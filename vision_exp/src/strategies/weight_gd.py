"""Weight-space GD strategy.

Full fine-tuning of the shared backbone, starting from an init point built from
the merged direction. Training data is the labeled validation budget
(``gd_label_source=gt``) or the unlabeled data with task-expert pseudo-labels
(``expert_soft`` / ``expert_hard``, see pseudo_labels.py). Init points:

  pretrained   -> base weights (delta = 0)
  avg          -> base + merged_delta(coeff=1/N)
  merged       -> base + merged_delta(coeff=1)
  coeff_best   -> base + merged_delta(coeff=lambda*)   (needs coeff search)
  random_<seed> -> i.i.d. N(0,1) over the shared float keys (true random)

Optional L2-SP (``ctx.gd_l2_sp``): train loss + (alpha/2)||W-W_init||^2, with AdamW
decay-to-zero disabled. Distance to init is logged to W&B only.

Checkpoints, when saved, are named by the init point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import get_cosine_schedule_with_warmup

from ..models import MultiTaskCLIPClassifier, avg_metrics
from ..subspace import SubspaceBasis, random_directions_like, record_step
from ._common import eval_multitask_all, pack_result, save_finetuned
from .context import StrategyContext
from .pseudo_labels import get_pseudo_loaders, result_key, step_loss


def _parse_random_seed(init: str) -> int:
    rest = init[len("random_"):]
    if not rest.isdigit():
        raise ValueError(f"random init must be 'random_<int seed>', got '{init}'")
    return int(rest)


def _init_weights(ctx: StrategyContext, init: str) -> Dict[str, torch.Tensor]:
    base_sd, keys, method = ctx.base_sd, ctx.keys, ctx.method
    if init == "pretrained":
        return {k: base_sd[k].float() for k in keys}
    if init == "avg":
        delta = method.merged_delta(base_sd, ctx.task_sds, keys, coeff=1.0 / len(ctx.tasks))
        return {k: base_sd[k].float() + delta[k] for k in keys}
    if init == "merged":
        delta = method.merged_delta(base_sd, ctx.task_sds, keys, coeff=1.0)
        return {k: base_sd[k].float() + delta[k] for k in keys}
    if init == "coeff_best":
        if ctx.best_lambda is None:
            raise ValueError("weight_gd init 'coeff_best' requires coeff search (best_lambda)")
        delta = method.merged_delta(base_sd, ctx.task_sds, keys, coeff=ctx.best_lambda)
        return {k: base_sd[k].float() + delta[k] for k in keys}
    if init.startswith("random_"):
        g = torch.Generator(device="cpu").manual_seed(_parse_random_seed(init))
        return {k: torch.randn(base_sd[k].shape, generator=g, dtype=torch.float32) for k in keys}
    raise ValueError(
        f"Unknown init '{init}' (expected pretrained, avg, merged, coeff_best, random_<seed>)"
    )


def _squared_dist_to_init(params, p0s) -> torch.Tensor:
    acc = None
    for p, p0 in zip(params, p0s):
        term = (p - p0).pow(2).sum()
        acc = term if acc is None else acc + term
    return acc if acc is not None else torch.zeros(())


def _build_optimizer(ctx: StrategyContext, params, weight_decay=None):
    opt = ctx.gd_optimizer.lower()
    if opt == "adamw":
        kwargs = {"lr": ctx.gd_lr}
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        return torch.optim.AdamW(params, **kwargs)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=ctx.gd_lr, momentum=ctx.gd_momentum)
    raise ValueError(f"Unknown gd_optimizer '{ctx.gd_optimizer}' (expected adamw, sgd)")


@dataclass
class _TrainSetup:
    """What one weight-GD run trains with, built once by ``_setup_train``."""
    optimizer: torch.optim.Optimizer
    scheduler: object
    steps_per_epoch: int
    train_loaders: Dict[str, object]  # valid budget (gt) or pseudo-labeled unlabeled data
    label_source: str
    alpha: float  # L2-SP strength (0 = off)
    params: list
    p0s: Optional[list]  # init weights for L2-SP


def _setup_train(model, ctx: StrategyContext) -> _TrainSetup:
    label_source = ctx.gd_label_source
    train_loaders = ctx.valid_loaders if label_source == "gt" else get_pseudo_loaders(ctx)[0]
    alpha = float(getattr(ctx, "gd_l2_sp", 0.0) or 0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    p0s = [p.detach().clone() for p in params] if alpha > 0 else None
    optimizer = _build_optimizer(ctx, params, weight_decay=0.0 if alpha > 0 else None)
    steps_per_epoch = max(len(l) for l in train_loaders.values())
    total_steps = steps_per_epoch * ctx.gd_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * ctx.gd_warmup_ratio), total_steps)
    return _TrainSetup(optimizer, scheduler, steps_per_epoch, train_loaders, label_source, alpha, params, p0s)


class _TrajectoryRecorder:
    """Optional in-memory trajectory recording for diagnostics
    (scripts/subspace_escape.py). Never built on the regular path, so
    weight_gd runs through experiment.py and profile_one are unaffected.

    At each optimizer step whose global index is in ``record_at`` it appends
    ``subspace.record_step(...)`` to ``trajectory``: the step's gradient (taken
    after every backward of the step, i.e. what the optimizer actually uses,
    including the L2-SP term when on), the weight update, and the weights
    after it, all relative to the task-vector subspace S."""

    def __init__(self, model, ctx: StrategyContext, trajectory: List[dict], record_at: Optional[set],
                 include_null_baseline: bool = False, metrics_fn=None, snapshot_fn=None):
        device, keys = ctx.device, ctx.keys
        directions = [
            {k: d[k].float().to(device) for k in keys}
            for d in ctx.method.basis(ctx.base_sd, ctx.task_sds, keys)
        ]
        self.basis = SubspaceBasis(directions, keys)
        # Same-rank random-direction baseline: with D (params) >> dim(S), any
        # fixed low-rank subspace looks nearly orthogonal to a generic vector,
        # so g_perp_ratio needs this to be interpretable at all.
        self.null_basis = (
            SubspaceBasis(random_directions_like(directions, keys), keys) if include_null_baseline else None
        )
        self.base_dev = {k: ctx.base_sd[k].float().to(device) for k in keys}
        self.params = dict(model.model.named_parameters())
        self.keys, self.trajectory, self.record_at = keys, trajectory, record_at or set()
        self.metrics_fn, self.snapshot_fn = metrics_fn, snapshot_fn
        self.step = 0
        self._w_before = self._g = None
        logging.info(
            f"  recording trajectory: S rank={self.basis.dim} "
            f"(from {len(directions)} task vector(s)) at {len(self.record_at)} step(s)"
        )

    def before_step(self) -> None:
        self._w_before = (
            {k: self.params[k].detach().clone() for k in self.keys} if self.step in self.record_at else None
        )

    def after_backward(self) -> None:
        if self._w_before is not None:
            self._g = {k: self.params[k].grad.detach() for k in self.keys}

    def after_step(self, model, epoch: int, loss: float) -> None:
        if self._w_before is not None:
            w_now = {k: self.params[k].detach() for k in self.keys}
            dw = {k: w_now[k] - self._w_before[k] for k in self.keys}
            self.trajectory.append(record_step(
                self.basis, self._g, dw, w_now, self.base_dev,
                step=self.step, epoch=epoch, loss=loss, null_basis=self.null_basis,
                extra=(self.metrics_fn(model) if self.metrics_fn is not None else None),
            ))
            if self.snapshot_fn is not None:
                # Hand the caller this step's weights (e.g. to keep the
                # S-orthogonal residual for an axis known only at the end).
                self.snapshot_fn(self.step, w_now)
            self._w_before = self._g = None
        self.step += 1


def _one_epoch(model, ctx: StrategyContext, st: _TrainSetup,
               rec: Optional[_TrajectoryRecorder] = None, epoch: int = 0):
    """One weight-GD epoch. Returns (avg_loss, avg_acc, avg_gt_acc, n_fwd, n_bwd);
    avg_acc is against the training targets (GT or pseudo), avg_gt_acc against GT.
    ``rec`` (diagnostics only) records the trajectory at its chosen steps."""
    tasks, device = ctx.tasks, ctx.device
    model.train()
    task_iters = {t: iter(l) for t, l in st.train_loaders.items()}
    epoch_loss = 0.0
    epoch_correct = epoch_total = 0
    epoch_gt_correct = 0  # diagnostic only under non-gt sources
    n_fwd = n_bwd = 0
    for _ in range(st.steps_per_epoch):
        st.optimizer.zero_grad()
        if rec is not None:
            rec.before_step()
        step_loss_sum = 0.0
        for task in tasks:
            try:
                batch = next(task_iters[task])
            except StopIteration:
                task_iters[task] = iter(st.train_loaders[task])
                batch = next(task_iters[task])
            logits, loss, targets = step_loss(model, batch, task, st.label_source, device, ctx.divergence)
            (loss / len(tasks)).backward()
            step_loss_sum += loss.item() / len(tasks)
            pred = logits.argmax(-1)
            n = targets.size(0)
            epoch_correct += (pred == targets).sum().item()
            epoch_gt_correct += (pred == batch["labels"].to(device)).sum().item()
            epoch_total += n
            n_fwd += n
            n_bwd += n
        if st.alpha > 0:
            (0.5 * st.alpha * _squared_dist_to_init(st.params, st.p0s)).backward()
        if rec is not None:
            rec.after_backward()
        st.optimizer.step()
        st.scheduler.step()
        epoch_loss += step_loss_sum
        if rec is not None:
            rec.after_step(model, epoch, step_loss_sum)
    return (epoch_loss / st.steps_per_epoch, epoch_correct / epoch_total,
            epoch_gt_correct / epoch_total, n_fwd, n_bwd)


def _train(model, ctx: StrategyContext, init: str, rec: Optional[_TrajectoryRecorder] = None) -> dict:
    from .. import wandb_util

    st = _setup_train(model, ctx)
    label_source, alpha, params, p0s = st.label_source, st.alpha, st.params, st.p0s
    key = result_key(label_source)
    patience = max(0, int(ctx.gd_patience))

    logging.info(
        f"\n=== Weight GD (init={init}, labels={label_source}, opt={ctx.gd_optimizer}, lr={ctx.gd_lr}, "
        f"epochs={ctx.gd_epochs}, patience={patience or 'off'}"
        + (f", l2_sp={alpha:g}" if alpha > 0 else "")
        + ") ==="
    )
    best_loss, best_state, best_epoch = float("inf"), None, 0
    epochs_since_improve = 0
    epochs_run = 0
    stopped_early = False
    for epoch in range(ctx.gd_epochs):
        avg_loss, avg_acc, avg_gt_acc, _, _ = _one_epoch(model, ctx, st, rec, epoch)
        epochs_run = epoch + 1
        if alpha > 0:
            with torch.no_grad():
                dist_sq = float(_squared_dist_to_init(params, p0s))
            monitored = avg_loss + 0.5 * alpha * dist_sq
            l2_sp_dist = dist_sq ** 0.5
        else:
            monitored, l2_sp_dist = avg_loss, None
        msg = f"  epoch {epochs_run}/{ctx.gd_epochs}  avg_loss={avg_loss:.4f}  avg_acc={avg_acc:.4f}"
        if label_source != "gt":
            msg += f"  gt_acc={avg_gt_acc:.4f}"
        logging.info(msg)
        wb = {"train_loss": avg_loss, "train_acc": avg_acc, "train_gt_acc": avg_gt_acc}
        if l2_sp_dist is not None:
            wb["l2_sp_dist"] = l2_sp_dist
            wb["train_loss_reg"] = monitored
        wandb_util.log_epoch(f"{key}/{init}", epochs_run, wb)
        if monitored < best_loss:
            best_loss = monitored
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epochs_run
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if patience > 0 and epochs_since_improve >= patience:
                stopped_early = True
                logging.info(
                    f"  early stop at epoch {epochs_run}: "
                    f"no train-loss improvement for {patience} epoch(s)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logging.info(
        f"  best epoch={best_epoch}/{epochs_run}  best_train_loss={best_loss:.4f}"
        + ("  (early stopped)" if stopped_early else "")
    )
    out = {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_train_loss": best_loss,
        "stopped_early": stopped_early,
        "patience": patience,
    }
    if alpha > 0:
        out["l2_sp"] = alpha
    return out


def run_one(
    ctx: StrategyContext,
    init: str,
    trajectory: Optional[List[dict]] = None,
    init_weights: Optional[Dict[str, torch.Tensor]] = None,
    record_at: Optional[set] = None,
    include_null_baseline: bool = False,
    return_state: bool = False,
    metrics_fn=None,
    snapshot_fn=None,
) -> dict:
    """Run weight_gd from ``init``. The keyword arguments are for diagnostics
    (scripts/subspace_escape.py) and change nothing when left at their
    defaults, as every regular run through experiment.py does.

    ``trajectory``: a list that gets one ``subspace.record_step`` dict appended
    at each optimizer step whose index is in ``record_at`` (in-memory only).
    ``metrics_fn(model)`` is merged into each recorded step (held-out metrics
    need an eval pass the recorder cannot do itself); ``snapshot_fn(step, w)``
    hands the caller the raw weights at each recorded step.
    ``init_weights``: a state dict used as the literal starting point; ``init``
    is then only a label (logging, checkpoint name).
    ``return_state``: add the final weights under ``"state_dict"`` -- not
    JSON-serializable, so callers must pop it before persisting.
    """
    tasks, keys, device = ctx.tasks, ctx.keys, ctx.device
    key = result_key(ctx.gd_label_source)
    # Drop the caching allocator's freed-but-not-returned memory before we
    # build a fresh classifier. This is the guard for the CUDA text-encoder
    # hang that showed up after weight_gd left the allocator fragmented.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"weight_gd(init={init}): building fresh classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, device, ctx.class_indices)
    W = init_weights if init_weights is not None else _init_weights(ctx, init)
    model.model.load_state_dict(W, strict=False)

    logging.info(f"weight_gd(init={init}): evaluating init on valid/test")
    init_valid = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
    init_test = eval_multitask_all(model, ctx.test_loaders, tasks, device)
    init_vacc, init_vloss = avg_metrics(init_valid)
    init_tacc, init_tloss = avg_metrics(init_test)
    logging.info(f"  init valid_avg_acc={init_vacc:.4f}  init test_avg_acc={init_tacc:.4f}")

    rec = None
    if trajectory is not None:
        rec = _TrajectoryRecorder(model, ctx, trajectory, record_at, include_null_baseline, metrics_fn, snapshot_fn)
    train_stats = _train(model, ctx, init, rec)
    del rec  # its basis copies live on the GPU

    logging.info(f"weight_gd(init={init}): evaluating on valid/test")
    valid_res = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
    test_res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
    logging.info(f"  valid_avg_acc={avg_metrics(valid_res)[0]:.4f}  test_avg_acc={avg_metrics(test_res)[0]:.4f}")

    extra = {
        "init": init, "label_source": ctx.gd_label_source,
        "epochs": ctx.gd_epochs, "learning_rate": ctx.gd_lr,
        "warmup_ratio": ctx.gd_warmup_ratio, "optimizer": ctx.gd_optimizer,
        "init_valid_avg_acc": init_vacc, "init_valid_avg_loss": init_vloss,
        "init_test_avg_acc": init_tacc, "init_test_avg_loss": init_tloss,
        **train_stats,
    }
    if ctx.gd_optimizer.lower() == "sgd":
        extra["momentum"] = ctx.gd_momentum
    if ctx.best_lambda is not None and init == "coeff_best":
        extra["best_lambda"] = ctx.best_lambda
    if ctx.gd_label_source != "gt":
        extra["divergence"] = ctx.divergence if ctx.gd_label_source == "expert_soft" else None
        extra["unlabeled_data"] = ctx.unlabeled_stats
        # pseudo-label quality: the expert's own accuracy on the unlabeled data
        extra["expert_on_unlabeled"] = get_pseudo_loaders(ctx)[1]
    if ctx.save_checkpoints:
        tag = f"{key}_{init}"
        logging.info(f"weight_gd(init={init}): saving checkpoint -> {tag}")
        extra["checkpoint"] = save_finetuned(ctx, model.model.state_dict(), keys, tag)
        logging.info(f"weight_gd(init={init}): checkpoint saved")
    if return_state:
        extra["state_dict"] = {k: v.detach().clone() for k, v in model.model.state_dict().items() if k in keys}
    return pack_result(valid_res, test_res, **extra)


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}


def profile_one(ctx: StrategyContext, init: str) -> dict:
    """Warm up one epoch, then time a second epoch (peak mem + wall + image counts)."""
    from ..cost import measure

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"weight_gd(init={init}): profile (warmup + 1 timed epoch)")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, ctx.device, ctx.class_indices)
    model.model.load_state_dict(_init_weights(ctx, init), strict=False)
    st = _setup_train(model, ctx)
    _one_epoch(model, ctx, st)
    with measure() as stats:
        avg_loss, avg_acc, _, n_fwd, n_bwd = _one_epoch(model, ctx, st)
        stats["n_fwd_images"] = n_fwd
        stats["n_bwd_images"] = n_bwd
    logging.info(
        f"  timed epoch  avg_loss={avg_loss:.4f}  avg_acc={avg_acc:.4f}  "
        f"wall={stats['action_wall_s']:.1f}s  "
        f"peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    return {"init": init, "action": "train_epoch", **stats}
