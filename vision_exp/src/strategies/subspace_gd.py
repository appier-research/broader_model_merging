"""Subspace GD strategy (simple autograd on the coefficient vector).

Optimizes a coefficient vector c with one entry per basis direction supplied by
the merging method:

    W(c) = base + sum_i c_i * dir_i

The weights are rebuilt as a differentiable tensor each step and the forward pass
runs against them via torch.func.functional_call, so d(loss)/dc comes straight
from autograd. No task-vector prestacking, no geometry bookkeeping -- the whole
job is gradient descent on c.

The number of coefficients is len(method.basis(...)): TA gives one per task,
TIES gives a single scalar. Nothing here assumes a particular count.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from torch.func import functional_call
from transformers import get_cosine_schedule_with_warmup

from ..models import MultiTaskCLIPClassifier, avg_metrics
from ._common import eval_multitask_all, pack_result, save_finetuned
from .context import StrategyContext


def _weights_from_coeffs(c, base_dev, dirs_dev, keys, prefix=""):
    n = len(dirs_dev)
    out = {}
    for k in keys:
        delta = dirs_dev[0][k] * c[0]
        for i in range(1, n):
            delta = delta + dirs_dev[i][k] * c[i]
        out[prefix + k] = base_dev[k] + delta
    return out


def _coeffs_to_list(c) -> List[float]:
    return [float(x) for x in c.detach().cpu().tolist()]


def _one_epoch(model, c, base_dev, dirs_dev, ctx, optimizer, scheduler, steps_per_epoch):
    """One subspace-GD epoch. Returns (avg_loss, avg_acc, n_fwd, n_bwd)."""
    keys, tasks, device = ctx.keys, ctx.tasks, ctx.device
    model.train()
    task_iters = {t: iter(l) for t, l in ctx.valid_loaders.items()}
    epoch_loss = epoch_correct = epoch_total = 0
    n_fwd = n_bwd = 0
    for _ in range(steps_per_epoch):
        optimizer.zero_grad()
        W = _weights_from_coeffs(c, base_dev, dirs_dev, keys, prefix="model.")
        total_loss = 0.0
        step_loss = 0.0
        step_n = 0
        for task in tasks:
            try:
                batch = next(task_iters[task])
            except StopIteration:
                task_iters[task] = iter(ctx.valid_loaders[task])
                batch = next(task_iters[task])
            pv = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            logits, loss = functional_call(model, W, args=(pv, task, y))
            total_loss = total_loss + loss / len(tasks)
            step_loss += loss.item() / len(tasks)
            n = y.size(0)
            epoch_correct += (logits.argmax(-1) == y).sum().item()
            epoch_total += n
            n_fwd += n
            step_n += n
        total_loss.backward()
        n_bwd += step_n
        optimizer.step()
        scheduler.step()
        epoch_loss += step_loss
    return epoch_loss / steps_per_epoch, epoch_correct / epoch_total, n_fwd, n_bwd


def _setup(ctx: StrategyContext, init: str):
    """Load basis onto GPU, build a frozen classifier, return trainables."""
    method, keys, tasks, device = ctx.method, ctx.keys, ctx.tasks, ctx.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dirs = method.basis(ctx.base_sd, ctx.task_sds, keys)
    n_dirs = len(dirs)
    base_dev = {k: ctx.base_sd[k].float().to(device) for k in keys}
    dirs_dev = [{k: d[k].float().to(device) for k in keys} for d in dirs]
    c = torch.nn.Parameter(
        method.default_coeffs(n_dirs, init, ctx.best_lambda, n_tasks=len(tasks)).to(device)
    )
    logging.info(f"subspace_gd(init={init}): building fresh classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, device, ctx.class_indices)
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW([c], lr=ctx.subspace_lr)
    steps_per_epoch = max(len(l) for l in ctx.valid_loaders.values())
    total_steps = steps_per_epoch * ctx.subspace_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * ctx.subspace_warmup_ratio), total_steps
    )
    return model, c, base_dev, dirs_dev, n_dirs, optimizer, scheduler, steps_per_epoch


def run_one(ctx: StrategyContext, init: str) -> dict:
    from .. import wandb_util

    model, c, base_dev, dirs_dev, n_dirs, optimizer, scheduler, steps_per_epoch = _setup(ctx, init)
    coeff_init = _coeffs_to_list(c)
    keys, tasks, device = ctx.keys, ctx.tasks, ctx.device

    patience = max(0, int(ctx.subspace_patience))
    logging.info(f"\n=== Subspace GD (init={init}) ===")
    logging.info(
        f"  optimizing {n_dirs} coefficient(s)  lr={ctx.subspace_lr}  "
        f"epochs={ctx.subspace_epochs}  patience={patience or 'off'}"
    )
    best_loss, best_coeff, best_epoch = float("inf"), None, 0
    epochs_since_improve = 0
    epochs_run = 0
    stopped_early = False
    for epoch in range(ctx.subspace_epochs):
        avg_loss, avg_acc, _, _ = _one_epoch(
            model, c, base_dev, dirs_dev, ctx, optimizer, scheduler, steps_per_epoch
        )
        epochs_run = epoch + 1
        logging.info(f"  epoch {epochs_run}/{ctx.subspace_epochs}  avg_loss={avg_loss:.4f}  avg_acc={avg_acc:.4f}")
        wandb_util.log_epoch(f"subspace_gd/{init}", epochs_run, {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
        })
        if avg_loss < best_loss:
            best_loss, best_coeff, best_epoch = avg_loss, c.detach().clone(), epochs_run
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

    if best_coeff is not None:
        c.data.copy_(best_coeff)
    logging.info(
        f"  best epoch={best_epoch}/{epochs_run}  best_train_loss={best_loss:.4f}"
        + ("  (early stopped)" if stopped_early else "")
    )

    logging.info(f"subspace_gd(init={init}): evaluating on valid/test")
    with torch.no_grad():
        W_eval = _weights_from_coeffs(c, base_dev, dirs_dev, keys)
    model.model.load_state_dict(W_eval, strict=False)
    model.eval()
    valid_res = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
    test_res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
    coeff_final = _coeffs_to_list(c)
    logging.info(
        f"  coeff_final={coeff_final}  "
        f"valid_avg_acc={avg_metrics(valid_res)[0]:.4f}  test_avg_acc={avg_metrics(test_res)[0]:.4f}"
    )

    extra = {
        "init": init,
        "subspace": True,
        "n_coeffs": n_dirs,
        "epochs": ctx.subspace_epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "stopped_early": stopped_early,
        "patience": patience,
        "learning_rate": ctx.subspace_lr,
        "warmup_ratio": ctx.subspace_warmup_ratio,
        "coefficients_init": coeff_init,
        "coefficients_final": coeff_final,
    }
    if ctx.best_lambda is not None and init == "coeff_best":
        extra["best_lambda"] = ctx.best_lambda
    if ctx.save_checkpoints:
        tag = f"subspace_gd_{init}"
        logging.info(f"subspace_gd(init={init}): saving checkpoint -> {tag}")
        extra["checkpoint"] = save_finetuned(ctx, model.model.state_dict(), keys, tag)
        logging.info(f"subspace_gd(init={init}): checkpoint saved")
    return pack_result(valid_res, test_res, **extra)


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}


def profile_one(ctx: StrategyContext, init: str) -> dict:
    """Warm up one epoch, then time a second epoch (peak mem + wall + image counts)."""
    from ..cost import measure

    logging.info(f"subspace_gd(init={init}): profile (warmup + 1 timed epoch)")
    model, c, base_dev, dirs_dev, n_dirs, optimizer, scheduler, steps_per_epoch = _setup(ctx, init)
    _one_epoch(model, c, base_dev, dirs_dev, ctx, optimizer, scheduler, steps_per_epoch)
    with measure() as stats:
        avg_loss, avg_acc, n_fwd, n_bwd = _one_epoch(
            model, c, base_dev, dirs_dev, ctx, optimizer, scheduler, steps_per_epoch
        )
        stats["n_fwd_images"] = n_fwd
        stats["n_bwd_images"] = n_bwd
    logging.info(
        f"  timed epoch  n_dirs={n_dirs}  avg_loss={avg_loss:.4f}  avg_acc={avg_acc:.4f}  "
        f"wall={stats['action_wall_s']:.1f}s  "
        f"peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    return {"init": init, "action": "train_epoch", "n_dirs": n_dirs, **stats}
