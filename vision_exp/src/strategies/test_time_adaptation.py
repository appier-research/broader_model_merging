"""Test-time adaptation of the merging coefficients: AdaMerging and DivMerge.

Both learn merging coefficients on *unlabeled* data (``ctx.unlabeled_loaders``,
by default a slice of each task's test split, as in the AdaMerging paper) and
differ only in the loss:

  adamerging   mean softmax entropy of the merged model on task t's batch
  divmerge     divergence (JS by default) between the task expert's logits and
               the merged model's logits on task t's batch

Parameterisation follows the reference implementation in model-merge-transfer:

  taskwise     one coefficient per basis direction          c: [n_dirs]
  layerwise    one coefficient per (parameter tensor, dir)  c: [n_keys, n_dirs]

Coefficients are clamped to [0, 1] in the forward pass. Directions come from
``method.basis`` (TA: one per task; TIES: one merged direction), so this works
for any merging method, like ``subspace_gd``. One *step* = one batch from every
task, summed loss, one optimizer update (the reference calls this an "epoch").
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..models import MultiTaskCLIPClassifier, avg_metrics
from ._common import eval_multitask_all, pack_result, save_finetuned
from .context import StrategyContext
from .pseudo_labels import get_pseudo_loaders, soft_divergence

METHODS = ("adamerging", "divmerge")
VARIANTS = ("taskwise", "layerwise")
LOG_EVERY = 50


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of softmax(logits)  [B]."""
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def _hparams(ctx: StrategyContext, name: str):
    """(steps, lr, prior); a None prior means 1/N (the ``avg`` init) for both methods."""
    avg = 1.0 / len(ctx.tasks)
    if name == "adamerging":
        return ctx.ada_steps, ctx.ada_lr, ctx.ada_prior if ctx.ada_prior is not None else avg
    if name == "divmerge":
        return ctx.div_steps, ctx.div_lr, ctx.div_prior if ctx.div_prior is not None else avg
    raise ValueError(f"Unknown unlabeled coefficient method '{name}' (expected {METHODS})")


def _init_coeffs(variant: str, n_keys: int, n_dirs: int, prior: float, device: str) -> torch.nn.Parameter:
    if variant == "taskwise":
        return torch.nn.Parameter(torch.full((n_dirs,), prior, device=device))
    if variant == "layerwise":
        return torch.nn.Parameter(torch.full((n_keys, n_dirs), prior, device=device))
    raise ValueError(f"Unknown variant '{variant}' (expected {VARIANTS})")


def _weights(c, variant, base_dev, dirs_dev, keys, prefix=""):
    lam = c.clamp(0.0, 1.0)
    out = {}
    for ki, k in enumerate(keys):
        row = lam if variant == "taskwise" else lam[ki]
        delta = dirs_dev[0][k] * row[0]
        for i in range(1, len(dirs_dev)):
            delta = delta + dirs_dev[i][k] * row[i]
        out[prefix + k] = base_dev[k] + delta
    return out


def _coeff_summary(c, variant) -> dict:
    lam = c.detach().clamp(0.0, 1.0).cpu()
    if variant == "taskwise":
        return {"coefficients": lam.tolist()}
    return {
        "coefficients_per_dir_mean": lam.mean(0).tolist(),
        "coefficients_per_dir_std": lam.std(0).tolist(),
        "coefficients": lam.tolist(),  # [n_keys, n_dirs], key order = ctx.keys
    }


def run_one(ctx: StrategyContext, name: str, variant: str) -> dict:
    from .. import wandb_util

    method, keys, tasks, device = ctx.method, ctx.keys, ctx.tasks, ctx.device
    steps, lr, prior = _hparams(ctx, name)
    tag = f"{name}/{variant}"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if name == "divmerge":
        train_loaders, expert_stats = get_pseudo_loaders(ctx)
    else:
        train_loaders, expert_stats = ctx.unlabeled_loaders, None

    dirs = method.basis(ctx.base_sd, ctx.task_sds, keys)
    n_dirs = len(dirs)
    base_dev = {k: ctx.base_sd[k].float().to(device) for k in keys}
    dirs_dev = [{k: d[k].float().to(device) for k in keys} for d in dirs]
    c = _init_coeffs(variant, len(keys), n_dirs, prior, device)

    logging.info(f"{tag}: building fresh classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, tasks, device, ctx.class_indices)
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam([c], lr=lr)

    logging.info(
        f"\n=== {name} ({variant}) ===\n"
        f"  coeffs={tuple(c.shape)}  n_dirs={n_dirs}  prior={prior:.4f}  steps={steps}  lr={lr}"
        + (f"  divergence={ctx.divergence}" if name == "divmerge" else "")
    )
    task_iters = {t: iter(l) for t, l in train_loaders.items()}
    running = 0.0
    final_loss = float("nan")
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        for task in tasks:
            try:
                batch = next(task_iters[task])
            except StopIteration:
                task_iters[task] = iter(train_loaders[task])
                batch = next(task_iters[task])
            pv = batch["pixel_values"].to(device)
            # Rebuild the merged weights per task so each backward() has its own
            # graph (one shared W would be freed by the first task's backward).
            W = _weights(c, variant, base_dev, dirs_dev, keys, prefix="model.")
            logits, _ = functional_call(model, W, args=(pv, task))
            if name == "adamerging":
                loss = softmax_entropy(logits).mean()
            else:
                loss = soft_divergence(logits, batch["expert_logits"].to(device), ctx.divergence)
            loss.backward()  # summed over tasks, as in the reference
            step_loss += loss.item()
        optimizer.step()
        running += step_loss
        final_loss = step_loss
        if step % LOG_EVERY == 0 or step == steps:
            n = LOG_EVERY if step % LOG_EVERY == 0 else step % LOG_EVERY
            avg = running / n
            running = 0.0
            lam = c.detach().clamp(0, 1)
            logging.info(f"  step {step}/{steps}  loss={avg:.4f}  coeff_mean={lam.mean().item():.4f}")
            wandb_util.log_epoch(tag, step, {"train_loss": avg, "coeff_mean": lam.mean().item()})

    with torch.no_grad():
        W_eval = _weights(c, variant, base_dev, dirs_dev, keys)
    model.model.load_state_dict(W_eval, strict=False)
    del base_dev, dirs_dev, W_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info(f"{tag}: evaluating on valid/test")
    valid_res = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
    test_res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
    logging.info(f"  valid_avg_acc={avg_metrics(valid_res)[0]:.4f}  test_avg_acc={avg_metrics(test_res)[0]:.4f}")

    extra = {
        "method": name, "variant": variant, "n_dirs": n_dirs, "n_coeffs": int(c.numel()),
        "steps": steps, "learning_rate": lr, "prior": prior, "final_train_loss": final_loss,
        "unlabeled_data": ctx.unlabeled_stats,
        **_coeff_summary(c, variant),
    }
    if name == "divmerge":
        extra["divergence"] = ctx.divergence
        extra["expert_on_unlabeled"] = expert_stats
    if ctx.save_checkpoints:
        ck = f"{name}_{variant}"
        extra["checkpoint"] = save_finetuned(ctx, model.model.state_dict(), keys, ck)
    return pack_result(valid_res, test_res, **extra)


def run(ctx: StrategyContext, name: str, variants: List[str]) -> Dict[str, dict]:
    return {v: run_one(ctx, name, v) for v in variants}
