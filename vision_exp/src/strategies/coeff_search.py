"""Coefficient search strategy.

Diagonal sweep of a single scalar coefficient lambda over the method's merged
direction. The best lambda is picked on the validation set; test metrics are
reported at that lambda. No gradient descent.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from ..models import MultiTaskCLIPClassifier, avg_metrics
from ._common import eval_multitask_all, pack_result
from .context import StrategyContext


def _lambda_grid(lo: float, hi: float, steps: int) -> List[float]:
    if steps < 2:
        raise ValueError("lambda_steps must be >= 2")
    return [round(lo + i * (hi - lo) / (steps - 1), 10) for i in range(steps)]


def run(ctx: StrategyContext) -> dict:
    method, keys, tasks = ctx.method, ctx.keys, ctx.tasks
    logging.info("coeff_search: building fresh classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, ctx.device, ctx.class_indices)

    lambdas = _lambda_grid(ctx.lambda_min, ctx.lambda_max, ctx.lambda_steps)
    logging.info(f"\n=== Coefficient search ({method.name}) ===")
    rows, best_lam, best_valid_acc = [], lambdas[0], -1.0
    for lam in lambdas:
        delta = method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=lam)
        W = {k: ctx.base_sd[k].float() + delta[k] for k in keys}
        model.model.load_state_dict(W, strict=False)
        valid_res = eval_multitask_all(model, ctx.valid_loaders, tasks, ctx.device)
        valid_acc, valid_loss = avg_metrics(valid_res)
        logging.info(f"  lambda={lam:.4f}  valid_avg_acc={valid_acc:.4f}  valid_avg_loss={valid_loss:.4f}")
        rows.append({"lambda": lam, "valid_avg_acc": valid_acc, "valid_avg_loss": valid_loss,
                     "valid_per_task": valid_res})
        if valid_acc > best_valid_acc:
            best_valid_acc, best_lam = valid_acc, lam

    best_delta = method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=best_lam)
    W = {k: ctx.base_sd[k].float() + best_delta[k] for k in keys}
    model.model.load_state_dict(W, strict=False)
    valid_best = eval_multitask_all(model, ctx.valid_loaders, tasks, ctx.device)
    test_best = eval_multitask_all(model, ctx.test_loaders, tasks, ctx.device)
    logging.info(
        f"  best lambda*={best_lam:.4f}  valid_avg_acc={avg_metrics(valid_best)[0]:.4f}  "
        f"test_avg_acc={avg_metrics(test_best)[0]:.4f}"
    )
    return pack_result(valid_best, test_best, best_lambda=best_lam, sweep=rows)


def _apply_lambda(ctx: StrategyContext, model, lam: float) -> None:
    delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, ctx.keys, coeff=lam)
    W = {k: ctx.base_sd[k].float() + delta[k] for k in ctx.keys}
    model.model.load_state_dict(W, strict=False)


@torch.no_grad()
def _count_eval(model, ctx: StrategyContext) -> int:
    """Valid eval; returns the number of images forwarded."""
    model.eval()
    n_fwd = 0
    for task in ctx.tasks:
        for batch in ctx.valid_loaders[task]:
            n_fwd += batch["labels"].size(0)
            model(batch["pixel_values"].to(ctx.device), task, batch["labels"].to(ctx.device))
    return n_fwd


def profile_one(ctx: StrategyContext) -> dict:
    """Warm up one lambda eval, then time the next (peak mem + wall + image counts)."""
    from ..cost import measure

    logging.info("coeff_search: profile (warmup λ + 1 timed valid eval)")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, ctx.device, ctx.class_indices)
    lambdas = _lambda_grid(ctx.lambda_min, ctx.lambda_max, ctx.lambda_steps)
    _apply_lambda(ctx, model, lambdas[0])
    eval_multitask_all(model, ctx.valid_loaders, ctx.tasks, ctx.device)
    _apply_lambda(ctx, model, lambdas[1])
    with measure() as stats:
        stats["n_fwd_images"] = _count_eval(model, ctx)
        stats["n_bwd_images"] = 0
    logging.info(
        f"  timed lambda={lambdas[1]:.4f}  wall={stats['action_wall_s']:.1f}s  "
        f"peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    return {"init": None, "action": "valid_eval", **stats}
