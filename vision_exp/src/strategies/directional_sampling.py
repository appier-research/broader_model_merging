"""Directional sampling around a merge init.

Samples
    W(α, β) = W* + sum_i (α_i - λ0) dir_i + β (-g_⊥)

where W* is the merge at a chosen init (pretrained / avg / merged / coeff_best),
{dir_i} is the method basis (TA: per-task vectors; TIES: one merged direction),
and g_⊥ is the valid-set loss gradient at W* with the basis component removed.
α_i ~ Unif[λ0 ± alpha], β ~ Unif[0, beta]. Selects the valid-best sample
(headline) and reports test acc for the valid top-5. Beat density is vs
coeff_search.valid_avg_acc (the coeff_best merge), not vs the box center.
"""

from __future__ import annotations

import logging
import statistics
from typing import Dict, List

import torch
from torch.func import functional_call

from ..models import MultiTaskCLIPClassifier, avg_metrics
from ._common import eval_multitask_all, pack_result
from .context import StrategyContext


def _flatten(sd: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.cat([sd[k].reshape(-1) for k in keys])


def _unflatten(vec: torch.Tensor, ref: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    out, offset = {}, 0
    for k in keys:
        n = ref[k].numel()
        out[k] = vec[offset:offset + n].view_as(ref[k])
        offset += n
    return out


def _project_orthogonal(g_flat: torch.Tensor, basis_flats: List[torch.Tensor]) -> torch.Tensor:
    """g minus its least-squares projection onto span(basis)."""
    g_perp = g_flat.clone()
    orth: List[torch.Tensor] = []
    for v in basis_flats:
        v = v.clone()
        for u in orth:
            v = v - torch.dot(v, u) * u
        n = torch.linalg.vector_norm(v)
        if float(n) > 1e-12:
            orth.append(v / n)
    for u in orth:
        g_perp = g_perp - torch.dot(g_perp, u) * u
    return g_perp


def _valid_loss_grad(model, W_star: Dict[str, torch.Tensor], ctx: StrategyContext) -> Dict[str, torch.Tensor]:
    """One full-valid-pass gradient of mean task loss at W* (no parameter update)."""
    keys, tasks, device = ctx.keys, ctx.tasks, ctx.device
    leaves = {k: W_star[k].detach().to(device).requires_grad_(True) for k in keys}
    W_call = {f"model.{k}": t for k, t in leaves.items()}
    for p in model.parameters():
        p.requires_grad_(False)

    n_steps = 0
    for task in tasks:
        for batch in ctx.valid_loaders[task]:
            pv = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            _, loss = functional_call(model, W_call, args=(pv, task, y))
            (loss / len(tasks)).backward()
            n_steps += 1
    if n_steps == 0:
        raise RuntimeError("directional_sampling: empty valid loaders")
    scale = 1.0 / n_steps
    grad = {}
    for k in keys:
        g = leaves[k].grad
        if g is None:
            raise RuntimeError(f"directional_sampling: missing grad for {k}")
        grad[k] = (g * scale).detach()
        leaves[k].grad = None
    return grad


def _apply(model, W: Dict[str, torch.Tensor]) -> None:
    model.model.load_state_dict(W, strict=False)


def _sample_weights(
    W_star: Dict[str, torch.Tensor],
    keys: List[str],
    dirs_dev: List[Dict[str, torch.Tensor]],
    g_perp: Dict[str, torch.Tensor],
    lam: float,
    alphas: List[float],
    beta: float,
) -> Dict[str, torch.Tensor]:
    W = {k: W_star[k].clone() for k in keys}
    for a, d in zip(alphas, dirs_dev):
        da = a - lam
        if da != 0.0:
            for k in keys:
                W[k] = W[k] + da * d[k]
    if beta != 0.0:
        for k in keys:
            W[k] = W[k] + beta * (-g_perp[k])
    return W


def _center_coeff(ctx: StrategyContext, init: str) -> float:
    if init == "pretrained":
        return 0.0
    if init == "avg":
        return 1.0 / len(ctx.tasks)
    if init == "merged":
        return 1.0
    if init == "coeff_best":
        if ctx.best_lambda is None:
            raise ValueError(
                "directional_sampling init 'coeff_best' requires best_lambda "
                "(from results/main_exp coeff_search)"
            )
        return float(ctx.best_lambda)
    raise ValueError(f"Unknown ds init '{init}' (expected pretrained, avg, merged, coeff_best)")


def run_one(ctx: StrategyContext, init: str) -> dict:
    if ctx.ds_samples < 1:
        raise ValueError("ds_samples must be >= 1")

    method, keys, tasks, device = ctx.method, ctx.keys, ctx.tasks, ctx.device
    lam = _center_coeff(ctx, init)
    alpha_r, beta_r = float(ctx.ds_alpha), float(ctx.ds_beta)
    n_samples = int(ctx.ds_samples)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(ctx.ds_seed))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dirs = method.basis(ctx.base_sd, ctx.task_sds, keys)
    n_dirs = len(dirs)
    base_dev = {k: ctx.base_sd[k].float().to(device) for k in keys}
    dirs_dev = [{k: d[k].float().to(device) for k in keys} for d in dirs]
    delta_star = method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=lam)
    W_star = {k: base_dev[k] + delta_star[k].to(device) for k in keys}

    logging.info(f"directional_sampling(init={init}): building classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, device, ctx.class_indices)

    logging.info(f"\n=== Directional sampling (init={init}) ===")
    logging.info(
        f"  λ0={lam:.4f}  alpha=±{alpha_r}  beta∈[0, {beta_r}]  "
        f"n_samples={n_samples}  n_dirs={n_dirs}  seed={ctx.ds_seed}"
    )
    logging.info(f"  computing ∇L at {init} (valid), then g_⊥")
    g = _valid_loss_grad(model, W_star, ctx)
    g_flat = _flatten(g, keys)
    basis_flats = [_flatten(d, keys) for d in dirs_dev]
    g_perp_flat = _project_orthogonal(g_flat, basis_flats)
    g_norm = float(torch.linalg.vector_norm(g_flat))
    g_perp_norm = float(torch.linalg.vector_norm(g_perp_flat))
    logging.info(f"  ||g||={g_norm:.6g}  ||g_⊥||={g_perp_norm:.6g}")
    g_perp = _unflatten(g_perp_flat, g, keys)
    del g_flat, g_perp_flat, basis_flats, g

    if ctx.coeff_best_valid_avg_acc is None:
        raise ValueError(
            "directional_sampling density requires coeff_search.valid_avg_acc "
            "(from results/main_exp or a coeff_search run in this JSON)"
        )
    coeff_best_valid_acc = float(ctx.coeff_best_valid_avg_acc)

    # Center valid acc (same protocol as samples). Density beats coeff_best.
    _apply(model, W_star)
    center_valid = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
    center_valid_acc, center_valid_loss = avg_metrics(center_valid)
    logging.info(f"  center valid_avg_acc={center_valid_acc:.4f}  valid_avg_loss={center_valid_loss:.4f}")
    logging.info(f"  coeff_best valid_avg_acc={coeff_best_valid_acc:.4f} (density threshold)")

    samples = []
    for i in range(n_samples):
        alphas = [float(torch.empty(1).uniform_(lam - alpha_r, lam + alpha_r, generator=rng))
                  for _ in range(n_dirs)]
        beta = float(torch.empty(1).uniform_(0.0, beta_r, generator=rng))
        W = _sample_weights(W_star, keys, dirs_dev, g_perp, lam, alphas, beta)
        _apply(model, W)
        valid_res = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
        vacc, vloss = avg_metrics(valid_res)
        samples.append({
            "idx": i, "alpha": alphas, "beta": beta,
            "valid_avg_acc": vacc, "valid_avg_loss": vloss,
        })
        logging.info(f"  sample {i + 1}/{n_samples}  valid_avg_acc={vacc:.4f}  valid_avg_loss={vloss:.4f}")

    n_beat_center = sum(1 for s in samples if s["valid_avg_acc"] > center_valid_acc)
    n_beat_coeff_best = sum(1 for s in samples if s["valid_avg_acc"] > coeff_best_valid_acc)
    density_valid = n_beat_coeff_best / n_samples
    top5_src = sorted(samples, key=lambda s: s["valid_avg_acc"], reverse=True)[:5]
    logging.info(
        f"  valid-best sample={top5_src[0]['idx']}  valid_avg_acc={top5_src[0]['valid_avg_acc']:.4f}  "
        f"density_valid={density_valid:.4f} ({n_beat_coeff_best}/{n_samples} beat coeff_best)"
    )

    logging.info(f"directional_sampling(init={init}): evaluating top-{len(top5_src)} on test")
    top5, valid_best, test_best = [], None, None
    for rank, s in enumerate(top5_src, 1):
        _apply(model, _sample_weights(W_star, keys, dirs_dev, g_perp, lam, s["alpha"], s["beta"]))
        test_res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
        tacc, tloss = avg_metrics(test_res)
        top5.append({
            "rank": rank, "idx": s["idx"],
            "alpha": s["alpha"], "beta": s["beta"],
            "valid_avg_acc": s["valid_avg_acc"], "valid_avg_loss": s["valid_avg_loss"],
            "test_avg_acc": tacc, "test_avg_loss": tloss,
        })
        logging.info(
            f"  top{rank} sample={s['idx']}  valid_avg_acc={s['valid_avg_acc']:.4f}  "
            f"test_avg_acc={tacc:.4f}"
        )
        if rank == 1:
            valid_best = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
            test_best = test_res

    top5_accs = [r["test_avg_acc"] for r in top5]
    top5_mean = statistics.mean(top5_accs)
    top5_std = statistics.stdev(top5_accs) if len(top5_accs) > 1 else 0.0
    logging.info(f"  top5 test_avg_acc={top5_mean:.4f} ± {top5_std:.4f} (n={len(top5_accs)})")

    extra = {
        "init": init,
        "center_coeff": lam,
        "alpha_range": alpha_r,
        "beta_max": beta_r,
        "n_samples": n_samples,
        "n_dirs": n_dirs,
        "ds_seed": ctx.ds_seed,
        "best_idx": top5[0]["idx"],
        "best_alpha": top5[0]["alpha"],
        "best_beta": top5[0]["beta"],
        "center_valid_avg_acc": center_valid_acc,
        "center_valid_avg_loss": center_valid_loss,
        "coeff_best_valid_avg_acc": coeff_best_valid_acc,
        "density_valid": density_valid,
        "n_beat_center": n_beat_center,
        "n_beat_coeff_best": n_beat_coeff_best,
        "grad_norm": g_norm,
        "grad_perp_norm": g_perp_norm,
        "top5_test_avg_acc": top5_mean,
        "top5_test_avg_acc_std": top5_std,
        "top5": top5,
        "samples": samples,
    }
    if init == "coeff_best":
        extra["best_lambda"] = lam
    return pack_result(valid_best, test_best, **extra)


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}
