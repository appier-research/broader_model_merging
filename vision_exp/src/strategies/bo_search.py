"""Bayesian optimization over the method's basis coefficients (optuna GPSampler).

Black-box maximization of the valid-pool classification accuracy over

    W(c) = base + sum_i c_i * dir_i

where {dir_i} = method.basis(...) -- TA/DARE: one direction per task; TIES /
TSV-M: the single merged direction, so n_dirs = 1. This is the same coefficient
space subspace_gd descends, but with a different objective: subspace_gd follows
the image CE loss and picks its lowest-loss epoch, while bo_search queries the
valid accuracy itself, so there is no loss/accuracy mismatch to fall through --
at the price of one full valid forward pass per trial.

Why BO rather than extending coeff_search: coeff_search sweeps one scalar along
the diagonal, but the per-task optima differ, and a per-task grid costs
lambda_steps ** n_tasks evaluations. A GP with log-EI (optuna's GPSampler)
covers the same continuous box in a few dozen trials.

Search box and warm start. The init gives a center c0 (coeff_best -> lambda* on
every direction, avg -> 1/N, merged -> 1, pretrained -> 0) and the box is
[max(bo_lower, c0_i - bo_radius), c0_i + bo_radius] per coefficient. The center
is the first trial (study.enqueue_trial), the next n_startup_trials - 1 are
uniform draws in the box, and every trial after that is a GPSampler proposal.
The only cross-strategy dependency is lambda* (for the coeff_best center), the
same one subspace_gd's coeff_best init already reads.

Vision models are small and everything runs in fp32, so base + basis stay
resident on the GPU (like subspace_gd) and each trial's W(c) is a per-key axpy.

Selection: the best observed trial by valid_avg_acc (ties -> earliest). Every
trial's coefficients and per-task metrics are kept in the result cell so an
offline re-selection (e.g. GP posterior mean) is possible without rerunning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from ..models import MultiTaskCLIPClassifier, avg_metrics
from ._common import eval_multitask_all, pack_result, save_finetuned
from .context import StrategyContext

BO_INITS = ("pretrained", "avg", "merged", "coeff_best")


def _require_optuna():
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError("bo_search needs optuna (pip install optuna); GPSampler also needs scipy") from e
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our logging.info lines are the record
    return optuna


def _center_coeffs(ctx: StrategyContext, init: str, n_dirs: int) -> List[float]:
    """Box center, one coefficient per basis direction. All four inits are scalar
    and broadcast across n_dirs (same convention as directional_sampling)."""
    if init == "pretrained":
        return [0.0] * n_dirs
    if init == "avg":
        return [1.0 / len(ctx.tasks)] * n_dirs
    if init == "merged":
        return [1.0] * n_dirs
    if init == "coeff_best":
        if ctx.best_lambda is None:
            raise ValueError("bo_search init 'coeff_best' requires best_lambda (run coeff_search first)")
        return [float(ctx.best_lambda)] * n_dirs
    raise ValueError(f"Unknown bo_search init '{init}' (expected one of {list(BO_INITS)})")


def _box(center: List[float], radius: float, lower: Optional[float]) -> List[Tuple[float, float]]:
    if radius <= 0:
        raise ValueError(f"bo_radius must be > 0 (got {radius})")
    out = []
    for c in center:
        lo, hi = c - radius, c + radius
        if lower is not None:
            lo = max(lo, lower)
        if hi <= lo:
            raise ValueError(f"empty search interval for center {c}: [{lo}, {hi}] (bo_lower={lower})")
        out.append((lo, hi))
    return out


def _fmt(coeffs: List[float]) -> str:
    return "[" + ", ".join(f"{c:.4f}" for c in coeffs) + "]"


def run_one(ctx: StrategyContext, init: str) -> dict:
    optuna = _require_optuna()
    method, keys, tasks, device = ctx.method, ctx.keys, ctx.tasks, ctx.device
    n_trials = int(ctx.bo_trials)
    if n_trials < 1:
        raise ValueError(f"bo_trials must be >= 1 (got {n_trials})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dirs = method.basis(ctx.base_sd, ctx.task_sds, keys)
    n_dirs = len(dirs)
    center = _center_coeffs(ctx, init, n_dirs)
    box = _box(center, float(ctx.bo_radius), ctx.bo_lower)
    # None -> 2 * n_dirs + 1 (the enqueued center + 2 * n_dirs uniform draws):
    # enough points for the GP to fit a length scale per direction before it
    # starts proposing, without spending the whole budget on startup.
    n_startup = int(ctx.bo_startup_trials) if ctx.bo_startup_trials is not None else 2 * n_dirs + 1
    n_startup = max(1, min(n_startup, n_trials))

    # fp32 base + directions resident on the GPU (same budget as subspace_gd);
    # each trial's W(c) is a per-key axpy with no CPU<->GPU copy.
    base_dev = {k: ctx.base_sd[k].float().to(device) for k in keys}
    dirs_dev = [{k: d[k].float().to(device) for k in keys} for d in dirs]

    logging.info(f"bo_search(init={init}): building fresh classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, device, ctx.class_indices)
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def load_coeffs(coeffs: List[float]) -> None:
        W = {}
        for k in keys:
            w = base_dev[k].clone()
            for c, d in zip(coeffs, dirs_dev):
                w = w + c * d[k]
            W[k] = w
        model.model.load_state_dict(W, strict=False)

    logging.info(f"\n=== Bayesian optimization (init={init}) ===")
    logging.info(
        f"  optuna {optuna.__version__} GPSampler  n_dirs={n_dirs}  center={_fmt(center)}  "
        f"radius=±{ctx.bo_radius}  lower={ctx.bo_lower}  n_trials={n_trials}  "
        f"n_startup={n_startup} (center + {n_startup - 1} uniform)  seed={ctx.bo_seed}"
    )
    for i, (lo, hi) in enumerate(box):
        label = tasks[i] if n_dirs == len(tasks) else "merged"
        logging.info(f"    c{i} ({label}): [{lo:.4f}, {hi:.4f}]")

    sampler = optuna.samplers.GPSampler(seed=int(ctx.bo_seed), n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({f"c{i}": c for i, c in enumerate(center)})

    def objective(trial) -> float:
        coeffs = [trial.suggest_float(f"c{i}", lo, hi) for i, (lo, hi) in enumerate(box)]
        load_coeffs(coeffs)
        res = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
        acc = avg_metrics(res)[0]
        trial.set_user_attr("valid_per_task", res)
        done = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_so_far = max(done + [acc])
        phase = "startup" if trial.number < n_startup else "gp"
        logging.info(
            f"  trial {trial.number + 1}/{n_trials} [{phase}]  c={_fmt(coeffs)}  "
            f"valid_avg_acc={acc:.4f}  best_so_far={best_so_far:.4f}"
        )
        return acc

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial  # highest value; optuna returns the earliest on ties
    best_coeffs = [float(best.params[f"c{i}"]) for i in range(n_dirs)]
    ranked = sorted(study.trials, key=lambda t: (-(t.value or 0.0), t.number))
    logging.info(f"  best trial={best.number + 1}  c={_fmt(best_coeffs)}  valid_avg_acc={best.value:.4f}")
    for t in ranked[:5]:
        logging.info(
            f"    #{t.number + 1:<3d} valid_avg_acc={t.value:.4f}  "
            f"c={_fmt([t.params[f'c{i}'] for i in range(n_dirs)])}"
        )

    logging.info(f"bo_search(init={init}): evaluating best trial on valid/test")
    load_coeffs(best_coeffs)
    model.eval()
    valid_res = eval_multitask_all(model, ctx.valid_loaders, tasks, device)
    test_res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
    logging.info(
        f"  coeff_final={best_coeffs}  valid_avg_acc={avg_metrics(valid_res)[0]:.4f}  "
        f"test_avg_acc={avg_metrics(test_res)[0]:.4f}"
    )

    extra = {
        "init": init,
        "subspace": True,
        "sampler": f"optuna-{optuna.__version__}/GPSampler",
        "n_coeffs": n_dirs,
        "n_trials": len(study.trials),
        "n_startup_trials": n_startup,
        "seed": int(ctx.bo_seed),
        "radius": float(ctx.bo_radius),
        "lower": ctx.bo_lower,
        "box": [[lo, hi] for lo, hi in box],
        "coefficients_init": center,
        "coefficients_final": best_coeffs,
        "best_trial": best.number,
        "trials": [
            {
                "number": t.number,
                "phase": "startup" if t.number < n_startup else "gp",
                "coefficients": [float(t.params[f"c{i}"]) for i in range(n_dirs)],
                "valid_avg_acc": t.value,
                "valid_per_task": t.user_attrs.get("valid_per_task"),
            }
            for t in study.trials
        ],
    }
    if ctx.best_lambda is not None and init == "coeff_best":
        extra["best_lambda"] = ctx.best_lambda
    if ctx.save_checkpoints:
        tag = f"bo_search_{init}"
        logging.info(f"bo_search(init={init}): saving checkpoint -> {tag}")
        extra["checkpoint"] = save_finetuned(ctx, model.model.state_dict(), keys, tag)
        logging.info(f"bo_search(init={init}): checkpoint saved")
    return pack_result(valid_res, test_res, **extra)


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}
