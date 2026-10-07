"""Bayesian optimization over the method's basis coefficients (optuna GPSampler).

Black-box maximization of the valid-pool generate()+score average over

    W(c) = base + sum_i c_i * dir_i

where {dir_i} = method.basis(...) -- TA/DARE: one direction per task; TIES /
TSV-M: the single merged direction, so n_dirs = 1. This is the same
coefficient space subspace_gd optimizes, with a different objective:
subspace_gd descends the teacher-forced SFT loss and picks its lowest-loss
epoch, and on the units run so far that loss and the valid score disagree
systematically (qwen3-0.6b x TA: the subspace_gd cells with the lowest train
loss are the worst-scoring ones). bo_search queries the score itself, so
there is no loss/score mismatch to fall through -- at the price of one full
valid generate() per trial (~1 min on a 0.6B model over 361 examples).

Why BO rather than extending coeff_search: coeff_search sweeps one scalar
along the diagonal, but the per-task optima differ (qwen3-0.6b x TA:
bank77/ddxplus peak near lambda=0.6, ifeval near 0.3, usefulness_judge near
0.2), and a per-task grid costs lambda_steps ** n_tasks evaluations. A GP with
log-EI (optuna's GPSampler) covers the same space in a few dozen.

Search box and warm start. The init gives a center c0 (coeff_best -> lambda*
on every direction, avg -> 1/N, merged -> 1, pretrained -> 0) and the box is
[max(bo_lower, c0_i - bo_radius), c0_i + bo_radius] per coefficient. The
center is the first trial (study.enqueue_trial), the next
n_startup_trials - 1 are uniform draws in the box, and every trial after
that is a GPSampler proposal. Nothing else is fed to the GP -- not
coeff_search's sweep, not directional_sampling's draws -- so bo_search is a
standalone method whose only cross-strategy dependency is lambda*, the same
one subspace_gd's coeff_best init already has.

Memory: the basis (base + n_dirs directions) is kept resident on the GPU in
the run dtype when it fits next to the model (bo_basis_device="auto": at
most ~60% of the card), otherwise it stays in the fp32 CPU copies experiment
already holds and each trial's W(c) is assembled on the CPU and copied in by
load_state_dict -- ~10-20 s per trial on a 4B model, negligible next to the
minutes-long valid generate(). This is what lets bo_search run on qwen3-4b x
TA (1 + 4 bf16 copies = 40 GB) on a 24 GB card, where subspace_gd cannot.

Selection: the best observed trial by valid_avg_score (ties -> earliest).
Small valid pools quantize the score (usefulness_judge: 25 examples -> 0.04
steps), so a best-observed pick carries winner's-curse risk on test; every
trial's coefficients and per-task scores are kept in the result cell so an
offline re-selection (e.g. GP posterior mean) is possible without rerunning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from .. import models
from ..models import avg_score, load_causal_lm
from ._common import (
    attach_unseen, clear_cache, eval_all_tasks, eval_all_tasks_loss, free_model, pack_result, save_finetuned,
)
from .context import StrategyContext
BO_INITS = ("pretrained", "avg", "merged", "coeff_best")
BASIS_DEVICES = ("auto", "cuda", "cpu")


def _require_optuna():
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError("bo_search needs optuna (pip install optuna); GPSampler also needs scipy + torch") from e
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our logging.info lines are the record
    return optuna


def _center_coeffs(ctx: StrategyContext, init: str, n_dirs: int) -> List[float]:
    """Box center, one coefficient per basis direction. All four inits are
    scalar and broadcast across n_dirs (same convention as directional_sampling)."""
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


def _resolve_basis_device(ctx: StrategyContext, n_dirs: int, dtype: torch.dtype) -> str:
    """Where base + the n_dirs directions live between trials: on the GPU (fast
    per-trial axpy, (1 + n_dirs) extra model-sized copies) or on the CPU (the
    fp32 state dicts experiment.py already holds; W(c) is built there and
    copied in each trial). "auto" keeps them on the GPU only if model +
    basis stay under ~60% of the card, leaving the rest for generate()."""
    choice = str(ctx.bo_basis_device).lower()
    if choice not in BASIS_DEVICES:
        raise ValueError(f"bo_basis_device must be one of {list(BASIS_DEVICES)} (got {choice!r})")
    if ctx.device != "cuda" or not torch.cuda.is_available():
        return "cpu"
    if choice != "auto":
        return choice
    n_params = sum(ctx.base_sd[k].numel() for k in ctx.keys)
    copy_bytes = n_params * torch.tensor([], dtype=dtype).element_size()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    need = (2 + n_dirs) * copy_bytes  # model + base + n_dirs directions
    dev = "cuda" if need <= 0.6 * total else "cpu"
    logging.info(
        f"  basis device: {dev} (auto -- model + base + {n_dirs} dirs = {need / 2**30:.1f} GB "
        f"vs {total / 2**30:.1f} GB card)"
    )
    return dev


def _setup(ctx: StrategyContext, init: str):
    """Basis (resident on the GPU or left as the CPU fp32 copies), search box,
    startup count, fresh eval-mode model. Returns (model, n_dirs, center, box,
    n_startup, basis_device, base_res, dirs_res, dtype)."""
    method, keys, device = ctx.method, ctx.keys, ctx.device
    dtype = models.resolve_dtype(ctx.dtype)
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
    # starts proposing, without spending half of a ~50-trial budget on them.
    n_startup = int(ctx.bo_startup_trials) if ctx.bo_startup_trials is not None else 2 * n_dirs + 1
    n_startup = max(1, min(n_startup, n_trials))

    basis_device = _resolve_basis_device(ctx, n_dirs, dtype)
    if basis_device == "cuda":
        # subspace_gd's budget: base + n_dirs directions resident in the run
        # dtype, so each trial's W(c) is a per-key axpy with no CPU->GPU copy.
        base_res = {k: ctx.base_sd[k].to(dtype).to(device) for k in keys}
        dirs_res = [{k: d[k].to(dtype).to(device) for k in keys} for d in dirs]
        del dirs
    else:
        # fp32 CPU copies (ctx.base_sd is shared with the rest of the run, not
        # duplicated); W(c) is built per key on the CPU, cast, and copied in.
        base_res, dirs_res = ctx.base_sd, dirs

    logging.info(f"bo_search(init={init}): building fresh model")
    model = load_causal_lm(ctx.base_model, device=device, dtype=dtype)
    model.eval()
    return model, n_dirs, center, box, n_startup, basis_device, base_res, dirs_res, dtype


def _load_coeffs(model, keys, base_res, dirs_res, coeffs: List[float], dtype) -> None:
    """W(c) = base + sum_i c_i dir_i into the model (a GPU axpy when the basis
    is resident, a CPU build + copy-in otherwise)."""
    W = {}
    with torch.no_grad():
        for k in keys:
            w = base_res[k]
            for c, d in zip(coeffs, dirs_res):
                w = w + c * d[k]
            W[k] = w if w.dtype == dtype else w.to(dtype)
    model.load_state_dict(W, strict=False)  # copies CPU-built W onto the GPU params
    del W


def _valid_score(ctx: StrategyContext, model) -> Dict[str, float]:
    """One trial's objective: generate+score the valid pool."""
    res = eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, ctx.tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)
    return res


def run_one(ctx: StrategyContext, init: str) -> dict:
    optuna = _require_optuna()
    keys, tasks, device = ctx.keys, ctx.tasks, ctx.device
    n_trials = int(ctx.bo_trials)
    model, n_dirs, center, box, n_startup, basis_device, base_res, dirs_res, dtype = _setup(ctx, init)

    def load_coeffs(coeffs: List[float]) -> None:
        _load_coeffs(model, keys, base_res, dirs_res, coeffs, dtype)

    def valid_score(coeffs: List[float]) -> Dict[str, float]:
        load_coeffs(coeffs)
        return _valid_score(ctx, model)

    logging.info(f"\n=== Bayesian optimization (init={init}) ===")
    logging.info(
        f"  optuna {optuna.__version__} GPSampler  n_dirs={n_dirs}  center={_fmt(center)}  "
        f"radius=±{ctx.bo_radius}  lower={ctx.bo_lower}  basis_device={basis_device}  "
        f"n_trials={n_trials}  n_startup={n_startup} (center + {n_startup - 1} uniform)  seed={ctx.bo_seed}"
    )
    for i, (lo, hi) in enumerate(box):
        logging.info(f"    c{i} ({tasks[i] if n_dirs == len(tasks) else 'merged'}): [{lo:.4f}, {hi:.4f}]")

    sampler = optuna.samplers.GPSampler(seed=int(ctx.bo_seed), n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({f"c{i}": c for i, c in enumerate(center)})

    def objective(trial) -> float:
        coeffs = [trial.suggest_float(f"c{i}", lo, hi) for i, (lo, hi) in enumerate(box)]
        res = valid_score(coeffs)
        val = avg_score(res)
        trial.set_user_attr("valid_per_task", res)
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_so_far = max([t.value for t in done] + [val])
        phase = "startup" if trial.number < n_startup else "gp"
        logging.info(
            f"  trial {trial.number + 1}/{n_trials} [{phase}]  c={_fmt(coeffs)}  "
            f"valid_avg_score={val:.4f}  best_so_far={best_so_far:.4f}"
        )
        return val

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial  # highest value; optuna returns the earliest on ties
    best_coeffs = [float(best.params[f"c{i}"]) for i in range(n_dirs)]
    valid_res = best.user_attrs["valid_per_task"]
    ranked = sorted(study.trials, key=lambda t: (-(t.value or 0.0), t.number))
    logging.info(f"  best trial={best.number + 1}  c={_fmt(best_coeffs)}  valid_avg_score={best.value:.4f}")
    for t in ranked[:5]:
        logging.info(f"    #{t.number + 1:<3d} valid_avg_score={t.value:.4f}  c={_fmt([t.params[f'c{i}'] for i in range(n_dirs)])}")

    logging.info(f"bo_search(init={init}): evaluating best trial on valid loss / test")
    load_coeffs(best_coeffs)
    # W(c*) is in the model; drop the resident copies before generate()'s KV
    # cache on the (larger) test pool needs the headroom -- same as subspace_gd.
    # (In the CPU case base_res is ctx.base_sd and stays alive via ctx.)
    del base_res, dirs_res
    clear_cache(device)
    valid_loss = eval_all_tasks_loss(
        model, ctx.tokenizer, ctx.sft_pools, tasks, device, ctx.loss_batch_size, ctx.max_seq_length,
    )
    clear_cache(device)
    test_res = eval_all_tasks(
        model, ctx.tokenizer, ctx.test_examples, tasks, device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    logging.info(
        f"bo_search(init={init}): valid_avg_score={avg_score(valid_res):.4f}  "
        f"valid_avg_loss={models.avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}"
    )

    extra = {
        "init": init,
        "subspace": True,
        "sampler": f"optuna-{optuna.__version__}/GPSampler",
        "n_coeffs": n_dirs,
        "n_trials": len(study.trials),
        "n_startup_trials": n_startup,
        "seed": int(ctx.bo_seed),
        "basis_device": basis_device,
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
                "valid_avg_score": t.value,
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
        extra["checkpoint"] = save_finetuned(ctx, model, ctx.tokenizer, tag)
        logging.info(f"bo_search(init={init}): checkpoint saved")
    result = pack_result(valid_res, test_res, valid_loss=valid_loss, **extra)
    attach_unseen(ctx, model, result)
    free_model(model, device)
    return result


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}


def profile_one(ctx: StrategyContext, init: str) -> dict:
    """Warm up one trial (the box center), then time a second trial at a
    uniform draw inside the box (peak mem + wall + token count).

    The timed action is everything one trial costs: assembling W(c) (on the
    CPU and copied in, or a GPU axpy when the basis is resident) plus the
    generate()+score pass -- the same accounting as coeff_search's merge. Peak
    memory includes the resident basis when bo_basis_device resolves to cuda.
    """
    import random

    from ..cost import TokenCounter, measure

    logging.info(f"bo_search(init={init}): profile (warmup trial + 1 timed trial)")
    model, n_dirs, center, box, n_startup, basis_device, base_res, dirs_res, dtype = _setup(ctx, init)
    _load_coeffs(model, ctx.keys, base_res, dirs_res, center, dtype)
    _valid_score(ctx, model)
    rng = random.Random(int(ctx.bo_seed))
    probe = [rng.uniform(lo, hi) for lo, hi in box]
    with measure() as stats, TokenCounter(model) as counter:
        _load_coeffs(model, ctx.keys, base_res, dirs_res, probe, dtype)
        _valid_score(ctx, model)
    stats["n_fwd_tokens"] = counter.tokens
    stats["n_bwd_tokens"] = 0
    logging.info(
        f"  timed trial  n_dirs={n_dirs}  basis_device={basis_device}  c={_fmt(probe)}  "
        f"tokens={counter.tokens}  wall={stats['action_wall_s']:.1f}s  "
        f"peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    del base_res, dirs_res
    free_model(model, ctx.device)
    return {"init": init, "action": "trial", "n_dirs": n_dirs, "basis_device": basis_device, **stats}
