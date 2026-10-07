#!/usr/bin/env python3
"""Profile one action per strategy and write results/cost_estimation/units/<unit>.json.

Times a warmed-up coeff-search valid eval (generate+score) /
bo_search trial (generate+score) / weight_gd_lora epoch / subspace_gd epoch,
records peak GPU memory, counts the tokens forwarded/backpropagated, then
scales by n_actions (lambda_steps / bo_trials / gd_epochs / subspace_epochs).
Does not measure scores -- plots read test_avg_score from the main results
root. Same shape as vision_exp/scripts/profile_cost.py.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cost import (
    BO_SEARCH_INIT, BO_TRIALS, DEFAULT_STRATEGIES, PROFILED_STRATEGIES, SUBSPACE_GD_INIT,
    WEIGHT_GD_INIT, WEIGHT_GD_LORA_INIT,
)
from src.cost_estimation import build_cfg_from_cli, run_cost_estimation
from src.merging import available_methods
from src.strategies import DEFAULT_INITS
from src.tasks import available_tasks


def _opt_float(s: str):
    return None if s.strip().lower() in ("none", "") else float(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base", help="HF id of the pretrained base model")
    p.add_argument("--arch", default="qwen3-0.6b", help="Short arch tag used in checkpoint dir names")
    p.add_argument("--method", required=True, help=f"Merging method: one of {available_methods()}")
    p.add_argument("--tasks", required=True, help=f"Comma-separated task labels, from {available_tasks()}")
    p.add_argument("--task-vectors-dir", default="checkpoints/task_vectors")
    p.add_argument("--task-checkpoints", default=None,
                   help="Explicit comma-separated checkpoint paths (aligned with --tasks); overrides resolution")
    p.add_argument("--budget", default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                   help=f"Comma-separated subset of {list(PROFILED_STRATEGIES)} "
                        f"(default: {','.join(DEFAULT_STRATEGIES)}; full-parameter weight_gd is opt-in)")
    p.add_argument("--force", default="", help="Comma-separated strategies to re-profile even if present")
    p.add_argument("--weight-gd-init", default=None, choices=DEFAULT_INITS,
                   help=f"Init to profile weight_gd at (default: per method, {WEIGHT_GD_INIT})")
    p.add_argument("--weight-gd-lora-init", default=None, choices=DEFAULT_INITS,
                   help=f"Init to profile weight_gd_lora at (default: per method, {WEIGHT_GD_LORA_INIT})")
    p.add_argument("--subspace-gd-init", default=None, choices=DEFAULT_INITS,
                   help=f"Init to profile subspace_gd at (default: {SUBSPACE_GD_INIT})")
    p.add_argument("--bo-init", default=None, choices=DEFAULT_INITS,
                   help=f"Box center to profile bo_search at (default: {BO_SEARCH_INIT})")

    # generation (coeff_search's valid pass) -- same defaults as scripts/merge_eval.py
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--gen-batch-size", type=int, default=4,
                   help="generate() batch; keep equal to --gd-batch-size / --subspace-batch-size so every "
                        "strategy's activation memory is comparable (see runs/cost_estimation/ta.sh)")
    p.add_argument("--loss-batch-size", type=int, default=4)

    p.add_argument("--lambda-min", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-steps", type=int, default=10)

    p.add_argument("--gd-epochs", type=int, default=5, help="weight_gd n_actions")
    p.add_argument("--gd-lr", type=float, default=3e-5)
    p.add_argument("--gd-warmup-ratio", type=float, default=0.1)
    p.add_argument("--gd-batch-size", type=int, default=4)
    p.add_argument("--gd-grad-accum-steps", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-lr", type=float, default=3e-4, help="weight_gd_lora learning rate")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default=None, help="comma list; default q/k/v/o + gate/up/down")

    p.add_argument("--subspace-epochs", type=int, default=5, help="subspace_gd n_actions")
    p.add_argument("--subspace-lr", type=float, default=1e-2)
    p.add_argument("--subspace-scheduler", default="cosine")
    p.add_argument("--subspace-warmup-ratio", type=float, default=0.1)
    p.add_argument("--subspace-batch-size", type=int, default=4)
    p.add_argument("--subspace-grad-accum-steps", type=int, default=1)
    p.add_argument("--subspace-basis-device", choices=("auto", "cuda", "cpu"), default="cuda",
                   help="cuda (default): base + N directions resident on the GPU, the layout subspace_gd "
                        "fundamentally needs; auto/cpu spill to host memory to fit smaller cards")
    p.add_argument("--subspace-basis-headroom-gib", type=float, default=6.0)

    p.add_argument("--bo-trials", type=int, default=None,
                   help=f"bo_search n_actions (default: per method, {BO_TRIALS})")
    p.add_argument("--bo-startup-trials", type=int, default=None)
    p.add_argument("--bo-radius", type=float, default=0.7)
    p.add_argument("--bo-lower", type=_opt_float, default=0.0, help="lower clamp; 'none' lifts it")
    p.add_argument("--bo-seed", type=int, default=42)
    p.add_argument("--bo-basis-device", choices=("auto", "cuda", "cpu"), default="cpu",
                   help="cpu (default): W(c) built on the CPU per trial, so bo_search's footprint is the model "
                        "plus generate(), like coeff_search; auto/cuda keep the basis resident for speed")

    p.add_argument("--ties-density", type=float, default=0.2)
    p.add_argument("--dare-drop-rate", type=float, default=0.5)
    p.add_argument("--dare-seed", type=int, default=42)
    p.add_argument("--svd-device", default=None, help="TSV-M SVD device: cuda | cpu")
    p.add_argument("--embedding-svd-device", default=None, help="TSV-M SVD device for embed/lm_head keys")

    p.add_argument("--coeff-source-dir", default="results",
                   help="Results root holding this unit's coeff_search.best_lambda (for coeff_best inits)")
    p.add_argument("--out-dir", default="results/cost_estimation")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if len(tasks) < 2:
        p.error("Provide at least 2 tasks")
    try:
        cfg = build_cfg_from_cli(args)
    except (FileNotFoundError, ValueError) as e:
        p.error(str(e))
    run_cost_estimation(
        cfg, coeff_source_dir=args.coeff_source_dir,
        weight_gd_init=args.weight_gd_init, weight_gd_lora_init=args.weight_gd_lora_init,
        subspace_gd_init=args.subspace_gd_init, bo_init=args.bo_init,
    )


if __name__ == "__main__":
    main()
