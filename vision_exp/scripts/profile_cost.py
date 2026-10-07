#!/usr/bin/env python3
"""Profile one action per strategy and write results/cost_estimation/units/<unit>.json.

Times a warmed-up coeff-search valid eval / GD epoch, records peak GPU memory,
counts vision forwards/backwards, then scales by n_actions (11 / 10 / 20).
Does not evaluate test accuracy — plots read that from results/main_exp.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cost_estimation import build_cfg_from_cli, run_cost_estimation
from src.merging import available_methods


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--method", required=True, help=f"one of {available_methods()}")
    p.add_argument("--tasks", required=True)
    p.add_argument("--task-vectors-dir", default="checkpoints/task_vectors")
    p.add_argument("--task-checkpoints", default=None)
    p.add_argument("--budget", default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", default="coeff_search,weight_gd,subspace_gd")
    p.add_argument("--force", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-steps", type=int, default=11)
    p.add_argument("--gd-epochs", type=int, default=10)
    p.add_argument("--gd-lr", type=float, default=1e-5)
    p.add_argument("--gd-warmup-ratio", type=float, default=0.1)
    p.add_argument("--subspace-epochs", type=int, default=20)
    p.add_argument("--subspace-lr", type=float, default=1e-2)
    p.add_argument("--subspace-warmup-ratio", type=float, default=0.1)
    p.add_argument("--ties-density", type=float, default=0.2)
    p.add_argument("--dare-drop-rate", type=float, default=0.5)
    p.add_argument("--dare-seed", type=int, default=42)
    p.add_argument("--coeff-source-dir", default="results/main_exp",
                   help="Unit JSON root for coeff_search.best_lambda")
    p.add_argument("--out-dir", default="results/cost_estimation")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if len(tasks) < 2:
        p.error("Provide at least 2 tasks")
    cfg = build_cfg_from_cli(args)
    run_cost_estimation(cfg, coeff_source_dir=args.coeff_source_dir)


if __name__ == "__main__":
    main()
