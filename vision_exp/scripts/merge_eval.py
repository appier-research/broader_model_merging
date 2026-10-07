#!/usr/bin/env python3
"""Run one merging experiment unit = (tasks, arch, method, budget[, seed]).

Evaluates the requested strategies (coeff_search, weight_gd, subspace_gd,
baselines) and merges the results into a deterministic JSON so that
full_comparison and data_scaling share overlapping cells. ``baselines`` is
test-only and budget-independent; the runner computes it once at budget=full and
the plots reuse those values as horizontal references at every other budget.
See the README for end-to-end usage.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment import RunConfig, run_experiment
from src import strategies
from src.merging import available_methods
from src.checkpoints import resolve_task_checkpoints


def _parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _opt_float(s: str):
    """Float CLI value, or None for 'none' (used by --bo-lower to lift the clamp)."""
    return None if s.strip().lower() in ("none", "") else float(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="HF id of the pretrained base (e.g. openai/clip-vit-base-patch32)")
    p.add_argument("--arch", required=True, help="Short arch tag used in filenames (e.g. vit-b-32)")
    p.add_argument("--method", required=True, help=f"Merging method: one of {available_methods()}")
    p.add_argument("--tasks", required=True, help="Comma-separated task labels (merged / selected on)")
    p.add_argument("--task-vectors-dir", default="checkpoints/task_vectors",
                   help="Folder holding all task-vector checkpoints; resolved per (task, arch)")
    p.add_argument("--task-checkpoints", default=None,
                   help="Explicit comma-separated checkpoint paths (aligned with --tasks); overrides --task-vectors-dir")
    p.add_argument("--budget", default="full", help="Valid budget: full | medium | low | <k> | <k>_per_class")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--strategies", default=",".join(strategies.ALL_STRATEGIES),
                   help=f"Comma-separated strategies from {list(strategies.ALL_STRATEGIES)}")
    p.add_argument("--weight-gd-inits", default=",".join(strategies.DEFAULT_INITS),
                   help=f"Init points for weight_gd from {list(strategies.DEFAULT_INITS)} "
                        f"(also random_<seed> for true-random weights)")
    p.add_argument("--subspace-gd-inits", default=",".join(strategies.DEFAULT_INITS),
                   help=f"Init points for subspace_gd from {list(strategies.DEFAULT_INITS)}")
    p.add_argument("--ds-inits", default="coeff_best",
                   help=f"Init points for directional_sampling from {list(strategies.DEFAULT_INITS)}")
    p.add_argument("--force", default="", help="Comma-separated strategies to recompute even if present")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-samples", type=int, default=None, help="Cap test set per task (shuffled); None = full")

    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-steps", type=int, default=11)

    p.add_argument("--gd-epochs", type=int, default=10)
    p.add_argument("--gd-lr", type=float, default=1e-5)
    p.add_argument("--gd-warmup-ratio", type=float, default=0.1)
    p.add_argument("--gd-optimizer", choices=("adamw", "sgd"), default="adamw")
    p.add_argument("--gd-momentum", type=float, default=0.0,
                   help="SGD momentum for weight_gd (0 = vanilla SGD)")
    p.add_argument("--gd-patience", type=int, default=5,
                   help="Early-stop weight_gd if train loss does not improve for this many epochs (0 = off)")
    p.add_argument("--gd-l2-sp", type=float, default=0.0,
                   help="L2-SP on weight_gd: CE + (alpha/2)||W-W_init||^2 (0 = off)")
    p.add_argument("--gd-label-source", choices=strategies.LABEL_SOURCES, default="gt",
                   help="weight_gd training signal: gt labels (valid budget), or task-expert pseudo-labels on "
                        "the unlabeled data (soft = divergence to expert logits, hard = expert argmax). "
                        "Non-gt results go under weight_gd_<source>.")

    p.add_argument("--unlabeled-source", choices=("test", "valid"), default="test",
                   help="Data the unlabeled strategies (adamerging, divmerge, weight_gd expert_*) train on")
    p.add_argument("--unlabeled-samples", type=int, default=None,
                   help="Unlabeled samples per task (None = whole split); same for every unlabeled strategy")
    p.add_argument("--divergence", choices=strategies.DIVERGENCES, default="js",
                   help="Divergence to expert logits for divmerge and weight_gd expert_soft")
    p.add_argument("--ada-variants", default=",".join(strategies.TTA_VARIANTS),
                   help=f"adamerging coefficient granularity from {list(strategies.TTA_VARIANTS)}")
    p.add_argument("--div-variants", default=",".join(strategies.TTA_VARIANTS),
                   help=f"divmerge coefficient granularity from {list(strategies.TTA_VARIANTS)}")
    p.add_argument("--ada-steps", type=int, default=1000, help="adamerging steps (one batch per task each)")
    p.add_argument("--ada-lr", type=float, default=1e-3)
    p.add_argument("--ada-prior", type=float, default=None,
                   help="adamerging coefficient init (default 1/N; the paper uses 0.3)")
    p.add_argument("--div-steps", type=int, default=1000, help="divmerge steps (one batch per task each)")
    p.add_argument("--div-lr", type=float, default=1e-2)
    p.add_argument("--div-prior", type=float, default=None, help="divmerge coefficient init (default 1/N)")

    p.add_argument("--subspace-epochs", type=int, default=20)
    p.add_argument("--subspace-lr", type=float, default=1e-2)
    p.add_argument("--subspace-warmup-ratio", type=float, default=0.1)
    p.add_argument("--subspace-patience", type=int, default=5,
                   help="Early-stop subspace_gd if train loss does not improve for this many epochs (0 = off)")

    p.add_argument("--ds-alpha", type=float, default=0.05,
                   help="Half-width of Unif[λ* ± alpha] for each basis coefficient")
    p.add_argument("--ds-beta", type=float, default=1e-5,
                   help="Max coefficient on -g_⊥ (β ~ Unif[0, beta])")
    p.add_argument("--ds-samples", type=int, default=64, help="Number of random (α, β) draws")
    p.add_argument("--ds-seed", type=int, default=42, help="RNG seed for directional_sampling")
    p.add_argument("--coeff-source-dir", default="results/main_exp",
                   help="Results root to read coeff_search.best_lambda from (same unit name)")

    p.add_argument("--bo-inits", default="coeff_best",
                   help=f"Box centers for bo_search from {list(strategies.BO_INITS)} (default: coeff_best only)")
    p.add_argument("--bo-trials", type=int, default=50,
                   help="bo_search: total trials (each = one valid forward pass), startup included")
    p.add_argument("--bo-startup-trials", type=int, default=None,
                   help="bo_search: trials before the GP starts proposing (default: 2 * n_dirs + 1)")
    p.add_argument("--bo-radius", type=float, default=0.4,
                   help="bo_search: half-width of the per-coefficient box around the init center")
    p.add_argument("--bo-lower", type=_opt_float, default=0.0,
                   help="bo_search: clamp every coefficient's lower bound (default 0.0; 'none' = no clamp)")
    p.add_argument("--bo-seed", type=int, default=42, help="bo_search: GPSampler / startup-draw seed")

    p.add_argument("--ties-density", type=float, default=0.2, help="TIES trim density (fraction kept)")
    p.add_argument("--dare-drop-rate", type=float, default=0.5, help="DARE drop rate (fraction zeroed)")
    p.add_argument("--dare-seed", type=int, default=42, help="DARE drop-mask seed")

    p.add_argument("--save-checkpoints", action="store_true", help="Save weight_gd / subspace_gd checkpoints (named by init)")
    p.add_argument("--multitask-checkpoint", default=None,
                   help="Multi-task ckpt for the 'baselines' upper_bound (defaults to auto-resolve inside --task-vectors-dir)")
    p.add_argument("--class-seed", type=int, default=None,
                   help="smaller_9task: seed for the class subset (requires --num-classes)")
    p.add_argument("--num-classes", type=int, default=None,
                   help="smaller_9task: total classes kept across tasks (requires --class-seed)")
    p.add_argument("--out-dir", default="results/main_exp")
    p.add_argument("--device", default=None)
    p.add_argument("--wandb-project", default="model_merging_main_exp",
                   help="W&B project (empty / --no-wandb to disable); WANDB_MODE defaults to online")
    p.add_argument("--wandb-group", default=None, help="W&B group (e.g. data_scaling/vit-b-32/ta)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    tasks = _parse_list(args.tasks)
    if len(tasks) < 2:
        p.error("Provide at least 2 tasks")

    if args.task_checkpoints:
        task_ckpts = _parse_list(args.task_checkpoints)
        if len(tasks) != len(task_ckpts):
            p.error(f"--tasks ({len(tasks)}) and --task-checkpoints ({len(task_ckpts)}) must align")
    else:
        try:
            task_ckpts = resolve_task_checkpoints(args.task_vectors_dir, tasks, args.arch)
        except (FileNotFoundError, ValueError) as e:
            p.error(str(e))
        for t, c in zip(tasks, task_ckpts):
            print(f"  resolved {t} -> {c}")

    method_kwargs = {}
    if args.method.lower() == "ties":
        method_kwargs["density"] = args.ties_density
    elif args.method.lower() == "dare":
        method_kwargs["drop_rate"] = args.dare_drop_rate
        method_kwargs["seed"] = args.dare_seed

    cfg = RunConfig(
        base_model=args.base_model, arch=args.arch, method=args.method,
        tasks=tasks, task_checkpoints=task_ckpts,
        budget=args.budget, seed=args.seed,
        strategies=_parse_list(args.strategies),
        weight_gd_inits=_parse_list(args.weight_gd_inits),
        subspace_gd_inits=_parse_list(args.subspace_gd_inits),
        ds_inits=_parse_list(args.ds_inits),
        bo_inits=_parse_list(args.bo_inits),
        force=set(_parse_list(args.force)),
        batch_size=args.batch_size, test_samples=args.test_samples,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max, lambda_steps=args.lambda_steps,
        gd_epochs=args.gd_epochs, gd_lr=args.gd_lr, gd_warmup_ratio=args.gd_warmup_ratio,
        gd_optimizer=args.gd_optimizer, gd_momentum=args.gd_momentum, gd_patience=args.gd_patience,
        gd_l2_sp=args.gd_l2_sp,
        gd_label_source=args.gd_label_source,
        unlabeled_source=args.unlabeled_source, unlabeled_samples=args.unlabeled_samples,
        divergence=args.divergence,
        ada_variants=_parse_list(args.ada_variants), div_variants=_parse_list(args.div_variants),
        ada_steps=args.ada_steps, ada_lr=args.ada_lr, ada_prior=args.ada_prior,
        div_steps=args.div_steps, div_lr=args.div_lr, div_prior=args.div_prior,
        subspace_epochs=args.subspace_epochs, subspace_lr=args.subspace_lr,
        subspace_warmup_ratio=args.subspace_warmup_ratio, subspace_patience=args.subspace_patience,
        ds_alpha=args.ds_alpha, ds_beta=args.ds_beta, ds_samples=args.ds_samples, ds_seed=args.ds_seed,
        coeff_source_dir=args.coeff_source_dir,
        bo_trials=args.bo_trials, bo_startup_trials=args.bo_startup_trials, bo_radius=args.bo_radius,
        bo_lower=args.bo_lower, bo_seed=args.bo_seed,
        method_kwargs=method_kwargs, save_checkpoints=args.save_checkpoints,
        out_dir=args.out_dir, device=args.device,
        task_vectors_dir=args.task_vectors_dir,
        multitask_checkpoint=args.multitask_checkpoint,
        class_seed=args.class_seed, num_classes=args.num_classes,
        wandb_project=None if args.no_wandb or not args.wandb_project else args.wandb_project,
        wandb_group=args.wandb_group,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
