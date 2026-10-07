"""Atomic experiment-unit runner.

One unit = (task_set, arch, merging_method, budget[, seed]). Non-full budgets
append ``__seed{N}`` so data_scaling can average multiple valid-subset draws.
``full`` omits the seed suffix (shared with full_comparison).

The valid pool partition is always fixed (seed 42); ``cfg.seed`` only affects
``apply_valid_budget`` subset selection.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from .checkpoints import resolve_multitask_checkpoint
from .class_subset import select_kept_classes
from .classnames import get_classnames
from .data import (
    apply_valid_budget, build_valid_pool, get_dataloader, get_dataloader_from_dataset,
    load_hf_dataset, parse_budget, restrict_classes,
)
from .merging import get_method
from .models import load_state_dict, shared_float_keys
from . import strategies
from .strategies import StrategyContext
from .strategies.baselines import ALL_CELLS as BASELINE_CELLS
from .strategies.pseudo_labels import result_key as weight_gd_key


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    base_model: str
    arch: str
    method: str
    tasks: List[str]
    task_checkpoints: List[str]
    budget: str = "full"
    seed: int = 42
    strategies: List[str] = field(default_factory=lambda: list(strategies.ALL_STRATEGIES))
    weight_gd_inits: List[str] = field(default_factory=lambda: list(strategies.DEFAULT_INITS))
    subspace_gd_inits: List[str] = field(default_factory=lambda: list(strategies.DEFAULT_INITS))
    ds_inits: List[str] = field(default_factory=lambda: ["coeff_best"])
    bo_inits: List[str] = field(default_factory=lambda: ["coeff_best"])
    force: Set[str] = field(default_factory=set)

    batch_size: int = 32
    test_samples: Optional[int] = None

    lambda_min: float = 0.0
    lambda_max: float = 1.0
    lambda_steps: int = 11

    gd_epochs: int = 10
    gd_lr: float = 1e-5
    gd_warmup_ratio: float = 0.1
    gd_optimizer: str = "adamw"  # adamw | sgd
    gd_momentum: float = 0.0
    gd_patience: int = 5
    gd_l2_sp: float = 0.0
    gd_label_source: str = "gt"  # gt | expert_soft | expert_hard (test-time adaptation)

    # Test-time adaptation: what adamerging / divmerge / weight_gd(expert_*) train on.
    unlabeled_source: str = "test"  # test (AdaMerging-style, transductive) | valid
    unlabeled_samples: Optional[int] = None  # per task; None = the whole split
    divergence: str = "js"  # divmerge + weight_gd expert_soft
    ada_variants: List[str] = field(default_factory=lambda: list(strategies.TTA_VARIANTS))
    div_variants: List[str] = field(default_factory=lambda: list(strategies.TTA_VARIANTS))
    ada_steps: int = 1000
    ada_lr: float = 1e-3
    ada_prior: Optional[float] = None  # None -> 1/N
    div_steps: int = 1000
    div_lr: float = 1e-2
    div_prior: Optional[float] = None

    subspace_epochs: int = 20
    subspace_lr: float = 1e-2
    subspace_warmup_ratio: float = 0.1
    subspace_patience: int = 5

    ds_alpha: float = 0.05
    ds_beta: float = 1e-5
    ds_samples: int = 64
    ds_seed: int = 42
    coeff_source_dir: str = "results/main_exp"

    bo_trials: int = 50
    bo_startup_trials: Optional[int] = None
    bo_radius: float = 0.4
    bo_lower: Optional[float] = 0.0
    bo_seed: int = 42

    method_kwargs: dict = field(default_factory=dict)

    save_checkpoints: bool = False
    out_dir: str = "results/main_exp"
    checkpoint_root: str = "checkpoints/finetuned"
    device: Optional[str] = None

    task_vectors_dir: str = "checkpoints/task_vectors"
    multitask_checkpoint: Optional[str] = None

    # smaller_9task. Both set, or both None. class_seed draws the subset;
    # num_classes is the total number of classes kept across tasks.
    class_seed: Optional[int] = None
    num_classes: Optional[int] = None
    kept_class_indices: Optional[Dict[str, List[int]]] = None

    wandb_project: Optional[str] = "model_merging_main_exp"
    wandb_group: Optional[str] = None


# --------------------------------------------------------------------------- #
# Naming / logging
# --------------------------------------------------------------------------- #
#
# Everything a run produces lives under the results root (cfg.out_dir), split
# into three subfolders:
#   units/  the per-unit JSON results (medium-level output)
#   logs/   the timestamped run logs
#   plots/  the figures drawn from the units (highest-level output; see plot.py)
UNITS_SUBDIR = "units"
LOGS_SUBDIR = "logs"
PLOTS_SUBDIR = "plots"
VALID_POOL_SEED = 42  # fixed train/valid partition; never vary with cfg.seed


def budget_label(budget) -> str:
    b = parse_budget(budget)
    return f"{b}per_class" if isinstance(b, int) else b


def unit_name(cfg: RunConfig) -> str:
    tasks = "-".join(cfg.tasks)
    bl = budget_label(cfg.budget)
    name = f"{len(cfg.tasks)}_tasks_{tasks}"
    name += f"__{cfg.arch}__{cfg.method}__{bl}"
    if bl != "full":
        name += f"__seed{cfg.seed}"
    if cfg.class_seed is not None:
        name += f"__cseed{cfg.class_seed}"
    return name


def apply_class_subset(cfg: RunConfig) -> None:
    """Drop tasks that receive no classes and record the kept label ids.

    No-op unless ``class_seed`` and ``num_classes`` are both set. The valid
    pool, the test split, and the classifier head all use ``kept_class_indices``.
    """
    if cfg.class_seed is None and cfg.num_classes is None:
        return
    if cfg.class_seed is None or cfg.num_classes is None:
        raise ValueError("class_seed and num_classes must be set together")
    if needs_unlabeled(cfg):
        # build_unlabeled_loaders does not restrict classes, so its labels would
        # not match the subset head.
        raise ValueError("a class subset cannot be combined with test-time adaptation strategies")
    kept = select_kept_classes(cfg.tasks, cfg.num_classes, cfg.class_seed)
    if len(kept) < 2:
        raise ValueError(
            f"num_classes={cfg.num_classes} left {len(kept)} task(s); need at least 2"
        )
    dropped = [t for t in cfg.tasks if t not in kept]
    ckpt = dict(zip(cfg.tasks, cfg.task_checkpoints))
    cfg.tasks = [t for t in cfg.tasks if t in kept]
    cfg.task_checkpoints = [ckpt[t] for t in cfg.tasks]
    cfg.kept_class_indices = kept
    if dropped:
        print(f"smaller_9task: dropping tasks with 0 classes: {', '.join(dropped)}")


class _ElapsedFormatter(logging.Formatter):
    """Prefix each line with [HH:MM:SS] elapsed since the formatter was built."""

    def __init__(self) -> None:
        super().__init__("%(message)s")
        self._start = time.monotonic()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = int(time.monotonic() - self._start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        return f"[{h:02d}:{m:02d}:{s:02d}] {super().format(record)}"


def setup_logging(out_dir: str, name: str) -> str:
    log_dir = os.path.join(out_dir, LOGS_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{name}__{stamp}.log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")]
    formatter = _ElapsedFormatter()
    for h in handlers:
        h.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    for name_ in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers", "filelock"):
        logging.getLogger(name_).setLevel(logging.WARNING)
    return log_path


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def needs_unlabeled(cfg: RunConfig) -> bool:
    return (
        any(s in cfg.strategies for s in strategies.TTA_STRATEGIES)
        or ("weight_gd" in cfg.strategies and cfg.gd_label_source != "gt")
    )


def unlabeled_tag(cfg: RunConfig) -> dict:
    """What identifies the unlabeled training data of a unit (checked on merge)."""
    return {"source": cfg.unlabeled_source, "samples_per_task": cfg.unlabeled_samples}


def build_unlabeled_loaders(cfg: RunConfig, transform, valid_loaders):
    """Loaders for the unlabeled strategies + stats. ``valid`` reuses the budget loaders."""
    if cfg.unlabeled_source == "valid":
        loaders = dict(valid_loaders)
    elif cfg.unlabeled_source == "test":
        loaders = {
            task: get_dataloader(
                task, split="test", transform=transform, batch_size=cfg.batch_size,
                shuffle=True, seed=VALID_POOL_SEED, num_samples=cfg.unlabeled_samples,
            )
            for task in cfg.tasks
        }
    else:
        raise ValueError(f"Unknown unlabeled_source '{cfg.unlabeled_source}' (expected test, valid)")
    stats = {**unlabeled_tag(cfg), "n_per_task": {t: len(l.dataset) for t, l in loaders.items()}}
    stats["n_total"] = sum(stats["n_per_task"].values())
    logging.info(
        f"\n=== Unlabeled loaders (source={cfg.unlabeled_source}, "
        f"samples_per_task={cfg.unlabeled_samples or 'all'}) ==="
    )
    for t, n in stats["n_per_task"].items():
        logging.info(f"  {t}: {n} unlabeled samples")
    logging.info(f"  total: {stats['n_total']} unlabeled samples")
    return loaders, stats


def build_budget_loaders(cfg: RunConfig, processor):
    def transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    valid_loaders, test_loaders, stats = {}, {}, {}
    total_used = total_pool = 0
    kept = cfg.kept_class_indices
    logging.info(f"\n=== Loaders (budget={budget_label(cfg.budget)}, select_seed={cfg.seed}) ===")
    for task in cfg.tasks:
        pool = build_valid_pool(task, seed=VALID_POOL_SEED)
        indices = None if kept is None else kept[task]
        if indices is None:
            n_classes = len(get_classnames(task))
        else:
            pool = restrict_classes(pool, indices)
            n_classes = len(indices)
            names = [get_classnames(task)[i] for i in indices]
            logging.info(f"  {task}: {n_classes} classes {names}")
        budget_ds, used, pool_size = apply_valid_budget(pool, cfg.budget, seed=cfg.seed)
        valid_loaders[task] = get_dataloader_from_dataset(
            budget_ds, transform=transform, batch_size=cfg.batch_size, shuffle=True,
        )
        if indices is None:
            test_loaders[task] = get_dataloader(
                task, split="test", transform=transform, batch_size=cfg.batch_size,
                shuffle=False, seed=VALID_POOL_SEED, num_samples=cfg.test_samples,
            )
        else:
            test_ds = restrict_classes(
                load_hf_dataset(task, split="test", num_samples=cfg.test_samples), indices,
            )
            test_loaders[task] = get_dataloader_from_dataset(
                test_ds, transform=transform, batch_size=cfg.batch_size, shuffle=False,
            )
        stats[task] = {
            "valid_pool_size": pool_size, "budget_used": used,
            "num_classes": n_classes, "avg_per_class": used / n_classes,
        }
        total_used += used
        total_pool += pool_size
        logging.info(f"  {task}: using {used} / {pool_size} valid samples ({stats[task]['avg_per_class']:.1f} per class)")
    logging.info(f"  total: using {total_used} / {total_pool} valid samples")
    stats["avg_instances_per_class"] = sum(stats[t]["avg_per_class"] for t in cfg.tasks) / len(cfg.tasks)
    stats["total_budget_used"] = total_used
    stats["total_valid_pool"] = total_pool
    unlabeled_loaders, unlabeled_stats = (
        build_unlabeled_loaders(cfg, transform, valid_loaders) if needs_unlabeled(cfg) else (None, None)
    )
    return valid_loaders, test_loaders, stats, unlabeled_loaders, unlabeled_stats


# --------------------------------------------------------------------------- #
# What to run (merge-into / skip-present / force)
# --------------------------------------------------------------------------- #

def _existing(out: dict) -> dict:
    return out.get("strategies", {})


def _assert_compatible(existing: dict, cfg: RunConfig) -> None:
    checks = {"tasks": cfg.tasks, "method": cfg.method, "budget": budget_label(cfg.budget), "seed": cfg.seed}
    if cfg.class_seed is not None:
        checks["class_seed"] = cfg.class_seed
        checks["num_classes"] = cfg.num_classes
    for field_, expected in checks.items():
        if field_ in existing and existing[field_] != expected:
            raise ValueError(f"Merge mismatch on {field_!r}: file has {existing[field_]!r}, cfg has {expected!r}")
    # Unlabeled cells in one unit must all come from the same unlabeled data;
    # use a different --out-dir for a different sample count / source.
    have = (existing.get("unlabeled_data") or {})
    if have and needs_unlabeled(cfg):
        want = unlabeled_tag(cfg)
        got = {k: have.get(k) for k in want}
        if got != want:
            raise ValueError(f"Merge mismatch on unlabeled data: file has {got!r}, cfg has {want!r}")
    if cfg.kept_class_indices is not None and "kept_classes" in existing:
        got = {t: existing["kept_classes"][t]["indices"] for t in cfg.tasks}
        if got != cfg.kept_class_indices:
            raise ValueError("Merge mismatch on kept class indices")


def _load_coeff_search(cfg: RunConfig) -> Tuple[float, float]:
    """Read coeff_search.best_lambda and valid_avg_acc from coeff_source_dir."""
    src = Path(cfg.coeff_source_dir) / UNITS_SUBDIR / f"{unit_name(cfg)}.json"
    if not src.exists():
        raise ValueError(
            f"directional_sampling needs coeff_search (best_lambda, valid_avg_acc); "
            f"no unit at {src}"
        )
    rec = json.load(open(src))
    cs = rec.get("strategies", {}).get("coeff_search") or {}
    lam = cs.get("best_lambda")
    vacc = cs.get("valid_avg_acc")
    if lam is None:
        raise ValueError(f"no strategies.coeff_search.best_lambda in {src}")
    if vacc is None:
        raise ValueError(f"no strategies.coeff_search.valid_avg_acc in {src}")
    return float(lam), float(vacc)


def _plan(cfg: RunConfig, out: dict, mt_ckpt: Optional[str]):
    """Return (run_coeff, weight_inits, subspace_inits, baseline_cells, ds_inits, bo_inits,
    ada_variants, div_variants).

    ``baseline_cells`` is the subset of :data:`BASELINE_CELLS` that this run
    should compute (missing from the JSON, or all when forced). ``upper_bound``
    is only proposed when a multi-task checkpoint has actually resolved.
    """
    done = _existing(out)
    force = cfg.force

    run_coeff = ("coeff_search" in cfg.strategies and
                 ("coeff_search" in force or "coeff_search" not in done))
    # Baselines are budget-independent test-only references; compute once per
    # (tasks, arch, method) at the 'full' budget, then plots pull them from the
    # full sibling for every other budget. The set of cells is computed
    # incrementally so a later rerun after the MT checkpoint appears just fills
    # in the missing upper_bound cell.
    baseline_cells: List[str] = []
    if "baselines" in cfg.strategies and budget_label(cfg.budget) == "full":
        have = {} if "baselines" in force else done.get("baselines", {})
        for cell in BASELINE_CELLS:
            if cell == "upper_bound" and mt_ckpt is None:
                continue
            if cell not in have:
                baseline_cells.append(cell)

    def inits_to_run(strat: str, requested: List[str], done_key: Optional[str] = None) -> List[str]:
        if strat not in cfg.strategies:
            return []
        have = done.get(done_key or strat, {})
        return [i for i in requested if (strat in force or i not in have)]

    return (
        run_coeff,
        # weight_gd cells are keyed by label source: weight_gd | weight_gd_expert_*
        inits_to_run("weight_gd", cfg.weight_gd_inits, weight_gd_key(cfg.gd_label_source)),
        inits_to_run("subspace_gd", cfg.subspace_gd_inits),
        baseline_cells,
        inits_to_run("directional_sampling", cfg.ds_inits),
        inits_to_run("bo_search", cfg.bo_inits),
        inits_to_run("adamerging", cfg.ada_variants),
        inits_to_run("divmerge", cfg.div_variants),
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_experiment(cfg: RunConfig) -> Optional[str]:
    apply_class_subset(cfg)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = unit_name(cfg)
    out_path = Path(cfg.out_dir) / UNITS_SUBDIR / f"{name}.json"

    output = {}
    if out_path.exists():
        output = json.load(open(out_path))
        _assert_compatible(output, cfg)

    # Resolve the multi-task checkpoint before planning: it decides whether the
    # upper_bound baseline cell can be filled on this run.
    mt_ckpt = cfg.multitask_checkpoint
    if mt_ckpt is None and "baselines" in cfg.strategies and budget_label(cfg.budget) == "full":
        mt_ckpt = resolve_multitask_checkpoint(cfg.task_vectors_dir, cfg.tasks, cfg.arch)

    (run_coeff, weight_inits, subspace_inits, baseline_cells, ds_inits, bo_inits,
     ada_variants, div_variants) = _plan(cfg, output, mt_ckpt)

    # coeff_best inits need a best_lambda from somewhere. bo_search can also fall
    # back to coeff_source_dir (like directional_sampling), so it is excluded here.
    have_coeff = run_coeff or ("coeff_search" in _existing(output))
    if ("coeff_best" in weight_inits or "coeff_best" in subspace_inits) and not have_coeff:
        raise ValueError(
            "init 'coeff_best' requires coeff_search results; include coeff_search in --strategies "
            "(or run it first into this JSON)."
        )

    if not (run_coeff or weight_inits or subspace_inits or baseline_cells or ds_inits or bo_inits
            or ada_variants or div_variants):
        skipping_baselines = (
            "baselines" in cfg.strategies
            and budget_label(cfg.budget) != "full"
            and "baselines" not in _existing(output)
        )
        note = " (baselines skipped: run at budget=full only)" if skipping_baselines else ""
        print(f"Nothing to run for {out_path} (all requested cells present; use --force to rerun){note}")
        return None

    from . import wandb_util

    log_path = setup_logging(cfg.out_dir, name)
    logging.info(f"Log -> {log_path}")
    logging.info(f"Unit: {name}")
    if cfg.kept_class_indices is not None:
        summary = ", ".join(f"{t}={len(ids)}" for t, ids in cfg.kept_class_indices.items())
        logging.info(f"Class subset: num_classes={cfg.num_classes}  cseed={cfg.class_seed}  {summary}")
    logging.info(
        f"Planned: coeff={run_coeff}  {weight_gd_key(cfg.gd_label_source)}={weight_inits}  "
        f"subspace_gd={subspace_inits}  baselines={baseline_cells}  "
        f"directional_sampling={ds_inits}  bo_search={bo_inits}  "
        f"adamerging={ada_variants}  divmerge={div_variants}"
    )
    if mt_ckpt and mt_ckpt != cfg.multitask_checkpoint:
        logging.info(f"Auto-resolved multitask checkpoint -> {mt_ckpt}")

    wandb_util.init_run(cfg, name)
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg.base_model)
        valid_loaders, test_loaders, data_stats, unlabeled_loaders, unlabeled_stats = (
            build_budget_loaders(cfg, processor)
        )

        logging.info("\n=== Loading checkpoints ===")
        base_sd = load_state_dict(cfg.base_model, device="cpu")
        task_sds = [load_state_dict(cp, device="cpu") for cp in cfg.task_checkpoints]
        keys = shared_float_keys(base_sd, *task_sds)
        method = get_method(cfg.method, **cfg.method_kwargs)

        bl = budget_label(cfg.budget)
        ckpt_dir = os.path.join(
            cfg.checkpoint_root, cfg.method, cfg.arch, "-".join(cfg.tasks),
            bl,
        )
        if bl != "full":
            ckpt_dir = os.path.join(ckpt_dir, f"seed{cfg.seed}")

        ctx = StrategyContext(
            method=method, base_model=cfg.base_model, base_sd=base_sd, task_sds=task_sds, keys=keys,
            tasks=cfg.tasks,
            valid_loaders=valid_loaders, test_loaders=test_loaders,
            device=device,
            lambda_min=cfg.lambda_min, lambda_max=cfg.lambda_max, lambda_steps=cfg.lambda_steps,
            gd_epochs=cfg.gd_epochs, gd_lr=cfg.gd_lr, gd_warmup_ratio=cfg.gd_warmup_ratio,
            gd_optimizer=cfg.gd_optimizer, gd_momentum=cfg.gd_momentum, gd_patience=cfg.gd_patience,
            gd_l2_sp=cfg.gd_l2_sp,
            gd_label_source=cfg.gd_label_source,
            unlabeled_loaders=unlabeled_loaders, unlabeled_stats=unlabeled_stats, divergence=cfg.divergence,
            ada_steps=cfg.ada_steps, ada_lr=cfg.ada_lr, ada_prior=cfg.ada_prior,
            div_steps=cfg.div_steps, div_lr=cfg.div_lr, div_prior=cfg.div_prior,
            subspace_epochs=cfg.subspace_epochs, subspace_lr=cfg.subspace_lr,
            subspace_warmup_ratio=cfg.subspace_warmup_ratio, subspace_patience=cfg.subspace_patience,
            ds_alpha=cfg.ds_alpha, ds_beta=cfg.ds_beta, ds_samples=cfg.ds_samples, ds_seed=cfg.ds_seed,
            bo_trials=cfg.bo_trials, bo_startup_trials=cfg.bo_startup_trials, bo_radius=cfg.bo_radius,
            bo_lower=cfg.bo_lower, bo_seed=cfg.bo_seed,
            save_checkpoints=cfg.save_checkpoints, checkpoint_dir=ckpt_dir,
            multitask_checkpoint=mt_ckpt,
            class_indices=cfg.kept_class_indices,
        )

        output.update({
            "tasks": cfg.tasks, "arch": cfg.arch, "method": cfg.method,
            "budget": bl, "seed": cfg.seed, "base_model": cfg.base_model,
            "data_stats": data_stats,
        })
        if unlabeled_stats is not None:
            output["unlabeled_data"] = unlabeled_stats
        if cfg.kept_class_indices is not None:
            output["class_seed"] = cfg.class_seed
            output["num_classes"] = cfg.num_classes
            output["kept_classes"] = {
                t: {
                    "indices": ids,
                    "names": [get_classnames(t)[i] for i in ids],
                }
                for t, ids in cfg.kept_class_indices.items()
            }
        strat_out = output.setdefault("strategies", {})

        if "coeff_search" in strat_out:
            ctx.best_lambda = strat_out["coeff_search"].get("best_lambda")
            ctx.coeff_best_valid_avg_acc = strat_out["coeff_search"].get("valid_avg_acc")

        def _persist(reason: str) -> None:
            output["config"] = _config_summary(cfg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2, default=float)
            logging.info(f"  [saved] {reason} -> {out_path.name}")

        if run_coeff:
            strat_out["coeff_search"] = strategies.coeff_search.run(ctx)
            ctx.best_lambda = strat_out["coeff_search"]["best_lambda"]
            ctx.coeff_best_valid_avg_acc = strat_out["coeff_search"]["valid_avg_acc"]
            wandb_util.log_result("coeff_search", strat_out["coeff_search"])
            _persist("coeff_search")

        wg_key = weight_gd_key(cfg.gd_label_source)
        for init in weight_inits:
            dst = strat_out.setdefault(wg_key, {})
            dst[init] = strategies.weight_gd.run_one(ctx, init)
            wandb_util.log_result(f"{wg_key}/{init}", dst[init])
            _persist(f"{wg_key}/{init}")

        for name, variants in (("adamerging", ada_variants), ("divmerge", div_variants)):
            for variant in variants:
                dst = strat_out.setdefault(name, {})
                dst[variant] = strategies.test_time_adaptation.run_one(ctx, name, variant)
                wandb_util.log_result(f"{name}/{variant}", dst[variant])
                _persist(f"{name}/{variant}")

        for init in subspace_inits:
            dst = strat_out.setdefault("subspace_gd", {})
            dst[init] = strategies.subspace_gd.run_one(ctx, init)
            wandb_util.log_result(f"subspace_gd/{init}", dst[init])
            _persist(f"subspace_gd/{init}")

        if baseline_cells:
            new_cells = strategies.baselines.run(ctx, ctx.multitask_checkpoint, cells=baseline_cells)
            strat_out.setdefault("baselines", {}).update(new_cells)
            for cell, res in new_cells.items():
                wandb_util.log_result(f"baselines/{cell}", res)
            _persist("baselines")

        if ds_inits and (ctx.best_lambda is None or ctx.coeff_best_valid_avg_acc is None):
            lam, vacc = _load_coeff_search(cfg)
            if ctx.best_lambda is None:
                ctx.best_lambda = lam
            if ctx.coeff_best_valid_avg_acc is None:
                ctx.coeff_best_valid_avg_acc = vacc
            logging.info(
                f"directional_sampling: λ*={ctx.best_lambda:.4f}  "
                f"coeff_best_valid_avg_acc={ctx.coeff_best_valid_avg_acc:.4f} "
                f"from {cfg.coeff_source_dir}"
            )
        for init in ds_inits:
            dst = strat_out.setdefault("directional_sampling", {})
            dst[init] = strategies.directional_sampling.run_one(ctx, init)
            wandb_util.log_result(f"directional_sampling/{init}", dst[init])
            _persist(f"directional_sampling/{init}")

        if "coeff_best" in bo_inits and ctx.best_lambda is None:
            ctx.best_lambda, _ = _load_coeff_search(cfg)
            logging.info(f"bo_search: λ*={ctx.best_lambda:.4f} from {cfg.coeff_source_dir}")
        for init in bo_inits:
            dst = strat_out.setdefault("bo_search", {})
            dst[init] = strategies.bo_search.run_one(ctx, init)
            wandb_util.log_result(f"bo_search/{init}", dst[init])
            _persist(f"bo_search/{init}")

        logging.info(f"\nDone -> {out_path}")
        return str(out_path)
    finally:
        wandb_util.finish()


def _config_summary(cfg: RunConfig) -> dict:
    d = dict(cfg.__dict__)
    d["force"] = sorted(cfg.force)
    d["budget"] = budget_label(cfg.budget)
    return d
