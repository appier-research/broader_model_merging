"""Atomic experiment-unit runner.

One unit = (task_set, arch, merging_method, budget). It holds a JSON of
per-strategy results. Same unit/merge-into-JSON pattern as vision_exp's
experiment.py: builds valid/test pools at the requested budget, loads the base
model and single-task checkpoints (task vectors), runs only the
strategy/init cells not already present (unless forced), and writes results to
a deterministic JSON and a timestamped log.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch

from . import data
from . import models
from . import strategies
from .checkpoints import resolve_multitask_checkpoint
from .merging import get_method
from .strategies import StrategyContext
from .strategies.baselines import ALL_CELLS as BASELINE_CELLS


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
    # Held-out tasks (runs/ood_generalization): test-only, no task vector, no
    # valid pool, never part of any selection criterion. Every strategy's
    # final weights get an extra unseen_test_* block in the unit JSON.
    unseen_tasks: List[str] = field(default_factory=list)
    # Cap per unseen task's test pool (None = full, or falls back to
    # test_samples). gsm8k/mbpp decode 512 tokens per example, so this is the
    # knob that keeps an OOD pass affordable.
    unseen_test_samples: Optional[int] = None
    # Few-shot prompting for the unseen tasks: None/0 = zero-shot (the
    # protocol of every seen task), "auto" = each task's DEFAULT_N_SHOT
    # (gsm8k 4, mbpp 3), or an int for all. Zero-shot on a held-out task
    # mostly measures format compliance (does the cell still emit \boxed{} /
    # a code block), which ifeval already scores -- see design.md "OOD generalization".
    unseen_n_shot: Optional[str] = None
    # Build the unseen prompts as raw completions (no chat template): the
    # pretrained *-Base reference only. Results land in unseen_raw_* fields.
    unseen_raw_prompt: bool = False
    budget: str = "full"
    seed: int = 42
    dtype: str = "bfloat16"
    strategies: List[str] = field(default_factory=lambda: list(strategies.ALL_STRATEGIES))
    weight_gd_inits: List[str] = field(default_factory=lambda: list(strategies.DEFAULT_INITS))
    weight_gd_lora_inits: List[str] = field(default_factory=lambda: list(strategies.DEFAULT_INITS))
    subspace_gd_inits: List[str] = field(default_factory=lambda: list(strategies.DEFAULT_INITS))
    force: Set[str] = field(default_factory=set)

    test_samples: Optional[int] = None

    # generation (used by coeff_search's sweep and every strategy's final eval)
    # None -- the default -- uses each task's own MAX_NEW_TOKENS (see
    # src/tasks/*.py); set an int to override every task uniformly.
    max_new_tokens: Optional[int] = None
    temperature: float = 0.01
    top_p: float = 0.95
    gen_batch_size: int = 16

    lambda_min: float = 0.1
    lambda_max: float = 1.0
    lambda_steps: int = 10

    gd_epochs: int = 5
    # weight_gd's LR. weight_gd_lora uses lora_lr below, not this one.
    gd_lr: float = 3e-5
    gd_warmup_ratio: float = 0.1
    gd_batch_size: int = 4
    gd_grad_accum_steps: int = 1
    max_seq_length: int = 2048
    loss_batch_size: int = 4

    lora_lr: float = 3e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    subspace_epochs: int = 5
    subspace_lr: float = 1e-2
    subspace_scheduler: str = "cosine"
    subspace_warmup_ratio: float = 0.1
    subspace_batch_size: int = 4
    subspace_grad_accum_steps: int = 1
    subspace_basis_device: str = "auto"
    subspace_basis_headroom_gib: float = 6.0

    ds_inits: List[str] = field(default_factory=lambda: ["coeff_best"])
    ds_alpha: float = 0.05
    ds_beta: float = 0.0
    ds_samples: int = 64
    ds_seed: int = 42
    ds_select_metric: str = "loss"
    ds_eval_test: bool = False
    ds_basis_device: str = "auto"
    ds_reeval_center: bool = False
    # Results root the directional_sampling density threshold (and lambda*) is
    # read from when this unit's own JSON has no coeff_search -- the ablation
    # writes to a side --out-dir, so it points back at the main results root.
    coeff_source_dir: str = "results"

    bo_inits: List[str] = field(default_factory=lambda: ["coeff_best"])
    bo_trials: int = 50
    bo_startup_trials: Optional[int] = None
    bo_radius: float = 0.7
    bo_lower: Optional[float] = 0.0
    bo_seed: int = 42
    bo_basis_device: str = "auto"

    method_kwargs: dict = field(default_factory=dict)

    save_checkpoints: bool = False
    out_dir: str = "results"
    checkpoint_root: str = "checkpoints/finetuned"
    device: Optional[str] = None

    # Multi-task upper-bound baseline: explicit override wins; otherwise the
    # runner auto-resolves inside task_vectors_dir. Only used by 'baselines'.
    task_vectors_dir: str = "checkpoints/task_vectors"
    multitask_checkpoint: Optional[str] = None


# --------------------------------------------------------------------------- #
# Naming / logging
# --------------------------------------------------------------------------- #

UNITS_SUBDIR = "units"
LOGS_SUBDIR = "logs"


def budget_label(budget) -> str:
    b = data.parse_budget(budget)
    return f"{b}per_class" if isinstance(b, int) else b


def unit_name(cfg: RunConfig) -> str:
    tasks = "-".join(cfg.tasks)
    name = f"{len(cfg.tasks)}_tasks_{tasks}"
    if cfg.unseen_tasks:
        name += f"__unseen_{'-'.join(cfg.unseen_tasks)}"
        if unseen_shots_enabled(cfg):
            name += "__fewshot"
    return name + f"__{cfg.arch}__{cfg.method}__{budget_label(cfg.budget)}"


def unseen_shots_enabled(cfg: RunConfig) -> bool:
    return any(data.resolve_n_shot(t, cfg.unseen_n_shot) > 0 for t in cfg.unseen_tasks)


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
    for name_ in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers", "filelock", "datasets"):
        logging.getLogger(name_).setLevel(logging.WARNING)
    return log_path


# --------------------------------------------------------------------------- #
# Data pools
# --------------------------------------------------------------------------- #

def build_budget_pools(cfg: RunConfig):
    valid_examples, test_examples, sft_pools, stats = {}, {}, {}, {}
    total_used = total_pool = 0
    logging.info(f"\n=== Data (budget={budget_label(cfg.budget)}) ===")
    for task in cfg.tasks:
        valid_pool, test_pool = data.build_valid_test(task, seed=cfg.seed)
        budget_examples, used, pool_size = data.apply_valid_budget(valid_pool, cfg.budget, seed=cfg.seed)

        if cfg.test_samples is not None and cfg.test_samples < len(test_pool):
            rng = random.Random(cfg.seed)
            idx = sorted(rng.sample(range(len(test_pool)), cfg.test_samples))
            test_pool = [test_pool[i] for i in idx]

        valid_examples[task] = budget_examples
        test_examples[task] = test_pool
        sft_pools[task] = data.build_sft_examples(task, budget_examples)

        stats[task] = {
            "valid_pool_size": pool_size, "budget_used": used,
            "test_size": len(test_pool), "sft_pool_size": len(sft_pools[task]),
        }
        total_used += used
        total_pool += pool_size
        logging.info(
            f"  {task}: using {used}/{pool_size} valid, {len(test_pool)} test, "
            f"{len(sft_pools[task])} sft examples"
        )
    logging.info(f"  total: using {total_used} / {total_pool} valid samples")
    stats["total_budget_used"] = total_used
    stats["total_valid_pool"] = total_pool
    return valid_examples, test_examples, sft_pools, stats


def _cap(pool: list, n: Optional[int], seed: int) -> list:
    if n is not None and n < len(pool):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(pool)), n))
        return [pool[i] for i in idx]
    return pool


def build_unseen_test_pools(cfg: RunConfig, stats: Optional[dict] = None):
    """Test pools for the held-out tasks: no valid pool, no SFT data, no GD.
    The pool is exactly what the task would get as a *seen* task (same
    fixed-seed split), so a task's number means the same thing either way."""
    if not cfg.unseen_tasks:
        return {}
    n = cfg.unseen_test_samples if cfg.unseen_test_samples is not None else cfg.test_samples
    out = {}
    logging.info("\n=== Unseen test pools (eval only) ===")
    for task in cfg.unseen_tasks:
        _, test_pool = data.build_valid_test(task, seed=cfg.seed)
        full = len(test_pool)
        pool = _cap(test_pool, n, cfg.seed)
        k = data.resolve_n_shot(task, cfg.unseen_n_shot)
        pool = data.apply_few_shot(task, pool, k, seed=cfg.seed, raw=cfg.unseen_raw_prompt)
        out[task] = pool
        if stats is not None:
            stats[task] = {"unseen": True, "test_size": len(pool), "test_pool_size": full, "n_shot": k,
                           "raw_prompt": bool(cfg.unseen_raw_prompt)}
        logging.info(f"  {task}: {len(pool)}/{full} test (unseen, {k}-shot{', raw prompt' if cfg.unseen_raw_prompt else ''})")
    return out


# --------------------------------------------------------------------------- #
# What to run (merge-into / skip-present / force)
# --------------------------------------------------------------------------- #

def _existing(out: dict) -> dict:
    return out.get("strategies", {})


def _assert_compatible(existing: dict, cfg: RunConfig) -> None:
    checks = {"tasks": cfg.tasks, "method": cfg.method, "budget": budget_label(cfg.budget), "seed": cfg.seed}
    if cfg.unseen_tasks or "unseen_tasks" in existing:
        checks["unseen_tasks"] = cfg.unseen_tasks
        checks["unseen_n_shot"] = {t: data.resolve_n_shot(t, cfg.unseen_n_shot) for t in cfg.unseen_tasks}
    for field_, expected in checks.items():
        if field_ in existing and existing[field_] != expected:
            raise ValueError(f"Merge mismatch on {field_!r}: file has {existing[field_]!r}, cfg has {expected!r}")


def _load_coeff_search(cfg: RunConfig) -> dict:
    """Read this unit's coeff_search cell from ``coeff_source_dir``.

    directional_sampling runs into a side results root (so the ablation's
    samples[] never lands in the main units), but its density threshold and
    its ``coeff_best`` center both come from the main run's coeff_search.
    """
    src = Path(cfg.coeff_source_dir) / UNITS_SUBDIR / f"{unit_name(cfg)}.json"
    if not src.exists():
        raise ValueError(
            f"directional_sampling needs coeff_search (best_lambda + valid metrics); no unit at {src}. "
            f"Run coeff_search into it first, or point --coeff-source-dir at a results root that has it."
        )
    cs = json.load(open(src)).get("strategies", {}).get("coeff_search") or {}
    if cs.get("best_lambda") is None:
        raise ValueError(f"no strategies.coeff_search.best_lambda in {src}")
    return cs


def _load_source_strategies(cfg: RunConfig) -> dict:
    """The whole ``strategies`` dict of this unit in ``coeff_source_dir`` (or
    {} when absent) -- directional_sampling's 'subspace_best' / 'bo_best'
    centers read subspace_gd / bo_search out of it, the same way 'coeff_best'
    reads coeff_search."""
    src = Path(cfg.coeff_source_dir) / UNITS_SUBDIR / f"{unit_name(cfg)}.json"
    if not src.exists():
        return {}
    return json.load(open(src)).get("strategies", {})


def _best_vector_coeffs(strat: dict, strategy: str) -> tuple:
    """Pick ``strategy``'s (subspace_gd or bo_search) best-performing cell and
    return its coefficient vector and recorded scores:
    (coeffs, source_init, valid_avg_score, test_avg_score) or (None,)*4.

    "Best" is by valid_avg_score -- the same selection signal every other
    strategy uses, and the only one available without re-running anything.
    Both strategies record the vector as ``coefficients_final`` (subspace_gd's
    trained coefficients, bo_search's best trial). Cells that never recorded
    it (an older partial) are skipped rather than silently centering the box
    on nothing.
    """
    best, best_score, best_init, best_test = None, None, None, None
    for init, cell in (strat.get(strategy) or {}).items():
        coeffs = cell.get("coefficients_final")
        score = cell.get("valid_avg_score")
        if not coeffs or score is None:
            continue
        if best_score is None or score > best_score:
            best, best_score, best_init, best_test = coeffs, score, init, cell.get("test_avg_score")
    return best, best_init, best_score, best_test


def _plan(cfg: RunConfig, out: dict, mt_ckpt: Optional[str]):
    """Return (run_coeff, weight_inits, weight_lora_inits, subspace_inits, baseline_cells, ds_inits,
    bo_inits) to compute.

    ``baseline_cells`` is the subset of :data:`BASELINE_CELLS` that this run
    should compute (missing from the JSON, or all when forced). ``upper_bound``
    is only proposed when a multi-task checkpoint has actually resolved.

    ``force`` is all-or-nothing per strategy: it recomputes every requested init
    of that strategy.
    """
    done = _existing(out)
    force = cfg.force

    run_coeff = ("coeff_search" in cfg.strategies and
                 ("coeff_search" in force or "coeff_search" not in done))

    # Baselines are budget-independent test-only references; compute once per
    # (tasks, arch, method) at the 'full' budget. The set of cells is computed
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

    def inits_to_run(strat: str, requested: List[str]) -> List[str]:
        if strat not in cfg.strategies:
            return []
        have = done.get(strat, {})
        return [i for i in requested if (strat in force or i not in have)]

    return (
        run_coeff,
        inits_to_run("weight_gd", cfg.weight_gd_inits),
        inits_to_run("weight_gd_lora", cfg.weight_gd_lora_inits),
        inits_to_run("subspace_gd", cfg.subspace_gd_inits),
        baseline_cells,
        inits_to_run("directional_sampling", cfg.ds_inits),
        inits_to_run("bo_search", cfg.bo_inits),
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_experiment(cfg: RunConfig) -> Optional[str]:
    overlap = set(cfg.tasks) & set(cfg.unseen_tasks)
    if overlap:
        raise ValueError(f"unseen_tasks overlap tasks: {sorted(overlap)}")
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

    run_coeff, weight_inits, weight_lora_inits, subspace_inits, baseline_cells, ds_inits, bo_inits = _plan(
        cfg, output, mt_ckpt
    )

    have_coeff = run_coeff or ("coeff_search" in _existing(output))
    if (
        ("coeff_best" in weight_inits or "coeff_best" in weight_lora_inits or "coeff_best" in subspace_inits
         or "coeff_best" in bo_inits)
        and not have_coeff
    ):
        raise ValueError(
            "init 'coeff_best' requires coeff_search results; include coeff_search in --strategies "
            "(or run it first into this JSON)."
        )

    if not (run_coeff or weight_inits or weight_lora_inits or subspace_inits or baseline_cells or ds_inits
            or bo_inits):
        skipping_baselines = (
            "baselines" in cfg.strategies
            and budget_label(cfg.budget) != "full"
            and "baselines" not in _existing(output)
        )
        note = " (baselines skipped: run at budget=full only)" if skipping_baselines else ""
        print(f"Nothing to run for {out_path} (all requested cells present; use --force to rerun){note}")
        return None

    log_path = setup_logging(cfg.out_dir, name)
    logging.info(f"Log -> {log_path}")
    logging.info(f"Unit: {name}")
    if cfg.unseen_tasks:
        logging.info(f"Unseen tasks (test only): {cfg.unseen_tasks}")
    logging.info(
        f"Planned: coeff={run_coeff}  weight_gd={weight_inits}  weight_gd_lora={weight_lora_inits}  "
        f"subspace_gd={subspace_inits}  baselines={baseline_cells}  directional_sampling={ds_inits}  "
        f"bo_search={bo_inits}"
    )
    if mt_ckpt and mt_ckpt != cfg.multitask_checkpoint:
        logging.info(f"Auto-resolved multitask checkpoint -> {mt_ckpt}")

    tokenizer = models.load_tokenizer(cfg.base_model)
    valid_examples, test_examples, sft_pools, data_stats = build_budget_pools(cfg)
    unseen_test_examples = build_unseen_test_pools(cfg, stats=data_stats)

    logging.info("\n=== Loading checkpoints ===")
    base_sd = models.load_state_dict(cfg.base_model, device="cpu")
    task_sds = [models.load_state_dict(cp, device="cpu") for cp in cfg.task_checkpoints]
    keys = models.shared_float_keys(base_sd, *task_sds)
    method = get_method(cfg.method, **cfg.method_kwargs)

    ckpt_dir = os.path.join(
        cfg.checkpoint_root, cfg.method, cfg.arch, "-".join(cfg.tasks),
        *(("unseen_" + "-".join(cfg.unseen_tasks),) if cfg.unseen_tasks else ()),
        budget_label(cfg.budget),
    )
    ctx = StrategyContext(
        method=method, base_model=cfg.base_model, base_sd=base_sd, task_sds=task_sds, keys=keys,
        tasks=cfg.tasks, tokenizer=tokenizer,
        valid_examples=valid_examples, test_examples=test_examples, sft_pools=sft_pools,
        unseen_tasks=list(cfg.unseen_tasks), unseen_test_examples=unseen_test_examples,
        device=device, dtype=cfg.dtype,
        max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p,
        gen_batch_size=cfg.gen_batch_size,
        lambda_min=cfg.lambda_min, lambda_max=cfg.lambda_max, lambda_steps=cfg.lambda_steps,
        gd_epochs=cfg.gd_epochs, gd_lr=cfg.gd_lr, gd_warmup_ratio=cfg.gd_warmup_ratio,
        gd_batch_size=cfg.gd_batch_size, gd_grad_accum_steps=cfg.gd_grad_accum_steps,
        max_seq_length=cfg.max_seq_length, loss_batch_size=cfg.loss_batch_size,
        lora_lr=cfg.lora_lr, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        subspace_epochs=cfg.subspace_epochs, subspace_lr=cfg.subspace_lr, subspace_scheduler=cfg.subspace_scheduler,
        subspace_warmup_ratio=cfg.subspace_warmup_ratio, subspace_batch_size=cfg.subspace_batch_size,
        subspace_grad_accum_steps=cfg.subspace_grad_accum_steps,
        subspace_basis_device=cfg.subspace_basis_device,
        subspace_basis_headroom_gib=cfg.subspace_basis_headroom_gib,
        ds_alpha=cfg.ds_alpha, ds_beta=cfg.ds_beta, ds_samples=cfg.ds_samples, ds_seed=cfg.ds_seed,
        ds_select_metric=cfg.ds_select_metric, ds_eval_test=cfg.ds_eval_test,
        ds_basis_device=cfg.ds_basis_device,
        ds_reeval_center=cfg.ds_reeval_center,
        bo_trials=cfg.bo_trials, bo_startup_trials=cfg.bo_startup_trials, bo_radius=cfg.bo_radius,
        bo_lower=cfg.bo_lower, bo_seed=cfg.bo_seed, bo_basis_device=cfg.bo_basis_device,
        save_checkpoints=cfg.save_checkpoints, checkpoint_dir=ckpt_dir,
        multitask_checkpoint=mt_ckpt,
    )

    # metadata / merge scaffolding
    output.update({
        "tasks": cfg.tasks, "arch": cfg.arch, "method": cfg.method,
        "budget": budget_label(cfg.budget), "seed": cfg.seed, "base_model": cfg.base_model,
        "data_stats": data_stats,
    })
    if cfg.unseen_tasks:
        output["unseen_tasks"] = list(cfg.unseen_tasks)
        output["unseen_n_shot"] = {t: data.resolve_n_shot(t, cfg.unseen_n_shot) for t in cfg.unseen_tasks}
    strat_out = output.setdefault("strategies", {})

    # best_lambda (+ the directional_sampling density thresholds): from prior
    # results or from a fresh coeff search
    if "coeff_search" in strat_out:
        _seed_coeff_state(ctx, strat_out["coeff_search"])
    _seed_subspace_best(ctx, strat_out)
    _seed_bo_best(ctx, strat_out)

    def _persist(reason: str) -> None:
        """Save the current output to out_path.

        Called after every strategy (and each GD init) so a run that dies
        partway through (e.g. OOM in subspace_gd) only loses the currently
        running cell, not everything that came before it.
        """
        output["config"] = _config_summary(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=float)
        logging.info(f"  [saved] {reason} -> {out_path.name}")

    if run_coeff:
        strat_out["coeff_search"] = strategies.coeff_search.run(ctx)
        _seed_coeff_state(ctx, strat_out["coeff_search"])
        _persist("coeff_search")

    for init in weight_inits:
        dst = strat_out.setdefault("weight_gd", {})
        dst[init] = strategies.weight_gd.run_one(ctx, init)
        _persist(f"weight_gd/{init}")

    for init in weight_lora_inits:
        dst = strat_out.setdefault("weight_gd_lora", {})
        dst[init] = strategies.weight_gd_lora.run_one(ctx, init)
        _persist(f"weight_gd_lora/{init}")

    for init in subspace_inits:
        dst = strat_out.setdefault("subspace_gd", {})
        dst[init] = strategies.subspace_gd.run_one(ctx, init)
        _persist(f"subspace_gd/{init}")

    if baseline_cells:
        new_cells = strategies.baselines.run(ctx, ctx.multitask_checkpoint, cells=baseline_cells)
        strat_out.setdefault("baselines", {}).update(new_cells)
        _persist("baselines")

    if ds_inits and (ctx.best_lambda is None or _ds_threshold(ctx, cfg) is None):
        # lambda* is required only by the coeff_best init; the coeff_search
        # valid metric is just a secondary density reference (the primary one
        # is the init point's own score), so a missing coeff_search is fatal
        # only when coeff_best was asked for.
        try:
            _seed_coeff_state(ctx, _load_coeff_search(cfg))
            if "coeff_best" in ds_inits:
                logging.info(
                    f"directional_sampling: coeff_best init: lambda*={ctx.best_lambda}  "
                    f"recorded valid_avg_{cfg.ds_select_metric}={_ds_threshold(ctx, cfg)} "
                    f"from {cfg.coeff_source_dir}"
                )
            else:
                logging.debug(f"directional_sampling: lambda*={ctx.best_lambda} read for reference only")
        except ValueError:
            if "coeff_best" in ds_inits:
                raise
            logging.info(
                f"directional_sampling: no coeff_search in {cfg.coeff_source_dir}; "
                "density is vs the init point only"
            )
    if "subspace_best" in ds_inits and ctx.subspace_best_coeffs is None:
        # Same fallback shape as coeff_best's: this unit's own JSON first (it
        # is the ablation's side root and usually has no subspace_gd), then the
        # main results root.
        _seed_subspace_best(ctx, _load_source_strategies(cfg))
        if ctx.subspace_best_coeffs is None:
            raise ValueError(
                "directional_sampling init 'subspace_best' requires subspace_gd's "
                "coefficients_final; none found in this unit or in "
                f"{cfg.coeff_source_dir}. Run subspace_gd into that unit first."
            )
        logging.info(
            f"directional_sampling: subspace_best init: center="
            f"{[round(c, 4) for c in ctx.subspace_best_coeffs]}  "
            f"recorded valid_avg_score={ctx.subspace_best_valid_avg_score}  "
            f"test_avg_score={ctx.subspace_best_test_avg_score} "
            f"(from subspace_gd/{ctx.subspace_best_source} in {cfg.coeff_source_dir})"
        )
    if "bo_best" in ds_inits and ctx.bo_best_coeffs is None:
        _seed_bo_best(ctx, _load_source_strategies(cfg))
        if ctx.bo_best_coeffs is None:
            raise ValueError(
                "directional_sampling init 'bo_best' requires bo_search's "
                "coefficients_final; none found in this unit or in "
                f"{cfg.coeff_source_dir}. Run bo_search into that unit first."
            )
        logging.info(
            f"directional_sampling: bo_best init: center="
            f"{[round(c, 4) for c in ctx.bo_best_coeffs]}  "
            f"recorded valid_avg_score={ctx.bo_best_valid_avg_score}  "
            f"test_avg_score={ctx.bo_best_test_avg_score} "
            f"(from bo_search/{ctx.bo_best_source} in {cfg.coeff_source_dir})"
        )
    for init in ds_inits:
        dst = strat_out.setdefault("directional_sampling", {})
        dst[init] = strategies.directional_sampling.run_one(ctx, init)
        _persist(f"directional_sampling/{init}")

    for init in bo_inits:
        dst = strat_out.setdefault("bo_search", {})
        dst[init] = strategies.bo_search.run_one(ctx, init)
        _persist(f"bo_search/{init}")

    logging.info(f"\nDone -> {out_path}")
    return str(out_path)


def _seed_subspace_best(ctx: StrategyContext, strat: dict) -> None:
    """Copy subspace_gd's best trained coefficient vector onto the context
    (directional_sampling's 'subspace_best' center). Does not overwrite a
    value already set."""
    if ctx.subspace_best_coeffs is not None:
        return
    coeffs, src, valid, test = _best_vector_coeffs(strat, "subspace_gd")
    if coeffs is not None:
        ctx.subspace_best_coeffs = coeffs
        ctx.subspace_best_source = src
        ctx.subspace_best_valid_avg_score = valid
        ctx.subspace_best_test_avg_score = test


def _seed_bo_best(ctx: StrategyContext, strat: dict) -> None:
    """Copy bo_search's best-trial coefficient vector onto the context
    (directional_sampling's 'bo_best' center). Does not overwrite a value
    already set."""
    if ctx.bo_best_coeffs is not None:
        return
    coeffs, src, valid, test = _best_vector_coeffs(strat, "bo_search")
    if coeffs is not None:
        ctx.bo_best_coeffs = coeffs
        ctx.bo_best_source = src
        ctx.bo_best_valid_avg_score = valid
        ctx.bo_best_test_avg_score = test


def _seed_coeff_state(ctx: StrategyContext, cs: dict) -> None:
    """Copy coeff_search's lambda* and valid metrics onto the context.

    ``best_lambda`` is the ``coeff_best`` init for every strategy; the two
    valid metrics are directional_sampling's density thresholds (which one is
    used depends on ds_select_metric). Existing values are not overwritten, so
    a threshold already read from --coeff-source-dir wins over a later partial.
    """
    for attr, key in (
        ("best_lambda", "best_lambda"),
        ("coeff_best_valid_avg_score", "valid_avg_score"),
        ("coeff_best_valid_avg_loss", "valid_avg_loss"),
        ("coeff_best_test_avg_score", "test_avg_score"),
    ):
        if getattr(ctx, attr) is None and cs.get(key) is not None:
            setattr(ctx, attr, cs[key])


def _ds_threshold(ctx: StrategyContext, cfg: RunConfig) -> Optional[float]:
    return (ctx.coeff_best_valid_avg_loss if cfg.ds_select_metric == "loss"
            else ctx.coeff_best_valid_avg_score)


def _config_summary(cfg: RunConfig) -> dict:
    d = dict(cfg.__dict__)
    d["force"] = sorted(cfg.force)
    d["budget"] = budget_label(cfg.budget)
    if not d.get("unseen_tasks"):
        for k in ("unseen_tasks", "unseen_test_samples", "unseen_n_shot", "unseen_raw_prompt"):
            d.pop(k, None)
    return d
