"""Profile one action per strategy and scale by n_actions.

Writes ``results/cost_estimation/units/<unit>.json``. Accuracy is not measured
here — plots read ``test_avg_acc`` from ``results/main_exp``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from . import cost
from . import strategies
from .experiment import (
    UNITS_SUBDIR,
    RunConfig,
    _assert_compatible,
    _config_summary,
    budget_label,
    build_budget_loaders,
    setup_logging,
    unit_name,
)
from .merging import get_method
from .models import load_state_dict, shared_float_keys
from .strategies import StrategyContext


def _load_best_lambda(source_dir: str, name: str) -> Optional[float]:
    path = Path(source_dir) / UNITS_SUBDIR / f"{name}.json"
    if not path.exists():
        return None
    rec = json.load(open(path))
    return rec.get("strategies", {}).get("coeff_search", {}).get("best_lambda")


def run_cost_estimation(cfg: RunConfig, coeff_source_dir: str = "results/main_exp") -> Optional[str]:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = unit_name(cfg)
    out_path = Path(cfg.out_dir) / UNITS_SUBDIR / f"{name}.json"

    output = {}
    if out_path.exists():
        output = json.load(open(out_path))
        _assert_compatible(output, cfg)

    wanted = [s for s in cfg.strategies if s in ("coeff_search", "weight_gd", "subspace_gd")]
    done = output.get("strategies", {})
    force = cfg.force
    todo = [s for s in wanted if s in force or s not in done]
    if not todo:
        print(f"Nothing to profile for {out_path} (all requested cells present; use --force to rerun)")
        return None

    log_path = setup_logging(cfg.out_dir, name)
    logging.info(f"Log -> {log_path}")
    logging.info(f"Unit: {name}")
    logging.info(f"Profile: {todo}")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    valid_loaders, test_loaders, data_stats, _, _ = build_budget_loaders(cfg, processor)  # no unlabeled data here

    logging.info("\n=== Loading checkpoints ===")
    base_sd = load_state_dict(cfg.base_model, device="cpu")
    task_sds = [load_state_dict(cp, device="cpu") for cp in cfg.task_checkpoints]
    keys = shared_float_keys(base_sd, *task_sds)
    method = get_method(cfg.method, **cfg.method_kwargs)

    best_lambda = _load_best_lambda(coeff_source_dir, name)
    if best_lambda is None and "subspace_gd" in todo:
        raise ValueError(
            f"subspace_gd profile needs coeff_search.best_lambda from "
            f"{coeff_source_dir}/units/{name}.json"
        )
    if best_lambda is not None:
        logging.info(f"  best_lambda={best_lambda}  (from {coeff_source_dir})")

    ctx = StrategyContext(
        method=method, base_model=cfg.base_model, base_sd=base_sd, task_sds=task_sds, keys=keys,
        tasks=cfg.tasks, valid_loaders=valid_loaders, test_loaders=test_loaders, device=device,
        lambda_min=cfg.lambda_min, lambda_max=cfg.lambda_max, lambda_steps=cfg.lambda_steps,
        gd_epochs=cfg.gd_epochs, gd_lr=cfg.gd_lr, gd_warmup_ratio=cfg.gd_warmup_ratio,
        gd_optimizer=cfg.gd_optimizer, gd_momentum=cfg.gd_momentum, gd_patience=cfg.gd_patience,
        subspace_epochs=cfg.subspace_epochs, subspace_lr=cfg.subspace_lr,
        subspace_warmup_ratio=cfg.subspace_warmup_ratio, subspace_patience=cfg.subspace_patience,
        best_lambda=best_lambda,
    )

    output.update({
        "tasks": cfg.tasks, "arch": cfg.arch, "method": cfg.method,
        "budget": budget_label(cfg.budget), "seed": cfg.seed,
        "base_model": cfg.base_model, "batch_size": cfg.batch_size,
        "data_stats": data_stats,
    })
    strat_out = output.setdefault("strategies", {})

    def _persist(reason: str) -> None:
        output["config"] = _config_summary(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=float)
        logging.info(f"  [saved] {reason} -> {out_path.name}")

    fwd_flops = output.get("fwd_flops_per_image")
    if fwd_flops is None:
        from transformers import AutoModel
        logging.info("calibrating vision FLOPs (one get_image_features batch)")
        pixels = cost.first_batch_pixels(valid_loaders, cfg.tasks, device)
        if pixels is None:
            raise RuntimeError("empty valid loaders; cannot calibrate FLOPs")
        vision = AutoModel.from_pretrained(cfg.base_model).to(device).eval()
        fwd_flops = cost.vision_fwd_flops_per_image(vision, pixels)
        output["fwd_flops_per_image"] = fwd_flops
        logging.info(f"  fwd_flops_per_image={fwd_flops:.4e}")
        del vision
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "coeff_search" in todo:
        raw = strategies.coeff_search.profile_one(ctx)
        strat_out["coeff_search"] = cost.finalize(raw, "coeff_search", cfg.lambda_steps, fwd_flops)
        _persist("coeff_search")

    if "weight_gd" in todo:
        init = cost.weight_gd_init(cfg.method)
        raw = strategies.weight_gd.profile_one(ctx, init)
        strat_out["weight_gd"] = cost.finalize(raw, "weight_gd", cfg.lambda_steps, fwd_flops)
        _persist(f"weight_gd/{init}")

    if "subspace_gd" in todo:
        init = cost.SUBSPACE_GD_INIT
        raw = strategies.subspace_gd.profile_one(ctx, init)
        strat_out["subspace_gd"] = cost.finalize(raw, "subspace_gd", cfg.lambda_steps, fwd_flops)
        _persist(f"subspace_gd/{init}")

    logging.info(f"\nDone -> {out_path}")
    return str(out_path)


def build_cfg_from_cli(args) -> RunConfig:
    """Shared by ``scripts/profile_cost.py`` so the CLI stays thin."""
    from .checkpoints import resolve_task_checkpoints as resolve

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if args.task_checkpoints:
        task_ckpts = [x.strip() for x in args.task_checkpoints.split(",") if x.strip()]
    else:
        task_ckpts = resolve(args.task_vectors_dir, tasks, args.arch)
        for t, c in zip(tasks, task_ckpts):
            print(f"  resolved {t} -> {c}")
    method_kwargs = {}
    if args.method.lower() == "ties":
        method_kwargs["density"] = args.ties_density
    elif args.method.lower() == "dare":
        method_kwargs["drop_rate"] = args.dare_drop_rate
        method_kwargs["seed"] = args.dare_seed
    return RunConfig(
        base_model=args.base_model, arch=args.arch, method=args.method,
        tasks=tasks, task_checkpoints=task_ckpts, budget=args.budget, seed=args.seed,
        strategies=[x.strip() for x in args.strategies.split(",") if x.strip()],
        force=set(x.strip() for x in (args.force or "").split(",") if x.strip()),
        batch_size=args.batch_size,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max, lambda_steps=args.lambda_steps,
        gd_epochs=args.gd_epochs, gd_lr=args.gd_lr, gd_warmup_ratio=args.gd_warmup_ratio,
        subspace_epochs=args.subspace_epochs, subspace_lr=args.subspace_lr,
        subspace_warmup_ratio=args.subspace_warmup_ratio,
        method_kwargs=method_kwargs, out_dir=args.out_dir, device=args.device,
        task_vectors_dir=args.task_vectors_dir, wandb_project=None,
    )
