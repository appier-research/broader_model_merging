"""Profile one action per strategy and scale by n_actions.

Writes ``results/cost_estimation/units/<unit>.json``. Scores are not measured
here -- plots read ``test_avg_score`` from the main results root (``results/``).
Same shape as vision_exp/src/cost_estimation.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from . import cost
from . import models
from . import strategies
from .data import get_sft_dataloader
from .experiment import (
    UNITS_SUBDIR,
    RunConfig,
    _assert_compatible,
    _config_summary,
    budget_label,
    build_budget_pools,
    setup_logging,
    unit_name,
)
from .merging import get_method
from .strategies import StrategyContext
from .tasks import get_task


def _load_best_lambda(source_dir: str, name: str) -> Optional[float]:
    path = Path(source_dir) / UNITS_SUBDIR / f"{name}.json"
    if not path.exists():
        return None
    rec = json.load(open(path))
    return rec.get("strategies", {}).get("coeff_search", {}).get("best_lambda")


def _calibration_batch(ctx: StrategyContext):
    """First SFT batch of the first task that has one, at loss_batch_size."""
    for task in ctx.tasks:
        pool = ctx.sft_pools.get(task)
        if not pool:
            continue
        dl = get_sft_dataloader(
            pool, ctx.tokenizer, batch_size=ctx.loss_batch_size, max_length=ctx.max_seq_length,
            shuffle=False, enable_thinking=getattr(get_task(task), "ENABLE_THINKING", True),
        )
        batch = next(iter(dl))
        return {k: v.to(ctx.device) for k, v in batch.items()}
    return None


def _calibrate_fwd_flops(ctx: StrategyContext) -> float:
    """FlopCounterMode on one teacher-forced forward of a fresh base model.

    Loaded with sdpa rather than flash_attention_2 so the attention matmuls
    are aten ops the counter can see (flash-attn's fused CUDA kernel is not).
    """
    from transformers import AutoModelForCausalLM

    batch = _calibration_batch(ctx)
    if batch is None:
        raise RuntimeError("no SFT examples in any task; cannot calibrate FLOPs")
    model = AutoModelForCausalLM.from_pretrained(
        ctx.base_model, torch_dtype=models.resolve_dtype(ctx.dtype), attn_implementation="sdpa",
    ).to(ctx.device)
    flops = cost.fwd_flops_per_token(model, batch)
    del model, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return flops


def run_cost_estimation(
    cfg: RunConfig,
    coeff_source_dir: str = "results",
    weight_gd_init: Optional[str] = None,
    weight_gd_lora_init: Optional[str] = None,
    subspace_gd_init: Optional[str] = None,
    bo_init: Optional[str] = None,
) -> Optional[str]:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = unit_name(cfg)
    out_path = Path(cfg.out_dir) / UNITS_SUBDIR / f"{name}.json"

    output = {}
    if out_path.exists():
        output = json.load(open(out_path))
        _assert_compatible(output, cfg)

    wanted = [s for s in cfg.strategies if s in cost.PROFILED_STRATEGIES]
    done = output.get("strategies", {})
    force = cfg.force
    todo = [s for s in wanted if s in force or s not in done]
    if not todo:
        print(f"Nothing to profile for {out_path} (all requested cells present; use --force to rerun)")
        return None

    w_init = weight_gd_init or cost.weight_gd_init(cfg.method)
    l_init = weight_gd_lora_init or cost.weight_gd_lora_init(cfg.method)
    s_init = subspace_gd_init or cost.SUBSPACE_GD_INIT
    b_init = bo_init or cost.BO_SEARCH_INIT

    log_path = setup_logging(cfg.out_dir, name)
    logging.info(f"Log -> {log_path}")
    logging.info(f"Unit: {name}")
    logging.info(
        f"Profile: {todo}  (inits: weight_gd={w_init}, weight_gd_lora={l_init}, "
        f"subspace_gd={s_init}, bo_search={b_init})"
    )

    tokenizer = models.load_tokenizer(cfg.base_model)
    valid_examples, test_examples, sft_pools, data_stats = build_budget_pools(cfg)

    logging.info("\n=== Loading checkpoints ===")
    base_sd = models.load_state_dict(cfg.base_model, device="cpu")
    task_sds = [models.load_state_dict(cp, device="cpu") for cp in cfg.task_checkpoints]
    keys = models.shared_float_keys(base_sd, *task_sds)
    method = get_method(cfg.method, **cfg.method_kwargs)

    best_lambda = _load_best_lambda(coeff_source_dir, name)
    needs_lambda = any(
        strat in todo and init == "coeff_best"
        for strat, init in (
            ("subspace_gd", s_init), ("weight_gd", w_init), ("weight_gd_lora", l_init), ("bo_search", b_init),
        )
    )
    if best_lambda is None and needs_lambda:
        raise ValueError(
            f"init 'coeff_best' needs coeff_search.best_lambda from {coeff_source_dir}/units/{name}.json"
        )
    if best_lambda is not None:
        logging.info(f"  best_lambda={best_lambda}  (from {coeff_source_dir})")

    ctx = StrategyContext(
        method=method, base_model=cfg.base_model, base_sd=base_sd, task_sds=task_sds, keys=keys,
        tasks=cfg.tasks, tokenizer=tokenizer,
        valid_examples=valid_examples, test_examples=test_examples, sft_pools=sft_pools,
        device=device, dtype=cfg.dtype,
        max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p,
        gen_batch_size=cfg.gen_batch_size,
        lambda_min=cfg.lambda_min, lambda_max=cfg.lambda_max, lambda_steps=cfg.lambda_steps,
        gd_epochs=cfg.gd_epochs, gd_lr=cfg.gd_lr, gd_warmup_ratio=cfg.gd_warmup_ratio,
        gd_batch_size=cfg.gd_batch_size, gd_grad_accum_steps=cfg.gd_grad_accum_steps,
        max_seq_length=cfg.max_seq_length, loss_batch_size=cfg.loss_batch_size,
        lora_lr=cfg.lora_lr, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        subspace_epochs=cfg.subspace_epochs, subspace_lr=cfg.subspace_lr, subspace_scheduler=cfg.subspace_scheduler,
        subspace_warmup_ratio=cfg.subspace_warmup_ratio, subspace_batch_size=cfg.subspace_batch_size,
        subspace_grad_accum_steps=cfg.subspace_grad_accum_steps,
        subspace_basis_device=cfg.subspace_basis_device,
        subspace_basis_headroom_gib=cfg.subspace_basis_headroom_gib,
        bo_trials=cfg.bo_trials, bo_startup_trials=cfg.bo_startup_trials, bo_radius=cfg.bo_radius,
        bo_lower=cfg.bo_lower, bo_seed=cfg.bo_seed, bo_basis_device=cfg.bo_basis_device,
        best_lambda=best_lambda,
    )

    output.update({
        "tasks": cfg.tasks, "arch": cfg.arch, "method": cfg.method,
        "budget": budget_label(cfg.budget), "seed": cfg.seed,
        "base_model": cfg.base_model, "dtype": cfg.dtype,
        "gen_batch_size": cfg.gen_batch_size, "loss_batch_size": cfg.loss_batch_size,
        "gd_batch_size": cfg.gd_batch_size, "gd_grad_accum_steps": cfg.gd_grad_accum_steps,
        "subspace_batch_size": cfg.subspace_batch_size, "subspace_grad_accum_steps": cfg.subspace_grad_accum_steps,
        "subspace_basis_device": cfg.subspace_basis_device,
        "data_stats": data_stats,
    })
    strat_out = output.setdefault("strategies", {})

    def _persist(reason: str) -> None:
        output["config"] = _config_summary(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=float)
        logging.info(f"  [saved] {reason} -> {out_path.name}")

    fwd_flops = output.get("fwd_flops_per_token")
    if fwd_flops is None:
        logging.info("calibrating forward FLOPs/token (one teacher-forced forward, sdpa)")
        fwd_flops = _calibrate_fwd_flops(ctx)
        output["fwd_flops_per_token"] = fwd_flops
        logging.info(f"  fwd_flops_per_token={fwd_flops:.4e}")

    def _n(strategy: str) -> int:
        return cost.n_actions(
            strategy, lambda_steps=cfg.lambda_steps, bo_trials=cfg.bo_trials,
            gd_epochs=cfg.gd_epochs, subspace_epochs=cfg.subspace_epochs,
        )

    if "coeff_search" in todo:
        raw = strategies.coeff_search.profile_one(ctx)
        strat_out["coeff_search"] = cost.finalize(raw, "coeff_search", _n("coeff_search"), fwd_flops, False)
        _persist("coeff_search")

    if "bo_search" in todo:
        raw = strategies.bo_search.profile_one(ctx, b_init)
        strat_out["bo_search"] = cost.finalize(raw, "bo_search", _n("bo_search"), fwd_flops, False)
        _persist(f"bo_search/{b_init}")

    if "weight_gd_lora" in todo:
        raw = strategies.weight_gd_lora.profile_one(ctx, l_init)
        strat_out["weight_gd_lora"] = cost.finalize(raw, "weight_gd_lora", _n("weight_gd_lora"), fwd_flops, True)
        _persist(f"weight_gd_lora/{l_init}")

    if "weight_gd" in todo:
        raw = strategies.weight_gd.profile_one(ctx, w_init)
        strat_out["weight_gd"] = cost.finalize(raw, "weight_gd", _n("weight_gd"), fwd_flops, True)
        _persist(f"weight_gd/{w_init}")

    if "subspace_gd" in todo:
        raw = strategies.subspace_gd.profile_one(ctx, s_init)
        strat_out["subspace_gd"] = cost.finalize(raw, "subspace_gd", _n("subspace_gd"), fwd_flops, True)
        _persist(f"subspace_gd/{s_init}")

    logging.info(f"\nDone -> {out_path}")
    return str(out_path)


def build_cfg_from_cli(args) -> RunConfig:
    """Shared by ``scripts/profile_cost.py`` so the CLI stays thin."""
    from .checkpoints import resolve_task_checkpoints as resolve

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if args.task_checkpoints:
        task_ckpts = [x.strip() for x in args.task_checkpoints.split(",") if x.strip()]
        if len(tasks) != len(task_ckpts):
            raise ValueError(f"--tasks ({len(tasks)}) and --task-checkpoints ({len(task_ckpts)}) must align")
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
    elif args.method.lower() == "tsvm":
        method_kwargs["svd_device"] = args.svd_device
        method_kwargs["embedding_svd_device"] = args.embedding_svd_device
    return RunConfig(
        base_model=args.base_model, arch=args.arch, method=args.method,
        tasks=tasks, task_checkpoints=task_ckpts, budget=args.budget, seed=args.seed,
        strategies=[x.strip() for x in args.strategies.split(",") if x.strip()],
        force=set(x.strip() for x in (args.force or "").split(",") if x.strip()),
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max, lambda_steps=args.lambda_steps,
        gd_epochs=args.gd_epochs, gd_lr=args.gd_lr, gd_warmup_ratio=args.gd_warmup_ratio,
        gd_batch_size=args.gd_batch_size, gd_grad_accum_steps=args.gd_grad_accum_steps,
        max_seq_length=args.max_seq_length, loss_batch_size=args.loss_batch_size,
        lora_lr=args.lora_lr, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_target_modules=(
            [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
            if args.lora_target_modules else None
        ),
        subspace_epochs=args.subspace_epochs, subspace_lr=args.subspace_lr,
        subspace_scheduler=args.subspace_scheduler,
        subspace_warmup_ratio=args.subspace_warmup_ratio, subspace_batch_size=args.subspace_batch_size,
        subspace_grad_accum_steps=args.subspace_grad_accum_steps,
        subspace_basis_device=args.subspace_basis_device,
        subspace_basis_headroom_gib=args.subspace_basis_headroom_gib,
        bo_trials=args.bo_trials if args.bo_trials is not None else cost.bo_trials(args.method.lower()),
        bo_startup_trials=args.bo_startup_trials, bo_radius=args.bo_radius, bo_lower=args.bo_lower,
        bo_seed=args.bo_seed, bo_basis_device=args.bo_basis_device,
        method_kwargs=method_kwargs, out_dir=args.out_dir, device=args.device,
        task_vectors_dir=args.task_vectors_dir,
    )
