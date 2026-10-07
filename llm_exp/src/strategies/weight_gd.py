"""Weight-space GD strategy.

Full fine-tuning of the shared decoder on each task's SFT pool, starting from
an init point built from the merged direction:

  pretrained    -> base weights (delta = 0)
  avg           -> base + merged_delta(coeff=1/N)        (N = number of tasks)
  merged        -> base + merged_delta(coeff=1)
  coeff_best    -> base + merged_delta(coeff=lambda*)   (needs coeff search)
  random_<seed> -> i.i.d. N(0,1) over the shared float keys (true random)

Minimizes next-token cross-entropy jointly over ``ctx.sft_pools`` (one
dataloader per task; ifeval's pool is the argilla-sourced substitute, not its
own eval set -- see design.md). Checkpoints, when saved, are named by the
init point.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from transformers import get_cosine_schedule_with_warmup

from ..data import get_sft_dataloader
from ..models import avg_loss, avg_score, load_causal_lm
from ..tasks import get_task
from ._common import (
    attach_unseen, clear_cache, eval_all_tasks, eval_all_tasks_loss, free_model, lm_loss, pack_result,
    save_finetuned,
)
from .context import StrategyContext


def _parse_random_seed(init: str) -> int:
    rest = init[len("random_"):]
    if not rest.isdigit():
        raise ValueError(f"random init must be 'random_<int seed>', got '{init}'")
    return int(rest)


def _init_weights(ctx: StrategyContext, init: str) -> Dict[str, torch.Tensor]:
    base_sd, keys, method = ctx.base_sd, ctx.keys, ctx.method
    if init == "pretrained":
        return dict(base_sd)
    if init == "avg":
        delta = method.merged_delta(base_sd, ctx.task_sds, keys, coeff=1.0 / len(ctx.tasks))
    elif init == "merged":
        delta = method.merged_delta(base_sd, ctx.task_sds, keys, coeff=1.0)
    elif init == "coeff_best":
        if ctx.best_lambda is None:
            raise ValueError("weight_gd init 'coeff_best' requires coeff search (best_lambda)")
        delta = method.merged_delta(base_sd, ctx.task_sds, keys, coeff=ctx.best_lambda)
    elif init.startswith("random_"):
        g = torch.Generator(device="cpu").manual_seed(_parse_random_seed(init))
        W = dict(base_sd)
        for k in keys:
            W[k] = torch.randn(base_sd[k].shape, generator=g, dtype=torch.float32)
        return W
    else:
        raise ValueError(
            f"Unknown init '{init}' (expected pretrained, avg, merged, coeff_best, random_<seed>)"
        )
    W = dict(base_sd)
    for k in keys:
        W[k] = base_sd[k].float() + delta[k]
    return W


def _build_dataloaders(ctx: StrategyContext):
    dataloaders = {}
    for t in ctx.tasks:
        pool = ctx.sft_pools.get(t)
        if pool:
            dataloaders[t] = get_sft_dataloader(
                pool, ctx.tokenizer, batch_size=ctx.gd_batch_size, max_length=ctx.max_seq_length,
                enable_thinking=getattr(get_task(t), "ENABLE_THINKING", True),
            )
    if not dataloaders:
        raise ValueError("No task in this unit has any SFT examples -- weight_gd cannot train.")
    return dataloaders


def _setup_train(model, ctx: StrategyContext, dataloaders):
    """Gradient checkpointing + optimizer + schedule. Returns (optimizer, scheduler, steps_per_epoch)."""
    # Full-parameter fine-tuning at gd_batch_size x max_seq_length retains every
    # layer's forward activations for backward; recompute them instead -- this
    # is what keeps ifeval/usefulness_judge's longer SFT examples from OOMing.
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ctx.gd_lr)
    accum = ctx.gd_grad_accum_steps
    micro_steps_per_epoch = max(len(dl) for dl in dataloaders.values())
    steps_per_epoch = -(-micro_steps_per_epoch // accum)  # ceil(micro_steps / accum)
    total_steps = steps_per_epoch * ctx.gd_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * ctx.gd_warmup_ratio), total_steps)
    return optimizer, scheduler, steps_per_epoch


def _one_epoch(model, ctx: StrategyContext, dataloaders, optimizer, scheduler, steps_per_epoch):
    """One weight-GD epoch. Returns (avg_loss, n_fwd_tokens, n_bwd_tokens)."""
    accum = ctx.gd_grad_accum_steps
    model.train()
    iters = {t: iter(dl) for t, dl in dataloaders.items()}
    epoch_loss = 0.0
    n_tokens = 0
    for _ in range(steps_per_epoch):
        optimizer.zero_grad()
        for task, dl in dataloaders.items():
            # Gradient accumulation: fetch+forward+backward accum many
            # gd_batch_size batches (one at a time, freed before the next)
            # before optimizer.step() -- effective batch size stays
            # gd_batch_size * accum, but peak activation memory only ever
            # sees one gd_batch_size-sized batch, regardless of how many
            # long sequences happen to land in the same logical batch.
            for _ in range(accum):
                try:
                    batch = next(iters[task])
                except StopIteration:
                    iters[task] = iter(dl)
                    batch = next(iters[task])
                batch = {k: v.to(ctx.device) for k, v in batch.items()}
                n_tokens += int(batch["input_ids"].numel())
                loss = lm_loss(model, batch)
                (loss / accum / len(dataloaders)).backward()
                epoch_loss += loss.item() / accum / len(dataloaders)
        optimizer.step()
        scheduler.step()
    # every position is forwarded once and backpropagated once
    return epoch_loss / steps_per_epoch, n_tokens, n_tokens


def _train(model, ctx: StrategyContext, dataloaders, label: str):
    optimizer, scheduler, steps_per_epoch = _setup_train(model, ctx, dataloaders)

    logging.info(f"\n=== Weight GD ({label}) ===")
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(ctx.gd_epochs):
        avg_loss, _, _ = _one_epoch(model, ctx, dataloaders, optimizer, scheduler, steps_per_epoch)
        logging.info(f"  epoch {epoch + 1}/{ctx.gd_epochs}  avg_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            # .to("cpu") (not a GPU .clone()): keeps this snapshot off the GPU
            # so it doesn't sit as a permanent extra full-model-sized copy for
            # the rest of training, nor double up momentarily with the next
            # epoch's snapshot when a later epoch improves on this one.
            best_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True  # re-enable KV cache for the eval-time generate() calls
    logging.info(f"  best epoch={best_epoch}/{ctx.gd_epochs}  best_train_loss={best_loss:.4f}")
    return best_loss, best_epoch


def run_one(ctx: StrategyContext, init: str) -> dict:
    tasks, device = ctx.tasks, ctx.device
    # Drop the caching allocator's freed-but-not-returned memory before we
    # build a fresh model. This is the guard for the CUDA hang that showed up
    # after a prior strategy left the allocator fragmented (see vision_exp).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"weight_gd(init={init}): building fresh model")
    model = load_causal_lm(ctx.base_model, device=device, dtype=ctx.dtype)
    W = _init_weights(ctx, init)
    model.load_state_dict(W, strict=True)

    dataloaders = _build_dataloaders(ctx)

    # Score the init point itself, before any GD step: this is the x-axis of
    # the init-vs-weight_gd scatter (scripts/plots/plot_init_scatter.py). For the
    # merge-derived inits it duplicates a baselines cell, but random_<seed>
    # inits have no baseline counterpart, so record it here for every init.
    logging.info(f"weight_gd(init={init}): evaluating init on valid/test")
    init_valid = eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, tasks, device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(device)
    init_test = eval_all_tasks(
        model, ctx.tokenizer, ctx.test_examples, tasks, device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(device)
    logging.info(
        f"  init valid_avg_score={avg_score(init_valid):.4f}  "
        f"init test_avg_score={avg_score(init_test):.4f}"
    )

    best_loss, best_epoch = _train(model, ctx, dataloaders, f"init={init}")

    logging.info(f"weight_gd(init={init}): evaluating on valid/test")
    valid_res = eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, tasks, device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
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
        f"weight_gd(init={init}): valid_avg_score={avg_score(valid_res):.4f}  "
        f"valid_avg_loss={avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}"
    )

    extra = {
        "init": init, "epochs": ctx.gd_epochs, "learning_rate": ctx.gd_lr,
        "warmup_ratio": ctx.gd_warmup_ratio, "best_train_loss": best_loss, "best_epoch": best_epoch,
        "init_valid_avg_score": avg_score(init_valid), "init_valid_per_task": init_valid,
        "init_test_avg_score": avg_score(init_test), "init_test_per_task": init_test,
    }
    if ctx.best_lambda is not None and init == "coeff_best":
        extra["best_lambda"] = ctx.best_lambda
    if ctx.save_checkpoints:
        tag = f"weight_gd_{init}"
        logging.info(f"weight_gd(init={init}): saving checkpoint -> {tag}")
        extra["checkpoint"] = save_finetuned(ctx, model, ctx.tokenizer, tag)
        logging.info(f"weight_gd(init={init}): checkpoint saved")
    result = pack_result(valid_res, test_res, valid_loss=valid_loss, **extra)
    attach_unseen(ctx, model, result)
    free_model(model, device)
    return result


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}


def profile_one(ctx: StrategyContext, init: str) -> dict:
    """Warm up one epoch, then time a second epoch (peak mem + wall + token counts).

    The timed epoch is the training loop only -- ``_train``'s best-state CPU
    snapshot and the post-training generate() eval are not part of it.
    """
    from ..cost import measure

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"weight_gd(init={init}): profile (warmup + 1 timed epoch)")
    model = load_causal_lm(ctx.base_model, device=ctx.device, dtype=ctx.dtype)
    model.load_state_dict(_init_weights(ctx, init), strict=True)
    dataloaders = _build_dataloaders(ctx)
    optimizer, scheduler, steps_per_epoch = _setup_train(model, ctx, dataloaders)
    _one_epoch(model, ctx, dataloaders, optimizer, scheduler, steps_per_epoch)
    with measure() as stats:
        avg_train_loss, n_fwd, n_bwd = _one_epoch(model, ctx, dataloaders, optimizer, scheduler, steps_per_epoch)
        stats["n_fwd_tokens"] = n_fwd
        stats["n_bwd_tokens"] = n_bwd
    logging.info(
        f"  timed epoch  avg_loss={avg_train_loss:.4f}  tokens={n_fwd}  "
        f"wall={stats['action_wall_s']:.1f}s  peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    del optimizer
    free_model(model, ctx.device)
    return {"init": init, "action": "train_epoch", "steps_per_epoch": steps_per_epoch, **stats}
