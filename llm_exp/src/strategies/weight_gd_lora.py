"""Weight-space GD strategy, LoRA variant.

Same init points / training loop shape as weight_gd.py, but the model is
wrapped in a PEFT LoRA adapter before training instead of being fully
fine-tuned. Only the adapter parameters (and their AdamW momentum/variance)
are trainable, so the optimizer-state memory that dominates weight_gd's
footprint on larger archs (e.g. qwen3-1.7b) collapses to a small fraction of
the full-parameter cost. The base weights themselves (the init point built
from the merged direction) are still loaded at full precision -- LoRA doesn't
touch that part of the memory picture, only the trainable/optimizer side.

The adapter is merged back into the base weights before eval/checkpointing
(``merge_and_unload``), so everything downstream (eval_all_tasks,
save_finetuned, subsequent task-vector extraction) sees a plain causal LM,
same as weight_gd's output -- no special-casing needed outside this file.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from ..data import get_sft_dataloader
from ..models import avg_loss, avg_score, load_causal_lm
from ..tasks import get_task
from ._common import (
    attach_unseen, clear_cache, eval_all_tasks, eval_all_tasks_loss, free_model, lm_loss, pack_result,
    save_finetuned,
)
from .context import StrategyContext
from .weight_gd import _init_weights

# Qwen3 (and most Llama-family) linear projection names -- attention + MLP.
# Covering both means LoRA can adjust the same weights full fine-tuning would.
_DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _wrap_lora(model, ctx: StrategyContext):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=ctx.lora_r,
        lora_alpha=ctx.lora_alpha,
        lora_dropout=ctx.lora_dropout,
        target_modules=ctx.lora_target_modules or _DEFAULT_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    # Required for gradient checkpointing when only adapter params are
    # trainable: the base model's embeddings/inputs have requires_grad=False,
    # so without this the checkpointed recompute graph has nothing to attach
    # gradients to and backward silently produces no grad for the adapter.
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    logging.info(
        f"  LoRA: r={ctx.lora_r} alpha={ctx.lora_alpha} target_modules={lora_cfg.target_modules} "
        f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.3f}%)"
    )
    return model


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
        raise ValueError("No task in this unit has any SFT examples -- weight_gd_lora cannot train.")
    return dataloaders


def _setup_train(model, ctx: StrategyContext, dataloaders):
    """Gradient checkpointing + AdamW over the adapter params + schedule.
    Returns (optimizer, scheduler, steps_per_epoch)."""
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ctx.lora_lr)
    accum = ctx.gd_grad_accum_steps
    micro_steps_per_epoch = max(len(dl) for dl in dataloaders.values())
    steps_per_epoch = -(-micro_steps_per_epoch // accum)  # ceil(micro_steps / accum)
    total_steps = steps_per_epoch * ctx.gd_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * ctx.gd_warmup_ratio), total_steps)
    return optimizer, scheduler, steps_per_epoch


def _one_epoch(model, ctx: StrategyContext, dataloaders, optimizer, scheduler, steps_per_epoch):
    """One LoRA epoch. Returns (avg_loss, n_fwd_tokens, n_bwd_tokens)."""
    accum = ctx.gd_grad_accum_steps
    model.train()
    iters = {t: iter(dl) for t, dl in dataloaders.items()}
    epoch_loss = 0.0
    n_tokens = 0
    for _ in range(steps_per_epoch):
        optimizer.zero_grad()
        for task, dl in dataloaders.items():
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

    logging.info(f"\n=== Weight GD LoRA ({label}) ===")
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(ctx.gd_epochs):
        avg_loss, _, _ = _one_epoch(model, ctx, dataloaders, optimizer, scheduler, steps_per_epoch)
        logging.info(f"  epoch {epoch + 1}/{ctx.gd_epochs}  avg_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Adapter-only state_dict (get_peft_model's state_dict() already
            # returns just the LoRA params) -- cheap to snapshot on CPU each
            # improving epoch, unlike a full-model snapshot.
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"weight_gd_lora(init={init}): building fresh model")
    model = load_causal_lm(ctx.base_model, device=device, dtype=ctx.dtype)
    W = _init_weights(ctx, init)
    model.load_state_dict(W, strict=True)
    model = _wrap_lora(model, ctx)

    dataloaders = _build_dataloaders(ctx)
    best_loss, best_epoch = _train(model, ctx, dataloaders, f"init={init}")

    logging.info(f"weight_gd_lora(init={init}): merging LoRA adapter into base weights")
    model = model.merge_and_unload()

    logging.info(f"weight_gd_lora(init={init}): evaluating on valid/test")
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
        f"weight_gd_lora(init={init}): valid_avg_score={avg_score(valid_res):.4f}  "
        f"valid_avg_loss={avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}"
    )

    extra = {
        "init": init, "epochs": ctx.gd_epochs, "learning_rate": ctx.lora_lr,
        "warmup_ratio": ctx.gd_warmup_ratio, "best_train_loss": best_loss, "best_epoch": best_epoch,
        "lora_r": ctx.lora_r, "lora_alpha": ctx.lora_alpha, "lora_dropout": ctx.lora_dropout,
    }
    if ctx.best_lambda is not None and init == "coeff_best":
        extra["best_lambda"] = ctx.best_lambda
    if ctx.save_checkpoints:
        tag = f"weight_gd_lora_{init}"
        logging.info(f"weight_gd_lora(init={init}): saving checkpoint -> {tag}")
        extra["checkpoint"] = save_finetuned(ctx, model, ctx.tokenizer, tag)
        logging.info(f"weight_gd_lora(init={init}): checkpoint saved")
    result = pack_result(valid_res, test_res, valid_loss=valid_loss, **extra)
    attach_unseen(ctx, model, result)
    free_model(model, device)
    return result


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}


def profile_one(ctx: StrategyContext, init: str) -> dict:
    """Warm up one epoch, then time a second epoch (peak mem + wall + token counts).

    The timed epoch is the LoRA training loop only -- the adapter snapshot,
    merge_and_unload and the post-training generate() eval are not part of it.
    """
    from ..cost import measure

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"weight_gd_lora(init={init}): profile (warmup + 1 timed epoch)")
    model = load_causal_lm(ctx.base_model, device=ctx.device, dtype=ctx.dtype)
    model.load_state_dict(_init_weights(ctx, init), strict=True)
    model = _wrap_lora(model, ctx)
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
    return {
        "init": init, "action": "train_epoch", "steps_per_epoch": steps_per_epoch,
        "lora_r": ctx.lora_r, "lora_alpha": ctx.lora_alpha, **stats,
    }
