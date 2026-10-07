"""Evaluation and packing helpers shared across strategies."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch

from ..data import get_sft_dataloader
from ..models import avg_loss, avg_score, generate, save_causal_lm
from ..tasks import get_task

# Fallback for any future task module that doesn't define its own
# MAX_NEW_TOKENS (see src/tasks/*.py).
_DEFAULT_MAX_NEW_TOKENS = 256


def lm_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Next-token cross-entropy for one SFT batch without materializing the
    [batch, seq, vocab] logits.

    ``model(**batch).loss`` (HF's default) runs lm_head over every position,
    upcasts the logits to fp32 and keeps them plus their log-softmax and
    gradient alive for backward: ~3.7 GiB per 2048-token sequence on Qwen's
    152k vocab, more than three 0.6B model copies, and the reason the GD
    strategies' peak memory said nothing about the methods. Here the decoder
    runs as usual (gradient checkpointing included) and Liger's fused
    linear-cross-entropy kernel folds lm_head + log-softmax + gradient into
    one chunked pass -- same loss, same gradients (to lm_head.weight, which is
    tied to the embeddings on Qwen3, and to the hidden states), no full-vocab
    tensor. Mean over the non-ignored (-100) targets, like HF.

    Works on a plain causal LM and on a PEFT-wrapped one (``get_base_model``);
    lm_head is never a LoRA target here so its weight is the base tensor.
    """
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = inner.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
    ).last_hidden_state
    h = hidden[:, :-1].reshape(-1, hidden.size(-1))
    y = batch["labels"][:, 1:].reshape(-1)
    return _FUSED_CE(inner.lm_head.weight, h, y)


class _LazyFusedCE:
    """Build the (stateless) Liger loss module on first use."""

    _fn = None

    def __call__(self, weight, h, y):
        if self._fn is None:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            self._fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")
        return self._fn(weight, h, y)


_FUSED_CE = _LazyFusedCE()


def clear_cache(device: str) -> None:
    """Drop the caching allocator's freed-but-not-returned blocks. Call this
    between generate()'s KV cache and eval_task_loss's logits tensor (batch *
    max_length * vocab_size, multiple GB at loss_batch_size=8/max_length=2048)
    -- without it, repeated back-to-back allocations of very different shapes
    (e.g. coeff_search's 10-point lambda sweep, each doing generate() then a
    loss forward) fragment the allocator until a later, individually-smaller
    allocation fails to find a contiguous block even though nominal free
    memory would fit it."""
    if device == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def eval_task_model(
    model, tokenizer, task_name: str, examples: List[dict], device: str,
    max_new_tokens: int, temperature: float, top_p: float, batch_size: int,
) -> float:
    if not examples:
        return 0.0
    task = get_task(task_name)
    inputs = [task.format_example(row) for row in examples]
    responses = generate(
        model, tokenizer, inputs, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, device=device, batch_size=batch_size,
        enable_thinking=getattr(task, "ENABLE_THINKING", True),
    )
    scores = [float(task.score(resp, row)) for resp, row in zip(responses, examples)]
    return sum(scores) / len(scores)


def eval_all_tasks(
    model, tokenizer, examples_by_task: Dict[str, list], tasks: List[str], device: str,
    max_new_tokens: Optional[int], temperature: float, top_p: float, batch_size: int,
) -> Dict[str, float]:
    """``max_new_tokens=None`` (the default) uses each task's own
    ``MAX_NEW_TOKENS`` (classification-style tasks need far fewer decode steps
    than ifeval's free-form responses); pass an int to override every task
    uniformly, e.g. for a fast smoke test."""
    return {
        t: eval_task_model(
            model, tokenizer, t, examples_by_task[t], device,
            max_new_tokens if max_new_tokens is not None
            else getattr(get_task(t), "MAX_NEW_TOKENS", _DEFAULT_MAX_NEW_TOKENS),
            temperature, top_p, batch_size,
        )
        for t in tasks
    }


@torch.no_grad()
def eval_task_loss(
    model, tokenizer, task_name: str, sft_examples: List[dict], device: str, batch_size: int, max_length: int,
) -> Optional[float]:
    """Mean per-token teacher-forced NLL over ``sft_examples`` (the same
    (input, target) pairs weight_gd/subspace_gd train on -- see
    data.build_sft_examples), or None if the task has no gold completion for
    any row in this pool (e.g. ifeval's test pool -- not used here, since
    only the valid pool's loss is tracked, but the guard is generic)."""
    if not sft_examples:
        return None
    task = get_task(task_name)
    dl = get_sft_dataloader(
        sft_examples, tokenizer, batch_size=batch_size, max_length=max_length,
        shuffle=False, enable_thinking=getattr(task, "ENABLE_THINKING", True),
    )
    was_training = model.training
    model.eval()
    total_loss_x_tokens, total_tokens = 0.0, 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        n_tokens = int((batch["labels"] != -100).sum().item())
        if n_tokens == 0:
            continue
        loss = model(**batch).loss
        total_loss_x_tokens += loss.item() * n_tokens
        total_tokens += n_tokens
    if was_training:
        model.train()
    return total_loss_x_tokens / total_tokens if total_tokens else None


def eval_all_tasks_loss(
    model, tokenizer, sft_pools_by_task: Dict[str, list], tasks: List[str], device: str,
    batch_size: int, max_length: int,
) -> Dict[str, Optional[float]]:
    return {
        t: eval_task_loss(model, tokenizer, t, sft_pools_by_task[t], device, batch_size, max_length)
        for t in tasks
    }


def eval_unseen(ctx, model) -> Dict[str, float]:
    """generate()+score the held-out tasks on the model's current weights,
    each at its own MAX_NEW_TOKENS (or ctx.max_new_tokens when overridden)."""
    clear_cache(ctx.device)
    res = eval_all_tasks(
        model, ctx.tokenizer, ctx.unseen_test_examples, ctx.unseen_tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)
    return res


def unseen_fields(unseen_res: Dict[str, float]) -> dict:
    return {"unseen_test_avg_score": avg_score(unseen_res), "unseen_test_per_task": unseen_res}


def attach_unseen(ctx, model, out: dict) -> dict:
    """Score the held-out tasks on the weights currently in ``model`` and add
    ``unseen_test_avg_score`` / ``unseen_test_per_task`` to a packed result.
    No-op when ctx.unseen_tasks is empty. Never touches valid_avg_score /
    test_avg_score (those stay seen-only), so selection is unaffected -- the
    LLM counterpart of vision_exp's strategies/_common.attach_unseen."""
    if not ctx.unseen_tasks:
        return out
    res = eval_unseen(ctx, model)
    out.update(unseen_fields(res))
    logging.info(f"  unseen_test_avg_score={out['unseen_test_avg_score']:.4f}  per_task={res}")
    return out


def pack_result(
    valid_res: Dict[str, float], test_res: Dict[str, float],
    valid_loss: Optional[Dict[str, Optional[float]]] = None, **extra,
) -> dict:
    out = {
        "valid_avg_score": avg_score(valid_res),
        "test_avg_score": avg_score(test_res),
        "valid_per_task": valid_res,
        "test_per_task": test_res,
    }
    if valid_loss is not None:
        out["valid_avg_loss"] = avg_loss(valid_loss)
        out["valid_per_task_loss"] = valid_loss
    out.update(extra)
    return out


def save_finetuned(ctx, model, tokenizer, tag: str) -> str:
    """Save a finetuned causal LM (weights + tokenizer) under checkpoint_dir/<tag>."""
    path = os.path.join(ctx.checkpoint_dir, tag)
    save_causal_lm(model, tokenizer, path)
    return path


def free_model(model, device: str) -> None:
    """Drop a strategy's model (and its optimizer state, already out of scope
    by the time this is called) before the next strategy loads its own. Each
    strategy loads a full fresh causal LM (~2.4GB fp32 for a 0.6B model, more
    with weight_gd's full-parameter AdamW state or subspace_gd's base/basis
    copies) -- run_experiment runs coeff_search/weight_gd/subspace_gd
    back-to-back in one process, so without this a 20GB GPU can OOM on the
    third strategy even though each one individually fits fine."""
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
