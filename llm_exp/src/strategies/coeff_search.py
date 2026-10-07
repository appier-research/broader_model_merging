"""Coefficient search strategy.

Diagonal sweep of a single scalar coefficient lambda over the method's merged
direction. At each lambda, generate + score on the valid pool; the best lambda
is picked by average score across tasks, and test metrics are reported at that
lambda. No gradients. Same shape as vision_exp's coeff_search.py, just
generation-based scoring instead of classification-head accuracy.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..models import avg_loss, avg_score, load_causal_lm
from ._common import attach_unseen, clear_cache, eval_all_tasks, eval_all_tasks_loss, free_model, pack_result
from .context import StrategyContext


def _lambda_grid(lo: float, hi: float, steps: int) -> List[float]:
    if steps < 2:
        raise ValueError("lambda_steps must be >= 2")
    return [round(lo + i * (hi - lo) / (steps - 1), 10) for i in range(steps)]


def _load_merged(model, base_sd: Dict, delta: Dict, keys: List[str]) -> None:
    W = dict(base_sd)
    for k in keys:
        W[k] = base_sd[k].float() + delta[k]
    model.load_state_dict(W, strict=True)


def run(ctx: StrategyContext) -> dict:
    method, keys, tasks = ctx.method, ctx.keys, ctx.tasks
    logging.info("coeff_search: building fresh model")
    model = load_causal_lm(ctx.base_model, device=ctx.device, dtype=ctx.dtype)
    tokenizer = ctx.tokenizer

    lambdas = _lambda_grid(ctx.lambda_min, ctx.lambda_max, ctx.lambda_steps)
    logging.info(f"\n=== Coefficient search ({method.name}) ===")
    rows, best_lam, best_valid = [], lambdas[0], -1.0
    for lam in lambdas:
        delta = method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=lam)
        _load_merged(model, ctx.base_sd, delta, keys)
        valid_res = eval_all_tasks(
            model, tokenizer, ctx.valid_examples, tasks, ctx.device,
            ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
        )
        valid_avg = avg_score(valid_res)
        clear_cache(ctx.device)  # generate()'s KV cache before the loss forward's logits tensor
        valid_loss = eval_all_tasks_loss(
            model, tokenizer, ctx.sft_pools, tasks, ctx.device, ctx.loss_batch_size, ctx.max_seq_length,
        )
        clear_cache(ctx.device)  # ... and before next lambda's generate() -- see clear_cache's docstring
        logging.info(f"  lambda={lam:.4f}  valid_avg_score={valid_avg:.4f}  valid_avg_loss={avg_loss(valid_loss)}")
        rows.append({
            "lambda": lam, "valid_avg_score": valid_avg, "valid_per_task": valid_res,
            "valid_avg_loss": avg_loss(valid_loss), "valid_per_task_loss": valid_loss,
        })
        if valid_avg > best_valid:
            best_valid, best_lam = valid_avg, lam

    best_delta = method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=best_lam)
    _load_merged(model, ctx.base_sd, best_delta, keys)
    valid_best = eval_all_tasks(
        model, tokenizer, ctx.valid_examples, tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)
    valid_best_loss = eval_all_tasks_loss(
        model, tokenizer, ctx.sft_pools, tasks, ctx.device, ctx.loss_batch_size, ctx.max_seq_length,
    )
    clear_cache(ctx.device)
    test_best = eval_all_tasks(
        model, tokenizer, ctx.test_examples, tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    logging.info(
        f"  best lambda*={best_lam:.4f}  valid_avg_score={avg_score(valid_best):.4f}  "
        f"test_avg_score={avg_score(test_best):.4f}"
    )
    result = pack_result(valid_best, test_best, valid_loss=valid_best_loss, best_lambda=best_lam, sweep=rows)
    attach_unseen(ctx, model, result)
    free_model(model, ctx.device)
    return result


def _valid_pass(ctx: StrategyContext, model) -> None:
    """The part of one sweep lambda that the method actually needs: generate +
    score the valid pool (lambda* is picked on valid_avg_score alone).

    ``run`` additionally does a teacher-forced loss forward per lambda, but
    that is diagnostics (directional_sampling's threshold, logging), not part
    of coefficient search -- and at loss_batch_size=4 its vocab-sized logits
    (4 x 2048 x 152k fp32) dominate the 0.6B peak, so it is left out here."""
    eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, ctx.tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)


def profile_one(ctx: StrategyContext) -> dict:
    """Warm up one lambda (merge + valid generate+score), then time the next
    (peak mem + wall + token count).

    The timed action is everything one sweep lambda costs: building the merge
    on the CPU (``merged_delta``), loading it into the model, and the valid
    generate+score. Only the diagnostic valid-loss forward is excluded (see
    ``_valid_pass``). FLOPs count the model forward alone; the merge's CPU
    elementwise work is negligible next to it.
    """
    from ..cost import TokenCounter, measure

    logging.info("coeff_search: profile (warmup lambda + 1 timed valid pass)")
    model = load_causal_lm(ctx.base_model, device=ctx.device, dtype=ctx.dtype)
    lambdas = _lambda_grid(ctx.lambda_min, ctx.lambda_max, ctx.lambda_steps)
    delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, ctx.keys, coeff=lambdas[0])
    _load_merged(model, ctx.base_sd, delta, ctx.keys)
    _valid_pass(ctx, model)
    del delta
    with measure() as stats, TokenCounter(model) as counter:
        delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, ctx.keys, coeff=lambdas[1])
        _load_merged(model, ctx.base_sd, delta, ctx.keys)
        del delta
        _valid_pass(ctx, model)
    stats["n_fwd_tokens"] = counter.tokens
    stats["n_bwd_tokens"] = 0
    logging.info(
        f"  timed lambda={lambdas[1]:.4f}  tokens={counter.tokens}  wall={stats['action_wall_s']:.1f}s  "
        f"peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    free_model(model, ctx.device)
    return {"init": None, "action": "valid_eval", **stats}
