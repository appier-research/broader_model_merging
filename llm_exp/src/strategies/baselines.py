"""Static reference points (no gradient descent).

Four budget-independent baselines, same shape as vision_exp's
``strategies/baselines.py``, generation-based scoring instead of a
classification-head accuracy:

  * ``pretrained``    -- raw base model (all coeffs = 0).
  * ``merged_avg``    -- base + method.merged_delta(coeff=1/N), where N =
                         len(tasks). For TA this is the classic task-vector
                         average.
  * ``merged_coeff1`` -- base + method.merged_delta(coeff=1.0).
  * ``upper_bound``   -- the joint multi-task checkpoint. Only included when a
                         checkpoint is found; silently skipped otherwise (no
                         multitask LLM checkpoint exists yet -- see
                         checkpoints.resolve_multitask_checkpoint).

Each cell reports test accuracy (the original, budget-independent reference
number) plus valid accuracy and valid teacher-forced loss -- unlike
coeff_search/weight_gd/subspace_gd, these aren't picked by any validation
criterion, so the valid numbers here are read-only context, not a selection
signal. Still computed once at ``budget=full`` only (same gate as before, in
``experiment._plan``): the valid pool at other budgets is a strict subset, so
nothing is gained by re-running these budget-independent cells against it.

``run`` is incremental: it only computes cells passed via the ``cells`` filter,
so once a run has produced e.g. {pretrained, merged_coeff1, merged_avg} and a
multitask checkpoint later appears, a rerun fills in just ``upper_bound``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from ..models import avg_loss, avg_score, load_causal_lm, load_state_dict
from ._common import attach_unseen, clear_cache, eval_all_tasks, eval_all_tasks_loss, free_model
from .context import StrategyContext

ALL_CELLS = ("pretrained", "merged_avg", "merged_coeff1", "upper_bound")


def _pack(valid_res: dict, valid_loss: dict, test_res: dict) -> dict:
    return {
        "valid_avg_score": avg_score(valid_res), "valid_per_task": valid_res,
        "valid_avg_loss": avg_loss(valid_loss), "valid_per_task_loss": valid_loss,
        "test_avg_score": avg_score(test_res), "test_per_task": test_res,
    }


def _weights_at(ctx: StrategyContext, coeff: float) -> dict:
    delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, ctx.keys, coeff=coeff)
    W = dict(ctx.base_sd)
    for k in ctx.keys:
        W[k] = ctx.base_sd[k].float() + delta[k]
    return W


def _eval(model, ctx: StrategyContext) -> tuple:
    """(valid_res, valid_loss, test_res) for the currently-loaded weights."""
    valid_res = eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, ctx.tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)
    valid_loss = eval_all_tasks_loss(
        model, ctx.tokenizer, ctx.sft_pools, ctx.tasks, ctx.device, ctx.loss_batch_size, ctx.max_seq_length,
    )
    clear_cache(ctx.device)
    test_res = eval_all_tasks(
        model, ctx.tokenizer, ctx.test_examples, ctx.tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    return valid_res, valid_loss, test_res


def run(
    ctx: StrategyContext,
    multitask_checkpoint: Optional[str] = None,
    cells: Optional[Iterable[str]] = None,
) -> dict:
    """Compute the requested baseline cells and return them as a dict.

    ``cells`` filters which of :data:`ALL_CELLS` to compute; ``None`` means all.
    ``upper_bound`` is silently dropped when ``multitask_checkpoint`` is None,
    even if requested.
    """
    todo = set(ALL_CELLS if cells is None else cells)
    logging.info(f"\n=== Baselines (no GD) === cells={sorted(todo)}")
    if not todo:
        return {}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("baselines: building fresh model")
    model = load_causal_lm(ctx.base_model, device=ctx.device, dtype=ctx.dtype)
    out: dict = {}

    if "pretrained" in todo:
        model.load_state_dict(dict(ctx.base_sd), strict=True)
        valid_res, valid_loss, test_res = _eval(model, ctx)
        logging.info(
            f"  pretrained         valid_avg_score={avg_score(valid_res):.4f}  "
            f"valid_avg_loss={avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}"
        )
        out["pretrained"] = attach_unseen(ctx, model, _pack(valid_res, valid_loss, test_res))

    if "merged_avg" in todo:
        coeff = 1.0 / len(ctx.tasks)
        model.load_state_dict(_weights_at(ctx, coeff), strict=True)
        valid_res, valid_loss, test_res = _eval(model, ctx)
        logging.info(
            f"  merged (coeff=1/N) valid_avg_score={avg_score(valid_res):.4f}  "
            f"valid_avg_loss={avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}"
        )
        out["merged_avg"] = attach_unseen(ctx, model, {"coeff": coeff, **_pack(valid_res, valid_loss, test_res)})

    if "merged_coeff1" in todo:
        model.load_state_dict(_weights_at(ctx, 1.0), strict=True)
        valid_res, valid_loss, test_res = _eval(model, ctx)
        logging.info(
            f"  merged (coeff=1)   valid_avg_score={avg_score(valid_res):.4f}  "
            f"valid_avg_loss={avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}"
        )
        out["merged_coeff1"] = attach_unseen(ctx, model, {"coeff": 1.0, **_pack(valid_res, valid_loss, test_res)})

    if "upper_bound" in todo:
        if multitask_checkpoint:
            mt_sd = load_state_dict(multitask_checkpoint, device="cpu")
            model.load_state_dict(mt_sd, strict=True)
            valid_res, valid_loss, test_res = _eval(model, ctx)
            logging.info(
                f"  upper_bound (MT)   valid_avg_score={avg_score(valid_res):.4f}  "
                f"valid_avg_loss={avg_loss(valid_loss)}  test_avg_score={avg_score(test_res):.4f}  "
                f"({multitask_checkpoint})"
            )
            out["upper_bound"] = attach_unseen(
                ctx, model, {"checkpoint": multitask_checkpoint, **_pack(valid_res, valid_loss, test_res)}
            )
        else:
            logging.info("  upper_bound: no multitask checkpoint found, skipping")

    free_model(model, ctx.device)
    return out
