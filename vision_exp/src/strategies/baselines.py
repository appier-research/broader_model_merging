"""Static, test-only reference points (no gradient descent).

Four budget-independent baselines that the plots draw as horizontal references:

  * ``pretrained``    -- raw base model on the test set (all coeffs = 0).
  * ``merged_avg``    -- base + method.merged_delta(coeff=1/N) on the test set,
                         where N = len(tasks). For TA this is the classic
                         task-vector average.
  * ``merged_coeff1`` -- base + method.merged_delta(coeff=1.0) on the test set.
  * ``upper_bound``   -- the joint multi-task checkpoint on the test set. Only
                         included when a checkpoint is found; silently skipped
                         otherwise.

Because none of these touch the validation budget, one run at ``budget=full`` is
enough per (tasks, arch, method); the runner gates that in ``_plan``.

``run`` is incremental: it only computes cells passed via the ``cells`` filter,
so once a run has produced e.g. {pretrained, merged_coeff1, merged_avg} and the
multi-task checkpoint later appears, a rerun fills in just ``upper_bound``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from ..models import MultiTaskCLIPClassifier, avg_metrics, load_state_dict
from ._common import eval_multitask_all
from .context import StrategyContext

ALL_CELLS = ("pretrained", "merged_avg", "merged_coeff1", "upper_bound")


def _pack_test_only(test_res: dict) -> dict:
    test_acc, test_loss = avg_metrics(test_res)
    return {
        "test_avg_acc": test_acc,
        "test_avg_loss": test_loss,
        "test_per_task": test_res,
    }


def _eval_at(model, base_sd, delta, keys, test_loaders, tasks, device) -> dict:
    """Load base+delta into the classifier and evaluate on the test loaders."""
    weights = {k: base_sd[k].float() + delta[k] for k in keys}
    model.model.load_state_dict(weights, strict=False)
    return eval_multitask_all(model, test_loaders, tasks, device)


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
    logging.info(f"\n=== Baselines (test only, no GD) === cells={sorted(todo)}")
    if not todo:
        return {}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("baselines: building fresh classifier")
    model = MultiTaskCLIPClassifier(ctx.base_model, ctx.tasks, ctx.device, ctx.class_indices)
    tasks, keys, device = ctx.tasks, ctx.keys, ctx.device
    out: dict = {}

    if "pretrained" in todo:
        model.model.load_state_dict(ctx.base_sd, strict=False)
        res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
        logging.info(f"  pretrained         test_avg_acc={avg_metrics(res)[0]:.4f}")
        out["pretrained"] = _pack_test_only(res)

    if "merged_avg" in todo:
        coeff = 1.0 / len(tasks)
        delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=coeff)
        res = _eval_at(model, ctx.base_sd, delta, keys, ctx.test_loaders, tasks, device)
        logging.info(f"  merged (coeff=1/N) test_avg_acc={avg_metrics(res)[0]:.4f}")
        out["merged_avg"] = {"coeff": coeff, **_pack_test_only(res)}

    if "merged_coeff1" in todo:
        delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=1.0)
        res = _eval_at(model, ctx.base_sd, delta, keys, ctx.test_loaders, tasks, device)
        logging.info(f"  merged (coeff=1)   test_avg_acc={avg_metrics(res)[0]:.4f}")
        out["merged_coeff1"] = {"coeff": 1.0, **_pack_test_only(res)}

    if "upper_bound" in todo:
        if multitask_checkpoint:
            mt_sd = load_state_dict(multitask_checkpoint, device="cpu")
            model.model.load_state_dict(mt_sd, strict=False)
            res = eval_multitask_all(model, ctx.test_loaders, tasks, device)
            logging.info(f"  upper_bound (MT)   test_avg_acc={avg_metrics(res)[0]:.4f}  ({multitask_checkpoint})")
            out["upper_bound"] = {"checkpoint": multitask_checkpoint, **_pack_test_only(res)}
        else:
            logging.info("  upper_bound: no multitask checkpoint found, skipping")

    return out
