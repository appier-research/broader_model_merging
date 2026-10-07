"""Thin optional W&B helpers for merge experiments (online by default).

All entry points no-op when W&B is unusable. That covers the obvious case (not
installed) and a subtler one: this repo writes run logs to ``vision_exp/wandb/``,
and scripts put the repo root first on ``sys.path``, so ``import wandb`` can
resolve to that *directory* as an empty namespace package. It imports fine but
has no attributes, so probing for ``run`` is what actually tells the two apart.
"""

from __future__ import annotations

import os
from typing import Optional

from .data import parse_budget


def _budget_label(budget) -> str:
    b = parse_budget(budget)
    return f"{b}per_class" if isinstance(b, int) else b


def _wandb():
    """The real wandb module, or None if it is missing or shadowed."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb if hasattr(wandb, "run") else None


def init_run(cfg, unit: str) -> None:
    if not getattr(cfg, "wandb_project", None):
        return
    os.environ.setdefault("WANDB_MODE", "online")
    wandb = _wandb()
    if wandb is None:
        return

    bl = _budget_label(cfg.budget)
    name = bl if bl == "full" else f"{bl}_seed{cfg.seed}"
    group = cfg.wandb_group or f"{cfg.arch}/{cfg.method}"
    cfg_dict = dict(cfg.__dict__)
    cfg_dict["force"] = sorted(getattr(cfg, "force", []) or [])
    cfg_dict["budget"] = bl
    wandb.init(
        project=cfg.wandb_project,
        group=group,
        name=name,
        config=cfg_dict,
        reinit=True,
        tags=[cfg.arch, cfg.method, bl],
    )
    wandb.run.summary["unit"] = unit


def log(data: dict, step: Optional[int] = None) -> None:
    wandb = _wandb()
    if wandb is None:
        return
    if wandb.run is not None:
        wandb.log(data, step=step)


def log_epoch(prefix: str, epoch: int, metrics: dict) -> None:
    """Log per-epoch metrics with a per-prefix x-axis (avoids global-step clashes across inits)."""
    wandb = _wandb()
    if wandb is None or wandb.run is None:
        return
    step_key = f"{prefix}/epoch"
    # define_metric is idempotent for the same keys within a run
    wandb.define_metric(step_key)
    wandb.define_metric(f"{prefix}/*", step_metric=step_key)
    payload = {step_key: epoch, **{f"{prefix}/{k}": v for k, v in metrics.items()}}
    wandb.log(payload)


def log_result(prefix: str, result: dict) -> None:
    payload = {}
    for k in ("valid_avg_acc", "valid_avg_loss", "test_avg_acc", "test_avg_loss",
              "density_valid", "n_beat_coeff_best"):
        if k in result:
            payload[f"{prefix}/{k}"] = result[k]
    if payload:
        # No global step — summary-style finals must not rewind the run step.
        log(payload)


def finish() -> None:
    wandb = _wandb()
    if wandb is None:
        return
    if wandb.run is not None:
        wandb.finish()
