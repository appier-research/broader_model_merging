"""Peak-memory / wall-time / model-FLOP helpers for cost estimation.

A paper run's cost is ``one timed action × n_actions``:

  coeff_search  one valid eval  ×  lambda_steps (11)
  weight_gd     one train epoch ×  10
  subspace_gd   one train epoch ×  20

Model FLOPs count vision matmuls only (backward ≈ 2× forward).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch

WEIGHT_GD_INIT = {"ta": "avg", "dare": "avg", "ties": "merged", "tsvm": "merged"}
SUBSPACE_GD_INIT = "coeff_best"
N_WEIGHT_GD_EPOCHS = 10
N_SUBSPACE_GD_EPOCHS = 20
BWD_FLOP_FACTOR = 2.0


def weight_gd_init(method: str) -> str:
    return WEIGHT_GD_INIT.get(method, "merged")


def n_actions(strategy: str, lambda_steps: int) -> int:
    if strategy == "coeff_search":
        return int(lambda_steps)
    if strategy == "weight_gd":
        return N_WEIGHT_GD_EPOCHS
    if strategy == "subspace_gd":
        return N_SUBSPACE_GD_EPOCHS
    raise ValueError(f"unknown strategy {strategy!r}")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def measure() -> Iterator[dict]:
    """Time a block and record CUDA peak memory. Caller fills n_fwd / n_bwd."""
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    stats = {"n_fwd_images": 0, "n_bwd_images": 0}
    yield stats
    _sync()
    allocated = reserved = 0
    if torch.cuda.is_available():
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    stats["action_wall_s"] = time.perf_counter() - t0
    stats["peak_mem_allocated_bytes"] = allocated
    stats["peak_mem_reserved_bytes"] = reserved


def vision_fwd_flops_per_image(vision_model, pixel_values: torch.Tensor) -> float:
    """One-image vision FLOPs via FlopCounterMode on get_image_features."""
    from torch.utils.flop_counter import FlopCounterMode

    vision_model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        vision_model.get_image_features(pixel_values=pixel_values)
    n = max(int(pixel_values.size(0)), 1)
    return float(fc.get_total_flops()) / n


def finalize(action: dict, strategy: str, lambda_steps: int, fwd_flops: float) -> dict:
    n = n_actions(strategy, lambda_steps)
    n_fwd = int(action.get("n_fwd_images", 0))
    n_bwd = int(action.get("n_bwd_images", 0))
    action_flops = n_fwd * fwd_flops + BWD_FLOP_FACTOR * n_bwd * fwd_flops
    action["n_actions"] = n
    action["action_model_flops"] = action_flops
    action["est_wall_s"] = float(action["action_wall_s"]) * n
    action["est_model_flops"] = action_flops * n
    return action


def first_batch_pixels(loaders: Dict[str, object], tasks, device: str) -> Optional[torch.Tensor]:
    for task in tasks:
        loader = loaders.get(task)
        if loader is None:
            continue
        try:
            batch = next(iter(loader))
        except StopIteration:
            continue
        return batch["pixel_values"].to(device)
    return None
