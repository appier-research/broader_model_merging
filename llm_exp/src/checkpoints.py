"""Resolve single-task checkpoint paths from the task-vectors folder.

Checkpoints (downloaded via admin/download_checkpoints.sh) live in
checkpoints/task_vectors/<task>_<arch>/checkpoint-<step>/, matching the GCS
bucket layout. Simpler than vision_exp's version: there's no CLIP-style
patch-token glue (``vit-b-32 -> patch32``) to worry about, since every
arch shares one training convention -- resolution
is just an exact ``<task>_<arch>`` match, then descend into the highest-step
``checkpoint-*`` subdir that actually holds model weights.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Optional


def _step(path: str) -> int:
    m = re.search(r"checkpoint-(?:step)?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _has_weights(d: str) -> bool:
    return (
        os.path.exists(os.path.join(d, "model.safetensors"))
        or os.path.exists(os.path.join(d, "model.safetensors.index.json"))
        or os.path.exists(os.path.join(d, "pytorch_model.bin"))
        or os.path.exists(os.path.join(d, "pytorch_model.bin.index.json"))
        or bool(glob.glob(os.path.join(d, "*.bin")))
    )


def _dir_with_weights(top: str) -> Optional[str]:
    """Return the directory under ``top`` that holds the weights: either
    ``top`` itself, or its highest-step checkpoint-* subdir."""
    if _has_weights(top):
        return top
    subs = [d for d in glob.glob(os.path.join(top, "checkpoint-*")) if os.path.isdir(d)]
    for d in sorted(subs, key=_step, reverse=True):
        if _has_weights(d):
            return d
    return None


def resolve_task_checkpoint(root: str, task: str, arch: str) -> str:
    """Resolve one task's checkpoint dir (the one containing the weights)."""
    top = os.path.join(root, f"{task}_{arch}")
    if not os.path.isdir(top):
        raise FileNotFoundError(f"No checkpoint dir for task='{task}' arch='{arch}': expected {top}")
    found = _dir_with_weights(top)
    if found is None:
        raise FileNotFoundError(f"No weights under {top} (nor any checkpoint-* subdir)")
    return found


def resolve_task_checkpoints(root: str, tasks: List[str], arch: str) -> List[str]:
    return [resolve_task_checkpoint(root, t, arch) for t in tasks]


def resolve_multitask_checkpoint(root: str, tasks: List[str], arch: str) -> Optional[str]:
    """Resolve the joint multi-task checkpoint for ``tasks`` (order-sensitive).

    Looks for an exact ``multitask_<t1>+<t2>+...+<tN>_<arch>`` dir under
    ``root`` and descends into the weight-bearing checkpoint-* subdir. Returns
    None when no such checkpoint exists so callers can treat the upper bound
    as optional, same as vision_exp's version.
    """
    if not tasks:
        return None
    top = os.path.join(root, f"multitask_{'+'.join(tasks)}_{arch}")
    if not os.path.isdir(top):
        return None
    return _dir_with_weights(top)
