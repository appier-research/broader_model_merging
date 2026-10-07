"""Resolve single-task checkpoint paths from a single task-vectors folder.

Checkpoints (downloaded from GCS or trained locally) all live in one directory,
e.g. checkpoints/task_vectors/, with names like:

    dtd_openai_clip-vit-base-patch32_ep10_bs32_lr1e-5/checkpoint-1062/
    gtsrb_openai_clip-vit-base-patch32_ep10_bs48_lr3e-5/checkpoint-4995/
    multitask_eurosat+gtsrb_clip-vit-base-patch32_.../checkpoint-final/
    dtd_vit-b-32/                                  # freshly trained (this repo)

The variable ep/bs/lr suffix and the checkpoint-* subdir make a fixed template
impossible, so we resolve by:

  1. exact match "<task>_<arch>" (this repo's training convention), else
  2. glob "<task>_*<patch>*" (HF/GCS convention), where patch is derived from arch,

then descend into the checkpoint-* subdir that actually holds model.safetensors.
Ambiguous matches raise, so a wrong checkpoint is never picked silently.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Optional

ARCH_TO_PATCH = {
    "vit-b-32": "patch32",
    "vit-b-16": "patch16",
    "vit-l-14": "patch14",
}


def arch_patch_token(arch: str) -> str:
    if arch not in ARCH_TO_PATCH:
        raise ValueError(f"Unknown arch '{arch}'. Known: {sorted(ARCH_TO_PATCH)}")
    return ARCH_TO_PATCH[arch]


def _step(path: str) -> int:
    # matches checkpoint-1062, checkpoint-10000, checkpoint-step10000
    m = re.search(r"checkpoint-(?:step)?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _has_weights(d: str) -> bool:
    return os.path.exists(os.path.join(d, "model.safetensors")) or bool(
        glob.glob(os.path.join(d, "*.bin"))
    )


def _dir_with_weights(top: str) -> Optional[str]:
    """Return the directory under ``top`` that holds the weights.

    Either ``top`` itself (has model.safetensors / .bin) or its highest-step
    checkpoint-* subdir.
    """
    if _has_weights(top):
        return top
    subs = [d for d in glob.glob(os.path.join(top, "checkpoint-*")) if os.path.isdir(d)]
    for d in sorted(subs, key=_step, reverse=True):
        if _has_weights(d):
            return d
    return None


def resolve_task_checkpoint(root: str, task: str, arch: str) -> str:
    """Resolve one task's checkpoint dir (the one containing the weights)."""
    exact = os.path.join(root, f"{task}_{arch}")
    if os.path.isdir(exact):
        found = _dir_with_weights(exact)
        if found is None:
            raise FileNotFoundError(f"No weights under {exact} (nor any checkpoint-* subdir)")
        return found

    patch = arch_patch_token(arch)
    matches = [
        p for p in sorted(glob.glob(os.path.join(root, f"{task}_*{patch}*")))
        if os.path.isdir(p) and not os.path.basename(p).startswith("multitask")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint for task='{task}' arch='{arch}' in {root} "
            f"(looked for '{task}_{arch}' or '{task}_*{patch}*')"
        )
    if len(matches) > 1:
        names = [os.path.basename(m) for m in matches]
        raise ValueError(
            f"Ambiguous checkpoints for task='{task}' arch='{arch}' in {root}: {names}. "
            f"Remove duplicates or pass explicit --task-checkpoints."
        )
    found = _dir_with_weights(matches[0])
    if found is None:
        raise FileNotFoundError(f"No weights under {matches[0]} (nor any checkpoint-* subdir)")
    return found


def resolve_task_checkpoints(root: str, tasks: List[str], arch: str) -> List[str]:
    return [resolve_task_checkpoint(root, t, arch) for t in tasks]


def resolve_multitask_checkpoint(root: str, tasks: List[str], arch: str) -> Optional[str]:
    """Resolve the joint multi-task checkpoint for ``tasks`` (order-sensitive).

    Looks for ``multitask_<t1>+<t2>+...+<tN>_*<patch>*/`` under ``root`` and
    descends into the weight-bearing checkpoint-* subdir. Returns None when no
    matching directory exists so callers can treat the upper bound as optional.
    """
    if not tasks:
        return None
    patch = arch_patch_token(arch)
    stem = "multitask_" + "+".join(tasks)
    matches = [
        p for p in sorted(glob.glob(os.path.join(root, f"{stem}_*{patch}*")))
        if os.path.isdir(p)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        names = [os.path.basename(m) for m in matches]
        raise ValueError(
            f"Ambiguous multitask checkpoints for tasks={tasks} arch='{arch}' in {root}: {names}. "
            f"Remove duplicates or pass explicit --multitask-checkpoint."
        )
    return _dir_with_weights(matches[0])
