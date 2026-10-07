"""Draw a fixed class subset for the smaller_9task runs.

Each task starts with ``n_classes // n_tasks`` classes. The ``n_classes % n_tasks``
extra classes go one each to the smallest tasks (ties keep the given task order).
A task is never given more classes than it has; a leftover slot moves to the
next-smallest task that still has room. Tasks left at zero are omitted.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .classnames import get_num_classes


def allocate_class_counts(tasks: List[str], n_classes: int) -> Dict[str, int]:
    """How many classes each task contributes. Values may be zero."""
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    available = {t: get_num_classes(t) for t in tasks}
    total = sum(available.values())
    if n_classes > total:
        raise ValueError(f"n_classes={n_classes} exceeds the {total} classes in {tasks}")

    n_tasks = len(tasks)
    base = n_classes // n_tasks
    counts = {t: min(base, available[t]) for t in tasks}
    leftover = n_classes - sum(counts.values())
    order = sorted(tasks, key=lambda t: (available[t], tasks.index(t)))
    while leftover > 0:
        placed = False
        for task in order:
            if counts[task] < available[task]:
                counts[task] += 1
                leftover -= 1
                placed = True
                if leftover == 0:
                    break
        if not placed:
            raise ValueError(f"could not place {n_classes} classes across {tasks}")
    return counts


def select_kept_classes(tasks: List[str], n_classes: int, class_seed: int) -> Dict[str, List[int]]:
    """Original label ids to keep, sorted, for every task that gets at least one.

    The draw is ``random.Random(class_seed)`` walked in ``tasks`` order, so the
    same seed always yields the same subset.
    """
    counts = allocate_class_counts(tasks, n_classes)
    rng = random.Random(class_seed)
    kept: Dict[str, List[int]] = {}
    for task in tasks:
        k = counts[task]
        if k <= 0:
            continue
        ids = list(range(get_num_classes(task)))
        rng.shuffle(ids)
        kept[task] = sorted(ids[:k])
    return kept
