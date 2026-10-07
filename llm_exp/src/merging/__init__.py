"""Registry of merging methods. Ported from vision_exp/src/merging/__init__.py."""

from __future__ import annotations

from typing import List

from .base import MergeMethod
from .dare import Dare
from .task_arithmetic import TaskArithmetic
from .ties import TiesMerging
from .tsv_m import TsvMerging

_REGISTRY = {
    "ta": TaskArithmetic,
    "ties": TiesMerging,
    "dare": Dare,
    "tsvm": TsvMerging,
}


def get_method(name: str, **kwargs) -> MergeMethod:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown merging method '{name}'. Available: {available_methods()}")
    return _REGISTRY[name](**kwargs)


def available_methods() -> List[str]:
    return sorted(_REGISTRY)
