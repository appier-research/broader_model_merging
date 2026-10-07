"""Merging-method registry.

Add a new method by implementing a :class:`MergeMethod` subclass and registering
it here. Everything downstream (strategies, experiment runner, scripts) works
through the registry, so no other file needs to change.
"""

from typing import Dict, Type

from .base import MergeMethod
from .task_arithmetic import TaskArithmetic
from .ties import TiesMerging
from .dare import Dare
from .tsv_m import TsvMerging

_REGISTRY: Dict[str, Type[MergeMethod]] = {
    "ta": TaskArithmetic,
    "ties": TiesMerging,
    "dare": Dare,
    "tsvm": TsvMerging,
}


def available_methods():
    return sorted(_REGISTRY)


def get_method(name: str, **kwargs) -> MergeMethod:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown merging method '{name}'. Available: {available_methods()}")
    return _REGISTRY[key](**kwargs)


__all__ = ["MergeMethod", "get_method", "available_methods"]
