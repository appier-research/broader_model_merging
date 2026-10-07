"""Coefficient-selection strategies.

Each strategy consumes a :class:`StrategyContext` and returns a JSON-serializable
result. ``coeff_search`` returns a single dict; ``weight_gd`` and ``subspace_gd``
return a dict keyed by init point.
"""

from . import (baselines, bo_search, coeff_search, directional_sampling, pseudo_labels,
               subspace_gd, test_time_adaptation, weight_gd)
from .test_time_adaptation import VARIANTS as TTA_VARIANTS
from .context import StrategyContext
from .pseudo_labels import DIVERGENCES, LABEL_SOURCES

# Strategies that fine-tune from one or more init points.
INIT_STRATEGIES = ("weight_gd", "subspace_gd", "bo_search")
ALL_STRATEGIES = ("coeff_search", "weight_gd", "subspace_gd", "baselines",
                  "directional_sampling", "bo_search", "adamerging", "divmerge")
# Test-time adaptation strategies: train on ctx.unlabeled_loaders (weight_gd
# joins when gd_label_source != gt).
TTA_STRATEGIES = ("adamerging", "divmerge")
DEFAULT_INITS = ("pretrained", "avg", "merged", "coeff_best")
# bo_search centers its search box on one of the four scalar inits (default:
# coeff_best only -- it is a standalone method, see strategies/bo_search.py).
BO_INITS = DEFAULT_INITS

__all__ = [
    "baselines",
    "coeff_search",
    "weight_gd",
    "subspace_gd",
    "directional_sampling",
    "bo_search",
    "StrategyContext",
    "INIT_STRATEGIES",
    "ALL_STRATEGIES",
    "DEFAULT_INITS",
    "BO_INITS",
    "LABEL_SOURCES",
    "DIVERGENCES",
    "TTA_STRATEGIES",
    "TTA_VARIANTS",
    "pseudo_labels",
    "test_time_adaptation",
]
