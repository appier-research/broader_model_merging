"""Coefficient-selection strategies.

Each strategy consumes a :class:`StrategyContext` and returns a
JSON-serializable result. ``coeff_search`` returns a single dict;
``weight_gd``, ``subspace_gd``, ``directional_sampling`` and ``bo_search`` return
a dict keyed by init point. ``baselines`` returns a dict keyed by cell name (see
baselines.ALL_CELLS).
"""

from . import baselines, bo_search, coeff_search, directional_sampling, subspace_gd, weight_gd, weight_gd_lora
from .context import StrategyContext

# Strategies that fine-tune from one or more init points.
INIT_STRATEGIES = ("weight_gd", "weight_gd_lora", "subspace_gd", "directional_sampling", "bo_search")
ALL_STRATEGIES = ("coeff_search", "weight_gd", "weight_gd_lora", "subspace_gd", "baselines",
                  "directional_sampling", "bo_search")
DEFAULT_INITS = ("pretrained", "avg", "merged", "coeff_best")
# directional_sampling additionally accepts two per-direction (vector) centers:
# 'subspace_best' = subspace_gd's trained coefficients, 'bo_best' = bo_search's
# best trial -- each from that strategy's best cell by valid score.
DS_INITS = DEFAULT_INITS + ("subspace_best", "bo_best")
# bo_search centers its search box on one of the four scalar inits (default:
# coeff_best only -- it is a standalone method, see strategies/bo_search.py).
BO_INITS = DEFAULT_INITS

__all__ = [
    "baselines",
    "coeff_search",
    "weight_gd",
    "weight_gd_lora",
    "subspace_gd",
    "directional_sampling",
    "bo_search",
    "StrategyContext",
    "INIT_STRATEGIES",
    "ALL_STRATEGIES",
    "DEFAULT_INITS",
    "DS_INITS",
    "BO_INITS",
]
