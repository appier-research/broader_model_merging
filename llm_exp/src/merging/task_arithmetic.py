"""Task Arithmetic (TA). Ported from vision_exp/src/merging/task_arithmetic.py.

The merged direction is the sum of task vectors, tau = sum_i tau_i, scaled by a
single coefficient. The subspace basis is the set of n per-task task vectors, so
subspace GD optimizes one coefficient per task.

``merged_delta`` caches the (coefficient-independent) sum, mirroring TIES:
coeff_search sweeps ~10 lambdas over the same task_sds, and recomputing the
sum per lambda meant re-materializing every task vector each time.
"""

from __future__ import annotations

from typing import List, Optional

from .base import MergeMethod, StateDict


class TaskArithmetic(MergeMethod):
    name = "ta"

    def __init__(self) -> None:
        self._cache_key = None
        self._cache_val: Optional[StateDict] = None

    def _summed(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> StateDict:
        """sum_i tau_i, cached across coefficients.

        Keyed on the identity of the inputs, same convention as
        TiesMerging._merged: experiment.py loads base_sd/task_sds once and
        holds them for the whole unit, so every coefficient reuses one sum.
        """
        key = (id(base_sd), tuple(id(s) for s in task_sds), tuple(keys))
        if key != self._cache_key:
            # Drop the old sum before building the new one so the two don't
            # coexist (a full model-sized copy each).
            self._cache_val = None
            self._cache_key = None
            self._cache_val = self.sum_task_vectors(base_sd, task_sds, keys)
            self._cache_key = key
        return self._cache_val

    def basis(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> List[StateDict]:
        return self.task_vectors(base_sd, task_sds, keys)

    def merged_delta(
        self,
        base_sd: StateDict,
        task_sds: List[StateDict],
        keys: List[str],
        coeff: float = 1.0,
    ) -> StateDict:
        summed = self._summed(base_sd, task_sds, keys)
        return {k: coeff * summed[k] for k in keys}
