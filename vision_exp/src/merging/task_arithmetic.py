"""Task Arithmetic (TA).

The merged direction is the sum of task vectors, tau = sum_i tau_i, scaled by a
single coefficient. The subspace basis is the set of n per-task task vectors, so
subspace GD optimizes one coefficient per task.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .base import MergeMethod, StateDict


class TaskArithmetic(MergeMethod):
    name = "ta"

    def basis(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> List[StateDict]:
        return self.task_vectors(base_sd, task_sds, keys)

    def merged_delta(
        self,
        base_sd: StateDict,
        task_sds: List[StateDict],
        keys: List[str],
        coeff: float = 1.0,
    ) -> StateDict:
        taus = self.task_vectors(base_sd, task_sds, keys)
        out = {k: torch.zeros_like(taus[0][k]) for k in keys}
        for tau in taus:
            for k in keys:
                out[k] += tau[k]
        return {k: coeff * out[k] for k in keys}
