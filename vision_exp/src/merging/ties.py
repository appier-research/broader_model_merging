"""TIES-Merging.

Three steps (Yadav et al., 2023):
  1. Trim   -- keep only the top ``density`` fraction of each task vector by
               magnitude (computed globally across all parameters of that task
               vector); zero the rest.
  2. Elect  -- for each parameter, the aggregate sign is the sign of the sum of
               the trimmed values across tasks.
  3. Merge  -- for each parameter, average the trimmed values whose sign agrees
               with the elected sign (disjoint mean).

This yields a single merged direction tau_ties, scaled by one coefficient. The
subspace basis is therefore a single direction (subspace GD optimizes one
scalar). ``density`` is configurable so the method stays flexible.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .base import MergeMethod, StateDict


def _trim(tau: StateDict, keys: List[str], density: float) -> StateDict:
    if density >= 1.0:
        return {k: tau[k].clone() for k in keys}
    all_abs = torch.cat([tau[k].abs().flatten() for k in keys])
    n = all_abs.numel()
    k_keep = max(1, int(round(density * n)))
    if k_keep >= n:
        return {k: tau[k].clone() for k in keys}
    # threshold = k_keep-th largest magnitude
    threshold = torch.kthvalue(all_abs, n - k_keep + 1).values
    return {
        k: torch.where(tau[k].abs() >= threshold, tau[k], torch.zeros_like(tau[k]))
        for k in keys
    }


def ties_merge(taus: List[StateDict], keys: List[str], density: float) -> StateDict:
    trimmed = [_trim(tau, keys, density) for tau in taus]
    merged: StateDict = {}
    for k in keys:
        stacked = torch.stack([t[k] for t in trimmed])          # (n_tasks, *shape)
        elected_sign = torch.sign(stacked.sum(0))               # (*shape,)
        agree = torch.where(torch.sign(stacked) == elected_sign, stacked, torch.zeros_like(stacked))
        count = (agree != 0).sum(0).clamp(min=1)
        merged[k] = agree.sum(0) / count
    return merged


class TiesMerging(MergeMethod):
    name = "ties"

    def __init__(self, density: float = 0.2):
        self.density = density
        self._cache_key = None
        self._cache_val = None

    def _merged(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> StateDict:
        key = (id(base_sd), tuple(id(s) for s in task_sds), tuple(keys), self.density)
        if key != self._cache_key:
            taus = self.task_vectors(base_sd, task_sds, keys)
            self._cache_val = ties_merge(taus, keys, self.density)
            self._cache_key = key
        return self._cache_val

    def basis(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> List[StateDict]:
        return [self._merged(base_sd, task_sds, keys)]

    def merged_delta(
        self,
        base_sd: StateDict,
        task_sds: List[StateDict],
        keys: List[str],
        coeff: float = 1.0,
    ) -> StateDict:
        merged = self._merged(base_sd, task_sds, keys)
        return {k: coeff * merged[k] for k in keys}
