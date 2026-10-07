"""DARE (Drop And REscale) merging.

Yu et al., 2024 ("Language Models are Super Mario: Absorbing Abilities from
Homologous Models as a Free Lunch"). Applied independently to each task
vector:

  1. Drop    -- each parameter of tau_i is zeroed with probability
                ``drop_rate`` (i.i.d. Bernoulli mask), sparsifying the
                (largely redundant) delta.
  2. Rescale -- surviving parameters are scaled by 1 / (1 - drop_rate) so the
                delta's expectation is unchanged.

The dropped-and-rescaled task vectors are then combined exactly like Task
Arithmetic: tau = sum_i tau_i_dare, scaled by a single coefficient. The
subspace basis is therefore the n per-task DARE vectors (one coefficient per
task, same shape as TA's basis) -- DARE is a preprocessing step on top of TA,
not a different combination rule.

``drop_rate`` and ``seed`` are configurable. The mask is deterministic given
(base_sd, task_sds, keys, drop_rate, seed) and cached so repeated calls
(``basis`` then ``merged_delta`` within one run) see the same dropped vectors.
"""

from __future__ import annotations

from typing import List

import torch

from .base import MergeMethod, StateDict


def _drop_and_rescale(
    tau: StateDict, keys: List[str], drop_rate: float, generator: torch.Generator
) -> StateDict:
    if drop_rate <= 0.0:
        return {k: tau[k].clone() for k in keys}
    keep_prob = 1.0 - drop_rate
    out: StateDict = {}
    for k in keys:
        mask = torch.rand(tau[k].shape, generator=generator, device="cpu").to(tau[k].device) < keep_prob
        out[k] = torch.where(mask, tau[k] / keep_prob, torch.zeros_like(tau[k]))
    return out


class Dare(MergeMethod):
    name = "dare"

    def __init__(self, drop_rate: float = 0.5, seed: int = 0):
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        self.drop_rate = drop_rate
        self.seed = seed
        self._cache_key = None
        self._cache_val = None

    def _dare_task_vectors(
        self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]
    ) -> List[StateDict]:
        key = (id(base_sd), tuple(id(s) for s in task_sds), tuple(keys), self.drop_rate, self.seed)
        if key != self._cache_key:
            taus = self.task_vectors(base_sd, task_sds, keys)
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self._cache_val = [_drop_and_rescale(tau, keys, self.drop_rate, generator) for tau in taus]
            self._cache_key = key
        return self._cache_val

    def basis(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> List[StateDict]:
        return self._dare_task_vectors(base_sd, task_sds, keys)

    def merged_delta(
        self,
        base_sd: StateDict,
        task_sds: List[StateDict],
        keys: List[str],
        coeff: float = 1.0,
    ) -> StateDict:
        taus = self._dare_task_vectors(base_sd, task_sds, keys)
        out = {k: torch.zeros_like(taus[0][k]) for k in keys}
        for tau in taus:
            for k in keys:
                out[k] += tau[k]
        return {k: coeff * out[k] for k in keys}
