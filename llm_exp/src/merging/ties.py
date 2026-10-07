"""TIES-Merging. Ported from vision_exp/src/merging/ties.py.

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

Memory
------
The straightforward implementation materialises every task vector at once
(``taus``), then every trimmed copy at once (``trimmed``), then stacks all
tasks per key. At 4B parameters that is 8+ full model-sized bf16 copies on top
of the base_sd/task_sds experiment.py keeps resident for the whole unit --
~22x one state dict, which on qwen3-4b (4.41B params, 8.2 GiB per state dict)
overruns a 125 GB box and swaps it to death.

``ties_merge_streaming`` below never holds more than one task vector at a time,
and accumulates the elect/merge statistics in a fixed number of model-sized
buffers. Same algorithm, ~5x less memory. It is not bit-identical to the
stacked version: summing left-to-right instead of pairwise moves the last bf16
ULP, which on near-tied entries can flip an elected sign. It accumulates in
fp32 to settle that in the accurate direction, so where the two differ the
streaming result is the closer one to the exact disjoint mean -- it matches an
fp64 reference exactly on every case in the equivalence tests. Scores computed
before this change are therefore comparable but not byte-reproducible; rerun a
unit if you need exact agreement.

Measured on the real qwen3-4b checkpoints (4 tasks, density 0.2): 49.3 GiB
resident for base_sd + 4 task_sds, 86.3 GiB peak through the merge, against
~181 GiB for the stacked version. The mirror change for Task Arithmetic is
``MergeMethod.sum_task_vectors``.
"""

from __future__ import annotations

from typing import Iterable, List

import torch

from .base import MergeMethod, StateDict


def _threshold(tau: StateDict, keys: List[str], density: float) -> torch.Tensor:
    """Magnitude cutoff keeping the top ``density`` fraction of ``tau``.

    Exact, and streams key-by-key instead of building the full concatenated
    ``|tau|`` vector that ``torch.kthvalue`` needs (one extra model-sized
    allocation, plus kthvalue's own working copy).

    bf16/fp16 have only 2**16 distinct bit patterns, so for a half-precision
    task vector we can bincount the magnitudes exactly into a 65536-bin
    histogram (0.5 MiB) and read the quantile straight off the cumulative
    counts. For any other dtype we fall back to concatenating -- fp32 task
    vectors are not what the checkpoints hold, so that path is not the one
    that has to be cheap.
    """
    dtype = tau[keys[0]].dtype
    if dtype not in (torch.bfloat16, torch.float16):
        all_abs = torch.cat([tau[k].abs().flatten() for k in keys])
        n = all_abs.numel()
        k_keep = max(1, int(round(density * n)))
        return torch.kthvalue(all_abs, n - k_keep + 1).values

    n_bins = 1 << 16
    counts = torch.zeros(n_bins, dtype=torch.int64)
    for k in keys:
        # |x| for a float is monotonic in its bit pattern, so the raw uint16 is
        # already a sort key; bincount it directly.
        bits = tau[k].abs().view(torch.int16).to(torch.int32) & 0xFFFF
        counts += torch.bincount(bits.flatten(), minlength=n_bins)

    values = torch.arange(n_bins, dtype=torch.int32).to(torch.int16).view(dtype).float()
    order = torch.argsort(values)
    cumulative = torch.cumsum(counts[order], 0)
    total = int(cumulative[-1].item())
    k_keep = max(1, int(round(density * total)))
    # Match torch.kthvalue(all_abs, n - k_keep + 1): that is the 1-indexed
    # ascending rank, i.e. 0-indexed rank n - k_keep, so the bin we want is the
    # first whose cumulative count exceeds it.
    pos = int(torch.searchsorted(cumulative, total - k_keep + 1).item())
    pos = min(pos, n_bins - 1)
    return values[order][pos].to(dtype)


def _trim(tau: StateDict, keys: List[str], density: float) -> StateDict:
    if density >= 1.0:
        return {k: tau[k].clone() for k in keys}
    threshold = _threshold(tau, keys, density)
    return {
        k: torch.where(tau[k].abs() >= threshold, tau[k], torch.zeros_like(tau[k]))
        for k in keys
    }


def _trim_in_place(tau: StateDict, keys: List[str], density: float) -> StateDict:
    """``_trim`` that reuses ``tau``'s storage.

    Only safe on a task vector we just built and own outright (see
    ``_iter_trimmed``); ``_trim`` stays non-destructive for external callers.
    """
    if density >= 1.0:
        return tau
    threshold = _threshold(tau, keys, density)
    for k in keys:
        tau[k].mul_(tau[k].abs() >= threshold)
    return tau


def _iter_trimmed(
    base_sd: StateDict, task_sds: List[StateDict], keys: List[str], density: float
) -> Iterable[StateDict]:
    """Yield each trimmed task vector in turn, holding only one at a time.

    The caller must finish with each one before advancing: the generator drops
    its reference on the next step so the memory can be reclaimed.
    """
    for sd in task_sds:
        tau = {k: sd[k] - base_sd[k] for k in keys}
        yield _trim_in_place(tau, keys, density)
        del tau


def ties_merge_streaming(
    base_sd: StateDict, task_sds: List[StateDict], keys: List[str], density: float
) -> StateDict:
    """Trim + elect + disjoint-mean, one task vector resident at a time.

    Two passes over the task vectors, because the elected sign needs the sum
    over all tasks before any task's agreement can be decided:

      pass 1  sum_i trim(tau_i)              -> elected sign
      pass 2  sum / count over sign-agreeing -> disjoint mean

    Trimming is deterministic, so recomputing each tau in pass 2 gives exactly
    the same values as pass 1. That costs a second subtract+trim per task and
    saves holding all n_tasks trimmed copies at once -- the right trade at 4B,
    where each copy is ~8 GiB.

    Peak extra memory is one fp32 accumulator plus two narrow (int8) buffers
    and the single live task vector, rather than 2 * n_tasks full copies plus
    an n_tasks-deep stack per key.
    """
    # Accumulate the running sums in fp32, not bf16. A streaming ``+=`` cannot be
    # bit-identical to the stacked ``torch.stack(...).sum(0)`` it replaces anyway
    # -- that sums pairwise, this sums left-to-right, and in bf16 the two
    # disagree by ~1 ULP per step, enough to flip an elected sign on a near-tied
    # entry. Rather than chase the old rounding, accumulate in fp32 and round
    # once at the end: against an fp64 reference the fp32 path is ~2e8x more
    # accurate than bf16, so where the two differ this is the better answer.
    #
    # The accumulators dominate peak memory here, so each is kept as narrow as
    # it can be: the sign election only needs a sign (int8, 1 byte) and the
    # disjoint-mean count only ever reaches n_tasks (int8), leaving exactly one
    # fp32 buffer live at a time.
    dtype = base_sd[keys[0]].dtype

    # Pass 1: elected sign = sign(sum_i trimmed_i). The fp32 running sum is
    # collapsed to an int8 sign per key as soon as that key is finished, so the
    # wide buffer never coexists with pass 2's.
    sum_acc: StateDict = {k: torch.zeros(base_sd[k].shape, dtype=torch.float32) for k in keys}
    for trimmed in _iter_trimmed(base_sd, task_sds, keys, density):
        for k in keys:
            sum_acc[k] += trimmed[k].float()
    elected_sign: StateDict = {}
    for k in keys:
        elected_sign[k] = torch.sign(sum_acc[k]).to(torch.int8)
        sum_acc[k] = None
    del sum_acc

    # Pass 2: disjoint mean over the entries whose sign matches the election.
    value_acc: StateDict = {k: torch.zeros(base_sd[k].shape, dtype=torch.float32) for k in keys}
    # int8 counts every task exactly once, so it only has to reach n_tasks.
    count_dtype = torch.int8 if len(task_sds) <= 127 else torch.int32
    count_acc: StateDict = {k: torch.zeros(base_sd[k].shape, dtype=count_dtype) for k in keys}
    for trimmed in _iter_trimmed(base_sd, task_sds, keys, density):
        for k in keys:
            t = trimmed[k]
            # Compare signs in the narrow dtype; only the accumulation widens.
            agree = torch.sign(t).to(torch.int8) == elected_sign[k]
            value_acc[k] += torch.where(agree, t, torch.zeros_like(t)).float()
            count_acc[k] += (agree & (t != 0)).to(count_dtype)

    merged: StateDict = {}
    for k in keys:
        merged[k] = (value_acc[k] / count_acc[k].clamp(min=1)).to(dtype)
        # Free as we go so the accumulators do not stay resident alongside the
        # finished result.
        value_acc[k] = None
        count_acc[k] = None
        elected_sign[k] = None
    return merged


def ties_merge(taus: List[StateDict], keys: List[str], density: float) -> StateDict:
    """Non-streaming TIES over pre-built task vectors.

    Kept for callers that already hold the taus (and for the equivalence test
    against ``ties_merge_streaming``); ``TiesMerging`` itself uses the
    streaming path.
    """
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
            # Drop the old merge before building the new one so the two don't
            # coexist (a full model-sized copy each), same as TA's _summed.
            self._cache_val = None
            self._cache_key = None
            self._cache_val = ties_merge_streaming(base_sd, task_sds, keys, self.density)
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
