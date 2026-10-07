"""TSV-Merge (TSV-M): Task Singular Vectors merging.

Gargiulo et al., "Task Singular Vectors: Reducing Task Interference in Model
Merging" (CVPR 2025), https://arxiv.org/abs/2412.00081 -- Algorithm 1.

Per weight *matrix* (any parameter with >= 2 dims; 1-D parameters such as
biases and LayerNorm weights/biases are merged by plain summation, since
SVD-based "task interference" isn't a meaningful notion for a vector):

  1. SVD each task's delta:              tau_i = U_i S_i V_i^T
  2. Truncate to the top k components,   k = floor(min(rows, cols) / n_tasks)
     (keeps ~1/n_tasks of each layer's rank, so the merged matrix stays
     roughly full-rank once all tasks are stacked back together).
  3. Concatenate the truncated factors across tasks:
        U = [U_1 | U_2 | ... | U_n]     (rows, n*k)
        V = [V_1 | V_2 | ... | V_n]     (cols, n*k)
        S = block_diag(S_1, ..., S_n)   (n*k, n*k)
  4. Orthogonalize U and V independently via the orthogonal Procrustes / polar
     factor of the concatenation itself: for A = P diag(D) Q^T (thin SVD), the
     nearest matrix with orthonormal columns is A_orth = P @ Q^T. This
     decorrelates the different tasks' directions ("Singular Task
     Interference") before they are added together -- the key difference from
     plain Task Arithmetic on the same truncated vectors.
  5. Reconstruct the merged layer:      M = U_orth @ S @ V_orth^T

The paper's alpha (default 1.0, no search needed) is just the scalar
coefficient here, so the subspace basis is a single merged direction -- same
shape as TIES.

Higher-rank tensors (e.g. Conv2d kernels, shape out x in x kh x kw) are
flattened to 2-D (out, in*kh*kw) for the SVD and reshaped back afterwards.
"""

from __future__ import annotations

from typing import List

import torch

from .base import MergeMethod, StateDict

Tensor = torch.Tensor


def _to_matrix(t: Tensor) -> Tensor:
    """Flatten every dim but the first into columns (e.g. conv kernels)."""
    return t.reshape(t.shape[0], -1)


def _orthogonalize(mat: Tensor) -> Tensor:
    """Nearest matrix with orthonormal columns (orthogonal Procrustes / polar
    factor), via the thin SVD: A = P diag(D) Q^T  ->  A_orth = P @ Q^T."""
    p, _, qh = torch.linalg.svd(mat, full_matrices=False)
    return p @ qh


def _tsv_merge_matrix(mats: List[Tensor], n_tasks: int) -> Tensor:
    rows, cols = mats[0].shape
    k = max(1, min(rows, cols) // n_tasks)

    us, ss, vs = [], [], []
    for mat in mats:
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        us.append(u[:, :k])
        ss.append(s[:k])
        vs.append(vh[:k, :].transpose(0, 1))  # columns = right singular vectors

    u_cat = torch.cat(us, dim=1)  # (rows, n*k)
    v_cat = torch.cat(vs, dim=1)  # (cols, n*k)
    s_block = torch.block_diag(*[torch.diag(s) for s in ss])  # (n*k, n*k)

    u_orth = _orthogonalize(u_cat)
    v_orth = _orthogonalize(v_cat)
    return u_orth @ s_block @ v_orth.transpose(0, 1)


class TsvMerging(MergeMethod):
    name = "tsvm"

    def __init__(self):
        self._cache_key = None
        self._cache_val = None

    def _merged(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> StateDict:
        key = (id(base_sd), tuple(id(s) for s in task_sds), tuple(keys))
        if key != self._cache_key:
            taus = self.task_vectors(base_sd, task_sds, keys)
            n_tasks = len(taus)
            merged: StateDict = {}
            for k in keys:
                vecs = [tau[k] for tau in taus]
                if vecs[0].dim() < 2 or n_tasks < 2:
                    # Not a matrix (bias/LayerNorm/etc.), or nothing to
                    # orthogonalize against -- plain task-arithmetic sum.
                    merged[k] = sum(vecs)
                    continue
                shape = vecs[0].shape
                mats = [_to_matrix(v) for v in vecs]
                merged[k] = _tsv_merge_matrix(mats, n_tasks).reshape(shape)
            self._cache_val = merged
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
