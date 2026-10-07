"""TSV-Merge (TSV-M): Task Singular Vectors merging. Ported from
vision_exp/src/merging/tsv_m.py.

Gargiulo et al., "Task Singular Vectors: Reducing Task Interference in Model
Merging" (CVPR 2025), https://arxiv.org/abs/2412.00081 -- Algorithm 1.

Per weight *matrix* (any parameter with >= 2 dims; 1-D parameters such as
biases and RMSNorm weights are merged by plain summation, since SVD-based
"task interference" isn't a meaningful notion for a vector):

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

Two LLM-specific deviations from the vision port:

  * dtype. The state dicts here are bf16 (see MergeMethod.task_vectors), but
    ``torch.linalg.svd`` has no bf16 CPU kernel ("linalg_svd_cpu not
    implemented for 'BFloat16'"), and an SVD is numerically delicate anyway.
    Each matrix is upcast to fp32 for its decomposition and the result cast
    back to the task vectors' dtype, so only one layer is ever held in fp32 --
    never a full fp32 copy of the model.
  * streaming. The deltas are formed one key at a time instead of via
    ``task_vectors`` (which materializes n_tasks full model-sized copies).
    Peak extra memory is the merged output plus the n deltas of the single key
    being processed -- on a 4B model that is the difference between ~7.5GB per
    task and a few hundred MB total.
  * device. The SVDs run on the GPU when one is available. ``torch.linalg.svd``
    runs wherever its input lives and the state dicts are CPU-resident, so
    without this the whole merge was CPU-bound: measured on qwen3-4b's real
    shapes (4 tasks), 80.8 min on 8 CPU threads against 9.2 min on an
    RTX PRO 4500 -- ~9x, and the merge is a one-time cost that blocks the first
    lambda of every coefficient sweep. See notes/tsvm_svd_cpu_bottleneck.md.

    Each key is moved to the GPU, decomposed, and brought straight back, so
    peak VRAM is the single largest key rather than the sum. The vocab-sized
    embed_tokens / lm_head keys are the one outlier -- ~17.7GB each on qwen3-4b
    against under 1.4GB for every other matrix -- while being only ~5% of the
    work, so ``embedding_svd_device`` can pin just those two to the CPU and hold
    peak VRAM to ~1.4GB. A cusolver failure or an unexpected OOM falls back to
    the CPU for that key too.

Higher-rank tensors are flattened to 2-D (out, rest) for the SVD and reshaped
back afterwards. LLM decoders are all Linear, so in practice every >= 2-D
parameter is already a matrix; the reshape only matters for the token
embedding / lm_head, which are matrices too.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from .base import MergeMethod, StateDict

Tensor = torch.Tensor


def _to_matrix(t: Tensor) -> Tensor:
    """Flatten every dim but the first into columns."""
    return t.reshape(t.shape[0], -1)


def _orthogonalize(mat: Tensor) -> Tensor:
    """Nearest matrix with orthonormal columns (orthogonal Procrustes / polar
    factor), via the thin SVD: A = P diag(D) Q^T  ->  A_orth = P @ Q^T."""
    p, _, qh = torch.linalg.svd(mat, full_matrices=False)
    return p @ qh


# The vocab-sized matrices: the token embedding and (when it is not tied to it)
# the output head. They are the only keys whose SVD does not comfortably fit
# beside a resident model on a smaller card -- on qwen3-4b the 151936x2560 pair
# peaks at ~17.7GB of VRAM each, where every other matrix in the model stays
# under 1.4GB. Matched by name rather than by size so the choice does not shift
# with the vocab or the card.
_EMBEDDING_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def _is_embedding_key(key: str) -> bool:
    return key.endswith(_EMBEDDING_SUFFIXES)


def _svd_device(explicit: Optional[str]) -> str:
    if explicit is not None:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_recoverable_cuda_error(exc: BaseException) -> bool:
    """Errors worth retrying on the CPU rather than aborting the run: an OOM
    the budget estimate failed to predict, or a cusolver/cusolvers convergence
    failure on a delicate matrix."""
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in ("out of memory", "cusolver", "linalg", "convergence", "cublas")
    )


def _tsv_merge_matrix(mats: List[Tensor], n_tasks: int) -> Tensor:
    """TSV-M on one layer. ``mats`` are fp32 2-D deltas; returns fp32."""
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

    def __init__(
        self, svd_device: Optional[str] = None, embedding_svd_device: Optional[str] = None
    ) -> None:
        """``svd_device``: where to run the decompositions. None (the default)
        uses the GPU when one is visible, else the CPU; pass "cpu" to force the
        old all-CPU behaviour.

        ``embedding_svd_device``: overrides that for the vocab-sized
        embed_tokens / lm_head keys. None (the default) means they follow
        ``svd_device`` like every other key; naming a device overrides it for
        those two unconditionally, in either direction.

        The override exists because those two keys are the only ones that need
        serious VRAM -- ~17.7GB each on qwen3-4b, against under 1.4GB for every
        other matrix -- while being only ~5% of the merge's total work. Passing
        "cpu" therefore buys a merge that shares a 24GB card with a resident
        model, for about 4 extra minutes. It is left to the caller rather than
        applied automatically, so that ``svd_device`` alone is taken at its
        word; ``runs/full_comparison/qwen3-4b_tsvm.sh`` opts in.

        Both are plumbed from merge_eval.py's --svd-device / --embedding-svd-device."""
        self.svd_device = svd_device
        self.embedding_svd_device = embedding_svd_device
        self._cache_key = None
        self._cache_val: Optional[StateDict] = None

    def _merged(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> StateDict:
        """The merged direction, cached across coefficients.

        Keyed on the identity of the inputs, same convention as
        TaskArithmetic._summed / TiesMerging._merged: experiment.py loads
        base_sd/task_sds once per unit, so coeff_search's lambda sweep reuses a
        single set of SVDs (which dominate this method's cost).
        """
        key = (id(base_sd), tuple(id(s) for s in task_sds), tuple(keys))
        if key != self._cache_key:
            # Drop the old merge before building the new one so the two don't
            # coexist (a full model-sized copy each).
            self._cache_val = None
            self._cache_key = None
            n_tasks = len(task_sds)
            device = _svd_device(self.svd_device)
            merged: StateDict = {}
            for k in keys:
                base = base_sd[k]
                # One key's deltas at a time -- see the module docstring.
                vecs = [sd[k] - base for sd in task_sds]
                if vecs[0].dim() < 2 or n_tasks < 2:
                    # Not a matrix (bias/RMSNorm/etc.), or nothing to
                    # orthogonalize against -- plain task-arithmetic sum.
                    acc = vecs[0]
                    for v in vecs[1:]:
                        acc = acc + v
                    merged[k] = acc
                    continue
                shape = vecs[0].shape
                dtype = vecs[0].dtype
                mats = [_to_matrix(v).float() for v in vecs]
                del vecs
                merged[k] = self._merge_one(mats, n_tasks, device, shape, dtype, k)
                del mats
            self._cache_val = merged
            self._cache_key = key
        return self._cache_val

    def _merge_one(
        self, mats: List[Tensor], n_tasks: int, device: str, shape, dtype, key: str
    ) -> Tensor:
        """One key's TSV-M, on the device chosen for it.

        The result always comes back to the CPU in the task vectors' dtype, so
        the merged state dict stays CPU-resident like every other method's.
        """
        if self.embedding_svd_device is not None and _is_embedding_key(key):
            device = self.embedding_svd_device
        use_gpu = device.startswith("cuda")
        if use_gpu:
            try:
                gpu_mats = [m.to(device, non_blocking=True) for m in mats]
                out = _tsv_merge_matrix(gpu_mats, n_tasks)
                result = out.reshape(shape).to(dtype).cpu()
                # Release this key's VRAM before the next one is staged;
                # ~250 keys of leftovers would fragment the allocator.
                del gpu_mats, out
                torch.cuda.empty_cache()
                return result
            except RuntimeError as exc:
                # cusolver can fail to converge, and a card that looked roomy can
                # still lose to fragmentation. Neither is worth losing the run
                # over -- redo this one key on the CPU.
                if not _is_recoverable_cuda_error(exc):
                    raise
                logging.warning(
                    f"tsvm: GPU SVD failed for a {tuple(shape)} key ({type(exc).__name__}: "
                    f"{str(exc).strip().splitlines()[0][:120]}); falling back to CPU for it."
                )
                torch.cuda.empty_cache()
        return _tsv_merge_matrix(mats, n_tasks).reshape(shape).to(dtype)

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
