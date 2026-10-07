"""Base interface for model-merging methods.

A merging method turns a set of single-task fine-tuned checkpoints (expressed as
task vectors tau_i = ft_i - base) into:

  * ``basis``         -- a list of directions in weight space. Subspace GD
                         optimizes one coefficient per direction. TA returns the
                         n per-task task vectors; TIES returns a single merged
                         direction (so subspace GD optimizes a single scalar).
  * ``merged_delta``  -- the weight-space delta (added to base) for a given
                         scalar coefficient, used to build the init point for
                         coefficient search and weight-space GD.
  * ``default_coeffs``-- the initial coefficient vector for subspace GD, given
                         an init name in {pretrained, merged, coeff_best}.

Every method operates over ``keys`` (the shared float parameter names) and works
with plain ``dict[str, Tensor]`` task vectors, keeping the interface framework
agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

Tensor = torch.Tensor
StateDict = Dict[str, Tensor]


class MergeMethod:
    """Abstract merging method. Subclasses implement the three hooks below."""

    name: str = "base"

    def task_vectors(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> List[StateDict]:
        """tau_i = ft_i - base for each task, over the shared float keys."""
        return [
            {k: sd[k].float() - base_sd[k].float() for k in keys}
            for sd in task_sds
        ]

    # --- hooks every method must provide ------------------------------------ #

    def basis(self, base_sd: StateDict, task_sds: List[StateDict], keys: List[str]) -> List[StateDict]:
        """Directions optimized (one coefficient each) by subspace GD."""
        raise NotImplementedError

    def merged_delta(
        self,
        base_sd: StateDict,
        task_sds: List[StateDict],
        keys: List[str],
        coeff: float = 1.0,
    ) -> StateDict:
        """Weight-space delta (added to base) at the given scalar coefficient."""
        raise NotImplementedError

    def default_coeffs(
        self,
        n_dirs: int,
        init: str,
        best_lambda: Optional[float] = None,
        n_tasks: Optional[int] = None,
    ) -> Tensor:
        """Initial coefficient vector (length n_dirs) for subspace GD.

        ``n_tasks`` is the number of tasks being merged (used only by ``avg`` to
        scale by 1/N). For TA it equals n_dirs; for TIES it doesn't (n_dirs=1).
        Falls back to n_dirs when not provided.
        """
        if init == "pretrained":
            return torch.zeros(n_dirs)
        if init == "avg":
            n = n_tasks if n_tasks is not None else n_dirs
            return torch.full((n_dirs,), 1.0 / n)
        if init == "merged":
            return torch.ones(n_dirs)
        if init == "coeff_best":
            if best_lambda is None:
                raise ValueError("coeff_best init requires best_lambda")
            return torch.full((n_dirs,), float(best_lambda))
        raise ValueError(f"Unknown init '{init}' (expected pretrained, avg, merged, coeff_best)")
