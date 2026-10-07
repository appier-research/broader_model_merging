"""Weight-space subspace geometry.

Builds an orthonormal basis for the subspace S spanned by a fixed set of
directions (typically the per-task task vectors from ``MergeMethod.basis``),
and provides the projection/decomposition primitives needed to check whether a
GD trajectory that starts inside S drifts away from it.

All vectors are plain ``dict[str, Tensor]`` over a shared ``keys`` list -- the
convention used throughout ``vision_exp`` (see ``models.py`` and
``scripts/subspace_orthogonality_sweep.py``, which has an independent copy of
the same vector-algebra primitives for a different sweep).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

Vec = Dict[str, torch.Tensor]


def vdot(a: Vec, b: Vec, keys: List[str]) -> float:
    return float(sum(torch.sum(a[k] * b[k]) for k in keys))


def vnorm(a: Vec, keys: List[str]) -> float:
    return math.sqrt(max(vdot(a, a, keys), 0.0))


def vsub(a: Vec, b: Vec, keys: List[str]) -> Vec:
    return {k: a[k] - b[k] for k in keys}


def vscale(a: Vec, s: float, keys: List[str]) -> Vec:
    return {k: a[k] * s for k in keys}


def vnormalize(a: Vec, keys: List[str]) -> Tuple[Vec, float]:
    n = vnorm(a, keys)
    if n == 0:
        raise ValueError("cannot normalize a zero vector")
    return vscale(a, 1.0 / n, keys), n


def project_out(a: Vec, basis: List[Vec], keys: List[str]) -> Vec:
    """``a`` with its projection onto each (orthonormal) vector in ``basis`` removed."""
    out = dict(a)
    for b in basis:
        c = vdot(out, b, keys)
        out = {k: out[k] - c * b[k] for k in keys}
    return out


def gram_schmidt_basis(vectors: List[Vec], keys: List[str], eps: float = 1e-8) -> List[Vec]:
    """Orthonormalize ``vectors``, dropping any direction that is (numerically)
    already in the span of the earlier ones -- e.g. two near-duplicate task
    vectors collapse to a single basis direction instead of a division by a
    near-zero norm."""
    basis: List[Vec] = []
    for v in vectors:
        r = project_out(v, basis, keys)
        n = vnorm(r, keys)
        if n < eps:
            continue
        basis.append(vscale(r, 1.0 / n, keys))
    return basis


class SubspaceBasis:
    """Orthonormal basis for a fixed weight-space subspace S = span(directions),
    built once and reused for every step of a GD run.

    ``coords`` / ``residual_norm`` never materialize the projection or residual
    as a full D-dimensional vector -- both reduce to a handful of dot products
    against the (small) basis, using Pythagoras for the residual norm.
    """

    def __init__(self, directions: List[Vec], keys: List[str]):
        self.keys = keys
        self.raw_dim = len(directions)
        self.basis = gram_schmidt_basis(directions, keys)
        self.dim = len(self.basis)
        if self.dim < self.raw_dim:
            logging.warning(
                "SubspaceBasis: %d of %d directions were linearly dependent "
                "(dropped by Gram-Schmidt) -- S has rank %d, not %d",
                self.raw_dim - self.dim, self.raw_dim, self.dim, self.raw_dim,
            )

    def coords(self, v: Vec) -> List[float]:
        """Coordinates of ``v`` against the orthonormal basis (its projection's
        coefficients, one per basis direction)."""
        return [vdot(v, b, self.keys) for b in self.basis]

    def residual_norm(self, v: Vec, coords: Optional[List[float]] = None) -> float:
        """``||v - P_S(v)||``, via Pythagoras: ||v||^2 = ||P_S(v)||^2 + ||residual||^2."""
        alpha = coords if coords is not None else self.coords(v)
        v_norm_sq = vdot(v, v, self.keys)
        par_energy = sum(a * a for a in alpha)
        return math.sqrt(max(0.0, v_norm_sq - par_energy))

    def residual_vector(self, v: Vec) -> Vec:
        """``v - P_S(v)`` as an actual vector (not just its norm) -- unlike
        ``residual_norm`` this does materialize a full D-dimensional tensor per
        key, so only reach for it when you need the direction itself (e.g. to
        build the MT-orthogonal residual), not just its magnitude."""
        return project_out(v, self.basis, self.keys)

    def decompose_dot(self, g: Vec, dw: Vec) -> Tuple[float, float, float]:
        """Split ``g . dw`` into an in-S part and a part orthogonal to S.

        Exact, not an approximation: since g_par.dw_perp = 0 and g_perp.dw_par
        = 0 for any g, dw, ``total == in_plane + out_of_plane`` identically.
        Returns ``(in_plane, out_of_plane, total)``.
        """
        alpha_g = self.coords(g)
        alpha_dw = self.coords(dw)
        total = vdot(g, dw, self.keys)
        in_plane = sum(ag * ad for ag, ad in zip(alpha_g, alpha_dw))
        return in_plane, total - in_plane, total


def combine(base: Vec, directions: List[Vec], coeffs: List[float], keys: List[str]) -> Vec:
    """``base + sum_i coeffs[i] * directions[i]`` -- the general per-direction
    version of ``models.apply_delta``/``merged_delta(coeff=...)``, which only
    support one scalar coefficient shared across every direction. Used to
    place a point anywhere in (an affine subspace around) ``base``, including
    off the diagonal that a single scalar coeff is confined to (e.g. a random
    point, or a ``subspace_gd`` result with a different coefficient per task).
    """
    out = {k: base[k].clone() for k in keys}
    for c, d in zip(coeffs, directions):
        if c == 0:
            continue
        out = {k: out[k] + c * d[k] for k in keys}
    return out


def random_directions_like(reference: List[Vec], keys: List[str], seed: int = 0) -> List[Vec]:
    """Same count/shapes as ``reference``, iid Gaussian entries -- a same-rank
    null-hypothesis subspace.

    In a D-dimensional weight space with D >> k, ANY fixed rank-k subspace
    captures only a ~sqrt(k/D) fraction of a generic vector's norm, so a raw
    ``g_perp_ratio`` near 1 is not, by itself, evidence that the task-vector
    subspace is special -- it is what you'd see for a random gradient too.
    Compare the task-vector ``SubspaceBasis`` against a ``SubspaceBasis`` built
    from this null to tell a real (if small) alignment from that baseline.

    Generated on CPU (dtype/shape matched, then moved to each reference
    tensor's device) so the null is reproducible across CPU/GPU runs for a
    given ``seed``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for ref in reference:
        out.append({
            k: torch.randn(ref[k].shape, generator=gen).to(ref[k].device, ref[k].dtype)
            for k in keys
        })
    return out


def record_step(
    basis: SubspaceBasis,
    g: Vec,
    dw: Vec,
    w_now: Vec,
    base: Vec,
    *,
    step: int,
    epoch: int,
    loss: float,
    null_basis: Optional["SubspaceBasis"] = None,
    extra: Optional[dict] = None,
) -> dict:
    """One in-memory trajectory point: loss-attribution + subspace-escape
    quantities for a single optimizer step.

    Cheap to keep every one of: a dozen floats plus one length-``basis.dim``
    list. Meant to be appended to a plain Python list held by the caller and
    consumed in the same process -- nothing here touches disk.

    ``g`` is the step's gradient (dict over ``keys``), ``dw`` is the actual
    parameter update applied by the optimizer (``w_after - w_before``, so this
    is correct for adaptive optimizers too, not just vanilla SGD), ``w_now`` is
    the parameter state *after* the step, and ``base`` is the anchor the
    subspace S is affine around (usually the pretrained backbone).

    ``null_basis``, when given a same-rank ``SubspaceBasis`` built from random
    directions (see ``random_directions_like``), adds ``g_perp_ratio_random``
    and ``residual_norm_random`` alongside the real ones -- in a D-dimensional
    weight space with D >> dim(S), a raw ``g_perp_ratio`` near 1 is not itself
    evidence of anything (any fixed low-rank subspace looks nearly orthogonal
    to a generic vector once D is large); this is the baseline to read it
    against.

    ``extra`` is merged into the returned dict -- used for anything this
    function has no way to compute itself, e.g. held-out valid/test metrics
    that need an eval pass over a dataloader.

    Note the third figure axis (the escape direction) is NOT recorded here:
    it is defined by the run's own start->end displacement and so is unknown
    until the run finishes. The caller snapshots each step's S-orthogonal
    residual and projects afterwards.
    """
    keys = basis.keys
    v = vsub(w_now, base, keys)
    alpha_v = basis.coords(v)
    residual_norm = basis.residual_norm(v, alpha_v)

    g_norm = vnorm(g, keys)
    alpha_g = basis.coords(g)
    g_par_norm = math.sqrt(sum(a * a for a in alpha_g))
    g_perp_norm = math.sqrt(max(0.0, g_norm ** 2 - g_par_norm ** 2))

    alpha_dw = basis.coords(dw)
    total_dot = vdot(g, dw, keys)
    local_in_plane = sum(ag * ad for ag, ad in zip(alpha_g, alpha_dw))
    local_out_plane = total_dot - local_in_plane

    out = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "g_norm": g_norm,
        "g_par_norm": g_par_norm,
        "g_perp_norm": g_perp_norm,
        "g_perp_ratio": (g_perp_norm / g_norm) if g_norm > 0 else 0.0,
        "local_total": total_dot,
        "local_in_plane": local_in_plane,
        "local_out_plane": local_out_plane,
        "lambda_hat": alpha_v,
        "residual_norm": residual_norm,
    }
    if null_basis is not None:
        alpha_g_null = null_basis.coords(g)
        g_par_norm_null = math.sqrt(sum(a * a for a in alpha_g_null))
        g_perp_norm_null = math.sqrt(max(0.0, g_norm ** 2 - g_par_norm_null ** 2))
        out["g_perp_ratio_random"] = (g_perp_norm_null / g_norm) if g_norm > 0 else 0.0
        out["residual_norm_random"] = null_basis.residual_norm(v)
    if extra:
        out.update(extra)
    return out
