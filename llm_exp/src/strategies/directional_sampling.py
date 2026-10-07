"""Directional sampling around a merge init.

Samples

    W(α, β) = W* + sum_i (α_i - λ0) dir_i + β (-g_⊥)

where W* is the merge at a chosen init (pretrained / avg / merged / coeff_best,
or the per-direction subspace_best / bo_best vectors),
{dir_i} is the method basis (TA: per-task vectors; TIES/TSV-M: one merged
direction), and g_⊥ is the valid-set loss gradient at W* (one joint gradient
over every task's SFT pool) with the basis component removed.
α_i ~ Unif[λ0 ± alpha], β ~ Unif[0, beta] independently.

Residency: W*, the basis and the gradient direction are held for the whole
run, on the GPU when 1 + n_dirs + n_grads copies fit next to the model and
generate()'s KV cache, otherwise on the CPU (``ds_basis_device``; "auto"
decides by memory), where each draw is combined and copied into the model key
by key.

Port of vision_exp/src/strategies/directional_sampling.py, with two changes
forced by the LLM setting:

* **Per-sample metric.** vision scores every draw with the same cheap
  classification-head accuracy it uses everywhere. Here the analogue would be
  ``eval_all_tasks`` -- a full autoregressive generate() over every task's
  valid pool per draw, minutes each, so ~128 draws is hours before the top-5
  test evals even start. The default per-draw metric is therefore the
  teacher-forced valid NLL (``ctx.ds_select_metric="loss"``, one forward pass
  per draw, the same ``eval_all_tasks_loss`` the other strategies already
  report), with lower = better. Set ``ds_select_metric="score"`` for the
  literal vision protocol (generate + score every draw) when the budget allows
  it. Either way the top-5 draws get the full generate+score treatment on
  valid *and* test, so the reported numbers are always comparable to every
  other strategy.
* **Test on every draw** (``ctx.ds_eval_test=True``). Picking the top-N by
  valid and reporting their test score conflates two things: whether the
  sampled region holds good points at all, and whether valid can find them
  (valid/test mismatch). With ``ds_eval_test`` each draw also gets a test
  generate()+score, so the result carries the whole distribution of test
  scores over the box (``test_dist``: median / quantiles / ``density_test`` =
  fraction of draws beating the init point's *test* score) instead of a
  valid-selected handful. The top-5 rows are then read off the per-draw evals
  rather than re-evaluated (which also removes the second generate() whose
  sampling noise made those valid numbers disagree with the sampling-phase
  ones). Costs one test pass per draw -- ~10x the valid pass here (3254 vs
  361 examples) -- so pair it with fewer draws.
* **Gradient without functional_call.** vision substitutes W* into the model
  via ``torch.func.functional_call``; subspace_gd here documents why that path
  is avoided on a causal LM (gradient checkpointing can silently recompute
  against the model's own parameters). W* is loaded into a real model instead
  and dL/dW is read off ``.grad`` after an ordinary backward -- the same
  trick subspace_gd uses (there projected onto the basis once per step).

The init point W*'s score is the number every draw is measured against
(``density_valid`` = fraction of draws beating it on the selection metric,
``density_test`` likewise on test). By default it is taken as *recorded by the
init's source cell* (coeff_search for coeff_best, subspace_gd for
subspace_best, bo_search for bo_best) -- the same number the main results
report, so this cell cannot disagree with them by a re-generation's sampling
noise. W* is re-evaluated in the run (valid score, valid loss, test score)
only when no such record exists (pretrained / avg / merged, or the "loss"
metric) or when ``ds_reeval_center`` asks for it; ``center_source`` says
which.

GPU residency is the minimum the sampler needs. With the joint gradient on the
GPU that is the model being evaluated, W*, the n_dirs basis directions and
(beta > 0) g_⊥ -- n_dirs + 3 full-model copies, constant through every phase;
when those do not fit (larger models) the resident copies
move to the CPU and the GPU holds the model alone. Everything else is built one
key at a time: W* is one dict (base and the init's delta are only ever used
through their sum), each draw is written straight into the model's parameters
instead of through an intermediate state dict, each gradient's projection
solves an n_dirs x n_dirs Gram system instead of keeping Gram-Schmidt's
orthonormalized copies resident, and the gradient is taken over from ``.grad``
rather than cloned. On a 1.7B model each copy is ~3.2 GiB, so the earlier
9-13 copy peaks (this file's first version) did not fit a 32 GB card.
"""

from __future__ import annotations

import logging
import statistics
from typing import Dict, List, Optional

import torch

from .. import models
from ..data import get_sft_dataloader
from ..models import avg_loss, avg_score, load_causal_lm
from ..tasks import get_task
from ._common import (
    clear_cache,
    eval_all_tasks,
    eval_all_tasks_loss,
    eval_unseen,
    unseen_fields,
    free_model,
    pack_result,
)
from .context import StrategyContext

SELECT_METRICS = ("loss", "score")
BASIS_DEVICES = ("auto", "cuda", "cpu")


# --------------------------------------------------------------------------- #
# Selection metric
# --------------------------------------------------------------------------- #

def _better(a: float, b: float, metric: str) -> bool:
    """Is ``a`` a better selection value than ``b``? Loss is minimized, score
    is maximized."""
    return a < b if metric == "loss" else a > b


def _sort_key(metric: str):
    return (lambda v: v) if metric == "loss" else (lambda v: -v)


# --------------------------------------------------------------------------- #
# Weight-space plumbing (streamed per key -- never a flat full-model vector)
# --------------------------------------------------------------------------- #

def _load_weights(
    model,
    wstar: Dict[str, torch.Tensor],
    dirs: List[Dict[str, torch.Tensor]],
    grads: List[Dict[str, torch.Tensor]],
    keys: List[str],
    lams: List[float],
    alphas: List[float],
    betas: List[float],
    device: str,
    dtype: torch.dtype,
) -> None:
    """Write W* + sum_i (α_i - λ0_i) dir_i - sum_t β_t g_t straight into
    ``model``'s parameters, one key at a time.

    No intermediate full-model state dict: ``load_state_dict`` would need the
    whole W materialized (one more full-model copy at peak) only to copy it
    into the parameters anyway. ``model.state_dict()`` hands back detached
    views sharing the parameters' storage, so ``copy_`` into them is the same
    write ``load_state_dict`` does. ``lams`` is the per-direction center: a
    scalar init (pretrained/avg/merged/coeff_best) passes the same value
    n_dirs times, a vector init (subspace_best / bo_best) one center per
    direction. ``grads`` is the list of (projected) gradient directions --
    the joint gradient, or empty when beta=0 -- with one β each.

    The sources (W*, dirs, grads) may live on the GPU (small models: the
    combination is a few bf16 axpys on the device) or on the CPU (when
    n_dirs + n_grads + 2 model copies would not fit next to generate()'s KV
    cache): then each key is combined on the CPU in fp32 and only the finished
    key is copied to the device, so the GPU holds exactly one model copy
    between draws. Per draw that is (n_dirs + n_grads) elementwise passes over
    the model on the CPU plus one host-to-device copy of the model -- seconds,
    against the minutes each draw's generate() costs.
    """
    offsets = [a - l for a, l in zip(alphas, lams)]
    on_cpu = wstar[keys[0]].device.type == "cpu"
    state = model.state_dict()
    with torch.no_grad():
        for k in keys:
            w = wstar[k].float() if on_cpu else wstar[k]
            for off, d in zip(offsets, dirs):
                if off != 0.0:
                    w = w + off * (d[k].float() if on_cpu else d[k])
            for b, g in zip(betas, grads):
                if b != 0.0:
                    w = w - b * (g[k].float() if on_cpu else g[k])
            if on_cpu:
                w = w.to(dtype)
            state[k].copy_(w, non_blocking=on_cpu)
    if on_cpu and device == "cuda":
        torch.cuda.synchronize()
    del state


def _resolve_basis_device(ctx: StrategyContext, n_copies: int, dtype: torch.dtype) -> str:
    """Where W*, the basis and the gradient directions live between draws.
    ``n_copies`` counts them (n_dirs + n_grads + 1 for W*); the model itself is
    one more. "auto" keeps them on the GPU only if all of that stays under
    ~60% of the card, leaving the rest for the gradient pass's activations and
    generate()'s KV cache -- same rule as bo_search._resolve_basis_device."""
    choice = str(ctx.ds_basis_device).lower()
    if choice not in BASIS_DEVICES:
        raise ValueError(f"ds_basis_device must be one of {list(BASIS_DEVICES)} (got {choice!r})")
    if ctx.device != "cuda" or not torch.cuda.is_available():
        return "cpu"
    if choice != "auto":
        return choice
    n_params = sum(ctx.base_sd[k].numel() for k in ctx.keys)
    copy_bytes = n_params * torch.tensor([], dtype=dtype).element_size()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    need = (1 + n_copies) * copy_bytes
    dev = "cuda" if need <= 0.6 * total else "cpu"
    logging.info(
        f"  basis device: {dev} (auto -- model + {n_copies} resident copies = {need / 2**30:.1f} GB "
        f"vs {total / 2**30:.1f} GB card)"
    )
    return dev


def _project_orthogonal(
    g: Dict[str, torch.Tensor], basis: List[Dict[str, torch.Tensor]], keys: List[str]
) -> Dict[str, torch.Tensor]:
    """g minus its least-squares projection onto span(basis), in place on ``g``.

    Only inner products: the Gram matrix G_ij = <dir_i, dir_j> and b_i =
    <g, dir_i> (both accumulated in fp64, see ``_dot``), then c = argmin
    ||G c - b|| via a minimum-norm least-squares solve so a dependent or
    all-zero direction simply drops out (what Gram-Schmidt's norm check did),
    and finally g -= sum_i c_i dir_i. This is the same projection as
    Gram-Schmidt against the orthonormalized basis, but keeps no extra
    full-model copies resident -- the orthonormalized directions were n_dirs
    of them, on top of the basis itself.
    """
    n = len(basis)
    if n == 0:
        return g
    gram = torch.zeros(n, n, dtype=torch.float64)
    rhs = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        rhs[i] = _dot(g, basis[i], keys)
        for j in range(i, n):
            gram[i, j] = gram[j, i] = _dot(basis[i], basis[j], keys)
    coeffs = torch.linalg.lstsq(gram, rhs.unsqueeze(1), driver="gelsd").solution.squeeze(1).tolist()
    with torch.no_grad():
        for k in keys:
            for c, d in zip(coeffs, basis):
                if c != 0.0:
                    g[k] -= c * d[k]
    return g


# Elements per fp64 chunk in _dot/_norm: 2^26 -> 512 MiB per temporary. A
# whole-key .double() of Qwen3's 151936 x 2048 embedding is 2.3 GiB, and the
# product needs two of those plus the result -- ~7 GiB of transients on top of
# the resident copies, which is exactly what OOM'd the 1.7B run on 32 GB.
_DOT_CHUNK = 1 << 26


def _dot(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], keys: List[str]) -> float:
    """<a, b> over the whole state dict, accumulated in fp64.

    The per-key products are summed at float64 rather than the tensors' own
    dtype: these run over bf16 weights, whose ~3 decimal digits of mantissa
    would otherwise turn a sum over billions of terms into noise, and the
    Gram system above is only as well-posed as this dot product is accurate.
    Each key is streamed through in ``_DOT_CHUNK``-element slices so the fp64
    temporaries stay bounded regardless of the largest key's size.
    """
    acc = 0.0
    for k in keys:
        x, y = a[k].reshape(-1), b[k].reshape(-1)
        for s in range(0, x.numel(), _DOT_CHUNK):
            acc += float(torch.dot(x[s:s + _DOT_CHUNK].double(), y[s:s + _DOT_CHUNK].double()))
    return acc


def _norm(a: Dict[str, torch.Tensor], keys: List[str]) -> float:
    return _dot(a, a, keys) ** 0.5


# --------------------------------------------------------------------------- #
# Gradient at W*
# --------------------------------------------------------------------------- #

def _valid_loss_grad(
    model, ctx: StrategyContext, keys: List[str], out_device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """One full pass of dL/dW over every task's SFT pool at the weights
    already loaded into ``model``, averaged over each task's batches and over
    the tasks used. The gradient is returned on ``out_device`` (default: where
    the model is).

    Mirrors weight_gd/subspace_gd's loop shape: per-task backward with
    gradient checkpointing on, so only one batch's activations are resident at
    a time. No optimizer, no step -- the gradient is read straight off
    ``.grad``. ``model.named_parameters()`` dedupes tied weights (Qwen3-0.6B
    ties lm_head to embed_tokens) to one name whose ``.grad`` already sums both
    usages, so this doesn't double count -- same reasoning as
    subspace_gd's ``_project_from_grads``.
    """
    dataloaders = {}
    for t in ctx.tasks:
        pool = ctx.sft_pools.get(t)
        if pool:
            dataloaders[t] = get_sft_dataloader(
                pool, ctx.tokenizer, batch_size=ctx.loss_batch_size, max_length=ctx.max_seq_length,
                shuffle=False, enable_thinking=getattr(get_task(t), "ENABLE_THINKING", True),
            )
    if not dataloaders:
        raise ValueError("No task in this unit has any SFT examples -- directional_sampling needs a gradient.")

    was_checkpointing = model.is_gradient_checkpointing
    model.gradient_checkpointing_enable()
    model.train()
    model.zero_grad(set_to_none=True)
    n_tasks = len(dataloaders)
    for task, dl in dataloaders.items():
        n_batches = len(dl)
        for batch in dl:
            batch = {k: v.to(ctx.device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / n_batches / n_tasks).backward()

    # Map every merged key to its gradient. named_parameters() dedupes tied
    # weights to a single name (Qwen3 ties lm_head.weight to
    # model.embed_tokens.weight) whose .grad already sums both usages'
    # contributions -- but the *other* name is still a live state-dict key, so
    # a plain name lookup would find no gradient for it. Resolve through the
    # underlying tensor identity so both aliases pick up that one shared
    # gradient (and, being aliases of one weight, they legitimately move
    # together under a -g_perp step).
    # named_parameters() is the only view that sees .grad, but it drops the
    # duplicate name of a tied weight; state_dict() keeps every name but hands
    # back detached tensors. Join them on the underlying storage: build
    # {data_ptr -> grad} from the parameters, then look each merged key's
    # state-dict tensor up by the same pointer.
    # The gradient tensors are taken over, not cloned: zero_grad(set_to_none)
    # below drops the parameters' references and this dict keeps the only
    # ones, so the full-model-sized gradient exists once, not twice. The one
    # exception is a tied weight whose *both* names are merged keys -- the
    # projection updates g[k] in place per key, so the second alias gets its
    # own copy rather than being subtracted from twice.
    by_ptr = {p.data_ptr(): p.grad for p in model.parameters() if p.grad is not None}
    state = model.state_dict()
    grad, missing, seen = {}, [], set()
    for k in keys:
        ptr = state[k].data_ptr()
        g = by_ptr.get(ptr)
        if g is None:
            missing.append(k)
        else:
            gk = g.detach().clone() if ptr in seen else g.detach()
            grad[k] = gk if out_device is None else gk.to(out_device)
            seen.add(ptr)
    if missing:
        raise RuntimeError(
            f"directional_sampling: no gradient for {len(missing)} merged key(s), "
            f"e.g. {missing[:3]}"
        )
    del state, by_ptr
    model.zero_grad(set_to_none=True)
    if not was_checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True  # re-enable KV cache for the eval-time generate() calls
    model.eval()
    clear_cache(ctx.device)
    return grad


# --------------------------------------------------------------------------- #
# Init center
# --------------------------------------------------------------------------- #

SUBSPACE_BEST_INIT = "subspace_best"
BO_BEST_INIT = "bo_best"


def _center_coeffs(ctx: StrategyContext, init: str, n_dirs: int) -> List[float]:
    """The sampling box's center, as one coefficient per basis direction.

    The four scalar inits (pretrained / avg / merged / coeff_best) put every
    direction at the same value, so they broadcast one number across n_dirs.
    ``subspace_best`` instead centers on subspace_gd's trained coefficient
    vector -- one value per direction, which is the whole point: a scalar
    center cannot express "pull bank77 down while pushing ifeval up", and that
    per-task asymmetry is what subspace_gd actually learned. ``bo_best`` is
    the same kind of vector center, taken from bo_search's best trial instead
    -- BO maximizes the valid score directly, so on the units where it beats
    both coeff_search and subspace_gd this is the strongest known point to
    sample around.
    """
    if init == "pretrained":
        return [0.0] * n_dirs
    if init == "avg":
        return [1.0 / len(ctx.tasks)] * n_dirs
    if init == "merged":
        return [1.0] * n_dirs
    if init == "coeff_best":
        if ctx.best_lambda is None:
            raise ValueError(
                "directional_sampling init 'coeff_best' requires best_lambda "
                "(from coeff_search in this JSON or --coeff-source-dir)"
            )
        return [float(ctx.best_lambda)] * n_dirs
    if init == SUBSPACE_BEST_INIT:
        c = ctx.subspace_best_coeffs
        if c is None:
            raise ValueError(
                "directional_sampling init 'subspace_best' requires subspace_gd's trained "
                "coefficients; run subspace_gd into this unit first, or point "
                "--coeff-source-dir at a results root whose unit has them"
            )
        if len(c) != n_dirs:
            raise ValueError(
                f"subspace_best coefficients have length {len(c)} but the method basis has "
                f"{n_dirs} direction(s) -- these must come from the same (method, tasks) unit"
            )
        return [float(x) for x in c]
    if init == BO_BEST_INIT:
        c = ctx.bo_best_coeffs
        if c is None:
            raise ValueError(
                "directional_sampling init 'bo_best' requires bo_search's best-trial "
                "coefficients; run bo_search into this unit first, or point "
                "--coeff-source-dir at a results root whose unit has them"
            )
        if len(c) != n_dirs:
            raise ValueError(
                f"bo_best coefficients have length {len(c)} but the method basis has "
                f"{n_dirs} direction(s) -- these must come from the same (method, tasks) unit"
            )
        return [float(x) for x in c]
    raise ValueError(
        f"Unknown ds init '{init}' (expected pretrained, avg, merged, coeff_best, "
        f"{SUBSPACE_BEST_INIT}, {BO_BEST_INIT})"
    )


def _wstar_at(
    ctx: StrategyContext, keys: List[str], lams: List[float],
    dirs: List[Dict[str, torch.Tensor]], dtype, device,
) -> Dict[str, torch.Tensor]:
    """W* = base + delta(init), materialized as ONE resident dict.

    base and the init's delta are only ever used through their sum, so keeping
    them apart would cost a full-model copy for nothing. A uniform center goes
    through ``method.merged_delta(coeff=lam)`` so every method keeps its own
    semantics; a per-direction center (subspace_best) has no scalar form, so
    it is built from the basis as sum_i c_i * dir_i -- exactly the
    reparametrization subspace_gd optimizes. base and delta are each rounded
    to the run dtype and then added, in that dtype, on the device -- the same
    arithmetic the first version of this file did with its separate base_dev
    and delta_star, so W* is bit-identical to what the existing 0.6b units
    were sampled around.
    """
    if len(set(lams)) == 1:
        delta = ctx.method.merged_delta(ctx.base_sd, ctx.task_sds, keys, coeff=lams[0])
    else:
        delta = {}
        for k in keys:
            acc = None
            for c, d in zip(lams, dirs):
                term = d[k].to(dtype).to(device) * c
                acc = term if acc is None else acc + term
            delta[k] = acc
    out = {}
    for k in keys:
        out[k] = ctx.base_sd[k].to(dtype).to(device) + delta[k].to(dtype).to(device)
        delta[k] = None  # free the delta as we go
    return out


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #

def run_one(ctx: StrategyContext, init: str) -> dict:
    if ctx.ds_samples < 1:
        raise ValueError("ds_samples must be >= 1")
    metric = ctx.ds_select_metric
    if metric not in SELECT_METRICS:
        raise ValueError(f"Unknown ds_select_metric '{metric}' (expected one of {list(SELECT_METRICS)})")

    method, keys, tasks, device = ctx.method, ctx.keys, ctx.tasks, ctx.device
    dtype = models.resolve_dtype(ctx.dtype)
    alpha_r, beta_r = float(ctx.ds_alpha), float(ctx.ds_beta)
    n_samples = int(ctx.ds_samples)
    eval_test = bool(ctx.ds_eval_test)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(ctx.ds_seed))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dirs = method.basis(ctx.base_sd, ctx.task_sds, keys)
    n_dirs = len(dirs)
    lams = _center_coeffs(ctx, init, n_dirs)
    # One joint gradient direction (the mean gradient over every task's SFT
    # pool), or none when beta=0.
    n_grads = 1 if beta_r != 0.0 else 0
    # Residency, in the run dtype: W* + n_dirs directions + n_grads gradient
    # directions, held unchanged through the gradient, sampling and top-5
    # phases, plus the model itself. On the GPU when that fits (n_dirs + 3
    # copies for the joint gradient -- see module docstring), otherwise on the
    # CPU with each draw combined there and copied in (see _load_weights).
    basis_device = _resolve_basis_device(ctx, 1 + n_dirs + n_grads, dtype)
    dirs_dev = [{k: d[k].to(dtype).to(basis_device) for k in keys} for d in dirs]
    wstar_dev = _wstar_at(ctx, keys, lams, dirs, dtype, basis_device)
    del dirs

    logging.info(f"directional_sampling(init={init}): building fresh model")
    model = load_causal_lm(ctx.base_model, device=device, dtype=dtype)

    logging.info(f"\n=== Directional sampling (init={init}) ===")
    lam_str = (f"{lams[0]:.4f}" if len(set(lams)) == 1
               else "[" + ", ".join(f"{l:.4f}" for l in lams) + "]")
    logging.info(
        f"  λ0={lam_str}  alpha=±{alpha_r}  beta∈[0, {beta_r}]  "
        f"n_samples={n_samples}  n_dirs={n_dirs}  seed={ctx.ds_seed}  select={metric}  "
        f"eval_test={'every draw' if eval_test else 'top-5 only'}  "
        f"n_grads={n_grads}  basis_device={basis_device}"
    )

    _load_weights(model, wstar_dev, dirs_dev, [], keys, lams, list(lams), [], device, dtype)

    # The joint gradient (mean over every task's SFT pool), projected
    # orthogonal to the basis.
    g_perps: List[Dict[str, torch.Tensor]] = []
    grad_norms: List[float] = []
    grad_perp_norms: List[float] = []
    if beta_r != 0.0:
        logging.info(f"  computing ∇L at {init} (SFT pools), then g_⊥")
        g = _valid_loss_grad(model, ctx, keys, out_device=basis_device)
        grad_norms.append(_norm(g, keys))
        g_perps.append(_project_orthogonal(g, dirs_dev, keys))
        grad_perp_norms.append(_norm(g_perps[-1], keys))
        logging.info(f"    ||g||={grad_norms[-1]:.6g}  ||g_⊥||={grad_perp_norms[-1]:.6g}")
    else:
        # β ≡ 0 makes the whole -g_⊥ axis a no-op; skip the backward pass (and
        # the full-model-sized gradient copy it leaves resident) entirely.
        logging.info("  beta=0 -> skipping the ∇L pass (samples stay in the basis subspace)")

    # The init point W*: the baseline every draw is measured against, and the
    # test score the top-k / test_dist numbers are read next to. By default it
    # is taken *as recorded by its source cell* (bo_search for bo_best,
    # subspace_gd for subspace_best, coeff_search for coeff_best) -- the same
    # number the main results report for that point, so nothing in this cell
    # can disagree with them by a re-generation's sampling noise. It is
    # re-evaluated here (valid score, valid loss, test score) only when that
    # is impossible: a scalar init with no source cell (pretrained / avg /
    # merged), a source without both scores, the "loss" selection metric (the
    # sources record no loss), or ds_reeval_center=True.
    src_name, src_valid, src_test = _init_source_scores(ctx, init)
    use_recorded = (not ctx.ds_reeval_center and metric == "score"
                    and src_valid is not None and src_test is not None)
    if use_recorded:
        center = {
            "valid_avg_score": src_valid, "valid_per_task": None,
            "valid_avg_loss": None, "valid_per_task_loss": None,
            "test_avg_score": src_test, "test_per_task": None,
        }
        center_source = "recorded"
        logging.info(
            f"  init point ({init}) taken as recorded by {src_name} (not re-evaluated): "
            f"valid_avg_score={_fmt(src_valid)}  test_avg_score={_fmt(src_test)}"
        )
    else:
        why = ("ds_reeval_center" if ctx.ds_reeval_center else
               f"select metric '{metric}'" if metric != "score" else
               "no source cell" if src_name is None else "source lacks a score")
        if src_name is not None:
            logging.info(
                f"  init point ({init}) as recorded by {src_name}: "
                f"valid_avg_score={_fmt(src_valid)}  test_avg_score={_fmt(src_test)}"
            )
        center = _eval_init_point(ctx, model)
        center_source = "re-evaluated"
        logging.info(
            f"  init point ({init}) re-evaluated here ({why}): valid_avg_score={_fmt(center['valid_avg_score'])}  "
            f"valid_avg_loss={_fmt(center['valid_avg_loss'])}  test_avg_score={_fmt(center['test_avg_score'])}"
        )
    center_valid_score, center_valid_loss = center["valid_avg_score"], center["valid_avg_loss"]
    center_value = center_valid_loss if metric == "loss" else center_valid_score
    logging.info(f"  density threshold = init point's valid_avg_{metric} = {center_value:.4f} ({center_source})")

    samples = []
    for i in range(n_samples):
        alphas = [float(torch.empty(1).uniform_(l - alpha_r, l + alpha_r, generator=rng))
                  for l in lams]
        betas = [float(torch.empty(1).uniform_(0.0, beta_r, generator=rng)) for _ in g_perps]
        _load_weights(model, wstar_dev, dirs_dev, g_perps, keys, lams, alphas, betas, device, dtype)
        row = {"idx": i, "alpha": alphas, "beta": betas[0] if betas else 0.0}
        if metric == "loss":
            clear_cache(device)
            vloss = eval_all_tasks_loss(
                model, ctx.tokenizer, ctx.sft_pools, tasks, device, ctx.loss_batch_size, ctx.max_seq_length,
            )
            row["valid_avg_loss"] = avg_loss(vloss)
            row["valid_per_task_loss"] = vloss
            value = row["valid_avg_loss"]
        else:
            clear_cache(device)
            vres = eval_all_tasks(
                model, ctx.tokenizer, ctx.valid_examples, tasks, device,
                ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
            )
            row["valid_avg_score"] = avg_score(vres)
            row["valid_per_task"] = vres
            value = row["valid_avg_score"]
        if value is None:
            raise RuntimeError(
                f"directional_sampling: sample {i} has no valid_avg_{metric} "
                f"(no task in this unit produced one)"
            )
        row["select_value"] = value
        msg = f"  sample {i + 1}/{n_samples}  valid_avg_{metric}={value:.4f}"
        if eval_test:
            clear_cache(device)
            tres = eval_all_tasks(
                model, ctx.tokenizer, ctx.test_examples, tasks, device,
                ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
            )
            row["test_avg_score"] = avg_score(tres)
            row["test_per_task"] = tres
            msg += f"  test_avg_score={_fmt(row['test_avg_score'])}"
        samples.append(row)
        logging.info(msg)
    clear_cache(device)

    n_beat_center = sum(1 for s in samples if _better(s["select_value"], center_value, metric))
    density_valid = n_beat_center / n_samples
    key = _sort_key(metric)
    top5_src = sorted(samples, key=lambda s: key(s["select_value"]))[:5]
    logging.info(
        f"  valid-best sample={top5_src[0]['idx']}  valid_avg_{metric}={top5_src[0]['select_value']:.4f}  "
        f"density_valid={density_valid:.4f} ({n_beat_center}/{n_samples} beat the init point)"
    )

    test_dist = None
    if eval_test:
        # The headline of this protocol: the distribution of test scores over
        # every draw, read against the init point's own test score. No
        # selection on valid is involved, so valid/test mismatch cannot hide
        # (or fake) anything here.
        test_scores = [s["test_avg_score"] for s in samples if s["test_avg_score"] is not None]
        center_test = center["test_avg_score"]
        n_beat_center_test = (sum(1 for t in test_scores if t > center_test)
                              if center_test is not None else None)
        test_dist = {
            "n": len(test_scores),
            "mean": statistics.mean(test_scores),
            "std": statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0,
            "median": statistics.median(test_scores),
            "min": min(test_scores),
            "max": max(test_scores),
            "q25": _quantile(test_scores, 0.25),
            "q75": _quantile(test_scores, 0.75),
            "n_beat_center": n_beat_center_test,
            "density_test": (n_beat_center_test / len(test_scores)
                             if n_beat_center_test is not None else None),
        }
        logging.info(
            f"  test over {test_dist['n']} draws: median={test_dist['median']:.4f}  "
            f"mean={test_dist['mean']:.4f} ± {test_dist['std']:.4f}  "
            f"[{test_dist['min']:.4f}, {test_dist['max']:.4f}]  "
            f"density_test={_fmt(test_dist['density_test'])} "
            f"({test_dist['n_beat_center']}/{test_dist['n']} beat the init point's test "
            f"{_fmt(center_test)})"
        )

    top5, valid_best, test_best, valid_best_loss = [], None, None, None
    if eval_test:
        # Every draw already has its test score, so the top-5 by valid need no
        # second pass: their rows are read straight off the samples. Only the
        # valid-best draw's valid *score* may be missing (loss metric), and
        # that is one generate() over the valid pool.
        logging.info(f"directional_sampling(init={init}): top-{len(top5_src)} by valid (from the per-draw evals)")
        for rank, s in enumerate(top5_src, 1):
            top5.append({
                "rank": rank, "idx": s["idx"], "alpha": s["alpha"], "beta": s["beta"],
                "select_value": s["select_value"],
                "valid_avg_score": s.get("valid_avg_score"), "valid_avg_loss": s.get("valid_avg_loss"),
                "test_avg_score": s["test_avg_score"], "test_per_task": s["test_per_task"],
            })
            logging.info(
                f"  top{rank} sample={s['idx']}  valid_avg_{metric}={s['select_value']:.4f}  "
                f"test_avg_score={_fmt(s['test_avg_score'])}"
            )
        best = top5_src[0]
        test_best = best["test_per_task"]
        valid_best = best.get("valid_per_task")
        if valid_best is None:
            _load_weights(model, wstar_dev, dirs_dev, g_perps, keys, lams, best["alpha"],
                          _betas_list(best["beta"], len(g_perps)), device, dtype)
            clear_cache(device)
            valid_best = eval_all_tasks(
                model, ctx.tokenizer, ctx.valid_examples, tasks, device,
                ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
            )
            top5[0]["valid_avg_score"] = avg_score(valid_best)
        valid_best_loss = best.get("valid_per_task_loss")
    else:
        # W* / basis / gradients stay resident through the top-5 evals and each
        # draw is rebuilt right before its eval: that is exactly the footprint
        # the sampling loop already ran generate() with, whereas materializing
        # the five weight sets up front would add five full-model copies at
        # once.
        logging.info(f"directional_sampling(init={init}): evaluating top-{len(top5_src)} on valid + test")
        for rank, s in enumerate(top5_src, 1):
            _load_weights(model, wstar_dev, dirs_dev, g_perps, keys, lams, s["alpha"],
                          _betas_list(s["beta"], len(g_perps)), device, dtype)
            clear_cache(device)
            valid_res = eval_all_tasks(
                model, ctx.tokenizer, ctx.valid_examples, tasks, device,
                ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
            )
            clear_cache(device)
            valid_loss_res = eval_all_tasks_loss(
                model, ctx.tokenizer, ctx.sft_pools, tasks, device, ctx.loss_batch_size, ctx.max_seq_length,
            )
            clear_cache(device)
            test_res = eval_all_tasks(
                model, ctx.tokenizer, ctx.test_examples, tasks, device,
                ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
            )
            clear_cache(device)
            top5.append({
                "rank": rank, "idx": s["idx"], "alpha": s["alpha"], "beta": s["beta"],
                "select_value": s["select_value"],
                "valid_avg_score": avg_score(valid_res), "valid_avg_loss": avg_loss(valid_loss_res),
                "test_avg_score": avg_score(test_res), "test_per_task": test_res,
            })
            logging.info(
                f"  top{rank} sample={s['idx']}  valid_avg_score={_fmt(avg_score(valid_res))}  "
                f"test_avg_score={_fmt(avg_score(test_res))}"
            )
            if rank == 1:
                valid_best, valid_best_loss, test_best = valid_res, valid_loss_res, test_res
    del wstar_dev, dirs_dev, g_perps
    clear_cache(device)

    top5_scores = [r["test_avg_score"] for r in top5 if r["test_avg_score"] is not None]
    top5_mean = statistics.mean(top5_scores) if top5_scores else None
    top5_std = statistics.stdev(top5_scores) if len(top5_scores) > 1 else 0.0
    logging.info(f"  top5 test_avg_score={_fmt(top5_mean)} ± {top5_std:.4f} (n={len(top5_scores)})")

    extra = {
        "init": init,
        "center_coeff": lams[0] if len(set(lams)) == 1 else None,
        "center_coeffs": lams,
        "select_metric": metric,
        "alpha_range": alpha_r,
        "beta_max": beta_r,
        "basis_device": basis_device,
        "n_samples": n_samples,
        "n_dirs": n_dirs,
        "ds_seed": ctx.ds_seed,
        "best_idx": top5[0]["idx"],
        "best_alpha": top5[0]["alpha"],
        "best_beta": top5[0]["beta"],
        # the init point W*: the baseline every draw and every top-k test
        # score should be read against
        # center_source: "recorded" (the source cell's numbers, the default)
        # or "re-evaluated" (a fresh valid/loss/test pass in this run)
        "center_source": center_source,
        "center_valid_avg_score": center_valid_score,
        "center_valid_avg_loss": center_valid_loss,
        "center_test_avg_score": center["test_avg_score"],
        "center_valid_per_task": center["valid_per_task"],
        "center_test_per_task": center["test_per_task"],
        "center_valid_per_task_loss": center["valid_per_task_loss"],
        # what the init's source cell recorded for this same point
        "init_source": src_name,
        "init_source_valid_avg_score": src_valid,
        "init_source_test_avg_score": src_test,
        # density_valid: fraction of draws beating the init point (its
        # re-evaluated valid metric above) on the selection metric.
        "density_valid": density_valid,
        "n_beat_center": n_beat_center,
        "grad_norm": grad_norms[0] if grad_norms else None,
        "grad_perp_norm": grad_perp_norms[0] if grad_perp_norms else None,
        "top5_test_avg_score": top5_mean,
        "top5_test_avg_score_std": top5_std,
        "top5": top5,
        # eval_test: every draw carries test_avg_score / test_per_task and
        # test_dist summarizes them (density_test = fraction beating the init
        # point's own test score). Otherwise test_dist is None and only the
        # top-5 have test scores.
        "eval_test": eval_test,
        "test_dist": test_dist,
        "samples": samples,
    }
    if init == "coeff_best":
        extra["best_lambda"] = lams[0]
    if init == SUBSPACE_BEST_INIT:
        extra["subspace_best_source"] = ctx.subspace_best_source
    if init == BO_BEST_INIT:
        extra["bo_best_source"] = ctx.bo_best_source
    result = pack_result(valid_best, test_best, valid_loss=valid_best_loss, **extra)
    if unseen_best is not None:
        result.update(unseen_fields(unseen_best))
    free_model(model, device)
    return result


def _eval_init_point(ctx: StrategyContext, model) -> dict:
    """Full evaluation of the weights currently loaded (W*): valid score, valid
    loss and test score -- the same three passes each top-k draw gets, so the
    init's numbers are directly comparable to theirs. Costs one extra test
    generate() per run, once."""
    clear_cache(ctx.device)
    vres = eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, ctx.tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)
    vloss = eval_all_tasks_loss(
        model, ctx.tokenizer, ctx.sft_pools, ctx.tasks, ctx.device,
        ctx.loss_batch_size, ctx.max_seq_length,
    )
    clear_cache(ctx.device)
    tres = eval_all_tasks(
        model, ctx.tokenizer, ctx.test_examples, ctx.tasks, ctx.device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(ctx.device)
    return {
        "valid_avg_score": avg_score(vres), "valid_per_task": vres,
        "valid_avg_loss": avg_loss(vloss), "valid_per_task_loss": vloss,
        "test_avg_score": avg_score(tres), "test_per_task": tres,
    }


def _init_source_scores(ctx: StrategyContext, init: str):
    """(source name, valid_avg_score, test_avg_score) the init's source cell
    recorded for the point we are centering on: coeff_search for coeff_best,
    subspace_gd / bo_search for the two vector inits. The three scalar
    constructions (pretrained / avg / merged) have no source cell -> (None,)*3.
    """
    if init == "coeff_best":
        return "coeff_search", ctx.coeff_best_valid_avg_score, ctx.coeff_best_test_avg_score
    if init == SUBSPACE_BEST_INIT:
        return (f"subspace_gd/{ctx.subspace_best_source}",
                ctx.subspace_best_valid_avg_score, ctx.subspace_best_test_avg_score)
    if init == BO_BEST_INIT:
        return (f"bo_search/{ctx.bo_best_source}",
                ctx.bo_best_valid_avg_score, ctx.bo_best_test_avg_score)
    return None, None, None


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _betas_list(beta: float, n_grads: int) -> List[float]:
    """A draw's scalar β as the per-direction list _load_weights takes."""
    return [float(beta)] * n_grads if n_grads else []


def _quantile(xs, q: float) -> float:
    """Linear-interpolated quantile (numpy's default), without numpy."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}
