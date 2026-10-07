"""Subspace GD strategy (simple autograd on the coefficient vector).

Optimizes a coefficient vector c with one entry per basis direction supplied by
the merging method:

    W(c) = base + sum_i c_i * dir_i

Since W(c) is linear in c, dL/dc_i = <dL/dW, dir_i>: load W(c) into the real
model and run an ordinary forward/backward, then project the weight-space
gradient onto each basis direction to get dL/dc. Mathematically identical to
differentiating through the reparametrization directly (verified against a toy
model), but the model is a plain nn.Module with real trainable parameters
throughout -- no torch.func.functional_call / parameter-substitution tricks --
so ordinary model.gradient_checkpointing_enable() just works, unlike
substituting weights via functional_call, where checkpointing's recomputation
during backward can silently see the model's own frozen parameters instead of
W(c) once the substitution context has closed.

The number of coefficients is len(method.basis(...)): TA gives one per task,
TIES gives a single scalar.

VRAM budget (the reason this file looks the way it does). Every full-model
copy is ~3.2 GiB in bf16 on qwen3-1.7b, and the naive layout -- model + its
.grad + a W(c) state dict + base + n_dirs directions, all resident -- is
(4 + n_dirs) copies: 22.7 GiB for TA before the first activation, which is
what OOMed every 1.7b TA run on a 24 GB card. Three things bring that down to
(2 + n_dirs) copies plus activations, and let the remainder spill to pinned
host memory when even that doesn't fit:

* **W(c) is materialized in place** into ``model.parameters()`` (per key, fp32
  accumulate, one bf16 round at the end), so there is no separate W dict.
* **Only real parameters are copied**, not state_dict keys: ``lm_head.weight``
  is tied to ``embed_tokens.weight`` on qwen3-0.6b/1.7b but is a separate
  state_dict entry, so the old per-key copies carried the 151936x2048
  embedding twice in every one of the (1 + n_dirs) copies.
* **Dot products are chunked fp32** -- no full-tensor ``.float()`` temporaries
  (the embedding's alone were 2 x 1.2 GiB).

The remaining model-sized buffer is ``.grad``: it accumulates across the
step's backward calls (one per task x grad-accum micro-batch) and is projected
onto the basis once per step, then freed. Projecting inside every backward via
gradient hooks would drop that buffer, but re-does the n_dirs dot products per
backward (16x per step at batch 1 x accum 4 x 4 tasks) and re-streams any
host-resident direction each time (measured 48 s/step on PCIe Gen3 x8); the
1.7b TA layout fits on a 32 GB card with the buffer (28.1 GB peak allocated),
so the once-per-step projection is the only mode.

``ctx.subspace_basis_device`` decides where base/dirs live: ``auto`` (default)
reserves one model-sized slot for ``.grad``, then keeps as many of them on the
GPU as fit under ``ctx.subspace_basis_headroom_gib`` of free memory --
directions first, base last -- and pins the rest in host memory, streamed per
key once per step. ``cuda``/``cpu`` force one placement for everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import get_scheduler

from .. import models
from ..data import get_sft_dataloader
from ..models import load_causal_lm
from ..tasks import get_task
from ._common import (
    attach_unseen, clear_cache, eval_all_tasks, eval_all_tasks_loss, free_model, lm_loss, pack_result,
    save_finetuned,
)
from .context import StrategyContext

StateDict = Dict[str, torch.Tensor]

# Elements per fp32 chunk in _dot: 2**26 * 4 B = 256 MiB per temporary.
_DOT_CHUNK = 1 << 26


def _coeffs_to_list(c) -> List[float]:
    return [float(x) for x in c.detach().cpu().tolist()]


def _param_names(model, keys: List[str]) -> List[str]:
    """The names subspace_gd operates over: ``model.named_parameters()`` (which
    dedupes tied weights), checked against the merge's shared float ``keys``."""
    names = [n for n, _ in model.named_parameters()]
    missing = [n for n in names if n not in keys]
    if missing:
        raise ValueError(
            f"{len(missing)} model parameter(s) have no task-vector key (e.g. {missing[:3]}); "
            "subspace_gd needs every trainable parameter covered by the merge."
        )
    return names


def _dot(g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """<g, d> accumulated in fp32 over 256 MiB chunks -- never materializes a
    full-tensor fp32 copy of either operand."""
    g, d = g.reshape(-1), d.reshape(-1)
    acc = torch.zeros((), dtype=torch.float32, device=g.device)
    for s in range(0, g.numel(), _DOT_CHUNK):
        acc += torch.dot(g[s : s + _DOT_CHUNK].float(), d[s : s + _DOT_CHUNK].float())
    return acc


def _pin(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.contiguous().pin_memory()
    except RuntimeError as e:  # pinned allocation can fail on a constrained host; pageable still works
        logging.warning(f"subspace_gd: pin_memory failed ({e}); using pageable host memory")
        return t.contiguous()


def _place_basis(
    pieces: List[StateDict], names: List[str], dtype: torch.dtype, device: str,
    placement: str, headroom_gib: float,
) -> tuple:
    """Copy each piece (a full-model state dict; the n_dirs directions first,
    base last) to the GPU or to pinned host memory. Returns (placed, n_on_gpu).

    ``auto``: one model-sized slot is reserved for the ``.grad`` buffer the
    once-per-step projection needs (see module docstring), then as many pieces
    as still fit under the headroom go on the GPU; the rest are pinned in host
    memory and streamed per key."""
    n_pieces = len(pieces)
    bytes_per_piece = sum(pieces[0][n].numel() * dtype.itemsize for n in names)
    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if placement == "cuda" or not on_cuda:
        n_gpu = n_pieces
    elif placement == "cpu":
        n_gpu = 0
    elif placement == "auto":
        free, _ = torch.cuda.mem_get_info()
        n_fit = int(max(0, (free - headroom_gib * 2**30) // bytes_per_piece))
        n_gpu = min(max(0, n_fit - 1), n_pieces)  # -1: the .grad buffer's slot
    else:
        raise ValueError(f"Unknown subspace_basis_device '{placement}' (expected auto, cuda, cpu)")

    placed = []
    for i, sd in enumerate(pieces):
        if i < n_gpu:
            placed.append({n: sd[n].to(dtype).to(device) for n in names})
        else:
            placed.append({n: _pin(sd[n].to(dtype)) for n in names})
    logging.info(
        f"  basis: {n_pieces} full-model copies x {bytes_per_piece / 2**30:.2f} GiB -- "
        f"{n_gpu} on {device}, {n_pieces - n_gpu} in pinned host memory (placement={placement})"
    )
    return placed, n_gpu


@torch.no_grad()
def _materialize(model, c: torch.Tensor, base: StateDict, dirs: List[StateDict], device: str) -> None:
    """Write W(c) = base + sum_i c_i dir_i straight into model.parameters().

    Per key: fp32 accumulate, one bf16 round on the final copy_ (the old
    per-key bf16 chain rounded after every add). Pieces already on ``device``
    make ``.to`` a no-op; pinned host pieces stream asynchronously."""
    coeffs = c.detach().float().tolist()
    for name, p in model.named_parameters():
        w = base[name].to(device, non_blocking=True).float()
        for ci, d in zip(coeffs, dirs):
            w.add_(d[name].to(device, non_blocking=True), alpha=ci)
        p.copy_(w)


@torch.no_grad()
def _project_from_grads(model, dirs: List[StateDict], device: str, grad_c: torch.Tensor) -> None:
    """dL/dc_i = <dL/dW, dir_i>, read off the .grad buffers accumulated over a
    whole step (per-task and grad-accum backward() calls sum into .grad exactly
    as a single backward() on the summed loss would). Host-resident directions
    stream once per step. ``named_parameters()`` dedupes tied weights
    (qwen3-0.6b/1.7b tie lm_head to embed_tokens) to one tensor whose .grad
    already sums both usages, so nothing is double-counted. Frees .grad after."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        for i, d in enumerate(dirs):
            grad_c[i] += _dot(p.grad, d[name].to(device, non_blocking=True))
        p.grad = None


def _build_dataloaders(ctx: StrategyContext):
    dataloaders = {}
    for t in ctx.tasks:
        pool = ctx.sft_pools.get(t)
        if pool:
            dataloaders[t] = get_sft_dataloader(
                pool, ctx.tokenizer, batch_size=ctx.subspace_batch_size, max_length=ctx.max_seq_length,
                enable_thinking=getattr(get_task(t), "ENABLE_THINKING", True),
            )
    if not dataloaders:
        raise ValueError("No task in this unit has any SFT examples -- subspace_gd cannot train.")
    return dataloaders


@dataclass
class _TrainState:
    """Everything one subspace-GD epoch needs, built once by _setup."""
    model: torch.nn.Module
    c: torch.nn.Parameter
    base_dev: StateDict
    dirs_dev: List[StateDict]
    n_dirs: int
    grad_c: torch.Tensor
    dataloaders: dict
    optimizer: torch.optim.Optimizer
    scheduler: object
    steps_per_epoch: int

    def release_basis(self) -> None:
        """Drop base/dirs (GPU and pinned host memory)."""
        self.base_dev, self.dirs_dev = {}, []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _setup(ctx: StrategyContext, init: str) -> _TrainState:
    """Fresh model with gradient checkpointing, basis placed per
    ``ctx.subspace_basis_device``, optimizer over c."""
    method, keys, tasks, device = ctx.method, ctx.keys, ctx.tasks, ctx.device
    dtype = models.resolve_dtype(ctx.dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info(f"subspace_gd(init={init}): building fresh model")
    # Model first, then the basis: `auto` placement reads the free memory the
    # model leaves behind.
    model = load_causal_lm(ctx.base_model, device=device, dtype=dtype)
    model.gradient_checkpointing_enable()
    names = _param_names(model, keys)

    dirs_cpu = method.basis(ctx.base_sd, ctx.task_sds, keys)
    n_dirs = len(dirs_cpu)
    placed, _ = _place_basis(
        dirs_cpu + [ctx.base_sd], names, dtype, device,
        ctx.subspace_basis_device, ctx.subspace_basis_headroom_gib,
    )
    dirs_dev, base_dev = placed[:n_dirs], placed[n_dirs]
    del dirs_cpu, placed

    c = torch.nn.Parameter(
        method.default_coeffs(n_dirs, init, ctx.best_lambda, n_tasks=len(tasks)).to(device)
    )
    grad_c = torch.zeros(n_dirs, dtype=torch.float32, device=device)

    dataloaders = _build_dataloaders(ctx)
    optimizer = torch.optim.AdamW([c], lr=ctx.subspace_lr)
    accum = ctx.subspace_grad_accum_steps
    micro_steps_per_epoch = max(len(dl) for dl in dataloaders.values())
    steps_per_epoch = -(-micro_steps_per_epoch // accum)  # ceil(micro_steps / accum)
    total_steps = steps_per_epoch * ctx.subspace_epochs
    scheduler = get_scheduler(
        ctx.subspace_scheduler, optimizer,
        num_warmup_steps=int(total_steps * ctx.subspace_warmup_ratio), num_training_steps=total_steps,
    )
    return _TrainState(
        model, c, base_dev, dirs_dev, n_dirs, grad_c, dataloaders, optimizer, scheduler, steps_per_epoch,
    )


def _one_epoch(st: _TrainState, ctx: StrategyContext):
    """One subspace-GD epoch. Returns (avg_loss, n_fwd_tokens, n_bwd_tokens)."""
    device = ctx.device
    accum = ctx.subspace_grad_accum_steps
    model, c, dataloaders = st.model, st.c, st.dataloaders
    model.train()
    iters = {t: iter(dl) for t, dl in dataloaders.items()}
    epoch_loss = 0.0
    n_tokens = 0
    for _ in range(st.steps_per_epoch):
        st.optimizer.zero_grad()
        model.zero_grad(set_to_none=True)
        st.grad_c.zero_()
        _materialize(model, c, st.base_dev, st.dirs_dev, device)

        step_loss = 0.0
        # Backward per task (instead of summing all tasks' losses into one
        # total_loss.backward()) so each task's forward activations free
        # before the next task's forward runs -- one task resident at a
        # time instead of n_tasks stacked together. model.parameters()'s
        # .grad accumulates the same total across the n backward() calls
        # as it would from a single backward() on the summed loss. Within a task,
        # further accumulate over `accum` subspace_batch_size fetches
        # before moving on -- effective batch size stays
        # subspace_batch_size * accum, but peak activation memory only
        # ever sees one subspace_batch_size-sized batch -- same reasoning
        # as weight_gd.
        for task, dl in dataloaders.items():
            for _ in range(accum):
                try:
                    batch = next(iters[task])
                except StopIteration:
                    iters[task] = iter(dl)
                    batch = next(iters[task])
                batch = {k: v.to(device) for k, v in batch.items()}
                n_tokens += int(batch["input_ids"].numel())
                loss = lm_loss(model, batch)
                (loss / accum / len(dataloaders)).backward()
                step_loss += loss.item() / accum / len(dataloaders)
        _project_from_grads(model, st.dirs_dev, device, st.grad_c)
        c.grad = st.grad_c.clone().to(c.dtype)
        st.optimizer.step()
        st.scheduler.step()
        epoch_loss += step_loss
    # every position is forwarded once and backpropagated once
    return epoch_loss / st.steps_per_epoch, n_tokens, n_tokens


def run_one(ctx: StrategyContext, init: str) -> dict:
    tasks, device = ctx.tasks, ctx.device
    st = _setup(ctx, init)
    model, c, n_dirs = st.model, st.c, st.n_dirs
    coeff_init = _coeffs_to_list(c)

    logging.info(f"\n=== Subspace GD (init={init}) ===")
    logging.info(
        f"  optimizing {n_dirs} coefficient(s)  lr={ctx.subspace_lr}  "
        f"scheduler={ctx.subspace_scheduler}  epochs={ctx.subspace_epochs}"
    )
    best_loss, best_coeff, best_epoch = float("inf"), None, 0
    for epoch in range(ctx.subspace_epochs):
        avg_loss, _, _ = _one_epoch(st, ctx)
        logging.info(f"  epoch {epoch + 1}/{ctx.subspace_epochs}  avg_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss, best_coeff, best_epoch = avg_loss, c.detach().clone(), epoch + 1

    if best_coeff is not None:
        c.data.copy_(best_coeff)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True  # re-enable KV cache for the eval-time generate() calls
    logging.info(f"  best epoch={best_epoch}/{ctx.subspace_epochs}  best_train_loss={best_loss:.4f}")

    logging.info(f"subspace_gd(init={init}): evaluating on valid/test")
    _materialize(model, c, st.base_dev, st.dirs_dev, device)
    # base_dev/dirs_dev (1 + n_dirs full-model-sized copies) have done their
    # job once W(c*) is in the model; free them (GPU and pinned host memory)
    # before generate()'s KV cache needs headroom instead of holding them
    # through the whole eval pass.
    st.release_basis()
    model.eval()
    valid_res = eval_all_tasks(
        model, ctx.tokenizer, ctx.valid_examples, tasks, device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    clear_cache(device)
    valid_loss = eval_all_tasks_loss(
        model, ctx.tokenizer, ctx.sft_pools, tasks, device, ctx.loss_batch_size, ctx.max_seq_length,
    )
    clear_cache(device)
    test_res = eval_all_tasks(
        model, ctx.tokenizer, ctx.test_examples, tasks, device,
        ctx.max_new_tokens, ctx.temperature, ctx.top_p, ctx.gen_batch_size,
    )
    logging.info(
        f"subspace_gd(init={init}): valid_avg_score={models.avg_score(valid_res):.4f}  "
        f"valid_avg_loss={models.avg_loss(valid_loss)}  test_avg_score={models.avg_score(test_res):.4f}"
    )
    coeff_final = _coeffs_to_list(c)

    extra = {
        "init": init,
        "subspace": True,
        "n_coeffs": n_dirs,
        "epochs": ctx.subspace_epochs,
        "learning_rate": ctx.subspace_lr,
        "scheduler": ctx.subspace_scheduler,
        "warmup_ratio": ctx.subspace_warmup_ratio,
        "coefficients_init": coeff_init,
        "coefficients_final": coeff_final,
        "best_train_loss": best_loss,
        "best_epoch": best_epoch,
    }
    if ctx.best_lambda is not None and init == "coeff_best":
        extra["best_lambda"] = ctx.best_lambda
    if ctx.save_checkpoints:
        tag = f"subspace_gd_{init}"
        logging.info(f"subspace_gd(init={init}): saving checkpoint -> {tag}")
        extra["checkpoint"] = save_finetuned(ctx, model, ctx.tokenizer, tag)
        logging.info(f"subspace_gd(init={init}): checkpoint saved")
    result = pack_result(valid_res, test_res, valid_loss=valid_loss, **extra)
    attach_unseen(ctx, model, result)
    free_model(model, device)
    return result


def run(ctx: StrategyContext, inits: List[str]) -> Dict[str, dict]:
    return {init: run_one(ctx, init) for init in inits}


def profile_one(ctx: StrategyContext, init: str) -> dict:
    """Warm up one epoch, then time a second epoch (peak mem + wall + token counts).

    Peak GPU memory includes the .grad buffer and whatever share of the
    (1 + n_dirs) base/basis copies ``ctx.subspace_basis_device`` placed on the
    GPU, exactly as the real training loop holds them; copies spilled to pinned
    host memory don't count.
    """
    from ..cost import measure

    logging.info(f"subspace_gd(init={init}): profile (warmup + 1 timed epoch)")
    st = _setup(ctx, init)
    _one_epoch(st, ctx)
    with measure() as stats:
        avg_train_loss, n_fwd, n_bwd = _one_epoch(st, ctx)
        stats["n_fwd_tokens"] = n_fwd
        stats["n_bwd_tokens"] = n_bwd
    logging.info(
        f"  timed epoch  n_dirs={st.n_dirs}  avg_loss={avg_train_loss:.4f}  tokens={n_fwd}  "
        f"wall={stats['action_wall_s']:.1f}s  peak={stats['peak_mem_allocated_bytes'] / 1e9:.2f} GB"
    )
    st.release_basis()
    free_model(st.model, ctx.device)
    return {"init": init, "action": "train_epoch", "n_dirs": st.n_dirs, "steps_per_epoch": st.steps_per_epoch, **stats}
