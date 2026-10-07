"""Peak-memory / wall-time / model-FLOP helpers for cost estimation.

Same shape as vision_exp/src/cost.py. A paper run's cost is
``one timed action x n_actions``:

  coeff_search  one lambda: CPU merge + load +
                valid generate+score (the sweep's
                diagnostic loss forward is not
                counted)                             x  lambda_steps (10)
  bo_search     one trial: W(c) build + load +
                valid generate+score                 x  bo_trials (50; 20 for 1-D methods)
  weight_gd_lora one train epoch (LoRA adapters)    x  gd_epochs (5)
  subspace_gd   one train epoch                      x  subspace_epochs (5)

(full-parameter weight_gd can still be profiled on request; the benchmark's
"Weight GD" point is the LoRA variant, which is what the main runs report at
every model size.)

Protocol (runs/cost_estimation/ta.sh): the plainest layout that fits one card
and the same per-step batch for every strategy, so peak memory is what each
method fundamentally needs -- with M = one bf16 model copy and N = tasks:
M + activations for coeff_search / bo_search (merge and W(c) built on the
CPU), M + adapters for weight_gd_lora, (N + 2) M for subspace_gd (base + N
directions + the model holding W(c), gradients projected as they land) --
rather than whichever fit-it-on-a-smaller-card trick the main run used.
The GD loss is Liger's fused linear cross-entropy (strategies/_common.lm_loss),
so no strategy pays for a materialized [seq, vocab] logits tensor -- HF's
default loss would add ~3.7 GiB per sequence on Qwen's 152k vocab, an artifact
of the loss implementation that generate() already avoids for the search
strategies. Gradient checkpointing stays on for the GD strategies: it is their
fixed training recipe, and its recompute is charged in FLOPs below.

vision_exp counts *images*; here the unit of work is the *token*: every
position the decoder processes, padding included (padding is real compute --
the matmuls run over it -- and the FLOP calibration below divides by the same
padded count, so the two stay consistent).

Model FLOPs = tokens x ``fwd_flops_per_token``, with the per-token number read
off ``FlopCounterMode`` on one teacher-forced forward (matmuls + attention;
sdpa, so the attention kernels are visible to the counter). This is a
linear-in-tokens approximation: generate()'s prefill skips the lm_head for
all but the last position and its KV-cached decode steps attend over a
growing cache, both of which the single per-token constant glosses over.
backward ~= 2x forward; the GD strategies train under gradient
checkpointing, which recomputes the forward during backward, so their
backward tokens cost (2 + 1)x forward.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch

# The init each GD strategy is profiled at, i.e. the cell plot_cost reads the
# score from. weight_gd: each method's own default coefficient (the "naive"
# init: 1/N for TA/DARE, 1 for TIES/TSVM). weight_gd_lora / subspace_gd: the
# best-scoring init on the main results (results/units, test_avg_score) --
# lambda* for LoRA on the per-task-direction methods and coeff=1 on the
# single-direction ones (14/16 units); coeff=1 for subspace_gd (6/8 units,
# and the only init run at 1.7b), where starting from lambda* actually lands
# far below coeff_search.
WEIGHT_GD_INIT = {"ta": "avg", "dare": "avg", "ties": "merged", "tsvm": "merged"}
WEIGHT_GD_LORA_INIT = {"ta": "coeff_best", "dare": "coeff_best", "ties": "merged", "tsvm": "merged"}
SUBSPACE_GD_INIT = "merged"
BO_SEARCH_INIT = "coeff_best"
# runs/bo_search/*.sh: 50 trials for the per-task-direction methods, 20 for the
# single-direction ones (TIES / TSV-M have a 1-D box).
BO_TRIALS = {"ta": 50, "dare": 50, "ties": 20, "tsvm": 20}
BWD_FLOP_FACTOR = 2.0
RECOMPUTE_FLOP_FACTOR = 1.0  # gradient checkpointing re-runs the forward in backward

PROFILED_STRATEGIES = ("coeff_search", "bo_search", "weight_gd_lora", "subspace_gd", "weight_gd")
DEFAULT_STRATEGIES = ("coeff_search", "bo_search", "weight_gd_lora", "subspace_gd")


def weight_gd_init(method: str) -> str:
    return WEIGHT_GD_INIT.get(method, "merged")


def weight_gd_lora_init(method: str) -> str:
    return WEIGHT_GD_LORA_INIT.get(method, "merged")


def bo_trials(method: str) -> int:
    return BO_TRIALS.get(method, 50)


def n_actions(strategy: str, *, lambda_steps: int, bo_trials: int, gd_epochs: int, subspace_epochs: int) -> int:
    if strategy == "coeff_search":
        return int(lambda_steps)
    if strategy == "bo_search":
        return int(bo_trials)
    if strategy in ("weight_gd", "weight_gd_lora"):
        return int(gd_epochs)
    if strategy == "subspace_gd":
        return int(subspace_epochs)
    raise ValueError(f"unknown strategy {strategy!r}")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def measure() -> Iterator[dict]:
    """Time a block and record CUDA peak memory. Caller fills n_fwd / n_bwd tokens."""
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    stats = {"n_fwd_tokens": 0, "n_bwd_tokens": 0}
    yield stats
    _sync()
    allocated = reserved = 0
    if torch.cuda.is_available():
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    stats["action_wall_s"] = time.perf_counter() - t0
    stats["peak_mem_allocated_bytes"] = allocated
    stats["peak_mem_reserved_bytes"] = reserved


class TokenCounter:
    """Count every position the model forwards (``input_ids.numel()`` per
    call) via a forward pre-hook on the top-level module.

    Covers generate() transparently: the prefill call contributes B x L and
    each KV-cached decode step contributes B (one new token per row). Use as
    a context manager around the block to count; read ``.tokens`` after.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._handle = None
        self.tokens = 0

    def _hook(self, module, args, kwargs) -> None:
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            ids = kwargs.get("inputs_embeds")
            if ids is not None:
                self.tokens += int(ids.shape[0] * ids.shape[1])
            return
        self.tokens += int(ids.numel())

    def __enter__(self) -> "TokenCounter":
        self._handle = self._model.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def batch_tokens(batch: Dict[str, torch.Tensor]) -> int:
    """Positions in one SFT batch (padded length x rows) -- the same count the
    forward pre-hook would see for ``model(**batch)``."""
    return int(batch["input_ids"].numel())


def fwd_flops_per_token(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    """Per-position forward FLOPs via FlopCounterMode on one full forward (no
    labels: the lm_head still runs over every position, but HF's loss path --
    fp32 logits over the whole vocab, ~13 GiB at batch 4 on a 4B model -- is
    skipped; its FLOPs are negligible next to the matmuls)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    n = max(batch_tokens(batch), 1)
    return float(fc.get_total_flops()) / n


def finalize(action: dict, strategy: str, n: int, fwd_flops: float, grad_checkpointing: bool) -> dict:
    n_fwd = int(action.get("n_fwd_tokens", 0))
    n_bwd = int(action.get("n_bwd_tokens", 0))
    bwd_factor = BWD_FLOP_FACTOR + (RECOMPUTE_FLOP_FACTOR if grad_checkpointing else 0.0)
    action_flops = n_fwd * fwd_flops + bwd_factor * n_bwd * fwd_flops
    action["n_actions"] = n
    action["grad_checkpointing"] = bool(grad_checkpointing)
    action["bwd_flop_factor"] = bwd_factor
    action["action_model_flops"] = action_flops
    action["est_wall_s"] = float(action["action_wall_s"]) * n
    action["est_model_flops"] = action_flops * n
    return action
