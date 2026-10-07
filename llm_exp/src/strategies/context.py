"""Shared context passed to every coefficient-selection strategy.

Groups the merge inputs (method + task vectors + eval pools) and the
per-strategy hyperparameters, same shape as vision_exp's StrategyContext.
Differences from vision_exp: ``valid_examples``/``test_examples`` are raw
example-dict lists (generation-based eval, no DataLoader) instead of image
DataLoaders, and ``sft_pools`` holds the (prompt, target) pairs weight_gd /
subspace_gd train on -- built once in experiment.py via
``data.build_sft_pool`` (which special-cases ifeval, see design.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..merging.base import MergeMethod


@dataclass
class StrategyContext:
    # merge inputs
    method: MergeMethod
    base_model: str
    base_sd: Dict[str, torch.Tensor]
    task_sds: List[Dict[str, torch.Tensor]]
    keys: List[str]
    tasks: List[str]
    tokenizer: object
    valid_examples: Dict[str, list]
    test_examples: Dict[str, list]
    sft_pools: Dict[str, list]
    device: str
    dtype: str = "bfloat16"  # GPU-resident model / merge tensors; base_sd/task_sds stay fp32 on CPU

    # Held-out (out-of-domain) tasks: test pools only. No task vector, no valid
    # pool, no SFT data -- nothing here ever feeds a selection criterion; every
    # strategy just scores its final weights on them via _common.attach_unseen
    # (same shape as vision_exp's unseen_tasks / unseen_test_loaders).
    unseen_tasks: List[str] = field(default_factory=list)
    unseen_test_examples: Dict[str, list] = field(default_factory=dict)

    # generation (used by coeff_search's sweep and by every strategy's final eval)
    # None -- the default -- uses each task's own MAX_NEW_TOKENS (see
    # src/tasks/*.py); set an int to override every task uniformly.
    max_new_tokens: Optional[int] = None
    temperature: float = 0.01
    top_p: float = 0.95
    gen_batch_size: int = 16

    # coefficient search (diagonal lambda sweep)
    lambda_min: float = 0.1
    lambda_max: float = 1.0
    lambda_steps: int = 10

    # weight-space GD
    gd_epochs: int = 5
    gd_lr: float = 3e-5  # weight_gd only; weight_gd_lora reads lora_lr below

    gd_warmup_ratio: float = 0.1
    gd_batch_size: int = 4  # per-GPU batch size fetched per forward/backward
    # Number of gd_batch_size batches to accumulate (per task) before
    # optimizer.step() -- effective batch size = gd_batch_size *
    # gd_grad_accum_steps, but peak activation memory only ever sees one
    # gd_batch_size-sized batch at a time.
    gd_grad_accum_steps: int = 1
    max_seq_length: int = 2048

    # weight-space GD, LoRA variant (weight_gd_lora) -- ignored by plain weight_gd
    #
    # lora_lr is weight_gd_lora's own learning rate; it does NOT fall back to
    # gd_lr. LoRA trains a tiny fraction of the parameters from a zero-init B,
    # so its usable LR sits an order of magnitude above full fine-tuning's
    # (held-out search on qwen3-0.6b: weight_gd 3e-5, weight_gd_lora 3e-4).
    # Sharing one knob forced the two strategies into separate merge_eval.py runs;
    # separate knobs let one run cover both at their own optima.
    lora_lr: float = 3e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # None -> weight_gd_lora's own default

    # valid-set teacher-forced loss (see _common.eval_all_tasks_loss) -- forward
    # only, no backward/optimizer state, so this can run at a larger batch size
    # than gd_batch_size/subspace_batch_size without the grad-accum machinery.
    loss_batch_size: int = 4

    # subspace GD (autograd on coefficient vector)
    subspace_epochs: int = 5
    subspace_lr: float = 1e-2
    # Any name transformers.get_scheduler() accepts: linear, cosine,
    # cosine_with_restarts, polynomial, constant, constant_with_warmup,
    # inverse_sqrt, ...
    subspace_scheduler: str = "cosine"
    subspace_warmup_ratio: float = 0.1
    subspace_batch_size: int = 4  # per-GPU batch size fetched per forward/backward
    subspace_grad_accum_steps: int = 1
    # Where subspace_gd keeps its (1 + n_dirs) full-model copies of base/basis:
    # auto = as many on the GPU as fit while leaving subspace_basis_headroom_gib
    # free (directions first, base last), the rest pinned in host memory and
    # streamed per key; cuda / cpu force one placement. See subspace_gd.py.
    subspace_basis_device: str = "auto"  # auto | cuda | cpu
    subspace_basis_headroom_gib: float = 6.0

    # directional sampling around an init; density vs coeff_search's valid
    # metric on the same axis the draws are ranked by (see
    # strategies/directional_sampling.py). ds_select_metric="loss" ranks draws
    # by teacher-forced valid NLL (one forward per draw); "score" ranks them by
    # generate()+score (vision_exp's literal protocol, far more expensive here).
    ds_alpha: float = 0.05
    ds_beta: float = 0.0
    ds_samples: int = 64
    ds_seed: int = 42
    ds_select_metric: str = "loss"  # loss | score
    # True: every draw also gets a test generate()+score, and the result
    # reports the test-score distribution over all draws (test_dist) instead
    # of re-evaluating the valid-selected top-5.
    ds_eval_test: bool = False
    # Where W*, the basis and the gradient directions live between draws:
    # "auto" keeps them on the GPU only if they fit next to the model and
    # generate()'s KV cache, else on the CPU (each draw is combined there and
    # copied in). See strategies/directional_sampling.py.
    ds_basis_device: str = "auto"  # auto | cuda | cpu
    # False (default): W*'s valid/test scores are the ones its source cell
    # recorded (bo_search / subspace_gd / coeff_search); True: re-evaluate W*
    # in the run. Scalar inits without a source are always re-evaluated.
    ds_reeval_center: bool = False

    # Bayesian optimization over the basis coefficients (optuna GPSampler),
    # maximizing valid_avg_score directly -- see strategies/bo_search.py. The
    # box is [max(bo_lower, c0 - bo_radius), c0 + bo_radius] per coefficient
    # around the init's center c0; bo_startup_trials=None -> 2 * n_dirs + 1
    # (the center itself + 2 * n_dirs uniform draws) before the GP proposes.
    bo_trials: int = 50
    bo_startup_trials: Optional[int] = None
    bo_radius: float = 0.7
    bo_lower: Optional[float] = 0.0  # None = no clamp (allow negative coefficients)
    bo_seed: int = 42
    # auto | cuda | cpu -- where base + basis live between trials (auto: GPU
    # only if model + basis fit in ~60% of the card; 4B models fall back to CPU).
    bo_basis_device: str = "auto"

    # checkpointing / cross-strategy state
    save_checkpoints: bool = False
    checkpoint_dir: Optional[str] = None
    best_lambda: Optional[float] = None
    # coeff_search's own valid metrics at lambda* -- the directional_sampling
    # density thresholds (one per selection metric).
    coeff_best_valid_avg_score: Optional[float] = None
    coeff_best_valid_avg_loss: Optional[float] = None
    coeff_best_test_avg_score: Optional[float] = None
    # subspace_gd's trained coefficient vector (one per basis direction) and
    # the init it came from -- the center for directional_sampling's
    # 'subspace_best' init. Unlike the four scalar inits this is per-direction.
    subspace_best_coeffs: Optional[List[float]] = None
    subspace_best_source: Optional[str] = None
    # bo_search's best-trial coefficient vector and the init it came from --
    # the center for directional_sampling's 'bo_best' init. Same shape as
    # subspace_best (one value per basis direction); differs in origin only.
    bo_best_coeffs: Optional[List[float]] = None
    bo_best_source: Optional[str] = None
    # The valid / test score the source cell recorded for each vector init --
    # what directional_sampling prints as "the init's score" next to its own
    # re-evaluation of the same point.
    subspace_best_valid_avg_score: Optional[float] = None
    subspace_best_test_avg_score: Optional[float] = None
    bo_best_valid_avg_score: Optional[float] = None
    bo_best_test_avg_score: Optional[float] = None

    # multi-task upper-bound checkpoint for the 'baselines' strategy (None when
    # no joint checkpoint has been resolved for these tasks/arch)
    multitask_checkpoint: Optional[str] = None
