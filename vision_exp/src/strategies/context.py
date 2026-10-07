"""Shared context passed to every coefficient-selection strategy.

Groups the merge inputs (method + task vectors + loaders) and the per-strategy
hyperparameters so strategy functions keep a small, uniform signature.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    valid_loaders: Dict[str, object]
    test_loaders: Dict[str, object]
    device: str

    # coefficient search (diagonal lambda sweep)
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    lambda_steps: int = 11

    # weight-space GD
    gd_epochs: int = 10
    gd_lr: float = 1e-5
    gd_warmup_ratio: float = 0.1
    gd_optimizer: str = "adamw"  # adamw | sgd (momentum=0 => vanilla SGD)
    gd_momentum: float = 0.0
    gd_patience: int = 5  # early-stop if train loss flat this many epochs; 0 = off
    gd_l2_sp: float = 0.0  # L2-SP: CE + (alpha/2)||W-W_init||^2; 0 = off
    # gt | expert_soft | expert_hard -- see strategies/pseudo_labels.py. Non-gt
    # sources train on the same valid budget but never read its labels
    # (test-time adaptation); results land under weight_gd_<source>.
    gd_label_source: str = "gt"

    # test-time adaptation (adamerging, divmerge, weight_gd expert_*): the data
    # those strategies train on (default: a slice of each task's test split)
    unlabeled_loaders: Optional[Dict[str, object]] = None
    unlabeled_stats: Optional[dict] = None
    divergence: str = "js"  # js | kl for divmerge and weight_gd expert_soft
    pseudo_loaders: Optional[Dict[str, object]] = None  # cache filled by pseudo_labels
    pseudo_stats: Optional[dict] = None

    # AdaMerging (entropy) / DivMerge (divergence to expert logits) coefficients;
    # one step = one batch per task. Defaults follow model-merge-transfer.
    ada_steps: int = 1000
    ada_lr: float = 1e-3
    ada_prior: Optional[float] = None  # None -> 1/N (the AdaMerging paper uses 0.3)
    div_steps: int = 1000
    div_lr: float = 1e-2
    div_prior: Optional[float] = None  # None -> 1/N

    # subspace GD (autograd on coefficient vector)
    subspace_epochs: int = 20
    subspace_lr: float = 1e-2
    subspace_warmup_ratio: float = 0.1
    subspace_patience: int = 5  # 0 = off

    # directional sampling around an init; density vs coeff_search.valid_avg_acc
    ds_alpha: float = 0.05
    ds_beta: float = 1e-5
    ds_samples: int = 64
    ds_seed: int = 42

    # Bayesian optimization over the basis coefficients (optuna GPSampler),
    # maximizing valid_avg_acc directly -- see strategies/bo_search.py. The box
    # is [max(bo_lower, c0 - bo_radius), c0 + bo_radius] per coefficient around
    # the init center c0; bo_startup_trials=None -> 2 * n_dirs + 1 (the enqueued
    # center + 2 * n_dirs uniform draws) before the GP starts proposing.
    bo_trials: int = 50
    bo_startup_trials: Optional[int] = None
    bo_radius: float = 0.4
    bo_lower: Optional[float] = -0.2  # None = no clamp (allow negative coefficients)
    bo_seed: int = 42

    # checkpointing / cross-strategy state
    save_checkpoints: bool = False
    checkpoint_dir: Optional[str] = None
    best_lambda: Optional[float] = None
    coeff_best_valid_avg_acc: Optional[float] = None

    # baselines: resolved multi-task checkpoint (None => upper_bound skipped)
    multitask_checkpoint: Optional[str] = None

    # smaller_9task: original label ids kept per task. None = every class.
    # Head row j is original label class_indices[task][j]; loader labels match.
    class_indices: Optional[Dict[str, List[int]]] = None
