"""Task registry.

Every task module exposes:
  - EVAL_SPLIT_SPEC: dict -- either {"pool": <hf_split>} (auto fixed-seed
    90/10 split into valid/test, see src/data.py) or
    {"valid": <hf_split>, "test": <hf_split>} (native split pair, used as-is).
  - load_split(hf_split) -> list[dict]     one HF split, formatted into rows.
  - format_example(row) -> str | list[dict]  prompt or chat messages.
  - score(response_str, row) -> bool | float
  - sft_target(row) -> Optional[str]       gold completion for GD strategies;
    None if the task has no canonical completion (see tasks/ifeval.py).

gsm8k and mbpp are the out-of-domain (unseen) tasks of runs/ood_generalization:
same contract, but no checkpoint is fine-tuned for them, so they are only
ever evaluated (test pool), never merged or trained on.
"""

from __future__ import annotations

from types import ModuleType

from . import bank77, ddxplus, gsm8k, ifeval, mbpp, usefulness_judge

_REGISTRY: dict[str, ModuleType] = {
    "bank77": bank77,
    "ddxplus": ddxplus,
    "gsm8k": gsm8k,
    "ifeval": ifeval,
    "mbpp": mbpp,
    "usefulness_judge": usefulness_judge,
}


def get_task(name: str) -> ModuleType:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_tasks() -> list[str]:
    return sorted(_REGISTRY)
