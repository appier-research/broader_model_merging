"""Dataset loading, valid/test split, budget subsampling, and SFT collation.

The LLM analogue of vision_exp's ``_train_valid_split``/``apply_valid_budget``:
every task's single available eval split gets the same fixed-seed treatment,
except tasks that declare a native valid/test split pair (see
tasks/*.EVAL_SPLIT_SPEC and tasks/ddxplus.py).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .models import RawPrompt, format_prompt
from .tasks import get_task

BudgetSpec = str  # "full" | "medium" | "low" | "<k>" | "<k>_per_class"


def parse_budget(budget):
    """Normalize a CLI budget string. Ported from vision_exp/src/data.py."""
    if isinstance(budget, int):
        return budget
    b = str(budget).strip().lower()
    if b in ("low", "medium", "full"):
        return b
    if b.endswith("_per_class"):
        b = b[: -len("_per_class")]
    if b.isdigit():
        return int(b)
    raise ValueError(
        f"Unknown budget '{budget}'. Expected low, medium, full, or an integer per-class/total count."
    )


# --------------------------------------------------------------------------- #
# Valid / test split
# --------------------------------------------------------------------------- #

def _fixed_seed_split(examples: List[Dict], valid_fraction: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, round(len(indices) * valid_fraction))
    valid_indices = sorted(indices[:n_valid])
    test_indices = sorted(indices[n_valid:])
    return [examples[i] for i in valid_indices], [examples[i] for i in test_indices]


def build_valid_test(task_name: str, seed: int = 42, valid_fraction: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """Return (valid_pool, test_pool) for a task.

    Tasks may define their own ``build_valid_test(seed) -> (valid, test)`` to
    override this entirely (currently just ifeval, whose valid pool has to
    come from a different dataset than its test pool -- see
    tasks/ifeval.py). Otherwise, dispatch on EVAL_SPLIT_SPEC: a "pool" spec
    loads one HF split and gets a fixed-seed split (default 90/10, mirrors
    vision_exp's train/valid split; a task can override the fraction via
    ``EVAL_SPLIT_SPEC["valid_fraction"]``, e.g. bank77's 50/50 -- see
    tasks/bank77.py). An explicit "valid"/"test" spec (currently just
    ddxplus) loads native splits directly, unmodified.

    Note: llm_exp doesn't run vision_exp's data-scaling (per-class budget
    sweep) experiments, so there's no need to keep every task's valid_fraction
    small/uniform the way vision_exp's fixed 10% does to preserve headroom for
    that sweep -- see design.md.
    """
    task = get_task(task_name)
    if hasattr(task, "build_valid_test"):
        return task.build_valid_test(seed=seed)
    spec = task.EVAL_SPLIT_SPEC
    if "pool" in spec:
        pool = task.load_split(spec["pool"])
        return _fixed_seed_split(pool, valid_fraction=spec.get("valid_fraction", valid_fraction), seed=seed)
    return task.load_split(spec["valid"]), task.load_split(spec["test"])


# --------------------------------------------------------------------------- #
# Few-shot prompts (unseen tasks only)
# --------------------------------------------------------------------------- #

RAW_SHOT_SEP = "\n\n###\n\n"  # delimiter between raw-completion shots; score() cuts the response at it


def resolve_n_shot(task_name: str, n_shot) -> int:
    """``None``/0 -> zero-shot, "auto" -> the task's DEFAULT_N_SHOT, else int."""
    if n_shot in (None, 0, "0", ""):
        return 0
    if n_shot == "auto":
        return int(getattr(get_task(task_name), "DEFAULT_N_SHOT", 0))
    return int(n_shot)


def shot_pool(task_name: str, seed: int = 42) -> List[Dict]:
    """Rows the shots are drawn from, per the task's SHOT_SPLIT: "valid" ->
    the task's own valid pool (never its test pool), otherwise an HF split
    name (e.g. gsm8k's "train")."""
    task = get_task(task_name)
    split = getattr(task, "SHOT_SPLIT", None)
    if split is None:
        raise ValueError(f"task {task_name!r} declares no SHOT_SPLIT; few-shot prompting is not supported for it")
    if split == "valid":
        return build_valid_test(task_name, seed=seed)[0]
    return task.load_split(split)


def apply_few_shot(task_name: str, rows: List[Dict], n_shot: int, seed: int = 42, raw: bool = False) -> List[Dict]:
    """Return copies of ``rows`` whose prompt carries ``n_shot`` worked
    examples. Each row gets its own shot draw (seeded by ``seed`` and the
    row's position, so the draw is reproducible and identical across cells):
    on qwen3-0.6b one fixed 4-shot set moved the same cell's gsm8k accuracy
    between 28 and 44 on identical test rows (format ~always fine), so a
    single shared set would make every absolute number a lottery ticket;
    per-row shots average that variance out inside one run at no extra cost.
    Every cell sees exactly the same (row, shots) pairs, so comparisons
    between cells stay paired.

    Chat form (default): the shots become user/assistant turn pairs between
    the system message and the query; the assistant turn is the task's
    ``sft_target`` (gsm8k: rationale ending in \\boxed{}, mbpp: the canonical
    solution in a ```python block) -- exactly the format ``score`` expects.

    ``raw=True`` instead builds one plain-text completion prompt via the
    task's ``raw_prompt(row, shots)`` (no chat template) and stores it as
    ``row["raw_prompt"]``, which the task's ``format_example`` returns as a
    :class:`RawPrompt`. Meant for the pretrained *-Base reference only.
    """
    task = get_task(task_name)
    if n_shot <= 0 and not raw:
        return rows
    pool = shot_pool(task_name, seed=seed) if n_shot > 0 else []
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        shots = random.Random(seed * 100_003 + i).sample(pool, n_shot) if n_shot > 0 else []
        if raw:
            row["raw_prompt"] = RawPrompt(task.raw_prompt(row, shots))
        else:
            msgs = list(row["messages"])
            system = [m for m in msgs if m["role"] == "system"][:1]
            query = msgs[-1]
            turns = []
            for s in shots:
                turns.append(s["messages"][-1])
                turns.append({"role": "assistant", "content": task.sft_target(s)})
            row["messages"] = system + turns + [query]
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Budget subsampling
# --------------------------------------------------------------------------- #

def _label_key(examples: List[Dict]) -> Optional[str]:
    return "label" if examples and "label" in examples[0] else None


def _select_per_class(by_label: dict, k: int, rng: random.Random) -> List[int]:
    selected = []
    for label, indices in by_label.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        if len(shuffled) < k:
            print(f"  warning: class {label} has {len(shuffled)} valid samples (< {k})")
        selected.extend(shuffled[: min(k, len(shuffled))])
    return sorted(selected)


def _stratified_fraction(by_label: dict, fraction: float, rng: random.Random) -> List[int]:
    selected = []
    for indices in by_label.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = max(1, round(len(indices) * fraction)) if fraction < 1.0 else len(indices)
        selected.extend(shuffled[:n])
    return sorted(selected)


def apply_valid_budget(
    examples: List[Dict],
    budget,
    per_class_low: int = 10,
    medium_fraction: float = 0.5,
    flat_low: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict], int, int]:
    """Subsample the valid pool according to the budget.

    Stratifies per-class when a ``label`` field is present (bank77, ddxplus,
    usefulness_judge); falls back to a flat random sample when it isn't
    (ifeval has no discrete label to stratify on).
    """
    mode = parse_budget(budget)
    if mode == "full":
        return examples, len(examples), len(examples)

    rng = random.Random(seed)
    label_key = _label_key(examples)

    if label_key is not None:
        by_label = defaultdict(list)
        for i, ex in enumerate(examples):
            by_label[ex[label_key]].append(i)
        if isinstance(mode, int):
            selected = _select_per_class(by_label, mode, rng)
        elif mode == "low":
            selected = _select_per_class(by_label, per_class_low, rng)
        elif mode == "medium":
            selected = _stratified_fraction(by_label, medium_fraction, rng)
        else:
            raise ValueError(f"Unknown budget mode '{mode}'.")
    else:
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        if isinstance(mode, int):
            n = min(mode, len(indices))
        elif mode == "low":
            n = min(flat_low, len(indices))
        elif mode == "medium":
            n = max(1, round(len(indices) * medium_fraction))
        else:
            raise ValueError(f"Unknown budget mode '{mode}'.")
        selected = sorted(indices[:n])

    return [examples[i] for i in selected], len(selected), len(examples)


# --------------------------------------------------------------------------- #
# SFT (prompt, target) pairs + tokenized collation for weight_gd / subspace_gd
# --------------------------------------------------------------------------- #

def build_sft_examples(task_name: str, examples: List[Dict]) -> List[Dict]:
    """{'input': str|messages, 'target': str} pairs for GD strategies, built
    from the same valid-pool rows used for coeff_search/eval (dropping any
    with no canonical target, i.e. task.sft_target returns None). Every task
    reuses its valid pool this way -- including ifeval, whose valid pool is
    itself sourced from argilla/ifeval-like-data (see tasks/ifeval.py) instead
    of its own eval set precisely so this generic path works uniformly."""
    task = get_task(task_name)
    out = []
    for row in examples:
        target = task.sft_target(row)
        if target is None:
            continue
        out.append({"input": task.format_example(row), "target": target})
    return out


class _ListDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _build_labeled_example(
    tokenizer, ex: Dict, max_length: int, enable_thinking: bool = True,
) -> Tuple[List[int], List[int]]:
    prompt_text = format_prompt(tokenizer, ex["input"], enable_thinking=enable_thinking)
    target_text = ex["target"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > max_length:
        keep_prompt = max(0, max_length - len(target_ids))
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt else []
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return input_ids, labels


def _collate_sft(
    batch: List[Dict], tokenizer, max_length: int, enable_thinking: bool = True,
) -> Dict[str, torch.Tensor]:
    pairs = [_build_labeled_example(tokenizer, ex, max_length, enable_thinking) for ex in batch]
    max_len = max(len(ids) for ids, _ in pairs)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in pairs:
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append([1] * len(ids) + [0] * pad_n)
        labels.append(labs + [-100] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def get_sft_dataloader(
    sft_examples: List[Dict],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    shuffle: bool = True,
    enable_thinking: bool = True,
) -> DataLoader:
    """``enable_thinking`` should match the ``ENABLE_THINKING`` of the task
    these examples came from (see src/tasks/*.py) so the training prompt and
    the eval-time generation prompt stay consistent."""
    import functools

    collate = functools.partial(
        _collate_sft, tokenizer=tokenizer, max_length=max_length, enable_thinking=enable_thinking,
    )
    return DataLoader(_ListDataset(sft_examples), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# --------------------------------------------------------------------------- #
# Eval-time batching (generation, no tokenization needed until models.generate)
# --------------------------------------------------------------------------- #

def chunked(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
