"""Google IFEval instruction-following benchmark.

Scoring is ported from model-merge-transfer/llm_evals/if_eval.py: soft/loose
accuracy via lighteval's instruction checkers, no single canonical completion
in google/IFEval itself.

Unlike bank77/ddxplus/usefulness_judge, ifeval's own eval set has no
completion to train on, so it can't supply both a valid pool and an SFT
target from the same source the way the other three tasks do. Instead,
``build_valid_test`` overrides the generic pool/native-split dispatch in
src/data.py: **valid** is sampled from argilla/ifeval-like-data (which does
have gold-ish completions, see ``_load_argilla_pool``), the same source
``sft_target`` reads from -- so weight_gd/subspace_gd train and coeff_search
validates on the same package of data, exactly like the other three tasks
(train == valid, both drawn from the budget-subsampled valid pool; **test**
stays the full, untouched google/IFEval benchmark as the true holdout).
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

import lighteval.tasks.tasks.ifeval.instructions_registry as instructions_registry

_VALID_POOL_SIZE = 60  # 1:9 valid:test ratio against the 541-example test pool, matching the other three tasks --
                       # keeps weight_gd/subspace_gd's tuning-data share consistent so more data doesn't quietly favor GD

# score() runs lighteval's format-sensitive instruction checkers (exact word
# count, no commas, all-caps, N bullet points, ...) directly against the
# response text; a <think>...</think> preamble breaks nearly all of them, the
# same problem as the other three tasks -- this is about evaluating merge
# quality, not reasoning ability, so thinking is off everywhere in this repo.
ENABLE_THINKING = False

# Unlike bank77/ddxplus/usefulness_judge's short classification-style
# answers, ifeval's gold completions are free-form prose satisfying several
# instructions at once, so this keeps the full generation budget.
MAX_NEW_TOKENS = 256


def load_split(hf_split: str) -> List[Dict]:
    """google/IFEval -- used only as the held-out test pool (see build_valid_test)."""
    ds = load_dataset("google/IFEval", split=hf_split)
    out = []
    for row in ds:
        messages = [{"role": "user", "content": row["prompt"]}]
        out.append({"messages": messages, "data": row})
    return out


def _load_argilla_pool() -> List[Dict]:
    """argilla/ifeval-like-data (``filtered`` config), kept only where the
    response satisfies its own instructions (prompt_level_strict_acc). Each
    row carries both the eval shape (``messages``/``data``, for
    format_example/score) and a ``response`` (for sft_target) -- one pool
    doubling as train+valid data, same as the other three tasks."""
    ds = load_dataset("argilla/ifeval-like-data", "filtered", split="train")
    ds = ds.filter(lambda r: bool(r["prompt_level_strict_acc"]))
    rows = []
    for row in ds:
        kwargs = json.loads(row["kwargs"]) if isinstance(row["kwargs"], str) else row["kwargs"]
        rows.append({
            "messages": [{"role": "user", "content": row["prompt"]}],
            "data": {"prompt": row["prompt"], "instruction_id_list": row["instruction_id_list"], "kwargs": kwargs},
            "response": row["response"],
        })
    return rows


def build_valid_test(seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    pool = _load_argilla_pool()
    indices = list(range(len(pool)))
    random.Random(seed).shuffle(indices)
    valid = [pool[i] for i in indices[:_VALID_POOL_SIZE]]
    test = load_split("train")  # full google/IFEval benchmark, never trained/tuned on
    return valid, test


def format_example(row: Dict):
    return row["messages"]


def _preprocess_response(response: str) -> List[str]:
    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()
    revised_response = response.replace("*", "")
    return [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        response_remove_first.replace("*", ""),
        response_remove_last.replace("*", ""),
        response_remove_both.replace("*", ""),
    ]


def score(response_str: str, row: Dict) -> float:
    data = row["data"]
    instruction_list = data["instruction_id_list"]
    all_kwargs = data["kwargs"]
    prompt = data["prompt"]
    all_responses = _preprocess_response(response_str)

    is_following_list_loose = []
    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
        instruction.build_description(**task_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break
        is_following_list_loose.append(is_following)

    if not is_following_list_loose:
        return 0.0
    return sum(is_following_list_loose) / len(is_following_list_loose)


def sft_target(row: Dict) -> Optional[str]:
    return row.get("response")
