"""GSM8K grade-school math (openai/gsm8k, ``main`` config).

Ported from model-merge-transfer/llm_evals/gsm8k.py: same system prompt; the
``score`` takes the last ``\\boxed{}`` when there is one (string match, then
numeric match) and otherwise falls back to the last number in the response
(lm-eval-harness's "flexible-extract"). The fallback matters: even 4-shot,
the 0.6b GD cells reason to the right number and then simply stop without
boxing it (a format habit from their SFT data), so strict boxed-only scoring
mostly measures ifeval-style format compliance rather than arithmetic --
which is what the seen tasks already score. Used as an **unseen /
out-of-domain** task in
runs/ood_generalization -- no task vector is merged for it, so it only ever
needs a test pool. The native valid/test split pair is still declared (and
``sft_target`` provided) so the same module would work as a seen task later.

Unlike the four in-domain tasks (short classification-style answers, 16
tokens), a GSM8K answer is a chain-of-thought ending in \\boxed{...}; the
generation budget is the main cost driver of an OOD eval pass, so
``--unseen-test-samples`` exists to cap the pool.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from datasets import load_dataset

EVAL_SPLIT_SPEC = {"valid": "train", "test": "test"}

# Few-shot (see data.apply_few_shot): shots come from the train split, never
# from test. 4-shot is the usual GSM8K base-model budget; the 0-shot chat
# prompt mostly measures whether a cell still emits \boxed{} at all (format),
# which is what the seen task ifeval already measures.
SHOT_SPLIT = "train"
DEFAULT_N_SHOT = 4

# Same reasoning as the other tasks: thinking is off everywhere in this repo
# (merge quality, not reasoning ability). The system prompt below asks for the
# reasoning in the visible answer instead, with the final number boxed.
ENABLE_THINKING = False

# Chain of thought + \boxed{}: 512 covers the base model's greedy answers
# comfortably (gold rationales are < 200 tokens); ifeval uses 256.
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You're now a MATH genius with deep thought, you will be given a math question, "
    "your task is to reasoning deeply and place your final answer in \\boxed{}"
)

_BOXED = re.compile(r"\\box(?:ed)?\{([^}]+)\}")
# flexible-extract fallback: the last number in the response (1,430 / 6.67 / -3 / $12)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _gold_answer(answer_field: str) -> str:
    return answer_field.split("####")[-1].replace("#", "").strip()


def load_split(hf_split: str) -> List[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split=hf_split)
    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            "question": row["question"],
            "rationale": row["answer"],
            "answer": _gold_answer(row["answer"]),
        }
        for row in ds
    ]


def format_example(row: Dict):
    return row.get("raw_prompt") or row["messages"]


def raw_prompt(row: Dict, shots: List[Dict]) -> str:
    """Plain completion prompt for a *-Base model (no chat template):
    Question/Answer pairs separated by data.RAW_SHOT_SEP, ending at
    "Answer:" so the model continues with the rationale + \\boxed{}."""
    from ..data import RAW_SHOT_SEP
    blocks = [f"Question: {s['question']}\nAnswer: {sft_target(s)}" for s in shots]
    blocks.append(f"Question: {row['question']}\nAnswer:")
    return RAW_SHOT_SEP.join(blocks)


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def score(response_str: str, row: Dict) -> bool:
    if row.get("raw_prompt"):
        # raw completion: stop at the next shot delimiter / next question
        from ..data import RAW_SHOT_SEP
        response_str = response_str.split(RAW_SHOT_SEP.strip())[0].split("\nQuestion:")[0]
    matches = _BOXED.findall(response_str)
    if matches:
        model_answer = matches[-1].strip()
    else:
        nums = _NUMBER.findall(response_str)
        if not nums:
            return False
        model_answer = nums[-1].rstrip(".")
    expected = str(row["answer"]).strip()
    if model_answer.lower() == expected.lower():
        return True
    a, b = _num(model_answer), _num(expected)
    return a is not None and b is not None and abs(a - b) < 1e-6


def sft_target(row: Dict) -> Optional[str]:
    """Gold rationale with the trailing ``#### N`` rewritten as ``\\boxed{N}``
    so the SFT target matches what ``score`` looks for."""
    body = row["rationale"].split("####")[0].rstrip()
    return f"{body}\n\\boxed{{{row['answer']}}}"
