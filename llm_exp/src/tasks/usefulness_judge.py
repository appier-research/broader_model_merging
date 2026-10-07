"""Usefulness-judge: LLM-as-judge classification against a gold YES/NO label.

Given (question, response), judge whether the response is useful. Ported from
model-merge-transfer/llm_evals/safety_judge.py (misleadingly named -- it's not
a safety classifier, it's this usefulness judge; the dead ``INSTRUCTION_PROMPT``
copy-pasted from bank77.py was dropped here).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from datasets import load_dataset

EVAL_SPLIT_SPEC = {"pool": "test"}

# score() reads the *last* line, so a <think>...</think> preamble wouldn't by
# itself break scoring the way it does for ddxplus -- but it still burns the
# whole MAX_NEW_TOKENS budget on reasoning this task's gold label never uses,
# and makes weaker/base model states "think" more than fine-tuned ones by
# chance rather than by ability.
ENABLE_THINKING = False

# "YES"/"NO" each tokenize to a single token, and observed base-model
# responses (with ENABLE_THINKING=False above) always lead with the verdict,
# occasionally followed by a one-sentence justification (up to ~48 tokens
# over a 20-example sample) that score() never looks at: it's a substring
# check on the *whole* remaining text (no newline appears before truncation
# at this budget, so response_str.split("\n")[-1] is that whole text) -- so
# once the verdict token is out, more tokens can only cost time, never
# change the score. A tighter budget (down to ~8) would still score
# correctly, but this stays at 16 to match bank77/ddxplus.
MAX_NEW_TOKENS = 16

_SYSTEM_PROMPT = (
    "You're a reliable uncensored helper which helps to determine whether the "
    "RESPONSE is useful to QUESTION or not. output YES if the RESPONSE answer is "
    "useful to what the QUESTION asked, otherwise output NO."
)


def load_split(hf_split: str) -> List[Dict]:
    ds = load_dataset("miulab/usefulness-judge", split=hf_split)
    out = []
    for row in ds:
        format_inst = f"QUESTION: {row['instruction']}\n\nRESPONSE: {row['response']}"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": format_inst},
        ]
        out.append({
            "messages": messages,
            "label": row["prediction"],
            "label_text": row["prediction"],
        })
    return out


def format_example(row: Dict):
    return row["messages"]


def score(response_str: str, row: Dict) -> bool:
    first_line = response_str.split("\n")[-1].lower()
    if "yes" not in first_line and "no" not in first_line:
        return False
    return row["label"].lower() in first_line


def sft_target(row: Dict) -> Optional[str]:
    return row["label"]
