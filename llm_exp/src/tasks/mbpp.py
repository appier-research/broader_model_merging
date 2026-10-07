"""MBPP python programming (laylarsssss/FusedMBPP, 500 test problems).

Ported from model-merge-transfer/llm_evals/mbpp.py with one deliberate
change: that version fetched its prompts from, and submitted completions to,
a running sandbox-fusion docker server. Here the prompt is built locally in
the standard MBPP form (task description + the assert tests the code must
pass) and ``score`` executes ``completion + tests`` in a python subprocess
with a timeout -- no docker, no network, same pass/fail semantics (all
``test_list`` asserts must pass). Used as an **unseen / out-of-domain** task
in runs/ood_generalization: no task vector, test pool only.

Model-generated code is executed on this machine (in a subprocess, in a temp
cwd, with a wall-clock timeout). That is how MBPP/HumanEval are normally
scored; keep it in mind if pointing this at an untrusted model.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

from datasets import load_dataset

# FusedMBPP ships a single 500-row ``test`` split; the shared fixed-seed 90/10
# split gives the same 450-row test pool whether mbpp is seen or unseen, so
# numbers stay comparable if it ever becomes a merged task.
EVAL_SPLIT_SPEC = {"pool": "test"}

# Few-shot (see data.apply_few_shot): shots come from the 50-row valid pool
# (never the 450-row test pool); 3-shot is the standard MBPP budget.
SHOT_SPLIT = "valid"
DEFAULT_N_SHOT = 3

ENABLE_THINKING = False

# A full MBPP solution is a short function; the code block plus any preamble
# fits in 512 with room to spare (ifeval: 256, gsm8k: 512).
MAX_NEW_TOKENS = 512

# Wall-clock limit per program (sandbox-fusion's run_timeout was 20s; MBPP
# solutions run in milliseconds, so 10s only ever catches infinite loops).
RUN_TIMEOUT_S = 10

SYSTEM_PROMPT = (
    "You're now a expert programmer which write code immediately in "
    "```python\n<your code here>\n``` code block"
)

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _literal(x):
    """FusedMBPP stores its list/dict columns as python-literal strings."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def _user_prompt(content: str, tests: List[str]) -> str:
    return (
        f"You are an expert Python programmer, and here is your task: {content} "
        f"Your code should pass these tests:\n\n" + "\n".join(tests) + "\n"
    )


def load_split(hf_split: str) -> List[Dict]:
    ds = load_dataset("laylarsssss/FusedMBPP", split=hf_split)
    rows = []
    for row in ds:
        tests = list(_literal(row["test_list"]))
        labels = _literal(row["labels"]) or {}
        setup = labels.get("test_setup_code", "") if isinstance(labels, dict) else ""
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(row["content"], tests)},
            ],
            "id": row["id"],
            "tests": tests,
            "setup_code": setup or "",
            "canonical_solution": row["canonical_solution"],
        })
    return rows


def format_example(row: Dict):
    return row.get("raw_prompt") or row["messages"]


def raw_prompt(row: Dict, shots: List[Dict]) -> str:
    """Plain completion prompt for a *-Base model: task + tests, then the
    solution inside a ```python fence; the query block opens the fence so the
    model continues with code (score() cuts at the closing fence)."""
    from ..data import RAW_SHOT_SEP
    blocks = [f"{s['messages'][-1]['content']}\n{sft_target(s)}" for s in shots]
    blocks.append(f"{row['messages'][-1]['content']}\n```python\n")
    return RAW_SHOT_SEP.join(blocks)


def extract_code(response_str: str) -> str:
    m = _CODE_BLOCK.findall(response_str)
    return m[0].strip() if m else response_str.strip()


def run_program(code: str, setup_code: str, tests: List[str], timeout_s: int = RUN_TIMEOUT_S) -> bool:
    program = "\n\n".join(p for p in (setup_code, code, "\n".join(tests)) if p)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path], cwd=tmp, capture_output=True, timeout=timeout_s,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


def score(response_str: str, row: Dict) -> bool:
    if row.get("raw_prompt"):
        # raw completion continues an open ```python fence: keep up to the closing fence
        from ..data import RAW_SHOT_SEP
        code = response_str.split("```")[0].split(RAW_SHOT_SEP.strip())[0].strip()
        return run_program(code, row["setup_code"], row["tests"])
    return run_program(extract_code(response_str), row["setup_code"], row["tests"])


def sft_target(row: Dict) -> Optional[str]:
    return f"```python\n{row['canonical_solution'].strip()}\n```"
