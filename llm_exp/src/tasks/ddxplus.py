"""DDXPlus medical diagnosis classification.

Patient profile -> one of 49 diagnoses. Ported from
model-merge-transfer/llm_evals/ddxplus_eval.py.

DDXPlus ships a native ``validate``/``test`` split pair, but the checkpoint's
own fine-tuning already used ``validate`` -- using it here would evaluate
merge quality on data the model has memorized. So both valid and test for
this task come from ``test`` only, via the same fixed-seed 90/10 split the
other pool-based tasks use (keeps GD's tuning-data share at 1:9, so more data
doesn't quietly favor weight_gd/subspace_gd over coeff_search). ``validate``
is never loaded.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional

from datasets import load_dataset

EVAL_SPLIT_SPEC = {"pool": "test"}

# score() only reads response_str.split("\n")[0] -- Qwen3's default chat
# template invites the model to open a <think>...</think> block first, and
# when it does the answer is never the first line, so this task is
# unscorable (not just slower) unless thinking is disabled.
ENABLE_THINKING = False

# gold "<label>. <label_text>" completions top out at 12 tokens; actual
# base-model generations with ENABLE_THINKING=False above topped out at 12
# tokens over a 20-example sample too. 16 leaves headroom without paying for
# a ~256-token decode on every generation call.
MAX_NEW_TOKENS = 16

LABEL2TEXT = {
    0: "Acute COPD exacerbation / infection",
    1: "Acute dystonic reactions",
    2: "Acute laryngitis",
    3: "Acute otitis media",
    4: "Acute pulmonary edema",
    5: "Acute rhinosinusitis",
    6: "Allergic sinusitis",
    7: "Anaphylaxis",
    8: "Anemia",
    9: "Atrial fibrillation",
    10: "Boerhaave",
    11: "Bronchiectasis",
    12: "Bronchiolitis",
    13: "Bronchitis",
    14: "Bronchospasm / acute asthma exacerbation",
    15: "Chagas",
    16: "Chronic rhinosinusitis",
    17: "Cluster headache",
    18: "Croup",
    19: "Ebola",
    20: "Epiglottitis",
    21: "GERD",
    22: "Guillain-Barré syndrome",
    23: "HIV (initial infection)",
    24: "Influenza",
    25: "Inguinal hernia",
    26: "Larygospasm",
    27: "Localized edema",
    28: "Myasthenia gravis",
    29: "Myocarditis",
    30: "PSVT",
    31: "Pancreatic neoplasm",
    32: "Panic attack",
    33: "Pericarditis",
    34: "Pneumonia",
    35: "Possible NSTEMI / STEMI",
    36: "Pulmonary embolism",
    37: "Pulmonary neoplasm",
    38: "SLE",
    39: "Sarcoidosis",
    40: "Scombroid food poisoning",
    41: "Spontaneous pneumothorax",
    42: "Spontaneous rib fracture",
    43: "Stable angina",
    44: "Tuberculosis",
    45: "URTI",
    46: "Unstable angina",
    47: "Viral pharyngitis",
    48: "Whooping cough",
}
TEXT2LABEL = {v.lower(): k for k, v in LABEL2TEXT.items()}
_OPTION_TEXT = "\n".join(f"{k}. {v}" for k, v in LABEL2TEXT.items())


def load_split(hf_split: str) -> List[Dict]:
    ds = load_dataset("appier-ai-research/StreamBench", "ddxplus", split=hf_split)
    out = []
    for row in ds:
        question = textwrap.dedent(f"""\
        Act as a medical doctor and diagnose the patient based on the provided patient profile.
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {_OPTION_TEXT}
        PATIENT PROFILE:
        {row['PATIENT_PROFILE']}
        Now provide the diagnosis for the patient in the following format: <number>. <diagnosis>""")
        label_text = row["PATHOLOGY"].lower().strip()
        assert label_text in TEXT2LABEL
        out.append({
            "prompt": question,
            "label": TEXT2LABEL[label_text],
            "label_text": row["PATHOLOGY"],
        })
    return out


def format_example(row: Dict):
    return row["prompt"]


def score(response_str: str, row: Dict) -> bool:
    try:
        first_line = response_str.split("\n")[0]
        numeric, category = first_line.split(".", maxsplit=1)
        numeric = int(numeric.strip())
        if numeric == int(row["label"]):
            return True
        return category.lower().strip() == row["label_text"].lower().strip()
    except (ValueError, IndexError, KeyError, AttributeError):
        return False


def sft_target(row: Dict) -> Optional[str]:
    return f"{row['label']}. {row['label_text']}"
