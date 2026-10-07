#!/usr/bin/env bash
# OOD generalization: qwen3-1.7b x ta. Scores the finished main unit's cells
# on the held-out math (gsm8k) and coding (mbpp) tasks -- no GD rerun; GD cells
# load their saved checkpoints (ARCH=qwen3-1.7b METHOD=ta bash admin/download_finetuned.sh).
# UNSEEN_TEST_SAMPLES caps each unseen pool (gsm8k 1319 / mbpp 450 full; 512-token decodes).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
METHOD="ta"
TASKS=(bank77 ddxplus ifeval usefulness_judge)
UNSEEN_TASKS=(gsm8k mbpp)
UNSEEN_TEST_SAMPLES="${UNSEEN_TEST_SAMPLES:-300}"
OUT_DIR="${OUT_DIR:-results/ood_generalization}"
run_ood_eval
