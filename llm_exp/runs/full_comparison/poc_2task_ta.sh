#!/usr/bin/env bash
# Quick POC: 2 tasks, Qwen3-0.6B x Task Arithmetic, low budget, coeff_search only.
# Fast end-to-end smoke test before launching the full 4-task runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
METHOD="ta"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="pretrained,avg,merged,coeff_best"
SUBSPACE_GD_INITS="pretrained,avg,merged,coeff_best"
BUDGETS="low"
TASKS=(bank77 ddxplus)

run_eval
