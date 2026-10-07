#!/usr/bin/env bash
# Draw score vs cost. Override rows with ROWS=memory,time,flops (default all three)
# and the strategies drawn with STRATEGIES=... (default leaves subspace_gd off).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ROWS="${ROWS:-memory,time,flops}"
STRATEGIES="${STRATEGIES:-coeff_search,bo_search,weight_gd_lora}"
METHOD="${METHOD:-ta}"
OUT_DIR="${OUT_DIR:-results/cost_estimation}"
ACC_DIR="${ACC_DIR:-results}"

(cd "$LLM_EXP_ROOT" && "$PYTHON" scripts/plots/plot_cost.py \
    --results-dir "$OUT_DIR" \
    --acc-dir "$ACC_DIR" \
    --method "$METHOD" \
    --rows "$ROWS" \
    --strategies "$STRATEGIES")
