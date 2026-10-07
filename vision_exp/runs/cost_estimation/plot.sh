#!/usr/bin/env bash
# Draw accuracy vs cost. Override rows with ROWS=memory,time,flops (default all three).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ROWS="${ROWS:-memory,time,flops}"
METHOD="${METHOD:-ta}"
OUT_DIR="${OUT_DIR:-results/cost_estimation}"
ACC_DIR="${ACC_DIR:-results/main_exp}"

(cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot_cost.py \
    --results-dir "$OUT_DIR" \
    --acc-dir "$ACC_DIR" \
    --method "$METHOD" \
    --rows "$ROWS")
