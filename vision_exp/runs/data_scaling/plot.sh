#!/usr/bin/env bash
# Data-scaling figures: one pair of arches per method (shared legend).
# Plotting 9-task setting
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
OUT_DIR="${OUT_DIR:-results/main_exp}"
METRIC="${PLOT_METRIC:-test_avg_acc}"
ARCHES="vit-b-32,vit-b-16"
METHODS=(ta ties)

tasks_csv="$(join_by , "${TASKS[@]}")"
for METHOD in "${METHODS[@]}"; do
    echo "======================================================================"
    echo "plot  arch=$ARCHES  method=$METHOD  tasks=$tasks_csv"
    echo "======================================================================"
    (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot.py data_scaling \
        --results-dir "$OUT_DIR" --tasks "$tasks_csv" \
        --arch "$ARCHES" --method "$METHOD" --metric "$METRIC")
done
