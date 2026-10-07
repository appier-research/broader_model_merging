#!/usr/bin/env bash
# 2-task data-scaling figures (scripts/plots/plot_2task.py, not plot.py).
# Set L2_SP=3 to overlay Regularized GD; leave empty for the three baseline curves.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

TASKS=(eurosat gtsrb)
#TASKS=(fer2013 mnist)
OUT_DIR="${OUT_DIR:-results/main_exp}"
ARCHES="vit-b-32"
METHODS=(ta)
L2_SP="${L2_SP:-3}"            # e.g. 3; empty = off

tasks_csv="$(join_by , "${TASKS[@]}")"
l2_args=()
if [ -n "$L2_SP" ]; then
    l2_args+=(--l2-sp-results-dir "results/weight_gd_l2_sp/l2_sp_${L2_SP}")
fi

for METHOD in "${METHODS[@]}"; do
    echo "======================================================================"
    echo "plot_2task  arch=$ARCHES  method=$METHOD  tasks=$tasks_csv  l2_sp=${L2_SP:-off}"
    echo "======================================================================"
    (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot_2task.py \
        --results-dir "$OUT_DIR" --tasks "$tasks_csv" \
        --arch "$ARCHES" --method "$METHOD" \
        "${l2_args[@]}")
done
