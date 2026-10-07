#!/usr/bin/env bash
# Cost estimation: 9-task TA at full budget, three CLIP sizes on one GPU.
# One warmed-up action per strategy, scaled by 11 / 10 / 20.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

METHOD="ta"
STRATEGIES="${STRATEGIES:-coeff_search,weight_gd,subspace_gd}"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
OUT_DIR="${OUT_DIR:-results/cost_estimation}"
COEFF_SOURCE_DIR="${COEFF_SOURCE_DIR:-results/main_exp}"
PLOT="${PLOT:-0}"

tasks_csv="$(join_by , "${TASKS[@]}")"

for spec in "vit-l-14:2" "vit-b-32:32" "vit-b-16:8"; do
    ARCH="${spec%%:*}"
    BATCH_SIZE="${spec##*:}"
    base_model="$(arch_base_model "$ARCH")"
    echo "======================================================================"
    echo "cost_estimation  method=$METHOD  arch=$ARCH  batch=$BATCH_SIZE"
    echo "======================================================================"
    (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/profile_cost.py \
        --base-model "$base_model" --arch "$ARCH" --method "$METHOD" \
        --tasks "$tasks_csv" \
        --task-vectors-dir "${TASK_VECTORS_DIR:-checkpoints/task_vectors}" \
        --budget full --seed 42 \
        --strategies "$STRATEGIES" \
        --batch-size "$BATCH_SIZE" \
        --out-dir "$OUT_DIR" \
        --coeff-source-dir "$COEFF_SOURCE_DIR" \
        ${EXTRA_ARGS:-})
done

if [ "$PLOT" = "1" ]; then
    bash "$(dirname "${BASH_SOURCE[0]}")/plot.sh"
fi
