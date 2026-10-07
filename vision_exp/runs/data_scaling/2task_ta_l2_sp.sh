#!/usr/bin/env bash
# 2-task data scaling: ViT-B/32 x TA, weight GD + L2-SP.
# coeff_search / subspace_gd / unregularized GD are reused from results/main_exp.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
STRATEGIES="weight_gd"
WEIGHT_GD_INITS="avg"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32
BUDGETS="full" #"64,32,16,8,4,2,1" 
SEED=42  #"42,43,44"
PLOT=0
WANDB_PROJECT="mm_2_task"

GD_L2_SP=3                    # set before running, e.g. 0.1
# comment out a line to skip that pair
PAIRS=(
    "eurosat gtsrb"
    "fer2013 mnist"
)

OUT_DIR="results/weight_gd_l2_sp/l2_sp_${GD_L2_SP}"

for pair in "${PAIRS[@]}"; do
    # shellcheck disable=SC2206
    TASKS=($pair)
    tasks_csv="$(join_by , "${TASKS[@]}")"
    WANDB_GROUP="l2_sp_${GD_L2_SP}/${ARCH}/${METHOD}/${TASKS[0]}-${TASKS[1]}"
    run_merge
    (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot.py data_scaling \
        --results-dir results/main_exp --l2-sp-results-dir "$OUT_DIR" \
        --tasks "$tasks_csv" --arch "$ARCH" --method "$METHOD")
done
