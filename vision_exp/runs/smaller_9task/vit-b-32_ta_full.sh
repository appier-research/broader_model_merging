#!/usr/bin/env bash
# Full valid pool for class seeds 0, 1, and 2, one after another.
# Full does not draw a data subset, so the image seed stays 42.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd"
WEIGHT_GD_INITS="avg"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

NUM_CLASSES=53
CLASS_SEEDS=(0 1 2)
SEED=42
PLOT=0
WANDB_PROJECT="smaller_9task"

for s in "${CLASS_SEEDS[@]}"; do
    CLASS_SEED="$s"
    OUT_DIR="results/smaller_9task_${NUM_CLASSES}/cseed${s}"
    run_merge
done
