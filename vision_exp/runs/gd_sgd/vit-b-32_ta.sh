#!/usr/bin/env bash
# Weight GD (SGD, momentum 0) for the TA columns of the SGD vs AdamW table.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
STRATEGIES="weight_gd"
WEIGHT_GD_INITS="avg"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32
BUDGETS="1,2,4,8,16,32,64,full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
SEED=42

GD_OPTIMIZER=sgd
GD_MOMENTUM=0
GD_LR="1e-3"
GD_EPOCHS=10
GD_PATIENCE=0

OUT_DIR="results/weight_gd_sgd/m${GD_MOMENTUM}"
PLOT=0
WANDB=0

run_merge
