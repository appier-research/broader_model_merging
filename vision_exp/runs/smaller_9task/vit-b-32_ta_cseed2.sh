#!/usr/bin/env bash
# 9-task data scaling, class seed 2. Data-selection seeds match poc_2task_ta.sh.
# Run cseed0/1/2 at the same time; each writes its own results root.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd"
WEIGHT_GD_INITS="avg"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32
BUDGETS="64,32,16,8,4,2,1"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

NUM_CLASSES=53
CLASS_SEED=2
SEED="42,43,44"
PLOT=0
WANDB_PROJECT="smaller_9task"
OUT_DIR="results/smaller_9task_${NUM_CLASSES}/cseed${CLASS_SEED}"

run_merge
