#!/usr/bin/env bash
# Weight GD from true-random vision weights (ViT-B/16). Writes a side results
# root so main_exp is untouched. METHOD=ta is a dummy (merge is unused).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-16"
METHOD="ta"
STRATEGIES="weight_gd"
WEIGHT_GD_INITS="random_42,random_43,random_44"
SUBSPACE_GD_INITS="pretrained"
BATCH_SIZE=8
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/weight_gd_random"
PLOT=0
WANDB=1
run_merge
