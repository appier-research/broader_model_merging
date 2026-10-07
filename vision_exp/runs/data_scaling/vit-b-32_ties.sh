#!/usr/bin/env bash
# Main Result 2 (data scaling): ViT-B/32 x Task Arithmetic across valid budgets.
# The `full` budget writes to the same JSON as full_comparison/vit-b-32_ta.sh,
# so that shared cell is computed once (whichever runs it first).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ties"
STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
WEIGHT_GD_INITS="merged,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32                                # vit-b-32: default
BUDGETS="64,32,16,8,4,2,1"                     # per-class counts 
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/main_exp"
SEED="42,43,44"
#EXTRA_ARGS="--save-checkpoints" 
run_merge
