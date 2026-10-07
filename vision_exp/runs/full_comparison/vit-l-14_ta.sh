#!/usr/bin/env bash
# Main Result 1 (full data): ViT-L/14 x Task Arithmetic.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-l-14"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
WEIGHT_GD_INITS="pretrained,avg,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=4                                 # vit-l-14: larger model + more tokens; conservative for subspace_gd
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/main_exp"
EXTRA_ARGS="--save-checkpoints" 
run_merge
