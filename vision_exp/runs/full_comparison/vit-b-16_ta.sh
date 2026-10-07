#!/usr/bin/env bash
# Main Result 1 (full data): ViT-B/16 x Task Arithmetic.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-16"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
WEIGHT_GD_INITS="pretrained,avg,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=8                                # vit-b-16: ~4x more attention tokens than b-32
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/main_exp"
EXTRA_ARGS="--save-checkpoints" 
run_merge
