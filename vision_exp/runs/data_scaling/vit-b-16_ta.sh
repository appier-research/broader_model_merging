#!/usr/bin/env bash
# Main Result 2 (data scaling): ViT-B/16 x Task Arithmetic across valid budgets.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-16"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
WEIGHT_GD_INITS="pretrained,avg,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=8                                # vit-b-16: ~4x more attention tokens than b-32
BUDGETS="full,64,32,16,8,4,2,1"                       # per-class counts + full
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/main_exp"
SEED="42,43,44"
EXTRA_ARGS="--save-checkpoints"                # store weight_gd / subspace_gd finetuned weights

run_merge
