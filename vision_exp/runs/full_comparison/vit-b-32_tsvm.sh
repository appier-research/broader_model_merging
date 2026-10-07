#!/usr/bin/env bash
# Main Result 1 (full data): ViT-B/32 x TSV-M.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="tsvm"
STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
WEIGHT_GD_INITS="merged,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32                                # vit-b-32: default
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
OUT_DIR="results/main_exp"
EXTRA_ARGS="--save-checkpoints"
run_merge
