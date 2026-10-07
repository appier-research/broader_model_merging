#!/usr/bin/env bash
# Main Result 1 (full data): ViT-B/16 x TIES.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-16"
METHOD="ties"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="merged,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32                                # vit-b-16: ~4x more attention tokens than b-32
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/ties_coeff_grid"
EXTRA_ARGS="--lambda-min 0.5 --lambda-max 0.9 --lambda-steps 41"
run_merge
