#!/usr/bin/env bash
# Main Result 1 (full data): ViT-L/14 x TSV-M coefficient grid search.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-l-14"
METHOD="tsvm"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="merged,coeff_best"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=4                                 # vit-l-14: larger model + more tokens
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

OUT_DIR="results/tsvm_coeff_grid"
# Coarse 0.0-1.0 sweep put the tsvm optimum at 0.7 (b-32), 0.8 (b-16),
# 0.9 (l-14) -- higher than ties -- so refine over 0.6-1.0. Same 0.01 interval
# as the ties grid; the upper edge covers an l-14 peak above 0.9.
EXTRA_ARGS="--lambda-min 0.6 --lambda-max 1.0 --lambda-steps 41"
run_merge
