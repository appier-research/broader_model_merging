#!/usr/bin/env bash
# 2-task data scaling: ViT-B/32 x TA, three selection seeds.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd"
WEIGHT_GD_INITS="avg"
SUBSPACE_GD_INITS="coeff_best"
BATCH_SIZE=32
BUDGETS="64,32,16,8,4,2,1"
TASKS=(eurosat gtsrb)

OUT_DIR="results/main_exp"
SEED="42,43,44"
PLOT=1

run_merge
