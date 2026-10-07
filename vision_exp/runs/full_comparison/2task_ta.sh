#!/usr/bin/env bash
# Quick POC: 2 tasks, ViT-B/32 x Task Arithmetic, full data, all strategies.
# Fast end-to-end smoke test before launching the 9-task runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
WEIGHT_GD_INITS="pretrained,avg,merged,coeff_best"
SUBSPACE_GD_INITS="pretrained,avg,merged,coeff_best"
BATCH_SIZE=32                                # vit-b-32: default
BUDGETS="full"
TASKS=(eurosat gtsrb)

OUT_DIR="results/main_exp"
EXTRA_ARGS="--save-checkpoints" 
run_merge
