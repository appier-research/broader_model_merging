#!/usr/bin/env bash
# Directional sampling: 9-task ViT-B/32 x TIES (full).
# coeff_best λ* is read from results/main_exp (same unit name).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ties"
STRATEGIES="directional_sampling"
WEIGHT_GD_INITS="avg"                        # required by _lib; unused
SUBSPACE_GD_INITS="coeff_best"               # required by _lib; unused
BATCH_SIZE=128
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
SEED=42
PLOT=0
OUT_DIR="results/directional_sampling"
WANDB_PROJECT="mm_directional_sampling"

DS_INIT="coeff_best"                         # pretrained | avg | merged | coeff_best
DS_ALPHA=0.1
DS_BETA=0.05
DS_SAMPLES=128

run_merge
