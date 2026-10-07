#!/usr/bin/env bash
# Directional sampling: Llama-3.2-1B-Instruct x Task Arithmetic, 4 tasks (full).
# coeff_best lambda* and the density threshold are read from $COEFF_SOURCE_DIR
# (the main results root), same unit name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
METHOD="ta"
STRATEGIES="directional_sampling"
WEIGHT_GD_INITS="avg"                        # required by _lib; unused
WEIGHT_GD_LORA_INITS="avg"                   # required by _lib; unused
SUBSPACE_GD_INITS="coeff_best"               # required by _lib; unused
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

OUT_DIR="${OUT_DIR:-results/directional_sampling}"
COEFF_SOURCE_DIR="${COEFF_SOURCE_DIR:-results}"

DS_INIT="avg"                 # pretrained | avg | merged | coeff_best
DS_ALPHA=0.1
DS_BETA=0                     # >0 adds the -g_perp axis (costs one backward pass over the SFT pools)
DS_SAMPLES=64
DS_SELECT_METRIC="loss"       # loss (cheap forward per draw) | score (generate+score per draw)

EXTRA_ARGS="--gen-batch-size 32 --loss-batch-size 4"
run_eval
