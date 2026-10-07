#!/usr/bin/env bash
# Qwen3-0.6B x DARE, all 4 tasks, full budget.
#
# Learning rates: weight_gd 3e-5, weight_gd_lora 3e-4.
#
# By default this fills in only the cells missing from the unit JSON. To redo
# an ENTIRE strategy: FORCE=weight_gd,weight_gd_lora ./qwen3-0.6b_dare.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
METHOD="dare"
STRATEGIES="coeff_search,weight_gd,weight_gd_lora,subspace_gd,baselines"
WEIGHT_GD_INITS="avg,merged,coeff_best"
WEIGHT_GD_LORA_INITS="avg,merged,coeff_best"
SUBSPACE_GD_INITS="pretrained,avg,merged,coeff_best"
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Held-out-selected learning rates (see the header note).
GD_LR="${GD_LR:-3e-5}"
LORA_LR="${LORA_LR:-3e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Checkpoints are large; scores do not depend on them being written.
SAVE_CKPT_ARG=""
if [ "${SAVE_CHECKPOINTS:-1}" != "0" ]; then SAVE_CKPT_ARG="--save-checkpoints"; fi

EXTRA_ARGS="$SAVE_CKPT_ARG --dare-drop-rate 0.5 --dare-seed 42 --gd-batch-size 1 --gd-grad-accum-steps 4 \
  --subspace-batch-size 1 --subspace-grad-accum-steps 4 \
  --gd-lr $GD_LR --lora-lr $LORA_LR --lora-r $LORA_R --lora-alpha $LORA_ALPHA"
run_eval
