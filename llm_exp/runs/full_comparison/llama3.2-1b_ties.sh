#!/usr/bin/env bash
# Llama-3.2-1B-Instruct x TIES, all 4 tasks, full budget.
#
# Checkpoints: ARCH=llama3.2-1b bash admin/download_checkpoints.sh
# See llama3.2-1b_ta.sh on why BASE_MODEL is the Instruct model.
#
# Learning rates: weight_gd 1e-4, weight_gd_lora 1e-3.
#
# By default this fills in only the cells missing from the unit JSON. To redo
# an ENTIRE strategy: FORCE=weight_gd,weight_gd_lora ./llama3.2-1b_ties.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
METHOD="ties"
# subspace_gd dropped at 1.7b+: too slow to finish. Re-add here to restore it.
STRATEGIES="coeff_search,weight_gd,weight_gd_lora,baselines"
WEIGHT_GD_INITS="pretrained,avg,merged,coeff_best"
WEIGHT_GD_LORA_INITS="pretrained,avg,merged,coeff_best"
SUBSPACE_GD_INITS="pretrained,avg,merged,coeff_best"
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Held-out-selected learning rates (see the header note).
GD_LR="${GD_LR:-1e-4}"
LORA_LR="${LORA_LR:-1e-3}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Checkpoints are large; scores do not depend on them being written.
SAVE_CKPT_ARG=""
if [ "${SAVE_CHECKPOINTS:-1}" != "0" ]; then SAVE_CKPT_ARG="--save-checkpoints"; fi

EXTRA_ARGS="$SAVE_CKPT_ARG --ties-density 0.2 --gd-batch-size 1 --gd-grad-accum-steps 4 --subspace-batch-size 1 --subspace-grad-accum-steps 4 \
  --gd-lr $GD_LR --lora-lr $LORA_LR --lora-r $LORA_R --lora-alpha $LORA_ALPHA"
run_eval
