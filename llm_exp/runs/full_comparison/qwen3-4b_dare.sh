#!/usr/bin/env bash
# Qwen3-4B x DARE, all 4 tasks, full budget.
#
# Checkpoints: ARCH=qwen3-4b bash admin/download_checkpoints.sh
#
# DARE is mask + rescale only -- no SVD, so it has none of TSV-M's one-time
# merge cost.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-4b"
BASE_MODEL="Qwen/Qwen3-4B-Base"
METHOD="dare"
# subspace_gd dropped at 1.7b+: too slow to finish. Re-add here to restore it.
# weight_gd (full fine-tuning) is NOT run at this size: Adam's two fp32 moments
# plus bf16 weights and grads are ~49GiB for 4.41B parameters, which OOMs in
# optimizer.step() on the 31.4GiB cards in this box at any batch size. It is
# replaced by weight_gd_lora, which trains 33M of 4.06B parameters (0.81%) and
# fits comfortably. Restoring weight_gd needs a bigger card, or FSDP/ZeRO or
# 8-bit Adam in src/strategies/weight_gd.py.
STRATEGIES="coeff_search,weight_gd_lora,baselines"
WEIGHT_GD_INITS="pretrained,avg,merged,coeff_best"        # unused while weight_gd is off
WEIGHT_GD_LORA_INITS="pretrained,avg,merged,coeff_best"
SUBSPACE_GD_INITS="pretrained,avg,merged,coeff_best"
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# LoRA hyperparameters. Learning rate: weight_gd_lora 1e-4.
LORA_LR="${LORA_LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Generation batch size. 8 rather than the 32 the 1.7b scripts use: at 4B the
# per-sequence KV cache is roughly double 1.7b's, and generation runs right
# after the merge while the model is resident. Raise it if you have headroom.
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"

# Checkpoints are large; scores do not depend on them being written.
SAVE_CKPT_ARG=""
if [ "${SAVE_CHECKPOINTS:-1}" != "0" ]; then SAVE_CKPT_ARG="--save-checkpoints"; fi

EXTRA_ARGS="$SAVE_CKPT_ARG --gen-batch-size $GEN_BATCH_SIZE --dare-drop-rate 0.5 --dare-seed 42 --gd-batch-size 1 --gd-grad-accum-steps 4 --subspace-batch-size 1 --subspace-grad-accum-steps 4 \
  --lora-lr $LORA_LR --lora-r $LORA_R --lora-alpha $LORA_ALPHA"
run_eval
