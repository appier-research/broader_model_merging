#!/usr/bin/env bash
# Qwen3-4B x TSV-M (Task Singular Vectors), all 4 tasks, full budget.
#
# Checkpoints: ARCH=qwen3-4b bash admin/download_checkpoints.sh
#
# TSV-M's merge is one big SVD pass that runs once per unit, before the first
# lambda of the coefficient sweep. It defaults to the GPU (see
# notes/tsvm_svd_cpu_bottleneck.md): measured on these shapes, ~9 min on the
# GPU against ~81 min on 8 CPU threads.
#
# --embedding-svd-device pins the 151936x2560 embed_tokens / lm_head pair to the
# CPU while the rest of the merge runs on the GPU: those two want ~17.7 GB of
# VRAM each where every other matrix stays under 1.4 GB, and they are only ~5%
# of the merge, so holding them back costs ~4 min and keeps this safe to run on
# a 24 GB card next to a resident model. Set EMBEDDING_SVD_DEVICE=cuda to put
# them on the GPU too, or ="" to just follow --svd-device.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-4b"
BASE_MODEL="Qwen/Qwen3-4B-Base"
METHOD="tsvm"
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

EMBEDDING_SVD_DEVICE="${EMBEDDING_SVD_DEVICE:-cpu}"

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

EMB_SVD_ARG=""
if [ -n "$EMBEDDING_SVD_DEVICE" ]; then EMB_SVD_ARG="--embedding-svd-device $EMBEDDING_SVD_DEVICE"; fi

EXTRA_ARGS="$SAVE_CKPT_ARG --gen-batch-size $GEN_BATCH_SIZE $EMB_SVD_ARG --gd-batch-size 1 --gd-grad-accum-steps 4 --subspace-batch-size 1 --subspace-grad-accum-steps 4 \
  --lora-lr $LORA_LR --lora-r $LORA_R --lora-alpha $LORA_ALPHA"
run_eval
