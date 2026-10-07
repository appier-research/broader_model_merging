#!/usr/bin/env bash
# Train the single-task checkpoints (task vectors) consumed by the merge runs.
# Edit ARCH / MODEL / TASKS as needed. Run from anywhere.
set -euo pipefail

VISION_EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$VISION_EXP_ROOT"

STEPS=10000
LR=1e-5
BATCH_SIZE=32
EVAL_STEPS=1000
SAVE_STEPS=1000
WANDB_PROJECT="${WANDB_PROJECT:-vision-exp-merging}"
ARCH="vit-b-32"
MODEL="openai/clip-vit-base-patch32"

TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

for TASK in "${TASKS[@]}"; do
    OUTPUT_DIR="checkpoints/task_vectors/${TASK}_${ARCH}"
    echo "=== Training single-task: ${TASK} -> ${OUTPUT_DIR} ==="
    WANDB_OPTS=()
    [ -n "$WANDB_PROJECT" ] && WANDB_OPTS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "${TASK}_${ARCH}")
    "$PYTHON" scripts/train_single.py \
        --task "$TASK" \
        --model "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --steps "$STEPS" \
        --eval-steps "$EVAL_STEPS" \
        --save-steps "$SAVE_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LR" \
        --warmup-ratio 0.1 \
        "${WANDB_OPTS[@]}"
done
