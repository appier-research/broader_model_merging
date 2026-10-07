#!/usr/bin/env bash
# Train the single-task checkpoints (task vectors) consumed by the merge runs,
# with axolotl (full fine-tuning; configs in configs/train/<arch>/<task>.yaml).
# Each run writes checkpoints/task_vectors/<task>_<arch>/checkpoint-<step>/,
# the layout src/checkpoints.py resolves. Run from anywhere.
#   ARCHS="qwen3-1.7b" TASKS="bank77 ifeval" bash runs/train/train_all_single_task_checkpoints.sh
set -euo pipefail

LLM_EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LLM_EXP_ROOT"

AXOLOTL="${AXOLOTL:-axolotl}"
ARCHS=(${ARCHS:-qwen3-0.6b qwen3-1.7b qwen3-4b llama3.2-1b})
TASKS=(${TASKS:-bank77 ddxplus ifeval usefulness_judge})

for ARCH in "${ARCHS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "=== Training single-task: ${TASK} x ${ARCH} ==="
        "$AXOLOTL" train "configs/train/${ARCH}/${TASK}.yaml"
    done
done
