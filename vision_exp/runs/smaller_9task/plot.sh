#!/usr/bin/env bash
# One data-scaling figure per class seed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
NUM_CLASSES="${NUM_CLASSES:-53}"
CLASS_SEEDS=(${CLASS_SEEDS:-0 1 2})
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
METRIC="${PLOT_METRIC:-test_avg_acc}"

tasks_csv="$(join_by , "${TASKS[@]}")"
for s in "${CLASS_SEEDS[@]}"; do
    out_dir="results/smaller_9task_${NUM_CLASSES}/cseed${s}"
    echo "plot  $out_dir"
    (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot.py data_scaling \
        --results-dir "$out_dir" --tasks "$tasks_csv" \
        --arch "$ARCH" --method "$METHOD" --metric "$METRIC" \
        --title "ViT-B/32 (TA), ${NUM_CLASSES} classes, cseed ${s}")
done
