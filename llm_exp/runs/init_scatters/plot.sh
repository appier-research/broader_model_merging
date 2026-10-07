#!/usr/bin/env bash
# Draw init-vs-weight_gd scatters. Override METHODS / ARCHS / METRIC as needed.
#
#   bash runs/init_scatters/plot.sh
#   ARCHS=qwen3-0.6b METHODS=pretrained,ta,ties bash runs/init_scatters/plot.sh
#   GD_STRATEGY=weight_gd ARCHS=qwen3-0.6b bash runs/init_scatters/plot.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

METHODS="${METHODS:-pretrained,ta,dare,ties,tsvm}"
ARCHS="${ARCHS:-qwen3-0.6b,qwen3-1.7b,qwen3-4b}"
# GD cells to plot. The LLM experiments run GD through LoRA (full-weight GD
# OOMs on qwen3-4b); pass GD_STRATEGY=weight_gd for the full-weight cells.
GD_STRATEGY="${GD_STRATEGY:-weight_gd_lora}"
METRIC="${METRIC:-test_avg_score}"
BUDGET="${BUDGET:-full}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RANDOM_DIR="${RANDOM_DIR:-results/weight_gd_random}"
TASKS_CSV="${TASKS_CSV:-bank77,ddxplus,ifeval,usefulness_judge}"

IFS=',' read -ra arch_arr <<< "$ARCHS"
for arch in "${arch_arr[@]}"; do
    arch="${arch// /}"
    [ -z "$arch" ] && continue
    echo "=== plot_init_scatter  arch=$arch  methods=$METHODS  metric=$METRIC  gd=$GD_STRATEGY ==="
    # An arch with no GD cells at all exits non-zero; keep going through the
    # remaining archs instead of tripping `set -e`.
    (cd "$LLM_EXP_ROOT" && "$PYTHON" scripts/plots/plot_init_scatter.py \
        --arch "$arch" \
        --methods "$METHODS" \
        --tasks "$TASKS_CSV" \
        --budget "$BUDGET" \
        --gd-strategy "$GD_STRATEGY" \
        --results-dir "$RESULTS_DIR" \
        --random-dir "$RANDOM_DIR" \
        --metric "$METRIC") || echo "  (no plot for $arch)"
done
