#!/usr/bin/env bash
# Draw init-vs-weight_gd scatters. Override METHODS / ARCHS / METRIC as needed.
#
#   bash runs/init_scatters/plot.sh
#   METHODS=pretrained,ta,dare,ties,random bash runs/init_scatters/plot.sh
#   ARCHS=vit-b-32 METHODS=pretrained,ta,ties bash runs/init_scatters/plot.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

METHODS="${METHODS:-random,pretrained,ta,dare,ties,tsvm}"
ARCHS="${ARCHS:-vit-b-32,vit-b-16,vit-l-14}"
METRIC="${METRIC:-test_avg_acc}"
RESULTS_DIR="${RESULTS_DIR:-results/main_exp}"
RANDOM_DIR="${RANDOM_DIR:-results/weight_gd_random}"
TASKS_CSV="${TASKS_CSV:-dtd,eurosat,fer2013,food101,gtsrb,mnist,resisc45,stanford-cars,sun397}"

IFS=',' read -ra arch_arr <<< "$ARCHS"
for arch in "${arch_arr[@]}"; do
    arch="${arch// /}"
    [ -z "$arch" ] && continue
    echo "=== plot_init_scatter  arch=$arch  methods=$METHODS  metric=$METRIC ==="
    (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot_init_scatter.py \
        --arch "$arch" \
        --methods "$METHODS" \
        --tasks "$TASKS_CSV" \
        --results-dir "$RESULTS_DIR" \
        --random-dir "$RANDOM_DIR" \
        --metric "$METRIC")
done
