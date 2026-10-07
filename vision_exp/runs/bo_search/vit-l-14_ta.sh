#!/usr/bin/env bash
# bo_search (optuna GPSampler) over basis coefficients, full budget: ViT-L/14 x TA.
#
# Maximizes valid_avg_acc over W(c) = base + sum_i c_i * dir_i (TA: 9 per-task
# coefficients). Box = coeff_best (lambda*) +/- BO_RADIUS per coefficient, clamped
# at BO_LOWER; the center is the first trial, then 2*n_dirs uniform startup draws,
# then GP proposals, BO_TRIALS in total. Each trial is one valid forward pass.
#
# base + 9 fp32 task vectors stay resident on the GPU (~12 GB); keep BATCH_SIZE
# modest so inference activations do not tip a 24 GB card over.
#
# Writes the 'bo_search' key into results/main_exp/units/<unit>.json alongside the
# paper cells -- cells merge, present strategies are skipped, so DO NOT --force.
# Needs coeff_search's lambda* already in that JSON. Reruns fill only missing cells.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-l-14"
METHOD="ta"
STRATEGIES="bo_search"
WEIGHT_GD_INITS="coeff_best"      # unused (strategy not in STRATEGIES) but required by _lib.sh
SUBSPACE_GD_INITS="coeff_best"    # unused, same
BATCH_SIZE=4                     # inference-only, but basis is resident: leave headroom
BUDGETS="full"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)
OUT_DIR="results/main_exp"

BO_INITS="${BO_INITS:-coeff_best}"
BO_TRIALS="${BO_TRIALS:-50}"
BO_RADIUS="${BO_RADIUS:-0.4}"
BO_LOWER="${BO_LOWER:--0.2}"
BO_SEED="${BO_SEED:-42}"

PLOT=0
run_merge
