#!/usr/bin/env bash
# meta-llama/Llama-3.2-1B-Instruct x dare, all 4 tasks, full budget: bo_search only.
#
# Bayesian optimization (optuna GPSampler) over 4 per-task coefficients,
# maximizing valid_avg_score directly. Box = lambda* +/- BO_RADIUS per
# coefficient, clamped at BO_LOWER; the center (coeff_best) is the first
# trial, then 2*n_dirs uniform startup draws, then GP proposals, BO_TRIALS in
# total. Each trial is one valid generate()+score pass.
#
# Needs coeff_search's lambda* already in results/units/<unit>.json (the
# runs/full_comparison script for this (arch, method) writes it); nothing else from other
# strategies is read. Base + basis directions stay on the GPU when they fit
# (BO_BASIS_DEVICE=auto), otherwise W(c) is built on the CPU each trial.
#
# BO_RADIUS defaults to 0.7 from the llama3.2-1b x TA radius sweep (results/bo_radius/):
# 0.3 pinned ifeval/usefulness_judge at the box edge
# (test 0.678); 0.5/0.7/0.9 gave test 0.689/0.702/0.709 with 0.7 converging fastest and
# proposing no duplicate points -- the gain came from ddxplus's coefficient (~0.58).
#
# Reruns fill in only missing cells. To redo: FORCE=bo_search bash <this script>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
METHOD="dare"
STRATEGIES="bo_search"
WEIGHT_GD_INITS="coeff_best"      # unused (strategy not in STRATEGIES) but required by run_eval
SUBSPACE_GD_INITS="coeff_best"    # unused, same
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

BO_INITS="${BO_INITS:-coeff_best}"
BO_TRIALS="${BO_TRIALS:-50}"
BO_RADIUS="${BO_RADIUS:-0.7}"
BO_LOWER="${BO_LOWER:-0.0}"
BO_SEED="${BO_SEED:-42}"
BO_BASIS_DEVICE="${BO_BASIS_DEVICE:-auto}"

SAVE_CKPT_ARG=""
if [ "${SAVE_CHECKPOINTS:-0}" != "0" ]; then SAVE_CKPT_ARG="--save-checkpoints"; fi

EXTRA_ARGS="$SAVE_CKPT_ARG --gen-batch-size 16"
run_eval
