#!/usr/bin/env bash
# Subspace upper bound: Qwen3-4B x TIES-Merging, refined coefficient grid.
#
# subspace_gd is not run at 4B: it holds base_dev plus one full-model-sized
# direction per basis vector resident on the GPU, on top of the model itself,
# and would OOM on the 31.4GiB cards in this box (weight_gd already does, for
# the same class of reason -- see qwen3-4b_ta.sh). This script brackets the
# reachable optimum by brute force instead of by gradient.
#
# For this method the grid is the *exact* subspace upper bound, not an
# approximation: basis() returns a single direction (the merged delta), so the
# subspace is one-dimensional and the lambda ray IS the whole subspace.
#
# Range: the coarse 0.1-1.0 x 10 sweep never bracketed the optimum -- valid was
# still climbing monotonically at the lambda=1.0 grid edge (0.8: 0.7361,
# 0.9: 0.7400, 1.0: 0.7485), so the peak lies at or above 1.0. Sweep 0.9-1.9 to
# push past that edge while keeping the 1.0 shoulder in view; 41 steps = 0.025
# interval. If it is STILL climbing at 1.9, extend again rather than taking the
# endpoint.
#
# (Unlike the tsvm grid, there is no valid/test disagreement to design around
# here: coeff_search's pick already IS lambda=1.0, and the baselines
# merged_coeff1 cell at the same lambda agrees to within noise -- valid
# 0.7455 vs 0.7501, test 0.6826 vs 0.6787.)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-4b"
BASE_MODEL="Qwen/Qwen3-4B-Base"
METHOD="ties"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
WEIGHT_GD_LORA_INITS="coeff_best"  # unused (weight_gd_lora not in STRATEGIES); ditto
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

LAMBDA_MIN="${LAMBDA_MIN:-0.9}"
LAMBDA_MAX="${LAMBDA_MAX:-1.9}"
LAMBDA_STEPS="${LAMBDA_STEPS:-41}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/ties_coeff_grid"
EXTRA_ARGS="--ties-density 0.2 --gen-batch-size $GEN_BATCH_SIZE --lambda-min $LAMBDA_MIN --lambda-max $LAMBDA_MAX --lambda-steps $LAMBDA_STEPS"
run_eval
