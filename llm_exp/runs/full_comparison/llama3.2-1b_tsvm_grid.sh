#!/usr/bin/env bash
# Subspace upper bound: Llama-3.2-1B x TSV-M, refined coefficient grid.
#
# For TSV-M this grid is the *exact* subspace upper bound, not an
# approximation: tsv_m.basis() returns a single direction (the merged delta),
# so the TSV-M subspace is one-dimensional and the lambda ray *is* the whole
# subspace -- same situation as TIES.
#
# This unit is the well-behaved one: unlike the qwen units (see
# qwen3-0.6b_tsvm_grid.sh), loss and score agree here -- loss bottoms at
# lambda=0.6 (0.506) and score peaks one grid step away at lambda=0.7 (0.841).
# subspace_gd was not run on this unit, but the agreement means it would have
# been informative if it had been; the grid is still the cheaper exact bound
# for a 1-D subspace.
#
# Range: the coarse 0.1-1.0 x 10 sweep brackets the peak -- 0.6: 0.831,
# 0.7: 0.841, 0.8: 0.801 -- so the optimum sits inside 0.5-0.9. 41 steps over
# that window = 0.01 interval, matching the TA/TIES grids.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

# See llama3.2-1b_ta.sh for why BASE_MODEL is the Instruct model.
ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
METHOD="tsvm"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/tsvm_coeff_grid"
EXTRA_ARGS="--gen-batch-size 32 --lambda-min 0.5 --lambda-max 0.9 --lambda-steps 41"
run_eval
