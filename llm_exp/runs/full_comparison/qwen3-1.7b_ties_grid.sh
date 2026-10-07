#!/usr/bin/env bash
# Subspace upper bound: Qwen3-1.7B x TIES-Merging, refined coefficient grid.
#
# For TIES this grid is the *exact* subspace upper bound, not an approximation:
# ties.basis() returns a single direction (the merged delta), so the TIES
# subspace is one-dimensional and the lambda ray *is* the whole subspace. That
# matters most here, because subspace_gd is dropped entirely at 1.7b+ (too slow
# to finish -- see qwen3-1.7b_ties.sh), leaving no gradient estimate at all.
#
# Range: the coarse 0.1-1.0 x 10 sweep never bracketed the optimum -- it was
# still climbing at the lambda=1.0 grid edge (0.8: 0.500, 0.9: 0.520,
# 1.0: 0.655), so the peak lies at or above 1.0. Sweep 0.8-1.8 to push past
# that edge while keeping the 1.0 shoulder in view; 41 steps over that window
# = 0.025 interval. Re-run narrower once the peak is located.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
METHOD="ties"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/ties_coeff_grid"
EXTRA_ARGS="--ties-density 0.2 --gen-batch-size 32 --lambda-min 0.8 --lambda-max 1.8 --lambda-steps 41"
run_eval
