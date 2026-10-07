#!/usr/bin/env bash
# Subspace upper bound: Qwen3-0.6B x TIES-Merging, refined coefficient grid.
#
# For TIES this grid is the *exact* subspace upper bound, not an approximation:
# ties.basis() returns a single direction (the merged delta), so the TIES
# subspace is one-dimensional and the lambda ray *is* the whole subspace.
# Anything subspace_gd can reach, a dense enough lambda sweep also reaches --
# which is what makes the comparison against subspace_gd's poor numbers clean.
#
# Range: no coarse 0.6b TIES sweep exists yet, so this reuses the 1.7b window.
# That sweep never bracketed its optimum -- it was still climbing at the
# lambda=1.0 grid edge (0.8: 0.500, 0.9: 0.520, 1.0: 0.655) -- so the peak lies
# at or above 1.0. Sweep 0.8-1.8 to push past that edge while keeping the 1.0
# shoulder in view; 41 steps over that window = 0.025 interval. Matching 1.7b
# also keeps the two archs' grids directly comparable point-for-point. Re-run
# narrower once the peak is located.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
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
