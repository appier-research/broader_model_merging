#!/usr/bin/env bash
# Subspace upper bound: Qwen3-1.7B x Task Arithmetic, refined coefficient grid.
#
# subspace_gd is dropped entirely at 1.7b+ (too slow to finish -- see
# qwen3-1.7b_ta.sh), so there is no gradient-based estimate of the reachable
# optimum inside the TA subspace at this scale at all. This script brackets it
# by brute force: a dense sweep of the diagonal c=(lambda,...,lambda) ray.
#
# Note the scope of the bound. TA's basis is one direction per task (4 here),
# so the full subspace is 4-D and coeff_search only walks its diagonal; this
# is the standard TA coefficient upper bound, not the 4-D subspace optimum.
#
# Range: the coarse 0.1-1.0 x 10 sweep peaked at lambda=0.6 (valid 0.808) on a
# broad plateau (0.4: 0.791, 0.5: 0.796, 0.7: 0.760), so the optimum sits
# inside 0.35-0.75. 41 steps over that window = 0.01 interval, matching the
# vision-side grids.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
METHOD="ta"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/ta_coeff_grid"
EXTRA_ARGS="--gen-batch-size 32 --lambda-min 0.35 --lambda-max 0.75 --lambda-steps 41"
run_eval
