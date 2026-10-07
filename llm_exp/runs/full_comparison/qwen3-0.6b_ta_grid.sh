#!/usr/bin/env bash
# Subspace upper bound: Qwen3-0.6B x Task Arithmetic, refined coefficient grid.
#
# subspace_gd underperforms plain coeff_search here (0.6b/ta: subspace_gd's
# best init "merged" reaches valid 0.718 / test 0.671, and its "coeff_best"
# init actually *degrades* to valid 0.496 / test 0.396 -- below the coeff_best
# point it started from), so we cannot read the reachable optimum inside the
# TA subspace off the gradient runs. This script instead brackets that optimum
# by brute force: a dense sweep of the diagonal c=(lambda,...,lambda) ray.
#
# Note the scope of the bound. TA's basis is one direction per task (4 here),
# so the full subspace is 4-D and coeff_search only walks its diagonal; this
# is the standard TA coefficient upper bound, not the 4-D subspace optimum.
#
# Range: the coarse 0.1-1.0 x 10 sweep peaked at lambda=0.3 (valid 0.686) with
# 0.2 and 0.5 both still ~0.63-0.67, so the optimum sits inside 0.15-0.55.
# 41 steps over that window = 0.01 interval, matching the vision-side grids.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
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
EXTRA_ARGS="--gen-batch-size 32 --lambda-min 0.15 --lambda-max 0.55 --lambda-steps 41"
run_eval
