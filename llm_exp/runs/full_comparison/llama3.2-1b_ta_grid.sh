#!/usr/bin/env bash
# Subspace upper bound: Llama-3.2-1B-Instruct x Task Arithmetic, refined grid.
#
# subspace_gd is dropped for this arch (too slow to finish -- see
# llama3.2-1b_ta.sh), so there is no gradient-based estimate of the reachable
# optimum inside the TA subspace here at all. This script brackets it by brute
# force: a dense sweep of the diagonal c=(lambda,...,lambda) ray.
#
# Note the scope of the bound. TA's basis is one direction per task (4 here),
# so the full subspace is 4-D and coeff_search only walks its diagonal; this
# is the standard TA coefficient upper bound, not the 4-D subspace optimum.
#
# Range: the coarse 0.1-1.0 x 10 sweep peaked at lambda=0.4 (valid 0.801) with
# a sharp falloff on the high side (0.5: 0.713, 0.6: 0.586) and 0.3 close
# behind (0.794), so the optimum sits inside 0.25-0.55 -- a tighter window than
# the qwen TA grids need. 31 steps over it = 0.01 interval, matching the
# vision-side grids.
#
# Checkpoints: ARCH=llama3.2-1b bash admin/download_checkpoints.sh
# See llama3.2-1b_ta.sh on why BASE_MODEL is the Instruct model.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
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
EXTRA_ARGS="--gen-batch-size 32 --lambda-min 0.25 --lambda-max 0.55 --lambda-steps 31"
run_eval
