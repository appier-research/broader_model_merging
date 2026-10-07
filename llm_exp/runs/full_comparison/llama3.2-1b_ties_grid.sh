#!/usr/bin/env bash
# Subspace upper bound: Llama-3.2-1B-Instruct x TIES-Merging, refined grid.
#
# For TIES this grid is the *exact* subspace upper bound, not an approximation:
# ties.basis() returns a single direction (the merged delta), so the TIES
# subspace is one-dimensional and the lambda ray *is* the whole subspace.
# That matters most here, because subspace_gd is dropped for this arch (too
# slow to finish -- see llama3.2-1b_ties.sh), leaving no gradient estimate.
#
# Range: unlike qwen3-1.7b TIES (which was still climbing at the lambda=1.0
# grid edge), the coarse 0.1-1.0 x 10 sweep here did bracket its peak: it rose
# to lambda=0.9 (valid 0.620) and then turned over at 1.0 (0.595), with 0.8
# just behind (0.607). So refine 0.7-1.1 rather than reusing the qwen 0.8-1.8
# window -- no need to hunt above 1.0 when the turnover is already in view.
# 41 steps over that window = 0.01 interval.
#
# Checkpoints: ARCH=llama3.2-1b bash admin/download_checkpoints.sh
# See llama3.2-1b_ta.sh on why BASE_MODEL is the Instruct model.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
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
EXTRA_ARGS="--ties-density 0.2 --gen-batch-size 32 --lambda-min 0.7 --lambda-max 1.1 --lambda-steps 41"
run_eval
