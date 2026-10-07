#!/usr/bin/env bash
# Subspace upper bound: Qwen3-0.6B x TSV-M, refined coefficient grid.
#
# For TSV-M this grid is the *exact* subspace upper bound, not an
# approximation: tsv_m.basis() returns a single direction (the merged delta),
# so the TSV-M subspace is one-dimensional and the lambda ray *is* the whole
# subspace -- same situation as TIES.
#
# subspace_gd cannot supply that bound here, and not because it is broken: it
# minimizes teacher-forced SFT loss, which on this unit is *anti-correlated*
# with the generation score over most of the range. The coarse sweep's own
# per-lambda losses show it -- loss bottoms at lambda=0.2 (1.728, score 0.320)
# while score peaks at lambda=0.7 (loss 3.448, score 0.740). All four
# subspace_gd inits duly converge to c=0.22 with train loss ~1.75, i.e. they
# land on the loss optimum and therefore near the score *minimum* (test
# 0.318-0.332). The gradient run is correct; its objective just is not the
# metric. Hence brute force.
#
# Range: the coarse 0.1-1.0 x 10 sweep brackets the peak cleanly -- 0.6: 0.721,
# 0.7: 0.740, 0.8: 0.716 -- so the optimum sits inside 0.5-0.9. 41 steps over
# that window = 0.01 interval, matching the TA/TIES grids.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
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
