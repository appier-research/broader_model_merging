#!/usr/bin/env bash
# Subspace upper bound: Qwen3-1.7B x TSV-M, refined coefficient grid.
#
# For TSV-M this grid is the *exact* subspace upper bound, not an
# approximation: tsv_m.basis() returns a single direction (the merged delta),
# so the TSV-M subspace is one-dimensional and the lambda ray *is* the whole
# subspace -- same situation as TIES.
#
# subspace_gd cannot supply that bound here, and not because it is broken: it
# minimizes teacher-forced SFT loss, which on this unit is *anti-correlated*
# with the generation score over the whole range. The coarse sweep's own
# per-lambda losses show it -- loss bottoms at lambda=0.3 (1.309, score 0.609)
# while score climbs monotonically to lambda=1.0 (loss 4.566, score 0.849).
# All four subspace_gd inits duly converge to c=0.23-0.26 with train loss
# ~1.32-1.34, i.e. they land on the loss optimum, scoring test 0.464-0.486
# against coeff_search's 0.784 at the same budget. The gradient run is
# correct; its objective just is not the metric. Hence brute force.
#
# Range: the coarse 0.1-1.0 x 10 sweep never bracketed the optimum -- it was
# still climbing at the lambda=1.0 grid edge (0.8: 0.793, 0.9: 0.815,
# 1.0: 0.849), so the peak lies at or above 1.0. Sweep 0.8-1.8 to push past
# that edge while keeping the 1.0 shoulder in view; 41 steps over that window
# = 0.025 interval. Re-run narrower once the peak is located.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
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
EXTRA_ARGS="--gen-batch-size 32 --lambda-min 0.8 --lambda-max 1.8 --lambda-steps 41"
run_eval
