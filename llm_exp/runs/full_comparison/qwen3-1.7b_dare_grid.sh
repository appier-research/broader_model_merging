#!/usr/bin/env bash
# Subspace upper bound: Qwen3-1.7B x DARE, refined coefficient grid.
#
# Note the scope of the bound. DARE's basis is one dropped-and-rescaled
# direction per task (4 here), so the full subspace is 4-D and coeff_search
# only walks its diagonal c=(lambda,...,lambda); this is the standard DARE
# coefficient upper bound, not the 4-D subspace optimum -- same caveat as the
# TA grid.
#
# subspace_gd is not available here anyway (dropped at 1.7b+, too slow to
# finish -- see qwen3-1.7b_dare.sh), and the 0.6b runs show it would not have
# helped: it minimizes teacher-forced SFT loss, which is anti-correlated with
# the generation score on this unit too -- loss bottoms at lambda=0.1 (1.248,
# score 0.438) while score peaks at lambda=0.5 (loss 6.521, score 0.778).
# Brute force is the only estimate of the bound.
#
# Range: the coarse 0.1-1.0 x 10 sweep brackets the peak -- 0.4: 0.763,
# 0.5: 0.778, 0.6: 0.768 -- so the optimum sits inside 0.3-0.7. 41 steps over
# that window = 0.01 interval, matching the TA grid.
#
# --dare-drop-rate/--dare-seed must match the main run (qwen3-1.7b_dare.sh):
# the drop masks are seeded, so a different seed samples a different subspace
# and the grid would no longer bound the same merge.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
METHOD="dare"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/dare_coeff_grid"
EXTRA_ARGS="--dare-drop-rate 0.5 --dare-seed 42 --gen-batch-size 32 \
  --lambda-min 0.3 --lambda-max 0.7 --lambda-steps 41"
run_eval
