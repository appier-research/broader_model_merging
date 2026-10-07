#!/usr/bin/env bash
# Subspace upper bound: Qwen3-0.6B x DARE, refined coefficient grid.
#
# Note the scope of the bound. DARE's basis is one dropped-and-rescaled
# direction per task (4 here), so the full subspace is 4-D and coeff_search
# only walks its diagonal c=(lambda,...,lambda); this is the standard DARE
# coefficient upper bound, not the 4-D subspace optimum -- same caveat as the
# TA grid.
#
# subspace_gd cannot supply the 4-D optimum here, and not because it is broken:
# it minimizes teacher-forced SFT loss, which on this unit is *anti-correlated*
# with the generation score. The coarse sweep's own per-lambda losses show it
# -- loss bottoms at lambda=0.1 (1.578, score 0.344) while score peaks at
# lambda=0.3 (loss 3.735, score 0.686). Three of the four inits converge to
# ~[0.3,-0.09,0.38,0.12] at train loss ~1.22-1.28 and score test 0.369-0.424.
# The exception proves the point: the "merged" init settles at a much *higher*
# train loss (4.52) and is the only one that scores well (test 0.685) -- lower
# loss buying a worse score is exactly the anti-correlation, not an optimizer
# fault. Hence brute force.
#
# Range: the coarse 0.1-1.0 x 10 sweep brackets the peak -- 0.2: 0.629,
# 0.3: 0.686, 0.4: 0.661 -- so the optimum sits inside 0.15-0.55. 41 steps
# over that window = 0.01 interval, matching the TA grid.
#
# --dare-drop-rate/--dare-seed must match the main run (qwen3-0.6b_dare.sh):
# the drop masks are seeded, so a different seed samples a different subspace
# and the grid would no longer bound the same merge.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
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
  --lambda-min 0.15 --lambda-max 0.55 --lambda-steps 41"
run_eval
