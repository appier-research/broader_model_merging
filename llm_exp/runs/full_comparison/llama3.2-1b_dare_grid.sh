#!/usr/bin/env bash
# Subspace upper bound: Llama-3.2-1B x DARE, refined coefficient grid.
#
# Note the scope of the bound. DARE's basis is one dropped-and-rescaled
# direction per task (4 here), so the full subspace is 4-D and coeff_search
# only walks its diagonal c=(lambda,...,lambda); this is the standard DARE
# coefficient upper bound, not the 4-D subspace optimum -- same caveat as the
# TA grid.
#
# This unit is the well-behaved one: unlike the qwen units (see
# qwen3-0.6b_dare_grid.sh), loss and score agree here -- both bottom/peak at
# lambda=0.3 (loss 0.537, score 0.796).
#
# Range: the coarse 0.1-1.0 x 10 sweep brackets the peak -- 0.2: 0.708,
# 0.3: 0.796, 0.4: 0.793 -- so the optimum sits inside 0.15-0.55, skewed
# toward the 0.3-0.4 plateau. 41 steps over that window = 0.01 interval,
# matching the TA grid.
#
# --dare-drop-rate/--dare-seed must match the main run (llama3.2-1b_dare.sh):
# the drop masks are seeded, so a different seed samples a different subspace
# and the grid would no longer bound the same merge.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

# See llama3.2-1b_ta.sh for why BASE_MODEL is the Instruct model.
ARCH="llama3.2-1b"
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
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
