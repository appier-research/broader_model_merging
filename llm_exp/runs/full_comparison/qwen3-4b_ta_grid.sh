#!/usr/bin/env bash
# Subspace upper bound: Qwen3-4B x Task Arithmetic, refined coefficient grid.
#
# subspace_gd is not run at 4B: it holds base_dev plus one full-model-sized
# direction per basis vector resident on the GPU, on top of the model itself,
# and would OOM on the 31.4GiB cards in this box (weight_gd already does, for
# the same class of reason -- see qwen3-4b_ta.sh). This script brackets the
# reachable optimum by brute force instead of by gradient.
#
# Note the scope of the bound. This method's basis is one direction per task
# (4 here), so the full subspace is 4-D and coeff_search only walks its
# diagonal c=(lambda,...,lambda). This is the standard coefficient upper
# bound, NOT the 4-D subspace optimum -- the true subspace optimum is >= this.
#
# Range: the coarse 0.1-1.0 x 10 sweep peaked at lambda=0.4 (valid 0.8875) with
# 0.3 at 0.8488 and 0.5 at 0.8729, so the optimum sits inside 0.25-0.65.
# 41 steps over that window = 0.01 interval, matching the other grids.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-4b"
BASE_MODEL="Qwen/Qwen3-4B-Base"
METHOD="ta"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
WEIGHT_GD_LORA_INITS="coeff_best"  # unused (weight_gd_lora not in STRATEGIES); ditto
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

LAMBDA_MIN="${LAMBDA_MIN:-0.25}"
LAMBDA_MAX="${LAMBDA_MAX:-0.65}"
LAMBDA_STEPS="${LAMBDA_STEPS:-41}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/ta_coeff_grid"
EXTRA_ARGS="--gen-batch-size $GEN_BATCH_SIZE --lambda-min $LAMBDA_MIN --lambda-max $LAMBDA_MAX --lambda-steps $LAMBDA_STEPS"
run_eval
