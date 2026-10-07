#!/usr/bin/env bash
# Subspace upper bound: Qwen3-4B x TSV-M, refined coefficient grid.
#
# subspace_gd is not run at 4B: it holds base_dev plus one full-model-sized
# direction per basis vector resident on the GPU, on top of the model itself,
# and would OOM on the 31.4GiB cards in this box (weight_gd already does, for
# the same class of reason -- see qwen3-4b_ta.sh). This script brackets the
# reachable optimum by brute force instead of by gradient.
#
# For this method the grid is the *exact* subspace upper bound, not an
# approximation: basis() returns a single direction (the merged delta), so the
# subspace is one-dimensional and the lambda ray IS the whole subspace.
#
# Range 0.45-1.15, deliberately wider than the valid peak suggests, because
# valid and test disagree about which lambda is better and the disagreement
# looks like noise in the selection metric rather than a real optimum at 0.6:
#
#   lambda=0.6 (coeff_search's pick)  valid 0.8979   test 0.7985
#   lambda=1.0 (baselines/merged_coeff1) valid 0.8804   test 0.8126
#
# Per task, 3 of the 4 are BETTER at 1.0 on both valid and test (bank77
# .88->.92 valid / .913->.953 test; ddxplus and ifeval likewise). The entire
# preference for 0.6 comes from usefulness_judge, whose valid pool is 25 rows:
# .96 vs .84 there is 24/25 vs 21/25, a three-row difference, against a 95%
# binomial band of about +/-0.12 at n=25. Because the selection metric is an
# unweighted 4-task mean, that one small pool swings it by 0.03 -- more than
# the 0.0175 valid gap that chose 0.6. On the 225-row test split the same task
# only differs .729 vs .707.
#
# The coarse valid curve also dips at 0.9 (0.8704) and rises again at 1.0
# (0.8819), which is not the shape of a real peak at 0.6.
#
# So sweep through 1.0 rather than stopping at 0.85: 0.4-1.2, 41 steps = 0.02
# interval, chosen so the grid lands exactly on both 0.6 and 1.0 and the new
# curve is directly comparable to the numbers above. If the refined curve is
# flat between ~0.6 and ~1.0, prefer the higher lambda -- it is the one the
# larger test split and 3 of 4 tasks agree on. Selection still happens on
# valid; this only ensures valid is asked about the right range.
#
# Range 0.4-1.2 also means this grid supersedes the merged_coeff1 baseline
# (lambda=1.0) rather than sitting beside it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-4b"
BASE_MODEL="Qwen/Qwen3-4B-Base"
METHOD="tsvm"
STRATEGIES="coeff_search"
WEIGHT_GD_INITS="coeff_best"       # unused (weight_gd not in STRATEGIES); satisfies run_eval's guard
WEIGHT_GD_LORA_INITS="coeff_best"  # unused (weight_gd_lora not in STRATEGIES); ditto
SUBSPACE_GD_INITS="coeff_best"     # unused (subspace_gd not in STRATEGIES); ditto
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

LAMBDA_MIN="${LAMBDA_MIN:-0.4}"
LAMBDA_MAX="${LAMBDA_MAX:-1.2}"
LAMBDA_STEPS="${LAMBDA_STEPS:-41}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"

# Own OUT_DIR: the unit JSON is keyed by (tasks, arch, method, budget) with no
# room for the lambda grid, so writing here would let _plan() see the main
# run's coeff_search cell as already-present and skip the refined sweep.
OUT_DIR="results/tsvm_coeff_grid"
EXTRA_ARGS="--embedding-svd-device cpu --gen-batch-size $GEN_BATCH_SIZE --lambda-min $LAMBDA_MIN --lambda-max $LAMBDA_MAX --lambda-steps $LAMBDA_STEPS"
run_eval
