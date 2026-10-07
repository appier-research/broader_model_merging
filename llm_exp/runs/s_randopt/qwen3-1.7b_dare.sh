#!/usr/bin/env bash
# Directional sampling: Qwen3-1.7B x DARE, 4 tasks (full), centered on bo_search's
# best trial. coeff_search's lambda* / density threshold and bo_search's
# coefficients are read from $COEFF_SOURCE_DIR (the main results root), same
# unit name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
METHOD="dare"
STRATEGIES="directional_sampling"
WEIGHT_GD_INITS="avg"                        # required by _lib; unused
WEIGHT_GD_LORA_INITS="avg"                   # required by _lib; unused
SUBSPACE_GD_INITS="coeff_best"               # required by _lib; unused
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Side folder: this protocol (alpha=0.05, 64 draws, test on every draw) is not
# comparable cell-for-cell with the valid-selected top-5 runs in
# results/directional_sampling.
OUT_DIR="${OUT_DIR:-results/directional_sampling_test_alpha0.05}"
COEFF_SOURCE_DIR="${COEFF_SOURCE_DIR:-results}"

# bo_best: on this unit BO (valid 0.8367) beat both the coefficient sweep
# (0.7878 on the 41-point grid) and subspace_gd (0.5758). DARE's basis is one dropped-and-rescaled
# task vector per task (same masks as merged_delta -- they are seeded and
# cached), so n_dirs = 4 like TA and the GPU footprint is TA's.
DS_INIT="bo_best"             # pretrained | avg | merged | coeff_best | subspace_best | bo_best
DS_ALPHA=0.05
# beta multiplies the raw g_perp, whose norm is ~130-290 on these units, so the
# weight-space step is beta * ||g_perp||. Binning the earlier beta_max=0.01 runs
# by that step: 0.01-0.1 beats the center 80-100% of the time, 0.1-0.3 degrades,
# >0.3 collapses to score 0 (75-80% of those draws were wasted there). 5e-4
# keeps every draw at a step <= ~0.07-0.15, matching the TA/TIES scripts.
DS_BETA=5e-4                  # >0 adds the -g_perp axis (costs one backward pass over the SFT pools)
# 64 draws, each also scored on the ~10x-larger test pool (3254 vs 361
# examples); a 1.7b DARE draw took ~27 min for the test pass alone in the old
# top-5 phase, so 64 draws is ~30 h.
DS_SAMPLES=64
# Score the TEST pool on every draw and report the test-score distribution over
# the box (test_dist / density_test) instead of a valid-selected top-5 -- see
# design.md's "Directional sampling".
DS_EVAL_TEST=1
DS_SELECT_METRIC="score"      # matches every other directional_sampling run

# Same residency as the 1.7b TA script (n_dirs + 3 = 7 bf16 copies, ~22.4 GiB):
# needs a 32 GB card; batch sizes measured there (see qwen3-1.7b_ta.sh).
# DARE also caches its 4 dropped task vectors in fp32 on the CPU (~28 GB for
# 1.7B) on top of the base/task state dicts -- ~60 GB host RAM in total.
EXTRA_ARGS="--gen-batch-size 8 --loss-batch-size 1"
run_eval
