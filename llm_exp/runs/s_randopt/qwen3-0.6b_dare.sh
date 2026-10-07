#!/usr/bin/env bash
# Directional sampling: Qwen3-0.6B x DARE, 4 tasks (full), centered on bo_search's
# best trial. coeff_search's lambda* / density threshold and bo_search's
# coefficients are read from $COEFF_SOURCE_DIR (the main results root), same
# unit name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
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

# bo_best: on this unit BO (valid 0.8027) beat both the coefficient sweep
# (0.6963 on the 41-point grid) and subspace_gd (0.7253). DARE's basis is one dropped-and-rescaled
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
# examples); a 0.6b draw is ~8 min with the test pass, so 64 draws is ~9 h.
DS_SAMPLES=64
# Score the TEST pool on every draw and report the test-score distribution over
# the box (test_dist / density_test) instead of a valid-selected top-5 -- see
# design.md's "Directional sampling".
DS_EVAL_TEST=1
DS_SELECT_METRIC="score"      # matches every other directional_sampling run

# DARE's basis is one dropped-and-rescaled vector per task, so its GPU residency
# is TA's (n_dirs + 3 = 7 bf16 copies, ~9.8 GB on a 24 GB card), not TIES's 4.
# The gradient pass then has ~13 GB left, and its fp32 logits + their grad at
# usefulness_judge's ~1900-token sequences cost ~2.3 GB per sequence: loss
# batch 4 OOMed there (2026-09-16), loss batch 2 is what the 0.6b TA script
# runs with the same residency. gen batch 32 matches TA/TIES.
EXTRA_ARGS="--gen-batch-size 32 --loss-batch-size 2"
run_eval
