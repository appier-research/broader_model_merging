#!/usr/bin/env bash
# Directional sampling: Qwen3-0.6B x Task Arithmetic, 4 tasks (full).
# coeff_best lambda* and the density threshold are read from $COEFF_SOURCE_DIR
# (the main results root), same unit name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-0.6b"
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
METHOD="ta"
STRATEGIES="directional_sampling"
WEIGHT_GD_INITS="avg"                        # required by _lib; unused
WEIGHT_GD_LORA_INITS="avg"                   # required by _lib; unused
SUBSPACE_GD_INITS="coeff_best"               # required by _lib; unused
BUDGETS="full"
TASKS=(bank77 ddxplus ifeval usefulness_judge)

# Side folder: this protocol (alpha=0.05, 64 draws, test on every draw) is not
# comparable cell-for-cell with the valid-selected top-5 runs in
# results/directional_sampling, nor with the alpha=0.1 / 64-draw cells in
# results/directional_sampling_test.
OUT_DIR="${OUT_DIR:-results/directional_sampling_test_alpha0.05}"
COEFF_SOURCE_DIR="${COEFF_SOURCE_DIR:-results}"

# bo_best centers the box on bo_search's best trial (per-direction vector,
# read from $COEFF_SOURCE_DIR). On this unit BO (valid 0.8035) beat both the
# coefficient sweep (0.7031) and subspace_gd (0.7173), so its point is the
# strongest known center. The coeff_best and subspace_best cells already in
# the unit JSON are kept; merge_eval.py only computes inits that are missing.
DS_INIT="bo_best"             # pretrained | avg | merged | coeff_best | subspace_best | bo_best
DS_ALPHA=0.05
# beta multiplies the raw g_perp, whose norm is ~130-290 on these units, so the
# weight-space step is beta * ||g_perp||. Binning the earlier beta_max=0.01 runs
# by that step: 0.01-0.1 beats the center 80-100% of the time, 0.1-0.3 degrades,
# >0.3 collapses to score 0 (75-80% of those draws were wasted there). 5e-4
# keeps every draw at a step <= ~0.07-0.15 across 0.6b/1.7b x TA/TIES.
DS_BETA=5e-4                  # >0 adds the -g_perp axis (costs one backward pass over the SFT pools)
# 64 draws, each also scored on the ~10x-larger test pool (3254 vs 361
# examples) -- a draw costs about what one old top-5 re-eval did (~8 min
# for 0.6b), so 128 draws would be ~17 h per 0.6b unit and ~53 h for 1.7b TA.
DS_SAMPLES=64
# Score the TEST pool on every draw and report the test-score distribution over
# the box (test_dist / density_test) instead of a valid-selected top-5 -- see
# design.md's "Directional sampling".
DS_EVAL_TEST=1
# score, not the "loss" default: the existing coeff_search cell for this unit
# predates loss tracking (valid_avg_loss is null), so the score threshold is
# the one actually available as a density reference. Costs a generate() pass
# per draw -- see design.md's "Directional sampling".
DS_SELECT_METRIC="score"

# gen batch 32, up from 16 (2026-09-14): the 0.6b TIES script already runs 32
# on the same 24 GB cards, and TA's three extra resident basis copies are only
# ~1.2 GiB each in bf16 at 0.6B. With a test pass on every draw generate()
# is the whole budget, so the larger batch roughly halves the run.
EXTRA_ARGS="--gen-batch-size 32 --loss-batch-size 2"
run_eval
