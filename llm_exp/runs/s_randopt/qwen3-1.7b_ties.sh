#!/usr/bin/env bash
# Directional sampling: Qwen3-1.7B x TIES, 4 tasks (full).
# coeff_best lambda* and the density threshold are read from $COEFF_SOURCE_DIR
# (the main results root), same unit name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
METHOD="ties"
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
# The main results root's TIES coeff_search sweep stops at lambda=1.0 and its
# lambda* is pinned to that boundary -- the real optimum lies outside it. The
# 41-point grid in results/ties_coeff_grid found lambda*=1.75 (valid
# 0.7042 vs the coarse sweep's), so read coeff_search from there instead.
COEFF_SOURCE_DIR="${COEFF_SOURCE_DIR:-results/ties_coeff_grid}"

# coeff_best (lambda*=1.75 from the grid above). subspace_best is doubly unavailable for this
# cell: the qwen3-1.7b TIES unit has no subspace_gd results at all, and TIES is
# single-direction (n_dirs=1) so that init would only supply another scalar
# rather than the per-task vector that makes it interesting for TA.
DS_INIT="coeff_best"          # pretrained | avg | merged | coeff_best | subspace_best
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
# score selection, matching the other directional_sampling runs so the archs
# stay comparable. (This unit does have coeff_search.valid_avg_loss = 2.6455,
# so DS_SELECT_METRIC="loss" would also work and is far cheaper per draw.)
DS_SELECT_METRIC="score"

# 1.7b weights: smaller eval batches than the 0.6b scripts (see the 1.7b TA
# script). TIES keeps one merged direction resident rather than TA's n_tasks
# basis copies, so this is the lighter of the two 1.7b runs.
EXTRA_ARGS="--gen-batch-size 16 --loss-batch-size 2"
run_eval
