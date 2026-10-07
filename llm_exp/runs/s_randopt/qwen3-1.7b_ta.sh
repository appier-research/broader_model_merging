#!/usr/bin/env bash
# Directional sampling: Qwen3-1.7B x Task Arithmetic, 4 tasks (full).
# coeff_best lambda* and the density threshold are read from $COEFF_SOURCE_DIR
# (the main results root), same unit name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="qwen3-1.7b"
BASE_MODEL="Qwen/Qwen3-1.7B-Base"
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

# bo_best: bo_search's best trial on this unit (valid 0.8536 / test 0.8051)
# beats the sweep's lambda*=0.6 (0.8098 / 0.7365) and subspace_gd/merged
# (0.6052), and even the coeff_best-centered directional_sampling top-1 (test
# 0.7697) -- so sample around the BO point. The coeff_best cell already in the
# unit JSON is kept; merge_eval.py only computes inits that are missing.
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
# score selection, matching the 0.6b run so the two archs stay comparable.
# (Unlike 0.6b, this unit does have coeff_search.valid_avg_loss, so
# DS_SELECT_METRIC="loss" would also work here and is far cheaper per draw.)
DS_SELECT_METRIC="score"

# GPU residency is n_dirs + 3 full-model copies (model, W*, 4 task vectors,
# g_perp; see directional_sampling.py), ~22.4 GiB in bf16 for 1.7B -- this
# needs a 32 GB card (GPU 2/3 here), a 24 GB 3090 cannot hold it. On 32 GB
# that leaves ~9 GiB for the gradient pass's activations and generate()'s KV
# cache, hence smaller eval batches than the 0.6b script's 32/4. Measured on
# GPU 2 (2026-09-05): gen batch 16 OOMs in the center eval's generate() (the
# process sat at 30.2 GiB), gen batch 8 is what fits. The loss batch is 1
# because the gradient pass upcasts the (batch, seq, 151936-vocab) logits to
# fp32 and keeps their grad alive too -- several GiB per sequence.
EXTRA_ARGS="--gen-batch-size 8 --loss-batch-size 1"
run_eval
