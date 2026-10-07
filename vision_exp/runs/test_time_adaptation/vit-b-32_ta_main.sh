#!/usr/bin/env bash
# Test-time adaptation: ViT-B/32 x Task Arithmetic. Every method below trains on the
# SAME unlabeled test data (UNLABELED_SAMPLES per task; empty = whole test split,
# transductive like the AdaMerging paper) and is evaluated on the test split:
#   adamerging  taskwise coefficients, entropy minimisation
#   divmerge    taskwise coefficients, JS to task-expert logits
#   weight_gd   full weights, expert_soft (JS to expert logits) / expert_hard (expert argmax)
#   baselines   pretrained / avg / merged / multitask upper bound (test-only references)
# Results: results/test_time_adaptation<tag>/units/<unit>.json under strategies.{adamerging,
# divmerge,weight_gd_expert_soft,weight_gd_expert_hard,baselines}.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

ARCH="vit-b-32"
METHOD="ta"
# coeff_best is excluded: lambda* comes from the labeled coeff_search, which
# would leak labels into test-time adaptation.
WEIGHT_GD_INITS="${WEIGHT_GD_INITS:-avg}"
SUBSPACE_GD_INITS=""
BATCH_SIZE=32
BUDGETS="full"
# Only the taskwise coefficient granularity (skip layerwise).
ADA_VARIANTS="${ADA_VARIANTS:-taskwise}"
DIV_VARIANTS="${DIV_VARIANTS:-taskwise}"
TASKS=(dtd eurosat fer2013 food101 gtsrb mnist resisc45 stanford-cars sun397)

export UNLABELED_SOURCE="${UNLABELED_SOURCE:-test}"
export UNLABELED_SAMPLES="${UNLABELED_SAMPLES:-1000}"      # per task; empty = all
OUT_DIR="${OUT_DIR:-results/test_time_adaptation${UNLABELED_SAMPLES:+_n$UNLABELED_SAMPLES}}"
PLOT=0

# Pass 1: coefficient baselines + weight_gd with soft pseudo-labels.
STRATEGIES="adamerging,divmerge,weight_gd"
export GD_LABEL_SOURCE=expert_soft
run_merge
# Pass 2: weight_gd with hard pseudo-labels (adamerging/divmerge cells already present -> skipped).
STRATEGIES="weight_gd"
export GD_LABEL_SOURCE=expert_hard
run_merge
