#!/usr/bin/env bash
# Cost estimation: 4-task TA at full budget, Qwen3 sizes back-to-back on one GPU.
# One warmed-up action per strategy, scaled by lambda_steps / bo_trials /
# gd_epochs / subspace_epochs (10 / 50 / 5 / 5). Scores are not re-measured --
# plot.sh reads test_avg_score out of the main results root ($COEFF_SOURCE_DIR).
#
# Protocol: every strategy runs the plainest layout that fits on one card, so
# peak memory reflects what each method fundamentally needs (M = one bf16
# model copy, N = number of tasks) rather than any fit-it-on-a-smaller-card
# trick:
#   coeff_search   M + activations      merge built on the CPU, forward only
#   bo_search      M + activations      basis stays on the CPU (bo_basis_device=cpu)
#   weight_gd_lora M + adapters + activations
#   subspace_gd    (N + 2) M + grad + activations   base + N directions resident
#                                       on the GPU (subspace_basis_device=cuda)
# and the SAME per-step batch (BS below, default 4 = the main runs'
# gd_batch_size) for generate and for the GD micro-batches, no gradient
# accumulation. The GD loss is Liger's fused linear cross-entropy
# (_common.lm_loss): HF's default materializes fp32 logits over the 152k vocab
# (~3.7 GiB per sequence, ~15 GiB at batch 4) which is an artifact of the loss
# implementation, not of any method -- and generate() already skips those
# logits for the search strategies. Gradient checkpointing stays on: it is the
# GD strategies' fixed recipe (charged in FLOPs), and without it 1.7B
# subspace_gd at batch 4 would not fit a 32 GB card.
#
# Per-arch spec is "<arch>:<base_model>:<batch_size>[:<strategies>]"; the
# optional 4th field overrides $STRATEGIES for that arch. Override the list
# with ARCHES="..." (space-separated specs).
#   qwen3-1.7b     subspace_gd needs ~24 GB under this protocol -> a 32 GB card.
#   qwen3-4b       coeff_search, bo_search, weight_gd_lora only (~10 / 10 / 13 GB):
#                  subspace_gd would need (N+2)M + grad = 56 GB, and the main
#                  results have no subspace_gd cell at 4B anyway.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

METHOD="ta"
STRATEGIES="${STRATEGIES:-coeff_search,bo_search,weight_gd_lora,subspace_gd}"
TASKS=(bank77 ddxplus ifeval usefulness_judge)
OUT_DIR="${OUT_DIR:-results/cost_estimation}"
COEFF_SOURCE_DIR="${COEFF_SOURCE_DIR:-results}"
PLOT="${PLOT:-1}"
ARCHES="${ARCHES:-qwen3-0.6b:Qwen/Qwen3-0.6B-Base:4 qwen3-1.7b:Qwen/Qwen3-1.7B-Base:4 qwen3-4b:Qwen/Qwen3-4B-Base:4:coeff_search,bo_search,weight_gd_lora}"

tasks_csv="$(join_by , "${TASKS[@]}")"

for spec in $ARCHES; do
    IFS=':' read -r ARCH BASE_MODEL BS ARCH_STRATEGIES <<< "$spec"
    ARCH_STRATEGIES="${ARCH_STRATEGIES:-$STRATEGIES}"
    echo "======================================================================"
    echo "cost_estimation  method=$METHOD  arch=$ARCH  base=$BASE_MODEL  batch=$BS (generate + GD micro-batch)"
    echo "strategies=$ARCH_STRATEGIES"
    echo "======================================================================"
    (cd "$LLM_EXP_ROOT" && "$PYTHON" scripts/profile_cost.py \
        --base-model "$BASE_MODEL" --arch "$ARCH" --method "$METHOD" \
        --tasks "$tasks_csv" \
        --task-vectors-dir "${TASK_VECTORS_DIR:-checkpoints/task_vectors}" \
        --budget full --seed 42 \
        --strategies "$ARCH_STRATEGIES" \
        --force "${FORCE:-}" \
        --gen-batch-size "$BS" \
        --gd-batch-size "$BS" --gd-grad-accum-steps 1 \
        --subspace-batch-size "$BS" --subspace-grad-accum-steps 1 \
        --bo-basis-device cpu --subspace-basis-device cuda \
        --out-dir "$OUT_DIR" \
        --coeff-source-dir "$COEFF_SOURCE_DIR" \
        ${EXTRA_ARGS:-})
done

if [ "$PLOT" = "1" ]; then
    bash "$(dirname "${BASH_SOURCE[0]}")/plot.sh"
fi
