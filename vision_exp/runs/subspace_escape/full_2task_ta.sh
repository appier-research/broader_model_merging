#!/usr/bin/env bash
# Full-scale run of scripts/subspace_escape.py: whole valid pool, whole
# test split, and enough weight_gd epochs to see real trajectory movement.
# For a quick sanity check before committing to this, edit VALID_K /
# TEST_SAMPLES / EPOCHS / NUM_POINTS below down to a few batches, 1 epoch.
#
# Idempotent: rerunning with the same --out-dir skips any row already in its
# results.json (including skipping coeff_search/subspace_gd entirely if their
# rows are cached) -- so raising NUM_RANDOM later only computes the new
# points. FORCE=1 recomputes everything.
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- what to run ------------------------------------------------------------
ARCH="vit-b-32"
METHOD="ta"
TASKS="eurosat,gtsrb"                          # exactly 2 (2-D task-vector subspace + MTR = 3D figure)
TASK_VECTORS_DIR="checkpoints/task_vectors"

# --- the three things you asked to control ----------------------------------
NUM_RANDOM=1                                    # how many random points on S to probe
NUM_POINTS=40                                   # 41 points per row (+1 for the t=0 start). 20 left the early
                                                #   LR transient spanning only ~4 points; 40 resolves it.
                                                #   Each point is a full valid + full test eval, so this is
                                                #   the main cost: ~6.4h vs ~3.4h at 20.
                                                # Each point costs a full valid + test eval pass.
RANDOM_SEED=0

# --- data budget: whole valid pool, whole test split ------------------------
VALID_K="full"                                  # "full" = the entire held-out valid pool, no per-class cap
TEST_SAMPLES=""                                 # empty = no cap (full test split); set a number to limit it
BATCH_SIZE=32                                   # vit-b-32 default; vit-b-16=16, vit-l-14=4 (GPU-memory bound)
DATA_SEED=42

# --- coeff_search (used for the coeff_best row) -----------------------------
LAMBDA_MIN=0.0
LAMBDA_MAX=1.0
LAMBDA_STEPS=11


# --- weight_gd, run from every row's point ----------------------------------
EPOCHS=10                                       # real training length; raise if trajectories still look flat
LR=3e-5
OPTIMIZER="adamw"                               # adamw | sgd
MOMENTUM=0.0
WARMUP_RATIO=0.1
PATIENCE=0                                      # 0 = no early stop -- keeps every row's trajectory full-length
                                                 # and comparable in the figure; set >0 if you only care about
                                                 # the table (final test_acc), not the trajectory shape.

# --- random-point coefficient range ------------------------------------------
RANDOM_COEFF_MIN=0.0
RANDOM_COEFF_MAX=1.0

OUT_DIR="results/subspace_escape/full_2task_ta"
DEVICE=""                                       # empty = auto (cuda if available, else cpu)
FORCE=0                                          # 1 = ignore results.json, recompute every row

case "$ARCH" in
    vit-b-32) BASE_MODEL="openai/clip-vit-base-patch32" ;;
    vit-b-16) BASE_MODEL="openai/clip-vit-base-patch16" ;;
    vit-l-14) BASE_MODEL="openai/clip-vit-large-patch14" ;;
    *) echo "unknown arch: $ARCH" >&2; exit 1 ;;
esac

echo "======================================================================"
echo "tasks=$TASKS  arch=$ARCH  method=$METHOD"
echo "num_random=$NUM_RANDOM  num_points=$NUM_POINTS  epochs=$EPOCHS  valid_k=$VALID_K"
echo "out_dir=$OUT_DIR"
echo "======================================================================"

device_args=()
[ -n "$DEVICE" ] && device_args=(--device "$DEVICE")
test_samples_args=()
[ -n "$TEST_SAMPLES" ] && test_samples_args=(--test-samples "$TEST_SAMPLES")
force_args=()
[ "$FORCE" = "1" ] && force_args=(--force)
(cd "$SCRIPT_DIR" && "$PYTHON" scripts/subspace_escape.py \
    --base-model "$BASE_MODEL" \
    --arch "$ARCH" --method "$METHOD" --tasks "$TASKS" \
    --task-vectors-dir "$TASK_VECTORS_DIR" \
    --num-random "$NUM_RANDOM" --num-points "$NUM_POINTS" --random-seed "$RANDOM_SEED" \
    --random-coeff-min "$RANDOM_COEFF_MIN" --random-coeff-max "$RANDOM_COEFF_MAX" \
    --valid-k "$VALID_K" --batch-size "$BATCH_SIZE" --data-seed "$DATA_SEED" \
    "${test_samples_args[@]}" \
    --lambda-min "$LAMBDA_MIN" --lambda-max "$LAMBDA_MAX" --lambda-steps "$LAMBDA_STEPS" \
    --epochs "$EPOCHS" --lr "$LR" --optimizer "$OPTIMIZER" --momentum "$MOMENTUM" \
    --warmup-ratio "$WARMUP_RATIO" --patience "$PATIENCE" \
    --out-dir "$OUT_DIR" \
    "${device_args[@]}" \
    "${force_args[@]}")

echo
echo "table   -> $OUT_DIR/table.csv"
echo "results -> $OUT_DIR/results.json  (rerun is idempotent)"
echo "figure: bash runs/subspace_escape/plot.sh"
