#!/usr/bin/env bash
# Shared helpers for the merge-experiment run scripts.
#
# Each run script declares config as plain variables, then calls `run_merge`:
#
#   ARCH="vit-b-32"
#   METHOD="ta"
#   STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
#   WEIGHT_GD_INITS="pretrained,merged,coeff_best"
#   SUBSPACE_GD_INITS="pretrained,merged,coeff_best"
#   BATCH_SIZE=32
#   BUDGETS="full"                       # or "full,64,32,..." for data_scaling
#   TASKS=(eurosat gtsrb)
#   OUT_DIR="results/main_exp"           # results root (units/ logs/ plots/)
#   SEED=42                              # or SEED="43,44" for several non-full draws
#   # GD_OPTIMIZER=sgd GD_MOMENTUM=0     # optional weight_gd optimizer overrides
#   # GD_LR=1e-5 GD_EPOCHS=10 GD_WARMUP_RATIO=0.1 GD_PATIENCE=5 GD_L2_SP=0.1
#   # GD_LR=1e-5 GD_EPOCHS=10 GD_WARMUP_RATIO=0.1 GD_PATIENCE=5
#   # GD_LABEL_SOURCE=gt|expert_soft|expert_hard   # weight_gd training signal
#   # UNLABELED_SOURCE=test UNLABELED_SAMPLES=      # data for adamerging/divmerge/expert_*
#   # DIVERGENCE=js ADA_VARIANTS= DIV_VARIANTS=      # (variants: taskwise,layerwise)
#   # ADA_STEPS= ADA_LR= ADA_PRIOR= DIV_STEPS= DIV_LR= DIV_PRIOR=
#   # SUBSPACE_PATIENCE=5                # early-stop (0 = off); Python defaults are 5
#   run_merge
#
# Seed rules:
#   * valid pool partition is fixed in Python (never varies with SEED)
#   * SEED (one value or comma list) only affects apply_valid_budget subset selection
#   * budget=full always uses seed 42 (no __seed suffix; selection is a no-op)
#   * other budgets append __seed{N}
#
# Optional env: PYTHON, TASK_VECTORS_DIR, EXTRA_ARGS, PLOT=0|1, PLOT_METRIC, WANDB=0,
#               WANDB_PROJECT, WANDB_GROUP,
#               DS_INIT, DS_ALPHA, DS_BETA, DS_SAMPLES, DS_SEED,
#               BO_INITS, BO_TRIALS, BO_STARTUP_TRIALS, BO_RADIUS, BO_LOWER, BO_SEED,
#               NUM_CLASSES, CLASS_SEED
set -euo pipefail

VISION_EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

arch_base_model() {
    case "$1" in
        vit-b-32) echo "openai/clip-vit-base-patch32" ;;
        vit-b-16) echo "openai/clip-vit-base-patch16" ;;
        vit-l-14) echo "openai/clip-vit-large-patch14" ;;
        *) echo "unknown arch: $1" >&2; return 1 ;;
    esac
}

join_by() { local IFS="$1"; shift; echo "$*"; }

# Resolve selection seeds from SEED (single value or comma list; default 42).
_selection_seeds() {
    local raw="${SEED:-42}" s
    IFS=',' read -ra s <<< "$raw"
    printf '%s\n' "${s[@]}"
}

run_merge() {
    : "${ARCH:?declare ARCH in the run script}"
    : "${METHOD:?declare METHOD in the run script}"
    : "${STRATEGIES:?declare STRATEGIES in the run script}"
    : "${WEIGHT_GD_INITS?declare WEIGHT_GD_INITS in the run script (may be empty)}"
    : "${SUBSPACE_GD_INITS?declare SUBSPACE_GD_INITS in the run script (may be empty)}"
    : "${BATCH_SIZE:?declare BATCH_SIZE in the run script (b-32=32, b-16=16, l-14=4)}"
    : "${BUDGETS:?declare BUDGETS in the run script}"
    if [ "${#TASKS[@]}" -eq 0 ]; then
        echo "declare TASKS=(...) in the run script" >&2; return 1
    fi

    local base_model tasks_csv budget seed out_dir script_dir wandb_group
    local -a seeds gd_args wandb_args
    base_model="$(arch_base_model "$ARCH")"
    tasks_csv="$(join_by , "${TASKS[@]}")"
    out_dir="${OUT_DIR:-results/main_exp}"
    mapfile -t seeds < <(_selection_seeds)
    gd_args=()
    [ -n "${GD_OPTIMIZER:-}" ] && gd_args+=(--gd-optimizer "$GD_OPTIMIZER")
    [ -n "${GD_MOMENTUM:-}" ] && gd_args+=(--gd-momentum "$GD_MOMENTUM")
    [ -n "${GD_LR:-}" ] && gd_args+=(--gd-lr "$GD_LR")
    [ -n "${GD_EPOCHS:-}" ] && gd_args+=(--gd-epochs "$GD_EPOCHS")
    [ -n "${GD_WARMUP_RATIO:-}" ] && gd_args+=(--gd-warmup-ratio "$GD_WARMUP_RATIO")
    [ -n "${GD_PATIENCE:-}" ] && gd_args+=(--gd-patience "$GD_PATIENCE")
    [ -n "${GD_L2_SP:-}" ] && gd_args+=(--gd-l2-sp "$GD_L2_SP")
    [ -n "${GD_LABEL_SOURCE:-}" ] && gd_args+=(--gd-label-source "$GD_LABEL_SOURCE")
    [ -n "${UNLABELED_SOURCE:-}" ] && gd_args+=(--unlabeled-source "$UNLABELED_SOURCE")
    [ -n "${UNLABELED_SAMPLES:-}" ] && gd_args+=(--unlabeled-samples "$UNLABELED_SAMPLES")
    [ -n "${DIVERGENCE:-}" ] && gd_args+=(--divergence "$DIVERGENCE")
    [ -n "${ADA_VARIANTS:-}" ] && gd_args+=(--ada-variants "$ADA_VARIANTS")
    [ -n "${DIV_VARIANTS:-}" ] && gd_args+=(--div-variants "$DIV_VARIANTS")
    [ -n "${ADA_STEPS:-}" ] && gd_args+=(--ada-steps "$ADA_STEPS")
    [ -n "${ADA_LR:-}" ] && gd_args+=(--ada-lr "$ADA_LR")
    [ -n "${ADA_PRIOR:-}" ] && gd_args+=(--ada-prior "$ADA_PRIOR")
    [ -n "${DIV_STEPS:-}" ] && gd_args+=(--div-steps "$DIV_STEPS")
    [ -n "${DIV_LR:-}" ] && gd_args+=(--div-lr "$DIV_LR")
    [ -n "${DIV_PRIOR:-}" ] && gd_args+=(--div-prior "$DIV_PRIOR")
    [ -n "${SUBSPACE_PATIENCE:-}" ] && gd_args+=(--subspace-patience "$SUBSPACE_PATIENCE")
    [ -n "${DS_INIT:-}" ] && gd_args+=(--ds-inits "$DS_INIT")
    [ -n "${DS_ALPHA:-}" ] && gd_args+=(--ds-alpha "$DS_ALPHA")
    [ -n "${DS_BETA:-}" ] && gd_args+=(--ds-beta "$DS_BETA")
    [ -n "${DS_SAMPLES:-}" ] && gd_args+=(--ds-samples "$DS_SAMPLES")
    [ -n "${DS_SEED:-}" ] && gd_args+=(--ds-seed "$DS_SEED")
    [ -n "${BO_INITS:-}" ] && gd_args+=(--bo-inits "$BO_INITS")
    [ -n "${BO_TRIALS:-}" ] && gd_args+=(--bo-trials "$BO_TRIALS")
    [ -n "${BO_STARTUP_TRIALS:-}" ] && gd_args+=(--bo-startup-trials "$BO_STARTUP_TRIALS")
    [ -n "${BO_RADIUS:-}" ] && gd_args+=(--bo-radius "$BO_RADIUS")
    [ -n "${BO_LOWER:-}" ] && gd_args+=(--bo-lower "$BO_LOWER")
    [ -n "${BO_SEED:-}" ] && gd_args+=(--bo-seed "$BO_SEED")
    [ -n "${NUM_CLASSES:-}" ] && gd_args+=(--num-classes "$NUM_CLASSES")
    [ -n "${CLASS_SEED:-}" ] && gd_args+=(--class-seed "$CLASS_SEED")
    script_dir="$(basename "$(cd "$(dirname "$0")" && pwd)")"
    wandb_group="${WANDB_GROUP:-${script_dir}/${ARCH}/${METHOD}}"
    wandb_args=()
    if [ "${WANDB:-1}" = "0" ]; then
        wandb_args=(--no-wandb)
    else
        [ -n "${WANDB_PROJECT:-}" ] && wandb_args+=(--wandb-project "$WANDB_PROJECT")
    fi

    IFS=',' read -ra budget_arr <<< "$BUDGETS"
    for budget in "${budget_arr[@]}"; do
        local -a seeds_for_budget
        if [ "$budget" = "full" ]; then
            seeds_for_budget=(42)
        else
            seeds_for_budget=("${seeds[@]}")
        fi
        for seed in "${seeds_for_budget[@]}"; do
            echo "======================================================================"
            echo "arch=$ARCH  method=$METHOD  budget=$budget  seed=$seed  out_dir=$out_dir"
            echo "strategies=$STRATEGIES"
            echo "weight_gd_inits=$WEIGHT_GD_INITS  subspace_gd_inits=$SUBSPACE_GD_INITS"
            echo "batch_size=$BATCH_SIZE  tasks=$tasks_csv"
            if [ -n "${CLASS_SEED:-}" ]; then
                echo "class_seed=$CLASS_SEED  num_classes=${NUM_CLASSES:-}"
            fi
            echo "======================================================================"
            (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/merge_eval.py \
                --base-model "$base_model" --arch "$ARCH" --method "$METHOD" \
                --tasks "$tasks_csv" \
                --task-vectors-dir "${TASK_VECTORS_DIR:-checkpoints/task_vectors}" \
                --budget "$budget" \
                --seed "$seed" \
                --strategies "$STRATEGIES" \
                --weight-gd-inits "$WEIGHT_GD_INITS" \
                --subspace-gd-inits "$SUBSPACE_GD_INITS" \
                --batch-size "$BATCH_SIZE" \
                --out-dir "$out_dir" \
                --wandb-group "$wandb_group" \
                "${wandb_args[@]}" \
                "${gd_args[@]}" \
                ${EXTRA_ARGS:-})
        done
    done

    if [ "${PLOT:-1}" = "1" ]; then
        run_default_plot "$tasks_csv"
    fi
}

run_default_plot() {
    local tasks_csv="$1" out_dir metric script_dir
    out_dir="${OUT_DIR:-results/main_exp}"
    metric="${PLOT_METRIC:-test_avg_acc}"
    script_dir="$(basename "$(cd "$(dirname "$0")" && pwd)")"

    case "$script_dir" in
        data_scaling)
            (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot.py data_scaling \
                --results-dir "$out_dir" --tasks "$tasks_csv" \
                --arch "$ARCH" --method "$METHOD" --metric "$metric")
            ;;
        full_comparison)
            (cd "$VISION_EXP_ROOT" && "$PYTHON" scripts/plots/plot.py full_comparison \
                --results-dir "$out_dir" --tasks "$tasks_csv" \
                --arch "$ARCH" --budget full --metric "$metric")
            ;;
        *)
            echo "run_default_plot: unknown folder '$script_dir'; skipping plot (set PLOT=0 to silence)" >&2
            ;;
    esac
}
