#!/usr/bin/env bash
# Shared helpers for the eval run scripts.
#
# Each run script declares the full config as plain variables (so you can read
# off exactly what is being run), then calls `run_eval`:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
#   ARCH="qwen3-0.6b"
#   BASE_MODEL="Qwen/Qwen3-0.6B-Base"
#   METHOD="ta"
#   STRATEGIES="coeff_search,weight_gd,subspace_gd,baselines"
#   WEIGHT_GD_INITS="pretrained,avg,merged,coeff_best"    # init points for weight_gd
#   WEIGHT_GD_LORA_INITS="pretrained,avg,merged,coeff_best"  # init points for weight_gd_lora
#   SUBSPACE_GD_INITS="pretrained,avg,merged,coeff_best"  # init points for subspace_gd
#   BUDGETS="full"                                        # comma list; per-class ints and/or full
#   TASKS=(bank77 ddxplus ifeval usefulness_judge)
#   UNSEEN_TASKS=(gsm8k mbpp)         # optional: held-out tasks, test-only (no task vectors)
#   UNSEEN_TEST_SAMPLES=300           # optional: cap each unseen test pool
#   UNSEEN_N_SHOT=auto                # optional: few-shot unseen prompts (auto = gsm8k 4 / mbpp 3)
#   run_eval
#
# runs/ood_generalization/ scores an EXISTING unit's cells on unseen tasks
# without re-running GD (scripts/ood_eval.py) via `run_ood_eval`, which reads
# the same ARCH / METHOD / TASKS / UNSEEN_TASKS / UNSEEN_TEST_SAMPLES plus:
#   UNIT_DIR          results root holding the source unit (default: results)
#   OOD_CELLS         comma list of cells to score (default: all but directional_sampling)
#   OOD_RAW_PROMPT=1  raw-completion prompts (pretrained *-Base reference; pair with OOD_CELLS=baselines/pretrained)
#   CHECKPOINT_ROOT   where the GD checkpoints live (default: checkpoints/finetuned;
#                     admin/download_finetuned.sh fills it from GCS)
#
# directional_sampling adds its own optional knobs (all forwarded only when
# set, so scripts that don't run it stay unchanged):
#   DS_INIT / DS_ALPHA / DS_BETA / DS_SAMPLES / DS_SEED / DS_SELECT_METRIC / DS_EVAL_TEST
#   DS_BASIS_DEVICE (auto | cuda | cpu) / DS_REEVAL_CENTER
#   COEFF_SOURCE_DIR  results root to read coeff_search from (default: results)
# bo_search likewise (forwarded only when set):
#   BO_INITS / BO_TRIALS / BO_STARTUP_TRIALS / BO_RADIUS / BO_LOWER / BO_SEED / BO_BASIS_DEVICE
#
# No plotting step: results land in results/units/*.json and
# results/logs/*.log; draw figures with scripts/plots/*.py.
#
# Infra-only knobs (not "what we run") stay as optional env vars:
#   PYTHON            python interpreter (default: python)
#   TASK_VECTORS_DIR  checkpoint folder (default: checkpoints/task_vectors)
#   EXTRA_ARGS        extra flags for scripts/merge_eval.py (e.g. --save-checkpoints)
#   OUT_DIR           results root (default: results)
#   FORCE             comma-separated strategies to recompute even if already
#                     present. All-or-nothing per strategy: it redoes EVERY
#                     requested init of that strategy.
set -euo pipefail

LLM_EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# Honour GPU= when a run script is invoked directly (admin/*.sh export
# CUDA_VISIBLE_DEVICES themselves, and an existing value wins).
if [ -n "${GPU:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

join_by() { local IFS="$1"; shift; echo "$*"; }

_unseen_csv() {
    if [ -n "${UNSEEN_TASKS+x}" ] && [ "${#UNSEEN_TASKS[@]}" -gt 0 ]; then
        join_by , "${UNSEEN_TASKS[@]}"
    fi
}

# Consumes the config declared by the run script: ARCH, BASE_MODEL, METHOD,
# STRATEGIES, WEIGHT_GD_INITS, SUBSPACE_GD_INITS, BUDGETS, TASKS. Checkpoints
# are resolved per (task, arch) from $TASK_VECTORS_DIR by scripts/merge_eval.py.
run_eval() {
    : "${ARCH:?declare ARCH in the run script}"
    : "${BASE_MODEL:?declare BASE_MODEL in the run script}"
    : "${METHOD:?declare METHOD in the run script}"
    : "${STRATEGIES:?declare STRATEGIES in the run script}"
    : "${WEIGHT_GD_INITS:?declare WEIGHT_GD_INITS in the run script}"
    : "${WEIGHT_GD_LORA_INITS:=$WEIGHT_GD_INITS}"
    : "${SUBSPACE_GD_INITS:?declare SUBSPACE_GD_INITS in the run script}"
    : "${BUDGETS:?declare BUDGETS in the run script}"
    if [ "${#TASKS[@]}" -eq 0 ]; then
        echo "declare TASKS=(...) in the run script" >&2; return 1
    fi

    local tasks_csv budget unseen_csv
    local -a ds_args
    tasks_csv="$(join_by , "${TASKS[@]}")"
    unseen_csv="$(_unseen_csv)"

    ds_args=()
    [ -n "$unseen_csv" ] && ds_args+=(--unseen-tasks "$unseen_csv")
    [ -n "${UNSEEN_TEST_SAMPLES:-}" ] && ds_args+=(--unseen-test-samples "$UNSEEN_TEST_SAMPLES")
    [ -n "${UNSEEN_N_SHOT:-}" ] && ds_args+=(--n-shot "$UNSEEN_N_SHOT")
    [ -n "${DS_INIT:-}" ] && ds_args+=(--ds-inits "$DS_INIT")
    [ -n "${DS_ALPHA:-}" ] && ds_args+=(--ds-alpha "$DS_ALPHA")
    [ -n "${DS_BETA:-}" ] && ds_args+=(--ds-beta "$DS_BETA")
    [ -n "${DS_SAMPLES:-}" ] && ds_args+=(--ds-samples "$DS_SAMPLES")
    [ -n "${DS_SEED:-}" ] && ds_args+=(--ds-seed "$DS_SEED")
    [ -n "${DS_SELECT_METRIC:-}" ] && ds_args+=(--ds-select-metric "$DS_SELECT_METRIC")
    case "${DS_EVAL_TEST:-}" in 1|true|yes) ds_args+=(--ds-eval-test) ;; esac
    [ -n "${DS_BASIS_DEVICE:-}" ] && ds_args+=(--ds-basis-device "$DS_BASIS_DEVICE")
    case "${DS_REEVAL_CENTER:-}" in 1|true|yes) ds_args+=(--ds-reeval-center) ;; esac
    [ -n "${COEFF_SOURCE_DIR:-}" ] && ds_args+=(--coeff-source-dir "$COEFF_SOURCE_DIR")
    [ -n "${BO_INITS:-}" ] && ds_args+=(--bo-inits "$BO_INITS")
    [ -n "${BO_TRIALS:-}" ] && ds_args+=(--bo-trials "$BO_TRIALS")
    [ -n "${BO_STARTUP_TRIALS:-}" ] && ds_args+=(--bo-startup-trials "$BO_STARTUP_TRIALS")
    [ -n "${BO_RADIUS:-}" ] && ds_args+=(--bo-radius "$BO_RADIUS")
    [ -n "${BO_LOWER:-}" ] && ds_args+=(--bo-lower "$BO_LOWER")
    [ -n "${BO_SEED:-}" ] && ds_args+=(--bo-seed "$BO_SEED")
    [ -n "${BO_BASIS_DEVICE:-}" ] && ds_args+=(--bo-basis-device "$BO_BASIS_DEVICE")

    IFS=',' read -ra budget_arr <<< "$BUDGETS"
    for budget in "${budget_arr[@]}"; do
        echo "======================================================================"
        echo "arch=$ARCH  method=$METHOD  budget=$budget"
        echo "strategies=$STRATEGIES"
        echo "weight_gd_inits=$WEIGHT_GD_INITS  weight_gd_lora_inits=$WEIGHT_GD_LORA_INITS  subspace_gd_inits=$SUBSPACE_GD_INITS"
        if [ -n "${FORCE:-}" ]; then echo "force=$FORCE  (ALL inits of these strategies are recomputed)"; fi
        echo "tasks=$tasks_csv  unseen=${unseen_csv:-}"
        echo "======================================================================"
        (cd "$LLM_EXP_ROOT" && "$PYTHON" scripts/merge_eval.py \
            --base-model "$BASE_MODEL" --arch "$ARCH" --method "$METHOD" \
            --tasks "$tasks_csv" \
            --task-vectors-dir "${TASK_VECTORS_DIR:-checkpoints/task_vectors}" \
            --budget "$budget" \
            --strategies "$STRATEGIES" \
            --weight-gd-inits "$WEIGHT_GD_INITS" \
            --weight-gd-lora-inits "$WEIGHT_GD_LORA_INITS" \
            --subspace-gd-inits "$SUBSPACE_GD_INITS" \
            --force "${FORCE:-}" \
            --out-dir "${OUT_DIR:-results}" \
            "${ds_args[@]}" \
            ${EXTRA_ARGS:-})
    done
}

# Post-hoc OOD eval of an existing unit (runs/ood_generalization/*.sh): loads
# each finished cell's weights (task-vector merges rebuilt, GD cells from their
# saved checkpoints) and scores UNSEEN_TASKS. Writes to
# $OUT_DIR/units/<unit>__unseen_<tasks>__...json (default OUT_DIR: results/ood_generalization).
run_ood_eval() {
    : "${ARCH:?declare ARCH in the run script}"
    : "${METHOD:?declare METHOD in the run script}"
    if [ "${#TASKS[@]}" -eq 0 ]; then echo "declare TASKS=(...) in the run script" >&2; return 1; fi
    if [ -z "$(_unseen_csv)" ]; then echo "declare UNSEEN_TASKS=(...) in the run script" >&2; return 1; fi

    local unit
    local -a ood_args
    unit="${UNIT_DIR:-results}/units/${#TASKS[@]}_tasks_$(join_by - "${TASKS[@]}")__${ARCH}__${METHOD}__full.json"
    ood_args=()
    [ -n "${UNSEEN_TEST_SAMPLES:-}" ] && ood_args+=(--unseen-test-samples "$UNSEEN_TEST_SAMPLES")
    [ -n "${UNSEEN_N_SHOT:-}" ] && ood_args+=(--n-shot "$UNSEEN_N_SHOT")
    [ "${OOD_RAW_PROMPT:-0}" = "1" ] && ood_args+=(--raw-prompt)
    [ -n "${OOD_CELLS:-}" ] && ood_args+=(--cells "$OOD_CELLS")
    [ -n "${FORCE:-}" ] && ood_args+=(--force)

    echo "======================================================================"
    echo "OOD eval  arch=$ARCH  method=$METHOD  unseen=$(_unseen_csv)  unit=$unit"
    echo "======================================================================"
    (cd "$LLM_EXP_ROOT" && "$PYTHON" scripts/ood_eval.py \
        --unit "$unit" \
        --unseen-tasks "$(_unseen_csv)" \
        --task-vectors-dir "${TASK_VECTORS_DIR:-checkpoints/task_vectors}" \
        --checkpoint-root "${CHECKPOINT_ROOT:-checkpoints/finetuned}" \
        --out-dir "${OUT_DIR:-results/ood_generalization}" \
        "${ood_args[@]}" \
        ${EXTRA_ARGS:-})
}
