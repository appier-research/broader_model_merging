#!/usr/bin/env bash
# Run several run scripts back-to-back on one GPU, logging each to its own file
# under admin/logs/. llm_exp/ and vision_exp/ keep identical copies.
#
#   GPU=0 bash admin/run_queue.sh runs/full_comparison/qwen3-1.7b_{ta,ties,dare,tsvm}.sh
#   GPU=1 TAG=s_randopt bash admin/run_queue.sh runs/s_randopt/*.sh
#
# Paths are relative to this experiment's root. Run scripts only compute the
# cells missing from their unit JSON, so re-running a queue is cheap: complete
# units print "Nothing to run".
#
# Env: GPU (default 0), PYTHON (default python), TAG (optional log-dir label).
# Exits non-zero if any script failed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -eq 0 ]; then
    echo "usage: [GPU=0] [TAG=name] bash admin/run_queue.sh <run script>..." >&2
    exit 2
fi
EXPERIMENTS=("$@")

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHON="${PYTHON:-python}"

LOG_DIR="admin/logs/queue${TAG:+_$TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILED=()

echo "Queue started"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Logs: $LOG_DIR"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo

for exp in "${EXPERIMENTS[@]}"; do
    name="$(echo "$exp" | sed 's|/|__|g' | sed 's|\.sh$||')"
    log_file="$LOG_DIR/${name}.log"

    echo "============================================================"
    echo "START: $exp"
    echo "TIME:  $(date '+%F %T')"
    echo "LOG:   $log_file"
    echo "============================================================"

    if bash "$exp" 2>&1 | tee "$log_file"; then
        echo "OK: $exp"
    else
        echo "FAIL: $exp"
        FAILED+=("$exp")
    fi

    echo
done

echo "============================================================"
echo "Queue finished at $(date '+%F %T')"
echo "============================================================"

if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All ${#EXPERIMENTS[@]} experiments finished successfully."
else
    echo "${#FAILED[@]} experiment(s) failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
