#!/usr/bin/env bash
# Draw the subspace-escape trajectory figure from full_2task_ta.sh's results.json
# (no GPU, no retraining) -> results/subspace_escape/full_2task_ta/trajectory_hero.{png,pdf}
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
"${PYTHON:-python}" scripts/plots/plot_subspace_escape.py
