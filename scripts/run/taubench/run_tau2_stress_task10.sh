#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

export TASK_IDS="${TASK_IDS:-0 1 3 4 5 6 10 13 26 28}"
export TASK_TAG="${TASK_TAG:-task10}"
export NUM_TASKS="${NUM_TASKS:-10}"

exec bash "$ROOT_DIR/scripts/run/taubench/run_tau2_stress_test_phase2_gpu567.sh" "$@"