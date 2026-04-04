#!/usr/bin/env bash
set -euo pipefail

# Stress test: sweep H2O heavy-hitter sizes with fixed window.
# Baseline and StreamingLLM serve as lower/upper bounds for comparison.
#
# This variant keeps Phase 1 on GPUs A/B/C and moves Phase 2 to GPUs D/E/F
# (default: 5, 6, 7).
#
# Usage:
#   bash scripts/run/taubench/run_tau2_stress_test_phase2_gpu567.sh
#   PHASE=1 bash scripts/run/taubench/run_tau2_stress_test_phase2_gpu567.sh
#   PHASE=2 bash scripts/run/taubench/run_tau2_stress_test_phase2_gpu567.sh
#
# Key overrides:
#   GPU_A=2 GPU_B=3 GPU_C=4 GPU_D=5 GPU_E=6 GPU_F=7 \
#   WINDOW_SIZE=1024 PHASE=2 bash scripts/run/taubench/run_tau2_stress_test_phase2_gpu567.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"

# Shared settings
MODEL_PATH="${MODEL_PATH:-./local_models/Qwen3.5-9B}"
DOMAIN="${DOMAIN:-airline}"
TASK_SPLIT="${TASK_SPLIT:-base}"
USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-4o}"
AGENT_LLM="openai/${SERVED_MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-800}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-800}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_TASKS="${NUM_TASKS:-30}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-256}"
EVICT_PERIOD="${EVICT_PERIOD:-16}"
SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
PHASE="${PHASE:-0}"  # 0=all, 1/2=specific phase
TASK_IDS="${TASK_IDS:-}"
TASK_TAG="${TASK_TAG:-}"

# Phase 1 GPU assignments
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
GPU_C="${GPU_C:-4}"

# Phase 2 GPU assignments (moved to another 3 GPUs)
GPU_D="${GPU_D:-5}"
GPU_E="${GPU_E:-6}"
GPU_F="${GPU_F:-7}"

# Fixed ports — each parallel job needs its own port.
PORT_BASELINE="${PORT_BASELINE:-8010}"
PORT_STREAMINGLLM="${PORT_STREAMINGLLM:-8011}"
PORT_H2O_64="${PORT_H2O_64:-8012}"
PORT_H2O_32="${PORT_H2O_32:-8013}"
PORT_H2O_128="${PORT_H2O_128:-8014}"
PORT_H2O_256="${PORT_H2O_256:-8015}"

# Environment
if [[ -f "$ROOT_DIR/tau2-bench/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/tau2-bench/.env"
  set +a
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[WARN] DEEPSEEK_API_KEY is empty. user-llm=$USER_LLM may fail if key is required."
fi

TAU2_SRC_DIR="$ROOT_DIR/tau2-bench/src"
if command -v tau2 >/dev/null 2>&1; then
  TAU2_CMD=(tau2)
  echo "[info] using tau2 from PATH"
elif [[ -d "$TAU2_SRC_DIR/tau2" ]]; then
  export PYTHONPATH="$TAU2_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  TAU2_CMD=(python -m tau2.cli)
  echo "[info] tau2 command not found; fallback to python -m tau2.cli"
else
  echo "[error] tau2 not found in PATH and local source missing at $TAU2_SRC_DIR"
  echo "[hint] install with: cd tau2-bench && pip install -e ."
  exit 1
fi

mkdir -p ./outputs

declare -a TASK_ID_ARRAY=()
SAVE_TO_SUFFIX=""

if [[ -n "$TASK_IDS" ]]; then
  normalized_task_ids="${TASK_IDS//,/ }"
  read -r -a TASK_ID_ARRAY <<< "$normalized_task_ids"
  if [[ "${#TASK_ID_ARRAY[@]}" -eq 0 ]]; then
    echo "[error] TASK_IDS is set but no valid task ids were parsed"
    exit 1
  fi
  TASK_TAG="${TASK_TAG:-subset}"
  SAVE_TO_SUFFIX="_${TASK_TAG}"
  if [[ "$NUM_TASKS" == "30" ]]; then
    NUM_TASKS="${#TASK_ID_ARRAY[@]}"
  fi
fi

wait_for_health() {
  local port="$1"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -sSf "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# run_config method gpu_id heavy port
#   method:  baseline | streamingllm | h2o
#   gpu_id:  CUDA device index
#   heavy:   heavy_hitter_size (ignored for baseline/streamingllm)
#   port:    TCP port for the api_server (must be unique per parallel job)
run_config() {
  local method="$1"
  local gpu_id="$2"
  local heavy="$3"
  local port="$4"

  local save_to
  local -a method_extra_args=()
  local -a tau2_selection_args=()

  case "$method" in
    baseline)
      save_to="stress_baseline_${NUM_TRIALS}x${NUM_TASKS}${SAVE_TO_SUFFIX}"
      ;;
    streamingllm)
      save_to="stress_streamingllm_${NUM_TRIALS}x${NUM_TASKS}_${SINK_SIZE}_${WINDOW_SIZE}${SAVE_TO_SUFFIX}"
      method_extra_args+=(
        --streaming-sink-size "$SINK_SIZE"
        --streaming-local-window-size "$WINDOW_SIZE"
      )
      ;;
    h2o)
      save_to="stress_h2o_${NUM_TRIALS}x${NUM_TASKS}_${SINK_SIZE}_${WINDOW_SIZE}_${heavy}${SAVE_TO_SUFFIX}"
      method_extra_args+=(
        --h2o-sink-size "$SINK_SIZE"
        --h2o-local-window-size "$WINDOW_SIZE"
        --h2o-heavy-hitter-size "$heavy"
      )
      ;;
  esac

  if [[ "${#TASK_ID_ARRAY[@]}" -gt 0 ]]; then
    tau2_selection_args+=(--task-ids "${TASK_ID_ARRAY[@]}")
  else
    tau2_selection_args+=(--num-tasks "$NUM_TASKS")
  fi

  local server_pid=""
  cleanup_config() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      echo "[cleanup] method=$method heavy=$heavy stop server pid=$server_pid"
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_config EXIT INT TERM

  local budget_desc
  case "$method" in
    baseline)     budget_desc="unbounded" ;;
    streamingllm) budget_desc="budget=$((SINK_SIZE + WINDOW_SIZE))" ;;
    h2o)          budget_desc="budget=$((SINK_SIZE + WINDOW_SIZE + heavy))" ;;
  esac

  echo "========================================"
  echo "[start] method=$method gpu=$gpu_id port=$port heavy=$heavy $budget_desc"

  CUDA_VISIBLE_DEVICES="$gpu_id" python api_server.py \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$port" \
    --method "$method" \
    --evict-period "$EVICT_PERIOD" \
    "${method_extra_args[@]}" \
    >"./outputs/${save_to}_server.log" 2>&1 &

  server_pid=$!
  echo "[info] method=$method server pid=$server_pid"

  if ! wait_for_health "$port"; then
    echo "[error] method=$method health check failed: http://${HOST}:${port}/health"
    return 1
  fi

  local agent_llm_args
  agent_llm_args=$(printf '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s}' "$HOST" "$port" "$AGENT_MAX_TOKENS")

  echo "[run] tau2 eval for method=$method heavy=$heavy"
  "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    "${tau2_selection_args[@]}" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_llm_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --save-to "$save_to" \
    >"./outputs/${save_to}_tau2.log" 2>&1

  echo "[done] method=$method heavy=$heavy save_to=$save_to"

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}

wait_group() {
  local -a pids=("$@")
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[error] job pid=$pid failed"
      failed=1
    fi
  done
  return "$failed"
}

# Phase 1: baseline + streamingllm + h2o(heavy=64)
run_phase1() {
  echo ""
  echo "=============================================================="
  echo "Phase 1: core 3-way comparison (window fixed)"
  echo "baseline | streamingllm(heavy=0) | h2o(heavy=64)"
  echo "GPUs: A=$GPU_A B=$GPU_B C=$GPU_C"
  echo "=============================================================="

  run_config "baseline"     "$GPU_A" 0  "$PORT_BASELINE"     &
  local pid_a=$!
  run_config "streamingllm" "$GPU_B" 0  "$PORT_STREAMINGLLM" &
  local pid_b=$!
  run_config "h2o"          "$GPU_C" 64 "$PORT_H2O_64"        &
  local pid_c=$!

  wait_group "$pid_a" "$pid_b" "$pid_c"
}

# Phase 2: h2o heavy sweep on GPUs 5/6/7 by default
run_phase2() {
  echo ""
  echo "=============================================================="
  echo "Phase 2: h2o heavy sweep (heavy=32,128,256)"
  echo "GPUs: D=$GPU_D E=$GPU_E F=$GPU_F"
  echo "=============================================================="

  run_config "h2o" "$GPU_D" 32  "$PORT_H2O_32"  &
  local pid_a=$!
  run_config "h2o" "$GPU_E" 128 "$PORT_H2O_128" &
  local pid_b=$!
  run_config "h2o" "$GPU_F" 256 "$PORT_H2O_256" &
  local pid_c=$!

  wait_group "$pid_a" "$pid_b" "$pid_c"
}

echo "=============================================================="
echo "H2O Heavy-Hitter Stress Test (Phase2 on GPU 5/6/7 variant)"
echo "Model:    $MODEL_PATH"
echo "Domain:   $DOMAIN  Split: $TASK_SPLIT"
echo "Trials:   $NUM_TRIALS  Tasks: $NUM_TASKS"
if [[ "${#TASK_ID_ARRAY[@]}" -gt 0 ]]; then
  echo "Task IDs: ${TASK_ID_ARRAY[*]}"
fi
echo "Sink:     $SINK_SIZE   Window (fixed): $WINDOW_SIZE"
echo "Agent max_tokens: $AGENT_MAX_TOKENS"
echo "Evict period:  $EVICT_PERIOD"
echo "Phase1 GPUs: A=$GPU_A B=$GPU_B C=$GPU_C"
echo "Phase2 GPUs: D=$GPU_D E=$GPU_E F=$GPU_F"
echo "Phase:    ${PHASE} (0=all)"
echo "=============================================================="

FAILED=0

case "$PHASE" in
  0)
    echo "[info] PHASE=0: run phase1 and phase2 in parallel"
    run_phase1 &
    local_phase1_pid=$!
    run_phase2 &
    local_phase2_pid=$!
    wait_group "$local_phase1_pid" "$local_phase2_pid" || FAILED=1
    ;;
  1) run_phase1 || FAILED=1 ;;
  2) run_phase2 || FAILED=1 ;;
  *)
    echo "[error] PHASE must be 0, 1, or 2"
    exit 1
    ;;
esac

echo ""
echo "========================================"
if [[ "$FAILED" -ne 0 ]]; then
  echo "Stress test finished with failures. Check ./outputs/stress_*_server.log and ./outputs/stress_*_tau2.log"
  exit 1
fi

echo "Stress test completed successfully."
echo "Results in: ./tau2-bench/data/simulations/"
echo "Logs in:    ./outputs/stress_*"