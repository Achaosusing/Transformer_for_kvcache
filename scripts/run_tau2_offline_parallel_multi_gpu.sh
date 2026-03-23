#!/usr/bin/env bash
set -euo pipefail

# Parallel version:
# - baseline / streamingllm / h2o run on different GPUs at the same time
# - each method starts its own api_server and tau2 evaluation concurrently

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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
NUM_TASKS="${NUM_TASKS:-50}"
STREAMING_SINK_SIZE="${STREAMING_SINK_SIZE:-4}"
STREAMING_LOCAL_WINDOW_SIZE="${STREAMING_LOCAL_WINDOW_SIZE:-256}"
H2O_SINK_SIZE="${H2O_SINK_SIZE:-4}"
H2O_LOCAL_WINDOW_SIZE="${H2O_LOCAL_WINDOW_SIZE:-256}"
H2O_HEAVY_HITTER_SIZE="${H2O_HEAVY_HITTER_SIZE:-128}"
PORT_BASELINE="${PORT_BASELINE:-8000}"
PORT_STREAMINGLLM="${PORT_STREAMINGLLM:-8001}"
PORT_H2O="${PORT_H2O:-8002}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"

SAVE_TO_BASELINE="baseline_${NUM_TRIALS}x${NUM_TASKS}"
SAVE_TO_STREAMINGLLM="streamingllm_${NUM_TRIALS}x${NUM_TASKS}_${STREAMING_SINK_SIZE}_${STREAMING_LOCAL_WINDOW_SIZE}"
SAVE_TO_H2O="h2o_${NUM_TRIALS}x${NUM_TASKS}_${H2O_SINK_SIZE}_${H2O_LOCAL_WINDOW_SIZE}_${H2O_HEAVY_HITTER_SIZE}"

# Frequently edited GPU bindings. Override these before running, e.g.:
# GPU_BASELINE=2 GPU_STREAMINGLLM=4 GPU_H2O=7 bash scripts/run_tau2_offline_parallel_multi_gpu.sh
GPU_BASELINE="${GPU_BASELINE:-2}"
GPU_STREAMINGLLM="${GPU_STREAMINGLLM:-3}"
GPU_H2O="${GPU_H2O:-4}"

# Optional: source tau2 env variables if file exists (for DEEPSEEK_API_KEY etc.)
if [[ -f "$ROOT_DIR/tau2-bench/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
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

run_one_method() {
  local method="$1"
  local port="$2"
  local save_to="$3"
  local gpu_id="$4"
  local -a method_extra_args=()

  case "$method" in
    streamingllm|streaming_llm)
      method_extra_args+=(
        --streaming-sink-size "$STREAMING_SINK_SIZE"
        --streaming-local-window-size "$STREAMING_LOCAL_WINDOW_SIZE"
      )
      ;;
    h2o)
      method_extra_args+=(
        --h2o-sink-size "$H2O_SINK_SIZE"
        --h2o-local-window-size "$H2O_LOCAL_WINDOW_SIZE"
        --h2o-heavy-hitter-size "$H2O_HEAVY_HITTER_SIZE"
      )
      ;;
  esac

  local server_pid=""

  cleanup_local() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      echo "[cleanup] method=$method stop server pid=$server_pid"
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_local EXIT INT TERM

  echo "========================================"
  echo "[start] method=$method gpu=$gpu_id port=$port"

  CUDA_VISIBLE_DEVICES="$gpu_id" python api_server.py \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$port" \
    --method "$method" \
    "${method_extra_args[@]}" \
    >"./outputs/${save_to}_server.log" 2>&1 &

  server_pid=$!
  echo "[info] method=$method server pid=$server_pid"

  if ! wait_for_health "$port"; then
    echo "[error] method=$method health check failed: http://${HOST}:${port}/health"
    return 1
  fi

  local -a tau2_extra_args=()
  if [[ -n "${NUM_TRIALS:-}" ]]; then
    tau2_extra_args+=(--num-trials "$NUM_TRIALS")
  fi
  if [[ -n "${NUM_TASKS:-}" ]]; then
    tau2_extra_args+=(--num-tasks "$NUM_TASKS")
  fi

  echo "[run] tau2 eval for method=$method"
  "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "{\"api_base\":\"http://${HOST}:${port}/v1\",\"api_key\":\"EMPTY\",\"temperature\":0.0}" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    "${tau2_extra_args[@]}" \
    --save-to "$save_to" \
    >"./outputs/${save_to}_tau2.log" 2>&1

  echo "[done] method=$method save_to=$save_to"
}

echo "[config] baseline: gpu=$GPU_BASELINE port=$PORT_BASELINE"
echo "[config] streamingllm: gpu=$GPU_STREAMINGLLM port=$PORT_STREAMINGLLM sink=$STREAMING_SINK_SIZE local=$STREAMING_LOCAL_WINDOW_SIZE"
echo "[config] h2o: gpu=$GPU_H2O port=$PORT_H2O sink=$H2O_SINK_SIZE local=$H2O_LOCAL_WINDOW_SIZE heavy=$H2O_HEAVY_HITTER_SIZE"
echo "[config] save baseline=$SAVE_TO_BASELINE"
echo "[config] save streamingllm=$SAVE_TO_STREAMINGLLM"
echo "[config] save h2o=$SAVE_TO_H2O"

run_one_method "baseline" "$PORT_BASELINE" "$SAVE_TO_BASELINE" "$GPU_BASELINE" &
PID_BASELINE=$!

run_one_method "streamingllm" "$PORT_STREAMINGLLM" "$SAVE_TO_STREAMINGLLM" "$GPU_STREAMINGLLM" &
PID_STREAMINGLLM=$!

run_one_method "h2o" "$PORT_H2O" "$SAVE_TO_H2O" "$GPU_H2O" &
PID_H2O=$!

PIDS=("$PID_BASELINE" "$PID_STREAMINGLLM" "$PID_H2O")
METHOD_NAMES=("baseline" "streamingllm" "h2o")

FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[error] ${METHOD_NAMES[$i]} job failed"
    FAILED=1
  else
    echo "[ok] ${METHOD_NAMES[$i]} job finished"
  fi
done

echo "========================================"
if [[ "$FAILED" -ne 0 ]]; then
  echo "Parallel run finished with failures. Check ./outputs/*_server.log and ./outputs/*_tau2.log"
  exit 1
fi

echo "All methods finished in parallel."
echo "Results in: ./tau2-bench/data/simulations/"