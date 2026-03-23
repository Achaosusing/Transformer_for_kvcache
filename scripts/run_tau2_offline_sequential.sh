#!/usr/bin/env bash
set -euo pipefail

# Sequentially evaluate baseline -> streamingllm -> h2o with tau2 official pipeline.
# For each method:
# 1) start api_server.py with fixed method
# 2) wait for /health
# 3) run `tau2 run`
# 4) stop server process and wait for full exit

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-./local_models/Qwen3.5-0.8B}"
DOMAIN="${DOMAIN:-airline}"
TASK_SPLIT="${TASK_SPLIT:-base}"
USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-4o}"
AGENT_LLM="openai/${SERVED_MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
TIMEOUT_SECONDS=800
TASK_TIMEOUT_SECONDS=800
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_TASKS="${NUM_TASKS:-10}"
PORT_BASELINE="${PORT_BASELINE:-8000}"
PORT_STREAMINGLLM="${PORT_STREAMINGLLM:-8001}"
PORT_H2O="${PORT_H2O:-8002}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"

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

cleanup_pid=""
cleanup() {
  if [[ -n "$cleanup_pid" ]] && kill -0 "$cleanup_pid" 2>/dev/null; then
    echo "[cleanup] stopping server pid=$cleanup_pid"
    kill "$cleanup_pid" 2>/dev/null || true
    wait "$cleanup_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

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

  echo "========================================"
  echo "[start] method=$method port=$port"

  python api_server.py \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$port" \
    --method "$method" \
    >"./outputs/${save_to}_server.log" 2>&1 &

  cleanup_pid=$!
  echo "[info] server pid=$cleanup_pid"

  if ! wait_for_health "$port"; then
    echo "[error] server health check failed: http://${HOST}:${port}/health"
    return 1
  fi

  echo "[run] tau2 eval for method=$method"
  tau2 run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "{\"api_base\":\"http://${HOST}:${port}/v1\",\"api_key\":\"EMPTY\",\"temperature\":0.0}" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    # --num-trials "$NUM_TRIALS" \
    # --num-tasks "$NUM_TASKS" \
    --save-to "$save_to"

  echo "[stop] method=$method pid=$cleanup_pid"
  kill "$cleanup_pid" 2>/dev/null || true
  wait "$cleanup_pid" 2>/dev/null || true
  cleanup_pid=""

  echo "[done] method=$method save_to=$save_to"
}

mkdir -p ./outputs

run_one_method "baseline" "$PORT_BASELINE" "transformer_baseline_50"
run_one_method "streamingllm" "$PORT_STREAMINGLLM" "transformer_streamingllm_50"
run_one_method "h2o" "$PORT_H2O" "transformer_h2o_50"

echo "========================================"
echo "All methods finished sequentially."
echo "Results in: ./tau2-bench/data/simulations/"
