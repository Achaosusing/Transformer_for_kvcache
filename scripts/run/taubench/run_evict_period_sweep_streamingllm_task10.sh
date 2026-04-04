#!/usr/bin/env bash
set -euo pipefail

# Sweep EVICT_PERIOD for StreamingLLM on task10 subset, single GPU.
#
# Fixed config: method=streamingllm, sink=4, window=128
# Sweep: evict_period in {1, 2, 4, 8, 12, 16}
# Runs sequentially on one GPU to avoid VRAM conflicts.
#
# Usage:
#   bash scripts/run/taubench/run_evict_period_sweep_streamingllm_task10.sh
#
# Key overrides:
#   GPU=3 bash scripts/run/taubench/run_evict_period_sweep_streamingllm_task10.sh
#   EVICT_PERIODS="1 4 16" bash scripts/run/taubench/run_evict_period_sweep_streamingllm_task10.sh
#   WINDOW_SIZE=64 bash scripts/run/taubench/run_evict_period_sweep_streamingllm_task10.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
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
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-256}"

GPU="${GPU:-2}"
PORT="${PORT:-8021}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"

# StreamingLLM fixed params — use window=128 (best single-run result so far)
SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-128}"

# task10 subset
TASK_IDS="${TASK_IDS:-0 1 3 4 5 6 10 13 26 28}"
TASK_TAG="${TASK_TAG:-task10}"
NUM_TASKS=10

# evict_period values to sweep (space-separated)
EVICT_PERIODS="${EVICT_PERIODS:-1 2 4 8 12 16}"

# ── Environment ───────────────────────────────────────────────────────────────
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

# Parse task ids into array
read -r -a TASK_ID_ARRAY <<< "${TASK_IDS//,/ }"

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# run_one_evict_period  evict_period
run_one_evict_period() {
  local evict_period="$1"
  local budget=$(( SINK_SIZE + WINDOW_SIZE ))
  local save_to="stress_streamingllm_1x${NUM_TASKS}_s${SINK_SIZE}_w${WINDOW_SIZE}_ep${evict_period}_${TASK_TAG}"

  local server_pid=""
  cleanup_run() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      echo "[cleanup] evict_period=$evict_period stop server pid=$server_pid"
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_run EXIT INT TERM

  echo ""
  echo "========================================"
  echo "[start] evict_period=$evict_period  gpu=$GPU  port=$PORT"
  echo "        sink=$SINK_SIZE  window=$WINDOW_SIZE  budget=$budget"
  echo "        save_to=$save_to"
  echo "========================================"

  CUDA_VISIBLE_DEVICES="$GPU" python api_server.py \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$PORT" \
    --method streamingllm \
    --evict-period "$evict_period" \
    --streaming-sink-size "$SINK_SIZE" \
    --streaming-local-window-size "$WINDOW_SIZE" \
    >"./outputs/${save_to}_server.log" 2>&1 &

  server_pid=$!
  echo "[info] evict_period=$evict_period server pid=$server_pid"

  if ! wait_for_health "$PORT"; then
    echo "[error] evict_period=$evict_period health check failed: http://${HOST}:${PORT}/health"
    return 1
  fi
  echo "[info] evict_period=$evict_period server ready"

  local agent_llm_args
  agent_llm_args=$(printf \
    '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s}' \
    "$HOST" "$PORT" "$AGENT_MAX_TOKENS")

  echo "[run] tau2 eval  evict_period=$evict_period"
  "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --task-ids "${TASK_ID_ARRAY[@]}" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_llm_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --save-to "$save_to" \
    >"./outputs/${save_to}_tau2.log" 2>&1

  echo "[done] evict_period=$evict_period  save_to=$save_to"

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
  trap - EXIT INT TERM
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=============================================================="
echo "StreamingLLM evict_period sweep on task10"
echo "GPU=$GPU  sink=$SINK_SIZE  window=$WINDOW_SIZE"
echo "budget=$(( SINK_SIZE + WINDOW_SIZE ))"
echo "periods: $EVICT_PERIODS"
echo "tasks: ${TASK_IDS}"
echo "=============================================================="

read -r -a PERIOD_ARRAY <<< "$EVICT_PERIODS"
for ep in "${PERIOD_ARRAY[@]}"; do
  run_one_evict_period "$ep"
done

echo ""
echo "=============================================================="
echo "All evict_period configs done."
echo "Results in: ./outputs/stress_streamingllm_*_ep*_task10*"
echo "Then analyze with:"
echo "  python scripts/analyze/taubench/analyze_evict_period_sweep.py"
echo "=============================================================="
