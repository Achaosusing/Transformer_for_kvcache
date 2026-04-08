#!/usr/bin/env bash
set -euo pipefail

# Launch one stateless baseline plus three session-decay alpha lanes in nohup.
# Default layout:
#   GPU 1 -> stateless
#   GPU 2 -> alpha 0.25
#   GPU 3 -> alpha 0.5
#   GPU 4 -> alpha 0.75

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
VENV_TAU2="$ROOT_DIR/.venv/bin/tau2"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_CMD="$VENV_PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python)"
else
  echo "[error] python not found; set PYTHON_BIN or create $VENV_PYTHON"
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-./local_models/Qwen3.5-9B}"
HOST="${HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-4o}"
USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"

DOMAIN="${DOMAIN:-airline}"
TASK_SPLIT="${TASK_SPLIT:-base}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_TASKS="${NUM_TASKS:-30}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-800}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-256}"

SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
HEAVY_HITTER_SIZE="${HEAVY_HITTER_SIZE:-64}"
EVICT_PERIOD="${EVICT_PERIOD:-16}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-800}"
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-10}"
KEEP_SERVERS_ON_EXIT="${KEEP_SERVERS_ON_EXIT:-0}"

ALPHAS="${ALPHAS:-0.25,0.5,0.75}"

GPU_STATELESS="${GPU_STATELESS:-2}"
GPU_ALPHA1="${GPU_ALPHA1:-3}"
GPU_ALPHA2="${GPU_ALPHA2:-4}"
GPU_ALPHA3="${GPU_ALPHA3:-5}"

PORT_STATELESS="${PORT_STATELESS:-8115}"
PORT_ALPHA1="${PORT_ALPHA1:-8116}"
PORT_ALPHA2="${PORT_ALPHA2:-8117}"
PORT_ALPHA3="${PORT_ALPHA3:-8118}"

RUN_TAG="${RUN_TAG:-airline_task30_alpha_grid_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/$RUN_TAG}"

if [[ -f "$ROOT_DIR/tau2-bench/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/tau2-bench/.env"
  set +a
fi

TAU2_SRC_DIR="$ROOT_DIR/tau2-bench/src"
if [[ -x "$VENV_TAU2" ]]; then
  TAU2_CMD=("$VENV_TAU2")
elif command -v tau2 >/dev/null 2>&1; then
  TAU2_CMD=(tau2)
elif [[ -d "$TAU2_SRC_DIR/tau2" ]]; then
  export PYTHONPATH="$TAU2_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  TAU2_CMD=("$PYTHON_CMD" -m tau2.cli)
else
  echo "[error] tau2 not found in PATH and local source missing at $TAU2_SRC_DIR"
  exit 1
fi

mkdir -p "$LOG_DIR"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

alpha_to_token() {
  local value
  value="$(trim "$1")"
  value="${value//./p}"
  printf '%s' "$value"
}

wait_health() {
  local port="$1"
  local pid="${2:-}"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

is_truthy() {
  local value="${1:-0}"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|on)
      return 0
      ;;
  esac
  return 1
}

LAST_STARTED_PID=""
declare -a SERVER_LABELS=()
declare -a SERVER_PIDS=()
declare -a TAU2_LABELS=()
declare -a TAU2_PIDS=()

record_server_pid() {
  local label="$1"
  local pid="$2"
  SERVER_LABELS+=("$label")
  SERVER_PIDS+=("$pid")
  printf '%s\n' "$pid" > "$LOG_DIR/server_${label}.pid"
}

record_tau2_pid() {
  local label="$1"
  local pid="$2"
  TAU2_LABELS+=("$label")
  TAU2_PIDS+=("$pid")
  printf '%s\n' "$pid" > "$LOG_DIR/tau2_${label}.pid"
}

wait_for_process_exit() {
  local pid="$1"
  local deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))

  while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 1
  done
  return 0
}

stop_tau2_processes() {
  local idx
  local label
  local pid

  for idx in "${!TAU2_PIDS[@]}"; do
    label="${TAU2_LABELS[$idx]}"
    pid="${TAU2_PIDS[$idx]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[cleanup] stop tau2 ${label} pid=${pid}"
      kill "$pid" 2>/dev/null || true
    fi
  done

  for idx in "${!TAU2_PIDS[@]}"; do
    label="${TAU2_LABELS[$idx]}"
    pid="${TAU2_PIDS[$idx]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null && ! wait_for_process_exit "$pid"; then
      echo "[cleanup] force kill tau2 ${label} pid=${pid}"
      kill -9 "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    rm -f "$LOG_DIR/tau2_${label}.pid"
  done

  TAU2_LABELS=()
  TAU2_PIDS=()
}

clear_tau2_tracking() {
  local idx
  local label

  for idx in "${!TAU2_LABELS[@]}"; do
    label="${TAU2_LABELS[$idx]}"
    rm -f "$LOG_DIR/tau2_${label}.pid"
  done

  TAU2_LABELS=()
  TAU2_PIDS=()
}

stop_server_processes() {
  local idx
  local label
  local pid

  if is_truthy "$KEEP_SERVERS_ON_EXIT"; then
    echo "[cleanup] KEEP_SERVERS_ON_EXIT=${KEEP_SERVERS_ON_EXIT}; leaving server processes running"
    SERVER_LABELS=()
    SERVER_PIDS=()
    return 0
  fi

  for idx in "${!SERVER_PIDS[@]}"; do
    label="${SERVER_LABELS[$idx]}"
    pid="${SERVER_PIDS[$idx]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[cleanup] stop server ${label} pid=${pid}"
      kill "$pid" 2>/dev/null || true
    fi
  done

  for idx in "${!SERVER_PIDS[@]}"; do
    label="${SERVER_LABELS[$idx]}"
    pid="${SERVER_PIDS[$idx]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null && ! wait_for_process_exit "$pid"; then
      echo "[cleanup] force kill server ${label} pid=${pid}"
      kill -9 "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    rm -f "$LOG_DIR/server_${label}.pid"
  done

  SERVER_LABELS=()
  SERVER_PIDS=()
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP

  if (( exit_code != 0 )); then
    stop_tau2_processes
  fi
  stop_server_processes
  exit "$exit_code"
}
trap cleanup EXIT INT TERM HUP

wait_for_server_health() {
  local label="$1"
  local port="$2"
  local pid="$3"

  if ! wait_health "$port" "$pid"; then
    echo "[error] server ${label} failed health check on ${HOST}:${port}; see $LOG_DIR/server_${label}.log"
    exit 1
  fi

  echo "[ready] server ${label} healthy on ${HOST}:${port}"
}

wait_for_tau2_jobs() {
  local idx
  local label
  local pid
  local wait_status
  local overall_status=0

  for idx in "${!TAU2_PIDS[@]}"; do
    label="${TAU2_LABELS[$idx]}"
    pid="${TAU2_PIDS[$idx]}"
    echo "[wait] tau2 ${label} pid=${pid}"
  done

  for idx in "${!TAU2_PIDS[@]}"; do
    label="${TAU2_LABELS[$idx]}"
    pid="${TAU2_PIDS[$idx]}"

    if wait "$pid"; then
      echo "[done] tau2 ${label} finished"
    else
      wait_status=$?
      echo "[error] tau2 ${label} exited with status ${wait_status}; see $LOG_DIR/tau2_${label}.log"
      overall_status=1
    fi

    rm -f "$LOG_DIR/tau2_${label}.pid"
  done

  TAU2_LABELS=()
  TAU2_PIDS=()
  return "$overall_status"
}

start_server() {
  local label="$1"
  local gpu="$2"
  local port="$3"

  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_CMD" api_server.py \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$port" \
    --method h2o \
    --h2o-sink-size "$SINK_SIZE" \
    --h2o-local-window-size "$WINDOW_SIZE" \
    --h2o-heavy-hitter-size "$HEAVY_HITTER_SIZE" \
    --evict-period "$EVICT_PERIOD" \
    > "$LOG_DIR/server_${label}.log" 2>&1 &

  LAST_STARTED_PID="$!"
  record_server_pid "$label" "$LAST_STARTED_PID"
  echo "[start] server ${label} pid=${LAST_STARTED_PID} gpu=${gpu} port=${port}"
}

start_tau2_stateless() {
  local port="$1"
  local save_to="h2o_session_ablation_stateless_${NUM_TRIALS}x${NUM_TASKS}_s${SINK_SIZE}_w${WINDOW_SIZE}_h${HEAVY_HITTER_SIZE}_${RUN_TAG}"
  local agent_llm_args

  agent_llm_args=$(printf '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s,"tau2_method_configs":{"h2o":{"sink_size":%s,"local_window_size":%s,"heavy_hitter_size":%s}}}' \
    "$HOST" "$port" "$AGENT_MAX_TOKENS" "$SINK_SIZE" "$WINDOW_SIZE" "$HEAVY_HITTER_SIZE")

  nohup "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --agent-llm "openai/$SERVED_MODEL_NAME" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_llm_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --num-tasks "$NUM_TASKS" \
    --save-to "$save_to" \
    > "$LOG_DIR/tau2_stateless.log" 2>&1 &

  LAST_STARTED_PID="$!"
  record_tau2_pid "stateless" "$LAST_STARTED_PID"
  echo "[start] tau2 stateless pid=${LAST_STARTED_PID} port=${port} save_to=${save_to}"
}

start_tau2_decay() {
  local label="$1"
  local port="$2"
  local alpha="$3"
  local alpha_token
  local save_to
  local agent_llm_args

  alpha_token="$(alpha_to_token "$alpha")"
  save_to="h2o_session_ablation_session_decay_${NUM_TRIALS}x${NUM_TASKS}_s${SINK_SIZE}_w${WINDOW_SIZE}_h${HEAVY_HITTER_SIZE}_a${alpha_token}_${RUN_TAG}"

  agent_llm_args=$(printf '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s,"tau2_enable_session_id":true,"tau2_session_namespace":"%s","tau2_session_score_alpha":%s,"tau2_method_configs":{"h2o":{"sink_size":%s,"local_window_size":%s,"heavy_hitter_size":%s}}}' \
    "$HOST" "$port" "$AGENT_MAX_TOKENS" "$save_to" "$alpha" "$SINK_SIZE" "$WINDOW_SIZE" "$HEAVY_HITTER_SIZE")

  nohup "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --agent-llm "openai/$SERVED_MODEL_NAME" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_llm_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --num-tasks "$NUM_TASKS" \
    --save-to "$save_to" \
    > "$LOG_DIR/tau2_${label}.log" 2>&1 &

  LAST_STARTED_PID="$!"
  record_tau2_pid "$label" "$LAST_STARTED_PID"
  echo "[start] tau2 ${label} pid=${LAST_STARTED_PID} port=${port} alpha=${alpha} save_to=${save_to}"
}

IFS=',' read -r -a alpha_values <<< "$ALPHAS"
if [[ ${#alpha_values[@]} -ne 3 ]]; then
  echo "[error] ALPHAS must contain exactly three comma-separated values, e.g. 0.25,0.5,0.75"
  exit 1
fi

alpha1="$(trim "${alpha_values[0]}")"
alpha2="$(trim "${alpha_values[1]}")"
alpha3="$(trim "${alpha_values[2]}")"

echo "[run_tag] $RUN_TAG"
echo "[log_dir] $LOG_DIR"
echo "[python] $PYTHON_CMD"
echo "[launcher_pid] $$"
echo "[layout] stateless@GPU${GPU_STATELESS} alpha=${alpha1}@GPU${GPU_ALPHA1} alpha=${alpha2}@GPU${GPU_ALPHA2} alpha=${alpha3}@GPU${GPU_ALPHA3}"
echo "[cleanup] KEEP_SERVERS_ON_EXIT=${KEEP_SERVERS_ON_EXIT} SHUTDOWN_GRACE_SECONDS=${SHUTDOWN_GRACE_SECONDS}"

start_server "stateless" "$GPU_STATELESS" "$PORT_STATELESS"
pid_stateless="$LAST_STARTED_PID"
start_server "alpha1" "$GPU_ALPHA1" "$PORT_ALPHA1"
pid_alpha1="$LAST_STARTED_PID"
start_server "alpha2" "$GPU_ALPHA2" "$PORT_ALPHA2"
pid_alpha2="$LAST_STARTED_PID"
start_server "alpha3" "$GPU_ALPHA3" "$PORT_ALPHA3"
pid_alpha3="$LAST_STARTED_PID"

wait_for_server_health "stateless" "$PORT_STATELESS" "$pid_stateless"
wait_for_server_health "alpha1" "$PORT_ALPHA1" "$pid_alpha1"
wait_for_server_health "alpha2" "$PORT_ALPHA2" "$pid_alpha2"
wait_for_server_health "alpha3" "$PORT_ALPHA3" "$pid_alpha3"

start_tau2_stateless "$PORT_STATELESS"
start_tau2_decay "alpha1" "$PORT_ALPHA1" "$alpha1"
start_tau2_decay "alpha2" "$PORT_ALPHA2" "$alpha2"
start_tau2_decay "alpha3" "$PORT_ALPHA3" "$alpha3"

echo "[wait] launcher will stay alive until all tau2 jobs finish"
echo "[watch] tail -f $LOG_DIR/tau2_*.log"
echo "[stop] kill $$"

if ! wait_for_tau2_jobs; then
  stop_server_processes
  exit 1
fi

stop_server_processes
if is_truthy "$KEEP_SERVERS_ON_EXIT"; then
  echo "[done] all tau2 jobs completed; server processes are still running"
  echo "[stop_servers] kill \$(cat $LOG_DIR/server_*.pid)"
else
  echo "[done] all tau2 jobs completed; server processes stopped"
fi