#!/usr/bin/env bash
set -euo pipefail

# H2O session ablation on tau2-bench.
# Compares two groups on the same fixed H2O server config:
#   1) stateless      : no session_id   (equivalent to old stateless H2O)
#   2) session_decay  : session_id + alpha=0.5
#
# Batch sweep usage:
#   DOMAINS="airline,retail" \
#   NUM_TASKS_LIST="30,100" \
#   MAX_CONCURRENCY_LIST="1,3" \
#   SESSION_ALPHAS="0.3,0.5" \
#   bash scripts/run/taubench/run_h2o_session_ablation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-./local_models/Qwen3.5-9B}"
DOMAIN="${DOMAIN:-airline}"
TASK_SPLIT="${TASK_SPLIT:-base}"
USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-4o}"
AGENT_LLM="openai/${SERVED_MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
PORT_STATELESS="${PORT_STATELESS:-8022}"
PORT_SESSION_DECAY="${PORT_SESSION_DECAY:-8023}"
GPU_STATELESS="${GPU_STATELESS:-6}"
GPU_SESSION_DECAY="${GPU_SESSION_DECAY:-7}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-800}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-800}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_TASKS="${NUM_TASKS:-50}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-256}"
EVICT_PERIOD="${EVICT_PERIOD:-16}"
SINK_SIZE="${SINK_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
HEAVY_HITTER_SIZE="${HEAVY_HITTER_SIZE:-64}"
SESSION_ALPHA="${SESSION_ALPHA:-0.5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SAVE_TAG="${SAVE_TAG:-}"

DOMAINS="${DOMAINS:-$DOMAIN}"
TASK_SPLITS="${TASK_SPLITS:-$TASK_SPLIT}"
NUM_TRIALS_LIST="${NUM_TRIALS_LIST:-$NUM_TRIALS}"
NUM_TASKS_LIST="${NUM_TASKS_LIST:-$NUM_TASKS}"
MAX_CONCURRENCY_LIST="${MAX_CONCURRENCY_LIST:-$MAX_CONCURRENCY}"
SINK_SIZES="${SINK_SIZES:-$SINK_SIZE}"
WINDOW_SIZES="${WINDOW_SIZES:-$WINDOW_SIZE}"
HEAVY_HITTER_SIZES="${HEAVY_HITTER_SIZES:-$HEAVY_HITTER_SIZE}"
SESSION_ALPHAS="${SESSION_ALPHAS:-$SESSION_ALPHA}"
EVICT_PERIODS="${EVICT_PERIODS:-$EVICT_PERIOD}"

if [[ -f "$ROOT_DIR/tau2-bench/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/tau2-bench/.env"
  set +a
fi

TAU2_SRC_DIR="$ROOT_DIR/tau2-bench/src"
if command -v tau2 >/dev/null 2>&1; then
  TAU2_CMD=(tau2)
elif [[ -d "$TAU2_SRC_DIR/tau2" ]]; then
  export PYTHONPATH="$TAU2_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  TAU2_CMD=(python -m tau2.cli)
else
  echo "[error] tau2 not found in PATH and local source missing at $TAU2_SRC_DIR"
  exit 1
fi

mkdir -p ./outputs

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_csv_into() {
  local -n out_ref="$1"
  local raw="$2"
  local item

  IFS=',' read -r -a out_ref <<< "$raw"
  if [[ ${#out_ref[@]} -eq 0 ]]; then
    out_ref=("$raw")
  fi
  for item in "${!out_ref[@]}"; do
    out_ref[$item]="$(trim "${out_ref[$item]}")"
  done
}

sanitize_token() {
  printf '%s' "$1" | tr ' /:' '___' | tr -cd '[:alnum:]_.-='
}

format_alpha_token() {
  local value
  value="$(sanitize_token "$1")"
  value="${value//./p}"
  printf '%s' "$value"
}

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

server_pid_stateless=""
server_pid_session_decay=""
CURRENT_SAVE_TAG=""
CURRENT_SAVE_SUFFIX=""
CURRENT_CONFIG_INDEX=0
TOTAL_CONFIGS=1

stop_servers() {
  local pid
  for pid in "$server_pid_stateless" "$server_pid_session_decay"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  server_pid_stateless=""
  server_pid_session_decay=""
}
cleanup() {
  stop_servers
}
trap cleanup EXIT INT TERM

start_server() {
  local label="$1"
  local gpu_id="$2"
  local port="$3"

  CUDA_VISIBLE_DEVICES="$gpu_id" python api_server.py \
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
    >"./outputs/h2o_session_ablation_${label}_server${CURRENT_SAVE_SUFFIX}.log" 2>&1 &

  local server_pid=$!
  case "$label" in
    stateless) server_pid_stateless="$server_pid" ;;
    session_decay) server_pid_session_decay="$server_pid" ;;
  esac

  if ! wait_for_health "$port"; then
    echo "[error] $label server health check failed on port $port"
    exit 1
  fi
}

ensure_resume_safe() {
  local sim_file="$1"
  local expected="$2"
  if [[ ! -f "$sim_file" ]]; then
    return 0
  fi

  local n_sims
  n_sims=$(python3 -c "import json; print(len(json.load(open('$sim_file')).get('simulations', [])))" 2>/dev/null || echo 0)
  if [[ "$n_sims" -lt "$expected" ]]; then
    echo "[warn] removing partial simulation file: $sim_file ($n_sims/$expected)"
    rm -f "$sim_file"
  fi
}

run_variant() {
  local label="$1"
  local enable_session="$2"
  local alpha="$3"
  local port="$4"

  local save_to="h2o_session_ablation_${label}_${NUM_TRIALS}x${NUM_TASKS}_s${SINK_SIZE}_w${WINDOW_SIZE}_h${HEAVY_HITTER_SIZE}${CURRENT_SAVE_SUFFIX}"
  local sim_file="$ROOT_DIR/tau2-bench/data/simulations/${save_to}.json"
  local expected_runs=$((NUM_TRIALS * NUM_TASKS))
  ensure_resume_safe "$sim_file" "$expected_runs"

  local agent_llm_args
  if [[ "$enable_session" == "1" ]]; then
    agent_llm_args=$(printf '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s,"tau2_enable_session_id":true,"tau2_session_namespace":"%s","tau2_session_score_alpha":%s,"tau2_method_configs":{"h2o":{"sink_size":%s,"local_window_size":%s,"heavy_hitter_size":%s}}}' \
      "$HOST" "$port" "$AGENT_MAX_TOKENS" "$save_to" "$alpha" "$SINK_SIZE" "$WINDOW_SIZE" "$HEAVY_HITTER_SIZE")
  else
    agent_llm_args=$(printf '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s,"tau2_method_configs":{"h2o":{"sink_size":%s,"local_window_size":%s,"heavy_hitter_size":%s}}}' \
      "$HOST" "$port" "$AGENT_MAX_TOKENS" "$SINK_SIZE" "$WINDOW_SIZE" "$HEAVY_HITTER_SIZE")
  fi

  echo "[run] $label -> $save_to"
  "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_llm_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --num-tasks "$NUM_TASKS" \
    --save-to "$save_to" \
    >"./outputs/${save_to}_tau2.log" 2>&1
}

build_config_tag() {
  local domain="$1"
  local task_split="$2"
  local num_trials="$3"
  local num_tasks="$4"
  local max_concurrency="$5"
  local sink_size="$6"
  local window_size="$7"
  local heavy_hitter_size="$8"
  local alpha="$9"
  local evict_period="${10}"

  local tag
  tag=$(printf 'dom%s_split%s_tr%s_nt%s_c%s_s%s_w%s_h%s_a%s_ep%s' \
    "$(sanitize_token "$domain")" \
    "$(sanitize_token "$task_split")" \
    "$(sanitize_token "$num_trials")" \
    "$(sanitize_token "$num_tasks")" \
    "$(sanitize_token "$max_concurrency")" \
    "$(sanitize_token "$sink_size")" \
    "$(sanitize_token "$window_size")" \
    "$(sanitize_token "$heavy_hitter_size")" \
    "$(format_alpha_token "$alpha")" \
    "$(sanitize_token "$evict_period")")

  if [[ -n "$SAVE_TAG" ]]; then
    tag+="_$(sanitize_token "$SAVE_TAG")"
  fi
  printf '%s' "$tag"
}

set_current_save_suffix() {
  if (( TOTAL_CONFIGS > 1 )); then
    CURRENT_SAVE_TAG="$(build_config_tag "$DOMAIN" "$TASK_SPLIT" "$NUM_TRIALS" "$NUM_TASKS" "$MAX_CONCURRENCY" "$SINK_SIZE" "$WINDOW_SIZE" "$HEAVY_HITTER_SIZE" "$SESSION_ALPHA" "$EVICT_PERIOD")"
  else
    CURRENT_SAVE_TAG="$(sanitize_token "$SAVE_TAG")"
  fi

  if [[ -n "$CURRENT_SAVE_TAG" ]]; then
    CURRENT_SAVE_SUFFIX="_${CURRENT_SAVE_TAG}"
  else
    CURRENT_SAVE_SUFFIX=""
  fi
}

print_config_header() {
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  H2O Session Ablation ${CURRENT_CONFIG_INDEX}/${TOTAL_CONFIGS}                         ║"
  echo "║  domain=$DOMAIN split=$TASK_SPLIT                          ║"
  echo "║  sink=$SINK_SIZE window=$WINDOW_SIZE heavy=$HEAVY_HITTER_SIZE         ║"
  echo "║  trials=$NUM_TRIALS tasks=$NUM_TASKS alpha=$SESSION_ALPHA concurrency=$MAX_CONCURRENCY   ║"
  echo "║  evict=$EVICT_PERIOD gpus: stateless=$GPU_STATELESS decay=$GPU_SESSION_DECAY        ║"
  if [[ -n "$CURRENT_SAVE_TAG" ]]; then
    echo "║  tag=$CURRENT_SAVE_TAG ║"
  fi
  echo "╚══════════════════════════════════════════════════════════════╝"
}

run_current_config() {
  print_config_header

  start_server "stateless" "$GPU_STATELESS" "$PORT_STATELESS"
  start_server "session_decay" "$GPU_SESSION_DECAY" "$PORT_SESSION_DECAY"

  run_variant "stateless" 0 0.0 "$PORT_STATELESS" &
  pid_stateless=$!
  run_variant "session_decay" 1 "$SESSION_ALPHA" "$PORT_SESSION_DECAY" &
  pid_session_decay=$!

  local failed=0
  local pid
  for pid in "$pid_stateless" "$pid_session_decay"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "[error] One or more ablation runs failed. Check ./outputs/h2o_session_ablation_*_server${CURRENT_SAVE_SUFFIX}.log and ./outputs/h2o_session_ablation_*_tau2.log"
    exit 1
  fi

  echo "[done] Simulation JSON saved under ./tau2-bench/data/simulations/"
  stop_servers
}

declare -a DOMAIN_VALUES
declare -a TASK_SPLIT_VALUES
declare -a NUM_TRIAL_VALUES
declare -a NUM_TASK_VALUES
declare -a MAX_CONCURRENCY_VALUES
declare -a SINK_SIZE_VALUES
declare -a WINDOW_SIZE_VALUES
declare -a HEAVY_HITTER_VALUES
declare -a SESSION_ALPHA_VALUES
declare -a EVICT_PERIOD_VALUES

parse_csv_into DOMAIN_VALUES "$DOMAINS"
parse_csv_into TASK_SPLIT_VALUES "$TASK_SPLITS"
parse_csv_into NUM_TRIAL_VALUES "$NUM_TRIALS_LIST"
parse_csv_into NUM_TASK_VALUES "$NUM_TASKS_LIST"
parse_csv_into MAX_CONCURRENCY_VALUES "$MAX_CONCURRENCY_LIST"
parse_csv_into SINK_SIZE_VALUES "$SINK_SIZES"
parse_csv_into WINDOW_SIZE_VALUES "$WINDOW_SIZES"
parse_csv_into HEAVY_HITTER_VALUES "$HEAVY_HITTER_SIZES"
parse_csv_into SESSION_ALPHA_VALUES "$SESSION_ALPHAS"
parse_csv_into EVICT_PERIOD_VALUES "$EVICT_PERIODS"

TOTAL_CONFIGS=$(( \
  ${#DOMAIN_VALUES[@]} * \
  ${#TASK_SPLIT_VALUES[@]} * \
  ${#NUM_TRIAL_VALUES[@]} * \
  ${#NUM_TASK_VALUES[@]} * \
  ${#MAX_CONCURRENCY_VALUES[@]} * \
  ${#SINK_SIZE_VALUES[@]} * \
  ${#WINDOW_SIZE_VALUES[@]} * \
  ${#HEAVY_HITTER_VALUES[@]} * \
  ${#SESSION_ALPHA_VALUES[@]} * \
  ${#EVICT_PERIOD_VALUES[@]} ))

config_index=0
for domain in "${DOMAIN_VALUES[@]}"; do
  for task_split in "${TASK_SPLIT_VALUES[@]}"; do
    for num_trials in "${NUM_TRIAL_VALUES[@]}"; do
      for num_tasks in "${NUM_TASK_VALUES[@]}"; do
        for max_concurrency in "${MAX_CONCURRENCY_VALUES[@]}"; do
          for sink_size in "${SINK_SIZE_VALUES[@]}"; do
            for window_size in "${WINDOW_SIZE_VALUES[@]}"; do
              for heavy_hitter_size in "${HEAVY_HITTER_VALUES[@]}"; do
                for session_alpha in "${SESSION_ALPHA_VALUES[@]}"; do
                  for evict_period in "${EVICT_PERIOD_VALUES[@]}"; do
                    ((config_index+=1))

                    DOMAIN="$domain"
                    TASK_SPLIT="$task_split"
                    NUM_TRIALS="$num_trials"
                    NUM_TASKS="$num_tasks"
                    MAX_CONCURRENCY="$max_concurrency"
                    SINK_SIZE="$sink_size"
                    WINDOW_SIZE="$window_size"
                    HEAVY_HITTER_SIZE="$heavy_hitter_size"
                    SESSION_ALPHA="$session_alpha"
                    EVICT_PERIOD="$evict_period"
                    CURRENT_CONFIG_INDEX="$config_index"

                    set_current_save_suffix
                    run_current_config
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

if (( TOTAL_CONFIGS > 1 )); then
  if [[ -n "$SAVE_TAG" ]]; then
    echo "[next] Analyze this sweep with: python scripts/analyze/taubench/analyze_h2o_session_ablation.py --tag-contains $(sanitize_token "$SAVE_TAG")"
  else
    echo "[next] Analyze all matching outputs with: python scripts/analyze/taubench/analyze_h2o_session_ablation.py"
  fi
else
  echo "[next] Analyze with: python scripts/analyze/taubench/analyze_h2o_session_ablation.py${CURRENT_SAVE_TAG:+ --tag ${CURRENT_SAVE_TAG}}"
fi