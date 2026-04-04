#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# 3D Sweep: evict_period × cache_budget × method   (task10, GPU 2+3)
# ══════════════════════════════════════════════════════════════════════════════
#
# Dimension 1 — evict_period:  {1, 4, 16}
# Dimension 2 — cache budget:  {132, 260, 516}
#   Budget = sink + window (StreamingLLM) = sink + window + heavy (H2O)
#   sink 固定 = 4；H2O 令 heavy = window 以保持与 StreamingLLM 同等 budget
#     budget=132 → SL(w=128)       H2O(w=64,  h=64)
#     budget=260 → SL(w=256)       H2O(w=128, h=128)
#     budget=516 → SL(w=512)       H2O(w=256, h=256)
# Dimension 3 — method:  baseline | streamingllm | h2o
#
# Grid: 1(baseline) + 9(SL) + 9(H2O) = 19 runs
# 两张 GPU 并行 → ~10 batches，预计 3-4 小时
#
# Usage:
#   bash scripts/run/taubench/run_3d_sweep_task10.sh
#
# Overrides:
#   GPU_A=2 GPU_B=3 bash scripts/run/taubench/run_3d_sweep_task10.sh
#   DRY_RUN=1 bash scripts/run/taubench/run_3d_sweep_task10.sh   # 只打印不执行
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"

# ── Shared Config ─────────────────────────────────────────────────────────────
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
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SINK_SIZE="${SINK_SIZE:-4}"

# GPU and port assignments (2 GPUs in parallel)
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
PORT_A="${PORT_A:-8020}"
PORT_B="${PORT_B:-8021}"

# task10 subset
TASK_IDS="${TASK_IDS:-0 1 3 4 5 6 10 13 26 28}"
TASK_TAG="task10"

DRY_RUN="${DRY_RUN:-0}"

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -f "$ROOT_DIR/tau2-bench/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/tau2-bench/.env"
  set +a
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[WARN] DEEPSEEK_API_KEY is empty. user-llm=$USER_LLM may fail."
fi

if command -v tau2 >/dev/null 2>&1; then
  TAU2_CMD=(tau2)
  echo "[info] using tau2 from PATH"
else
  TAU2_SRC_DIR="$ROOT_DIR/tau2-bench/src"
  if [[ -d "$TAU2_SRC_DIR/tau2" ]]; then
    export PYTHONPATH="$TAU2_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
    TAU2_CMD=(python -m tau2.cli)
    echo "[info] tau2 fallback to python -m tau2.cli"
  else
    echo "[error] tau2 not found"; exit 1
  fi
fi

mkdir -p ./outputs
read -r -a TASK_ID_ARRAY <<< "${TASK_IDS//,/ }"

# ══════════════════════════════════════════════════════════════════════════════
# Configuration grid
# Format: "method|window|heavy|evict_period"
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS=(
  # ── baseline (1 run, uses SDPA, no cache compression) ──
  "baseline|0|0|1"

  # ── streamingllm: 3 budgets × 3 evict_periods = 9 runs (SDPA) ──
  # budget=132 (w=128)
  "streamingllm|128|0|1"
  "streamingllm|128|0|4"
  "streamingllm|128|0|16"
  # budget=260 (w=256)
  "streamingllm|256|0|1"
  "streamingllm|256|0|4"
  "streamingllm|256|0|16"
  # budget=516 (w=512)
  "streamingllm|512|0|1"
  "streamingllm|512|0|4"
  "streamingllm|512|0|16"

  # ── h2o: 3 budgets × 3 evict_periods = 9 runs (eager) ──
  # budget=132 (w=64, h=64)
  "h2o|64|64|1"
  "h2o|64|64|4"
  "h2o|64|64|16"
  # budget=260 (w=128, h=128)
  "h2o|128|128|1"
  "h2o|128|128|4"
  "h2o|128|128|16"
  # budget=516 (w=256, h=256)
  "h2o|256|256|1"
  "h2o|256|256|4"
  "h2o|256|256|16"
)

TOTAL_CONFIGS="${#CONFIGS[@]}"

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
wait_for_health() {
  local port="$1"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -sSf "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# make_save_to method window heavy ep → save_to name
make_save_to() {
  local method="$1" window="$2" heavy="$3" ep="$4"
  case "$method" in
    baseline)
      echo "sweep3d_baseline_${TASK_TAG}"
      ;;
    streamingllm)
      echo "sweep3d_sl_s${SINK_SIZE}_w${window}_ep${ep}_${TASK_TAG}"
      ;;
    h2o)
      echo "sweep3d_h2o_s${SINK_SIZE}_w${window}_h${heavy}_ep${ep}_${TASK_TAG}"
      ;;
  esac
}

# run_one_config method window heavy ep gpu port
run_one_config() {
  local method="$1" window="$2" heavy="$3" ep="$4" gpu="$5" port="$6"
  local save_to
  save_to=$(make_save_to "$method" "$window" "$heavy" "$ep")

  # Compute budget for display
  local budget_desc
  case "$method" in
    baseline)     budget_desc="unbounded" ;;
    streamingllm) budget_desc="budget=$((SINK_SIZE + window))" ;;
    h2o)          budget_desc="budget=$((SINK_SIZE + window + heavy))" ;;
  esac

  echo ""
  echo "┌──────────────────────────────────────────────────────────┐"
  echo "│ [START] method=$method  ep=$ep  $budget_desc"
  echo "│         gpu=$gpu  port=$port  save_to=$save_to"
  echo "└──────────────────────────────────────────────────────────┘"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] skipping actual execution"
    return 0
  fi

  # Build server args
  local -a server_args=(
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --host "$HOST"
    --port "$port"
    --method "$method"
    --evict-period "$ep"
    --attn-implementation auto
  )

  case "$method" in
    streamingllm)
      server_args+=(
        --streaming-sink-size "$SINK_SIZE"
        --streaming-local-window-size "$window"
      )
      ;;
    h2o)
      server_args+=(
        --h2o-sink-size "$SINK_SIZE"
        --h2o-local-window-size "$window"
        --h2o-heavy-hitter-size "$heavy"
        --collect-period 0
      )
      ;;
  esac

  # Start server
  CUDA_VISIBLE_DEVICES="$gpu" python api_server.py "${server_args[@]}" \
    >"./outputs/${save_to}_server.log" 2>&1 &
  local server_pid=$!
  echo "[info] server pid=$server_pid (gpu=$gpu)"

  # Health check
  if ! wait_for_health "$port"; then
    echo "[ERROR] health check failed for $save_to (port=$port)"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi
  echo "[info] server ready ($save_to)"

  # Run tau2 evaluation
  local agent_llm_args
  agent_llm_args=$(printf \
    '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s}' \
    "$HOST" "$port" "$AGENT_MAX_TOKENS")

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

  local tau2_exit=$?

  # Stop server
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true

  if [[ $tau2_exit -eq 0 ]]; then
    echo "[DONE] ✓ $save_to"
  else
    echo "[FAIL] ✗ $save_to (tau2 exit=$tau2_exit)"
  fi
  return $tau2_exit
}

# ══════════════════════════════════════════════════════════════════════════════
# Main: process configs in pairs (2 GPUs)
# ══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  3D Sweep: evict_period × budget × method                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:  $MODEL_PATH"
echo "║  GPUs:   $GPU_A, $GPU_B    Ports: $PORT_A, $PORT_B"
echo "║  Sink:   $SINK_SIZE (fixed)"
echo "║  Tasks:  ${TASK_IDS}"
echo "║  Configs: $TOTAL_CONFIGS total"
echo "║"
echo "║  Dim1 evict_period: {1, 4, 16}"
echo "║  Dim2 budget:       {132, 260, 516}"
echo "║  Dim3 method:       {baseline, streamingllm, h2o}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

STARTED_AT=$SECONDS
COMPLETED=0
FAILED=0

for (( i=0; i < TOTAL_CONFIGS; i+=2 )); do
  batch_num=$(( i/2 + 1 ))
  batch_total=$(( (TOTAL_CONFIGS + 1) / 2 ))
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Batch $batch_num / $batch_total"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Parse config A
  IFS='|' read -r method_a window_a heavy_a ep_a <<< "${CONFIGS[$i]}"

  # Launch config A on GPU_A
  run_one_config "$method_a" "$window_a" "$heavy_a" "$ep_a" "$GPU_A" "$PORT_A" &
  pid_a=$!

  # Launch config B on GPU_B (if exists)
  pid_b=""
  if (( i+1 < TOTAL_CONFIGS )); then
    IFS='|' read -r method_b window_b heavy_b ep_b <<< "${CONFIGS[$((i+1))]}"
    run_one_config "$method_b" "$window_b" "$heavy_b" "$ep_b" "$GPU_B" "$PORT_B" &
    pid_b=$!
  fi

  # Wait for both
  if wait "$pid_a"; then
    COMPLETED=$((COMPLETED + 1))
  else
    FAILED=$((FAILED + 1))
    echo "[WARN] config A in batch $batch_num failed"
  fi

  if [[ -n "$pid_b" ]]; then
    if wait "$pid_b"; then
      COMPLETED=$((COMPLETED + 1))
    else
      FAILED=$((FAILED + 1))
      echo "[WARN] config B in batch $batch_num failed"
    fi
  fi
done

ELAPSED=$(( SECONDS - STARTED_AT ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  3D Sweep COMPLETE                                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Completed: $COMPLETED / $TOTAL_CONFIGS"
echo "║  Failed:    $FAILED"
echo "║  Elapsed:   ${ELAPSED_MIN} min (${ELAPSED} sec)"
echo "║"
echo "║  Results in: ./outputs/sweep3d_*_${TASK_TAG}*"
echo "║"
echo "║  Next step — analyze:"
echo "║    python scripts/analyze/taubench/analyze_3d_sweep.py"
echo "╚══════════════════════════════════════════════════════════════╝"

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
