#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Comprehensive StreamingLLM Sweep
#   window_size : {32, 64, 128, 256}
#   evict_period: {1, 2, 4, 8, 12, 16}
#   Total       : 4 × 6 = 24 configs (6 already in sweep3d → skip, 18 new)
#   GPUs        : 2, 3, 4  (3-way parallel)
#   Naming      : sl_s4_w{window}_ep{ep}_task10
#
# Usage:
#   bash scripts/run/run_sl_full_sweep_task10.sh
#   DRY_RUN=1 bash scripts/run/run_sl_full_sweep_task10.sh
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"

SIM_DIR="$ROOT_DIR/tau2-bench/data/simulations"

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
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"   # 3 slots/GPU × 0.30 ≈ 0.90 total
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SINK_SIZE=4

# 6 worker slots = 2 GPUs × 3 slots each
# GPU 2: ports 8030 8031 8032
# GPU 3: ports 8033 8034 8035
WORKER_GPUS=(6 6 6 7 7 7)
WORKER_PORTS=(8030 8031 8032 8033 8034 8035)
NUM_WORKERS=${#WORKER_GPUS[@]}   # 6

# task10 subset
TASK_IDS="${TASK_IDS:-0 1 3 4 5 6 10 13 26 28}"
TASK_TAG="task10"

DRY_RUN="${DRY_RUN:-0}"

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -f "$ROOT_DIR/tau2-bench/.env" ]]; then
  set -a; source "$ROOT_DIR/tau2-bench/.env"; set +a
fi
[[ -z "${DEEPSEEK_API_KEY:-}" ]] && echo "[WARN] DEEPSEEK_API_KEY is empty."

if command -v tau2 >/dev/null 2>&1; then
  TAU2_CMD=(tau2)
else
  TAU2_SRC="$ROOT_DIR/tau2-bench/src"
  [[ -d "$TAU2_SRC/tau2" ]] || { echo "[error] tau2 not found"; exit 1; }
  export PYTHONPATH="$TAU2_SRC${PYTHONPATH:+:$PYTHONPATH}"
  TAU2_CMD=(python -m tau2.cli)
fi

mkdir -p ./outputs
read -r -a TASK_ID_ARRAY <<< "${TASK_IDS//,/ }"

# ══════════════════════════════════════════════════════════════════════════════
# Full 24-config grid: "window|ep"
# ══════════════════════════════════════════════════════════════════════════════
ALL_CONFIGS=(
  "32|1"   "32|2"   "32|4"   "32|8"   "32|12"  "32|16"
  "64|1"   "64|2"   "64|4"   "64|8"   "64|12"  "64|16"
  "128|1"  "128|2"  "128|4"  "128|8"  "128|12" "128|16"
  "256|1"  "256|2"  "256|4"  "256|8"  "256|12" "256|16"
)

# ── Skip-check: returns 0 (run) or 1 (skip) ──────────────────────────────────
# Checks both the new canonical name AND the old sweep3d name.
should_skip() {
  local window="$1" ep="$2"
  local canonical="$SIM_DIR/sl_s4_w${window}_ep${ep}_${TASK_TAG}.json"
  local legacy="$SIM_DIR/sweep3d_sl_s4_w${window}_ep${ep}_${TASK_TAG}.json"
  if [[ -f "$canonical" ]]; then
    echo "[SKIP] already exists: sl_s4_w${window}_ep${ep}_${TASK_TAG}"
    return 1
  fi
  if [[ -f "$legacy" ]]; then
    echo "[SKIP] covered by sweep3d: sweep3d_sl_s4_w${window}_ep${ep}_${TASK_TAG}"
    return 1
  fi
  return 0
}

# Build the list of configs to actually run
PENDING_CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
  IFS='|' read -r window ep <<< "$cfg"
  if should_skip "$window" "$ep"; then
    PENDING_CONFIGS+=("$cfg")
  fi
done

TOTAL="${#PENDING_CONFIGS[@]}"

# ── Helpers ───────────────────────────────────────────────────────────────────
wait_for_health() {
  local port="$1"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -sSf "http://${HOST}:${port}/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

# run_one  window ep gpu port
run_one() {
  local window="$1" ep="$2" gpu="$3" port="$4"
  local budget=$(( SINK_SIZE + window ))
  local save_to="sl_s4_w${window}_ep${ep}_${TASK_TAG}"

  echo ""
  echo "┌──────────────────────────────────────────────────────────┐"
  printf "│ [START] sl  w=%-3s  ep=%-2s  budget=%-3s  gpu=%s  port=%s\n" \
    "$window" "$ep" "$budget" "$gpu" "$port"
  echo "│         save_to=$save_to"
  echo "└──────────────────────────────────────────────────────────┘"

  if [[ "$DRY_RUN" == "1" ]]; then echo "[DRY_RUN] skip"; return 0; fi

  # Remove incomplete sim file to avoid tau2's interactive "resume?" prompt.
  # A complete run for task10 has 10 simulations; anything less is a partial/corrupt file.
  local sim_file="$SIM_DIR/${save_to}.json"
  if [[ -f "$sim_file" ]]; then
    local n_sims
    n_sims=$(python3 -c "import json; d=json.load(open('$sim_file')); print(len(d.get('simulations',[])))" 2>/dev/null || echo 0)
    if (( n_sims < 10 )); then
      echo "[WARN] removing incomplete sim file ($n_sims/10 sims): $sim_file"
      rm -f "$sim_file"
    fi
  fi

  # Start api_server
  CUDA_VISIBLE_DEVICES="$gpu" python api_server.py \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$port" \
    --method streamingllm \
    --streaming-sink-size "$SINK_SIZE" \
    --streaming-local-window-size "$window" \
    --evict-period "$ep" \
    --attn-implementation sdpa \
    >"./outputs/${save_to}_server.log" 2>&1 &
  local server_pid=$!
  echo "[info] server pid=$server_pid (gpu=$gpu port=$port)"

  if ! wait_for_health "$port"; then
    echo "[ERROR] health check failed: $save_to"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi
  echo "[info] server ready"

  local agent_args
  agent_args=$(printf \
    '{"api_base":"http://%s:%s/v1","api_key":"EMPTY","temperature":0.0,"max_tokens":%s}' \
    "$HOST" "$port" "$AGENT_MAX_TOKENS")

  "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --task-ids "${TASK_ID_ARRAY[@]}" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --save-to "$save_to" \
    >"./outputs/${save_to}_tau2.log" 2>&1
  local tau2_exit=$?

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true

  [[ $tau2_exit -eq 0 ]] \
    && echo "[DONE] ✓ $save_to" \
    || echo "[FAIL] ✗ $save_to (exit=$tau2_exit)"
  return $tau2_exit
}

# ══════════════════════════════════════════════════════════════════════════════
# Main: 3-way parallel over GPU_A / GPU_B / GPU_C
# ══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  StreamingLLM Full Sweep (task10)  — 6-way parallel        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model : $MODEL_PATH"
echo "║  Workers: GPU2×3(8030-32)  GPU3×3(8033-35)"
echo "║  mem_util/slot: $GPU_MEMORY_UTILIZATION  (3 slots × 0.30 ≈ 0.90)"
echo "║  Sink  : $SINK_SIZE   Windows: {32,64,128,256}   ep: {1,2,4,8,12,16}"
echo "║  Total grid : 24   Pending (skip existing): $TOTAL"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

STARTED_AT=$SECONDS
COMPLETED=0
FAILED=0

batch_num=0
for (( i=0; i < TOTAL; i+=NUM_WORKERS )); do
  batch_num=$(( batch_num + 1 ))
  batch_last=$(( (TOTAL + NUM_WORKERS - 1) / NUM_WORKERS ))
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Batch $batch_num / $batch_last"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  declare -a batch_pids=()
  for (( slot=0; slot<NUM_WORKERS; slot++ )); do
    idx=$(( i + slot ))
    [[ $idx -ge $TOTAL ]] && break
    IFS='|' read -r window ep <<< "${PENDING_CONFIGS[$idx]}"
    run_one "$window" "$ep" "${WORKER_GPUS[$slot]}" "${WORKER_PORTS[$slot]}" &
    batch_pids+=($!)
  done

  for pid in "${batch_pids[@]}"; do
    if wait "$pid"; then
      COMPLETED=$(( COMPLETED + 1 ))
    else
      FAILED=$(( FAILED + 1 ))
      echo "[WARN] a job in batch $batch_num failed (pid=$pid)"
    fi
  done
done

ELAPSED=$(( SECONDS - STARTED_AT ))
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  StreamingLLM Sweep COMPLETE                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Ran / Completed / Failed : $TOTAL / $COMPLETED / $FAILED"
echo "║  Elapsed : $(( ELAPSED/60 )) min (${ELAPSED}s)"
echo "║"
echo "║  New results in : ./outputs/sl_s4_w*_ep*_task10*"
echo "║  Combined analysis (includes sweep3d):"
echo "║    python scripts/analyze/analyze_sl_full_sweep.py"
echo "╚══════════════════════════════════════════════════════════════╝"

[[ $FAILED -gt 0 ]] && exit 1 || exit 0
