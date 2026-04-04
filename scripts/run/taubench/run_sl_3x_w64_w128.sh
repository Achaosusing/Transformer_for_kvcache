#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# StreamingLLM 3-Trial Sweep — w64 & w128, all evict_period values
#
#   Purpose : Quantify evict_period effect with 3 independent trials
#             to separate signal from randomness.
#
#   Grid    : w={64, 128} × ep={1,2,4,8,12,16} = 12 configs per trial
#   Trials  : r1, r2, r3  → 36 total runs
#   Naming  : sl3x_w{w}_ep{ep}_task10_r{n}
#             (distinct from sl_s4_* and sweep3d_sl_s4_*)
#
#   GPUs    : 6, 7  (3 slots each → 6-way parallel)
#   Ports   : 8060 8061 8062  (GPU6)
#             8063 8064 8065  (GPU7)
#   Batches : 2 batches × 3 trials = 6 batches total
#
# Usage:
#   bash scripts/run/taubench/run_sl_3x_w64_w128.sh
#   DRY_RUN=1 bash scripts/run/taubench/run_sl_3x_w64_w128.sh
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
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"   # 3 slots/GPU × 0.30 ≈ 0.90
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SINK_SIZE=4

# 6 worker slots: GPU6 × 3 + GPU7 × 3
WORKER_GPUS=(6 6 6 7 7 7)
WORKER_PORTS=(8060 8061 8062 8063 8064 8065)
NUM_WORKERS=${#WORKER_GPUS[@]}   # 6

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

# ── Config grid: w={64,128} × ep={1,2,4,8,12,16} ─────────────────────────────
BASE_CONFIGS=(
  "64|1"   "64|2"   "64|4"   "64|8"   "64|12"  "64|16"
  "128|1"  "128|2"  "128|4"  "128|8"  "128|12" "128|16"
)
TRIAL_IDS=(1 2 3)

# ── Build pending list: (window|ep|trial) ────────────────────────────────────
# Skip only if this exact trial's file is already complete (≥10 sims).
# Existing sl_s4_* / sweep3d_sl_s4_* files are NOT treated as trials here.
PENDING=()
for trial in "${TRIAL_IDS[@]}"; do
  for cfg in "${BASE_CONFIGS[@]}"; do
    IFS='|' read -r window ep <<< "$cfg"
    save_to="sl3x_w${window}_ep${ep}_${TASK_TAG}_r${trial}"
    f="$SIM_DIR/${save_to}.json"
    if [[ -f "$f" ]]; then
      n=$(python3 -c "import json; print(len(json.load(open('$f')).get('simulations',[])))" 2>/dev/null || echo 0)
      if (( n >= 10 )); then
        echo "[SKIP] complete (${n}/10) — ${save_to}"
        continue
      else
        echo "[WARN] incomplete (${n}/10) — ${save_to}  (will re-run)"
      fi
    fi
    PENDING+=("${window}|${ep}|${trial}")
  done
done

TOTAL="${#PENDING[@]}"

# ── Helpers ───────────────────────────────────────────────────────────────────
wait_for_health() {
  local port="$1"
  local deadline=$(( SECONDS + TIMEOUT_SECONDS ))
  while (( SECONDS < deadline )); do
    if curl -sSf "http://${HOST}:${port}/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

# run_one  window ep trial gpu port
run_one() {
  local window="$1" ep="$2" trial="$3" gpu="$4" port="$5"
  local budget=$(( SINK_SIZE + window ))
  local save_to="sl3x_w${window}_ep${ep}_${TASK_TAG}_r${trial}"

  echo ""
  echo "┌──────────────────────────────────────────────────────────┐"
  printf "│ [START] sl  w=%-3s  ep=%-2s  r=%s  budget=%-3s  gpu=%s  port=%s\n" \
    "$window" "$ep" "$trial" "$budget" "$gpu" "$port"
  echo "│         save_to=$save_to"
  echo "└──────────────────────────────────────────────────────────┘"

  if [[ "$DRY_RUN" == "1" ]]; then echo "[DRY_RUN] skip"; return 0; fi

  # Remove incomplete sim file to prevent tau2's interactive "resume?" prompt
  local sim_file="$SIM_DIR/${save_to}.json"
  if [[ -f "$sim_file" ]]; then
    local n_sims
    n_sims=$(python3 -c "import json; print(len(json.load(open('$sim_file')).get('simulations',[])))" 2>/dev/null || echo 0)
    if (( n_sims < 10 )); then
      echo "[WARN] removing incomplete sim file (${n_sims}/10): $sim_file"
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
# Main: 6-way parallel (GPU6×3 + GPU7×3)
# ══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  StreamingLLM 3-Trial Sweep — w64 & w128                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model  : $MODEL_PATH"
echo "║  Workers: GPU6×3(8060-62)  GPU7×3(8063-65)"
echo "║  Grid   : w={64,128} × ep={1,2,4,8,12,16} × 3 trials"
echo "║  Naming : sl3x_w{w}_ep{ep}_task10_r{1,2,3}"
echo "║  Total  : 36 configs  Pending: $TOTAL"
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
  printf "  Batch %d / %d\n" "$batch_num" "$batch_last"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  declare -a batch_pids=()
  for (( slot=0; slot<NUM_WORKERS; slot++ )); do
    idx=$(( i + slot ))
    [[ $idx -ge $TOTAL ]] && break
    IFS='|' read -r window ep trial <<< "${PENDING[$idx]}"
    run_one "$window" "$ep" "$trial" "${WORKER_GPUS[$slot]}" "${WORKER_PORTS[$slot]}" &
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
echo "║  3-Trial Sweep COMPLETE                                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Total / Completed / Failed : $TOTAL / $COMPLETED / $FAILED"
echo "║  Elapsed : $(( ELAPSED/60 )) min (${ELAPSED}s)"
echo "║"
echo "║  Run analysis after completion:"
echo "║    python scripts/analyze/taubench/analyze_sl_3x_ep_effect.py"
echo "╚══════════════════════════════════════════════════════════════╝"

[[ $FAILED -gt 0 ]] && exit 1 || exit 0
