#!/usr/bin/env bash
set -euo pipefail

# Strict StreamingLLM evict_period measurement.
#
# Design goals:
#   1. Fix window to remove the main confound.
#   2. Use more tasks than task10 so ep effects are not hidden by 0.1 reward granularity.
#   3. Repeat with different task subsets to estimate variance.
#   4. Randomize ep-to-GPU-slot assignment each repeat to reduce system bias.
#   5. Keep the exact same task subset within a repeat across all ep values for paired comparison.
#
# Default design:
#   - window fixed at 128
#   - task pool: a feasible task subset provided by user, or airline/base if no
#     feasible-pool override is supplied
#   - tasks per repeat: sampled without replacement from that task pool
#   - repeats: 5
#   - ep values: {1,2,4,8,12,16}
#   - one repeat per batch, 6-way parallel across GPU6/GPU7
#
# By default this yields 50 task instances per ep (10 tasks x 5 repeats), while
# preserving paired comparisons inside each repeat. If you have a larger stable
# feasible pool, increase TASKS_PER_REPEAT or provide a larger TASK_POOL_IDS.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"

SIM_DIR="$ROOT_DIR/tau2-bench/data/simulations"
AIRLINE_SPLIT_FILE="$ROOT_DIR/tau2-bench/data/tau2/domains/airline/split_tasks.json"
OUT_DIR="$ROOT_DIR/outputs/strict_ep_measurement"
MANIFEST_DIR="$OUT_DIR/manifests"

# Shared config
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
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"

# Strict experiment design
WINDOW="${WINDOW:-128}"
SINK_SIZE="${SINK_SIZE:-4}"
TASK_POOL_SPLIT="${TASK_POOL_SPLIT:-base}"
TASK_POOL_IDS="${TASK_POOL_IDS:-}"
TASK_POOL_FILE="${TASK_POOL_FILE:-}"
TASKS_PER_REPEAT="${TASKS_PER_REPEAT:-10}"
NUM_REPEATS="${NUM_REPEATS:-5}"
MASTER_SEED="${MASTER_SEED:-20260331}"
EVICT_PERIODS="${EVICT_PERIODS:-1 2 4 8 12 16}"
DRY_RUN="${DRY_RUN:-0}"

# 6 worker slots: GPU6 x 3 + GPU7 x 3
WORKER_GPUS=(6 6 6 7 7 7)
WORKER_PORTS=(8060 8061 8062 8063 8064 8065)
NUM_WORKERS=${#WORKER_GPUS[@]}

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

mkdir -p "$OUT_DIR" "$MANIFEST_DIR"

[[ -f "$AIRLINE_SPLIT_FILE" ]] || { echo "[error] missing split file: $AIRLINE_SPLIT_FILE"; exit 1; }

wait_for_health() {
  local port="$1"
  local deadline=$(( SECONDS + TIMEOUT_SECONDS ))
  while (( SECONDS < deadline )); do
    if curl -sSf "http://${HOST}:${port}/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

save_to_name() {
  local ep="$1" repeat="$2"
  printf "sl_epstrict_w%s_pool%s_n%s_ep%s_rep%02d_s%s" \
    "$WINDOW" "$TASK_POOL_SPLIT" "$TASKS_PER_REPEAT" "$ep" "$repeat" "$MASTER_SEED"
}

generate_repeat_plan() {
  python3 - "$AIRLINE_SPLIT_FILE" "$TASK_POOL_SPLIT" "$TASK_POOL_IDS" "$TASK_POOL_FILE" "$TASKS_PER_REPEAT" "$NUM_REPEATS" "$MASTER_SEED" "$EVICT_PERIODS" <<'PY'
import json
import random
import sys

split_file, split_name, task_pool_ids, task_pool_file, tasks_per_repeat, num_repeats, master_seed, eps_str = sys.argv[1:9]
tasks_per_repeat = int(tasks_per_repeat)
num_repeats = int(num_repeats)
master_seed = int(master_seed)
eps = [int(x) for x in eps_str.split() if x]

if task_pool_ids.strip():
  pool = [x for x in task_pool_ids.replace(',', ' ').split() if x]
elif task_pool_file.strip():
  with open(task_pool_file) as f:
    raw = f.read().strip()
  try:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
      raise SystemExit(f"TASK_POOL_FILE must contain a JSON list, got {type(parsed).__name__}")
    pool = [str(x) for x in parsed]
  except json.JSONDecodeError:
    pool = [x for x in raw.replace(',', ' ').split() if x]
else:
  with open(split_file) as f:
    split_map = json.load(f)

  if split_name not in split_map:
    raise SystemExit(f"unknown split: {split_name}")

  pool = list(split_map[split_name])

pool = [str(x) for x in pool]
pool = list(dict.fromkeys(pool))
if tasks_per_repeat > len(pool):
    raise SystemExit(
    f"TASKS_PER_REPEAT={tasks_per_repeat} exceeds pool size {len(pool)}"
    )
if len(eps) != 6:
    raise SystemExit(f"expected 6 ep values for 6 workers, got {len(eps)}: {eps}")

for repeat in range(1, num_repeats + 1):
    rng = random.Random(master_seed + repeat)
    task_ids = rng.sample(pool, tasks_per_repeat)
    ep_order = eps[:]
    rng.shuffle(ep_order)
    print(f"{repeat}|{' '.join(task_ids)}|{' '.join(str(x) for x in ep_order)}")
PY
}

run_one() {
  local repeat="$1" ep="$2" gpu="$3" port="$4" task_ids_str="$5"
  local budget=$(( SINK_SIZE + WINDOW ))
  local save_to
  save_to=$(save_to_name "$ep" "$repeat")

  echo ""
  echo "┌──────────────────────────────────────────────────────────┐"
  printf "│ [START] strict-ep  rep=%-2s ep=%-2s w=%-3s budget=%-3s gpu=%s port=%s\n" \
    "$repeat" "$ep" "$WINDOW" "$budget" "$gpu" "$port"
  echo "│         save_to=$save_to"
  echo "└──────────────────────────────────────────────────────────┘"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] tasks=$task_ids_str"
    return 0
  fi

  local sim_file="$SIM_DIR/${save_to}.json"
  if [[ -f "$sim_file" ]]; then
    local n_sims
    n_sims=$(python3 -c "import json; print(len(json.load(open('$sim_file')).get('simulations',[])))" 2>/dev/null || echo 0)
    if (( n_sims < TASKS_PER_REPEAT )); then
      echo "[WARN] removing incomplete sim file (${n_sims}/${TASKS_PER_REPEAT}): $sim_file"
      rm -f "$sim_file"
    fi
  fi

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
    --streaming-local-window-size "$WINDOW" \
    --evict-period "$ep" \
    --attn-implementation sdpa \
    >"$OUT_DIR/${save_to}_server.log" 2>&1 &
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

  local -a task_id_array
  read -r -a task_id_array <<< "$task_ids_str"

  "${TAU2_CMD[@]}" run \
    --domain "$DOMAIN" \
    --task-split-name "$TASK_SPLIT" \
    --task-ids "${task_id_array[@]}" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --agent-llm-args "$agent_args" \
    --user-llm-args '{"temperature":0.0}' \
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-trials "$NUM_TRIALS" \
    --save-to "$save_to" \
    >"$OUT_DIR/${save_to}_tau2.log" 2>&1
  local tau2_exit=$?

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true

  [[ $tau2_exit -eq 0 ]] \
    && echo "[DONE] $save_to" \
    || echo "[FAIL] $save_to (exit=$tau2_exit)"
  return $tau2_exit
}

declare -A REPEAT_TASK_IDS=()
declare -A REPEAT_EP_ORDER=()
PENDING=()

while IFS='|' read -r repeat task_ids ep_order; do
  REPEAT_TASK_IDS["$repeat"]="$task_ids"
  REPEAT_EP_ORDER["$repeat"]="$ep_order"

  manifest_path="$MANIFEST_DIR/repeat_$(printf '%02d' "$repeat").txt"
  {
    echo "repeat=$repeat"
    echo "window=$WINDOW"
    echo "task_pool_split=$TASK_POOL_SPLIT"
    echo "task_pool_ids=${TASK_POOL_IDS:-<from split/file>}"
    echo "task_pool_file=${TASK_POOL_FILE:-<none>}"
    echo "tasks_per_repeat=$TASKS_PER_REPEAT"
    echo "master_seed=$MASTER_SEED"
    echo "task_ids=$task_ids"
    echo "ep_order=$ep_order"
  } > "$manifest_path"

  read -r -a ep_array <<< "$ep_order"
  for ep in "${ep_array[@]}"; do
    save_to=$(save_to_name "$ep" "$repeat")
    sim_path="$SIM_DIR/${save_to}.json"
    if [[ -f "$sim_path" ]]; then
      n=$(python3 -c "import json; print(len(json.load(open('$sim_path')).get('simulations',[])))" 2>/dev/null || echo 0)
      if (( n >= TASKS_PER_REPEAT )); then
        echo "[SKIP] complete (${n}/${TASKS_PER_REPEAT}) - ${save_to}"
        continue
      fi
      echo "[WARN] incomplete (${n}/${TASKS_PER_REPEAT}) - ${save_to}  (will re-run)"
    fi
    PENDING+=("${repeat}|${ep}|${task_ids}")
  done
done < <(generate_repeat_plan)

TOTAL="${#PENDING[@]}"
BATCHES=$(( (TOTAL + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "=============================================================="
echo "Strict StreamingLLM evict_period measurement"
echo "window=$WINDOW  sink=$SINK_SIZE  budget=$(( SINK_SIZE + WINDOW ))"
echo "task pool split=$TASK_POOL_SPLIT  tasks/repeat=$TASKS_PER_REPEAT"
echo "task pool ids=${TASK_POOL_IDS:-<from split/file>}"
echo "task pool file=${TASK_POOL_FILE:-<none>}"
echo "repeats=$NUM_REPEATS  master_seed=$MASTER_SEED"
echo "ep values=$EVICT_PERIODS"
echo "workers=GPU6x3 + GPU7x3  total jobs=$TOTAL  batches=$BATCHES"
echo "manifest dir=$MANIFEST_DIR"
echo "=============================================================="

STARTED_AT=$SECONDS
COMPLETED=0
FAILED=0
batch_num=0

for (( i=0; i < TOTAL; i+=NUM_WORKERS )); do
  batch_num=$(( batch_num + 1 ))
  echo ""
  echo "=============================================================="
  printf "Batch %d / %d\n" "$batch_num" "$BATCHES"
  repeat_preview=$(echo "${PENDING[$i]}" | cut -d'|' -f1)
  echo "repeat=$repeat_preview  task_ids=${REPEAT_TASK_IDS[$repeat_preview]}"
  echo "ep order=${REPEAT_EP_ORDER[$repeat_preview]}"
  echo "=============================================================="

  declare -a batch_pids=()
  for (( slot=0; slot<NUM_WORKERS; slot++ )); do
    idx=$(( i + slot ))
    [[ $idx -ge $TOTAL ]] && break
    IFS='|' read -r repeat ep task_ids <<< "${PENDING[$idx]}"
    run_one "$repeat" "$ep" "${WORKER_GPUS[$slot]}" "${WORKER_PORTS[$slot]}" "$task_ids" &
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
echo "=============================================================="
echo "Strict measurement complete"
echo "Total / Completed / Failed : $TOTAL / $COMPLETED / $FAILED"
echo "Elapsed : $(( ELAPSED/60 )) min (${ELAPSED}s)"
echo "Analyze with: python scripts/analyze/taubench/analyze_sl_ep_strict_measurement.py"
echo "=============================================================="

[[ $FAILED -gt 0 ]] && exit 1 || exit 0