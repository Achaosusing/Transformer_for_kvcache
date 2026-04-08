# Oracle KV Token Eval

本仓库面向本地 Transformer 模型，提供三种 KV cache 策略的统一评测入口：

- `baseline`
- `streamingllm` / `streaming_llm`
- `h2o`

当前主入口：

- `api_server.py`：OpenAI 风格 HTTP 服务
- `offline_infer.py`：离线批量推理入口
- `src/api.py`：统一 Python 调用接口
- [docs/algorithms.md](docs/algorithms.md)：算法与实现细节
- [docs/attention_analysis_guide.md](docs/attention_analysis_guide.md)：attention-by-role 分析说明

## 快速开始

### 推荐安装方式

```bash
bash scripts/run/bootstrap_all.sh
source ./.venv/bin/activate
```

这会：

- 创建或复用 `.venv`
- 安装 `requirements.txt`
- 以 editable 模式安装根项目
- 验证 `tau2` 命令和 `src` 包可用

### 手动安装

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 启动服务

建议为不同方法启动不同端口，便于对比评测。

```bash
# baseline
python api_server.py \
  --model-path ./local_models/Qwen3.5-9B \
  --served-model-name gpt-4o \
  --device cuda \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8000 \
  --method baseline

# streamingLLM
python api_server.py \
  --model-path ./local_models/Qwen3.5-9B \
  --served-model-name gpt-4o \
  --device cuda \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8001 \
  --method streamingllm \
  --streaming-sink-size 4 \
  --streaming-local-window-size 256

# h2o
python api_server.py \
  --model-path ./local_models/Qwen3.5-9B \
  --served-model-name gpt-4o \
  --device cuda \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8002 \
  --method h2o \
  --h2o-sink-size 4 \
  --h2o-local-window-size 256 \
  --h2o-heavy-hitter-size 128 \
  --evict-period 1 \
  --collect-period 0
```

### 服务端关键参数

- `--model-path`：本地模型目录。
- `--served-model-name`：对外暴露给 `/v1/models` 和响应 `model` 字段的名称。
- `--method`：固定当前服务实例的方法；可选 `baseline` / `streamingllm` / `streaming_llm` / `h2o`。
- `--max-new-tokens`：服务端默认生成上限。
- `--streaming-sink-size` / `--streaming-local-window-size`：streamingLLM 预算参数。
- `--h2o-sink-size` / `--h2o-local-window-size` / `--h2o-heavy-hitter-size`：H2O 预算参数。
- `--evict-period`：按批次裁剪 cache。`1` 表示每个 decode step 都严格裁剪；更大的值会减少裁剪开销，但允许 cache 临时超预算 `evict_period - 1` 个 token。
- `--collect-period`：H2O 收集 attention 的步频。`0` 表示跟随 `evict_period`；`1` 表示每步收集；更大值会减少 attention 读取次数。
- `--attn-implementation`：服务端默认 `auto`，当前会解析为 `sdpa`。这是为了避免长 prefill 时的全量 attention 矩阵 OOM。

## 接口列表

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/evaluate`

当前行为以 `api_server.py` 为准：

- `stream=true` 目前不支持。
- 如果服务通过 `--method` 固定了方法，所有请求都会只走该方法。
- 如果服务未固定方法：
  - `/v1/evaluate` 默认同时运行 `baseline`、`streamingllm`、`h2o`
  - `/v1/chat/completions` 和 `/v1/completions` 默认只运行 `baseline`
- `/v1/chat/completions` 和 `/v1/completions` 也可以通过请求体里的 `methods` 指定多方法，此时响应里的 `choices` 会额外带上 `method` 字段。
- `h2o` 的 `/v1/chat/completions` 现在支持可选 `session_id`。当后续请求满足“上一轮完整历史 + 严格尾部追加”时，服务端会尝试复用上一轮的 H2O 活动 KV snapshot，而不是重新 prefill 全部历史。
- 上面的会话复用路径只在“单方法 `h2o` chat 请求”里生效；当前不支持和 `max_input_tokens` 组合使用。
- `h2o` 的 chat 路径支持 OpenAI 风格的 `tools`、`tool_choice`（当前不支持显式对象形式的 `tool_choice`），响应里会在适用时返回 `tool_calls`。

### curl 示例

```bash
# 健康检查
curl -sS http://127.0.0.1:8000/health

# 模型列表
curl -sS http://127.0.0.1:8000/v1/models

# chat/completions
curl -sS -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己"}
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0
  }'

# h2o + session_id + tools
curl -sS -X POST "http://127.0.0.1:8002/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "session_id": "demo-session-1",
    "methods": ["h2o"],
    "messages": [
      {"role": "user", "content": "帮我查询订单 123 的状态"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "lookup_order",
          "description": "查询订单状态",
          "parameters": {
            "type": "object",
            "properties": {
              "order_id": {"type": "string"}
            },
            "required": ["order_id"]
          }
        }
      }
    ],
    "temperature": 0.0,
    "top_p": 1.0,
    "method_configs": {
      "h2o": {
        "sink_size": 4,
        "local_window_size": 256,
        "heavy_hitter_size": 128,
        "session_score_alpha": 0.5
      }
    }
  }'

# evaluate：一次比较三种方法
curl -sS -X POST "http://127.0.0.1:8000/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "id": "chat_0",
        "messages": [
          {"role": "user", "content": "请简述 Transformer 的核心思想。"}
        ]
      }
    ],
    "methods": ["baseline", "streamingllm", "h2o"],
    "method_configs": {
      "streamingllm": {"sink_size": 4, "local_window_size": 256},
      "h2o": {"sink_size": 4, "local_window_size": 256, "heavy_hitter_size": 128}
    },
    "max_new_tokens": 64
  }'
```

## 三种 KV 策略的当前实现

更完整的说明见 [docs/algorithms.md](docs/algorithms.md)。这里仅列当前代码里的关键事实：

- `baseline`：手写逐步缓存解码，不做 cache 裁剪。
- `streamingLLM`：prompt 在进入生成前会先裁成 `sink + recent`；生成时继续按位置滑窗裁剪，cache 预算是 `sink_size + local_window_size`。
- `h2o`：当前主路径不会在 prefill 阶段读取 full-prompt attention。实现流程是：
  1. 对完整 prompt 做普通 prefill
  2. 用全零 `score_counters` 初始化当前活动 cache
  3. 若 prompt 已超预算，首次裁剪会退化为“零分 + recent tie-break”，效果接近 `sink + recent`
  4. 只有在 decode 过程中，且满足“本步需要裁剪”或“达到 `collect_period`”时，才会调用带 attention 的增量前向并累积分数
  5. 分数只对当前活动 cache 生效；被驱逐 token 的历史分数不会保留
  6. `chat/completions` 的会话模式里，跨轮恢复时会先对上一轮 `score_counters` 做 Max 归一化，再乘 `session_score_alpha`，然后只对严格追加的尾部 token 做增量续写

如果旧文档还写着“h2o 用 full-prompt attention 完成首次 heavy-hitter 打分”，那已经不是当前实现。

## Python 侧调用

```python
from src.api import OracleKVProjectAPI

api = OracleKVProjectAPI(
    model_path="./local_models/Qwen3.5-9B",
    device="cuda",
    gpu_memory_utilization=0.9,
    dtype="auto",
    attn_implementation="sdpa",
)
```

统一入口在 `src/api.py`。核心公开方法是 `evaluate(...)`。

## 离线推理

`offline_infer.py` 支持两种输入格式：

- `jsonl`：每行一个样本，支持 `prompt` 或 `messages`
- `tau2`：从 tau2 domain 目录读取 `tasks.json`、`split_tasks.json` 和可选的 `policy.md`

### JSONL 输入格式

```json
{"id":"prompt_1","prompt":"请简述什么是 Transformer。"}
{"id":"chat_1","messages":[{"role":"system","content":"你是一个简洁的中文助手。"},{"role":"user","content":"你的模型是什么？"}]}
```

仓库当前不再维护根目录 `data/` 示例集。JSONL 输入文件请自行准备，然后把路径传给 `--input-jsonl`。

### JSONL 示例

```bash
python offline_infer.py \
  --dataset-format jsonl \
  --model-path ./local_models/Qwen3.5-9B \
  --input-jsonl /path/to/offline_samples.jsonl \
  --output-dir ./outputs/offline_infer \
  --methods baseline streamingllm h2o \
  --device cuda \
  --dtype auto \
  --max-new-tokens 80 \
  --h2o-heavy-hitter-size 128
```

### tau2 domain 示例

仓库内可直接使用的 airline domain 在：

- `./tau2-bench/data/tau2/domains/airline`

示例命令：

```bash
python offline_infer.py \
  --dataset-format tau2 \
  --tau2-domain-dir ./tau2-bench/data/tau2/domains/airline \
  --tau2-split test \
  --tau2-include-policy \
  --model-path ./local_models/Qwen3.5-9B \
  --output-dir ./outputs/offline_infer_tau2_airline \
  --methods baseline streamingllm h2o \
  --device cuda \
  --dtype auto \
  --max-new-tokens 80 \
  --h2o-heavy-hitter-size 128
```

### 离线推理的当前语义

- 当 `--methods` 传入多个方法时，脚本会按方法顺序拉起独立子进程，逐个完成推理，再汇总 `all_results.jsonl` 和 `summary.jsonl`。这样做主要是为了隔离显存状态，减少多方法混跑带来的碎片或 OOM。
- `tau2` 模式会把每个 task 压成一条 `user` 消息；如果传了 `--tau2-include-policy`，会额外把 `policy.md` 放成一条 `system` 消息。
- 因此 `offline_infer.py --dataset-format tau2` 适合做“单轮离线对比”，不等价于 tau2-bench 的真实多轮 agent 对话仿真。
- `offline_infer.py` 的 `--attn-implementation auto` 在方法集合里包含 `h2o` 时会解析成 `eager`；如果你在长上下文上更关心 prefill 显存，可以显式传 `--attn-implementation sdpa` 再自行验证内存与结果。

### 输出文件

无论 `jsonl` 还是 `tau2` 模式，`--output-dir` 下都会生成：

- `baseline.jsonl`
- `streamingllm.jsonl`
- `h2o.jsonl`
- `all_results.jsonl`
- `summary.jsonl`

如果传入 `--save-step-trace`，每条结果还会带上 `step_trace` 字段，记录 `full_context_tokens`、`kept_tokens` 和 `kept_ratio`。

## tau2-bench 全链路接入

按方法分别评测时，把 tau2 的 `agent-llm` 指向不同端口即可：

- `baseline`：`http://<host>:8000/v1`
- `streamingLLM`：`http://<host>:8001/v1`
- `h2o`：`http://<host>:8002/v1`

推荐先看这两个和当前 `api_server.py` CLI 对齐的脚本：

- `scripts/run/taubench/run_tau2_offline_sequential.sh`
- `scripts/run/taubench/run_tau2_offline_parallel_multi_gpu.sh`

补充说明：

- 这些脚本默认只把 `agent-llm` 指到本地 OpenAI-compatible server。
- `USER_LLM` 默认仍是 `deepseek/deepseek-chat`，所以脚本默认不是“全链路离线”。
- 如果你要做真正的本地闭环评测，还需要把 `user-llm` 一并改到本地 provider / api_base。
- `scripts/run/taubench/` 下还有一批 sweep / stress 实验脚本，但其中一些带有实验性参数。实际可用参数以 `api_server.py` 当前 CLI 为准。

## Attention 分析

attention-by-role 分析脚本当前位于：

- `scripts/analyze/taubench/analyze_attention_by_role.py`

对应说明文档：

- [docs/attention_analysis_guide.md](docs/attention_analysis_guide.md)

默认输出目录是 `outputs/attention_task10`。脚本支持一次读取一个或多个 simulation JSON 文件。

## 目录说明

- `api_server.py`：HTTP 服务入口
- `offline_infer.py`：离线批量推理入口
- `src/api.py`：统一评测 API
- `src/model.py`：模型加载、tokenization、cache 裁剪和 attention 聚合
- `src/methods/baseline.py`：baseline policy
- `src/methods/streaming_llm.py`：streamingLLM policy
- `src/methods/h2o.py`：H2O policy
- `scripts/run/bootstrap_all.sh`：推荐的一键环境初始化脚本
- `scripts/run/taubench/`：tau2 自动化运行脚本
- `scripts/analyze/taubench/`：tau2 结果分析脚本
- `tau2-bench/`：tau2 上游代码和数据