# Oracle KV Token Eval

本项目提供 baseline、streamingLLM、h2o 三种 KV 保留策略，并封装了 OpenAI 风格接口。

算法与实现细节说明见 [docs/algorithms.md](docs/algorithms.md)。

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

这条命令现在会统一安装：

- 根项目运行所需依赖
- tau2-bench 的第三方依赖
- 本地可编辑安装的 tau2 包（等价于 `pip install -e ./tau2-bench`）

### 2) 启动服务（baseline 示例）

```bash
python api_server.py \
  --model-path ./local_models/Qwen3.5-9B \
  --served-model-name gpt-4o \
  --device cuda \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8000 \
  --method baseline
```

## 三种方法分别启动

建议用不同端口分别启动 3 个进程，便于对比评测。

```bash
# baseline
python api_server.py --model-path ./local_models/Qwen3.5-9B --served-model-name gpt-4o --device cuda --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8000 --method baseline

# streamingLLM
python api_server.py --model-path ./local_models/Qwen3.5-9B --served-model-name gpt-4o --device cuda --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8001 --method streamingllm

# h2o
python api_server.py --model-path ./local_models/Qwen3.5-9B --served-model-name gpt-4o --device cuda --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8002 --method h2o
```

## 关键参数说明

- --model-path: 本地模型目录。
- --served-model-name: 对外暴露的模型名（如 gpt-4o），用于 /v1/models 和响应 model 字段。
- --method: 固定当前服务实例的方法，可选 baseline / streamingllm / streaming_llm / h2o。
- --gpu-memory-utilization: 单进程可用显存上限比例，范围 (0, 1]。这是上限，不会强制占满。

## 接口列表

- GET /health
- GET /v1/models
- POST /v1/chat/completions
- POST /v1/completions
- POST /v1/evaluate

说明：

- baseline、streamingLLM、h2o 现在都走手写逐步解码路径。
- h2o 在同一解码框架上额外维护在线 score 计数器和 cache 裁剪逻辑。

## curl 测试

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
    "messages": [{"role": "user", "content": "请用一句话介绍你自己"}],
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0
  }'

# completions
curl -sS -X POST "http://127.0.0.1:8002/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "prompt": "请简述什么是Transformer。",
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0
  }'
```


## tau2-bench 接入建议

按方法分别评测时，将 base_url 指向不同端口：

- baseline: http://<host>:8000/v1
- streamingLLM: http://<host>:8001/v1
- h2o: http://<host>:8002/v1

模型名可统一写 gpt-4o（通过 --served-model-name 控制）。

## Python 侧调用

统一接口在 src/api.py，建议按包路径导入：

```python
from src.api import OracleKVProjectAPI

api = OracleKVProjectAPI(
    model_path="./local_models/Qwen3.5-9B",
    device="cuda",
    gpu_memory_utilization=0.9,
)
```

## 目录说明

- src/methods/baseline.py: baseline（full context）
- src/methods/streaming_llm.py: streamingLLM
- src/methods/h2o.py: h2o
- api_server.py: HTTP 服务入口
- offline_infer.py: 唯一离线推理入口（baseline / streamingLLM / h2o）

## KV Cache 说明

- baseline: 使用手写逐步缓存解码，KV cache 不裁剪。
- streamingLLM: 使用手写逐步缓存解码；在进入生成前保留 Sink Tokens + Recent Tokens，在生成过程中持续以滑动窗口方式裁剪 cache，cache 大小保持在 `sink_size + local_window_size` 以内。
- h2o: 以 streamingLLM 思想为基础，额外维护 Heavy Hitters。实现上会先对 full prompt 做一次带 attention 的 prefill，基于真实 attention 分数完成首次 h2o 裁剪；之后每个缓存 token 维护 score 计数器并按步累积注意力权重。当缓存超过预算时，仅在非 sink、非 recent 的普通 token 中淘汰累计得分最低者。

## 离线推理

`offline_infer.py` 现在支持两种数据格式：

- `jsonl`：兼容原有 `prompt` / `messages` 输入
- `tau2`：自动读取 `tasks.json`、`split_tasks.json`，并把任务转换为模型输入消息

当 `--methods` 传入多个方法时，脚本会按顺序执行并且每个方法使用独立子进程：

1. 先跑 `baseline`，进程结束
2. 再跑 `streamingllm`，进程结束
3. 再跑 `h2o`，进程结束

最后父进程汇总 `all_results.jsonl` 和 `summary.jsonl`。

默认数据集位置：

- 你自己的数据建议放在 [data/raw](data/raw) 下，例如 [data/raw/offline_samples.jsonl](data/raw/offline_samples.jsonl)
- 项目里提供了示例文件 [data/raw/offline_samples.example.jsonl](data/raw/offline_samples.example.jsonl)

默认结果输出位置：

- [outputs/offline_infer](outputs/offline_infer)
- 其中会生成 baseline.jsonl、streamingllm.jsonl、h2o.jsonl、all_results.jsonl、summary.jsonl
- 三种方法会分别独立运行后再汇总，便于逐方法对比

数据格式：

```json
{"id":"prompt_1","prompt":"请简述什么是Transformer。"}
{"id":"chat_1","messages":[{"role":"system","content":"你是一个简洁的中文助手。"},{"role":"user","content":"你的模型是什么？"}]}
```

启动方式：

```bash
cp data/raw/offline_samples.example.jsonl data/raw/offline_samples.jsonl

python offline_infer.py \
  --dataset-format jsonl \
  --model-path ./local_models/Qwen3.5-9B \
  --input-jsonl ./data/raw/offline_samples.jsonl \
  --output-dir ./outputs/offline_infer \
  --methods baseline streamingllm h2o \
  --device cuda \
  --dtype auto \
  --max-new-tokens 80 \
  --h2o-heavy-hitter-size 128
```

使用 tau2 airline 数据集评测（推荐）：

```bash
python offline_infer.py \
  --dataset-format tau2 \
  --tau2-domain-dir ./data/tau2/domains/airline \
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

如果需要输出逐步缓存轨迹：

```bash
python offline_infer.py \
  --model-path ./local_models/Qwen3.5-9B \
  --input-jsonl ./data/raw/offline_samples.jsonl \
  --output-dir ./outputs/offline_infer \
  --methods baseline streamingllm h2o \
  --dataset-format jsonl \
  --save-step-trace
```

tau2 相关参数：

- `--tau2-domain-dir`：如 `./data/tau2/domains/airline`
- `--tau2-split`：`train` / `test` / `base` / `all`
- `--tau2-include-policy`：把 `policy.md` 作为 system message 注入
- `--tau2-limit`：只跑前 N 条任务，便于快速 smoke test

## 离线功能 Smoke Test

项目里新增了两套小测试集，专门用于验证离线推理流程是否正常：

- JSONL smoke：`data/raw/offline_smoke_samples.jsonl`
- tau2 smoke：`data/tau2/domains/airline_smoke`

快速测试（JSONL）：

```bash
python offline_infer.py \
  --dataset-format jsonl \
  --input-jsonl ./data/raw/offline_smoke_samples.jsonl \
  --output-dir ./outputs/offline_smoke_jsonl \
  --methods baseline streamingllm h2o \
  --device cuda \
  --max-new-tokens 32 \
  --h2o-heavy-hitter-size 64
```

快速测试（tau2 smoke）：

```bash
python offline_infer.py \
  --dataset-format tau2 \
  --tau2-domain-dir ./data/tau2/domains/airline_smoke \
  --tau2-split base \
  --tau2-include-policy \
  --output-dir ./outputs/offline_smoke_tau2 \
  --methods baseline streamingllm h2o \
  --device cuda \
  --max-new-tokens 32 \
  --h2o-heavy-hitter-size 64
```

可以修改的地方：

- 数据文件路径：启动命令里的 --input-jsonl
- 输出目录：启动命令里的 --output-dir
- 模型路径：启动命令里的 --model-path
- 数据格式：--dataset-format jsonl/tau2
- streamingLLM 参数：--streaming-sink-size、--streaming-local-window-size
- h2o 参数：--h2o-sink-size、--h2o-local-window-size、--h2o-heavy-hitter-size

## tau2-bench 压力测试

### 脚本

项目提供三个评测脚本：

| 脚本 | 用途 |
|------|------|
| `scripts/run/run_tau2_offline_sequential.sh` | 单 GPU 顺序评测 baseline → streamingllm → h2o |
| `scripts/run/run_tau2_offline_parallel_multi_gpu.sh` | 多 GPU 并行评测 |
| `scripts/run/run_tau2_stress_test.sh` | H2O heavy-hitter 压力测试（推荐） |

### 压力测试

```bash
# 全部 6 组配置（baseline + streamingllm + h2o×4）
bash scripts/run/run_tau2_stress_test.sh

# Phase 1：baseline + streamingllm(heavy=0) + h2o(heavy=64) 三方核心对比
PHASE=1 bash scripts/run/run_tau2_stress_test.sh

# Phase 2：h2o heavy sweep（heavy=32, 128, 256）
PHASE=2 bash scripts/run/run_tau2_stress_test.sh
```

当前脚本默认参数：

| 参数 | 默认值 |
|------|--------|
| `MODEL_PATH` | `./local_models/Qwen3.5-9B` |
| `DOMAIN` | `airline` |
| `TASK_SPLIT` | `base` |
| `USER_LLM` | `deepseek/deepseek-chat` |
| `SERVED_MODEL_NAME` | `gpt-4o` |
| `HOST` | `127.0.0.1` |
| `TIMEOUT_SECONDS` | `800` |
| `TASK_TIMEOUT_SECONDS` | `800` |
| `MAX_CONCURRENCY` | `3` |
| `NUM_TRIALS` | `1` |
| `NUM_TASKS` | `50` |
| `AGENT_MAX_TOKENS` | `1024` |
| `SINK_SIZE` | `4` |
| `WINDOW_SIZE` | `128` |
| `GPU_MEMORY_UTILIZATION` | `0.9` |
| `DEVICE` | `cuda` |
| `DTYPE` | `auto` |
| `PHASE` | `0` |
| `GPU_A` | `2` |
| `GPU_B` | `3` |
| `GPU_C` | `4` |
| `PORT_BASELINE` | `8010` |
| `PORT_STREAMINGLLM` | `8011` |
| `PORT_H2O_64` | `8012` |
| `PORT_H2O_32` | `8013` |
| `PORT_H2O_128` | `8014` |
| `PORT_H2O_256` | `8015` |

参数网格（`WINDOW_SIZE` 固定，扫描 `heavy_hitter_size`）：

| 方法 | sink | window(固定) | heavy | cache 预算 |
|------|------|------------|-------|-----------|
| baseline | - | - | - | 无限制 |
| streamingllm | 4 | 128 | 0 | 132 ← 下界参考 |
| h2o | 4 | 128 | 32 | 164 |
| h2o | 4 | 128 | 64 | 196 |
| h2o | 4 | 128 | 128 | 260 |
| h2o | 4 | 128 | 256 | 388 |

说明：脚本当前不再包含 `heavy=512`，因为在默认 `AGENT_MAX_TOKENS=1024` 下，这个预算已经明显接近不裁剪场景，压缩对比度不够高。

Phase 组织方式：

| Phase | 内容 |
|------|------|
| `0` | 依次运行 Phase 1 和 Phase 2 |
| `1` | `baseline` + `streamingllm` + `h2o(heavy=64)` |
| `2` | `h2o(heavy=32)` + `h2o(heavy=128)` + `h2o(heavy=256)` |

可通过环境变量覆盖：

```bash
# 更换 window 大小（所有配置统一更换）
WINDOW_SIZE=256 bash scripts/run/run_tau2_stress_test.sh

# 完整覆盖示例
GPU_A=0 GPU_B=1 GPU_C=2 \
NUM_TASKS=30 \
WINDOW_SIZE=128 \
AGENT_MAX_TOKENS=1024 \
SINK_SIZE=4 \
bash scripts/run/run_tau2_stress_test.sh

tau2-bench 中 agent 需要生成：
- 工具调用 JSON（30-100 tokens）
- 自然语言回复（100-300 tokens）
- 复杂策略解释（200-500 tokens）

默认 `max_tokens=128` 容易导致回复被截断。当前项目脚本已统一提升默认值。建议：

- **tau2-bench 压力测试**：使用 1024（脚本默认值）
- **快速 smoke test**：使用 256
- **通用离线推理**：根据任务长度自定义
