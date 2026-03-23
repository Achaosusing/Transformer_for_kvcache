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
- streamingLLM: 使用手写逐步缓存解码；仅在进入生成前保留 Sink Tokens + Recent Tokens。
- h2o: 以 streamingLLM 为基础，额外维护 Heavy Hitters。每个缓存 token 有一个 score 计数器，按每步解码时的注意力权重累计更新；当缓存超过预算时，仅在非 sink、非 recent 的普通 token 中淘汰累计得分最低者。

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
