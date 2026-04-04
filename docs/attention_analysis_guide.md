# Attention-by-Role 分析说明

本文档对应当前脚本：

- `scripts/analyze/taubench/analyze_attention_by_role.py`

目标是说明这个脚本到底在测什么、怎么划分角色、如何定义 turn distance，以及最终输出了哪些文件。

## 1. 脚本位置与默认行为

当前脚本默认参数以源码为准：

- 脚本路径：`scripts/analyze/taubench/analyze_attention_by_role.py`
- 默认模型：`./local_models/Qwen3.5-9B`
- 默认 simulation 文件：`tau2-bench/data/simulations/` 下的 4 个 `stress_baseline_1x10_task10_*.json`
- 默认输出目录：`outputs/attention_task10`
- 默认设备：`cuda:2`

脚本在加载模型时会：

1. 使用本地模型文件（`local_files_only=True`）
2. 打开 `trust_remote_code=True`
3. 强制 `attn_implementation="eager"`，因为它需要 `output_attentions=True`

## 2. `fine_role` 的当前定义

脚本最终输出的细粒度角色不止四类，而是最多六类：

| 原始条件 | `fine_role` |
|---|---|
| 第 0 条消息且 `role == "assistant"` | `asst_greeting` |
| `role == "assistant"` 且内容命中工具调用正则 | `asst_tool_call` |
| `role == "assistant"` 且未命中工具调用正则 | `asst_gen` |
| `role == "user"` | `user_msg` |
| `role == "system"` | `system` |
| `role == "tool"` | `tool_result` |

工具调用正则当前是：

```python
TOOL_CALL_RE = re.compile(
    r'```json\s*\{|\[call_tool\]|\[function_call\]|<tool_call>',
    re.IGNORECASE,
)
```

因此，当前脚本不只识别旧格式的 `[call_tool]`，也会把以 ```json 开头的工具调用块识别为 `asst_tool_call`。

### 2.1 第一条 assistant 消息的特殊处理

tau2-bench 对话通常以 assistant greeting 开头，但 Qwen 的 chat template 通常要求对话从 system 或 user 开始。

为了解决这个问题，脚本里的 `_normalise_messages(...)` 会：

1. 把第 0 条 `assistant` 消息在 tokenization 阶段临时改写成 `system`
2. 但在 span 统计阶段，仍然保留它原本的语义标签 `asst_greeting`

也就是说：

- 模型看到的是“system-like greeting”
- 输出统计里看到的是 `asst_greeting`

这也是为什么 `asst_greeting` 常被当作一种近似的 attention sink 来看待。

## 3. Turn 是怎么定义的

这个脚本把“assistant 消息的出现”视为 turn 边界。

### 3.1 分析点如何选择

对每个 simulation：

1. 找出所有 `role == "assistant"` 的消息索引
2. 把这些索引视为可分析的 prefill 快照点
3. 如果 `--max-turns > 0`，只分析最后 N 个 assistant turn
4. 如果 `--max-turns == 0`，则分析全部 assistant turn

所以，脚本不是只看“最终一轮”，而是可以在一段完整对话的多个时刻重复做 attention 快照。

### 3.2 `asst_turns_dist` 的定义

对任意一条历史消息，`asst_turns_dist` 表示：

- 这条消息与当前分析点之间，间隔了多少个 assistant 消息

因此：

- `0`：当前 assistant turn 内的消息
- `1`：上一轮 assistant turn 及其附近消息
- `2`：再上一轮

这个距离不是按“消息条数”算，而是按“assistant turn 数”算。

## 4. 测量流程

脚本对每个分析点都会重复以下流程：

1. 取 `context_msgs = messages[:turn_idx + 1]`
2. 用 `_safe_apply_template(...)` 把这段历史转成一条完整 token 序列
3. 用 `compute_message_spans(...)` 通过增量 tokenization 推出每条消息的 token span
4. 对整段 token 做一次完整前向：

```python
out = model(input_ids=input_ids, use_cache=False, output_attentions=True)
```

5. 从每一层 attention 里取“最后一个 query”的注意力行
6. 先对 head 求平均，再对有效 layer 求平均
7. 得到长度为 `seq_len` 的 attention 向量
8. 按消息 span 聚合为：
   - `attn_mean`
   - `attn_sum`
   - `attn_frac`

需要注意两点：

1. 这里是完整 prefill 分析，不走 KV cache decode。
2. 这里测的是“当前分析点最后一个 token 回望历史时关注哪里”，不是服务端运行时的完整在线裁剪过程。

## 5. span 是怎么求出来的

`compute_message_spans(...)` 的做法是：

1. 对 `messages[:1]` tokenization，得到第 0 条消息结束位置
2. 对 `messages[:2]` tokenization，得到第 1 条消息结束位置
3. 依次递增
4. 相邻两次长度之差，就是新增那条消息的 token span

如果 chat template 失败，脚本会退回到“只对该消息内容做 tokenization”的近似路径，因此极端异常数据也不会直接让整个分析崩掉。

## 6. 输出文件

默认输出目录是：

- `outputs/attention_task10`

当前会生成以下文件：

| 文件 | 内容 |
|---|---|
| `attention_by_role.csv` | 原始逐消息记录 |
| `attn_decay_by_role.png` | 左图 `attn_mean`，右图 `attn_frac` 的按距离衰减曲线 |
| `attn_heatmap.png` | `(asst_turns_dist, fine_role)` 的平均 `attn_frac` 热图 |
| `role_share_per_turn.png` | 不同 `context_turn` 上的角色 attention 占比堆叠图 |
| `token_count_vs_attn.png` | `token_count` 与 `attn_mean` 的散点图 |

### 6.1 CSV 字段

当前 `attention_by_role.csv` 的主要字段有 14 列：

| 列名 | 含义 |
|---|---|
| `sim_id` | simulation id 的前 12 位 |
| `context_turn` | 当前分析点对应的消息索引 |
| `total_msgs` | 该 simulation 的总消息数 |
| `msg_idx` | 被统计消息在对话中的索引 |
| `msg_dist` | 与当前分析点的消息索引距离 |
| `asst_turns_dist` | 与当前分析点的 assistant-turn 距离 |
| `role` | 原始 role |
| `fine_role` | 细粒度角色标签 |
| `attn_mean` | 该消息 span 内每个 token 的平均注意力 |
| `attn_sum` | 该消息 span 的注意力总和 |
| `token_count` | 该消息 span 的 token 数 |
| `seq_len` | 当前分析点的总 token 长度 |
| `attn_frac` | 该消息占总注意力的比例 |
| `src_file` | 这条记录来自哪个 simulation 文件 |

## 7. `attn_mean` 和 `attn_frac` 的区别

这两个字段经常被混用，但含义不同：

| 指标 | 含义 |
|---|---|
| `attn_mean` | 每个 token 平均受到多少注意力，更偏“密度” |
| `attn_frac` | 这条消息整体拿走了多少注意力，更偏“总预算占比” |

如果你关心“哪类消息值得更多 KV 预算”，通常更应该看 `attn_frac`。

如果你关心“同等长度下哪类 token 更容易被盯住”，通常更应该看 `attn_mean`。

## 8. 示例命令

### 单文件分析

```bash
python3 scripts/analyze/taubench/analyze_attention_by_role.py \
  --model-path ./local_models/Qwen3.5-9B \
  --sim-file ./tau2-bench/data/simulations/stress_baseline_1x10_task10_1st.json \
  --device cuda:2
```

### 多文件分析

```bash
python3 scripts/analyze/taubench/analyze_attention_by_role.py \
  --model-path ./local_models/Qwen3.5-9B \
  --sim-file \
    ./tau2-bench/data/simulations/stress_baseline_1x10_task10_1st.json \
    ./tau2-bench/data/simulations/stress_baseline_1x10_task10_2nd.json \
    ./tau2-bench/data/simulations/stress_baseline_1x10_task10_3rd.json \
    ./tau2-bench/data/simulations/stress_baseline_1x10_task10_4th.json \
  --device cuda:2 \
  --output-dir ./outputs/attention_task10
```

## 9. 读图时应注意什么

当前脚本测的是“prefill 快照下最后一个 query 的平均 attention”，因此它最适合回答：

- 当前轮生成前，模型在回望历史时更偏向哪类消息
- 不同角色的注意力占比如何随 assistant-turn 距离衰减

它不直接回答：

- 在线 H2O 在完整 decode 过程中每一步的真实裁剪轨迹
- 某个 token 是否最终一定会被保留在 cache 中

如果你要把这个脚本的结果拿去解释 H2O 行为，应该把它看成“role-level 先验观察”，而不是 H2O 运行时分数更新的逐步替代品。