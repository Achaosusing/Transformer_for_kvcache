# Attention-by-Role 分析说明

本文档说明 `scripts/analyze_attention_by_role.py` 的完整工作原理，包括：
- 四种 `fine_role` 的定义与识别方法
- 多轮对话中「轮次」的划分依据
- 衰减表（Attention Fraction Decay Table）的测量过程

---

## 1. 四种 `fine_role` 的定义与识别

### 数据来源

`fine_role` 是对 tau2-bench simulation JSON 中原始 `role` 字段的**细粒度重新分类**。

原始 JSON 里每条消息的格式：

```json
{
  "role": "assistant" | "user" | "system" | "tool",
  "content": "...(消息正文)..."
}
```

分类逻辑在 `classify_role()` 函数（[scripts/analyze_attention_by_role.py](../scripts/analyze_attention_by_role.py#L75)）以及 `compute_message_spans()` 中完成：

```
原始 role       +  附加条件                        →  fine_role
─────────────────────────────────────────────────────────────
assistant      +  消息索引 == 0（对话首条）         →  asst_greeting
assistant      +  content 含 ```json 或工具调用标记 →  asst_tool_call
assistant      +  其他正常回复                      →  asst_gen
user           +  任何 content                      →  user_msg
system         +  任何 content                      →  system
tool           +  任何 content                      →  tool_result
```

工具调用的判断使用正则（[第 71 行](../scripts/analyze_attention_by_role.py#L71)）：

```python
TOOL_CALL_RE = re.compile(r"\[call_tool\]|\[function_call\]|<tool_call>", re.IGNORECASE)
```

---

### 1.1 `asst_greeting`

**定义**：对话的**第 0 条消息**，且 `role == "assistant"`。

这是 tau2-bench 的固定格式——agent 先向用户打招呼，还没有获取任何上下文。  
在 token 层面它只有 5~15 个 token，功能上接近 system prompt，是典型的 **attention sink**。

**在真实数据中的位置**（文件 `tau2-bench/data/simulations/stress_baseline_1x30_1st.json`，sim_id=b8ff8f92）：

```
[00] role=assistant  fine_role=[asst_greeting]
     content: Hi! How can I help you today?
```

---

### 1.2 `user_msg`

**定义**：任何 `role == "user"` 的消息。

用户发来的自然语言请求、补充信息、追问。

```
[01] role=user  fine_role=[user_msg]
     content: Hi, I have a couple of things I need help with. First, I'd like to
              know the total balance of my gift cards...

[03] role=user  fine_role=[user_msg]
     content: Sure, my user ID is mohamed_silva_9265.

[09] role=user  fine_role=[user_msg]
     content: Thanks for the balance details. I don't have my reservation ID handy
              —could you help me find it?
```

**注意**：user_msg 的 per-token attention 密度是最低的（0.16），但衰减极慢（半衰期 6.3 轮），意味着用户的原始意图在整个对话中始终保持一定的参考价值。

---

### 1.3 `asst_tool_call`

**定义**：`role == "assistant"` 且 content 中含有工具调用标记（JSON 代码块或 `[call_tool]` 等）。

在 tau2-bench 中，agent 调用工具通过在 content 中嵌入 ` ```json {...} ``` ` 代码块实现。

```
[04] role=assistant  fine_role=[asst_tool_call]
     content: Thank you for providing your user ID. Let me first check your profile
              to see the balances of your gift cards and travel certificates.
              ```json
              {"get_user": {"user_id": "mohamed_silva_9265"}}
              ```

[10] role=assistant  fine_role=[asst_tool_call]
     content: I can help you locate your reservation ID. Let me retrieve your
              reservations.
              ```json
              {"get_reservations": {"user_id": "mohamed_silva_9265"}}
              ```

[12] role=assistant  fine_role=[asst_tool_call]
     content: ```json
              {"get_reservations": {"user_id": "mohamed_silva_9265"}}
              ```
```

消息 [12] 是纯工具调用（没有自然语言前缀），消息 [04] 和 [10] 是带解释的工具调用。两者都被归为 `asst_tool_call`。

**关键特征**：per-token 密度高（1.45），但衰减极快（半衰期仅 0.8 轮）。距当前轮 2 轮以上，其 attention 接近 0。

---

### 1.4 `asst_gen`

**定义**：`role == "assistant"` 且 content 中**不含**工具调用标记，是正常的文字回复。

```
[02] role=assistant  fine_role=[asst_gen]
     content: To assist you with your requests, I first need to obtain your user
              ID. Could you please provide your user ID?

[08] role=assistant  fine_role=[asst_gen]
     content: Based on the profile information retrieved, here are the details...
              - Gift Cards: $100 and $50, total $150
              - Travel Certificates: $200

[24] role=assistant  fine_role=[asst_gen]
     content: Based on the reservation details for RES789012:
              - Current Cabin Class: Economy
              - Trip Type: Round Trip
```

**关键特征**：占 token 总量最多（约 60%），per-token attention 中等（密度 0.29），衰减速度中等（半衰期 1.3 轮）。

---

## 2. 轮次（Turn）的划分方式

### 2.1 什么是一「轮」

本脚本以**assistant 消息的出现**作为轮次边界。

```
消息序列示意：
  [0] assistant ← 轮次 0 (asst_greeting, dist=∞ from current)
  [1] user       ← 轮次 0 的 user 部分
  [2] assistant ← 轮次 1
  [3] user
  [4] assistant ← 轮次 2         ← 若我们在这里分析
  [5] user
  [6] assistant ← 轮次 3 (current，分析点)
```

**当我们站在轮次 3（消息 [6]）做 attention 分析时：**

- `asst_turns_dist = 0`：消息 [6]（当前轮 assistant）
- `asst_turns_dist = 1`：消息 [4]（上一轮 assistant）及其前的 [5] user
- `asst_turns_dist = 2`：消息 [2]（两轮前 assistant）及其前的 [3] user

具体计算逻辑在 `analyse_simulation()` 中（[第 284 行](../scripts/analyze_attention_by_role.py#L284)）：

```python
asst_turns_before = sum(
    1 for j in range(msg_i, turn_idx)
    if messages[j].get("role") == "assistant"
)
```

即：消息 `msg_i` 与当前分析点 `turn_idx` 之间，历经了几个 assistant 消息，就是这条消息的 `asst_turns_dist`。

### 2.2 「新一轮对话的开始」的标志

在 tau2-bench 的 airline 场景中，**每一个 user 消息紧跟 assistant 消息之后**就代表进入新一轮交互。具体来说：

- **user 消息出现** → assistant 需要做出回应 → 新一轮开始
- **没有独立的特殊符号标记**轮次边界，轮次由 `role` 字段的交替切换隐式定义

对话结束（任务终止）的特殊标记是：
```
user content: "###TRANSFER###"   （请求转接人工）
```
或者 tau2-bench 内部的 `termination_reason` 字段（`"max_turns"` / `"agent"` / `"user"`）。

---

## 3. 衰减表的测量原理（如何得到 Attention Fraction Decay Table）

### 3.1 整体流程图

```
tau2-bench simulation JSON (多轮对话历史)
          │
          ▼
  选取若干 assistant 轮次作为"分析点"
  （每个分析点 = 一个 prefill 快照）
          │
          ▼
  把该分析点之前的所有消息 tokenize 成一段完整的 token 序列
  [system/greeting tokens | user1 tokens | asst1 tokens | user2 tokens | ... | current-asst tokens]
          │
          ▼
  将整段 token 序列送入模型做一次完整的前向传播（prefill），
  同时开启 output_attentions=True
          │
          ▼
  提取「最后一个 query（=当前轮最后一个 token）」对所有历史 token 的注意力权重
  attn[seq_len]   ← 这就是模型「此刻最关注历史哪些位置」
          │
          ▼
  把 attn 向量按消息的 token span 分组
  → 每条消息得到一个 attn_sum 和 attn_mean
  → 除以总和得到 attn_frac（归一化）
          │
          ▼
  记录 (fine_role, asst_turns_dist, attn_frac) 三元组
          │
          ▼
  对多个分析点、多个 simulation 聚合平均
  → 按 (asst_turns_dist, fine_role) 分组 → 衰减表
```

### 3.2 关键代码逐步解释

#### 第一步：选择分析点

```python
# analyse_simulation() 中：
assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
turns_to_analyse = assistant_indices[-max_turns:]   # 取最后几个 assistant 轮
```

每个 assistant 消息的位置就是一个"快照点"。取最靠后的是因为越靠后历史越丰富，衰减趋势越完整。

#### 第二步：构造完整 token 序列

```python
context_msgs = messages[:turn_idx + 1]   # 截取到当前轮（含）
token_ids = _safe_apply_template(tokenizer, context_msgs)
# → 一维 list[int]，长度就是 seq_len
```

用 Qwen3.5 的 chat template 把消息格式化成模型能理解的 token 序列：

```
<|im_start|>system\nHi! How can I help...<|im_end|>
<|im_start|>user\nHi, I want...<|im_end|>
<|im_start|>assistant\nTo assist...<|im_end|>
...（更多轮次）...
<|im_start|>assistant\n当前轮内容<|im_end|>
```

#### 第三步：计算 token span

```python
spans = compute_message_spans(tokenizer, context_msgs)
# → [(start, end, role, fine_role), ...]
```

原理：对 `messages[0..k]` 逐步 tokenize，相邻两次长度之差就是第 k 条消息的 token 区间：

```
tokenize(msgs[0..0]) → len=12     → span[0] = [0, 12)
tokenize(msgs[0..1]) → len=35     → span[1] = [12, 35)
tokenize(msgs[0..2]) → len=58     → span[2] = [35, 58)
...
```

#### 第四步：提取注意力权重

```python
# extract_last_query_attention() 中：
out = model(input_ids=input_ids, use_cache=False, output_attentions=True)
# out.attentions: tuple，每个元素对应一个 Transformer 层
# 形状：[batch=1, heads, seq, seq]

for layer_attn in out.attentions:
    row = layer_attn[0, :, -1, :]   # 取最后一个 query（当前轮末尾 token）的注意力行
    acc += row.mean(axis=0)          # 对所有 head 求均值
```

`attn[j]` 表示：当模型生成当前轮的下一个 token 时，对历史第 `j` 个 token 的关注程度。  

由于 Qwen3.5-0.8B 是混合架构（24 层中只有 6 层是全注意力层，其余 18 层是线性注意力/Mamba），只有这 6 层返回非 None 的 attention tensor，代码会自动跳过 None 层。

#### 第五步：按 span 聚合

```python
for msg_i, (start, end, role, fine_role) in enumerate(spans):
    span_attn = attn[start:end]           # 这条消息的所有 token 的注意力
    attn_sum = span_attn.sum()
    attn_frac = attn_sum / attn.sum()     # 这条消息占总注意力的比例
    asst_turns_dist = ...                 # 距当前轮的 assistant 轮数
    records.append({fine_role, asst_turns_dist, attn_frac, ...})
```

#### 第六步：统计聚合出衰减表

```python
# print_summary() 中：
by_dist = (
    df[df["asst_turns_dist"] <= 6]
    .groupby(["asst_turns_dist", "fine_role"])["attn_frac"]
    .mean()
    .unstack()
)
```

对 6 个 simulation × 多个分析点 × 多条消息的所有记录，按 `(asst_turns_dist, fine_role)` 分组求平均，得到最终的二维衰减表。

### 3.3 为什么用「最后一个 token 的 attention」

这模拟的是**模型做下一步 decode 时的决策状态**——即在生成下一个 token 之前，模型「回望」历史时对每个位置的关注程度。这正是 H2O 方法中 `score_counters` 的实际含义（每一步 decode 时最后一个 query token 的 attention）。

### 3.4 `attn_frac` vs `attn_mean` 的区别

| 指标 | 公式 | 含义 |
|------|------|------|
| `attn_mean` | `span_attn.mean()` | 该消息**每个 token** 平均得到多少注意力（与长度无关）|
| `attn_frac` | `span_attn.sum() / total` | 该消息**整体**占全局注意力的比例（与长度正相关）|

衰减表使用 `attn_frac`，因为它反映的是「这条消息在 KV cache 中值多少预算」，而非单个 token 的密度。

---

## 4. 输出文件说明

运行后在 `outputs/attention_analysis/` 生成：

| 文件 | 内容 |
|------|------|
| `attention_by_role.csv` | 每条消息×每个分析点的原始记录，13 列 |
| `attn_decay_by_role.png` | 折线图：attention 随轮次距离的衰减曲线（左：per-token mean；右：fraction） |
| `attn_heatmap.png` | 热图：(asst_turns_dist, fine_role) × attn_frac |
| `role_share_per_turn.png` | 堆叠柱状图：各分析点的 role attention 占比 |
| `token_count_vs_attn.png` | 散点图：消息长度 vs per-token attention |

CSV 主要列含义：

| 列名 | 含义 |
|------|------|
| `sim_id` | simulation 的前 12 位 id |
| `context_turn` | 分析点对应的消息索引 |
| `msg_idx` | 该消息在对话中的索引 |
| `asst_turns_dist` | 该消息距当前分析点相隔的 assistant 轮次数（0=当前轮） |
| `fine_role` | 细粒度角色标签 |
| `attn_frac` | 该消息的注意力占总注意力的比例 |
| `attn_mean` | 该消息每个 token 的平均注意力分数 |
| `token_count` | 该消息的 token 数 |
| `seq_len` | 分析点的总 token 数 |
