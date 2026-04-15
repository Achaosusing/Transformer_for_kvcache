# baseline、streamingLLM、H2O、DTA-H2O 算法与实现说明

本文档以当前仓库实现为准，主要对应以下文件：

- `api_server.py`
- `src/api.py`
- `src/model.py`
- `src/chat_format.py`
- `src/methods/baseline.py`
- `src/methods/streaming_llm.py`
- `src/methods/h2o.py`
- `src/methods/dta_h2o.py`

当前仓库支持四种方法：

- `baseline`
- `streamingllm` / `streaming_llm`
- `h2o`
- `dta_h2o`

其中，`baseline`、`streamingLLM`、`h2o` 是核心对照组；`dta_h2o` 是在 H2O 基础上增加时间衰减、分层驱逐和反级联保护后的扩展版本。

## 1. 统一执行框架

### 1.1 离线评测主路径

通过 Python 直接调用 `OracleKVProjectAPI.evaluate(...)` 时，四种方法共享同一套高层流程：

1. `normalize_sample(...)` 统一样本格式。
2. `LocalTransformerModel.format_prompt_ids(...)` 将 `prompt` 或 `messages` 转为 token ids。
3. `OracleKVProjectAPI._build_policy(...)` 根据 `method` 和配置构造 policy。
4. 路由到具体生成路径。

当前实际路由为：

- `baseline` -> `OracleKVProjectAPI._generate_with_manual_cache(...)`
- `streamingllm` / `streaming_llm` -> `prune_streaming_prompt(...)` -> `OracleKVProjectAPI._generate_with_streaming_cache(...)`
- `h2o` / `dta_h2o` -> `OracleKVProjectAPI._generate_with_h2o(...)`

这些路径共用的底层模型操作包括：

- `prefill_next_token_logits(...)`
- `prefill_next_token_logits_with_attention(...)`
- `next_token_logits_from_cache(...)`
- `next_token_logits_from_cache_with_attention(...)`
- `prune_past_key_values(...)`
- `sample_next_token(...)`

### 1.2 OpenAI 风格 Chat 路径

`api_server.py` 的 `POST /v1/chat/completions` 有一条额外的 H2O 专用路径：

- 当请求是单方法 `h2o` 或 `dta_h2o` 时，会走 `_h2o_chat_response(...)`
- 这条路径支持：
  - canonical chat 格式化
  - per-token `role_ids`
  - DTA-H2O 的 per-token `turn_ids`
  - 可选的多轮 session 快照保存 / 恢复
  - 工具调用文本的解析与回填

也就是说：

- 离线评测路径重点关注“单轮 prompt -> generate”
- Chat H2O 路径额外处理“多轮 history 对齐、session restore、角色感知衰减”

### 1.3 `save_step_trace`

`save_step_trace=True` 时，只会额外记录每个 decode step 的：

- `full_context_tokens`
- `kept_tokens`
- `kept_ratio`

它不会改变任何实际裁剪逻辑。

## 2. baseline

### 2.1 算法定义

baseline 是 full-context 策略，不丢弃任何历史 token：

$$
\mathrm{Keep}(t) = \{0, 1, 2, \dots, t - 1\}
$$

### 2.2 当前实现

当前实现走 `OracleKVProjectAPI._generate_with_manual_cache(...)`：

1. 对完整 `token_ids` 做一次普通 prefill。
2. 得到下一 token 的 logits 和 `past_key_values`。
3. 进入逐步 decode。
4. 每一步都只在已有 cache 上追加，不做裁剪。

`src/methods/baseline.py` 中的 `BaselineFullAttentionPolicy` 只是统一接口下的占位 policy；真正的 baseline 生成路径不依赖任何 eviction 选择。

### 2.3 特点

优点：

1. 与标准自回归解码最接近。
2. 没有近似裁剪误差。
3. 是所有缓存裁剪方法的直接对照基线。

代价：

1. KV cache 会随上下文长度持续增长。
2. 长上下文下显存和计算开销最高。

## 3. streamingLLM

### 3.1 算法定义

streamingLLM 只保留两类 token：

1. `sink tokens`：序列最前面的若干 token
2. `recent tokens`：序列末尾最近的若干 token

缓存结构为：

$$
\mathrm{KV\ Cache} = \mathrm{Sink} + \mathrm{Recent}
$$

预算为：

$$
\mathrm{Budget} = \mathrm{sink\_size} + \mathrm{local\_window\_size}
$$

### 3.2 Prompt 阶段

当前实现会先调用 `prune_streaming_prompt(full_ids, policy)`，在 prefill 之前对 prompt 做静态裁剪：

- 保留前 `sink_size` 个 token
- 保留末尾 `local_window_size` 个 token

如果 `len(full_ids) <= cache_budget`，则不裁剪。

这意味着 streamingLLM 的 prompt 阶段与 H2O 不同：

- streamingLLM：先裁 prompt，再 prefill
- H2O / DTA-H2O：先对完整 prompt prefill，再决定是否初始裁剪

### 3.3 Decode 阶段

当前 decode 逻辑位于 `OracleKVProjectAPI._generate_with_streaming_cache(...)`。

设：

- 当前活动 cache 长度为 `active_token_count`
- 下一个 token 加入后的长度为 `next_total = active_token_count + 1`
- 预算为 `cache_budget`

当满足：

$$
\mathrm{next\_total} > \mathrm{cache\_budget} + \mathrm{evict\_period} - 1
$$

时，触发裁剪。代码会直接丢弃最老的非 sink token：

- `sink` 区间固定保留
- 非 sink 部分从左侧按需删除 `excess = next_total - cache_budget` 个位置

等价地说，streamingLLM 的 decode eviction 不是“按分数选 token”，而是“固定 sink + 向右滑动窗口”。

### 3.4 `evict_period` 的含义

- `evict_period = 1`：每一步都严格保持在预算内
- `evict_period > 1`：允许 cache 临时超预算最多 `evict_period - 1` 个 token，以减少裁剪频率

### 3.5 特点

优点：

1. 实现简单，行为稳定。
2. cache 预算容易解释。
3. 很适合作为“只保留 sink + recent”的对照组。

限制：

1. 中间历史 token 会整体丢弃。
2. 无法保留那些不在 recent window 内、但长期重要的 token。

## 4. H2O

### 4.1 算法定义

H2O 在 streamingLLM 的基础上增加了 `heavy hitters`：

1. `sink tokens` 必保留
2. `recent tokens` 必保留
3. 中间区域按累计分数选出 `heavy_hitter_size` 个高分 token

缓存结构为：

$$
\mathrm{KV\ Cache} = \mathrm{Sink} + \mathrm{Heavy\ Hitters} + \mathrm{Recent}
$$

预算为：

$$
\mathrm{Budget} = \mathrm{sink\_size} + \mathrm{local\_window\_size} + \mathrm{heavy\_hitter\_size}
$$

### 4.2 分数定义

H2O 为当前活动 cache 中的每个位置维护一个累计分数 $S_i$。

每次采集 attention 时，代码只取“最后一个 query 对全部 key 的注意力分布”，并做如下聚合：

1. 取每一层最后一个 query 对所有 key 的注意力行
2. 先在 head 维上做平均
3. 再在有效 layer 维上做平均

记单次采集得到的向量为 $s_t$，则 vanilla H2O 的更新规则为：

$$
S_i \leftarrow S_i + s_t(i)
$$

这里的 $s_t(i)$ 不是 pre-softmax 的原始 qk，相当于：

$$
s_t(i) =
\frac{1}{|L_t|}
\sum_{l \in L_t}
\frac{1}{H_l}
\sum_{h=1}^{H_l}
\operatorname{softmax}
\left(
\frac{q_{l,h,t}K_{l,h,1:n}^{\top}}{\sqrt{d}} + m
\right)_i
$$

因此，当前实现里的 H2O 分数有四个重要性质：

1. 它是后 softmax 概率，不是原始 qk logits。
2. 它只看“最后一个 query”的回看分布，不是整张 attention matrix 的汇总。
3. 它是跨 head、跨 layer 平均后的结果。
4. 它只对“当前仍在活动 cache 中的 token”持续累积。

### 4.3 Prefill 阶段

H2O 的初始化由 `OracleKVProjectAPI.initialize_h2o_state(...)` 完成。

核心步骤如下：

1. 对完整 prompt 做 prefill。
2. 同时采集 prefill 阶段最后一个 query 的 attention 分数。
3. 初始化 `score_counters`。
4. 如果 prompt 已超出预算，立即做一次初始裁剪。

当前代码里，prefill attention 的采集方式取决于 `attn_implementation`：

- 当 `attn_implementation == "sdpa"` 时：
  - 使用 `SDPAAttentionCapture`
  - 它会包裹 `torch.nn.functional.scaled_dot_product_attention`
  - 额外提取最后一个 query 的注意力分布
  - 但实际前向仍走原始 SDPA kernel
- 其他情况下：
  - 调用 `prefill_next_token_logits_with_attention(...)`
  - 通过 `output_attentions=True` 取回注意力

这意味着一个很重要的结论：

- 当前 H2O 的首次裁剪，已经是“prefill attention 驱动”的
- 它不是旧版本里那种“全零分数初始化后再靠 recency tie-break” 的近似行为

### 4.4 初始裁剪

若 `active_token_count > policy.cache_budget`，当前实现会直接依据 prefill 得到的 `score_counters` 做第一次保留集合选择：

- 普通 H2O：`policy.select_keep_tensor(...)`
- DTA-H2O：`policy.select_keep_tensor_tiered(...)`

随后同步裁剪：

- `score_counters`
- `past_key_values`
- `role_tags`（若存在）
- `turn_ids`（若存在）

### 4.5 Decode 阶段

H2O 的 decode 核心在 `OracleKVProjectAPI._advance_h2o_state_with_token(...)` 中。

对每一个将要追加的 token，执行顺序是：

1. 先判断“加入新 token 后”是否会超过 `budget + evict_period - 1`
2. 若需要裁剪：
   - 先构造 `extended_scores = old_scores + [0]`
   - 也就是把“新 token 的初始零分”一并纳入选择逻辑
   - 根据 `next_total_tokens = active_token_count + 1` 计算 keep 集合
   - 但真正裁剪时，只会裁旧 cache；新 token 此时尚未写入 KV cache
3. 将新 token 追加到 state 中：
   - `score_counters` 追加一个 0
   - `role_tags` 追加 `ROLE_GENERATED`（若存在）
   - `turn_ids` 追加 `current_turn_id`（若存在）
4. 决定本步是否需要采集 attention
5. 采集时再更新 `score_counters`

### 4.6 Attention 采集频率

当前实现不是每一步都读取 attention。

只有满足以下任一条件时，才会调用 `next_token_logits_from_cache_with_attention(...)`：

1. 本步发生 eviction
2. `steps_since_collect >= collect_period`

否则只调用 `next_token_logits_from_cache(...)`，不返回 attention。

因此：

- `collect_period = 1` 最接近“每步在线更新分数”
- 更大的 `collect_period` 会减少 attention 调用次数，但分数更新更稀疏

### 4.7 Heavy Hitter 选择逻辑

`src/methods/h2o.py` 中的 `H2OPolicy.select_keep_tensor(...)` 会：

1. 永久保留 `sink`
2. 永久保留 `recent`
3. 对中间候选区按 `score_counters` 取 top-k

当分数相同时，代码会施加一个极小的“越新越优先”偏置：

$$
\mathrm{adjusted\_score} = \mathrm{score} + \varepsilon \cdot \mathrm{recent\_bias}
$$

因此在完全同分时，H2O 会偏向保留更靠后的 token。

### 4.8 作用域与边界

当前 H2O 的分数和选择只针对“仍在活动 cache 中的 token”：

1. token 被驱逐后，其分数不会被单独保留
2. 当前 vanilla H2O 没有被驱逐 token 的召回机制
3. 当前 vanilla H2O 也不会对已驱逐 token 做重算或回填

## 5. DTA-H2O

### 5.1 目标

DTA-H2O 是在 H2O 上增加三类能力的扩展版本：

1. 时间衰减
2. 分层驱逐
3. 反级联保护

实现位于 `src/methods/dta_h2o.py` 与 `src/api.py`。

### 5.2 时间衰减分数更新

DTA-H2O 使用的累计方式为：

$$
S_i \leftarrow \gamma \cdot S_i + s_t(i)
$$

其中：

- $\gamma = \texttt{dta\_gamma}$
- 当 `gamma = 1.0` 时，退化为 vanilla H2O

代码对应 `OracleKVProjectAPI._accumulate_dta_h2o_scores(...)`。

### 5.3 额外状态

相比 vanilla H2O，DTA-H2O 还会维护：

- `turn_ids`：每个 cache 位置属于哪一轮对话
- `current_turn_id`：当前轮次编号
- `ghost_buffer`：最近被驱逐 token 的元数据环形缓冲区

对应运行时结构体：

```python
H2ORuntimeState(
    past_key_values,
    score_counters,
    active_token_count,
    steps_since_collect,
    role_tags,
    turn_ids,
    current_turn_id,
    ghost_buffer,
)
```

### 5.4 分层驱逐

`DTAH2OPolicy.select_keep_tensor_tiered(...)` 的核心逻辑如下。

先构造保护集合：

1. `sink`
2. `recent`
3. 若 `system_anchor=True`，则在固定预算内优先保留 `ROLE_SYSTEM`

剩余候选 token 再按轮次分为：

1. `current turn`
2. `history turns`

设剩余预算为 `remaining_budget`，则：

$$
\mathrm{current\_budget} = \lfloor \mathrm{remaining\_budget} \cdot \mathrm{current\_turn\_ratio} \rfloor
$$

$$
\mathrm{history\_budget} = \mathrm{remaining\_budget} - \mathrm{current\_budget}
$$

若某一层候选不足，会把溢出的预算重新分配给另一层。随后在各自层内再按分数做 top-k，仍然沿用“同分时更偏向新 token”的 tie-break。

### 5.5 `system_anchor`

若 `system_anchor=True`，则角色为 `ROLE_SYSTEM` 的 token 会优先占用 heavy-hitter 预算；若系统 token 数量超过剩余预算，则按分数与最近性选择其中一部分保留，以维持 cache 上界。

这使 DTA-H2O 能对系统提示和工具定义保持更稳定的跨轮保留。

### 5.6 Ghost Buffer 与反级联保护

`GhostBuffer` 是一个固定容量的环形缓冲区，记录最近被驱逐 token 的：

- `score`
- `role`
- `turn_id`

在下一次 eviction 决策前，如果某个历史轮次在 ghost buffer 中的驱逐数量已经达到阈值（当前实现为 `max(1, capacity // 4)`），`get_anti_cascade_boost(...)` 会给该轮仍存活的 token 施加一个额外 boost：

$$
\mathrm{boost} = \mathrm{max\_score} \times 0.1 \times \mathrm{factor}
$$

它的目的不是永久改变分数定义，而是在“某一历史轮次正在被连续清空”时，暂时抬高该轮剩余 token 的生存概率，避免级联式驱逐。

### 5.7 何时退化为普通 H2O

如果同时满足：

- `current_turn_ratio = 1.0`
- 所有 token 都属于同一 `turn`
- `system_anchor = False`
- `gamma = 1.0`

则 DTA-H2O 的行为会非常接近普通 H2O。

## 6. 角色标签与轮次标签

### 6.1 `role_tags`

H2O / DTA-H2O 的 chat 路径会为每个 token 构造角色标签，常量定义在 `src/chat_format.py`：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `ROLE_SYSTEM` | 0 | system 消息 |
| `ROLE_USER` | 1 | user 消息 |
| `ROLE_ASSISTANT` | 2 | assistant 历史消息 |
| `ROLE_TOOL` | 3 | tool 返回消息 |
| `ROLE_GENERATED` | 4 | 当前轮新生成 token |

`role_tags` 会在以下操作中同步维护：

- 初始化
- eviction
- 追加新 token
- `trim_h2o_state_tail(...)`
- `clone_h2o_state(...)`

### 6.2 `turn_ids`

只有 DTA-H2O 会额外使用 `turn_ids`。

`build_token_role_and_turn_ids(...)` 的轮次规则是：

- system 为 `turn 0`
- 每遇到一条 user 消息，`turn_counter += 1`
- assistant 和 tool 消息继承前一条 user 的 turn
- `add_generation_prompt=True` 时，预留的 assistant generation prompt 也继承当前 turn

这样就能把“当前轮”和“历史轮”在 token 级别上对齐到真实 prompt。

## 7. 多轮 Session 复用

### 7.1 生效条件

多轮 session 复用只在以下条件同时满足时生效：

1. 路径是 `POST /v1/chat/completions`
2. 请求只包含单方法 `h2o` 或 `dta_h2o`
3. 服务端显式启用了 `--enable-session`

否则每个请求都按单轮执行：

- 重新 tokenize
- 重新 prefill
- 重新初始化 H2O state

### 7.2 为什么要用 canonical chat

session 复用依赖“下一轮 prompt 的 token 前缀，等于上一轮完整 history 的 token 序列”。

为保证这一点，H2O chat 路径会使用：

- `format_canonical_chat(...)`
- `build_token_role_ids(...)`
- `build_token_role_and_turn_ids(...)`

这套逻辑会：

1. 把消息、工具定义、assistant thinking 区块、tool response 统一成稳定文本格式
2. 再用“增长前缀 tokenize”的方式给每一段片段分配 role / turn
3. 尽量保证多轮 history 在 token 级别前缀稳定

### 7.3 Snapshot 查找

当前 session 存储由 `LRUSessionStore` 管理，支持两种查找方式：

1. 显式 `session_id` 精确查找
2. 自动前缀匹配 `find_by_prefix(prompt_ids, signature)`

自动前缀匹配的实际策略为：

1. 先对 prompt 的前 64 个 token 求 hash，缩小候选范围
2. 再做完整前缀校验
3. 选择“最长可匹配 history 前缀”的 snapshot

### 7.4 Snapshot 命中条件

即使找到了候选 snapshot，也必须同时满足以下条件才会真正复用：

1. `signature` 一致
2. `tools` 一致
3. 当前消息数大于 snapshot 中保存的消息数
4. 当前消息前缀与 snapshot 的消息完全一致
5. 当前 `prompt_ids` 的前缀与 snapshot 的 `history_token_ids` 一致

其中 `signature` 当前包含：

- `sink_size`
- `local_window_size`
- `heavy_hitter_size`
- `evict_period`
- `collect_period`
- `session_score_alpha`
- `dta_gamma`
- `current_turn_ratio`

### 7.5 恢复阶段

命中 snapshot 后，会执行：

1. `restore_h2o_state(...)`
2. `continue_h2o_state(appended_ids, ...)`

其中 `restore_h2o_state(...)` 会：

1. 把 snapshot 的 state clone 到当前设备
2. 对旧分数执行跨轮衰减
3. 重置 `steps_since_collect = 0`
4. `current_turn_id += 1`
5. 清空 `ghost_buffer`

### 7.6 跨轮分数衰减

#### 7.6.1 统一衰减

若没有 `role_tags` 或没提供角色 alpha，则回退到旧版统一衰减：

$$
\mathrm{decayed} =
\frac{S}{\max(S)} \cdot \alpha
$$

对应 `apply_max_normalized_h2o_decay(...)`。

#### 7.6.2 角色感知衰减

若有 `role_tags`，则按角色分别做 max-normalize：

$$
S_i^{\mathrm{decayed}} =
\frac{S_i}{\max_{j:\mathrm{role}(j)=r} S_j} \cdot \alpha_r
\quad \text{其中 } r = \mathrm{role}(i)
$$

当前默认值为：

| 角色 | 默认 alpha |
| --- | --- |
| system | 0.9 |
| tool | 0.7 |
| user | 0.3 |
| assistant | 0.3 |

这表示：

- system token 跨轮几乎不衰减
- tool token 适度保留
- user / assistant 历史内容更快让位给新一轮输入

### 7.7 Snapshot 保存

生成完成后，服务端会把 assistant 输出重新整理成 history，再尝试保存 snapshot。

这里有一个实现细节很重要：

- 模型“实际生成的 token 序列”
- canonical history 文本中“最终应该写入的 token 序列”

二者不一定完全一一对应，例如：

- 生成结果被解析出 `tool_calls`
- EOS 被剔除
- canonical assistant 历史里有额外闭合片段

因此保存 snapshot 前，代码会做一次对齐：

1. 若 history 末尾比原始生成多出 closure token，则调用 `continue_h2o_state(...)` 把 closure 补入 state
2. 若原始生成比 history 末尾更长，则调用 `trim_h2o_state_tail(...)` 截掉多余尾部

只有成功对齐后，才会把 snapshot 以 CPU 版本写入 LRU。

## 8. 注意力实现与采集细节

### 8.1 `--attn-implementation auto`

当前 `api_server.py` 中，`--attn-implementation auto` 会被实际解析成：

```text
sdpa
```

也就是说，当前服务端默认并不会对 H2O 自动切回 eager。

H2O 之所以仍然能采到分数，是因为：

- prefill 阶段可通过 `SDPAAttentionCapture` 在 SDPA 上补采最后一行 attention
- decode 阶段在需要采集的步上也会优先走同样的 SDPA capture 路径

### 8.2 为什么默认更偏向 SDPA

原因是：

1. 长 prompt prefill 时，不希望物化完整 attention tensor
2. H2O 真正需要的只是“最后一个 query 的回看分布”
3. 用 SDPA wrapper 可以在保留高效 kernel 的同时，只额外取出这部分分数

## 9. 常见边界与易错点

### 9.1 四种方法的 prompt 处理并不相同

- `baseline`：完整 prompt 直接 prefill
- `streamingLLM`：先静态裁 prompt，再 prefill
- `h2o` / `dta_h2o`：完整 prompt 先 prefill 并采分，再按预算做初始裁剪

### 9.2 `collect_period` 的默认值有两套来源

- 直接调用 `OracleKVProjectAPI.evaluate(...)` 时，默认 `collect_period=1`
- 通过 `api_server.py` 启动服务时，CLI 默认 `--collect-period 0`
- 但服务端会把 `0` 解释为“跟随 `evict_period`”

因此文档或实验脚本在比较结果时，必须区分：

- Python API 默认值
- HTTP 服务默认值

### 9.3 当前没有通用的“被驱逐 token 召回”

- vanilla H2O 不会重新注入已驱逐 token
- DTA-H2O 的 ghost buffer 只会影响“下一次驱逐的打分偏置”
- 它不是完整的 token 回收池

### 9.4 `session_id` 目前不能和 `max_input_tokens` 同时使用

`api_server.py` 当前会直接拒绝：

```text
session_id + max_input_tokens
```

原因是截断后的 prompt 与 snapshot 历史对齐规则尚未单独实现。

## 10. 方法对比总结

| 方法 | Prompt 阶段 | Decode 阶段 | 预算组成 | 是否依赖 attention 分数 | 是否支持多轮 session 复用 |
| --- | --- | --- | --- | --- | --- |
| baseline | 完整 prompt prefill | 不裁剪 | 无固定预算 | 否 | 否 |
| streamingLLM | 先裁 prompt，再 prefill | 固定 sink + 滑动 recent window | `sink + window` | 否 | 否 |
| h2o | 完整 prompt prefill 并采分，必要时初始裁剪 | sink + recent 保护，中间区按累计分数选 HH | `sink + window + hh` | 是 | 有条件支持 |
| dta_h2o | 与 H2O 相同 | 在 H2O 基础上加入时间衰减、分层驱逐、ghost buffer | `sink + window + hh` | 是 | 有条件支持 |

如果只关心“最忠实于原模型的结果”，应看 `baseline`。

如果只关心“最简单、最稳定的固定预算裁剪”，应看 `streamingLLM`。

如果关心“在固定预算下尽量保留长期重要 token”，应看 `H2O`。

如果关心“多轮对话、角色差异、历史轮次保护”，应看 `DTA-H2O`。

这里的“有条件支持”特指：

- 仅限 `api_server.py` 的单方法 chat 路径
- 且服务端开启了 `--enable-session`
