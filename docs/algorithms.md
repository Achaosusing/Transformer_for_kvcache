# baseline、streamingLLM、h2o 算法说明

本文说明项目中三种 KV Cache 策略的算法定义、统一执行框架和关键实现细节。

对应代码位置：

- `src/methods/baseline.py`
- `src/methods/streaming_llm.py`
- `src/methods/h2o.py`
- `src/api.py`
- `src/model.py`

## 1. 统一执行框架

统一流程在 `src/api.py` 中实现：

1. 先把输入文本转换成 `full_ids`。
2. 根据方法决定初始保留哪些 token。
3. 对保留后的 token 做一次 prefill，得到：
   - 下一 token 的 logits
   - `past_key_values`
4. 进入逐步解码循环。
5. 每一步：
   - 从 logits 采样下一个 token
   - 把该 token 追加到输出序列
   - 使用 `past_key_values` 做一次增量前向
   - 更新 logits 和 cache

在代码里，这个统一框架分成两条路径：

- baseline 和 streamingLLM：`OracleKVProjectAPI._generate_with_manual_cache`
- h2o：`OracleKVProjectAPI._generate_with_h2o`

两条路径的共同点是：

1. 都使用 `prefill_next_token_logits(...)` 或它的 attention 版来初始化 cache。
2. 都使用 `next_token_logits_from_cache(...)` 或它的 attention 版进行增量解码。
3. 都调用同一个 `sample_next_token(...)` 做采样。

因此三者在模型、采样逻辑、增量缓存使用方式上保持同一大框架，主要差异集中在“哪些 token 被保留在 KV Cache 中”。

## 2. baseline

### 2.1 算法定义

baseline 是 full-context 策略，不丢弃任何 token。

它的 policy 很简单：

$$
\text{Keep}(t) = \{0, 1, 2, \dots, t-1\}
$$

也就是说，在长度为 $t$ 的上下文里，全部 token 都保留。

### 2.2 实现位置

在 `src/methods/baseline.py` 中：

- `BaselineFullAttentionPolicy.select_keep_indices(...)`

其行为是直接返回 `range(total_tokens)`。

### 2.3 运行方式

在 `src/api.py` 中，baseline 的执行流程是：

1. `pruned_ids = full_ids`
2. 用 `_generate_with_manual_cache(...)` 进入统一逐步解码

因此 baseline 的特征是：

1. 初始 prompt 完整进入 KV Cache。
2. 后续生成出的 token 也持续追加到 cache 中。
3. 没有任何裁剪或淘汰逻辑。

### 2.4 特点

优点：

1. 最接近标准自回归解码。
2. 语义信息最完整。
3. 作为对比基线最直接。

缺点：

1. KV Cache 会随上下文长度持续增长。
2. 长上下文下显存和计算开销最大。

## 3. streamingLLM

### 3.1 算法定义

streamingLLM 的基本思想是只保留两类 token：

1. Sink Tokens：序列最前面的若干 token
2. Recent Tokens：序列末尾最近的若干 token

因此它的缓存结构是：

$$
\text{KV Cache} = \text{Sink Tokens} + \text{Recent Tokens}
$$

其中：

- `sink_size` 控制前缀保留数量
- `local_window_size` 控制最近窗口大小

### 3.2 实现位置

在 `src/methods/streaming_llm.py` 中：

- `StreamingLLMPolicy.select_keep_indices(...)`

具体做法是：

1. 计算 `sink_end = min(sink_size, total_tokens)`
2. 计算 `tail_start = max(sink_end, total_tokens - local_window_size)`
3. 保留：
   - `[0, sink_end)`
   - `[tail_start, total_tokens)`
4. 强制把最后一个 token 加入保留集

### 3.3 运行方式

在 `src/api.py` 中，streamingLLM 的执行流程是：

1. 对 `full_ids` 调用 policy 选出初始 `keep_idx`
2. 生成 `pruned_ids = [full_ids[i] for i in keep_idx]`
3. 将 `pruned_ids` 送入 `_generate_with_manual_cache(...)`

这里要注意：

1. streamingLLM 当前实现只对初始 prompt 做一次静态裁剪。
2. 进入逐步解码后，新生成 token 会继续追加进 cache。
3. 它不会像 h2o 那样在解码过程中持续做淘汰决策。

### 3.4 特点

优点：

1. 结构简单。
2. 初始 cache 大小容易控制。
3. 很适合作为“仅靠 recency + sink”的对照方法。

缺点：

1. 中间历史 token 会整体丢弃。
2. 无法保留那些虽然较早但持续重要的 token。

## 4. h2o

### 4.1 算法定义

h2o 在 streamingLLM 的基础上增加了 Heavy Hitters。

它的思想是：

1. Sink Tokens 必保留。
2. Recent Tokens 必保留。
3. 其余普通 token 根据历史累计注意力得分决定是否保留。

因此它的缓存结构可以写成：

$$
\text{KV Cache} = \text{Sink Tokens} + \text{Heavy Hitters} + \text{Recent Tokens}
$$

缓存总预算为：

$$
\text{Budget} = \text{sink\_size} + \text{local\_window\_size} + \text{heavy\_hitter\_size}
$$

### 4.2 Score 计数器

h2o 的核心是每个缓存 token 都有一个 score 计数器 $S_i$。

#### 初始化

当 token 进入当前活动 cache 时，初始化：

$$
S_i = 0
$$

#### 每步更新

在第 $t$ 步解码时，当前 query 会对 cache 中所有 key 产生注意力权重。项目里的实现做的是：

1. 读取模型返回的每层 attention
2. 取最后一个 query 对所有 key 的注意力
3. 在 head 维上求平均
4. 再在 layer 维上求平均

对应的聚合结果记作 $s_{t,i}$，随后更新：

$$
S_i^{(t)} = S_i^{(t-1)} + s_{t,i}
$$

这表示：如果一个 token 在很多步里持续被关注，它的累计分数会越来越高。

### 4.3 实现位置

#### policy 部分

在 `src/methods/h2o.py` 中：

- `select_streaming_keep_indices(...)`
  - 用来构造初始的 streamingLLM 风格 cache
- `select_keep_indices(...)`
  - 当 cache 超预算时，决定保留哪些 token

#### 运行时部分

在 `src/api.py` 中：

- `_generate_with_h2o(...)`
  - 管理活动 cache、score 计数器和裁剪时机
- `_accumulate_h2o_scores(...)`
  - 把当前步 attention 聚合值累加到 score 计数器

#### attention 读取部分

在 `src/model.py` 中：

- `_aggregate_last_query_attention(...)`
  - 将多层、多头 attention 聚合成当前步每个 key token 的一个分数
- `prefill_next_token_logits_with_attention(...)`
  - prefill 时返回 logits、cache 和 attention 分数
- `next_token_logits_from_cache_with_attention(...)`
  - 增量解码时返回 logits、cache 和 attention 分数

#### cache 裁剪部分

在 `src/model.py` 中：

- `prune_past_key_values(...)`

它会根据保留索引，直接在 `past_key_values` 上裁剪 key/value 张量，而不是重算整段上下文。

### 4.4 淘汰逻辑

当活动 cache 的 token 数量即将超过预算时：

1. 先构造一个“新 token 加入后的候选总序列长度”
2. sink 区间和 recent 区间直接保护，不参与淘汰
3. 在中间普通 token 中，根据 score 计数器选出最高的 `heavy_hitter_size` 个
4. 只保留：
   - sink tokens
   - selected heavy hitters
   - recent tokens

这对应于“淘汰得分最低的普通 token”的思想。

### 4.5 细节补充

1. h2o 的初始 cache 不是 full prompt，而是先经过一次 streamingLLM 风格筛选。
2. score 计数器只对当前活动 cache 中的 token 生效。
3. 当 token 被淘汰后，它的 score 不再保留，也不会再次进入 cache。
4. 当前实现用的是 layer/head 平均后的 attention 累加，而不是简单的单层单头值。
5. 在分数相同的情况下，实现中用极小的 recent tie-break 偏向更靠后的 token，以减少完全相同分数时的不稳定性。

### 4.6 特点

优点：

1. 在 streamingLLM 基础上保留了“长期重要 token”。
2. 比纯 recent window 更能覆盖远处关键信息。
3. 通过直接裁剪 `past_key_values`，避免了旧式整段重建。

代价：

1. 需要在 h2o 路径中开启 `output_attentions=True`。
2. 每步都要更新 score 计数器。
3. 需要在超预算时做 cache 剪枝。

## 5. 三种方法的对比

### 5.1 保留策略对比

1. baseline：保留全部 token
2. streamingLLM：保留 sink + recent
3. h2o：保留 sink + heavy hitters + recent

### 5.2 动态性对比

1. baseline：无动态裁剪
2. streamingLLM：仅初始静态裁剪
3. h2o：解码过程中动态维护和裁剪 cache

### 5.3 对实现开销的影响

1. baseline：实现最简单，cache 最大
2. streamingLLM：实现简单，初始 cache 更小
3. h2o：实现最复杂，需要 attention 聚合、score 累计和 cache 剪枝

## 6. 当前实现边界

当前项目的 h2o 已经是在线缓存管理版本，但仍有一些边界需要明确：

1. streamingLLM 目前只在 prompt 阶段裁剪一次，不会在生成过程中继续滑动裁剪。
2. h2o 的分数只来源于模型返回的 attention，不额外引入其他启发式信号。
3. 三种方法已经统一到同一类手写逐步解码框架，但 h2o 为了实现 score 计数器，仍然必须额外读取 attention。

## 7. 参数建议

### baseline

baseline 没有算法参数，主要受模型和解码参数控制。

### streamingLLM

可调参数：

1. `sink_size`
2. `local_window_size`

经验上：

1. `sink_size` 不宜过大，一般只需要少量前缀 token。
2. `local_window_size` 决定近期上下文保留能力。

### h2o

可调参数：

1. `sink_size`
2. `local_window_size`
3. `heavy_hitter_size`

经验上：

1. `heavy_hitter_size` 越大，越接近 baseline。
2. `heavy_hitter_size` 越小，越依赖 score 计数器筛出真正重要的历史 token。
3. 若任务对远距离依赖强，通常需要给 h2o 留出足够 heavy hitter 预算。