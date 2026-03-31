# evict_period Sweep Analysis Report
## h2o
- Fixed config: sink=4, window=64, heavy=32, budget=100
- Sweep range: ep ∈ [1, 1, 2, 4, 8, 12, 16]
- **Best**: ep=1  → success=100.0%
- **Worst**: ep=1  → success=40.0%
- Range of improvement: 60.0%

| evict_period | success_rate | timeout_rate | avg_turns | n |
|---|---|---|---|---|
| 1 | 0.400 | 0.600 | 43.9 | 10 |
| 1 | 1.000 | 0.000 | 17.0 | 2 |
| 2 | 0.600 | 0.400 | 53.9 | 10 |
| 4 | 0.600 | 0.400 | 38.6 | 10 |
| 8 | 0.700 | 0.300 | 36.0 | 10 |
| 12 | 0.900 | 0.100 | 32.6 | 10 |
| 16 | 0.800 | 0.200 | 34.6 | 10 |

## Key Observations

- `evict_period=1` means pruning happens every single decode step (exact on-budget). This is the most aggressive setting and tends to hurt quality because:
  - Heavy-hitter scores accumulate over fewer tokens, making selection noisier.
  - The cache is always exactly at budget; the model never has a chance to see slightly more context.

- Higher `evict_period` allows the cache to briefly exceed budget by `ep-1` tokens before batch-pruning. Benefits:
  - H2O accumulates more attention signal per selection → more stable heavy-hitter ranking.
  - StreamingLLM window is unaffected, but overhead is reduced.

- There is a **sweet spot**: too large an ep means the cache grows significantly above budget before pruning, causing a larger disruptive drop. For budget=100, ep≈8–12 empirically gives the best accuracy.

## Recommendation

- **h2o**: use `evict_period=1` as default.
