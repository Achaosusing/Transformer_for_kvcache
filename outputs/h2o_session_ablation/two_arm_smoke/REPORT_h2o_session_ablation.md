# H2O Session Ablation Report
## Run Summary
| variant | success_rate | avg_duration | timeout_rate | avg_turns | n |
|---|---:|---:|---:|---:|---:|
| Stateless | 0.300 | 199.65 | 0.000 | 32.53 | 30 |
| Session + Decay | 0.300 | 319.59 | 0.067 | 29.23 | 30 |

## Paired Delta vs Stateless
| variant | reward_delta_mean | duration_delta_mean | timeout_delta_mean |
|---|---:|---:|---:|
| Session + Decay | +0.000 | +119.94 | +0.067 |
