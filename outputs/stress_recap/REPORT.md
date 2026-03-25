# Stress Test Recap

## Scope
- old30: earlier 30-task stress results
- new10: newly rerun task10 results with filenames containing task10
- complete runs used in main plots: old30=36, new10=26
- incomplete runs excluded from main curves: 4

## Old30 Summary
- Baseline mean reward: 0.3333
- Baseline mean timeout: 0.0750
- StreamingLLM best: budget=1028 reward=0.3333 timeout=0.0000
- StreamingLLM main drop: 260 -> 132 (drop=0.1000)
- H2O earliest/best plateau candidate: budget=292 reward=0.3333 timeout=0.0000 window=256 heavy=32
- H2O best-curve main drop: 132 -> 100 (drop=0.0667)

## New10 Summary
- Baseline mean reward: 0.9250
- Baseline mean timeout: 0.0500
- StreamingLLM best: budget=132 reward=0.8000 timeout=0.2000
- StreamingLLM main drop: 100 -> 36 (drop=0.2000)
- H2O earliest/best plateau candidate: budget=100 reward=1.0000 timeout=0.0000 window=64 heavy=32
- H2O best-curve main drop: 100 -> 68 (drop=0.5000)

## Incomplete New10 Runs
- stress_h2o_1x10_4_160_128_task10.json: actual_tasks=7 expected_tasks=10 reward=1.0000
- stress_h2o_1x10_4_160_256_task10.json: actual_tasks=8 expected_tasks=10 reward=1.0000
- stress_h2o_1x10_4_160_32_task10.json: actual_tasks=7 expected_tasks=10 reward=1.0000
- stress_streamingllm_1x10_4_64_task10.json: actual_tasks=9 expected_tasks=10 reward=0.3333

## Output Files
- comparison_success.png: side-by-side success curves for old30 and new10
- comparison_timeout.png: side-by-side timeout curves for old30 and new10
- old30_h2o_heatmap.png: H2O heatmap for 30-task results
- new10_h2o_heatmap.png: H2O heatmap for 10-task rerun results
- *_aggregated_configs.csv: per-config aggregated metrics
- new10_task_reward_summary.csv: task-level reward summary for the rerun task10 set
