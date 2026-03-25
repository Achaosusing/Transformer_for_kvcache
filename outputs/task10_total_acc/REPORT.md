# Total Target-10 ACC Report

## Scope
- Included source A: old30 full runs, but only the target task IDs 0/1/3/4/5/6/10/13/26/28 are counted
- Included source B: new10 reruns with tag task10, only complete 10-task files are counted
- Excluded: incomplete new10 files
- Valid run files counted: 62
- Valid task instances counted across all methods: 620

## Overall Method ACC
- baseline: acc=0.9625, correct=77, total=80
- h2o: acc=0.8545, correct=376, total=440
- streamingllm: acc=0.7100, correct=71, total=100

## By Source Group
- new10 / baseline: acc=0.9250, correct=37, total=40
- new10 / h2o: acc=0.8316, correct=158, total=190
- new10 / streamingllm: acc=0.6667, correct=20, total=30
- old30 / baseline: acc=1.0000, correct=40, total=40
- old30 / h2o: acc=0.8720, correct=218, total=250
- old30 / streamingllm: acc=0.7286, correct=51, total=70

## Earliest Best H2O Points
- budget=68 acc=0.5500, total=20, window=32, heavy=32
- budget=100 acc=0.8500, total=20, window=64, heavy=32
- budget=132 acc=0.9000, total=10, window=96, heavy=32
- budget=164 acc=0.7500, total=20, window=128, heavy=32
- budget=196 acc=0.9000, total=20, window=128, heavy=64
- budget=228 acc=0.9000, total=10, window=192, heavy=32
- budget=260 acc=0.9000, total=20, window=128, heavy=128
- budget=292 acc=1.0000, total=10, window=256, heavy=32
