# Commit Prep Notes (2026-03-23)

## 1. Purpose
Prepare a clean commit for KV-cache method parameterization and naming normalization.

## 2. Reviewed Changes
### Core code changes (recommended to commit)
- `api_server.py`
  - Add explicit CLI args for method-specific configs:
    - `--streaming-sink-size`
    - `--streaming-local-window-size`
    - `--h2o-sink-size`
    - `--h2o-local-window-size`
    - `--h2o-heavy-hitter-size`
  - Inject fixed-method default `method_configs` when server is launched with `--method`.

- `scripts/run_tau2_offline_sequential.sh`
  - Pass explicit method parameters to `api_server.py`.
  - Enable `--num-trials` and `--num-tasks` (previously commented out).
  - Use parameterized `save_to` naming:
    - `baseline_{num_trials}x{num_tasks}`
    - `streamingllm_{num_trials}x{num_tasks}_{streaming_sink_size}_{streaming_local_window_size}`
    - `h2o_{num_trials}x{num_tasks}_{h2o_sink_size}_{h2o_local_window_size}_{h2o_heavy_hitter_size}`

- `scripts/run_tau2_offline_parallel_multi_gpu.sh`
  - New parallel runner script.
  - Same explicit parameter passing and naming convention as sequential script.

- `scripts/bootstrap_all.sh`
  - New environment bootstrap helper script.

## 3. Do NOT Commit (generated / environment / experiment artifacts)
### Keep out for this commit
- `tau2-bench/` (explicitly excluded by project policy)
- `local_models/`
- cache artifacts (`__pycache__/`, `*.pyc`, `.venv/`)
- runtime outputs/logs in `outputs/`
- packaging metadata generated locally:
  - `src/oracle_kv_token_eval.egg-info/PKG-INFO`
  - `src/oracle_kv_token_eval.egg-info/requires.txt`
- environment lock drift if not intentionally updated:
  - `uv.lock`

## 4. Review Findings (before commit)
1. Generated artifacts mixed into working tree
   - `outputs/*` renamed/new logs and deleted old logs are currently in git status.
   - Risk: noisy history and oversized commit.

2. Local packaging metadata changed
   - `src/oracle_kv_token_eval.egg-info/*` changed due local install/build actions.
   - Risk: non-source, machine-specific churn.

3. Lockfile changed significantly
   - `uv.lock` contains broad dependency source/index and package graph changes.
   - Risk: dependency drift unrelated to this feature unless lockfile update was intentional.

## 5. Recommended Commit Scope
Commit only these files:
- `api_server.py`
- `scripts/run_tau2_offline_sequential.sh`
- `scripts/run_tau2_offline_parallel_multi_gpu.sh`
- `scripts/bootstrap_all.sh`
- `docs/commit_prep_2026-03-23.md`

## 6. Suggested Commands
```bash
# 1) Stage only source + docs intended for this feature
git add \
  api_server.py \
  scripts/run_tau2_offline_sequential.sh \
  scripts/run_tau2_offline_parallel_multi_gpu.sh \
  scripts/bootstrap_all.sh \
  docs/commit_prep_2026-03-23.md

# 2) Verify staged content
git diff --staged --stat
git diff --staged

# 3) Commit
git commit -m "feat: parameterize streamingllm/h2o runs and normalize result naming"
```

## 7. Optional Follow-up (separate commit)
If you want to prevent future accidental commits of logs/artifacts, add ignore rules in a separate housekeeping commit:
- `outputs/`
- `.venv/`
- `src/oracle_kv_token_eval.egg-info/`

(Do this only if aligned with your team workflow.)
