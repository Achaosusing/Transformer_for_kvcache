# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and common commands

- Recommended environment bootstrap:
  ```bash
  conda create -n oracle-kv python=3.10 -y
  conda activate oracle-kv
  python -m pip install -U pip setuptools wheel
  python -m pip install -e '.[dev]'
  ```
- Install analysis extras when needed:
  ```bash
  python -m pip install -e '.[dev,analysis]'
  ```
- Optional uv setup:
  ```bash
  uv sync --extra dev
  uv sync --extra dev --extra analysis
  ```
- Run the test suite:
  ```bash
  pytest -q
  ```
- Run one test file:
  ```bash
  pytest tests/test_dta_h2o.py -q
  ```
- Run one test case:
  ```bash
  pytest tests/test_dta_h2o.py::test_name -q
  ```
- Lint:
  ```bash
  ruff check .
  ```
- Start the OpenAI-compatible API server (replace method/config as needed):
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
- Other supported methods are `streamingllm` / `streaming_llm`, `h2o`, and `dta_h2o`.

## High-level architecture

- `api_server.py` is the FastAPI/CLI entrypoint. It exposes `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, and `/v1/evaluate`, validates OpenAI-style requests, and wraps the H2O session restore/save flow around request handling.
- `src/api.py` is the core runtime orchestrator. `OracleKVProjectAPI` builds the selected policy, routes requests into baseline vs. streamingLLM vs. H2O paths, and owns the H2O state machine (`initialize_h2o_state`, `continue_h2o_state`, `generate_from_h2o_state`, restore/trim helpers).
- `src/model.py` is the Hugging Face model wrapper. It handles tokenizer/model loading, prompt formatting, token generation, KV-cache pruning, and `SDPAAttentionCapture`, which collects last-query attention scores while still using SDPA kernels.
- `src/chat_format.py` is the canonical chat/tool serialization layer. It normalizes OpenAI-style messages and tool calls, renders a stable chat prompt, and builds per-token role/turn tags consumed by H2O/DTA-H2O and session decay logic.
- `src/session_store.py` is the multi-turn H2O memory layer. It stores `H2OChatSessionSnapshot` objects in an LRU cache and supports both exact lookup by `session_id` and automatic longest-prefix lookup over tokenized history plus a runtime signature.
- `src/methods/__init__.py` is the method registry. Policy implementations live under `src/methods/`:
  - baseline: full attention, no cache pruning
  - streamingLLM: sink + recent-window pruning
  - H2O: sink + recent + heavy-hitter retention
  - DTA-H2O: H2O plus temporal decay, current-turn/history budget split, system-anchor preservation, and ghost-buffer anti-cascade protection

## Important behavioral details

- If the server is started with `--method`, every request is forced onto that single method. If not fixed, `/v1/evaluate` defaults to multiple methods while chat/completions defaults to `baseline`.
- H2O/DTA-H2O session reuse only applies to single-method chat requests when `--enable-session` is enabled.
- Multi-turn reuse depends on three pieces staying aligned when editing code: canonical chat formatting (`src/chat_format.py`), token role/turn tagging, and session signatures (`src/session_store.py`).
- DTA-H2O is intentionally a strict extension of H2O: with `gamma=1.0`, `current_turn_ratio=1.0`, `system_anchor=False`, and `ghost_buffer_size=0`, it collapses back to vanilla H2O behavior.

## Tests as executable documentation

- `tests/test_chat_format_and_session.py` covers canonical chat stability, tool-call formatting/parsing, and session-signature behavior.
- `tests/test_model_cache_prune.py` covers supported cache shapes and cache-pruning behavior in `LocalTransformerModel`.
- `tests/test_dta_h2o.py` covers DTA-H2O invariants such as temporal decay, system-anchor protection, tiered budgeting, and ghost-buffer behavior.

## Supporting docs and scripts

- `docs/algorithms.md` explains the intended algorithmic behavior across baseline, streamingLLM, H2O, and DTA-H2O.
- `docs/attention_analysis_guide.md` explains the attention-by-role analysis workflow.
- `scripts/run/taubench/` contains benchmark/run orchestration scripts.
- `scripts/analyze/taubench/` contains analysis scripts for TauBench/Tau2 outputs.
