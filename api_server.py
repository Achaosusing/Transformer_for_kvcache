#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.api import H2ORuntimeState, OracleKVProjectAPI
from src.chat_format import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    build_token_role_ids,
    extract_tool_calls_from_text,
    normalize_chat_messages,
    normalize_tool_definitions,
)


class EvalRequest(BaseModel):
    samples: list[dict[str, Any]] = Field(default_factory=list)
    methods: list[str] | None = None
    method_configs: dict[str, dict[str, Any]] | None = None
    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    stop_on_eos: bool = True
    max_input_tokens: int | None = None
    save_step_trace: bool = False


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    session_id: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    methods: list[str] | None = None
    method_configs: dict[str, dict[str, Any]] | None = None
    max_input_tokens: int | None = None
    save_step_trace: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    methods: list[str] | None = None
    method_configs: dict[str, dict[str, Any]] | None = None
    max_input_tokens: int | None = None
    save_step_trace: bool = False


@dataclass
class H2OChatSessionSnapshot:
    session_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    history_token_ids: list[int]
    runtime_state: H2ORuntimeState
    signature: tuple[Any, ...]


class LRUSessionStore:
    """Bounded LRU store for H2O session snapshots with automatic prefix matching.

    Supports two lookup modes:
    1. Exact key lookup via ``get(key)`` — used when the client provides a
       ``session_id``.
    2. Automatic token-prefix lookup via ``find_by_prefix(prompt_ids, signature)``
       — scans all snapshots and returns the one whose ``history_token_ids`` is
       the longest prefix of *prompt_ids* (with matching signature).  This makes
       multi-turn session reuse work transparently for any OpenAI-compatible
       client that does NOT send ``session_id``.
    """

    _PREFIX_HASH_LEN = 64  # number of leading tokens used for the fast hash index

    def __init__(self, max_size: int) -> None:
        self._store: OrderedDict[str, H2OChatSessionSnapshot] = OrderedDict()
        self._max_size = max_size
        # Fast index: hash of the first _PREFIX_HASH_LEN token ids → set of store keys.
        self._prefix_index: dict[str, set[str]] = {}

    @staticmethod
    def _hash_token_prefix(token_ids: list[int], length: int = 64) -> str:
        """Deterministic hash of the first *length* token ids."""
        prefix = token_ids[:length]
        raw = ",".join(map(str, prefix))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> H2OChatSessionSnapshot | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: H2OChatSessionSnapshot) -> None:
        # Remove old index entry if updating an existing key.
        if key in self._store:
            old_snap = self._store[key]
            old_h = self._hash_token_prefix(old_snap.history_token_ids, self._PREFIX_HASH_LEN)
            bucket = self._prefix_index.get(old_h)
            if bucket is not None:
                bucket.discard(key)
                if not bucket:
                    del self._prefix_index[old_h]
            self._store.move_to_end(key)

        self._store[key] = value

        # Add to prefix index.
        h = self._hash_token_prefix(value.history_token_ids, self._PREFIX_HASH_LEN)
        self._prefix_index.setdefault(h, set()).add(key)

        # Evict oldest entries if over capacity.
        while len(self._store) > self._max_size:
            evicted_key, evicted_snap = self._store.popitem(last=False)
            eh = self._hash_token_prefix(evicted_snap.history_token_ids, self._PREFIX_HASH_LEN)
            bucket = self._prefix_index.get(eh)
            if bucket is not None:
                bucket.discard(evicted_key)
                if not bucket:
                    del self._prefix_index[eh]

    def find_by_prefix(
        self,
        prompt_ids: list[int],
        signature: tuple[Any, ...],
    ) -> H2OChatSessionSnapshot | None:
        """Find the best snapshot whose ``history_token_ids`` is a prefix of
        *prompt_ids* and whose signature matches.

        Uses a two-stage strategy:
        1. Hash the first *_PREFIX_HASH_LEN* tokens of *prompt_ids* and check
           the index for candidate keys (O(1) average).
        2. Among candidates (or all entries on hash miss), verify full prefix
           match and pick the longest.

        Returns ``None`` if no matching snapshot is found.
        """
        if not self._store:
            return None

        prompt_len = len(prompt_ids)
        h = self._hash_token_prefix(prompt_ids, self._PREFIX_HASH_LEN)
        candidate_keys = self._prefix_index.get(h)

        if candidate_keys:
            candidates = [(k, self._store[k]) for k in candidate_keys if k in self._store]
        else:
            # Fallback: scan all entries (handles cases where prompt is shorter
            # than _PREFIX_HASH_LEN or hash bucket was pruned).
            candidates = list(self._store.items())

        best: H2OChatSessionSnapshot | None = None
        best_len = 0

        for key, snap in candidates:
            if snap.signature != signature:
                continue
            hist_len = len(snap.history_token_ids)
            if hist_len > prompt_len:
                continue
            if hist_len <= best_len:
                continue
            if prompt_ids[:hist_len] == snap.history_token_ids:
                best = snap
                best_len = hist_len

        if best is not None:
            self._store.move_to_end(best.session_id)

        return best


def _build_h2o_session_signature(
    method_cfg: dict[str, Any],
    *,
    evict_period: int,
    collect_period: int,
    alpha: float,
) -> tuple[Any, ...]:
    return (
        int(method_cfg.get("sink_size", 4)),
        int(method_cfg.get("local_window_size", 256)),
        int(method_cfg.get("heavy_hitter_size", 128)),
        int(evict_period),
        int(collect_period),
        float(alpha),
    )


def _build_response_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None

    response_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        response_calls.append(
            {
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": str(function.get("name", "")),
                    "arguments": json.dumps(function.get("arguments", {}), ensure_ascii=False),
                },
            }
        )
    return response_calls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HTTP API server for KV evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument(
        "--method",
        choices=["baseline", "streamingllm", "streaming_llm", "h2o"],
        default=None,
        help="If set, this server instance always evaluates with one method.",
    )
    parser.add_argument("--streaming-sink-size", type=int, default=4)
    parser.add_argument("--streaming-local-window-size", type=int, default=256)
    parser.add_argument("--h2o-sink-size", type=int, default=4)
    parser.add_argument("--h2o-local-window-size", type=int, default=256)
    parser.add_argument("--h2o-heavy-hitter-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--evict-period", type=int, default=1,
        help="Batch eviction period: prune cache every N tokens instead of every token. "
             "1=exact per-token eviction (default). Higher values reduce pruning overhead.",
    )
    parser.add_argument(
        "--attn-implementation", type=str, default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention implementation. 'auto' uses sdpa for baseline/streamingllm "
             "and eager for h2o (which needs output_attentions).",
    )
    parser.add_argument(
        "--collect-period", type=int, default=0,
        help="H2O attention collection period (steps between collecting attention scores). "
             "0=same as evict_period (default). 1=every step. Higher=fewer attention calls.",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=64,
        help="Maximum number of H2O session snapshots to keep in memory (LRU eviction).",
    )
    parser.add_argument(
        "--enable-session", action="store_true", default=False,
        help="Enable H2O multi-turn session reuse (snapshot save/restore, role-aware "
             "decay, automatic prefix matching). When disabled (default), every H2O "
             "request is treated as a standalone turn — useful as an ablation baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fixed_method = args.method
    served_model_name = args.served_model_name or args.model_path.rstrip("/").split("/")[-1]
    fixed_method_configs: dict[str, dict[str, Any]] = {}
    if fixed_method in ("streamingllm", "streaming_llm"):
        fixed_method_configs[fixed_method] = {
            "sink_size": args.streaming_sink_size,
            "local_window_size": args.streaming_local_window_size,
        }
    elif fixed_method == "h2o":
        fixed_method_configs[fixed_method] = {
            "sink_size": args.h2o_sink_size,
            "local_window_size": args.h2o_local_window_size,
            "heavy_hitter_size": args.h2o_heavy_hitter_size,
        }

    # Resolve attention implementation
    attn_impl = args.attn_implementation
    if attn_impl == "auto":
        attn_impl = "sdpa"

    # Resolve collect_period
    collect_period = args.collect_period if args.collect_period > 0 else args.evict_period

    evaluator = OracleKVProjectAPI(
        model_path=args.model_path,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        allow_remote_files=args.allow_remote_files,
        attn_implementation=attn_impl,
    )
    h2o_sessions = LRUSessionStore(args.max_sessions)
    enable_session: bool = args.enable_session

    def _h2o_chat_response(
        req: ChatCompletionRequest,
        req_method_configs: dict[str, dict[str, Any]],
        normalized_messages: list[dict[str, Any]],
        active_tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if req.session_id and req.max_input_tokens is not None:
            raise HTTPException(status_code=400, detail="session_id cannot be combined with max_input_tokens yet")

        method_cfg = req_method_configs.get("h2o", {})
        score_alpha = float(method_cfg.get("session_score_alpha", 0.5))
        if score_alpha < 0.0:
            raise HTTPException(status_code=400, detail="session_score_alpha must be >= 0")

        # Role-aware decay alphas (Scheme C).
        role_alphas: dict[int, float] = {
            ROLE_SYSTEM: float(method_cfg.get("role_alpha_system", 0.9)),
            ROLE_USER: float(method_cfg.get("role_alpha_user", 0.3)),
            ROLE_ASSISTANT: float(method_cfg.get("role_alpha_assistant", 0.3)),
            ROLE_TOOL: float(method_cfg.get("role_alpha_tool", 0.7)),
        }

        policy = evaluator._build_policy("h2o", method_cfg)
        signature = _build_h2o_session_signature(
            method_cfg,
            evict_period=args.evict_period,
            collect_period=collect_period,
            alpha=score_alpha,
        )
        prompt_ids = evaluator.model.format_prompt_ids(
            prompt=None,
            messages=normalized_messages,
            max_input_tokens=req.max_input_tokens,
            add_generation_prompt=True,
            tools=active_tools,
            canonical_chat=True,
        )
        prompt_len = len(prompt_ids)

        # Build per-token role ids for role-aware decay.
        role_ids = build_token_role_ids(
            normalized_messages,
            evaluator.model.tokenizer,
            tools=active_tools,
            add_generation_prompt=True,
        )
        # Align with possible truncation from max_input_tokens.
        if len(role_ids) > prompt_len:
            role_ids = role_ids[len(role_ids) - prompt_len:]
        elif len(role_ids) < prompt_len:
            role_ids = role_ids + [ROLE_ASSISTANT] * (prompt_len - len(role_ids))

        t0 = time.perf_counter()
        cached_logits = None
        state = None

        # --- Session lookup: explicit session_id OR automatic prefix match ---
        snapshot = None
        if enable_session:
            if req.session_id:
                snapshot = h2o_sessions.get(req.session_id)
            else:
                snapshot = h2o_sessions.find_by_prefix(prompt_ids, signature)

        if snapshot is not None:
            messages_match = (
                snapshot.signature == signature
                and snapshot.tools == active_tools
                and len(normalized_messages) > len(snapshot.messages)
                and normalized_messages[: len(snapshot.messages)] == snapshot.messages
                and prompt_ids[: len(snapshot.history_token_ids)] == snapshot.history_token_ids
            )
            if messages_match:
                appended_ids = prompt_ids[len(snapshot.history_token_ids):]
                if appended_ids:
                    state = evaluator.restore_h2o_state(
                        snapshot.runtime_state,
                        alpha=score_alpha,
                        role_alphas=role_alphas,
                    )
                    cached_logits, state = evaluator.continue_h2o_state(
                        appended_ids,
                        policy,
                        state,
                        evict_period=args.evict_period,
                        collect_period=collect_period,
                    )

        if cached_logits is None or state is None:
            cached_logits, state = evaluator.initialize_h2o_state(prompt_ids, policy, role_ids=role_ids)

        generated_ids, traces, _, state = evaluator.generate_from_h2o_state(
            cached_logits=cached_logits,
            state=state,
            policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_on_eos=True,
            save_step_trace=req.save_step_trace,
            prompt_len=prompt_len,
            evict_period=args.evict_period,
            collect_period=collect_period,
        )
        elapsed = time.perf_counter() - t0

        output_text = evaluator.model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        assistant_content, parsed_tool_calls = extract_tool_calls_from_text(output_text)
        response_tool_calls = _build_response_tool_calls(parsed_tool_calls)

        assistant_history_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if response_tool_calls:
            assistant_history_message["tool_calls"] = response_tool_calls
        normalized_history_messages = normalize_chat_messages(
            normalized_messages + [assistant_history_message]
        )

        # --- Snapshot save: only when session reuse is enabled ---
        if enable_session:
            store_key = req.session_id or f"auto_{LRUSessionStore._hash_token_prefix(prompt_ids, LRUSessionStore._PREFIX_HASH_LEN)}_{len(normalized_messages)}"

            history_ids = evaluator.model.format_prompt_ids(
                prompt=None,
                messages=normalized_history_messages,
                max_input_tokens=None,
                add_generation_prompt=False,
                tools=active_tools,
                canonical_chat=True,
            )
            eos_token_id = evaluator.model.tokenizer.eos_token_id
            inserted_generated_ids = generated_ids
            if inserted_generated_ids and eos_token_id is not None and inserted_generated_ids[-1] == eos_token_id:
                inserted_generated_ids = inserted_generated_ids[:-1]
            history_suffix_ids = history_ids[len(prompt_ids):]

            snapshot_state = state
            can_store_snapshot = False
            if len(history_suffix_ids) >= len(inserted_generated_ids) and history_suffix_ids[: len(inserted_generated_ids)] == inserted_generated_ids:
                closure_ids = history_suffix_ids[len(inserted_generated_ids):]
                if closure_ids:
                    _, snapshot_state = evaluator.continue_h2o_state(
                        closure_ids,
                        policy,
                        snapshot_state,
                        evict_period=args.evict_period,
                        collect_period=collect_period,
                    )
                can_store_snapshot = True
            elif len(inserted_generated_ids) > len(history_suffix_ids) and inserted_generated_ids[: len(history_suffix_ids)] == history_suffix_ids:
                trim_count = len(inserted_generated_ids) - len(history_suffix_ids)
                snapshot_state = evaluator.trim_h2o_state_tail(snapshot_state, trim_count)
                can_store_snapshot = True

            if can_store_snapshot:
                h2o_sessions.set(store_key, H2OChatSessionSnapshot(
                    session_id=store_key,
                    messages=normalized_history_messages,
                    tools=active_tools,
                    history_token_ids=history_ids,
                    runtime_state=evaluator.clone_h2o_state(snapshot_state, "cpu"),
                    signature=signature,
                ))

        choice_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if response_tool_calls:
            choice_message["tool_calls"] = response_tool_calls

        choice: dict[str, Any] = {
            "index": 0,
            "message": choice_message,
            "finish_reason": "tool_calls" if response_tool_calls else "stop",
            "method": "h2o",
        }
        if req.save_step_trace:
            choice["step_trace"] = [trace.model_dump() if hasattr(trace, "model_dump") else trace.__dict__ for trace in traces]

        model_name = req.model or served_model_name
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": len(generated_ids),
                "total_tokens": prompt_len + len(generated_ids),
            },
        }

    app = FastAPI(title="Oracle KV Eval API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/evaluate")
    def evaluate(req: EvalRequest) -> dict[str, Any]:
        methods = [fixed_method] if fixed_method else (req.methods or ["baseline", "streamingllm", "h2o"])
        req_method_configs = req.method_configs or {}
        if fixed_method and fixed_method_configs and fixed_method not in req_method_configs:
            req_method_configs = {**req_method_configs, **fixed_method_configs}
        max_new_tokens = req.max_new_tokens if req.max_new_tokens is not None else args.max_new_tokens
        return evaluator.evaluate(
            samples=req.samples,
            methods=methods,
            method_configs=req_method_configs,
            max_new_tokens=max_new_tokens,
            evict_period=args.evict_period,
            collect_period=collect_period,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_on_eos=req.stop_on_eos,
            max_input_tokens=req.max_input_tokens,
            save_step_trace=req.save_step_trace,
        )

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported yet")

        methods = [fixed_method] if fixed_method else (req.methods or ["baseline"])
        req_method_configs = req.method_configs or {}
        if fixed_method and fixed_method_configs and fixed_method not in req_method_configs:
            req_method_configs = {**req_method_configs, **fixed_method_configs}

        max_new_tokens = req.max_tokens if req.max_tokens is not None else args.max_new_tokens
        method_keys = [method.lower() for method in methods]
        normalized_messages = normalize_chat_messages(
            [message.model_dump(exclude_none=True) for message in req.messages]
        )

        if isinstance(req.tool_choice, dict):
            raise HTTPException(status_code=400, detail="Explicit tool_choice objects are not supported yet")
        active_tools = None if req.tool_choice == "none" else normalize_tool_definitions(req.tools)

        if req.tools and (len(method_keys) != 1 or method_keys[0] != "h2o"):
            raise HTTPException(status_code=400, detail="tools are currently supported only for single-method h2o chat requests")

        if len(method_keys) == 1 and method_keys[0] == "h2o":
            return _h2o_chat_response(
                req, req_method_configs, normalized_messages,
                active_tools, max_new_tokens,
            )

        sample = {
            "id": "chat_0",
            "messages": normalized_messages,
        }
        result = evaluator.evaluate(
            samples=[sample],
            methods=methods,
            method_configs=req_method_configs,
            max_new_tokens=max_new_tokens,
            evict_period=args.evict_period,
            collect_period=collect_period,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_on_eos=True,
            max_input_tokens=req.max_input_tokens,
            save_step_trace=req.save_step_trace,
        )

        choices: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        for idx, method in enumerate(methods):
            rows = result["results_by_method"].get(method.lower(), [])
            if not rows:
                continue
            row = rows[0]
            prompt_tokens = max(prompt_tokens, int(row["prompt_tokens"]))
            completion_tokens = max(completion_tokens, int(row["generated_tokens"]))
            choices.append(
                {
                    "index": idx,
                    "message": {
                        "role": "assistant",
                        "content": row["output_text"],
                    },
                    "finish_reason": "stop",
                    "method": method.lower(),
                }
            )

        model_name = req.model or served_model_name
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported yet")

        methods = [fixed_method] if fixed_method else (req.methods or ["baseline"])
        req_method_configs = req.method_configs or {}
        if fixed_method and fixed_method_configs and fixed_method not in req_method_configs:
            req_method_configs = {**req_method_configs, **fixed_method_configs}
        sample = {
            "id": "completion_0",
            "prompt": req.prompt,
        }
        max_new_tokens = req.max_tokens if req.max_tokens is not None else args.max_new_tokens
        result = evaluator.evaluate(
            samples=[sample],
            methods=methods,
            method_configs=req_method_configs,
            max_new_tokens=max_new_tokens,
            evict_period=args.evict_period,
            collect_period=collect_period,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_on_eos=True,
            max_input_tokens=req.max_input_tokens,
            save_step_trace=req.save_step_trace,
        )

        choices: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        for idx, method in enumerate(methods):
            rows = result["results_by_method"].get(method.lower(), [])
            if not rows:
                continue
            row = rows[0]
            prompt_tokens = max(prompt_tokens, int(row["prompt_tokens"]))
            completion_tokens = max(completion_tokens, int(row["generated_tokens"]))
            choices.append(
                {
                    "index": idx,
                    "text": row["output_text"],
                    "finish_reason": "stop",
                    "method": method.lower(),
                }
            )

        model_name = req.model or served_model_name
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
