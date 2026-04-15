#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from typing import Any, Literal

import os

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import uvicorn

from src.api import OracleKVProjectAPI
from src.chat_format import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    build_token_role_and_turn_ids,
    build_token_role_ids,
    extract_tool_calls_from_text,
    normalize_chat_messages,
    normalize_tool_definitions,
    render_chat_content,
)
from src.session_store import (
    H2OChatSessionSnapshot,
    LRUSessionStore,
    build_h2o_session_signature,
    hash_signature,
)


class EvalRequest(BaseModel):
    samples: list[dict[str, Any]] = Field(default_factory=list)
    methods: list[str] | None = None
    method_configs: dict[str, dict[str, Any]] | None = None
    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    stop_on_eos: bool = True
    max_input_tokens: int | None = Field(default=None, gt=0)
    save_step_trace: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None

class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    methods: list[str] | None = None
    method_configs: dict[str, dict[str, Any]] | None = None
    max_input_tokens: int | None = Field(default=None, gt=0)
    save_step_trace: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    methods: list[str] | None = None
    method_configs: dict[str, dict[str, Any]] | None = None
    max_input_tokens: int | None = Field(default=None, gt=0)
    save_step_trace: bool = False


def _validate_normalized_messages(messages: list[dict[str, Any]]) -> None:
    for idx, message in enumerate(messages):
        try:
            render_chat_content(message.get("content"))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content for message {idx}: {exc}",
            ) from exc


def _validate_tool_choice(tool_choice: str | dict[str, Any] | None) -> None:
    if isinstance(tool_choice, dict):
        raise HTTPException(status_code=400, detail="Explicit tool_choice objects are not supported yet")
    if tool_choice in (None, "auto", "none"):
        return
    if tool_choice == "required":
        raise HTTPException(status_code=400, detail="tool_choice='required' is not supported yet")
    raise HTTPException(status_code=400, detail=f"Unsupported tool_choice: {tool_choice}")


def _as_bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


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
        choices=["baseline", "streamingllm", "streaming_llm", "h2o", "dta_h2o"],
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
    # DTA-H2O specific parameters.
    parser.add_argument(
        "--dta-gamma", type=float, default=0.95,
        help="DTA-H2O temporal decay factor (0,1). Applied per attention collection step.",
    )
    parser.add_argument(
        "--dta-current-turn-ratio", type=float, default=0.6,
        help="Fraction of heavy-hitter budget reserved for current turn tokens.",
    )
    parser.add_argument(
        "--dta-ghost-buffer-size", type=int, default=32,
        help="Size of the eviction ghost buffer for anti-cascade protection (0=disabled).",
    )
    parser.add_argument(
        "--dta-system-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize system-role tokens within the cache budget (DTA-H2O).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for Bearer token auth. Also reads API_KEY env var. "
             "When set, all endpoints require 'Authorization: Bearer <key>'.",
    )
    return parser


def _resolve_h2o_method_config(
    req_method_configs: dict[str, Any],
    fixed_method: str | None,
    default_dta_gamma: float,
) -> tuple[str, dict[str, Any], float, float, dict[int, float]]:
    """Determine H2O variant, config, decay params, and role alphas."""
    h2o_method_key = (
        "dta_h2o"
        if "dta_h2o" in req_method_configs or fixed_method == "dta_h2o"
        else "h2o"
    )
    method_cfg = req_method_configs.get(
        h2o_method_key, req_method_configs.get("h2o", {}),
    )
    dta_gamma = float(method_cfg.get(
        "dta_gamma", 1.0 if h2o_method_key == "h2o" else default_dta_gamma,
    ))
    score_alpha = float(method_cfg.get("session_score_alpha", 0.5))
    if score_alpha < 0.0:
        raise HTTPException(
            status_code=400, detail="session_score_alpha must be >= 0",
        )

    role_alphas: dict[int, float] = {
        ROLE_SYSTEM: float(method_cfg.get("role_alpha_system", 0.9)),
        ROLE_USER: float(method_cfg.get("role_alpha_user", 0.3)),
        ROLE_ASSISTANT: float(method_cfg.get("role_alpha_assistant", 0.3)),
        ROLE_TOOL: float(method_cfg.get("role_alpha_tool", 0.7)),
    }
    for _role_key, _alpha in role_alphas.items():
        if not (0.0 <= _alpha <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"role_alpha for role {_role_key} must be in [0, 1], got {_alpha}",
            )
    return h2o_method_key, method_cfg, dta_gamma, score_alpha, role_alphas


def _prepare_h2o_prompt_data(
    evaluator: "OracleKVProjectAPI",
    h2o_method_key: str,
    normalized_messages: list[dict[str, Any]],
    active_tools: list[dict[str, Any]] | None,
    max_input_tokens: int | None,
) -> tuple[list[int], list[int], list[int] | None]:
    """Tokenize prompt and build per-token role and turn ID arrays."""
    prompt_ids = evaluator.model.format_prompt_ids(
        prompt=None,
        messages=normalized_messages,
        max_input_tokens=max_input_tokens,
        add_generation_prompt=True,
        tools=active_tools,
        canonical_chat=True,
    )
    prompt_len = len(prompt_ids)

    turn_ids_list: list[int] | None = None
    if h2o_method_key == "dta_h2o":
        role_ids, turn_ids_list = build_token_role_and_turn_ids(
            normalized_messages,
            evaluator.model.tokenizer,
            tools=active_tools,
            add_generation_prompt=True,
        )
    else:
        role_ids = build_token_role_ids(
            normalized_messages,
            evaluator.model.tokenizer,
            tools=active_tools,
            add_generation_prompt=True,
        )

    if len(role_ids) > prompt_len:
        role_ids = role_ids[len(role_ids) - prompt_len:]
        if turn_ids_list is not None:
            turn_ids_list = turn_ids_list[len(turn_ids_list) - prompt_len:]
    elif len(role_ids) < prompt_len:
        role_ids = role_ids + [ROLE_ASSISTANT] * (prompt_len - len(role_ids))
        if turn_ids_list is not None:
            last_turn = turn_ids_list[-1] if turn_ids_list else 0
            turn_ids_list = turn_ids_list + [last_turn] * (
                prompt_len - len(turn_ids_list)
            )

    return prompt_ids, role_ids, turn_ids_list


def _save_h2o_session_snapshot(
    h2o_sessions: LRUSessionStore,
    evaluator: "OracleKVProjectAPI",
    store_key: str,
    normalized_history_messages: list[dict[str, Any]],
    active_tools: list[dict[str, Any]] | None,
    prompt_ids: list[int],
    generated_ids: list[int],
    state: "H2ORuntimeState",
    policy: Any,
    signature: tuple[Any, ...],
    evict_period: int,
    collect_period: int,
    dta_gamma: float,
) -> None:
    """Persist H2O state snapshot if response tokens align with history template."""
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
    if (
        inserted_generated_ids
        and eos_token_id is not None
        and inserted_generated_ids[-1] == eos_token_id
    ):
        inserted_generated_ids = inserted_generated_ids[:-1]
    history_suffix_ids = history_ids[len(prompt_ids):]

    snapshot_state = state
    can_store = False
    if (
        len(history_suffix_ids) >= len(inserted_generated_ids)
        and history_suffix_ids[: len(inserted_generated_ids)]
        == inserted_generated_ids
    ):
        closure_ids = history_suffix_ids[len(inserted_generated_ids):]
        if closure_ids:
            _, snapshot_state = evaluator.continue_h2o_state(
                closure_ids,
                policy,
                snapshot_state,
                evict_period=evict_period,
                collect_period=collect_period,
                dta_gamma=dta_gamma,
            )
        can_store = True
    elif (
        len(inserted_generated_ids) > len(history_suffix_ids)
        and inserted_generated_ids[: len(history_suffix_ids)]
        == history_suffix_ids
    ):
        trim_count = len(inserted_generated_ids) - len(history_suffix_ids)
        snapshot_state = evaluator.trim_h2o_state_tail(
            snapshot_state, trim_count,
        )
        can_store = True

    if can_store:
        h2o_sessions.set(
            store_key,
            H2OChatSessionSnapshot(
                session_id=store_key,
                messages=normalized_history_messages,
                tools=active_tools,
                history_token_ids=history_ids,
                runtime_state=evaluator.clone_h2o_state(
                    snapshot_state, "cpu",
                ),
                signature=signature,
            ),
        )


def _try_restore_from_session(
    h2o_sessions: LRUSessionStore,
    evaluator: "OracleKVProjectAPI",
    enable_session: bool,
    session_id: str | None,
    prompt_ids: list[int],
    signature: tuple[Any, ...],
    normalized_messages: list[dict[str, Any]],
    active_tools: list[dict[str, Any]] | None,
    score_alpha: float,
    role_alphas: dict[int, float],
    policy: Any,
    evict_period: int,
    collect_period: int,
    dta_gamma: float,
) -> tuple[Any, Any]:
    """Try to restore H2O state from a previous session snapshot.

    Returns (cached_logits, state) — both None if no match found.
    """
    if not enable_session:
        return None, None

    snapshot = (
        h2o_sessions.get(session_id)
        if session_id
        else h2o_sessions.find_by_prefix(prompt_ids, signature)
    )
    if snapshot is None:
        return None, None

    messages_match = (
        snapshot.signature == signature
        and snapshot.tools == active_tools
        and len(normalized_messages) > len(snapshot.messages)
        and normalized_messages[: len(snapshot.messages)] == snapshot.messages
        and prompt_ids[: len(snapshot.history_token_ids)]
        == snapshot.history_token_ids
    )
    if not messages_match:
        return None, None

    appended_ids = prompt_ids[len(snapshot.history_token_ids):]
    if not appended_ids:
        return None, None

    state = evaluator.restore_h2o_state(
        snapshot.runtime_state,
        alpha=score_alpha,
        role_alphas=role_alphas,
    )
    cached_logits, state = evaluator.continue_h2o_state(
        appended_ids, policy, state,
        evict_period=evict_period,
        collect_period=collect_period,
        dta_gamma=dta_gamma,
    )
    return cached_logits, state


def _build_chat_completion_result(
    req: "ChatCompletionRequest",
    served_model_name: str,
    method_key: str,
    assistant_content: str,
    response_tool_calls: list[dict[str, Any]] | None,
    traces: list[Any],
    prompt_len: int,
    generated_ids: list[int],
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response dict."""
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
        "method": method_key,
    }
    if req.save_step_trace:
        choice["step_trace"] = [
            trace.model_dump() if hasattr(trace, "model_dump")
            else trace.__dict__
            for trace in traces
        ]

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
    elif fixed_method == "dta_h2o":
        fixed_method_configs[fixed_method] = {
            "sink_size": args.h2o_sink_size,
            "local_window_size": args.h2o_local_window_size,
            "heavy_hitter_size": args.h2o_heavy_hitter_size,
            "dta_gamma": args.dta_gamma,
            "current_turn_ratio": args.dta_current_turn_ratio,
            "system_anchor": args.dta_system_anchor,
            "ghost_buffer_size": args.dta_ghost_buffer_size,
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
            raise HTTPException(
                status_code=400,
                detail="session_id cannot be combined with max_input_tokens yet",
            )

        h2o_method_key, method_cfg, dta_gamma, score_alpha, role_alphas = (
            _resolve_h2o_method_config(
                req_method_configs, fixed_method, args.dta_gamma,
            )
        )
        policy = evaluator.build_policy(h2o_method_key, method_cfg)
        signature = build_h2o_session_signature(
            h2o_method_key, method_cfg,
            evict_period=args.evict_period,
            collect_period=collect_period,
            alpha=score_alpha, role_alphas=role_alphas,
        )

        prompt_ids, role_ids, turn_ids_list = _prepare_h2o_prompt_data(
            evaluator, h2o_method_key, normalized_messages,
            active_tools, req.max_input_tokens,
        )
        prompt_len = len(prompt_ids)

        t0 = time.perf_counter()
        cached_logits, state = _try_restore_from_session(
            h2o_sessions, evaluator, enable_session, req.session_id,
            prompt_ids, signature, normalized_messages, active_tools,
            score_alpha, role_alphas, policy,
            args.evict_period, collect_period, dta_gamma,
        )
        if cached_logits is None or state is None:
            cached_logits, state = evaluator.initialize_h2o_state(
                prompt_ids, policy,
                role_ids=role_ids, turn_ids_list=turn_ids_list,
            )

        generated_ids, traces, _, state = evaluator.generate_from_h2o_state(
            cached_logits=cached_logits, state=state, policy=policy,
            max_new_tokens=max_new_tokens,
            temperature=req.temperature, top_p=req.top_p,
            stop_on_eos=True, save_step_trace=req.save_step_trace,
            prompt_len=prompt_len,
            evict_period=args.evict_period,
            collect_period=collect_period, dta_gamma=dta_gamma,
        )
        elapsed = time.perf_counter() - t0

        output_text = evaluator.model.tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        )
        assistant_content, parsed_tool_calls = extract_tool_calls_from_text(
            output_text,
        )
        response_tool_calls = _build_response_tool_calls(parsed_tool_calls)

        assistant_msg: dict[str, Any] = {
            "role": "assistant", "content": assistant_content,
        }
        if response_tool_calls:
            assistant_msg["tool_calls"] = response_tool_calls
        history_messages = normalize_chat_messages(
            normalized_messages + [assistant_msg],
        )

        if enable_session:
            store_key = req.session_id or (
                f"auto_"
                f"{hashlib.sha256(','.join(map(str, prompt_ids)).encode()).hexdigest()[:20]}_"
                f"{len(normalized_messages)}_{hash_signature(signature)}"
            )
            _save_h2o_session_snapshot(
                h2o_sessions, evaluator, store_key,
                history_messages, active_tools, prompt_ids,
                generated_ids, state, policy, signature,
                args.evict_period, collect_period, dta_gamma,
            )

        return _build_chat_completion_result(
            req, served_model_name, h2o_method_key,
            assistant_content, response_tool_calls, traces,
            prompt_len, generated_ids,
        )

    app = FastAPI(title="Oracle KV Eval API", version="1.0.0")

    # --- API key authentication ---
    api_key = args.api_key or os.environ.get("API_KEY")
    if api_key:
        _bearer_scheme = HTTPBearer()

        async def _verify_api_key(
            credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
        ) -> None:
            if credentials.credentials != api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")

        app.dependency_overrides[_verify_api_key] = _verify_api_key
        app.router.dependencies.append(Depends(_verify_api_key))

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
        try:
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
        except ValueError as exc:
            raise _as_bad_request(exc) from exc

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
        _validate_normalized_messages(normalized_messages)

        _validate_tool_choice(req.tool_choice)
        active_tools = None if req.tool_choice == "none" else normalize_tool_definitions(req.tools)

        if req.tools and (len(method_keys) != 1 or method_keys[0] not in ("h2o", "dta_h2o")):
            raise HTTPException(status_code=400, detail="tools are currently supported only for single-method h2o/dta_h2o chat requests")

        if len(method_keys) == 1 and method_keys[0] in ("h2o", "dta_h2o"):
            try:
                return _h2o_chat_response(
                    req, req_method_configs, normalized_messages,
                    active_tools, max_new_tokens,
                )
            except ValueError as exc:
                raise _as_bad_request(exc) from exc

        sample = {
            "id": "chat_0",
            "messages": normalized_messages,
        }
        try:
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
        except ValueError as exc:
            raise _as_bad_request(exc) from exc

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
        try:
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
        except ValueError as exc:
            raise _as_bad_request(exc) from exc

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
