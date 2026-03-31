#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.api import OracleKVProjectAPI


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
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
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
        if fixed_method == "h2o":
            attn_impl = "eager"  # H2O needs output_attentions → must use eager
        else:
            attn_impl = "sdpa"  # baseline/streamingllm → use SDPA for speed

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
        sample = {
            "id": "chat_0",
            "messages": [m.model_dump() for m in req.messages],
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
