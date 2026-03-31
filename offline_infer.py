#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import read_jsonl, set_global_seed, write_jsonl

if TYPE_CHECKING:
    from src.api import OracleKVProjectAPI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline runner for baseline/streamingLLM/h2o with per-method outputs"
    )
    parser.add_argument("--model-path", default="./local_models/Qwen3.5-9B")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "jsonl", "tau2"],
        default="auto",
        help="Input dataset format. auto will infer from provided arguments.",
    )
    parser.add_argument(
        "--tau2-domain-dir",
        default="./data/tau2/domains/airline",
        help="tau2 domain directory containing tasks.json / split_tasks.json / policy.md",
    )
    parser.add_argument(
        "--tau2-split",
        choices=["train", "test", "base", "all"],
        default="test",
        help="tau2 split name. all uses every task in tasks.json.",
    )
    parser.add_argument(
        "--tau2-include-policy",
        action="store_true",
        help="If set, prepend policy.md as system message for each tau2 sample.",
    )
    parser.add_argument(
        "--tau2-limit",
        type=int,
        default=0,
        help="Optional cap for number of tau2 tasks to run (0 means no cap).",
    )
    parser.add_argument("--output-dir", default="./outputs/offline_infer")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "streamingllm", "h2o"],
        choices=["baseline", "streamingllm", "streaming_llm", "h2o"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming-sink-size", type=int, default=4)
    parser.add_argument("--streaming-local-window-size", type=int, default=256)
    parser.add_argument("--h2o-sink-size", type=int, default=4)
    parser.add_argument("--h2o-local-window-size", type=int, default=256)
    parser.add_argument("--h2o-heavy-hitter-size", type=int, default=128)
    parser.add_argument("--save-step-trace", action="store_true")
    parser.add_argument(
        "--attn-implementation", type=str, default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention implementation. 'auto' uses sdpa for baseline/streamingllm "
             "and eager for h2o (which needs output_attentions).",
    )
    parser.add_argument("--_child-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def build_method_configs(args: argparse.Namespace) -> dict[str, dict[str, float | int]]:
    return {
        "streamingllm": {
            "sink_size": args.streaming_sink_size,
            "local_window_size": args.streaming_local_window_size,
        },
        "streaming_llm": {
            "sink_size": args.streaming_sink_size,
            "local_window_size": args.streaming_local_window_size,
        },
        "h2o": {
            "sink_size": args.h2o_sink_size,
            "local_window_size": args.h2o_local_window_size,
            "heavy_hitter_size": args.h2o_heavy_hitter_size,
        },
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compose_tau2_user_content(task: dict[str, Any]) -> str:
    scenario = task.get("user_scenario", {}) or {}
    instructions = scenario.get("instructions", {}) or {}

    lines: list[str] = []
    reason = str(instructions.get("reason_for_call", "")).strip()
    task_inst = str(instructions.get("task_instructions", "")).strip()
    known_info = str(instructions.get("known_info", "")).strip()
    unknown_info = str(instructions.get("unknown_info", "")).strip()
    persona = str(scenario.get("persona", "")).strip()

    if reason:
        lines.append("Reason for call:")
        lines.append(reason)
    if task_inst:
        lines.append("Task instructions:")
        lines.append(task_inst)
    if known_info:
        lines.append("Known info:")
        lines.append(known_info)
    if unknown_info and unknown_info.lower() != "none":
        lines.append("Unknown info:")
        lines.append(unknown_info)
    if persona and persona.lower() != "none":
        lines.append("Persona:")
        lines.append(persona)

    if not lines:
        return "Please help with this airline support task."
    return "\n".join(lines)


def load_tau2_samples(
    domain_dir: Path,
    split: str,
    include_policy: bool,
    limit: int,
) -> list[dict[str, Any]]:
    tasks_path = domain_dir / "tasks.json"
    split_path = domain_dir / "split_tasks.json"
    policy_path = domain_dir / "policy.md"

    if not tasks_path.exists():
        raise FileNotFoundError(f"tau2 tasks file not found: {tasks_path}")

    tasks = _read_json(tasks_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected list in {tasks_path}")

    selected_ids: set[str] | None = None
    if split != "all":
        if not split_path.exists():
            raise FileNotFoundError(f"tau2 split file not found: {split_path}")
        split_obj = _read_json(split_path)
        if not isinstance(split_obj, dict) or split not in split_obj:
            raise ValueError(f"Split '{split}' not found in {split_path}")
        selected_ids = {str(x) for x in split_obj[split]}

    policy_text = ""
    if include_policy and policy_path.exists():
        policy_text = policy_path.read_text(encoding="utf-8").strip()

    samples: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id", ""))
        if selected_ids is not None and task_id not in selected_ids:
            continue

        user_content = _compose_tau2_user_content(task)
        messages: list[dict[str, str]] = []
        if policy_text:
            messages.append({"role": "system", "content": policy_text})
        messages.append({"role": "user", "content": user_content})

        samples.append(
            {
                "id": f"tau2_{task_id}",
                "messages": messages,
            }
        )

        if limit > 0 and len(samples) >= limit:
            break

    return samples


def load_samples_from_args(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    fmt = args.dataset_format
    if fmt == "auto":
        fmt = "jsonl" if args.input_jsonl else "tau2"

    if fmt == "jsonl":
        if not args.input_jsonl:
            raise ValueError("--input-jsonl is required when --dataset-format jsonl")
        input_path = Path(args.input_jsonl)
        samples = read_jsonl(input_path)
        return samples, str(input_path)

    domain_dir = Path(args.tau2_domain_dir)
    samples = load_tau2_samples(
        domain_dir=domain_dir,
        split=args.tau2_split,
        include_policy=args.tau2_include_policy,
        limit=max(0, int(args.tau2_limit)),
    )
    return samples, f"{domain_dir} (split={args.tau2_split})"


def summarize_rows(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "method": method,
            "samples": 0,
            "avg_elapsed_sec": 0.0,
            "avg_generated_tokens": 0.0,
            "tokens_per_sec": 0.0,
        }

    total_elapsed = sum(float(r.get("elapsed_sec", 0.0)) for r in rows)
    total_generated = sum(int(r.get("generated_tokens", 0)) for r in rows)
    avg_elapsed = total_elapsed / len(rows)
    avg_generated = total_generated / len(rows)
    tps = (total_generated / total_elapsed) if total_elapsed > 0 else 0.0
    return {
        "method": method,
        "samples": len(rows),
        "avg_elapsed_sec": round(avg_elapsed, 4),
        "avg_generated_tokens": round(avg_generated, 2),
        "tokens_per_sec": round(tps, 3),
    }


def run_one_method(
    api: "OracleKVProjectAPI",
    method: str,
    method_configs: dict[str, dict[str, float | int]],
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    result = api.evaluate(
        samples=samples,
        methods=[method],
        method_configs=method_configs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_on_eos=True,
        max_input_tokens=args.max_input_tokens,
        save_step_trace=args.save_step_trace,
    )
    return result.get("results_by_method", {}).get(method.lower(), [])


def _build_child_command(args: argparse.Namespace, method: str) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve())]

    cmd.extend(["--model-path", args.model_path])
    if args.input_jsonl:
        cmd.extend(["--input-jsonl", args.input_jsonl])
    cmd.extend(["--dataset-format", args.dataset_format])
    cmd.extend(["--tau2-domain-dir", args.tau2_domain_dir])
    cmd.extend(["--tau2-split", args.tau2_split])
    if args.tau2_include_policy:
        cmd.append("--tau2-include-policy")
    cmd.extend(["--tau2-limit", str(args.tau2_limit)])

    cmd.extend(["--output-dir", args.output_dir])
    cmd.extend(["--methods", method])

    cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--top-p", str(args.top_p)])
    cmd.extend(["--device", args.device])
    cmd.extend(["--dtype", args.dtype])
    cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_input_tokens is not None:
        cmd.extend(["--max-input-tokens", str(args.max_input_tokens)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.allow_remote_files:
        cmd.append("--allow-remote-files")
    cmd.extend(["--seed", str(args.seed)])

    cmd.extend(["--streaming-sink-size", str(args.streaming_sink_size)])
    cmd.extend(["--streaming-local-window-size", str(args.streaming_local_window_size)])
    cmd.extend(["--h2o-sink-size", str(args.h2o_sink_size)])
    cmd.extend(["--h2o-local-window-size", str(args.h2o_local_window_size)])
    cmd.extend(["--h2o-heavy-hitter-size", str(args.h2o_heavy_hitter_size)])
    if args.save_step_trace:
        cmd.append("--save-step-trace")
    cmd.extend(["--attn-implementation", args.attn_implementation])

    cmd.append("--_child-run")
    return cmd


def run_methods_in_subprocesses(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.lower() for m in args.methods]

    print(f"Running methods sequentially in separate processes: {', '.join(methods)}")
    for method in methods:
        cmd = _build_child_command(args, method)
        print(f"[parent] starting method={method}")
        subprocess.run(cmd, check=True)
        print(f"[parent] finished method={method}")

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        rows = read_jsonl(output_dir / f"{method}.jsonl")
        all_rows.extend(rows)
        summary_rows.append(summarize_rows(method, rows))

    write_jsonl(output_dir / "all_results.jsonl", all_rows)
    write_jsonl(output_dir / "summary.jsonl", summary_rows)

    print(f"Output directory: {output_dir}")
    for row in summary_rows:
        print(
            f"[{row['method']}] samples={row['samples']} avg_elapsed_sec={row['avg_elapsed_sec']} "
            f"avg_generated_tokens={row['avg_generated_tokens']} tokens_per_sec={row['tokens_per_sec']}"
        )


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.seed)

    methods = [m.lower() for m in args.methods]
    if len(methods) > 1 and not args._child_run:
        run_methods_in_subprocesses(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, dataset_desc = load_samples_from_args(args)
    if not samples:
        raise ValueError(f"No samples found for dataset: {dataset_desc}")

    # Resolve attention implementation
    attn_impl = args.attn_implementation
    if attn_impl == "auto":
        method_set = {m.lower() for m in methods}
        if "h2o" in method_set:
            attn_impl = "eager"
        else:
            attn_impl = "sdpa"

    from src.api import OracleKVProjectAPI

    api = OracleKVProjectAPI(
        model_path=args.model_path,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        allow_remote_files=args.allow_remote_files,
        attn_implementation=attn_impl,
    )
    method_configs = build_method_configs(args)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, object]] = []
    for method in methods:
        rows = run_one_method(api, method, method_configs, samples, args)
        write_jsonl(output_dir / f"{method}.jsonl", rows)
        all_rows.extend(rows)
        summary_rows.append(summarize_rows(method, rows))

    write_jsonl(output_dir / "all_results.jsonl", all_rows)
    write_jsonl(output_dir / "summary.jsonl", summary_rows)

    print(f"Input dataset: {dataset_desc}")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {', '.join(methods)}")
    for row in summary_rows:
        print(
            f"[{row['method']}] samples={row['samples']} avg_elapsed_sec={row['avg_elapsed_sec']} "
            f"avg_generated_tokens={row['avg_generated_tokens']} tokens_per_sec={row['tokens_per_sec']}"
        )


if __name__ == "__main__":
    main()