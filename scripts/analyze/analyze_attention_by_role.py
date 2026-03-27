#!/usr/bin/env python3
"""
Attention-by-role analysis for multi-turn KV cache research.

For each simulation in tau2-bench, we:
  1. Build the chat context incrementally (turn by turn).
  2. At the start of each *assistant* turn (i.e., after all prior messages
     have been tokenized), run a prefill with output_attentions=True to
     extract the last-query attention vector (shape [seq_len]).
  3. Map each token position back to its originating message index + role.
  4. Aggregate attention by (role, turn_distance_from_current) and save
     summary CSVs + PNG plots.

Usage (single file):
  python3 scripts/analyze_attention_by_role.py \
      --model-path ./local_models/Qwen3.5-9B \
      --sim-file tau2-bench/data/simulations/stress_baseline_1x10_task10_1st.json \
      --device cuda:2

Usage (all four task10 files, GPU 2):
  python3 scripts/analyze_attention_by_role.py \
      --model-path ./local_models/Qwen3.5-9B \
      --sim-file \
          tau2-bench/data/simulations/stress_baseline_1x10_task10_1st.json \
          tau2-bench/data/simulations/stress_baseline_1x10_task10_2nd.json \
          tau2-bench/data/simulations/stress_baseline_1x10_task10_3rd.json \
          tau2-bench/data/simulations/stress_baseline_1x10_task10_4th.json \
      --device cuda:2 \
      --output-dir outputs/attention_task10

Parameters:
  --sim-file   one or more JSON files to analyse (space-separated after the flag)
  --max-sims   per-file limit on simulations (0 or omit = use all sims in each file)
  --max-turns  per-sim limit on assistant turns to analyse (0 = all turns)
  --device     cuda:N to target GPU N; "auto" picks cuda:0 if CUDA is available
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse per-role attention in multi-turn conversations.")
    p.add_argument("--model-path", default=str(ROOT / "local_models" / "Qwen3.5-9B"))
    p.add_argument(
        "--sim-file",
        nargs="+",
        default=[
            str(ROOT / "tau2-bench" / "data" / "simulations" / "stress_baseline_1x10_task10_1st.json"),
            str(ROOT / "tau2-bench" / "data" / "simulations" / "stress_baseline_1x10_task10_2nd.json"),
            str(ROOT / "tau2-bench" / "data" / "simulations" / "stress_baseline_1x10_task10_3rd.json"),
            str(ROOT / "tau2-bench" / "data" / "simulations" / "stress_baseline_1x10_task10_4th.json"),
        ],
        help="One or more simulation JSON files to analyse (space-separated).",
    )
    p.add_argument(
        "--max-sims", type=int, default=0,
        help="Per-file limit on simulations to analyse (0 = all sims in each file).",
    )
    p.add_argument(
        "--max-turns", type=int, default=0,
        help="Per-sim limit on assistant turns to analyse (0 = all turns).",
    )
    p.add_argument("--output-dir", default=str(ROOT / "outputs" / "attention_task10"))
    p.add_argument(
        "--device", default="cuda:2",
        help="Device to run on: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:2', etc.",
    )
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Role classification helper
# ─────────────────────────────────────────────────────────────────────────────
# tau2-bench assistant messages sometimes contain embedded "tool call" blocks
# such as:  [call_tool]\n[function_name]\n[param]: value\n
# We detect these to refine the assistant role into "assistant_tool_call".
# tau2-bench uses ```json {...}``` blocks for tool calls.
# Older formats ([call_tool], <tool_call>) kept as fallback.
TOOL_CALL_RE = re.compile(
    r'```json\s*\{|\[call_tool\]|\[function_call\]|<tool_call>',
    re.IGNORECASE,
)
TOOL_RESULT_RE = re.compile(
    r"I have retrieved|tool result|function result|<tool_response>|<function_results>",
    re.IGNORECASE,
)


def classify_role(role: str, content: str) -> str:
    """More granular role classification for plotting."""
    if role == "system":
        return "system"
    if role == "tool":
        return "tool_result"
    if role == "user":
        return "user_msg"
    if role == "assistant":
        if TOOL_CALL_RE.search(content or ""):
            return "asst_tool_call"
        return "asst_gen"
    return role


# ─────────────────────────────────────────────────────────────────────────────
# Span computation via incremental tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_messages(messages: list[dict]) -> tuple[list[dict], int]:
    """
    tau2-bench conversations start with an assistant greeting at index 0.
    Qwen3.5's chat template requires conversations to start with
    system/user.  We remap the FIRST assistant message → system so that
    the template is satisfied.  Also removes any trailing incomplete pair
    (e.g. a lone assistant message with no following user).

    Returns (normalised_messages, greeting_remapped: 0 or 1).
    """
    normalised = []
    greeting_remapped = 0
    for i, m in enumerate(messages):
        if i == 0 and m.get("role") == "assistant":
            # Promote the greeting to a system message.
            normalised.append({"role": "system", "content": m.get("content", "") or ""})
            greeting_remapped = 1
        else:
            normalised.append(m)
    return normalised, greeting_remapped


def _safe_apply_template(tokenizer, messages: list[dict]) -> list[int] | None:
    """
    Apply chat template (tokenize=False then encode) after normalising the list.
    Returns token ids as list[int], or None on failure.
    """
    if not messages:
        return []
    normalised, _ = _normalise_messages(messages)
    try:
        text = tokenizer.apply_chat_template(
            normalised, tokenize=False, add_generation_prompt=False
        )
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return None


def compute_message_spans(
    tokenizer,
    messages: list[dict],
) -> list[tuple[int, int, str, str]]:
    """
    Returns [(start_tok, end_tok, role, fine_role), ...] for each original message.

    The first assistant message (greeting) is remapped to 'system' internally
    for tokenisation, but its span still carries the original role.
    Span for message k = [prev_len, len(tokenise(messages[:k+1]))].
    """
    spans: list[tuple[int, int, str, str]] = []
    prev_len = 0
    for k in range(len(messages)):
        ids = _safe_apply_template(tokenizer, messages[: k + 1])
        if ids is None:
            # Fallback: approximate by content length
            content = messages[k].get("content", "") or ""
            tok_ids = tokenizer.encode(content, add_special_tokens=False)
            curr_len = prev_len + len(tok_ids)
        else:
            curr_len = len(ids)
        role = messages[k].get("role", "unknown")
        content = messages[k].get("content", "") or ""
        fine_role = classify_role(role, content)
        # The first assistant message is functionally a greeting / system token
        if k == 0 and role == "assistant":
            fine_role = "asst_greeting"
        if curr_len > prev_len:
            spans.append((prev_len, curr_len, role, fine_role))
        prev_len = curr_len
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_last_query_attention(
    model,
    tokenizer,
    token_ids: list[int],
    device: str,
) -> np.ndarray:
    """
    Run a full prefill with output_attentions=True and return the averaged
    last-query attention vector (shape [seq_len]).

    We average over all heads AND all layers to get a single importance score
    per past token position.
    """
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, output_attentions=True)

    # out.attentions: tuple of [batch, heads, seq, seq] per layer
    seq_len = len(token_ids)
    acc = np.zeros(seq_len, dtype=np.float64)
    valid = 0
    for layer_attn in out.attentions:
        if layer_attn is None:
            continue
        # layer_attn: [1, heads, seq, seq]  — take last-query row
        row = layer_attn[0, :, -1, :].float().cpu().numpy()  # [heads, seq]
        acc += row.mean(axis=0)
        valid += 1

    if valid == 0:
        return acc
    return acc / valid


# ─────────────────────────────────────────────────────────────────────────────
# Per-simulation analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_simulation(
    sim: dict,
    model,
    tokenizer,
    device: str,
    max_turns: int,
) -> list[dict]:
    """
    Returns a list of records with columns:
      sim_id, turn_idx, total_turns, turn_dist, fine_role, attn_mean, attn_sum,
      token_count, seq_len_at_turn
    """
    messages = sim.get("messages", []) or []
    sim_id = sim.get("id", "unknown")[:12]

    # Identify assistant turn indices (the turns we'll analyse)
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        return []

    records = []

    # Analyse up to max_turns assistant turns (0 means all turns).
    # Taking all turns provides full coverage of both early and late conversation
    # phases; taking only the last N captures the rich-history phase only.
    if max_turns and max_turns > 0:
        turns_to_analyse = assistant_indices[-max_turns:]
    else:
        turns_to_analyse = assistant_indices

    for turn_idx in turns_to_analyse:
        # Context = all messages UP TO AND INCLUDING this assistant turn
        context_msgs = messages[:turn_idx + 1]

        # Compute token spans
        spans = compute_message_spans(tokenizer, context_msgs)
        if not spans:
            continue

        # Get full token IDs for this context
        token_ids = _safe_apply_template(tokenizer, context_msgs)
        if token_ids is None:
            print(f"  [warn] tokenize failed at turn {turn_idx}", file=sys.stderr)
            continue

        seq_len = len(token_ids)
        if seq_len < 4:
            continue

        # Extract last-query attention (query = last token of this turn)
        try:
            attn = extract_last_query_attention(model, tokenizer, token_ids, device)
        except Exception as e:
            print(f"  [warn] attention failed at turn {turn_idx} seq_len={seq_len}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue

        # turn_dist: 0 = current assistant turn, 1 = one turn ago, etc.
        # We count assistant turns as the "distance unit"
        asst_turn_pos = assistant_indices.index(turn_idx)

        for msg_i, (start, end, role, fine_role) in enumerate(spans):
            if end > seq_len:
                end = seq_len
            if start >= end:
                continue
            span_attn = attn[start:end]
            # Distance in message index from current (turn_idx)
            msg_dist = turn_idx - msg_i  # higher = further in the past
            # Distance in assistant-turns
            asst_turns_before = sum(
                1 for j in range(msg_i, turn_idx)
                if messages[j].get("role") == "assistant"
            )
            records.append(
                {
                    "sim_id": sim_id,
                    "context_turn": turn_idx,
                    "total_msgs": len(messages),
                    "msg_idx": msg_i,
                    "msg_dist": msg_dist,
                    "asst_turns_dist": asst_turns_before,
                    "role": role,
                    "fine_role": fine_role,
                    "attn_mean": float(span_attn.mean()),
                    "attn_sum": float(span_attn.sum()),
                    "token_count": int(end - start),
                    "seq_len": seq_len,
                    # normalised so sum over all spans = 1
                    "attn_frac": float(span_attn.sum() / (attn.sum() + 1e-12)),
                }
            )

        total_attn = sum(r["attn_sum"] for r in records if r["context_turn"] == turn_idx
                         and r["sim_id"] == sim_id)
        print(
            f"  turn={turn_idx:2d}  seq_len={seq_len:5d}  "
            f"spans={len(spans):2d}  attn_sum={total_attn:.4f}"
        )

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "user_msg":       "#2196F3",   # blue
    "asst_gen":       "#9C27B0",   # purple
    "asst_tool_call": "#FF9800",   # orange
    "asst_greeting":  "#795548",   # brown (initial greeting / system-like)
    "tool_result":    "#4CAF50",   # green
    "system":         "#F44336",   # red
}

ROLE_ORDER = ["system", "asst_greeting", "user_msg", "asst_tool_call", "asst_gen", "tool_result"]


def plot_attn_decay_by_role(df: pd.DataFrame, out_dir: Path) -> None:
    """Line plot: attention fraction vs asst_turns_dist, grouped by fine_role."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: mean attention (mean over span tokens) ---
    ax = axes[0]
    grouped = df.groupby(["asst_turns_dist", "fine_role"])["attn_mean"].mean().reset_index()
    for role in ROLE_ORDER:
        sub = grouped[grouped["fine_role"] == role]
        if sub.empty:
            continue
        ax.plot(
            sub["asst_turns_dist"], sub["attn_mean"],
            marker="o", label=role, color=ROLE_COLORS.get(role, "grey"),
        )
    ax.set_xlabel("Assistant-turn distance from current (0=current, larger=older)")
    ax.set_ylabel("Mean per-token attention score")
    ax.set_title("Per-token attention vs turn distance")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.invert_xaxis()

    # --- Right: fraction of total attention ---
    ax2 = axes[1]
    grouped2 = df.groupby(["asst_turns_dist", "fine_role"])["attn_frac"].mean().reset_index()
    for role in ROLE_ORDER:
        sub = grouped2[grouped2["fine_role"] == role]
        if sub.empty:
            continue
        ax2.plot(
            sub["asst_turns_dist"], sub["attn_frac"],
            marker="s", label=role, color=ROLE_COLORS.get(role, "grey"),
        )
    ax2.set_xlabel("Assistant-turn distance from current")
    ax2.set_ylabel("Fraction of total attention")
    ax2.set_title("Attention fraction vs turn distance")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.invert_xaxis()

    plt.tight_layout()
    out_path = out_dir / "attn_decay_by_role.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


def plot_attn_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: average attention for each (asst_dist, role) cell."""
    pivot = (
        df.groupby(["asst_turns_dist", "fine_role"])["attn_frac"]
        .mean()
        .unstack(fill_value=0.0)
    )
    # Reorder columns
    cols = [c for c in ROLE_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.5), max(4, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"dist={i}" for i in pivot.index])
    ax.set_title("Avg attention fraction  (row=turn distance, col=token role)")
    plt.colorbar(im, ax=ax, label="attn_frac")
    plt.tight_layout()
    out_path = out_dir / "attn_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


def plot_role_share_per_turn(df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar: for each context_turn, show role share of total attention."""
    # Average across simulations at same context turn length
    pivot = (
        df.groupby(["context_turn", "fine_role"])["attn_frac"]
        .sum()
        .unstack(fill_value=0.0)
    )
    cols = [c for c in ROLE_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.8), 5))
    bottom = np.zeros(len(pivot))
    for role in cols:
        vals = pivot[role].values
        ax.bar(pivot.index, vals, bottom=bottom,
               color=ROLE_COLORS.get(role, "grey"), label=role)
        bottom += vals
    ax.set_xlabel("Context turn index (=num preceding msgs)")
    ax.set_ylabel("Summed attention fraction")
    ax.set_title("Attention share by role at each prefill point")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out_path = out_dir / "role_share_per_turn.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


def plot_token_count_vs_attn(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: token count of message vs its normalised mean attention."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for role in ROLE_ORDER:
        sub = df[df["fine_role"] == role]
        if sub.empty:
            continue
        ax.scatter(
            sub["token_count"], sub["attn_mean"],
            alpha=0.4, label=role, color=ROLE_COLORS.get(role, "grey"), s=20,
        )
    ax.set_xlabel("Token count of message span")
    ax.set_ylabel("Mean attention score per token")
    ax.set_title("Message length vs per-token attention")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "token_count_vs_attn.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY: average attention fraction by fine_role")
    print("=" * 70)
    by_role = df.groupby("fine_role").agg(
        attn_frac_mean=("attn_frac", "mean"),
        attn_frac_std=("attn_frac", "std"),
        attn_mean_mean=("attn_mean", "mean"),
        token_count_mean=("token_count", "mean"),
        n_spans=("attn_frac", "count"),
    ).sort_values("attn_frac_mean", ascending=False)
    print(by_role.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY: attention fraction decay by (fine_role, asst_turns_dist)")
    print("(turn_dist=0 is current assistant turn, higher = older)")
    print("=" * 70)
    by_dist = (
        df[df["asst_turns_dist"] <= 6]
        .groupby(["asst_turns_dist", "fine_role"])["attn_frac"]
        .mean()
        .unstack(fill_value=0.0)
    )
    cols = [c for c in ROLE_ORDER if c in by_dist.columns]
    print(by_dist[cols].to_string())

    print("\n" + "=" * 70)
    print("INSIGHTS")
    print("=" * 70)
    # Compute ratio: attn at dist=0 vs dist=2 for user_msg
    for role in ["user_msg", "asst_gen", "asst_tool_call", "tool_result"]:
        sub = df[df["fine_role"] == role].groupby("asst_turns_dist")["attn_frac"].mean()
        if sub.empty:
            continue
        d0 = sub.get(0, None)
        d2 = sub.get(2, None)
        d4 = sub.get(4, None)
        parts = [f"  {role:<20}"]
        if d0 is not None:
            parts.append(f"dist0={d0:.4f}")
        if d2 is not None:
            ratio = d0 / (d2 + 1e-12) if d0 is not None else float("nan")
            parts.append(f"dist2={d2:.4f} (drop ratio={ratio:.1f}x)")
        if d4 is not None:
            ratio = d0 / (d4 + 1e-12) if d0 is not None else float("nan")
            parts.append(f"dist4={d4:.4f} (drop ratio={ratio:.1f}x)")
        print("  ".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # Validate device string early to give a clear error message.
    if device_str.startswith("cuda"):
        parts = device_str.split(":")
        gpu_idx = int(parts[1]) if len(parts) > 1 else 0
        n_gpus = torch.cuda.device_count()
        if gpu_idx >= n_gpus:
            print(f"[error] Requested {device_str} but only {n_gpus} GPU(s) visible."
                  f" Available: cuda:0 .. cuda:{n_gpus-1}", file=sys.stderr)
            sys.exit(1)
    print(f"[info] device={device_str}  model={args.model_path}")

    dtype = torch.bfloat16 if (device_str.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
    if args.dtype not in ("auto", ""):
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(args.dtype, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",  # required for output_attentions=True
    )
    model.eval().to(device_str)
    print(f"[info] model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # ── 2. Load simulations from all specified files ──────────────────────
    sim_files = [Path(p) for p in args.sim_file]
    print(f"[info] {len(sim_files)} sim file(s) to process:")
    for sf in sim_files:
        print(f"       {sf}")

    # ── 3. Run analysis ────────────────────────────────────────────────────
    all_records: list[dict] = []
    for file_i, sim_path in enumerate(sim_files):
        print(f"\n{'='*60}")
        print(f"[file {file_i+1}/{len(sim_files)}] {sim_path.name}")
        print(f"{'='*60}")
        with open(sim_path) as f:
            data = json.load(f)
        simulations = data.get("simulations", [])
        if args.max_sims and args.max_sims > 0:
            simulations = simulations[: args.max_sims]
        print(f"[info] analysing {len(simulations)} simulations")

        for sim_i, sim in enumerate(simulations):
            sim_id_short = sim.get("id", "?")[:8]
            task_id = sim.get("task_id", "?")
            msgs_total = len(sim.get("messages", []) or [])
            print(f"\n[sim {sim_i+1}/{len(simulations)}]  id={sim_id_short}  "
                  f"task={task_id}  msgs={msgs_total}")
            recs = analyse_simulation(
                sim=sim,
                model=model,
                tokenizer=tokenizer,
                device=device_str,
                max_turns=args.max_turns,
            )
            # Tag each record with the source filename for traceability
            src_tag = sim_path.stem
            for r in recs:
                r["src_file"] = src_tag
            all_records.extend(recs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_records:
        print("[error] No records collected. Check model/data paths.")
        sys.exit(1)

    # ── 4. Save CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    csv_path = out_dir / "attention_by_role.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[data] {len(df)} records saved → {csv_path}")

    # ── 5. Print summary ───────────────────────────────────────────────────
    print_summary(df)

    # ── 6. Save plots ──────────────────────────────────────────────────────
    print("\n[plots] generating...")
    plot_attn_decay_by_role(df, out_dir)
    plot_attn_heatmap(df, out_dir)
    plot_role_share_per_turn(df, out_dir)
    plot_token_count_vs_attn(df, out_dir)
    print(f"\n[done] all outputs in {out_dir}/")


if __name__ == "__main__":
    main()
