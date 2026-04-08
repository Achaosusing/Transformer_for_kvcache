#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILENAME_RE = re.compile(
    r"^h2o_session_ablation_(?P<variant>stateless|session_decay)_(?P<trials>\d+)x(?P<tasks>\d+)_s(?P<sink>\d+)_w(?P<window>\d+)_h(?P<heavy>\d+)(?:_(?P<tag>.+))?$"
)
VARIANT_ORDER = ["stateless", "session_decay"]
VARIANT_LABELS = {
    "stateless": "Stateless",
    "session_decay": "Session + Decay",
}


def parse_filename(path: Path) -> dict[str, object] | None:
    match = FILENAME_RE.match(path.stem)
    if match is None:
        return None
    meta = match.groupdict()
    return {
        "variant": meta["variant"],
        "num_trials": int(meta["trials"]),
        "num_tasks": int(meta["tasks"]),
        "sink": int(meta["sink"]),
        "window": int(meta["window"]),
        "heavy": int(meta["heavy"]),
        "tag": meta.get("tag"),
    }


def load_records(
    sim_dir: Path,
    tag: str | None,
    tag_contains: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_records: list[dict[str, object]] = []
    task_records: list[dict[str, object]] = []

    for path in sorted(sim_dir.glob("h2o_session_ablation_*.json")):
        meta = parse_filename(path)
        if meta is None:
            continue
        if tag is not None and meta["tag"] != tag:
            continue
        if tag_contains is not None and tag_contains not in str(meta.get("tag") or ""):
            continue

        payload = json.loads(path.read_text())
        sims = payload.get("simulations", [])
        rewards = [float(sim.get("reward_info", {}).get("reward", math.nan)) for sim in sims]
        rewards = [reward for reward in rewards if not math.isnan(reward)]
        durations = [float(sim.get("duration", math.nan)) for sim in sims]
        durations = [duration for duration in durations if not math.isnan(duration)]
        timeout_rate = (
            sum(sim.get("termination_reason") == "task_timeout" for sim in sims) / len(sims)
            if sims else math.nan
        )
        avg_turns = (
            sum(len(sim.get("messages", [])) for sim in sims) / len(sims)
            if sims else math.nan
        )

        run_records.append(
            {
                **meta,
                "source_file": path.name,
                "n": len(sims),
                "success_rate": sum(rewards) / len(rewards) if rewards else math.nan,
                "avg_duration": sum(durations) / len(durations) if durations else math.nan,
                "timeout_rate": timeout_rate,
                "avg_turns": avg_turns,
            }
        )

        for sim in sims:
            task_records.append(
                {
                    **meta,
                    "source_file": path.name,
                    "trial": int(sim.get("trial", 0)),
                    "task_id": str(sim.get("task_id")),
                    "reward": float(sim.get("reward_info", {}).get("reward", math.nan)),
                    "duration": float(sim.get("duration", math.nan)),
                    "termination_reason": sim.get("termination_reason"),
                }
            )

    return pd.DataFrame.from_records(run_records), pd.DataFrame.from_records(task_records)


def build_paired_deltas(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame()

    baseline = task_df[task_df["variant"] == "stateless"]
    if baseline.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    baseline_keyed = baseline.set_index(["trial", "task_id", "num_trials", "num_tasks", "sink", "window", "heavy", "tag"])

    for variant in [item for item in VARIANT_ORDER if item != "stateless"]:
        cur_df = task_df[task_df["variant"] == variant]
        for _, row in cur_df.iterrows():
            key = (
                row["trial"],
                row["task_id"],
                row["num_trials"],
                row["num_tasks"],
                row["sink"],
                row["window"],
                row["heavy"],
                row["tag"],
            )
            if key not in baseline_keyed.index:
                continue
            ref = baseline_keyed.loc[key]
            rows.append(
                {
                    "variant": variant,
                    "task_id": row["task_id"],
                    "reward_delta": float(row["reward"]) - float(ref["reward"]),
                    "duration_delta": float(row["duration"]) - float(ref["duration"]),
                    "timeout_delta": float(row["termination_reason"] == "task_timeout") - float(ref["termination_reason"] == "task_timeout"),
                }
            )
    return pd.DataFrame.from_records(rows)


def save_csv(run_df: pd.DataFrame, task_df: pd.DataFrame, paired_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(output_dir / "h2o_session_ablation_runs.csv", index=False)
    task_df.to_csv(output_dir / "h2o_session_ablation_tasks.csv", index=False)
    paired_df.to_csv(output_dir / "h2o_session_ablation_paired_deltas.csv", index=False)


def save_summary_plot(run_df: pd.DataFrame, output_dir: Path) -> None:
    if run_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = run_df.copy()
    ordered["variant"] = pd.Categorical(ordered["variant"], categories=VARIANT_ORDER, ordered=True)
    ordered = ordered.sort_values("variant")
    labels = [VARIANT_LABELS[str(variant)] for variant in ordered["variant"]]
    colors = ["#7f8c8d", "#2ecc71"][: len(labels)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(labels, ordered["success_rate"], color=colors)
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, ordered["avg_duration"], color=colors)
    axes[1].set_title("Avg Duration (s)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(labels, ordered["timeout_rate"], color=colors)
    axes[2].set_title("Timeout Rate")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("H2O Session Ablation")
    fig.tight_layout()
    fig.savefig(output_dir / "h2o_session_ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_delta_plot(paired_df: pd.DataFrame, output_dir: Path) -> None:
    if paired_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = paired_df.groupby("variant", as_index=False).agg(
        reward_delta_mean=("reward_delta", "mean"),
        duration_delta_mean=("duration_delta", "mean"),
    )
    grouped["variant"] = pd.Categorical(
        grouped["variant"],
        categories=[item for item in VARIANT_ORDER if item != "stateless"],
        ordered=True,
    )
    grouped = grouped.sort_values("variant")
    labels = [VARIANT_LABELS[str(variant)] for variant in grouped["variant"]]
    colors = ["#2ecc71"][: len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, grouped["reward_delta_mean"], color=colors)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Reward Delta vs Stateless")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, grouped["duration_delta_mean"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Duration Delta vs Stateless")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "h2o_session_ablation_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(run_df: pd.DataFrame, paired_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT_h2o_session_ablation.md"
    lines: list[str] = ["# H2O Session Ablation Report\n"]

    if run_df.empty:
        lines.append("No matching simulation files found.\n")
        report_path.write_text("".join(lines), encoding="utf-8")
        return

    ordered = run_df.copy()
    ordered["variant"] = pd.Categorical(ordered["variant"], categories=VARIANT_ORDER, ordered=True)
    ordered = ordered.sort_values("variant")

    lines.append("## Run Summary\n")
    lines.append("| variant | success_rate | avg_duration | timeout_rate | avg_turns | n |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for _, row in ordered.iterrows():
        lines.append(
            f"| {VARIANT_LABELS[str(row['variant'])]} | {row['success_rate']:.3f} | {row['avg_duration']:.2f} | {row['timeout_rate']:.3f} | {row['avg_turns']:.2f} | {int(row['n'])} |\n"
        )

    if not paired_df.empty:
        lines.append("\n## Paired Delta vs Stateless\n")
        grouped = paired_df.groupby("variant", as_index=False).agg(
            reward_delta_mean=("reward_delta", "mean"),
            duration_delta_mean=("duration_delta", "mean"),
            timeout_delta_mean=("timeout_delta", "mean"),
        )
        lines.append("| variant | reward_delta_mean | duration_delta_mean | timeout_delta_mean |\n")
        lines.append("|---|---:|---:|---:|\n")
        for _, row in grouped.iterrows():
            lines.append(
                f"| {VARIANT_LABELS[str(row['variant'])]} | {row['reward_delta_mean']:+.3f} | {row['duration_delta_mean']:+.2f} | {row['timeout_delta_mean']:+.3f} |\n"
            )

    report_path.write_text("".join(lines), encoding="utf-8")


def print_summary(run_df: pd.DataFrame, paired_df: pd.DataFrame) -> None:
    if run_df.empty:
        print("[warn] No matching H2O session ablation files found.")
        return

    ordered = run_df.copy()
    ordered["variant"] = pd.Categorical(ordered["variant"], categories=VARIANT_ORDER, ordered=True)
    ordered = ordered.sort_values("variant")

    print("\nH2O session ablation summary")
    print("=" * 60)
    print(f"{'variant':<18} {'success':>8} {'duration(s)':>12} {'timeout':>8} {'n':>4}")
    for _, row in ordered.iterrows():
        print(
            f"{VARIANT_LABELS[str(row['variant'])]:<18} {row['success_rate']:>8.3f} {row['avg_duration']:>12.2f} {row['timeout_rate']:>8.3f} {int(row['n']):>4}"
        )

    if not paired_df.empty:
        print("\nPaired deltas vs Stateless")
        print("-" * 60)
        grouped = paired_df.groupby("variant", as_index=False).agg(
            reward_delta_mean=("reward_delta", "mean"),
            duration_delta_mean=("duration_delta", "mean"),
            timeout_delta_mean=("timeout_delta", "mean"),
        )
        for _, row in grouped.iterrows():
            print(
                f"{VARIANT_LABELS[str(row['variant'])]:<18} reward={row['reward_delta_mean']:+.3f} duration={row['duration_delta_mean']:+.2f}s timeout={row['timeout_delta_mean']:+.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze H2O session ablation results.")
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        default=Path("tau2-bench/data/simulations"),
        help="Directory containing tau2 simulation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/h2o_session_ablation"),
        help="Directory to save CSVs, figures, and report.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional save tag suffix to filter matching ablation files.",
    )
    parser.add_argument(
        "--tag-contains",
        type=str,
        default=None,
        help="Optional substring filter for batch sweep tags.",
    )
    args = parser.parse_args()

    run_df, task_df = load_records(args.simulations_dir, args.tag, args.tag_contains)
    paired_df = build_paired_deltas(task_df)
    print_summary(run_df, paired_df)
    save_csv(run_df, task_df, paired_df, args.output_dir)
    save_summary_plot(run_df, args.output_dir)
    save_delta_plot(paired_df, args.output_dir)
    write_report(run_df, paired_df, args.output_dir)


if __name__ == "__main__":
    main()