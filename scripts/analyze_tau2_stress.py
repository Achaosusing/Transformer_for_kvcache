#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator


BASELINE_RE = re.compile(r"stress_baseline_1x\d+(?:_(?P<repeat>\d+(?:st|nd|rd|th)))?$")
STREAMING_RE = re.compile(
    r"stress_streamingllm_1x\d+_(?P<sink>\d+)_(?P<window>\d+)(?:_(?P<repeat>\d+(?:st|nd|rd|th)))?$"
)
H2O_RE = re.compile(
    r"stress_h2o_1x\d+_(?P<sink>\d+)_(?P<window>\d+)_(?P<heavy>\d+)(?:_(?P<repeat>\d+(?:st|nd|rd|th)))?$"
)
DEFAULT_STABLE_TASK_IDS = ["0", "1", "3", "4", "5", "6", "10", "13", "26", "28"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tau2 stress-test simulation JSON files and draw summary plots."
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        help="Optional task IDs to filter before computing reward curves.",
    )
    parser.add_argument(
        "--use-stable-task10",
        action="store_true",
        help="Analyze only the 10 tasks: 0 1 3 4 5 6 10 13 26 28.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default=None,
        help="Optional output directory name under outputs/.",
    )
    return parser.parse_args()


def resolve_task_ids(args: argparse.Namespace) -> list[str] | None:
    if args.task_ids and args.use_stable_task10:
        raise ValueError("Use either --task-ids or --use-stable-task10, not both.")
    if args.use_stable_task10:
        return DEFAULT_STABLE_TASK_IDS.copy()
    if args.task_ids:
        return [str(task_id) for task_id in args.task_ids]
    return None


def detect_config(path: Path) -> dict | None:
    stem = path.stem

    match = BASELINE_RE.fullmatch(stem)
    if match:
        return {
            "method": "baseline",
            "sink": pd.NA,
            "window": pd.NA,
            "heavy": pd.NA,
            "repeat": match.group("repeat") or "run1",
            "total_budget": pd.NA,
        }

    match = STREAMING_RE.fullmatch(stem)
    if match:
        sink = int(match.group("sink"))
        window = int(match.group("window"))
        return {
            "method": "streamingllm",
            "sink": sink,
            "window": window,
            "heavy": 0,
            "repeat": match.group("repeat") or "run1",
            "total_budget": sink + window,
        }

    match = H2O_RE.fullmatch(stem)
    if match:
        sink = int(match.group("sink"))
        window = int(match.group("window"))
        heavy = int(match.group("heavy"))
        return {
            "method": "h2o",
            "sink": sink,
            "window": window,
            "heavy": heavy,
            "repeat": match.group("repeat") or "run1",
            "total_budget": sink + window + heavy,
        }

    return None


def build_run_record(config: dict, path: Path, simulations: list[dict]) -> dict:
    rewards = [simulation["reward_info"]["reward"] for simulation in simulations]
    durations = [simulation.get("duration", 0.0) for simulation in simulations]
    turns = [len(simulation.get("messages", [])) for simulation in simulations]
    terminations = Counter(simulation["termination_reason"] for simulation in simulations)

    return {
        **config,
        "source_file": path.name,
        "num_tasks": len(simulations),
        "success_rate": sum(rewards) / len(rewards),
        "timeout_rate": terminations.get("task_timeout", 0) / len(simulations),
        "user_stop_rate": terminations.get("user_stop", 0) / len(simulations),
        "avg_duration_s": sum(durations) / len(durations),
        "avg_turns": sum(turns) / len(turns),
    }


def load_results(
    simulations_dir: Path, task_ids: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_records = []
    simulation_records = []
    task_id_filter = set(task_ids) if task_ids is not None else None

    for path in sorted(simulations_dir.glob("stress_*.json")):
        config = detect_config(path)
        if config is None:
            continue

        payload = json.loads(path.read_text())
        simulations = payload["simulations"]
        if task_id_filter is not None:
            simulations = [simulation for simulation in simulations if str(simulation["task_id"]) in task_id_filter]
        if not simulations:
            continue

        run_records.append(build_run_record(config, path, simulations))
        for simulation in simulations:
            simulation_records.append(
                {
                    **config,
                    "source_file": path.name,
                    "task_id": str(simulation["task_id"]),
                    "reward": float(simulation["reward_info"]["reward"]),
                    "termination_reason": simulation["termination_reason"],
                    "duration_s": float(simulation.get("duration", 0.0)),
                    "turns": len(simulation.get("messages", [])),
                }
            )

    if not run_records:
        raise FileNotFoundError(f"No stress result files found in {simulations_dir}")

    sim_df = pd.DataFrame.from_records(simulation_records)
    raw_df = pd.DataFrame.from_records(run_records)
    numeric_columns = [
        "sink",
        "window",
        "heavy",
        "total_budget",
        "reward",
        "duration_s",
        "turns",
        "success_rate",
        "timeout_rate",
        "user_stop_rate",
        "avg_duration_s",
        "avg_turns",
    ]
    for column in numeric_columns:
        if column in sim_df.columns:
            sim_df[column] = pd.to_numeric(sim_df[column], errors="coerce")
        if column in raw_df.columns:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    agg_df = (
        raw_df.groupby(["method", "sink", "window", "heavy", "total_budget"], dropna=False)
        .agg(
            runs=("source_file", "count"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            timeout_rate_mean=("timeout_rate", "mean"),
            timeout_rate_std=("timeout_rate", "std"),
            user_stop_rate_mean=("user_stop_rate", "mean"),
            avg_duration_s_mean=("avg_duration_s", "mean"),
            avg_turns_mean=("avg_turns", "mean"),
            num_tasks=("num_tasks", "max"),
        )
        .reset_index()
        .sort_values(["method", "total_budget", "window", "heavy"], na_position="last")
    )
    agg_df[["success_rate_std", "timeout_rate_std"]] = agg_df[
        ["success_rate_std", "timeout_rate_std"]
    ].fillna(0.0)
    return sim_df, raw_df, agg_df


def build_best_h2o_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    h2o_df = agg_df[agg_df["method"] == "h2o"].copy()
    if h2o_df.empty:
        return h2o_df

    h2o_df = h2o_df.sort_values(
        ["total_budget", "success_rate_mean", "timeout_rate_mean", "window", "heavy"],
        ascending=[True, False, True, True, True],
    )
    best_h2o_df = h2o_df.drop_duplicates(subset=["total_budget"], keep="first")
    return best_h2o_df.sort_values("total_budget")


def build_task_summary(sim_df: pd.DataFrame) -> pd.DataFrame:
    task_summary_df = (
        sim_df.assign(task_timeout=(sim_df["termination_reason"] == "task_timeout").astype(float))
        .groupby(["task_id", "method"], dropna=False)
        .agg(
            runs=("reward", "count"),
            reward_mean=("reward", "mean"),
            timeout_rate=("task_timeout", "mean"),
            avg_duration_s=("duration_s", "mean"),
            avg_turns=("turns", "mean"),
        )
        .reset_index()
    )
    task_summary_df = task_summary_df.sort_values(
        ["task_id", "method"],
        key=lambda series: series.map(lambda value: int(value) if str(value).isdigit() else value),
    )
    return task_summary_df


def detect_kink(curve_df: pd.DataFrame) -> dict | None:
    curve_df = curve_df.dropna(subset=["total_budget", "success_rate_mean"]).sort_values(
        "total_budget", ascending=False
    )
    if len(curve_df) < 2:
        return None

    curve_df = curve_df.reset_index(drop=True)
    drops = curve_df["success_rate_mean"] - curve_df["success_rate_mean"].shift(-1)
    drops = drops.iloc[:-1]
    positive_drops = drops[drops > 0]
    peak_row = curve_df.loc[curve_df["success_rate_mean"].idxmax()]
    if positive_drops.empty:
        return {
            "plateau_success": float(curve_df["success_rate_mean"].max()),
            "peak_budget_min": int(peak_row["total_budget"]),
            "drop": 0.0,
            "high_budget": None,
            "low_budget": None,
        }

    idx = int(positive_drops.idxmax())
    high_row = curve_df.iloc[idx]
    low_row = curve_df.iloc[idx + 1]
    return {
        "plateau_success": float(curve_df["success_rate_mean"].max()),
        "peak_budget_min": int(peak_row["total_budget"]),
        "drop": float(positive_drops.loc[idx]),
        "high_budget": int(high_row["total_budget"]),
        "low_budget": int(low_row["total_budget"]),
    }


def write_csv_outputs(
    sim_df: pd.DataFrame, raw_df: pd.DataFrame, agg_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_h2o_df = build_best_h2o_df(agg_df)
    task_summary_df = build_task_summary(sim_df)

    sim_df.sort_values(["task_id", "method", "total_budget", "window", "heavy", "source_file"], na_position="last").to_csv(
        output_dir / "stress_raw_simulations.csv", index=False
    )
    raw_df.sort_values(["method", "total_budget", "window", "heavy", "source_file"], na_position="last").to_csv(
        output_dir / "stress_raw_runs.csv", index=False
    )
    agg_df.to_csv(output_dir / "stress_aggregated_configs.csv", index=False)
    best_h2o_df.to_csv(output_dir / "stress_best_h2o_by_budget.csv", index=False)
    task_summary_df.to_csv(output_dir / "stress_task_summary.csv", index=False)
    return best_h2o_df


def add_error_band(ax: plt.Axes, df: pd.DataFrame, value_col: str, std_col: str, color: str) -> None:
    if df.empty:
        return
    lower = (df[value_col] - df[std_col]).clip(lower=0.0)
    upper = (df[value_col] + df[std_col]).clip(upper=1.0)
    ax.fill_between(df["total_budget"], lower, upper, color=color, alpha=0.12)


def plot_success_vs_budget(
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    best_h2o_df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    baseline_runs = raw_df[raw_df["method"] == "baseline"]
    baseline_mean = baseline_runs["success_rate"].mean()

    streaming_df = agg_df[agg_df["method"] == "streamingllm"].sort_values("total_budget")
    h2o_df = agg_df[agg_df["method"] == "h2o"].sort_values("total_budget")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    heavy_levels = sorted(h2o_df["heavy"].dropna().unique())
    heavy_palette = dict(zip(heavy_levels, sns.color_palette("crest", n_colors=max(len(heavy_levels), 1))))

    for heavy in heavy_levels:
        subset = h2o_df[h2o_df["heavy"] == heavy]
        ax.scatter(
            subset["total_budget"],
            subset["success_rate_mean"],
            color=heavy_palette[heavy],
            alpha=0.65,
            s=70,
            label=f"H2O configs (heavy={int(heavy)})",
        )

    stream_color = "#D97706"
    h2o_color = "#2563EB"
    baseline_color = "#374151"

    ax.plot(
        streaming_df["total_budget"],
        streaming_df["success_rate_mean"],
        color=stream_color,
        marker="o",
        linewidth=2.5,
        label="StreamingLLM",
    )
    add_error_band(ax, streaming_df, "success_rate_mean", "success_rate_std", stream_color)

    ax.plot(
        best_h2o_df["total_budget"],
        best_h2o_df["success_rate_mean"],
        color=h2o_color,
        marker="o",
        linewidth=2.8,
        label="H2O best per budget",
    )
    ax.axhline(
        baseline_mean,
        color=baseline_color,
        linestyle="--",
        linewidth=2.0,
        label=f"Baseline (unbounded) = {baseline_mean:.3f}",
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.set_ylim(0.0, 1.05 if baseline_mean > 0.5 else 0.42)
    ax.set_xlabel("Total KV cache budget (sink + window + heavy)")
    ax.set_ylabel("Pass rate / average reward")
    ax.set_title(f"tau2 stress test{title_suffix}: success rate vs KV cache budget")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "stress_budget_success.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_timeout_vs_budget(
    agg_df: pd.DataFrame, best_h2o_df: pd.DataFrame, output_dir: Path, title_suffix: str = ""
) -> None:
    streaming_df = agg_df[agg_df["method"] == "streamingllm"].sort_values("total_budget")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    stream_color = "#D97706"
    h2o_color = "#2563EB"

    ax.plot(
        streaming_df["total_budget"],
        streaming_df["timeout_rate_mean"],
        color=stream_color,
        marker="o",
        linewidth=2.5,
        label="StreamingLLM",
    )
    add_error_band(ax, streaming_df, "timeout_rate_mean", "timeout_rate_std", stream_color)

    ax.plot(
        best_h2o_df["total_budget"],
        best_h2o_df["timeout_rate_mean"],
        color=h2o_color,
        marker="o",
        linewidth=2.8,
        label="H2O best per budget",
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Total KV cache budget (sink + window + heavy)")
    ax.set_ylabel("Task-timeout rate")
    ax.set_title(f"tau2 stress test{title_suffix}: timeout rate vs KV cache budget")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "stress_budget_timeout.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_h2o_heatmap(agg_df: pd.DataFrame, output_dir: Path, title_suffix: str = "") -> None:
    h2o_df = agg_df[agg_df["method"] == "h2o"].copy()
    if h2o_df.empty:
        return

    heatmap_df = (
        h2o_df.pivot_table(
            index="window",
            columns="heavy",
            values="success_rate_mean",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    heatmap_df.index = [int(value) for value in heatmap_df.index]
    heatmap_df.columns = [int(value) for value in heatmap_df.columns]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Pass rate / average reward"},
        ax=ax,
        vmin=0.0,
        vmax=max(0.35, float(heatmap_df.max().max())),
    )
    ax.set_title(f"H2O success rate{title_suffix} by local window and heavy-hitter size")
    ax.set_xlabel("Heavy-hitter size")
    ax.set_ylabel("Local window size")
    fig.tight_layout()
    fig.savefig(output_dir / "stress_h2o_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary_text(
    sim_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    best_h2o_df: pd.DataFrame,
    task_ids: list[str] | None,
) -> str:
    baseline_runs = raw_df[raw_df["method"] == "baseline"]
    baseline_success = baseline_runs["success_rate"].mean()
    baseline_timeout = baseline_runs["timeout_rate"].mean()

    streaming_df = agg_df[agg_df["method"] == "streamingllm"].sort_values("total_budget")
    streaming_kink = detect_kink(streaming_df)
    h2o_kink = detect_kink(best_h2o_df)

    best_streaming = None
    if not streaming_df.empty:
        best_streaming = streaming_df.sort_values(
            ["success_rate_mean", "timeout_rate_mean", "total_budget"],
            ascending=[False, True, True],
        ).iloc[0]

    task_reward_pivot = (
        sim_df.groupby(["task_id", "method"], dropna=False)["reward"].mean().unstack(fill_value=0.0)
    )
    task_reward_pivot = task_reward_pivot.sort_index(key=lambda index: index.astype(int))

    lines = [
        "tau2 stress-test summary",
        f"selected_task_count={int(sim_df['task_id'].nunique())}",
        f"selected_task_ids={' '.join(task_ids) if task_ids else 'ALL'}",
        f"baseline_unbounded_success={baseline_success:.4f}",
        f"baseline_unbounded_timeout={baseline_timeout:.4f}",
        "",
        "streamingllm_by_budget:",
    ]
    for row in streaming_df.itertuples(index=False):
        lines.append(
            "  "
            f"budget={int(row.total_budget)} success={row.success_rate_mean:.4f} "
            f"timeout={row.timeout_rate_mean:.4f} runs={int(row.runs)}"
        )

    if best_streaming is not None:
        lines.extend(
            [
                "",
                "streamingllm_best:",
                "  "
                f"budget={int(best_streaming.total_budget)} success={best_streaming.success_rate_mean:.4f} "
                f"timeout={best_streaming.timeout_rate_mean:.4f}",
            ]
        )
    if streaming_kink is not None:
        lines.extend(
            [
                "streamingllm_kink:",
                "  "
                f"largest_drop={streaming_kink['drop']:.4f} "
                f"interval={streaming_kink['high_budget']}->{streaming_kink['low_budget']} "
                f"plateau_success={streaming_kink['plateau_success']:.4f}",
            ]
        )

    lines.append("")
    lines.append("best_h2o_by_budget:")
    for row in best_h2o_df.itertuples(index=False):
        lines.append(
            "  "
            f"budget={int(row.total_budget)} success={row.success_rate_mean:.4f} "
            f"timeout={row.timeout_rate_mean:.4f} window={int(row.window)} heavy={int(row.heavy)}"
        )

    if h2o_kink is not None:
        lines.extend(
            [
                "",
                "h2o_best_kink:",
                "  "
                f"largest_drop={h2o_kink['drop']:.4f} "
                f"interval={h2o_kink['high_budget']}->{h2o_kink['low_budget']} "
                f"plateau_success={h2o_kink['plateau_success']:.4f}",
            ]
        )

    lines.extend(["", "task_mean_reward:"])
    for task_id, row in task_reward_pivot.iterrows():
        row_desc = " ".join(f"{method}={value:.3f}" for method, value in row.items())
        lines.append(f"  task={task_id} {row_desc}")
    return "\n".join(lines) + "\n"


def build_output_dir_name(task_ids: list[str] | None, output_dir_name: str | None) -> str:
    if output_dir_name:
        return output_dir_name
    if task_ids is None:
        return "stress_analysis"
    return "stress_analysis_tasks_" + "_".join(task_ids)


def main() -> None:
    args = parse_args()
    task_ids = resolve_task_ids(args)
    repo_dir = Path(__file__).resolve().parents[1]
    simulations_dir = repo_dir / "tau2-bench" / "data" / "simulations"
    output_dir = repo_dir / "outputs" / build_output_dir_name(task_ids, args.output_dir_name)
    title_suffix = ""
    if task_ids is not None:
        title_suffix = f" ({len(task_ids)} selected tasks)"

    sns.set_theme(style="whitegrid", context="talk")
    sim_df, raw_df, agg_df = load_results(simulations_dir, task_ids=task_ids)
    best_h2o_df = write_csv_outputs(sim_df, raw_df, agg_df, output_dir)

    plot_success_vs_budget(raw_df, agg_df, best_h2o_df, output_dir, title_suffix=title_suffix)
    plot_timeout_vs_budget(agg_df, best_h2o_df, output_dir, title_suffix=title_suffix)
    plot_h2o_heatmap(agg_df, output_dir, title_suffix=title_suffix)

    summary_text = build_summary_text(sim_df, raw_df, agg_df, best_h2o_df, task_ids)
    (output_dir / "stress_summary.txt").write_text(summary_text)
    print(summary_text, end="")
    print(f"saved_artifacts={output_dir}")


if __name__ == "__main__":
    main()
