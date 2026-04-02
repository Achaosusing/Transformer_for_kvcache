#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


def is_repeat_token(token: str) -> bool:
    return any(token.endswith(suffix) for suffix in ORDINAL_SUFFIXES) and token[:-2].isdigit()


def parse_stress_filename(path: Path) -> dict | None:
    tokens = path.stem.split("_")
    if len(tokens) < 3 or tokens[0] != "stress":
        return None

    method = tokens[1]
    trial_task = tokens[2]
    if "x" not in trial_task:
        return None

    num_trials_str, expected_tasks_str = trial_task.split("x", 1)
    if not num_trials_str.isdigit() or not expected_tasks_str.isdigit():
        return None

    expected_tasks = int(expected_tasks_str)
    tail = tokens[3:]
    repeat = "run1"
    tag = None

    if method == "baseline":
        if tail and tail[-1] and is_repeat_token(tail[-1]):
            repeat = tail.pop()
        if tail:
            tag = "_".join(tail)
        sink = math.nan
        window = math.nan
        heavy = math.nan
        total_budget = math.nan
    elif method == "streamingllm":
        if len(tail) < 2:
            return None
        sink = int(tail[0])
        window = int(tail[1])
        heavy = 0
        remaining = tail[2:]
        if remaining and is_repeat_token(remaining[-1]):
            repeat = remaining.pop()
        if remaining:
            tag = "_".join(remaining)
        total_budget = sink + window
    elif method == "h2o":
        if len(tail) < 3:
            return None
        sink = int(tail[0])
        window = int(tail[1])
        heavy = int(tail[2])
        remaining = tail[3:]
        if remaining and is_repeat_token(remaining[-1]):
            repeat = remaining.pop()
        if remaining:
            tag = "_".join(remaining)
        total_budget = sink + window + heavy
    else:
        return None

    if expected_tasks == 30 and tag is None:
        cohort = "old30"
    elif expected_tasks == 10 and tag == "task10":
        cohort = "new10"
    else:
        cohort = "other"

    return {
        "method": method,
        "num_trials": int(num_trials_str),
        "expected_tasks": expected_tasks,
        "sink": sink,
        "window": window,
        "heavy": heavy,
        "total_budget": total_budget,
        "repeat": repeat,
        "tag": tag,
        "cohort": cohort,
    }


def load_records(simulations_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_records = []
    sim_records = []

    for path in sorted(simulations_dir.glob("stress_*.json")):
        meta = parse_stress_filename(path)
        if meta is None or meta["cohort"] == "other":
            continue

        payload = json.loads(path.read_text())
        simulations = payload["simulations"]
        rewards = [simulation["reward_info"]["reward"] for simulation in simulations]
        timeout_rate = sum(simulation["termination_reason"] == "task_timeout" for simulation in simulations) / len(simulations)
        avg_duration = sum(simulation.get("duration", 0.0) for simulation in simulations) / len(simulations)
        avg_turns = sum(len(simulation.get("messages", [])) for simulation in simulations) / len(simulations)

        run_records.append(
            {
                **meta,
                "source_file": path.name,
                "actual_tasks": len(simulations),
                "is_complete": len(simulations) == meta["expected_tasks"],
                "success_rate": sum(rewards) / len(simulations),
                "timeout_rate": timeout_rate,
                "avg_duration_s": avg_duration,
                "avg_turns": avg_turns,
            }
        )

        for simulation in simulations:
            sim_records.append(
                {
                    **meta,
                    "source_file": path.name,
                    "actual_tasks": len(simulations),
                    "is_complete": len(simulations) == meta["expected_tasks"],
                    "task_id": str(simulation["task_id"]),
                    "reward": float(simulation["reward_info"]["reward"]),
                    "termination_reason": simulation["termination_reason"],
                    "duration_s": float(simulation.get("duration", 0.0)),
                    "turns": len(simulation.get("messages", [])),
                }
            )

    run_df = pd.DataFrame.from_records(run_records)
    sim_df = pd.DataFrame.from_records(sim_records)
    return run_df, sim_df


def aggregate_configs(run_df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        run_df.groupby(["cohort", "method", "sink", "window", "heavy", "total_budget"], dropna=False)
        .agg(
            runs=("source_file", "count"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            timeout_rate_mean=("timeout_rate", "mean"),
            timeout_rate_std=("timeout_rate", "std"),
            avg_duration_s_mean=("avg_duration_s", "mean"),
            avg_turns_mean=("avg_turns", "mean"),
        )
        .reset_index()
        .sort_values(["cohort", "method", "total_budget", "window", "heavy"], na_position="last")
    )
    agg_df[["success_rate_std", "timeout_rate_std"]] = agg_df[["success_rate_std", "timeout_rate_std"]].fillna(0.0)
    return agg_df


def build_best_h2o_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    h2o_df = agg_df[agg_df["method"] == "h2o"].copy()
    if h2o_df.empty:
        return h2o_df
    h2o_df = h2o_df.sort_values(
        ["cohort", "total_budget", "success_rate_mean", "timeout_rate_mean", "window", "heavy"],
        ascending=[True, True, False, True, True, True],
    )
    return h2o_df.drop_duplicates(subset=["cohort", "total_budget"], keep="first")


def detect_kink(curve_df: pd.DataFrame) -> dict | None:
    curve_df = curve_df.dropna(subset=["total_budget", "success_rate_mean"]).sort_values("total_budget", ascending=False)
    if len(curve_df) < 2:
        return None
    curve_df = curve_df.reset_index(drop=True)
    drops = curve_df["success_rate_mean"] - curve_df["success_rate_mean"].shift(-1)
    drops = drops.iloc[:-1]
    positive_drops = drops[drops > 0]
    if positive_drops.empty:
        return None
    idx = int(positive_drops.idxmax())
    return {
        "drop": float(positive_drops.loc[idx]),
        "high_budget": int(curve_df.iloc[idx]["total_budget"]),
        "low_budget": int(curve_df.iloc[idx + 1]["total_budget"]),
    }


def add_error_band(ax: plt.Axes, df: pd.DataFrame, mean_col: str, std_col: str, color: str) -> None:
    if df.empty:
        return
    lower = (df[mean_col] - df[std_col]).clip(lower=0.0)
    upper = (df[mean_col] + df[std_col]).clip(upper=1.0)
    ax.fill_between(df["total_budget"], lower, upper, color=color, alpha=0.10)


def plot_success_comparison(agg_df: pd.DataFrame, best_h2o_df: pd.DataFrame, output_dir: Path) -> None:
    cohorts = ["old30", "new10"]
    titles = {"old30": "Old 30-task stress", "new10": "New 10-task rerun"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)
    for ax, cohort in zip(axes, cohorts):
        cohort_df = agg_df[agg_df["cohort"] == cohort]
        baseline_df = cohort_df[cohort_df["method"] == "baseline"]
        streaming_df = cohort_df[cohort_df["method"] == "streamingllm"].sort_values("total_budget")
        cohort_best_h2o_df = best_h2o_df[best_h2o_df["cohort"] == cohort].sort_values("total_budget")

        if not baseline_df.empty:
            baseline_mean = baseline_df["success_rate_mean"].mean()
            ax.axhline(
                baseline_mean,
                color="#374151",
                linestyle="--",
                linewidth=2.0,
                label=f"Baseline = {baseline_mean:.3f}",
            )

        ax.plot(
            streaming_df["total_budget"],
            streaming_df["success_rate_mean"],
            color="#D97706",
            marker="o",
            linewidth=2.5,
            label="StreamingLLM",
        )
        add_error_band(ax, streaming_df, "success_rate_mean", "success_rate_std", "#D97706")

        ax.plot(
            cohort_best_h2o_df["total_budget"],
            cohort_best_h2o_df["success_rate_mean"],
            color="#2563EB",
            marker="o",
            linewidth=2.8,
            label="H2O best per budget",
        )

        ax.set_title(titles[cohort])
        ax.set_xlabel("Total KV cache budget")
        ax.set_ylabel("Pass rate / average reward")
        ax.grid(True, axis="y", alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_success.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_timeout_comparison(agg_df: pd.DataFrame, best_h2o_df: pd.DataFrame, output_dir: Path) -> None:
    cohorts = ["old30", "new10"]
    titles = {"old30": "Old 30-task stress", "new10": "New 10-task rerun"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, cohort in zip(axes, cohorts):
        cohort_df = agg_df[agg_df["cohort"] == cohort]
        streaming_df = cohort_df[cohort_df["method"] == "streamingllm"].sort_values("total_budget")
        cohort_best_h2o_df = best_h2o_df[best_h2o_df["cohort"] == cohort].sort_values("total_budget")

        ax.plot(
            streaming_df["total_budget"],
            streaming_df["timeout_rate_mean"],
            color="#D97706",
            marker="o",
            linewidth=2.5,
            label="StreamingLLM",
        )
        add_error_band(ax, streaming_df, "timeout_rate_mean", "timeout_rate_std", "#D97706")
        ax.plot(
            cohort_best_h2o_df["total_budget"],
            cohort_best_h2o_df["timeout_rate_mean"],
            color="#2563EB",
            marker="o",
            linewidth=2.8,
            label="H2O best per budget",
        )
        ax.set_title(titles[cohort])
        ax.set_xlabel("Total KV cache budget")
        ax.set_ylabel("Timeout rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_timeout.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_h2o_heatmap(agg_df: pd.DataFrame, cohort: str, output_dir: Path) -> None:
    cohort_h2o_df = agg_df[(agg_df["cohort"] == cohort) & (agg_df["method"] == "h2o")].copy()
    if cohort_h2o_df.empty:
        return
    heatmap_df = (
        cohort_h2o_df.pivot_table(index="window", columns="heavy", values="success_rate_mean", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    heatmap_df.index = [int(value) for value in heatmap_df.index]
    heatmap_df.columns = [int(value) for value in heatmap_df.columns]
    title = "H2O success rate heatmap: old30" if cohort == "old30" else "H2O success rate heatmap: new10"
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Pass rate / average reward"},
        vmin=0.0,
        vmax=max(0.35, float(heatmap_df.max().max())),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Heavy-hitter size")
    ax.set_ylabel("Local window size")
    fig.tight_layout()
    fig.savefig(output_dir / f"{cohort}_h2o_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_new10_task_summary(sim_df: pd.DataFrame) -> pd.DataFrame:
    task_df = sim_df[(sim_df["cohort"] == "new10") & (sim_df["is_complete"])].copy()
    if task_df.empty:
        return task_df
    summary_df = (
        task_df.groupby(["task_id", "method"], dropna=False)
        .agg(
            runs=("reward", "count"),
            reward_mean=("reward", "mean"),
            timeout_rate=("termination_reason", lambda values: sum(value == "task_timeout" for value in values) / len(values)),
        )
        .reset_index()
        .sort_values(["task_id", "method"], key=lambda series: series.map(lambda value: int(value) if str(value).isdigit() else value))
    )
    return summary_df


def summarize_cohort(agg_df: pd.DataFrame, best_h2o_df: pd.DataFrame, cohort: str) -> list[str]:
    cohort_df = agg_df[agg_df["cohort"] == cohort]
    baseline_df = cohort_df[cohort_df["method"] == "baseline"]
    streaming_df = cohort_df[cohort_df["method"] == "streamingllm"].sort_values("total_budget")
    h2o_best_df = best_h2o_df[best_h2o_df["cohort"] == cohort].sort_values("total_budget")

    lines = []
    if not baseline_df.empty:
        lines.append(f"- Baseline mean reward: {baseline_df['success_rate_mean'].mean():.4f}")
        lines.append(f"- Baseline mean timeout: {baseline_df['timeout_rate_mean'].mean():.4f}")

    if not streaming_df.empty:
        best_stream = streaming_df.sort_values(["success_rate_mean", "timeout_rate_mean", "total_budget"], ascending=[False, True, True]).iloc[0]
        lines.append(
            f"- StreamingLLM best: budget={int(best_stream.total_budget)} reward={best_stream.success_rate_mean:.4f} timeout={best_stream.timeout_rate_mean:.4f}"
        )
        kink = detect_kink(streaming_df)
        if kink is not None:
            lines.append(
                f"- StreamingLLM main drop: {kink['high_budget']} -> {kink['low_budget']} (drop={kink['drop']:.4f})"
            )

    if not h2o_best_df.empty:
        best_h2o = h2o_best_df.sort_values(["success_rate_mean", "timeout_rate_mean", "total_budget"], ascending=[False, True, True]).iloc[0]
        lines.append(
            f"- H2O earliest/best plateau candidate: budget={int(best_h2o.total_budget)} reward={best_h2o.success_rate_mean:.4f} timeout={best_h2o.timeout_rate_mean:.4f} window={int(best_h2o.window)} heavy={int(best_h2o.heavy)}"
        )
        kink = detect_kink(h2o_best_df)
        if kink is not None:
            lines.append(
                f"- H2O best-curve main drop: {kink['high_budget']} -> {kink['low_budget']} (drop={kink['drop']:.4f})"
            )
    return lines


def write_report(
    output_dir: Path,
    run_df: pd.DataFrame,
    complete_run_df: pd.DataFrame,
    incomplete_run_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    best_h2o_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
) -> None:
    old30_count = int(len(complete_run_df[complete_run_df["cohort"] == "old30"]))
    new10_count = int(len(complete_run_df[complete_run_df["cohort"] == "new10"]))
    lines = [
        "# Stress Test Recap",
        "",
        "## Scope",
        "- old30: earlier 30-task stress results",
        "- new10: newly rerun task10 results with filenames containing task10",
        f"- complete runs used in main plots: old30={old30_count}, new10={new10_count}",
        f"- incomplete runs excluded from main curves: {len(incomplete_run_df)}",
        "",
        "## Old30 Summary",
        *summarize_cohort(agg_df, best_h2o_df, "old30"),
        "",
        "## New10 Summary",
        *summarize_cohort(agg_df, best_h2o_df, "new10"),
        "",
        "## Incomplete New10 Runs",
    ]
    if incomplete_run_df.empty:
        lines.append("- None")
    else:
        for row in incomplete_run_df.sort_values(["cohort", "source_file"]).itertuples(index=False):
            lines.append(
                f"- {row.source_file}: actual_tasks={int(row.actual_tasks)} expected_tasks={int(row.expected_tasks)} reward={row.success_rate:.4f}"
            )

    lines.extend([
        "",
        "## Output Files",
        "- comparison_success.png: side-by-side success curves for old30 and new10",
        "- comparison_timeout.png: side-by-side timeout curves for old30 and new10",
        "- old30_h2o_heatmap.png: H2O heatmap for 30-task results",
        "- new10_h2o_heatmap.png: H2O heatmap for 10-task rerun results",
        "- *_aggregated_configs.csv: per-config aggregated metrics",
        "- new10_task_reward_summary.csv: task-level reward summary for the rerun task10 set",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[2]
    simulations_dir = repo_dir / "tau2-bench" / "data" / "simulations"
    output_dir = repo_dir / "outputs" / "stress_recap"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    run_df, sim_df = load_records(simulations_dir)
    run_df.to_csv(output_dir / "all_runs.csv", index=False)

    incomplete_run_df = run_df[~run_df["is_complete"]].copy()
    complete_run_df = run_df[run_df["is_complete"]].copy()
    incomplete_run_df.to_csv(output_dir / "incomplete_runs.csv", index=False)

    complete_run_df.to_csv(output_dir / "complete_runs.csv", index=False)
    agg_df = aggregate_configs(complete_run_df)
    agg_df.to_csv(output_dir / "aggregated_configs.csv", index=False)

    for cohort in ["old30", "new10"]:
        cohort_df = complete_run_df[complete_run_df["cohort"] == cohort]
        cohort_df.to_csv(output_dir / f"{cohort}_complete_runs.csv", index=False)
        aggregate_configs(cohort_df).to_csv(output_dir / f"{cohort}_aggregated_configs.csv", index=False)

    best_h2o_df = build_best_h2o_df(agg_df)
    best_h2o_df.to_csv(output_dir / "best_h2o_by_budget.csv", index=False)
    task_summary_df = build_new10_task_summary(sim_df)
    task_summary_df.to_csv(output_dir / "new10_task_reward_summary.csv", index=False)

    plot_success_comparison(agg_df, best_h2o_df, output_dir)
    plot_timeout_comparison(agg_df, best_h2o_df, output_dir)
    plot_h2o_heatmap(agg_df, "old30", output_dir)
    plot_h2o_heatmap(agg_df, "new10", output_dir)
    write_report(output_dir, run_df, complete_run_df, incomplete_run_df, agg_df, best_h2o_df, task_summary_df)

    print(f"saved_artifacts={output_dir}")


if __name__ == "__main__":
    main()