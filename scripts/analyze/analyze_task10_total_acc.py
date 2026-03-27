#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TARGET_TASK_IDS = {"0", "1", "3", "4", "5", "6", "10", "13", "26", "28"}
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
        if tail and is_repeat_token(tail[-1]):
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
        source_group = "old30"
    elif expected_tasks == 10 and tag == "task10":
        source_group = "new10"
    else:
        return None

    return {
        "method": method,
        "expected_tasks": expected_tasks,
        "sink": sink,
        "window": window,
        "heavy": heavy,
        "total_budget": total_budget,
        "repeat": repeat,
        "tag": tag,
        "source_group": source_group,
    }


def load_valid_task10_samples(simulations_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_records = []
    sample_records = []

    for path in sorted(simulations_dir.glob("stress_*.json")):
        meta = parse_stress_filename(path)
        if meta is None:
            continue

        payload = json.loads(path.read_text())
        simulations = payload["simulations"]
        actual_tasks = len(simulations)

        is_valid_run = False
        if meta["source_group"] == "old30":
            is_valid_run = True
        elif meta["source_group"] == "new10":
            is_valid_run = actual_tasks == 10

        if not is_valid_run:
            continue

        target_samples = [simulation for simulation in simulations if str(simulation["task_id"]) in TARGET_TASK_IDS]
        if meta["source_group"] == "new10" and len(target_samples) != 10:
            continue

        run_records.append(
            {
                **meta,
                "source_file": path.name,
                "actual_tasks": actual_tasks,
                "target_task_count": len(target_samples),
                "target_acc": sum(sim["reward_info"]["reward"] for sim in target_samples) / len(target_samples),
            }
        )

        for simulation in target_samples:
            sample_records.append(
                {
                    **meta,
                    "source_file": path.name,
                    "task_id": str(simulation["task_id"]),
                    "reward": float(simulation["reward_info"]["reward"]),
                    "termination_reason": simulation["termination_reason"],
                }
            )

    run_df = pd.DataFrame.from_records(run_records)
    sample_df = pd.DataFrame.from_records(sample_records)
    return run_df, sample_df


def aggregate_method_acc(sample_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sample_df.groupby("method", dropna=False)
        .agg(
            total_task_instances=("reward", "count"),
            total_correct=("reward", "sum"),
            acc=("reward", "mean"),
        )
        .reset_index()
        .sort_values("method")
    )


def aggregate_method_source_acc(sample_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sample_df.groupby(["source_group", "method"], dropna=False)
        .agg(
            total_task_instances=("reward", "count"),
            total_correct=("reward", "sum"),
            acc=("reward", "mean"),
        )
        .reset_index()
        .sort_values(["source_group", "method"])
    )


def aggregate_config_acc(sample_df: pd.DataFrame) -> pd.DataFrame:
    config_df = (
        sample_df.groupby(["method", "sink", "window", "heavy", "total_budget"], dropna=False)
        .agg(
            total_task_instances=("reward", "count"),
            total_correct=("reward", "sum"),
            acc=("reward", "mean"),
        )
        .reset_index()
        .sort_values(["method", "total_budget", "window", "heavy"], na_position="last")
    )
    return config_df


def build_best_h2o_df(config_df: pd.DataFrame) -> pd.DataFrame:
    h2o_df = config_df[config_df["method"] == "h2o"].copy()
    if h2o_df.empty:
        return h2o_df
    h2o_df = h2o_df.sort_values(["total_budget", "acc", "total_task_instances", "window", "heavy"], ascending=[True, False, False, True, True])
    return h2o_df.drop_duplicates(subset=["total_budget"], keep="first")


def build_task_acc_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    task_acc_df = (
        sample_df.groupby(["task_id", "method"], dropna=False)
        .agg(
            total_task_instances=("reward", "count"),
            total_correct=("reward", "sum"),
            acc=("reward", "mean"),
        )
        .reset_index()
        .sort_values(["task_id", "method"], key=lambda series: series.map(lambda value: int(value) if str(value).isdigit() else value))
    )
    return task_acc_df


def plot_method_acc_bar(method_acc_df: pd.DataFrame, output_dir: Path) -> None:
    palette = {"baseline": "#4B5563", "streamingllm": "#D97706", "h2o": "#2563EB"}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.barplot(data=method_acc_df, x="method", y="acc", hue="method", palette=palette, dodge=False, legend=False, ax=ax)
    for row in method_acc_df.itertuples(index=False):
        ax.text(row.method, row.acc + 0.015, f"{row.acc:.3f}\n({int(row.total_correct)}/{int(row.total_task_instances)})", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Method")
    ax.set_ylabel("Total valid task10 ACC")
    ax.set_title("Overall ACC on all valid target-10 task instances")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_method_acc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_budget_curves(config_df: pd.DataFrame, best_h2o_df: pd.DataFrame, method_acc_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    baseline_acc = float(method_acc_df.loc[method_acc_df["method"] == "baseline", "acc"].iloc[0])
    streaming_df = config_df[config_df["method"] == "streamingllm"].sort_values("total_budget")
    ax.axhline(baseline_acc, color="#4B5563", linestyle="--", linewidth=2.0, label=f"Baseline overall = {baseline_acc:.3f}")
    ax.plot(streaming_df["total_budget"], streaming_df["acc"], color="#D97706", marker="o", linewidth=2.5, label="StreamingLLM")
    ax.plot(best_h2o_df["total_budget"], best_h2o_df["acc"], color="#2563EB", marker="o", linewidth=2.8, label="H2O best per budget")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Total KV cache budget")
    ax.set_ylabel("ACC on all valid target-10 task instances")
    ax.set_title("Combined target-10 ACC vs KV cache budget")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_budget_acc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_h2o_heatmap(config_df: pd.DataFrame, output_dir: Path) -> None:
    h2o_df = config_df[config_df["method"] == "h2o"].copy()
    heatmap_df = h2o_df.pivot_table(index="window", columns="heavy", values="acc", aggfunc="mean").sort_index().sort_index(axis=1)
    heatmap_df.index = [int(value) for value in heatmap_df.index]
    heatmap_df.columns = [int(value) for value in heatmap_df.columns]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Combined ACC"},
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title("H2O combined ACC heatmap on valid target-10 task instances")
    ax.set_xlabel("Heavy-hitter size")
    ax.set_ylabel("Local window size")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_h2o_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, run_df: pd.DataFrame, method_acc_df: pd.DataFrame, method_source_df: pd.DataFrame, best_h2o_df: pd.DataFrame) -> None:
    total_valid_samples = int(len(pd.read_csv(output_dir / "valid_task_samples.csv")))
    lines = [
        "# Total Target-10 ACC Report",
        "",
        "## Scope",
        "- Included source A: old30 full runs, but only the target task IDs 0/1/3/4/5/6/10/13/26/28 are counted",
        "- Included source B: new10 reruns with tag task10, only complete 10-task files are counted",
        "- Excluded: incomplete new10 files",
        f"- Valid run files counted: {len(run_df)}",
        f"- Valid task instances counted across all methods: {total_valid_samples}",
        "",
        "## Overall Method ACC",
    ]
    for row in method_acc_df.itertuples(index=False):
        lines.append(f"- {row.method}: acc={row.acc:.4f}, correct={int(row.total_correct)}, total={int(row.total_task_instances)}")

    lines.extend(["", "## By Source Group"]) 
    for row in method_source_df.itertuples(index=False):
        lines.append(f"- {row.source_group} / {row.method}: acc={row.acc:.4f}, correct={int(row.total_correct)}, total={int(row.total_task_instances)}")

    lines.extend(["", "## Earliest Best H2O Points"]) 
    for row in best_h2o_df.sort_values("total_budget").head(8).itertuples(index=False):
        lines.append(
            f"- budget={int(row.total_budget)} acc={row.acc:.4f}, total={int(row.total_task_instances)}, window={int(row.window)}, heavy={int(row.heavy)}"
        )

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[2]
    simulations_dir = repo_dir / "tau2-bench" / "data" / "simulations"
    output_dir = repo_dir / "outputs" / "task10_total_acc"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    run_df, sample_df = load_valid_task10_samples(simulations_dir)
    run_df.to_csv(output_dir / "valid_runs.csv", index=False)
    sample_df.to_csv(output_dir / "valid_task_samples.csv", index=False)

    method_acc_df = aggregate_method_acc(sample_df)
    method_acc_df.to_csv(output_dir / "method_total_acc.csv", index=False)

    method_source_df = aggregate_method_source_acc(sample_df)
    method_source_df.to_csv(output_dir / "method_source_acc.csv", index=False)

    config_df = aggregate_config_acc(sample_df)
    config_df.to_csv(output_dir / "config_total_acc.csv", index=False)

    best_h2o_df = build_best_h2o_df(config_df)
    best_h2o_df.to_csv(output_dir / "best_h2o_by_budget.csv", index=False)

    task_acc_df = build_task_acc_summary(sample_df)
    task_acc_df.to_csv(output_dir / "task_total_acc.csv", index=False)

    plot_method_acc_bar(method_acc_df, output_dir)
    plot_budget_curves(config_df, best_h2o_df, method_acc_df, output_dir)
    plot_h2o_heatmap(config_df, output_dir)
    write_report(output_dir, run_df, method_acc_df, method_source_df, best_h2o_df)
    print(f"saved_artifacts={output_dir}")


if __name__ == "__main__":
    main()