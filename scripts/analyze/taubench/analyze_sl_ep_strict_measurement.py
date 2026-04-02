"""Strict analysis for StreamingLLM evict_period measurement.

Expected input files come from scripts/run/run_sl_ep_strict_measurement.sh.

Design assumptions:
    - fixed window across all runs
    - same task subset shared across all ep values within each repeat
    - multiple repeats with different task subsets

Recommended statistical interpretation:
    1. Primary: mean reward / success rate across repeats
    2. Primary paired: per-repeat delta vs reference ep=1
    3. Secondary: timeout rate and mean duration
    4. Diagnostic: per-task win/loss/tie against ep=1 within each repeat
"""

import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")
SIM_DIR = os.path.join(ROOT_DIR, "tau2-bench/data/simulations")
OUT_DIR = os.path.join(ROOT_DIR, "outputs/strict_ep_measurement")
REF_EP = 1

FILENAME_RE = re.compile(
    r"sl_epstrict_w(?P<window>\d+)_pool(?P<pool>[A-Za-z0-9_]+)_n(?P<n>\d+)_ep(?P<ep>\d+)_rep(?P<repeat>\d+)_s(?P<seed>\d+)\.json$"
)


def mean_std_ci(values):
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr)) if len(arr) else math.nan
    std = float(np.std(arr, ddof=0)) if len(arr) else math.nan
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) else math.nan
    return mean, std, ci95


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(SIM_DIR, "sl_epstrict_w*_pool*_n*_ep*_rep*_s*.json"))):
        name = os.path.basename(path)
        match = FILENAME_RE.match(name)
        if not match:
            continue

        meta = {key: int(value) if value.isdigit() else value for key, value in match.groupdict().items()}
        with open(path) as f:
            payload = json.load(f)

        sims = payload.get("simulations", [])
        rewards = []
        durations = []
        timeout_count = 0
        task_reward = {}
        task_duration = {}
        task_term = {}
        for sim in sims:
            task_id = str(sim.get("task_id"))
            reward = sim.get("reward_info", {}).get("reward", math.nan)
            duration = sim.get("duration", math.nan)
            term = sim.get("termination_reason")
            if not math.isnan(reward):
                rewards.append(float(reward))
                task_reward[task_id] = float(reward)
            if not math.isnan(duration):
                durations.append(float(duration))
                task_duration[task_id] = float(duration)
            task_term[task_id] = term
            if term == "task_timeout":
                timeout_count += 1

        n_tasks = len(sims)
        success_rate = float(np.mean(rewards)) if rewards else math.nan
        avg_duration = float(np.mean(durations)) if durations else math.nan
        timeout_rate = timeout_count / n_tasks if n_tasks else math.nan

        runs.append(
            {
                **meta,
                "path": path,
                "n_tasks_observed": n_tasks,
                "success_rate": success_rate,
                "mean_reward": success_rate,
                "avg_duration": avg_duration,
                "timeout_rate": timeout_rate,
                "task_reward": task_reward,
                "task_duration": task_duration,
                "task_term": task_term,
            }
        )
    return runs


def build_tables(runs):
    by_ep = defaultdict(list)
    by_repeat_ep = {}
    windows = set()
    pools = set()
    repeats = set()
    expected_n = set()
    for run in runs:
        by_ep[run["ep"]].append(run)
        by_repeat_ep[(run["repeat"], run["ep"])] = run
        windows.add(run["window"])
        pools.add(run["pool"])
        repeats.add(run["repeat"])
        expected_n.add(run["n"])
    return by_ep, by_repeat_ep, sorted(windows), sorted(pools), sorted(repeats), sorted(expected_n)


def summarize_runs(by_ep):
    rows = []
    for ep in sorted(by_ep):
        reward_vals = [run["mean_reward"] for run in by_ep[ep]]
        timeout_vals = [run["timeout_rate"] for run in by_ep[ep]]
        dur_vals = [run["avg_duration"] for run in by_ep[ep]]
        reward_mean, reward_std, reward_ci = mean_std_ci(reward_vals)
        timeout_mean, timeout_std, timeout_ci = mean_std_ci(timeout_vals)
        dur_mean, dur_std, dur_ci = mean_std_ci(dur_vals)
        rows.append(
            {
                "ep": ep,
                "n_repeats": len(by_ep[ep]),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_ci95": reward_ci,
                "timeout_mean": timeout_mean,
                "timeout_std": timeout_std,
                "timeout_ci95": timeout_ci,
                "duration_mean": dur_mean,
                "duration_std": dur_std,
                "duration_ci95": dur_ci,
            }
        )
    return rows


def paired_deltas(by_repeat_ep, repeats, ref_ep=REF_EP):
    deltas = defaultdict(list)
    task_pairs = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0, "missing": 0})
    for repeat in repeats:
        ref_run = by_repeat_ep.get((repeat, ref_ep))
        if not ref_run:
            continue
        for (rep_key, ep), run in by_repeat_ep.items():
            if rep_key != repeat or ep == ref_ep:
                continue
            deltas[(ep, "reward")].append(run["mean_reward"] - ref_run["mean_reward"])
            deltas[(ep, "timeout")].append(run["timeout_rate"] - ref_run["timeout_rate"])
            deltas[(ep, "duration")].append(run["avg_duration"] - ref_run["avg_duration"])

            common_tasks = set(ref_run["task_reward"]) & set(run["task_reward"])
            if not common_tasks:
                task_pairs[ep]["missing"] += 1
                continue
            for task_id in common_tasks:
                cur = run["task_reward"][task_id]
                ref = ref_run["task_reward"][task_id]
                if cur > ref:
                    task_pairs[ep]["win"] += 1
                elif cur < ref:
                    task_pairs[ep]["loss"] += 1
                else:
                    task_pairs[ep]["tie"] += 1
    return deltas, task_pairs


def fig_reward(rows):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    eps = [row["ep"] for row in rows]
    means = [row["reward_mean"] for row in rows]
    cis = [row["reward_ci95"] for row in rows]
    ax.errorbar(eps, means, yerr=cis, fmt="-o", capsize=4, color="#1f77b4", lw=2)
    ax.set_title("Strict evict_period measurement: reward vs ep")
    ax.set_xlabel("evict_period")
    ax.set_ylabel("mean reward across repeats")
    ax.set_xticks(eps)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_duration(rows):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    eps = [row["ep"] for row in rows]
    means = [row["duration_mean"] for row in rows]
    cis = [row["duration_ci95"] for row in rows]
    ax.errorbar(eps, means, yerr=cis, fmt="-o", capsize=4, color="#ff7f0e", lw=2)
    ax.set_title("Strict evict_period measurement: duration vs ep")
    ax.set_xlabel("evict_period")
    ax.set_ylabel("avg duration (s) across repeats")
    ax.set_xticks(eps)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_delta(deltas):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    metrics = [("reward", "Reward delta vs ep=1"), ("duration", "Duration delta vs ep=1")]
    for ax, (metric, title) in zip(axes, metrics):
        eps = sorted({ep for ep, key in deltas if key == metric})
        means = []
        cis = []
        for ep in eps:
            vals = deltas[(ep, metric)]
            mean, _std, ci = mean_std_ci(vals)
            means.append(mean)
            cis.append(ci)
        ax.errorbar(eps, means, yerr=cis, fmt="-o", capsize=4, lw=2)
        ax.axhline(0.0, color="gray", lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlabel("evict_period")
        ax.set_xticks(eps)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("delta")
    axes[1].set_ylabel("seconds")
    fig.tight_layout()
    return fig


def print_summary(rows, deltas, task_pairs, windows, pools, repeats, expected_n):
    print("=" * 76)
    print("Strict StreamingLLM evict_period measurement")
    print("=" * 76)
    print(f"window={windows}  task_pool={pools}  repeats={repeats}  tasks_per_repeat={expected_n}")
    print(f"reference ep={REF_EP}")

    print("\nPrimary summary (across repeats)")
    print("ep   repeats   reward_mean+-sd   95%CI    timeout_mean   duration_mean+-sd")
    for row in rows:
        print(
            f"{row['ep']:<4} {row['n_repeats']:<8} "
            f"{row['reward_mean']:.3f}+-{row['reward_std']:.3f}  {row['reward_ci95']:.3f}   "
            f"{row['timeout_mean']:.3f}         {row['duration_mean']:.1f}+-{row['duration_std']:.1f}s"
        )

    print("\nPaired delta vs ep=1 (repeat-blocked)")
    print("ep   reward_delta_mean+-sd   duration_delta_mean+-sd   timeout_delta_mean+-sd")
    for ep in sorted({ep for ep, key in deltas if key == 'reward'}):
        reward_mean, reward_std, _ = mean_std_ci(deltas[(ep, "reward")])
        duration_mean, duration_std, _ = mean_std_ci(deltas[(ep, "duration")])
        timeout_mean, timeout_std, _ = mean_std_ci(deltas[(ep, "timeout")])
        print(
            f"{ep:<4} {reward_mean:+.3f}+-{reward_std:.3f}        "
            f"{duration_mean:+.1f}+-{duration_std:.1f}s              "
            f"{timeout_mean:+.3f}+-{timeout_std:.3f}"
        )

    print("\nPer-task paired win/loss/tie vs ep=1")
    print("ep   win   loss   tie")
    for ep in sorted(task_pairs):
        item = task_pairs[ep]
        print(f"{ep:<4} {item['win']:<5} {item['loss']:<6} {item['tie']:<4}")

    best_reward = max(rows, key=lambda row: (row["reward_mean"], -row["duration_mean"]))
    fastest = min(rows, key=lambda row: row["duration_mean"])
    print("\nInterpretation")
    print(
        f"best mean reward ep={best_reward['ep']} ({best_reward['reward_mean']:.3f}), "
        f"fastest ep={fastest['ep']} ({fastest['duration_mean']:.1f}s)."
    )
    reward_spread = max(row["reward_mean"] for row in rows) - min(row["reward_mean"] for row in rows)
    duration_spread = max(row["duration_mean"] for row in rows) - min(row["duration_mean"] for row in rows)
    print(f"reward spread across ep={reward_spread:.3f}")
    print(f"duration spread across ep={duration_spread:.1f}s")
    print("=" * 76)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("No strict measurement files found. Run scripts/run/run_sl_ep_strict_measurement.sh first.")

    by_ep, by_repeat_ep, windows, pools, repeats, expected_n = build_tables(runs)
    rows = summarize_runs(by_ep)
    deltas, task_pairs = paired_deltas(by_repeat_ep, repeats)

    print_summary(rows, deltas, task_pairs, windows, pools, repeats, expected_n)

    figures = [
        ("fig1_reward_vs_ep_ci.png", fig_reward(rows)),
        ("fig2_duration_vs_ep_ci.png", fig_duration(rows)),
        ("fig3_delta_vs_ref.png", fig_delta(deltas)),
    ]
    for filename, fig in figures:
        path = os.path.join(OUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()