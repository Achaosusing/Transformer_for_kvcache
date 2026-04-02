#!/usr/bin/env python3
"""
Analyze the 3D sweep: evict_period × cache_budget × method.

Produces 4 figures:
  Fig 1 — Reward vs evict_period, faceted by budget (SL vs H2O vs baseline)
  Fig 2 — Reward vs budget, faceted by evict_period (SL vs H2O vs baseline)
  Fig 3 — Reward heatmap (ep × budget) per method, side-by-side
  Fig 4 — Avg task duration (speed) heatmap per method

Usage:
    python scripts/analyze/analyze_3d_sweep.py
    python scripts/analyze/analyze_3d_sweep.py --sim-dir tau2-bench/data/simulations \
        --output-dir outputs/sweep3d_analysis
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Regex parsers ──────────────────────────────────────────────────────────────
_SIM_FILE_RE = re.compile(
    r"^sweep3d_"
    r"(?P<method>baseline|sl|h2o)"
    r"(?:_s(?P<sink>\d+))?"
    r"(?:_w(?P<window>\d+))?"
    r"(?:_h(?P<heavy>\d+))?"
    r"(?:_ep(?P<ep>\d+))?"
    r"_task10\.json$"
)

ROOT = Path(__file__).resolve().parents[2]


def parse_sim_file(path: Path) -> dict | None:
    m = _SIM_FILE_RE.match(path.name)
    if not m:
        return None
    method_short = m.group("method")
    method = {"baseline": "baseline", "sl": "streamingllm", "h2o": "h2o"}[method_short]
    sink = int(m.group("sink") or 4)
    window = int(m.group("window") or 0)
    heavy = int(m.group("heavy") or 0)
    ep = int(m.group("ep") or 1)
    budget = sink + window + heavy

    data = json.loads(path.read_text())
    sims = data.get("simulations", [])

    rewards = []
    durations = []
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is not None:
            rewards.append(float(r))
        d = s.get("duration")
        if d is not None:
            durations.append(float(d))

    avg_reward = float(np.mean(rewards)) if rewards else float("nan")
    avg_duration = float(np.mean(durations)) if durations else float("nan")
    n_tasks = len(sims)
    n_success = sum(1 for r in rewards if r >= 1.0)

    return {
        "method": method,
        "sink": sink,
        "window": window,
        "heavy": heavy,
        "ep": ep,
        "budget": budget,
        "avg_reward": avg_reward,
        "avg_duration": avg_duration,
        "n_tasks": n_tasks,
        "n_success": n_success,
    }


def load_data(sim_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(sim_dir.glob("sweep3d_*.json")):
        row = parse_sim_file(path)
        if row:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No sweep3d_*.json files found in {sim_dir}")
    return pd.DataFrame(rows)


# ── Plotting helpers ───────────────────────────────────────────────────────────
METHOD_COLORS = {
    "baseline": "#555555",
    "streamingllm": "#2196F3",
    "h2o": "#E91E63",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "streamingllm": "StreamingLLM",
    "h2o": "H2O",
}
METHOD_MARKERS = {
    "baseline": "D",
    "streamingllm": "o",
    "h2o": "s",
}
EP_VALUES = [1, 4, 16]
BUDGET_VALUES = [132, 260, 516]
BUDGET_LABELS = {132: "B=132\n(small)", 260: "B=260\n(mid)", 516: "B=516\n(large)"}
EP_LABELS = {1: "ep=1\n(every step)", 4: "ep=4", 16: "ep=16\n(infrequent)"}


def _baseline_reward(df: pd.DataFrame) -> float:
    b = df[df.method == "baseline"]["avg_reward"]
    return float(b.iloc[0]) if len(b) > 0 else float("nan")


# ── Figure 1: Reward vs evict_period, faceted by budget ───────────────────────
def plot_fig1_reward_vs_ep(df: pd.DataFrame, ax_row: list, baseline_r: float) -> None:
    """3 subplots: one per budget. Each: SL vs H2O line by ep."""
    for ax, budget in zip(ax_row, BUDGET_VALUES):
        sub = df[(df.budget == budget) & (df.method != "baseline")]
        for method in ("streamingllm", "h2o"):
            mdf = sub[sub.method == method].sort_values("ep")
            if mdf.empty:
                continue
            ax.plot(
                mdf["ep"], mdf["avg_reward"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2, markersize=8,
                label=METHOD_LABELS[method],
            )

        # Baseline dashed line
        ax.axhline(baseline_r, color=METHOD_COLORS["baseline"],
                   linestyle="--", linewidth=1.5, label="Baseline")

        ax.set_title(BUDGET_LABELS[budget], fontsize=11)
        ax.set_xticks(EP_VALUES)
        ax.set_xlabel("evict_period", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", alpha=0.35)
        ax.legend(fontsize=8)

    ax_row[0].set_ylabel("Avg Reward", fontsize=11)


# ── Figure 2: Reward vs budget, faceted by ep ─────────────────────────────────
def plot_fig2_reward_vs_budget(df: pd.DataFrame, ax_row: list, baseline_r: float) -> None:
    for ax, ep in zip(ax_row, EP_VALUES):
        sub = df[(df.ep == ep) & (df.method != "baseline")]
        for method in ("streamingllm", "h2o"):
            mdf = sub[sub.method == method].sort_values("budget")
            if mdf.empty:
                continue
            ax.plot(
                mdf["budget"], mdf["avg_reward"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2, markersize=8,
                label=METHOD_LABELS[method],
            )

        ax.axhline(baseline_r, color=METHOD_COLORS["baseline"],
                   linestyle="--", linewidth=1.5, label="Baseline")

        ax.set_title(EP_LABELS[ep], fontsize=11)
        ax.set_xticks(BUDGET_VALUES)
        ax.set_xticklabels(["B=132", "B=260", "B=516"], fontsize=9)
        ax.set_xlabel("Cache Budget", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", alpha=0.35)
        ax.legend(fontsize=8)

    ax_row[0].set_ylabel("Avg Reward", fontsize=11)


# ── Figure 3: Reward heatmap (ep × budget) ────────────────────────────────────
def plot_heatmaps(df: pd.DataFrame, fig, axes, value_col: str,
                  title: str, fmt: str, vmin: float, vmax: float,
                  cmap: str = "RdYlGn") -> None:
    methods_plot = [("streamingllm", "StreamingLLM"), ("h2o", "H2O")]
    for ax, (method, label) in zip(axes, methods_plot):
        mdf = df[df.method == method]
        # Pivot: rows=ep, cols=budget
        pivot = mdf.pivot_table(index="ep", columns="budget",
                                values=value_col, aggfunc="mean")
        # Ensure correct order
        pivot = pivot.reindex(index=EP_VALUES, columns=BUDGET_VALUES)

        im = ax.imshow(pivot.values, aspect="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(BUDGET_VALUES)))
        ax.set_xticklabels(["B=132", "B=260", "B=516"], fontsize=10)
        ax.set_yticks(range(len(EP_VALUES)))
        ax.set_yticklabels([f"ep={e}" for e in EP_VALUES], fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cache Budget", fontsize=10)
        ax.set_ylabel("evict_period", fontsize=10)

        # Annotate cells
        for i in range(len(EP_VALUES)):
            for j in range(len(BUDGET_VALUES)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    cell_text = fmt.format(val)
                    ax.text(j, i, cell_text, ha="center", va="center",
                            fontsize=12, fontweight="bold",
                            color="black" if 0.35 < (val - vmin) / (vmax - vmin + 1e-9) < 0.75 else "white")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)


def build_all_figures(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_r = _baseline_reward(df)

    # ── Fig 1: Reward vs ep, faceted by budget ────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig1.suptitle("Reward vs evict_period  (faceted by budget)", fontsize=13, fontweight="bold")
    plot_fig1_reward_vs_ep(df, axes1, baseline_r)
    fig1.tight_layout()
    p1 = output_dir / "fig1_reward_vs_ep.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {p1}")

    # ── Fig 2: Reward vs budget, faceted by ep ────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig2.suptitle("Reward vs cache budget  (faceted by evict_period)", fontsize=13, fontweight="bold")
    plot_fig2_reward_vs_budget(df, axes2, baseline_r)
    fig2.tight_layout()
    p2 = output_dir / "fig2_reward_vs_budget.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p2}")

    # ── Fig 3: Reward heatmap ─────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))
    plot_heatmaps(df[df.method != "baseline"], fig3, axes3,
                  value_col="avg_reward",
                  title=f"Reward Heatmap (ep × budget)   baseline={baseline_r:.2f}",
                  fmt="{:.2f}", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    fig3.tight_layout()
    p3 = output_dir / "fig3_reward_heatmap.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {p3}")

    # ── Fig 4: Speed heatmap (avg task duration) ──────────────────────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(11, 4))
    # Also add baseline duration as annotation
    baseline_dur = df[df.method == "baseline"]["avg_duration"]
    baseline_dur_val = float(baseline_dur.iloc[0]) if len(baseline_dur) > 0 else float("nan")
    plot_heatmaps(df[df.method != "baseline"], fig4, axes4,
                  value_col="avg_duration",
                  title=f"Avg Task Duration (s) — Speed   baseline={baseline_dur_val:.0f}s",
                  fmt="{:.0f}", vmin=100, vmax=600, cmap="RdYlGn_r")
    fig4.tight_layout()
    p4 = output_dir / "fig4_speed_heatmap.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {p4}")

    # ── Fig 5: Combined overview ───────────────────────────────────────────────
    # Side-by-side bar: reward per config, sorted
    non_base = df[df.method != "baseline"].copy()
    non_base["label"] = non_base.apply(
        lambda r: f"{'SL' if r.method=='streamingllm' else 'H2O'}\nB={r.budget}\nep={r.ep}", axis=1
    )
    non_base = non_base.sort_values(["method", "budget", "ep"])

    fig5, ax5 = plt.subplots(figsize=(18, 5))
    x = np.arange(len(non_base))
    bars = ax5.bar(
        x, non_base["avg_reward"],
        color=[METHOD_COLORS[m] for m in non_base["method"]],
        alpha=0.85, width=0.7, edgecolor="white",
    )
    ax5.axhline(baseline_r, color=METHOD_COLORS["baseline"],
                linestyle="--", linewidth=2, label=f"Baseline ({baseline_r:.2f})", zorder=5)
    ax5.set_xticks(x)
    ax5.set_xticklabels(non_base["label"], fontsize=7)
    ax5.set_ylim(0, 1.15)
    ax5.set_ylabel("Avg Reward", fontsize=11)
    ax5.set_title("All Configs Reward  (blue=StreamingLLM, red=H2O, dashed=Baseline)",
                  fontsize=12, fontweight="bold")
    ax5.grid(axis="y", alpha=0.35)
    # Value labels
    for bar, val in zip(bars, non_base["avg_reward"]):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax5.legend(fontsize=10)
    fig5.tight_layout()
    p5 = output_dir / "fig5_all_configs_bar.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved: {p5}")

    # ── Fig 6: Scatter reward vs speed (Pareto front) ─────────────────────────
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    for method in ("streamingllm", "h2o"):
        mdf = df[df.method == method]
        sc = ax6.scatter(
            mdf["avg_duration"], mdf["avg_reward"],
            c=[METHOD_COLORS[method]] * len(mdf),
            marker=METHOD_MARKERS[method], s=120, zorder=4,
            label=METHOD_LABELS[method], alpha=0.85,
        )
        for _, row in mdf.iterrows():
            ax6.annotate(
                f"B={row.budget}\nep={row.ep}",
                (row.avg_duration, row.avg_reward),
                textcoords="offset points", xytext=(6, 4), fontsize=6.5,
            )
    # Baseline
    b_row = df[df.method == "baseline"].iloc[0]
    ax6.scatter([b_row.avg_duration], [b_row.avg_reward],
                c=METHOD_COLORS["baseline"], marker="D", s=180, zorder=5,
                label="Baseline")
    ax6.annotate("Baseline", (b_row.avg_duration, b_row.avg_reward),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax6.set_xlabel("Avg Task Duration (s)  ← faster", fontsize=11)
    ax6.set_ylabel("Avg Reward  ↑ better", fontsize=11)
    ax6.set_title("Quality-Speed Pareto  (top-left is best)", fontsize=12, fontweight="bold")
    ax6.grid(alpha=0.3)
    ax6.legend(fontsize=10)
    fig6.tight_layout()
    p6 = output_dir / "fig6_pareto.png"
    fig6.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"  Saved: {p6}")


def print_summary_table(df: pd.DataFrame) -> None:
    baseline_r = _baseline_reward(df)
    print("\n" + "=" * 78)
    print("FULL RESULTS TABLE")
    print("=" * 78)
    print(f"{'Config':<40} {'Reward':>8} {'vs Base':>8} {'AvgDur(s)':>10} {'N_ok':>6}")
    print("-" * 78)

    # Baseline first
    b = df[df.method == "baseline"].iloc[0]
    print(f"{'baseline':40} {b.avg_reward:8.3f} {'—':>8} {b.avg_duration:10.1f} {int(b.n_success):6d}/{int(b.n_tasks)}")

    for _, row in df[df.method != "baseline"].sort_values(["method", "budget", "ep"]).iterrows():
        cfg = f"{row.method}  B={row.budget}  ep={row.ep}"
        delta = row.avg_reward - baseline_r
        sign = "+" if delta >= 0 else ""
        print(f"{cfg:<40} {row.avg_reward:8.3f} {sign}{delta:7.3f} {row.avg_duration:10.1f} {int(row.n_success):6d}/{int(row.n_tasks)}")

    print("=" * 78)

    # Key observations
    best = df.loc[df.avg_reward.idxmax()]
    fastest_good = df[(df.avg_reward >= baseline_r) & (df.method != "baseline")]
    print("\nKEY FINDINGS")
    print(f"  Best reward:  {best.method} B={best.budget} ep={best.ep}  → {best.avg_reward:.3f}")
    if len(fastest_good) > 0:
        frow = fastest_good.loc[fastest_good.avg_duration.idxmin()]
        print(f"  Fastest ≥ baseline:  {frow.method} B={frow.budget} ep={frow.ep}"
              f"  reward={frow.avg_reward:.3f}  dur={frow.avg_duration:.1f}s")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze 3D sweep results")
    p.add_argument("--sim-dir", default=str(ROOT / "tau2-bench/data/simulations"),
                   help="Directory containing sweep3d_*.json files")
    p.add_argument("--output-dir", default=str(ROOT / "outputs/sweep3d_analysis"),
                   help="Where to save figures and CSV")
    return p


def main() -> None:
    args = build_parser().parse_args()
    sim_dir = Path(args.sim_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading simulations from: {sim_dir}")
    df = load_data(sim_dir)
    print(f"  {len(df)} configs loaded\n")

    print_summary_table(df)

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep3d_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}\n")

    print("Generating figures...")
    build_all_figures(df, output_dir)
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
