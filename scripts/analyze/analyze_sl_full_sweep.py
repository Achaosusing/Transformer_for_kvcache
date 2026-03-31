"""
Comprehensive StreamingLLM sweep analysis.

Reads ALL sl_s4_* and sweep3d_sl_s4_* simulation files, merges them into a
(window, ep) matrix, and produces four figures saved to
outputs/sl_full_sweep_analysis/
"""

import json
import glob
import os
import re
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SIM_DIR  = os.path.join(os.path.dirname(__file__), "../../tau2-bench/data/simulations")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "../../outputs/sl_full_sweep_analysis")
TASK_TAG = "task10"
MIN_SIMS = 10   # threshold for "complete"

WINDOWS  = [32, 64, 128, 256, 512]
EPS      = [1, 2, 4, 8, 12, 16]

COLORS   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS  = ["o", "s", "^", "D", "v"]

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    """Returns dict: (window, ep) -> dict with keys n, reward, duration, file"""
    patterns = [
        f"{SIM_DIR}/sl_s4_w*_ep*_{TASK_TAG}.json",
        f"{SIM_DIR}/sweep3d_sl_s4_w*_ep*_{TASK_TAG}.json",
    ]
    data = {}
    for pat in patterns:
        for fpath in glob.glob(pat):
            m = re.search(r"_w(\d+)_ep(\d+)_" + TASK_TAG, os.path.basename(fpath))
            if not m:
                continue
            w, ep = int(m.group(1)), int(m.group(2))
            d = json.load(open(fpath))
            sims = d.get("simulations", [])
            n = len(sims)
            rewards  = [s["reward_info"]["reward"] for s in sims if "reward_info" in s]
            durations = [s.get("duration", math.nan) for s in sims]
            avg_r  = sum(rewards)  / len(rewards)  if rewards  else math.nan
            valid_d = [x for x in durations if not math.isnan(x)]
            avg_d  = sum(valid_d)  / len(valid_d)  if valid_d  else math.nan
            # prefer the newer sl_s4_ over sweep3d_ if both exist for same key
            if (w, ep) not in data or "sweep3d" in os.path.basename(data[(w, ep)]["file"]):
                data[(w, ep)] = dict(n=n, reward=avg_r, duration=avg_d,
                                     complete=(n >= MIN_SIMS), file=fpath)
    return data

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_matrix(data, key, windows=WINDOWS, eps=EPS):
    mat = np.full((len(windows), len(eps)), math.nan)
    for i, w in enumerate(windows):
        for j, ep in enumerate(eps):
            v = data.get((w, ep))
            if v and v["complete"]:
                mat[i, j] = v[key]
    return mat

def fmt_cell(v, fmt=".2f"):
    return format(v, fmt) if not math.isnan(v) else "—"

# ── Figures ───────────────────────────────────────────────────────────────────
def fig_reward_vs_ep(data):
    """Fig1: Reward vs evict_period, one line per window size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, w in enumerate(WINDOWS):
        xs, ys = [], []
        for ep in EPS:
            v = data.get((w, ep))
            if v and v["complete"]:
                xs.append(ep)
                ys.append(v["reward"])
        if xs:
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], label=f"w={w}")
    ax.axhline(0.9, ls="--", color="gray", lw=1, label="baseline (0.90)")
    ax.set_xlabel("evict_period (ep)")
    ax.set_ylabel("avg reward")
    ax.set_title("StreamingLLM: Reward vs evict_period (by window size)")
    ax.set_xticks(EPS)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="window size", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def fig_duration_vs_ep(data):
    """Fig2: Avg duration vs evict_period, one line per window size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, w in enumerate(WINDOWS):
        xs, ys = [], []
        for ep in EPS:
            v = data.get((w, ep))
            if v and v["complete"]:
                xs.append(ep)
                ys.append(v["duration"])
        if xs:
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], label=f"w={w}")
    ax.set_xlabel("evict_period (ep)")
    ax.set_ylabel("avg task duration (s)")
    ax.set_title("StreamingLLM: Speed vs evict_period (by window size)")
    ax.set_xticks(EPS)
    ax.legend(title="window size", loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def fig_reward_heatmap(data):
    """Fig3: Reward heatmap (window × ep)."""
    mat = get_matrix(data, "reward")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, vmin=0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="avg reward")
    ax.set_xticks(range(len(EPS)));      ax.set_xticklabels([f"ep={e}" for e in EPS])
    ax.set_yticks(range(len(WINDOWS)));  ax.set_yticklabels([f"w={w}" for w in WINDOWS])
    ax.set_title("StreamingLLM Reward Heatmap (window × evict_period)")
    for i in range(len(WINDOWS)):
        for j in range(len(EPS)):
            v = mat[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    fig.tight_layout()
    return fig

def fig_duration_heatmap(data):
    """Fig4: Duration heatmap (window × ep)."""
    mat = get_matrix(data, "duration")
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="avg duration (s)")
    ax.set_xticks(range(len(EPS)));      ax.set_xticklabels([f"ep={e}" for e in EPS])
    ax.set_yticks(range(len(WINDOWS)));  ax.set_yticklabels([f"w={w}" for w in WINDOWS])
    ax.set_title("StreamingLLM Duration Heatmap (window × evict_period)")
    for i in range(len(WINDOWS)):
        for j in range(len(EPS)):
            v = mat[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        fontsize=9, color="black")
    fig.tight_layout()
    return fig

def fig_pareto(data):
    """Fig5: Quality-Speed Pareto scatter (all complete configs)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, w in enumerate(WINDOWS):
        for j, ep in enumerate(EPS):
            v = data.get((w, ep))
            if not (v and v["complete"]):
                continue
            r, dur = v["reward"], v["duration"]
            ax.scatter(dur, r, color=COLORS[i % len(COLORS)],
                       marker=MARKERS[i % len(MARKERS)], s=80, zorder=3)
            ax.annotate(f"ep={ep}", (dur, r),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
    # legend for window sizes
    for i, w in enumerate(WINDOWS):
        ax.scatter([], [], color=COLORS[i % len(COLORS)],
                   marker=MARKERS[i % len(MARKERS)], label=f"w={w}")
    ax.axhline(0.9, ls="--", color="gray", lw=1, label="baseline reward")
    ax.set_xlabel("avg task duration (s)  [lower = faster]")
    ax.set_ylabel("avg reward  [higher = better]")
    ax.set_title("StreamingLLM: Quality–Speed Pareto (all window × ep)")
    ax.legend(title="window size", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── Text summary ──────────────────────────────────────────────────────────────
def print_summary(data):
    reward_mat  = get_matrix(data, "reward")
    dur_mat     = get_matrix(data, "duration")

    print("=" * 70)
    print("StreamingLLM Full Sweep — Summary")
    print("=" * 70)

    # completeness
    total = len(WINDOWS) * len(EPS)
    complete = sum(1 for w in WINDOWS for ep in EPS
                   if data.get((w, ep), {}).get("complete"))
    missing = [(w, ep) for w in WINDOWS for ep in EPS
               if not data.get((w, ep), {}).get("complete")]
    print(f"\nCompleteness: {complete}/{total} configs")
    if missing:
        print(f"  Missing/incomplete: {missing}")

    # reward matrix
    print("\n--- Reward matrix ---")
    print(f"  {'w\\ep':>6}", "  ".join(f"ep={e:>2}" for e in EPS))
    for i, w in enumerate(WINDOWS):
        row = "  ".join(fmt_cell(reward_mat[i, j]) for j in range(len(EPS)))
        print(f"  w={w:<5} {row}")

    # duration matrix
    print("\n--- Avg Duration (s) matrix ---")
    print(f"  {'w\\ep':>6}", "  ".join(f"ep={e:>2}" for e in EPS))
    for i, w in enumerate(WINDOWS):
        row = "  ".join(fmt_cell(dur_mat[i, j], ".0f") for j in range(len(EPS)))
        print(f"  w={w:<5} {row}")

    # ep effect on each window (reward std)
    print("\n--- ep effect on reward (std across ep per window) ---")
    for i, w in enumerate(WINDOWS):
        vals = [reward_mat[i, j] for j in range(len(EPS)) if not math.isnan(reward_mat[i, j])]
        if len(vals) >= 2:
            std = np.std(vals)
            mean = np.mean(vals)
            print(f"  w={w:<5}  mean={mean:.3f}  std={std:.3f}  "
                  f"range=[{min(vals):.2f}, {max(vals):.2f}]")

    # ep effect on speed
    print("\n--- ep effect on duration: ep=1 vs ep=16 per window ---")
    for i, w in enumerate(WINDOWS):
        d1  = dur_mat[i, EPS.index(1)]  if 1  in EPS else math.nan
        d16 = dur_mat[i, EPS.index(16)] if 16 in EPS else math.nan
        if not (math.isnan(d1) or math.isnan(d16)):
            pct = (d1 - d16) / d1 * 100
            print(f"  w={w:<5}  ep=1: {d1:.0f}s  ep=16: {d16:.0f}s  "
                  f"speedup: {pct:+.1f}%")

    # best configs at each budget level (reward first, then speed)
    print("\n--- Best config per window (by reward, then speed) ---")
    print(f"  {'window':>8}  {'best_ep':>7}  {'reward':>8}  {'dur(s)':>8}")
    for i, w in enumerate(WINDOWS):
        candidates = [
            (reward_mat[i, j], -dur_mat[i, j], EPS[j])
            for j in range(len(EPS))
            if not (math.isnan(reward_mat[i, j]) or math.isnan(dur_mat[i, j]))
        ]
        if candidates:
            best = max(candidates)
            print(f"  w={w:<6}  ep={best[2]:>5}  {best[0]:>8.2f}  {-best[1]:>8.0f}")

    print("=" * 70)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    data = load_data()
    os.makedirs(OUT_DIR, exist_ok=True)

    print_summary(data)

    # check completeness
    missing = [(w, ep) for w in WINDOWS for ep in EPS
               if not data.get((w, ep), {}).get("complete")]
    if missing:
        print(f"\n[WARN] {len(missing)} config(s) incomplete, skipping their data points in figures.")

    figs = [
        ("fig1_reward_vs_ep.png",    fig_reward_vs_ep(data)),
        ("fig2_duration_vs_ep.png",  fig_duration_vs_ep(data)),
        ("fig3_reward_heatmap.png",  fig_reward_heatmap(data)),
        ("fig4_duration_heatmap.png",fig_duration_heatmap(data)),
        ("fig5_pareto.png",          fig_pareto(data)),
    ]

    for fname, fig in figs:
        path = os.path.join(OUT_DIR, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[saved] {path}")

    print(f"\nDone. {len(figs)} figures written to {OUT_DIR}")

if __name__ == "__main__":
    main()
