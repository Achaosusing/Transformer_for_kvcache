"""
StreamingLLM 3-Trial Analysis — w=64 & w=128, quantify evict_period effect.

Reads:
  · sl3x_w{64,128}_ep{ep}_task10_r{1,2,3}.json   [3 new independent trials]
  · sl_s4_w{w}_ep{ep}_task10.json                 [existing data, "ref"]
  · sweep3d_sl_s4_w{w}_ep{ep}_task10.json         [existing data, "ref"]

For each (window, ep) pair:
  - Computes per-trial reward and duration
  - Computes mean ± std across available trials
  - Answers: is w=128 ep=4's low reward (0.40 in ref) signal or noise?

Outputs:  outputs/sl_3x_analysis/
  fig1_reward_errbar.png    mean±std reward vs ep, w=64 & w=128 side-by-side
  fig2_trial_scatter.png    individual trial dots + mean line (per window)
  fig3_duration_errbar.png  mean±std duration vs ep
  fig4_noise_table.png      std table as heatmap (which ep configs are noisy?)
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
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SIM_DIR  = os.path.join(os.path.dirname(__file__), "../../tau2-bench/data/simulations")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "../../outputs/sl_3x_analysis")
TASK_TAG = "task10"
MIN_SIMS = 10

WINDOWS    = [64, 128]
EPS        = [1, 2, 4, 8, 12, 16]
TRIAL_IDS  = [1, 2, 3]

W_COLORS   = {64: "#1f77b4", 128: "#ff7f0e"}
EP_COLORS  = dict(zip(EPS, plt.cm.tab10(np.linspace(0, 0.8, len(EPS)))))
TRIAL_MARKERS = ["o", "s", "^"]

# ── Data loading ──────────────────────────────────────────────────────────────
def _parse_sim(fpath):
    """Return (avg_reward, avg_duration, reward_list, n_sims) or None if <10 sims."""
    with open(fpath) as f:
        d = json.load(f)
    sims = d.get("simulations", [])
    if len(sims) < MIN_SIMS:
        return None
    rewards   = [s["reward_info"]["reward"] for s in sims if "reward_info" in s]
    durations = [s.get("duration", math.nan) for s in sims]
    avg_r = sum(rewards) / len(rewards) if rewards else math.nan
    valid_d = [x for x in durations if not math.isnan(x)]
    avg_d = sum(valid_d) / len(valid_d) if valid_d else math.nan
    return avg_r, avg_d, rewards, len(sims)


def load_3x_trials():
    """
    Return dict: (window, ep, trial) -> dict(reward, duration, rewards, n)
    Covers sl3x_w{64,128}_ep{ep}_task10_r{trial}.json
    """
    data = {}
    for w in WINDOWS:
        for ep in EPS:
            for trial in TRIAL_IDS:
                name = f"sl3x_w{w}_ep{ep}_{TASK_TAG}_r{trial}"
                fpath = os.path.join(SIM_DIR, f"{name}.json")
                if not os.path.exists(fpath):
                    continue
                result = _parse_sim(fpath)
                if result is None:
                    print(f"[WARN] incomplete (<{MIN_SIMS} sims): {name}")
                    continue
                avg_r, avg_d, rewards, n = result
                data[(w, ep, trial)] = dict(reward=avg_r, duration=avg_d,
                                            rewards=rewards, n=n, file=fpath)
    return data


def load_ref():
    """
    Return dict: (window, ep) -> dict(reward, duration, rewards, n)
    Uses sl_s4_* and sweep3d_sl_s4_* as reference (prefer sl_s4_ over sweep3d_).
    """
    patterns = [
        f"{SIM_DIR}/sl_s4_w*_ep*_{TASK_TAG}.json",
        f"{SIM_DIR}/sweep3d_sl_s4_w*_ep*_{TASK_TAG}.json",
    ]
    data = {}
    for pat in patterns:
        for fpath in sorted(glob.glob(pat)):
            m = re.search(r"_w(\d+)_ep(\d+)_" + TASK_TAG, os.path.basename(fpath))
            if not m:
                continue
            w, ep = int(m.group(1)), int(m.group(2))
            if w not in WINDOWS:
                continue
            result = _parse_sim(fpath)
            if result is None:
                continue
            avg_r, avg_d, rewards, n = result
            # prefer non-sweep3d if already loaded
            if (w, ep) not in data or "sweep3d" in os.path.basename(data[(w, ep)]["file"]):
                data[(w, ep)] = dict(reward=avg_r, duration=avg_d,
                                     rewards=rewards, n=n, file=fpath)
    return data


def compute_stats(trial_data):
    """
    Given trial_data from load_3x_trials(), compute per-(window, ep) stats:
      mean, std, median, min, max, n_trials, per_trial_rewards
    Returns: dict (window, ep) -> stats dict
    """
    stats = {}
    for w in WINDOWS:
        for ep in EPS:
            vals = [trial_data[(w, ep, t)]["reward"]
                    for t in TRIAL_IDS if (w, ep, t) in trial_data]
            durs = [trial_data[(w, ep, t)]["duration"]
                    for t in TRIAL_IDS if (w, ep, t) in trial_data]
            if not vals:
                continue
            stats[(w, ep)] = dict(
                rewards=vals,
                durations=durs,
                mean=float(np.mean(vals)),
                std=float(np.std(vals, ddof=0)),
                median=float(np.median(vals)),
                r_min=min(vals),
                r_max=max(vals),
                dur_mean=float(np.mean(durs)) if durs else math.nan,
                dur_std=float(np.std(durs, ddof=0)) if len(durs) > 1 else math.nan,
                n_trials=len(vals),
            )
    return stats


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_reward_errbar(stats, ref):
    """
    Fig1: Mean±std reward vs evict_period, one subplot per window.
    Also overlays individual trial points and reference data marker.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, w in zip(axes, WINDOWS):
        color = W_COLORS[w]
        xs_mean, ys_mean, ys_std = [], [], []
        for ep in EPS:
            s = stats.get((w, ep))
            if s:
                xs_mean.append(ep)
                ys_mean.append(s["mean"])
                ys_std.append(s["std"])

        if xs_mean:
            n_trials = stats.get((w, xs_mean[0]), {}).get("n_trials", "?")
            # error-bar line (mean ± std)
            ax.errorbar(xs_mean, ys_mean, yerr=ys_std,
                        fmt="-o", color=color, lw=2, ms=7, capsize=4, capthick=1.5,
                        label=f"w={w} mean±std  (n={n_trials})")

            # individual trial points (jittered slightly)
            for t_idx, t in enumerate(TRIAL_IDS):
                trial_xs, trial_ys = [], []
                for ep in EPS:
                    from_data = None
                    # look up trial_data via stats
                    s = stats.get((w, ep))
                    if s and t_idx < len(s["rewards"]):
                        trial_xs.append(ep + (t_idx - 1) * 0.25)
                        trial_ys.append(s["rewards"][t_idx])
                if trial_xs:
                    ax.scatter(trial_xs, trial_ys,
                               marker=TRIAL_MARKERS[t_idx], color=color,
                               alpha=0.45, s=40, zorder=4,
                               label=f"  trial r{t}")

        # reference data (×)
        ref_xs, ref_ys = [], []
        for ep in EPS:
            r = ref.get((w, ep))
            if r:
                ref_xs.append(ep)
                ref_ys.append(r["reward"])
        if ref_xs:
            ax.scatter(ref_xs, ref_ys, marker="x", color="gray",
                       s=60, linewidths=2, zorder=5, label="ref (original)")

        ax.axhline(0.9, ls="--", color="dimgray", lw=1, alpha=0.6, label="baseline (0.90)")
        ax.set_title(f"w={w}  — Reward vs evict_period")
        ax.set_xlabel("evict_period (ep)")
        ax.set_ylabel("avg reward" if w == WINDOWS[0] else "")
        ax.set_xticks(EPS)
        ax.set_ylim(-0.05, 1.10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("StreamingLLM 3-Trial Analysis: Reward vs evict_period", y=1.01)
    fig.tight_layout()
    return fig


def fig_trial_scatter(stats, ref):
    """
    Fig2: For each (window, ep), show vertical strip of 3 trial rewards
    plus mean line and reference ×.  One subplot per window.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, w in zip(axes, WINDOWS):
        color = W_COLORS[w]
        for ep in EPS:
            s = stats.get((w, ep))
            if not s:
                continue
            rewards = s["rewards"]
            # jitter x slightly
            jitter = (np.arange(len(rewards)) - (len(rewards) - 1) / 2) * 0.18
            for i, (rew, jit) in enumerate(zip(rewards, jitter)):
                ax.scatter(ep + jit, rew,
                           marker=TRIAL_MARKERS[i % len(TRIAL_MARKERS)],
                           color=color, alpha=0.7, s=50, zorder=4)
            # mean bar
            ax.hlines(s["mean"], ep - 0.35, ep + 0.35,
                      colors=color, lw=2.5, zorder=5)
            # ±std range
            ax.vlines(ep, s["mean"] - s["std"], s["mean"] + s["std"],
                      colors=color, lw=1, linestyle=":", zorder=3)

        # reference ×
        for ep in EPS:
            r = ref.get((w, ep))
            if r:
                ax.scatter(ep, r["reward"], marker="x", color="gray",
                           s=70, linewidths=2, zorder=6)

        ax.axhline(0.9, ls="--", color="dimgray", lw=1, alpha=0.6)
        ax.set_title(f"w={w}  — Trial spread per ep")
        ax.set_xlabel("evict_period (ep)")
        ax.set_ylabel("reward" if w == WINDOWS[0] else "")
        ax.set_xticks(EPS)
        ax.set_ylim(-0.05, 1.10)
        ax.grid(True, alpha=0.25)

        # custom legend
        for i in range(len(TRIAL_IDS)):
            ax.scatter([], [], marker=TRIAL_MARKERS[i], color=color,
                       alpha=0.7, s=50, label=f"r{TRIAL_IDS[i]}")
        ax.scatter([], [], marker="x", color="gray", s=70, linewidths=2, label="ref")
        ax.plot([], [], color=color, lw=2.5, label="mean")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("StreamingLLM: Individual trial rewards (strips + mean bar)", y=1.01)
    fig.tight_layout()
    return fig


def fig_duration_errbar(stats, ref):
    """
    Fig3: Mean±std duration vs ep, one subplot per window.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, w in zip(axes, WINDOWS):
        color = W_COLORS[w]
        xs, ys_mean, ys_std = [], [], []
        for ep in EPS:
            s = stats.get((w, ep))
            if s and not math.isnan(s["dur_mean"]):
                xs.append(ep)
                ys_mean.append(s["dur_mean"])
                ys_std.append(s["dur_std"] if not math.isnan(s.get("dur_std", math.nan)) else 0)

        if xs:
            ax.errorbar(xs, ys_mean, yerr=ys_std,
                        fmt="-o", color=color, lw=2, ms=7, capsize=4, capthick=1.5,
                        label=f"w={w} mean±std")

        # reference
        ref_xs, ref_ys = [], []
        for ep in EPS:
            r = ref.get((w, ep))
            if r and not math.isnan(r["duration"]):
                ref_xs.append(ep)
                ref_ys.append(r["duration"])
        if ref_xs:
            ax.scatter(ref_xs, ref_ys, marker="x", color="gray",
                       s=60, linewidths=2, zorder=5, label="ref (original)")

        ax.set_title(f"w={w}  — Duration vs evict_period")
        ax.set_xlabel("evict_period (ep)")
        ax.set_ylabel("avg task duration (s)" if w == WINDOWS[0] else "")
        ax.set_xticks(EPS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("StreamingLLM 3-Trial Analysis: Speed vs evict_period", y=1.01)
    fig.tight_layout()
    return fig


def fig_noise_heatmap(stats, ref):
    """
    Fig4: Reward std heatmap — shows which (window, ep) are noisy.
    Left panel: std from 3 new trials.
    Right panel: |mean_3x - ref| deviation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    # panel 1 — std of 3 trials
    std_mat = np.full((len(WINDOWS), len(EPS)), math.nan)
    for i, w in enumerate(WINDOWS):
        for j, ep in enumerate(EPS):
            s = stats.get((w, ep))
            if s:
                std_mat[i, j] = s["std"]

    im1 = axes[0].imshow(std_mat, vmin=0, vmax=0.35, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im1, ax=axes[0], label="reward std")
    axes[0].set_xticks(range(len(EPS)));    axes[0].set_xticklabels([f"ep={e}" for e in EPS])
    axes[0].set_yticks(range(len(WINDOWS))); axes[0].set_yticklabels([f"w={w}" for w in WINDOWS])
    axes[0].set_title("Reward std (across 3 new trials)")
    for i in range(len(WINDOWS)):
        for j in range(len(EPS)):
            v = std_mat[i, j]
            if not math.isnan(v):
                axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=9, color="black")

    # panel 2 — |mean_3x - ref| deviation
    dev_mat = np.full((len(WINDOWS), len(EPS)), math.nan)
    for i, w in enumerate(WINDOWS):
        for j, ep in enumerate(EPS):
            s = stats.get((w, ep))
            r = ref.get((w, ep))
            if s and r:
                dev_mat[i, j] = abs(s["mean"] - r["reward"])

    im2 = axes[1].imshow(dev_mat, vmin=0, vmax=0.50, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im2, ax=axes[1], label="|mean_new − ref|")
    axes[1].set_xticks(range(len(EPS)));    axes[1].set_xticklabels([f"ep={e}" for e in EPS])
    axes[1].set_yticks(range(len(WINDOWS))); axes[1].set_yticklabels([f"w={w}" for w in WINDOWS])
    axes[1].set_title("Deviation from reference run")
    for i in range(len(WINDOWS)):
        for j in range(len(EPS)):
            v = dev_mat[i, j]
            if not math.isnan(v):
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=9, color="black")

    fig.suptitle("Reward noise by (window, evict_period)", y=1.02)
    fig.tight_layout()
    return fig


# ── Text summary ──────────────────────────────────────────────────────────────
def print_summary(stats, ref, trial_data):
    print("=" * 72)
    print("StreamingLLM 3-Trial Analysis — w=64 & w=128")
    print("=" * 72)

    # coverage
    n_complete = sum(1 for w in WINDOWS for ep in EPS if stats.get((w, ep)))
    n_total    = len(WINDOWS) * len(EPS)
    n_ref      = sum(1 for w in WINDOWS for ep in EPS if ref.get((w, ep)))
    print(f"\nNew 3-trial coverage : {n_complete}/{n_total} configs complete")
    print(f"Reference coverage   : {n_ref}/{n_total} configs")

    # trial counts
    trial_counts = {(w, ep): stats[(w, ep)]["n_trials"]
                    for w in WINDOWS for ep in EPS if (w, ep) in stats}
    min_t = min(trial_counts.values()) if trial_counts else 0
    max_t = max(trial_counts.values()) if trial_counts else 0
    print(f"Trials per config    : {min_t}–{max_t}")

    # mean±std table
    print("\n--- Reward mean ± std (3 new trials) ---")
    header = "  " + " ".join(f" ep={e:>2}" for e in EPS)
    print(header)
    for w in WINDOWS:
        cells = []
        for ep in EPS:
            s = stats.get((w, ep))
            if s:
                cells.append(f"{s['mean']:.2f}±{s['std']:.2f}")
            else:
                cells.append("  —    ")
        print(f"  w={w:<4}  " + "  ".join(cells))

    # reference reward table
    print("\n--- Reward reference (original single run) ---")
    print(header)
    for w in WINDOWS:
        cells = []
        for ep in EPS:
            r = ref.get((w, ep))
            cells.append(f"  {r['reward']:.2f}   " if r else "   —    ")
        print(f"  w={w:<4}  " + "  ".join(cells))

    # deviation table
    print("\n--- |mean_3x - ref| deviation ---")
    print(header)
    for w in WINDOWS:
        cells = []
        for ep in EPS:
            s = stats.get((w, ep))
            r = ref.get((w, ep))
            if s and r:
                cells.append(f"  {abs(s['mean']-r['reward']):.2f}   ")
            else:
                cells.append("   —    ")
        print(f"  w={w:<4}  " + "  ".join(cells))

    # noise ranking: highest std configs
    print("\n--- Top noisy configs (highest reward std) ---")
    ranked = sorted(
        [(w, ep, stats[(w, ep)]["std"]) for w in WINDOWS for ep in EPS if (w, ep) in stats],
        key=lambda x: -x[2]
    )
    for w, ep, std in ranked[:8]:
        s = stats[(w, ep)]
        ref_r = ref.get((w, ep), {}).get("reward", math.nan)
        ref_str = f"  ref={ref_r:.2f}" if not math.isnan(ref_r) else ""
        print(f"  w={w:<4} ep={ep:<3}  mean={s['mean']:.2f}  std={std:.2f}  "
              f"range=[{s['r_min']:.2f},{s['r_max']:.2f}]{ref_str}")

    # key question: w=128 ep=4
    print("\n--- KEY QUESTION: Is w=128 ep=4 low reward (ref=0.40) a real effect? ---")
    s128_4 = stats.get((128, 4))
    r128_4 = ref.get((128, 4))
    if s128_4:
        n_high = sum(1 for v in s128_4["rewards"] if v >= 0.6)
        verdict = "likely NOISE" if s128_4["mean"] >= 0.55 else "possibly REAL"
        print(f"  3-trial rewards : {s128_4['rewards']}")
        print(f"  mean ± std      : {s128_4['mean']:.3f} ± {s128_4['std']:.3f}")
        print(f"  ref single run  : {r128_4['reward']:.2f}" if r128_4 else "  ref: —")
        print(f"  trials ≥ 0.60   : {n_high}/{s128_4['n_trials']}")
        print(f"  Verdict         : {verdict}")
        if s128_4["std"] > 0.15:
            print(f"  [NOTE] high variance — ep=4 effect at w=128 is unreliable (need more trials)")
    else:
        print("  [NOT YET RUN] — run sl3x_w128_ep4_task10_r{1,2,3} first")

    # ep effect on duration
    print("\n--- ep effect on speed (mean duration s/task) ---")
    header_dur = "  " + " ".join(f"  ep={e:>2}" for e in EPS)
    print(header_dur)
    for w in WINDOWS:
        cells = []
        for ep in EPS:
            s = stats.get((w, ep))
            cells.append(f"   {s['dur_mean']:.0f}  " if (s and not math.isnan(s["dur_mean"])) else "   —  ")
        print(f"  w={w:<4}  " + "  ".join(cells))

    print("=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    trial_data = load_3x_trials()
    ref        = load_ref()
    stats      = compute_stats(trial_data)

    if not stats:
        print("[ERROR] No complete 3-trial data found.")
        print(f"  Expected files: {SIM_DIR}/sl3x_w{{64,128}}_ep*_{TASK_TAG}_r{{1,2,3}}.json")
        print("  Run scripts/run/run_sl_3x_w64_w128.sh first.")
        sys.exit(1)

    print_summary(stats, ref, trial_data)

    n_complete = sum(1 for w in WINDOWS for ep in EPS if stats.get((w, ep)))
    if n_complete < len(WINDOWS) * len(EPS):
        print(f"\n[WARN] Only {n_complete}/{len(WINDOWS)*len(EPS)} configs have complete 3-trial data. "
              "Partial results shown.")

    figs = [
        ("fig1_reward_errbar.png",   fig_reward_errbar(stats, ref)),
        ("fig2_trial_scatter.png",   fig_trial_scatter(stats, ref)),
        ("fig3_duration_errbar.png", fig_duration_errbar(stats, ref)),
        ("fig4_noise_heatmap.png",   fig_noise_heatmap(stats, ref)),
    ]

    for fname, fig in figs:
        path = os.path.join(OUT_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {path}")

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
