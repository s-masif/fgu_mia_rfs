"""Publication plots for the four OMNeT++/PPO runs.

Usage: put this file in the folder with the four CSVs (or edit SRC below), then run
    python make_plots.py
Requires: numpy, pandas, matplotlib, scipy. Output: ./plots/*.png, *.pdf and summary CSVs.

Scope: episodes 1-800 only. Main analysis window: episodes 751-800 (last 50 of the first 800).
95% CI = Student-t interval of the mean over the 50 per-episode values (within-run variability;
each experiment is a single training seed, so seed-to-seed variability is NOT included).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SRC = "."        # folder containing the four CSV files
OUT = "plots"    # output folder for figures and tables
os.makedirs(OUT, exist_ok=True)

MAX_EP, WIN_LO, WIN_HI, ROLL = 800, 751, 800, 25
RUNS = {  # label: (file, color, linestyle)  -- Okabe-Ito colour-blind-safe palette
    "APPO":          ("APPO_metrics_30_users_1500.csv",                 "#0072B2", "-"),
    "PPO":           ("PPO_baseline_metrics_30_users_1500.csv",         "#E69F00", "-"),
    "APPO (masked)": ("APPO_masked_metrics_30_users_1500.csv",          "#56B4E9", "--"),
    "PPO (masked)":  ("PPO_baseline_masked_metrics_30_users_1500.csv",  "#D55E00", "--"),
}
HATCH = {"APPO": "", "PPO": "", "APPO (masked)": "//", "PPO (masked)": "//"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})

# ---------------- load + derived metrics ----------------
D = {}
for name, (f, _, _) in RUNS.items():
    d = pd.read_csv(os.path.join(SRC, f))
    d = d[d.episode <= MAX_EP].copy()
    d["completion_pct"] = 100 * d.completion_ratio
    d["high_pct"] = 100 * d.high_priority_completion
    d["low_pct"] = 100 * d.low_priority_completion
    d["priority_gap_pct"] = d.high_pct - d.low_pct
    d["offload_pct"] = 100 * d.offload_ratio
    d["inactive_correct_pct"] = 100 * d.inactive_correct_ratio
    d["a_reward_per_task"] = d.a_reward / 500                           # mean active-user reward per slot
    d["completed_tasks"] = d.completion_ratio * d.ET_active_count       # approx. completed tasks per episode
    d["energy_per_completed_mj"] = d.avg_energy_mj / d.completion_ratio # energy per successfully completed task
    D[name] = d
    assert len(d) == MAX_EP, (name, len(d))


def ci95(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return m, h


def window(d):
    return d[(d.episode >= WIN_LO) & (d.episode <= WIN_HI)]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


CI_NOTE = f"Error bars: 95% CI of the mean over episodes {WIN_LO}–{WIN_HI} (n = 50, single seed per method)."
ROLL_NOTE = f"Lines: {ROLL}-episode rolling mean; shading: 95% CI of the rolling mean."

# ---------------- 1. learning curves ----------------
CURVES = [
    ("completion_pct", "Overall completion (%)", "Overall task completion"),
    ("high_pct", "High-priority completion (%)", "High-priority task completion"),
    ("low_pct", "Low-priority completion (%)", "Low-priority task completion"),
    ("avg_delay_ms", "Average delay (ms)", "Average task delay"),
    ("a_reward_per_task", "Active-user reward per slot", "Active-user reward"),
    ("offload_pct", "Offloaded tasks (%)", "Offloading ratio"),
    ("avg_energy_mj", "Energy per task (mJ)", "Energy consumption per task"),
    ("priority_gap_pct", "High − low completion (pp)", "Priority differentiation"),
]


def curve(ax, key, runs=RUNS):
    for name in runs:
        d = D[name]
        s = d[key].rolling(ROLL, min_periods=ROLL)
        m, sd = s.mean(), s.std()
        h = sd / np.sqrt(ROLL) * stats.t.ppf(0.975, ROLL - 1)
        _, c, ls = RUNS[name]
        ax.plot(d.episode, m, color=c, ls=ls, lw=1.4, label=name)
        ax.fill_between(d.episode, m - h, m + h, color=c, alpha=0.18, lw=0)
    ax.axvspan(WIN_LO, WIN_HI, color="grey", alpha=0.12, lw=0)
    ax.set_xlim(0, MAX_EP)
    ax.set_xlabel("Episode")


for key, ylab, title in CURVES:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    curve(ax, key)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.text(0.01, -0.04, ROLL_NOTE + f" Grey band: analysis window ({WIN_LO}–{WIN_HI}).", fontsize=7, color="0.35")
    save(fig, f"curve_{key}")

# multi-panel overview
fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
for ax, (key, ylab, title) in zip(axes.flat, [CURVES[i] for i in (0, 1, 2, 3, 5, 6)]):
    curve(ax, key); ax.set_ylabel(ylab); ax.set_title(title)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.text(0.01, -0.01, ROLL_NOTE + f" Grey band: analysis window ({WIN_LO}–{WIN_HI}).", fontsize=8, color="0.35")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "overview_learning_curves")

# inactive-user handling (unmasked runs only; masked runs are 100% by construction)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
curve(ax, "inactive_correct_pct", runs=["APPO", "PPO"])
ax.set_ylabel("Inactive users kept local (%)"); ax.set_title("Inactive-user action correctness (unmasked runs)")
ax.axhline(25, color="0.4", lw=0.8, ls=":"); ax.text(10, 27, "uniform random policy (25%)", fontsize=7, color="0.35")
ax.legend(loc="lower right", frameon=False)
fig.text(0.01, -0.04, ROLL_NOTE + " Masked runs force the local action (100%) and are omitted.", fontsize=7, color="0.35")
save(fig, "curve_inactive_correct")

# ---------------- 2. final-window bar charts with 95% CI ----------------
summary = []
for name in RUNS:
    w = window(D[name])
    row = {"experiment": name}
    for key in ["completion_pct", "high_pct", "low_pct", "priority_gap_pct", "avg_delay_ms", "a_reward_per_task",
                "offload_pct", "avg_energy_mj", "energy_per_completed_mj", "completed_tasks", "inactive_correct_pct"]:
        m, h = ci95(w[key]); row[key + "_mean"] = m; row[key + "_ci95"] = h
    summary.append(row)
S = pd.DataFrame(summary)
S.round(4).to_csv(os.path.join(OUT, "summary_episodes_751_800.csv"), index=False)


def bars(ax, key, ylab, title, zoom=True):
    names = list(RUNS)
    ms = [S.loc[S.experiment == n, key + "_mean"].item() for n in names]
    hs = [S.loc[S.experiment == n, key + "_ci95"].item() for n in names]
    x = np.arange(len(names))
    ax.bar(x, ms, yerr=hs, capsize=4, color=[RUNS[n][1] for n in names], edgecolor="black", lw=0.6,
           hatch=[HATCH[n] for n in names], error_kw=dict(lw=1, ecolor="black"))
    for xi, m, h in zip(x, ms, hs):
        ax.text(xi, m + h, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in names])
    ax.set_ylabel(ylab); ax.set_title(title); ax.grid(axis="x", visible=False)
    if zoom:  # bars differ by small amounts: zoom the axis and say so
        lo, hi = min(m - h for m, h in zip(ms, hs)), max(m + h for m, h in zip(ms, hs))
        pad = (hi - lo) * 0.6 + 1e-9
        ax.set_ylim(lo - pad, hi + pad * 1.2)


BAR_SET = [
    ("completion_pct", "Completion (%)", "Overall task completion"),
    ("avg_delay_ms", "Average delay (ms)", "Average task delay"),
    ("high_pct", "Completion (%)", "High-priority completion"),
    ("low_pct", "Completion (%)", "Low-priority completion"),
    ("a_reward_per_task", "Reward per slot", "Active-user reward"),
    ("offload_pct", "Offloaded tasks (%)", "Offloading ratio"),
    ("avg_energy_mj", "Energy per task (mJ)", "Energy per task"),
    ("energy_per_completed_mj", "Energy per completed task (mJ)", "Energy efficiency"),
    ("priority_gap_pct", "High − low completion (pp)", "Priority differentiation"),
]
for key, ylab, title in BAR_SET:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars(ax, key, ylab, f"{title} (episodes {WIN_LO}–{WIN_HI})", zoom=key != "priority_gap_pct")
    if key == "priority_gap_pct":
        ax.axhline(0, color="black", lw=0.8)
    fig.text(0.01, -0.06, CI_NOTE + (" y-axis zoomed." if key != "priority_gap_pct" else ""), fontsize=7, color="0.35")
    save(fig, f"bar_{key}")

# grouped completion by priority
fig, ax = plt.subplots(figsize=(7.2, 3.8))
groups = [("completion_pct", "Overall"), ("high_pct", "High priority"), ("low_pct", "Low priority")]
bw = 0.2
for i, name in enumerate(RUNS):
    for g, (key, _) in enumerate(groups):
        m = S.loc[S.experiment == name, key + "_mean"].item(); h = S.loc[S.experiment == name, key + "_ci95"].item()
        ax.bar(g + (i - 1.5) * bw, m, bw, yerr=h, capsize=3, color=RUNS[name][1], edgecolor="black", lw=0.6,
               hatch=HATCH[name], label=name if g == 0 else None, error_kw=dict(lw=0.9))
        ax.text(g + (i - 1.5) * bw, m + h + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(range(3), [g[1] for g in groups]); ax.set_ylabel("Completion (%)")
ax.set_ylim(66, 77); ax.grid(axis="x", visible=False)
ax.set_title(f"Task completion by priority (episodes {WIN_LO}–{WIN_HI})")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.text(0.01, -0.12, CI_NOTE + " y-axis starts at 66%.", fontsize=7, color="0.35")
save(fig, "bar_completion_by_priority")

# ---------------- 3. distributions in the analysis window ----------------
for key, ylab, title in [("completion_pct", "Completion (%)", "Per-episode completion"),
                         ("avg_delay_ms", "Average delay (ms)", "Per-episode delay")]:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [window(D[n])[key].values for n in RUNS]
    vp = ax.violinplot(data, showextrema=False)
    for body, n in zip(vp["bodies"], RUNS):
        body.set_facecolor(RUNS[n][1]); body.set_alpha(0.35); body.set_edgecolor("black")
    ax.boxplot(data, widths=0.15, showfliers=True, medianprops=dict(color="black"),
               flierprops=dict(markersize=3))
    ax.set_xticks(range(1, 5), [n.replace(" (", "\n(") for n in RUNS]); ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylab); ax.set_title(f"{title} distribution (episodes {WIN_LO}–{WIN_HI})")
    fig.text(0.01, -0.06, "Violin: kernel density; box: median and IQR; whiskers: 1.5×IQR; n = 50 episodes each.",
             fontsize=7, color="0.35")
    save(fig, f"dist_{key}")

# ---------------- 4. effect sizes: APPO minus PPO (Welch 95% CI) ----------------
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    return diff, stats.t.ppf(0.975, df) * np.sqrt(va + vb)


EFFECTS = [("completion_pct", "Overall completion (pp)"), ("high_pct", "High-priority completion (pp)"),
           ("low_pct", "Low-priority completion (pp)"), ("avg_delay_ms", "Average delay (ms)"),
           ("offload_pct", "Offloading ratio (pp)")]
rows = []
fig, axes = plt.subplots(1, len(EFFECTS), figsize=(13, 2.9), sharey=True)
pairs = [("APPO", "PPO", "Unmasked", "o"), ("APPO (masked)", "PPO (masked)", "Masked", "s")]
for ax, (key, lab) in zip(axes, EFFECTS):
    for j, (a, b, pl, mk) in enumerate(pairs):
        diff, h = welch(window(D[a])[key], window(D[b])[key])
        rows.append(dict(metric=key, comparison=f"{a} − {b}", diff=diff, ci95=h))
        ax.errorbar(diff, j, xerr=h, fmt=mk, color=RUNS[a][1], capsize=4, ms=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(lab, fontsize=9); ax.set_yticks([0, 1], ["Unmasked\nAPPO − PPO", "Masked\nAPPO − PPO"])
    ax.set_ylim(-0.7, 1.7); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
fig.suptitle(f"Effect of attention: APPO minus PPO (episodes {WIN_LO}–{WIN_HI})", y=1.04)
fig.text(0.01, -0.07, "Points: difference in means; bars: Welch 95% CI (n = 50 episodes per run, single seed). "
         "Completion: positive favours APPO; delay: negative favours APPO; offloading ratio is descriptive (no better direction).", fontsize=7, color="0.35")
save(fig, "effect_appo_minus_ppo")
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "effect_sizes_episodes_751_800.csv"), index=False)

# ---------------- 5. learning speed ----------------
fig, ax = plt.subplots(figsize=(5.6, 3.4))
targets = [68, 69, 70, 71]
bw = 0.2
speed_rows = []
for i, name in enumerate(RUNS):
    roll = D[name].completion_pct.rolling(ROLL, min_periods=ROLL).mean()
    eps = []
    for t in targets:
        hit = D[name].episode[roll >= t]
        e = int(hit.iloc[0]) if len(hit) else np.nan
        eps.append(e); speed_rows.append(dict(experiment=name, target_pct=t, first_episode=e))
    xs = np.arange(len(targets)) + (i - 1.5) * bw
    ax.bar(xs, np.nan_to_num(eps, nan=0), bw, color=RUNS[name][1], edgecolor="black", lw=0.6,
           hatch=HATCH[name], label=name)
    for x, e in zip(xs, eps):
        ax.text(x, (e if not np.isnan(e) else 0) + 8, "n/r" if np.isnan(e) else str(e), ha="center",
                fontsize=6.5, rotation=90, va="bottom")
ax.set_xticks(range(len(targets)), [f"≥ {t}%" for t in targets]); ax.grid(axis="x", visible=False)
ax.set_xlabel(f"Completion target ({ROLL}-episode rolling mean)"); ax.set_ylabel("First episode reaching target")
ax.set_ylim(0, 900); ax.set_title("Learning speed (lower is faster)")
ax.legend(ncol=2, frameon=False, fontsize=7.5, loc="upper left")
fig.text(0.01, -0.06, "n/r: target not reached within 800 episodes. 25 is the earliest possible value (rolling window). Single seed per method.", fontsize=7, color="0.35")
save(fig, "learning_speed")
pd.DataFrame(speed_rows).to_csv(os.path.join(OUT, "learning_speed.csv"), index=False)

# ---------------- 6. trade-off scatter: delay vs completion ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for name in RUNS:
    r = S[S.experiment == name].iloc[0]
    ax.errorbar(r.avg_delay_ms_mean, r.completion_pct_mean, xerr=r.avg_delay_ms_ci95, yerr=r.completion_pct_ci95,
                fmt="o" if "masked" not in name else "s", color=RUNS[name][1], capsize=3, ms=7, label=name)
ax.set_xlabel("Average delay (ms)  ← better"); ax.set_ylabel("Overall completion (%)  → better")
ax.set_title(f"Delay–completion trade-off (episodes {WIN_LO}–{WIN_HI})")
ax.legend(frameon=False, loc="best")
fig.text(0.01, -0.05, CI_NOTE, fontsize=7, color="0.35")
save(fig, "tradeoff_delay_completion")

print(S[["experiment", "completion_pct_mean", "completion_pct_ci95", "avg_delay_ms_mean", "avg_delay_ms_ci95",
         "high_pct_mean", "low_pct_mean", "energy_per_completed_mj_mean"]].round(3).to_string(index=False))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(speed_rows).pivot(index="experiment", columns="target_pct", values="first_episode"))
