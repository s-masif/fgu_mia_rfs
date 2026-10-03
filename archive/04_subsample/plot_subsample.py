from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

mpl.use("Agg")

GAP = "train_test_gap"
AUC = "shokri_mia_auc"

DATASET_COLORS = {
    "roman_empire":   "#bcbd22",
    "amazon_ratings": "#17becf",
    "minesweeper":    "#7f7f7f",
    "tolokers":       "#aec7e8",
    "Computers":      "#2ca02c",
    "Photo":          "#9467bd",
    "WikiCS":         "#e377c2",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    })


def load(csv):
    df = pd.read_csv(csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str) == "nan")]
    return df


# ── Figure A: pooled gap-vs-AUC scatter ─────────────────────────────────────
def fig_gap_auc(df, out):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = df[GAP].values.astype(float)
    y = df[AUC].values.astype(float)

    for ds in sorted(df.dataset.unique()):
        m = df.dataset.values == ds
        ax.scatter(df[GAP].values[m], df[AUC].values[m], s=26, alpha=0.6,
                   color=DATASET_COLORS.get(ds, "#3b6fb0"),
                   edgecolors="none", label=ds)

    # pooled fit + chance line
    keep = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    xs = np.linspace(x[keep].min(), x[keep].max(), 100)
    ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    rho, p = spearmanr(x[keep], y[keep])
    ax.text(0.03, 0.97,
            f"Spearman $\\rho$ = {rho:.2f}\np < 0.001\nn = {int(keep.sum())}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))

    ax.set_xlabel("Train / test accuracy gap")
    ax.set_ylabel("MIA AUC")
    ax.legend(fontsize=8, loc="lower right", frameon=False, title="Dataset")
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out}.pdf / .png")


# ── Figure B: per-dataset gap & AUC vs fraction (twin axis) ─────────────────
def fig_clean_tier(df, out,
                   datasets=("roman_empire", "amazon_ratings",
                             "minesweeper", "tolokers")):
    present = [d for d in datasets if d in df.dataset.unique()]
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharex=True)
    if n == 1:
        axes = [axes]

    fracs = sorted(df.fraction.unique(), reverse=True)  # 1.0 → 0.25

    for ax, ds in zip(axes, present):
        sub = df[df.dataset == ds]
        g = (sub.groupby("fraction")
                .agg(gap=(GAP, "mean"), auc=(AUC, "mean"),
                     gap_s=(GAP, "std"), auc_s=(AUC, "std"))
                .reindex(fracs))
        xpos = np.arange(len(fracs))

        # left axis: gap
        c_gap = "#d62728"
        ax.errorbar(xpos, g["gap"], yerr=g["gap_s"], color=c_gap,
                    marker="o", markersize=4, capsize=3, linewidth=1.6,
                    label="gap")
        ax.set_ylabel("Train / test gap", color=c_gap)
        ax.tick_params(axis="y", labelcolor=c_gap)
        ax.axhline(0, color=c_gap, linestyle=":", linewidth=0.7, alpha=0.5)

        # right axis: AUC
        ax2 = ax.twinx()
        c_auc = "#1f77b4"
        ax2.errorbar(xpos, g["auc"], yerr=g["auc_s"], color=c_auc,
                     marker="s", markersize=4, capsize=3, linewidth=1.6,
                     label="MIA AUC")
        ax2.set_ylabel("MIA AUC", color=c_auc)
        ax2.tick_params(axis="y", labelcolor=c_auc)
        ax2.axhline(0.5, color=c_auc, linestyle=":", linewidth=0.7, alpha=0.5)
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{f:.2f}" for f in fracs])
        ax.set_xlabel("Subsample fraction")
        ax.set_title(ds)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mia_subsample_results/raw_data/mia_subsample_raw.csv")
    ap.add_argument("--out-dir", default="mia_subsample_results/plots")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = load(args.csv)
    fig_gap_auc(df, out / "subsample_gap_auc")
    fig_clean_tier(df, out / "subsample_clean_tier")
    print("✅ Done.")


if __name__ == "__main__":
    main()
