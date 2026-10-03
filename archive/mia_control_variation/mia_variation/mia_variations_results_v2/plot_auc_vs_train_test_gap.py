"""
plot_figure1_gap.py
===================
Figure 1 for the paper, exactly as the professor requested:

    "Make a single scatter plot: train/test gap on the x axis, MIA AUC on the
     y axis, one point for every configuration you have already run, across
     datasets, partitioners, values of K and forget ratios. If the points fall
     on one curve, we are done and that is Figure 1."

This script produces:

  figure1_gap_vs_auc.{pdf,png}   THE figure — plain scatter, one dot per config,
                                 a single fitted line, nothing else.

  (separate, optional supporting figures:)
  supp_gap_vs_auc_by_partitioner.{pdf,png}   same scatter, coloured by partitioner
  supp_gap_vs_auc_by_dataset.{pdf,png}       same scatter, coloured by dataset

Usage
-----
  python plot_figure1_gap.py \\
      --dir     mia_variation_results/raw_results \\
      --out-dir mia_variation_results/plots
"""
from __future__ import annotations

import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr, pearsonr

mpl.use("Agg")

GAP_COL = "train_test_gap"
AUC_COL = "shokri_mia_auc"
PRIMARY_KEY = ["dataset", "scenario", "seed",
               "partitioning", "num_clients", "forget_ratio"]


def setup_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        120,
        "savefig.dpi":       220,
        "savefig.bbox":      "tight",
    })


def load_all(csv_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(csv_dir / "mia_variations_raw*.csv")))
    if not files:
        # fall back to any csv in the directory
        files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {csv_dir}")
    dfs = [pd.read_csv(f) for f in files]
    print(f"Loaded {len(files)} CSV(s):")
    for f, d in zip(files, dfs):
        print(f"  {Path(f).name}: {len(d)} rows")
    df = pd.concat(dfs, ignore_index=True)
    # dedup on the full config key if all key columns exist
    if all(c in df.columns for c in PRIMARY_KEY):
        df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
    # keep only rows with both quantities present
    df = df[~df[GAP_COL].isna() & ~df[AUC_COL].isna()].copy()
    print(f"Usable rows (gap & AUC present, deduped): {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — the plain scatter the professor asked for
# ═══════════════════════════════════════════════════════════════════════════
def figure1(df: pd.DataFrame, out_path: Path):
    x = df[GAP_COL].values
    y = df[AUC_COL].values

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # One dot per configuration — single colour, no grouping.
    ax.scatter(x, y, s=26, alpha=0.55, color="#3b6fb0", edgecolors="none")

    # Single fitted line through all points.
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=2, alpha=0.9)

    # Chance reference.
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Stats in the corner.
    rho, p_s = spearmanr(x, y)
    r, p_p   = pearsonr(x, y)
    R2 = r ** 2
    p_str = "< 0.001" if p_s < 0.001 else f"= {p_s:.3f}"
    ax.text(0.03, 0.97,
            f"Spearman ρ = {rho:.2f}\nR² = {R2:.2f}\np {p_str}\nn = {len(df)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))

    ax.set_xlabel("Train / test accuracy gap")
    ax.set_ylabel("MIA AUC")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png   (FIGURE 1)")


# ═══════════════════════════════════════════════════════════════════════════
#  Supporting figures — same scatter, coloured. Kept SEPARATE from Figure 1.
# ═══════════════════════════════════════════════════════════════════════════
def supp_colored(df: pd.DataFrame, out_path: Path, color_col: str, title: str):
    if color_col not in df.columns:
        print(f"⚠ {color_col} not in data, skipping {out_path.name}")
        return

    x_all = df[GAP_COL].values
    y_all = df[AUC_COL].values

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    palette = plt.cm.tab10.colors
    groups = list(dict.fromkeys(df[color_col].tolist()))
    cmap = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    for g in groups:
        m = df[color_col].values == g
        ax.scatter(df[GAP_COL].values[m], df[AUC_COL].values[m],
                   s=26, alpha=0.6, color=cmap[g], edgecolors="none", label=str(g))

    # Single pooled fit line (same as Figure 1) so the reader sees every
    # group lies along the *same* curve.
    slope, intercept = np.polyfit(x_all, y_all, 1)
    xs = np.linspace(x_all.min(), x_all.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=2, alpha=0.9,
            label="pooled fit")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Train / test accuracy gap")
    ax.set_ylabel("MIA AUC")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right", frameon=False)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",     default="mia_variation_results/raw_results")
    ap.add_argument("--out-dir", default="mia_variation_results/plots")
    args = ap.parse_args()

    csv_dir = Path(args.dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    df = load_all(csv_dir)

    # THE figure — exactly what the professor asked for.
    figure1(df, out_dir / "figure1_gap_vs_auc")

    # Separate supporting figures (not Figure 1).
    supp_colored(df, out_dir / "supp_gap_vs_auc_by_partitioner",
                 "partitioning", "Gap vs MIA AUC — coloured by partitioner")
    supp_colored(df, out_dir / "supp_gap_vs_auc_by_dataset",
                 "dataset", "Gap vs MIA AUC — coloured by dataset")

    print(f"\n✅ Done → {out_dir}")


if __name__ == "__main__":
    main()