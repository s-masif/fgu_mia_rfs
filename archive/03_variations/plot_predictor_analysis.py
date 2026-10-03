from __future__ import annotations

import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr, pearsonr

mpl.use("Agg")

AUC_COL = "shokri_mia_auc"
PRIMARY_KEY = ["dataset", "scenario", "seed",
               "partitioning", "num_clients", "forget_ratio"]

# Predictor column -> (axis label, x-scale)
PREDICTORS = {
    "train_test_gap": ("Train / test accuracy gap", "linear"),
    "n_nodes":        ("Number of nodes (log scale)", "log"),
    "n_classes":      ("Number of classes",           "linear"),
    "modularity":     ("Partition modularity Q",      "linear"),
}


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
        files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {csv_dir}")
    dfs = [pd.read_csv(f) for f in files]
    print(f"Loaded {len(files)} CSV(s):")
    for f, d in zip(files, dfs):
        print(f"  {Path(f).name}: {len(d)} rows")
    df = pd.concat(dfs, ignore_index=True)
    if all(c in df.columns for c in PRIMARY_KEY):
        df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
    df = df[~df[AUC_COL].isna()].copy()
    print(f"Usable rows (AUC present, deduped): {len(df)}")
    return df


def _fit_and_stats(ax, x, y, xscale):
    """Draw fitted line + stats box on ax. Fit is done in the plotted space
    (log-x if xscale=='log'). Returns (rho, R2)."""
    keep = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[keep], y[keep]
    if xscale == "log":
        x_fit = np.log10(x)
    else:
        x_fit = x

    # Spearman is scale-invariant; Pearson/R2 computed in the fit space.
    rho, p_s = spearmanr(x, y)
    r, _     = pearsonr(x_fit, y)
    R2 = r ** 2

    slope, intercept = np.polyfit(x_fit, y, 1)
    xs_fit = np.linspace(x_fit.min(), x_fit.max(), 100)
    xs_plot = 10 ** xs_fit if xscale == "log" else xs_fit
    ax.plot(xs_plot, slope * xs_fit + intercept, color="black",
            linewidth=2, alpha=0.9)

    p_str = "< 0.001" if p_s < 0.001 else f"= {p_s:.3f}"
    ax.text(0.03, 0.97,
            f"Spearman ρ = {rho:.2f}\nR² = {R2:.2f}\np {p_str}\nn = {keep.sum()}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))
    return rho, R2


def single_scatter(df, col, label, xscale, out_path):
    if col not in df.columns:
        print(f"⚠ {col} not in data, skipping {out_path.name}")
        return
    x = df[col].values.astype(float)
    y = df[AUC_COL].values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(x, y, s=26, alpha=0.55, color="#3b6fb0", edgecolors="none")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    if xscale == "log":
        ax.set_xscale("log")
    _fit_and_stats(ax, x, y, xscale)
    ax.set_xlabel(label)
    ax.set_ylabel("MIA AUC")
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


def compare_2x2(df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (col, (label, xscale)) in zip(axes, PREDICTORS.items()):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' missing", transform=ax.transAxes,
                    ha="center"); continue
        x = df[col].values.astype(float)
        y = df[AUC_COL].values.astype(float)
        ax.scatter(x, y, s=18, alpha=0.5, color="#3b6fb0", edgecolors="none")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        if xscale == "log":
            ax.set_xscale("log")
        _fit_and_stats(ax, x, y, xscale)
        ax.set_xlabel(label)
        ax.set_ylabel("MIA AUC")
    fig.suptitle("What predicts MIA AUC on the retrained model?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png   (comparison figure)")


def print_ranked_table(df):
    """Print correlations ranked so the reader sees gap wins."""
    print("\n" + "=" * 62)
    print(" Predictor vs MIA AUC — ranked by |Spearman rho|")
    print("=" * 62)
    rows = []
    y = df[AUC_COL].values.astype(float)
    for col, (label, xscale) in PREDICTORS.items():
        if col not in df.columns: continue
        x = df[col].values.astype(float)
        keep = ~np.isnan(x) & ~np.isnan(y)
        rho, p = spearmanr(x[keep], y[keep])
        xf = np.log10(x[keep]) if xscale == "log" else x[keep]
        r, _ = pearsonr(xf, y[keep])
        rows.append((label, rho, p, r**2, keep.sum()))
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"  {'Predictor':28s} {'ρ':>7s} {'p':>10s} {'R²':>7s} {'n':>5s}")
    print("  " + "-" * 58)
    for label, rho, p, R2, n in rows:
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"  {label:28s} {rho:+7.3f} {p_str:>10s} {R2:7.3f} {n:5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",     default="mia_variations_results_v2/raw_data")
    ap.add_argument("--out-dir", default="mia_variations_results_v2/plots")
    args = ap.parse_args()

    csv_dir = Path(args.dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = load_all(csv_dir)

    single_scatter(df, "train_test_gap", "Train / test accuracy gap", "linear",
                   out_dir / "pred_gap")
    single_scatter(df, "n_nodes",   "Number of nodes (log scale)", "log",
                   out_dir / "pred_nodes")
    single_scatter(df, "n_classes", "Number of classes",           "linear",
                   out_dir / "pred_classes")
    single_scatter(df, "modularity","Partition modularity Q",      "linear",
                   out_dir / "pred_modularity")
    compare_2x2(df, out_dir / "pred_compare_2x2")

    print_ranked_table(df)
    print(f"\n✅ Done → {out_dir}")


if __name__ == "__main__":
    main()
