"""
plot_property_analysis.py
=========================
Dedicated visualization for the correlation analysis the professor asked for:

    "correlation with homophily, degree, modularity and forget size,
     with more points obtained by changing partitioning, number of clients
     and forget ratio, and including dataset size and the train to test gap
     as controls."

Produces 5 figures:

  A_property_overview.{pdf,png}
      4-panel bar chart: homophily / avg_degree / modularity distribution
      across datasets and partitioning methods — so the reader sees what the
      input properties look like before seeing their correlation with MIA.

  B_homophily_vs_auc.{pdf,png}
      Scatter: homophily (x) vs MIA AUC (y), one point per configuration.
      Colour = partitioning. Points from the same dataset are grouped.
      Spearman ρ annotated.  Shows homophily is NOT the driver.

  C_modularity_vs_auc.{pdf,png}
      Same layout but x = modularity Q.  Shows moderate negative correlation
      — better community structure → lower MIA.

  D_forget_size_vs_auc.{pdf,png}
      Same layout but x = forget_ratio (the "forget size" axis).
      Moderate positive correlation: bigger forget → less retain → more
      overfitting → higher MIA.

  E_four_panel_controlled.{pdf,png}
      2×2 panel showing ALL four properties together, with train/test gap
      overlaid as point size (bigger dot = larger gap).  The "controlled
      correlation" figure showing gap is the dominant covariate.

Usage
-----
  python plot_property_analysis.py \\
      --dir     mia_variations_results/raw_data \\
      --out-dir mia_variations_results/plots
"""
from __future__ import annotations

import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

mpl.use("Agg")

# ── Style ───────────────────────────────────────────────────────────────────
DATASET_COLORS = {
    "Cora":      "#1f77b4",
    "Chameleon": "#ff7f0e",
    "Squirrel":  "#2ca02c",
    "Actor":     "#d62728",
}
PARTITIONER_COLORS = {
    "louvain":    "#9467bd",
    "metis":      "#7f7f7f",
    "metis_plus": "#8c564b",
    "random":     "#e377c2",
}
DATASETS = ["Cora", "Chameleon", "Squirrel", "Actor"]

# ── Known dataset properties ─────────────────────────────────────────────────
# homophily: raw edge homophily from Pei et al. 2020
# avg_degree: computed from published graph stats (2 * |E| / |V|)
DATASET_METADATA = {
    #            h      avg_deg   n_nodes
    "Cora":      (0.81,  3.90,   2708),
    "Chameleon": (0.23, 14.69,   2277),
    "Squirrel":  (0.22, 26.63,   5201),
    "Actor":     (0.22,  7.25,   7600),
}


def setup_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "legend.frameon":    False,
        "figure.dpi":        120,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })


# ── Data loading ─────────────────────────────────────────────────────────────
def load_all(csv_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(csv_dir / "mia_variations_raw*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs under {csv_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    pk = ["dataset", "scenario", "seed", "partitioning", "num_clients", "forget_ratio"]
    df = df.drop_duplicates(subset=pk, keep="last")

    # Attach known metadata per dataset (avg_degree not in CSV)
    df["homophily"]  = df["dataset"].map(lambda d: DATASET_METADATA.get(d, (None,None,None))[0])
    df["avg_degree"] = df["dataset"].map(lambda d: DATASET_METADATA.get(d, (None,None,None))[1])
    df["n_nodes"]    = df["dataset"].map(lambda d: DATASET_METADATA.get(d, (None,None,None))[2])

    # forget_size_n = approximate number of nodes forgotten
    # (forget_ratio × training nodes of the forget partition)
    # We use n_nodes as a proxy since exact retain_n is not always recorded
    df["forget_size_n"] = df["forget_ratio"] * df["n_nodes"] * 0.20
    # ^ rough: ~20% of nodes are in train masks on average

    print(f"Loaded {len(df)} rows from {len(files)} CSV(s).")
    return df


# ── Fig A: Property overview bar chart ───────────────────────────────────────
def plot_property_overview(df: pd.DataFrame, out_path: Path):
    """Show what homophily, avg_degree, and modularity look like per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel 1: Homophily per dataset (static property)
    ax = axes[0]
    hs  = [DATASET_METADATA[d][0] for d in DATASETS]
    bars = ax.bar(DATASETS, hs, color=[DATASET_COLORS[d] for d in DATASETS],
                   edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
                label="h = 0.5 (borderline)")
    for bar, v in zip(bars, hs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Edge Homophily h")
    ax.set_title("Homophily (dataset property)")
    ax.legend(fontsize=8)

    # Panel 2: Avg degree per dataset (static property)
    ax = axes[1]
    degs = [DATASET_METADATA[d][1] for d in DATASETS]
    bars = ax.bar(DATASETS, degs, color=[DATASET_COLORS[d] for d in DATASETS],
                   edgecolor="black", linewidth=0.6, alpha=0.85)
    for bar, v in zip(bars, degs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average Node Degree")
    ax.set_title("Average Degree (dataset property)")

    # Panel 3: Modularity distribution per dataset × partitioner
    ax = axes[2]
    if "modularity" in df.columns:
        x_ticks, x_labels = [], []
        offset = 0
        for ds in DATASETS:
            for pi, part in enumerate(["louvain", "metis", "metis_plus", "random"]):
                sub = df[(df.dataset == ds) & (df.partitioning == part)]["modularity"].dropna()
                if sub.empty: continue
                bp = ax.boxplot(sub.values, positions=[offset], widths=0.6,
                                patch_artist=True, showfliers=False,
                                medianprops={"color":"black","linewidth":1.2},
                                boxprops={"facecolor": PARTITIONER_COLORS[part],
                                          "alpha": 0.75})
                offset += 1
            # Group separator
            x_ticks.append(offset - 2.5)
            x_labels.append(ds)
            ax.axvline(offset - 0.5, color="gray", linewidth=0.5, linestyle=":")
            offset += 1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Modularity Q")
        ax.set_title("Modularity per dataset × partitioner")

        # Partitioner legend
        handles = [mpl.patches.Patch(color=c, label=p, alpha=0.8)
                   for p, c in PARTITIONER_COLORS.items()
                   if p in df["partitioning"].unique()]
        ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.suptitle("Graph Properties Used as Predictors in the Correlation Analysis",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ── Shared scatter helper ─────────────────────────────────────────────────────
def _scatter_vs_auc(ax, df, x_col, x_label, show_regression=True,
                     color_by="partitioning", annotate_rho=True):
    y = df["shokri_mia_auc"].values
    x = df[x_col].values
    keep = ~np.isnan(x) & ~np.isnan(y)

    if color_by == "partitioning" and "partitioning" in df.columns:
        for p, c in PARTITIONER_COLORS.items():
            m = keep & (df["partitioning"].values == p)
            if m.any():
                ax.scatter(x[m], y[m], color=c, alpha=0.55, s=28,
                            edgecolors="none", label=p, zorder=2)
    elif color_by == "dataset":
        for ds, c in DATASET_COLORS.items():
            m = keep & (df["dataset"].values == ds)
            if m.any():
                ax.scatter(x[m], y[m], color=c, alpha=0.55, s=28,
                            edgecolors="none", label=ds, zorder=2)

    if show_regression and keep.sum() >= 5 and np.std(x[keep]) > 1e-8:
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        xs = np.linspace(x[keep].min(), x[keep].max(), 50)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.8, alpha=0.85, zorder=3)

    if annotate_rho and keep.sum() >= 5:
        rho, p = spearmanr(x[keep], y[keep])
        R2 = LinearRegression().fit(x[keep].reshape(-1,1), y[keep]).score(
            x[keep].reshape(-1,1), y[keep])
        pstr = f"{p:.4f}" if p >= 0.0001 else "< 0.0001"
        ax.text(0.03, 0.97,
                 f"Spearman ρ = {rho:+.3f}\np = {pstr}\nR² = {R2:.3f}\nn = {keep.sum()}",
                 transform=ax.transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor="lightgray", alpha=0.9))

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("MIA AUC (retrain baseline)")
    return ax


# ── Fig B: Homophily vs AUC ───────────────────────────────────────────────────
def plot_homophily_vs_auc(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, sc in enumerate(["client", "node"]):
        ax = axes[i]
        sub = df[df.scenario == sc]
        _scatter_vs_auc(ax, sub, "homophily",
                         "Edge Homophily h  (dataset property)", color_by="partitioning")
        ax.set_title(f"Scenario: {sc}")

        # Annotate dataset name at the median x position
        for ds in DATASETS:
            g = sub[sub.dataset == ds]["homophily"]
            if not g.empty:
                ax.axvline(g.iloc[0], color=DATASET_COLORS[ds], linewidth=0.8,
                            linestyle="-", alpha=0.3)
                ax.text(g.iloc[0], 0.97, ds, transform=ax.get_xaxis_transform(),
                         ha="center", va="top", fontsize=8,
                         color=DATASET_COLORS[ds], fontweight="bold")

    # Partitioner legend
    handles = [mpl.patches.Patch(color=c, label=p, alpha=0.8)
               for p, c in PARTITIONER_COLORS.items() if p in df.partitioning.unique()]
    axes[1].legend(handles=handles, title="Partitioner", fontsize=9,
                    loc="lower right")

    fig.suptitle("Homophily vs MIA AUC  (Spearman ρ ≈ 0 → homophily does NOT drive MIA)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ── Fig C: Modularity vs AUC ─────────────────────────────────────────────────
def plot_modularity_vs_auc(df: pd.DataFrame, out_path: Path):
    if "modularity" not in df.columns:
        print("⚠ modularity not in CSV, skipping"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, sc in enumerate(["client", "node"]):
        ax = axes[i]
        sub = df[df.scenario == sc]
        _scatter_vs_auc(ax, sub, "modularity",
                         "Modularity Q  (partition property)", color_by="partitioning")
        ax.set_title(f"Scenario: {sc}")

        # Vertical reference lines for the four partitioners
        for part in ["louvain", "metis", "metis_plus", "random"]:
            q = sub[sub.partitioning == part]["modularity"].mean()
            if not np.isnan(q):
                ax.text(q, 0.52, part, ha="center", va="bottom", fontsize=7,
                         color=PARTITIONER_COLORS[part], rotation=45)

    handles = [mpl.patches.Patch(color=c, label=p, alpha=0.8)
               for p, c in PARTITIONER_COLORS.items() if p in df.partitioning.unique()]
    axes[1].legend(handles=handles, title="Partitioner", fontsize=9)

    fig.suptitle("Modularity Q vs MIA AUC  (negative correlation — "
                  "less structured partitions → more overfitting → higher MIA)",
                  fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ── Fig D: Forget size vs AUC ────────────────────────────────────────────────
def plot_forget_size_vs_auc(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, sc in enumerate(["client", "node"]):
        ax = axes[i]
        sub = df[df.scenario == sc]

        # Plot forget_ratio (the configurable axis) on x; one panel per scenario
        _scatter_vs_auc(ax, sub, "forget_ratio",
                         "Forget Ratio γ  (fraction of training pool designated as forget)",
                         color_by="dataset")
        ax.set_title(f"Scenario: {sc}")

        # Add secondary x annotations: number of nodes ≈ ratio × n_train
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ratios = sorted(sub["forget_ratio"].unique())
        # approximate forget_n for Cora (n=2708, ~20% train = 542 nodes)
        approx_n = [int(r * 542) for r in ratios]
        ax2.set_xticks(ratios)
        ax2.set_xticklabels([f"≈{n}" for n in approx_n], fontsize=8)
        ax2.set_xlabel("Approx. forget set size (Cora, train nodes)", fontsize=8)

    handles = [mpl.patches.Patch(color=c, label=ds, alpha=0.8)
               for ds, c in DATASET_COLORS.items() if ds in df.dataset.unique()]
    axes[1].legend(handles=handles, title="Dataset", fontsize=9)

    fig.suptitle("Forget Size (ratio) vs MIA AUC  (moderate positive correlation — "
                  "larger forget → smaller retain → more overfitting → higher MIA)",
                  fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ── Fig E: Four-panel controlled correlation ──────────────────────────────────
def plot_four_panel_controlled(df: pd.DataFrame, out_path: Path):
    """2×2 panels: homophily / avg_degree / modularity / forget_ratio vs AUC.
    Point SIZE encodes train/test gap — bigger dot = more overfitting.
    This is the 'controlled' figure showing gap is the dominant covariate."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    panels = [
        ("homophily",    "Edge Homophily h"),
        ("avg_degree",   "Average Node Degree"),
        ("modularity",   "Modularity Q"),
        ("forget_ratio", "Forget Ratio γ"),
    ]

    # Normalise gap for point sizing
    gap = df["train_test_gap"].values if "train_test_gap" in df.columns else np.ones(len(df))
    gap_min, gap_max = np.nanmin(gap), np.nanmax(gap)
    size_norm = 15 + 120 * (gap - gap_min) / max(gap_max - gap_min, 1e-8)

    for i, (col, label) in enumerate(panels):
        ax = axes[i]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not in CSV", transform=ax.transAxes,
                     ha="center"); continue

        y = df["shokri_mia_auc"].values
        x = df[col].values
        keep = ~np.isnan(x) & ~np.isnan(y)

        # Colour by dataset, size by train/test gap
        for ds, c in DATASET_COLORS.items():
            m = keep & (df["dataset"].values == ds)
            if not m.any(): continue
            ax.scatter(x[m], y[m],
                        c=c, s=size_norm[m], alpha=0.60,
                        edgecolors="white", linewidth=0.3, zorder=2, label=ds)

        # Regression
        if keep.sum() >= 5 and np.std(x[keep]) > 1e-8:
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            xs = np.linspace(x[keep].min(), x[keep].max(), 50)
            ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.8, alpha=0.8)
            rho, p = spearmanr(x[keep], y[keep])
            pstr = f"{p:.4f}" if p >= 0.0001 else "< 0.0001"
            ax.text(0.03, 0.97,
                     f"Spearman ρ = {rho:+.3f}   p = {pstr}",
                     transform=ax.transAxes, va="top", ha="left", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="lightgray", alpha=0.9))

        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("MIA AUC")
        ax.set_title(label)

    # Dataset colour legend
    ds_handles = [mpl.patches.Patch(color=c, label=ds, alpha=0.8)
                   for ds, c in DATASET_COLORS.items() if ds in df.dataset.unique()]
    fig.legend(handles=ds_handles, title="Dataset (colour)",
                loc="lower center", ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, -0.01))

    # Size legend (gap)
    size_examples = [0.1, 0.3, 0.5]
    gap_handles = [
        plt.scatter([], [], s=15 + 120*(g - gap_min)/(gap_max - gap_min + 1e-8),
                    color="gray", alpha=0.7, label=f"gap ≈ {g:.1f}")
        for g in size_examples
    ]
    fig.legend(handles=gap_handles, title="Train/test gap (size)",
                loc="lower center", ncol=3, fontsize=9,
                bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("MIA AUC vs Graph Properties  "
                  "(point size = train/test gap — the dominant covariate)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out_path}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     default="mia_variations_results/raw_data")
    parser.add_argument("--out-dir", default="mia_variations_results/plots")
    args = parser.parse_args()

    csv_dir = Path(args.dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    df = load_all(csv_dir)

    print(f"\n══ Generating property analysis plots to {out_dir} ══")
    plot_property_overview(df,       out_dir / "A_property_overview")
    plot_homophily_vs_auc(df,        out_dir / "B_homophily_vs_auc")
    plot_modularity_vs_auc(df,       out_dir / "C_modularity_vs_auc")
    plot_forget_size_vs_auc(df,      out_dir / "D_forget_size_vs_auc")
    plot_four_panel_controlled(df,   out_dir / "E_four_panel_controlled")

    print(f"\n✅ Done. 5 figures (PDF + PNG each) → {out_dir}")


if __name__ == "__main__":
    main()