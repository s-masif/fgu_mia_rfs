# # # """
# # # plot_variations.py
# # # ==================
# # # Comprehensive plotting for the Phase 2 variation sweep.

# # # Reads all all_variations_*.csv files under --dir and produces:

# # #   01_sensitivity_partitioning.{pdf,png}    2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001
# # #   02_sensitivity_num_clients.{pdf,png}     Same, sweeping K
# # #   03_sensitivity_forget_ratio.{pdf,png}    Same, sweeping forget ratio
# # #   04_correlation_scatters.{pdf,png}        Grid of scatter plots for candidate predictors
# # #   05_headline_gap_vs_auc.{pdf,png}         Single big scatter — the paper's main finding
# # #   06_distribution_boxplots.{pdf,png}       Per-dataset AUC distributions across all configs
# # #   07_regression_r2.{pdf,png}               Bar chart of R² per predictor set

# # # Usage
# # # -----
# # #   python plot_variations.py \\
# # #       --dir     variation_all_results \\
# # #       --out-dir variation_all_results/plots
# # # """
# # # from __future__ import annotations

# # # import argparse, glob
# # # from pathlib import Path

# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import matplotlib as mpl
# # # from scipy.stats import spearmanr

# # # # Force non-interactive backend so this runs cleanly on a server / in tmux
# # # mpl.use("Agg")


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  Style
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # DATASET_COLORS = {
# # #     "Cora":      "#1f77b4",
# # #     "Chameleon": "#ff7f0e",
# # #     "Squirrel":  "#2ca02c",
# # #     "Actor":     "#d62728",
# # # }

# # # def _dataset_color(ds, _cache={}):
# # #     """Return a stable colour for any dataset name, including *_filtered.
# # #     Known names use DATASET_COLORS; unknown names get a palette colour."""
# # #     if ds in DATASET_COLORS:
# # #         return DATASET_COLORS[ds]
# # #     # map a *_filtered name to its base colour if the base is known
# # #     base = ds.replace("_filtered", "")
# # #     if base in DATASET_COLORS:
# # #         return DATASET_COLORS[base]
# # #     if ds not in _cache:
# # #         palette = plt.cm.tab10.colors
# # #         _cache[ds] = palette[len(_cache) % len(palette)]
# # #     return _cache[ds]

# # # def _datasets_in(df):
# # #     """Datasets actually present, in a stable order."""
# # #     return sorted(df["dataset"].unique())

# # # PARTITIONER_COLORS = {
# # #     "louvain":    "#9467bd",
# # #     "metis":      "#7f7f7f",
# # #     "metis_plus": "#8c564b",
# # #     "random":     "#e377c2",
# # # }
# # # SCENARIO_STYLE = {"client": "-", "node": "--"}
# # # METRIC_LABELS = {
# # #     "shokri_mia_auc":              "MIA AUC",
# # #     "shokri_mia_balanced_acc":     "Balanced Accuracy",
# # #     "shokri_mia_tpr_at_fpr_0.01":  "TPR @ FPR ≤ 0.01",
# # #     "shokri_mia_tpr_at_fpr_0.001": "TPR @ FPR ≤ 0.001",
# # # }
# # # FOUR_METRICS = list(METRIC_LABELS.keys())


# # # def setup_style():
# # #     plt.rcParams.update({
# # #         "font.family":       "sans-serif",
# # #         "font.size":         10,
# # #         "axes.titlesize":    11,
# # #         "axes.labelsize":    10,
# # #         "axes.spines.top":   False,
# # #         "axes.spines.right": False,
# # #         "axes.grid":         True,
# # #         "grid.alpha":        0.3,
# # #         "grid.linestyle":    "-",
# # #         "legend.frameon":    False,
# # #         "figure.dpi":        120,
# # #         "savefig.dpi":       200,
# # #         "savefig.bbox":      "tight",
# # #     })


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  Data loading
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # PRIMARY_KEY = ["dataset", "scenario", "seed",
# # #                "partitioning", "num_clients", "forget_ratio"]


# # # def _read_csv_aligned(path) -> pd.DataFrame:
# # #     """Read a CSV whose data rows may carry extra *trailing* fields beyond the
# # #     header width (e.g. stray trailing commas from a later writer version).
# # #     Keeps only the header's columns; extra trailing empties are dropped and
# # #     blank cells become NaN. Verified lossless for header-aligned rows."""
# # #     cols = pd.read_csv(path, nrows=0).columns.tolist()
# # #     return pd.read_csv(path, header=0, names=cols, usecols=range(len(cols)))


# # # def load_all(csv_dir: Path) -> pd.DataFrame:
# # #     files = sorted(glob.glob(str(csv_dir / "all_variations_*.csv")))
# # #     if not files:
# # #         raise FileNotFoundError(f"No CSVs found under {csv_dir}")
# # #     dfs = [_read_csv_aligned(f) for f in files]
# # #     print(f"Loaded {len(files)} CSV(s):")
# # #     for f, df in zip(files, dfs):
# # #         print(f"  {Path(f).name}: {len(df)} rows")
# # #     df = pd.concat(dfs, ignore_index=True)
# # #     df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
# # #     print(f"After dedup: {len(df)} rows")
# # #     return df


# # # def load_sweep(csv_dir: Path, sweep_name: str) -> pd.DataFrame:
# # #     path = csv_dir / f"all_variations_{sweep_name}.csv"
# # #     if not path.exists():
# # #         print(f"⚠ Missing sweep CSV: {path.name}")
# # #         return pd.DataFrame()
# # #     return _read_csv_aligned(path)


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  1-3. Sensitivity plots — one axis per figure, 2×2 grid of metrics
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # def plot_sensitivity(df: pd.DataFrame, axis: str, axis_label: str,
# # #                       out_path: Path, is_categorical: bool = False):
# # #     """2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001 vs. `axis`.
# # #     One line per (dataset, scenario)."""
# # #     if df.empty:
# # #         print(f"⚠ empty df for {axis} sensitivity, skipping")
# # #         return

# # #     metrics = [m for m in FOUR_METRICS if m in df.columns]
# # #     if not metrics:
# # #         print(f"⚠ no metric columns present for {axis}")
# # #         return

# # #     fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
# # #     axes = axes.flatten()

# # #     axis_vals = sorted(df[axis].unique())

# # #     for i, metric in enumerate(metrics):
# # #         ax = axes[i]
# # #         for ds in _datasets_in(df):
# # #             for sc in ["client", "node"]:
# # #                 sub = df[(df.dataset == ds) & (df.scenario == sc)]
# # #                 if sub.empty: continue
# # #                 agg = sub.groupby(axis)[metric].agg(["mean", "std"]).reindex(axis_vals)
# # #                 x = np.arange(len(axis_vals)) if is_categorical else np.asarray(axis_vals, dtype=float)
# # #                 ax.errorbar(
# # #                     x, agg["mean"], yerr=agg["std"],
# # #                     color=_dataset_color(ds),
# # #                     linestyle=SCENARIO_STYLE[sc],
# # #                     marker="o", markersize=4, capsize=3, linewidth=1.4,
# # #                     label=f"{ds} ({sc})",
# # #                 )
# # #         ax.set_title(METRIC_LABELS[metric])
# # #         ax.set_xlabel(axis_label)
# # #         if is_categorical:
# # #             ax.set_xticks(np.arange(len(axis_vals)))
# # #             ax.set_xticklabels(axis_vals, rotation=15, ha="right")
# # #         # Reference lines
# # #         if metric == "shokri_mia_auc" or metric == "shokri_mia_balanced_acc":
# # #             ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
# # #         elif metric.startswith("shokri_mia_tpr_at_fpr"):
# # #             # A random attacker's TPR at FPR α is α itself
# # #             α = float(metric.rsplit("_", 1)[-1])
# # #             ax.axhline(α, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
# # #                         label="_random baseline")

# # #     # Legend outside on the right
# # #     handles, labels = axes[0].get_legend_handles_labels()
# # #     fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.14, 0.5),
# # #                fontsize=9, title="Dataset (scenario)")
# # #     fig.suptitle(f"Sensitivity to {axis_label}", fontsize=13, fontweight="bold")
# # #     fig.tight_layout(rect=[0, 0, 0.98, 0.97])
# # #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# # #     fig.savefig(str(out_path) + ".png")
# # #     plt.close(fig)
# # #     print(f"→ {out_path}.pdf / .png")


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  4. Correlation scatter grid — MIA AUC vs each candidate predictor
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # CANDIDATE_PREDICTORS = [
# # #     ("train_test_gap", "Train/Test Accuracy Gap"),
# # #     ("modularity",     "Modularity Q"),
# # #     ("homophily",      "Homophily"),
# # #     ("n_nodes",        "Total Nodes"),
# # #     ("num_clients",    "Number of Clients K"),
# # #     ("forget_ratio",   "Forget Ratio"),
# # # ]


# # # def plot_correlation_grid(df: pd.DataFrame, out_path: Path):
# # #     """3×2 grid: MIA AUC vs each predictor, one point per (dataset, config)."""
# # #     preds = [(c, l) for c, l in CANDIDATE_PREDICTORS if c in df.columns]
# # #     if not preds:
# # #         print("⚠ no predictor columns present"); return

# # #     rows, cols = 3, 2
# # #     fig, axes = plt.subplots(rows, cols, figsize=(11, 12))
# # #     axes = axes.flatten()

# # #     y = df["shokri_mia_auc"].values

# # #     for i, (col, label) in enumerate(preds):
# # #         ax = axes[i]
# # #         x = df[col].values
# # #         keep = ~np.isnan(x) & ~np.isnan(y)

# # #         # Colour by partitioning if available
# # #         if "partitioning" in df.columns:
# # #             for p, colour in PARTITIONER_COLORS.items():
# # #                 m = keep & (df["partitioning"].values == p)
# # #                 if m.any():
# # #                     ax.scatter(x[m], y[m], color=colour, alpha=0.55, s=22,
# # #                                 edgecolors="none", label=p)
# # #         else:
# # #             ax.scatter(x[keep], y[keep], alpha=0.55, s=22, edgecolors="none")

# # #         # Linear fit
# # #         if keep.sum() >= 5 and np.std(x[keep]) > 1e-8:
# # #             slope, intercept = np.polyfit(x[keep], y[keep], 1)
# # #             xs = np.linspace(x[keep].min(), x[keep].max(), 50)
# # #             ax.plot(xs, slope * xs + intercept, color="black",
# # #                      linewidth=1.5, alpha=0.8, linestyle="--")
# # #             rho, p = spearmanr(x[keep], y[keep])
# # #             ax.text(0.02, 0.95,
# # #                      f"Spearman ρ = {rho:+.3f}\np = {p:.4f}\nn = {keep.sum()}",
# # #                      transform=ax.transAxes, va="top", ha="left", fontsize=9,
# # #                      bbox=dict(boxstyle="round,pad=0.35",
# # #                                facecolor="white", edgecolor="lightgray", alpha=0.9))

# # #         ax.set_xlabel(label)
# # #         ax.set_ylabel("MIA AUC")
# # #         ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)

# # #     # Hide unused
# # #     for j in range(len(preds), rows * cols):
# # #         axes[j].set_visible(False)

# # #     # Common partitioner legend
# # #     if "partitioning" in df.columns:
# # #         handles = [mpl.patches.Patch(color=c, label=p)
# # #                     for p, c in PARTITIONER_COLORS.items()
# # #                     if p in df["partitioning"].unique()]
# # #         fig.legend(handles=handles, loc="lower right",
# # #                     bbox_to_anchor=(0.98, 0.02), title="Partitioner",
# # #                     fontsize=9, ncol=len(handles))

# # #     fig.suptitle("MIA AUC vs. Candidate Predictors (n = {} across sweep configs)".format(len(df)),
# # #                   fontsize=13, fontweight="bold")
# # #     fig.tight_layout(rect=[0, 0.04, 1, 0.97])
# # #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# # #     fig.savefig(str(out_path) + ".png")
# # #     plt.close(fig)
# # #     print(f"→ {out_path}.pdf / .png")


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  5. Headline scatter — the paper's main finding in one figure
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # def plot_headline_gap_vs_auc(df: pd.DataFrame, out_path: Path):
# # #     """Train/test gap vs MIA AUC — colored by partitioning, marker per dataset.
# # #     The paper's headline figure."""
# # #     if "train_test_gap" not in df.columns:
# # #         print("⚠ train_test_gap missing"); return

# # #     fig, ax = plt.subplots(figsize=(9, 6.5))

# # #     _marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*"]
# # #     markers = {ds: _marker_pool[i % len(_marker_pool)]
# # #                for i, ds in enumerate(_datasets_in(df))}

# # #     for ds in _datasets_in(df):
# # #         for p in ["louvain", "metis", "metis_plus", "random"]:
# # #             sub = df[(df.dataset == ds) & (df.partitioning == p)]
# # #             if sub.empty: continue
# # #             ax.scatter(
# # #                 sub["train_test_gap"], sub["shokri_mia_auc"],
# # #                 marker=markers[ds], color=PARTITIONER_COLORS.get(p, "gray"),
# # #                 alpha=0.65, s=45, edgecolors="none",
# # #             )

# # #     # Global regression line
# # #     x = df["train_test_gap"].values; y = df["shokri_mia_auc"].values
# # #     keep = ~np.isnan(x) & ~np.isnan(y)
# # #     slope, intercept = np.polyfit(x[keep], y[keep], 1)
# # #     xs = np.linspace(x[keep].min(), x[keep].max(), 50)
# # #     ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9,
# # #              label=f"OLS fit (slope = {slope:+.2f})")

# # #     rho, p = spearmanr(x[keep], y[keep])
# # #     from sklearn.linear_model import LinearRegression
# # #     R2 = LinearRegression().fit(x[keep].reshape(-1, 1), y[keep]).score(
# # #         x[keep].reshape(-1, 1), y[keep])
# # #     ax.text(0.02, 0.98,
# # #              f"Spearman ρ = {rho:+.3f}\np < {max(p, 1e-16):.1e}\nR² = {R2:.3f}\nn = {keep.sum()}",
# # #              transform=ax.transAxes, va="top", ha="left", fontsize=10,
# # #              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
# # #                        edgecolor="lightgray", alpha=0.92))

# # #     ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
# # #     ax.set_xlabel("Train / Test Accuracy Gap  (retrain baseline)")
# # #     ax.set_ylabel("MIA AUC")
# # #     ax.set_title("MIA AUC tracks overfitting, not unlearning quality",
# # #                   fontsize=12, fontweight="bold")

# # #     # Custom legends: partitioner (colour) + dataset (marker)
# # #     p_handles = [plt.Line2D([0], [0], marker="s", color="w",
# # #                               markerfacecolor=c, markersize=9, label=p)
# # #                   for p, c in PARTITIONER_COLORS.items() if p in df.partitioning.unique()]
# # #     ds_handles = [plt.Line2D([0], [0], marker=m, color="gray",
# # #                                markersize=9, linestyle="", label=ds)
# # #                    for ds, m in markers.items()]

# # #     leg1 = ax.legend(handles=p_handles, title="Partitioner",
# # #                        loc="lower right", bbox_to_anchor=(1.0, 0.0),
# # #                        fontsize=9)
# # #     ax.add_artist(leg1)
# # #     ax.legend(handles=ds_handles, title="Dataset",
# # #                 loc="lower right", bbox_to_anchor=(0.82, 0.0),
# # #                 fontsize=9)

# # #     fig.tight_layout()
# # #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# # #     fig.savefig(str(out_path) + ".png")
# # #     plt.close(fig)
# # #     print(f"→ {out_path}.pdf / .png")


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  6. Distribution boxplots — how much MIA AUC varies within each dataset
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # def plot_boxplots(df: pd.DataFrame, out_path: Path):
# # #     """Per-dataset MIA AUC boxplot across all sweep configs. Shows the range
# # #     of MIA behavior on the retrained baseline — the paper's variability story."""
# # #     fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

# # #     for i, sc in enumerate(["client", "node"]):
# # #         ax = axes[i]
# # #         data, colors = [], []
# # #         labels = []
# # #         for ds in _datasets_in(df):
# # #             sub = df[(df.dataset == ds) & (df.scenario == sc)]
# # #             if sub.empty: continue
# # #             data.append(sub["shokri_mia_auc"].values)
# # #             colors.append(_dataset_color(ds))
# # #             labels.append(ds)

# # #         bp = ax.boxplot(data, patch_artist=True, showmeans=True,
# # #                           meanprops={"marker": "o", "markerfacecolor": "white",
# # #                                     "markeredgecolor": "black", "markersize": 5},
# # #                           medianprops={"color": "black", "linewidth": 1.4},
# # #                           widths=0.55)
# # #         for patch, colour in zip(bp["boxes"], colors):
# # #             patch.set_facecolor(colour); patch.set_alpha(0.65)

# # #         # Scatter the individual points on top
# # #         for j, arr in enumerate(data, 1):
# # #             jitter = np.random.default_rng(0).normal(0, 0.06, size=len(arr))
# # #             ax.scatter(j + jitter, arr, color=colors[j - 1],
# # #                         alpha=0.35, s=14, edgecolors="none", zorder=3)

# # #         ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
# # #         ax.set_xticks(range(1, len(labels) + 1))
# # #         ax.set_xticklabels(labels)
# # #         ax.set_ylabel("MIA AUC" if i == 0 else "")
# # #         ax.set_title(f"Scenario: {sc}")

# # #     fig.suptitle("MIA AUC distribution across all sweep configurations",
# # #                   fontsize=13, fontweight="bold")
# # #     fig.tight_layout(rect=[0, 0, 1, 0.96])
# # #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# # #     fig.savefig(str(out_path) + ".png")
# # #     plt.close(fig)
# # #     print(f"→ {out_path}.pdf / .png")


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  7. Regression R² bar chart — how much variance each predictor set explains
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # def plot_regression_r2(df: pd.DataFrame, out_path: Path):
# # #     """Bar chart of R² for different predictor combinations."""
# # #     try:
# # #         from sklearn.linear_model import LinearRegression
# # #     except ImportError:
# # #         print("⚠ sklearn missing, skipping regression plot"); return

# # #     y = df["shokri_mia_auc"].values
# # #     keep_y = ~np.isnan(y); y = y[keep_y]

# # #     def _r2(cols):
# # #         if not all(c in df.columns for c in cols): return None
# # #         X = df[cols].values[keep_y]
# # #         m = ~np.isnan(X).any(axis=1)
# # #         if m.sum() < 5: return None
# # #         return LinearRegression().fit(X[m], y[m]).score(X[m], y[m])

# # #     models = [
# # #         ("Train/test gap only",     ["train_test_gap"]),
# # #         ("Homophily only",          ["homophily"]),
# # #         ("Dataset size only",       ["n_nodes"]),
# # #         ("Modularity only",         ["modularity"]),
# # #         ("Gap + homophily",         ["train_test_gap", "homophily"]),
# # #         ("Gap + size",              ["train_test_gap", "n_nodes"]),
# # #         ("Gap + modularity",        ["train_test_gap", "modularity"]),
# # #         ("All predictors",          ["train_test_gap", "homophily", "n_nodes",
# # #                                       "modularity", "forget_ratio", "num_clients"]),
# # #     ]
# # #     labels, r2s = [], []
# # #     for name, cols in models:
# # #         r2 = _r2(cols)
# # #         if r2 is not None:
# # #             labels.append(name); r2s.append(r2)

# # #     fig, ax = plt.subplots(figsize=(9, 5.5))
# # #     y_pos = np.arange(len(labels))
# # #     colors = ["#1f77b4" if "gap" in l.lower() else "#c0c0c0" for l in labels]
# # #     colors[-1] = "#2ca02c"     # highlight "all predictors"

# # #     bars = ax.barh(y_pos, r2s, color=colors, edgecolor="black",
# # #                      linewidth=0.6, alpha=0.85)
# # #     ax.set_yticks(y_pos); ax.set_yticklabels(labels)
# # #     ax.invert_yaxis()
# # #     ax.set_xlabel("R² for predicting MIA AUC")
# # #     ax.set_xlim(0, max(r2s) * 1.15)
# # #     for bar, r2 in zip(bars, r2s):
# # #         ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
# # #                  f"{r2:.3f}", va="center", fontsize=9)
# # #     ax.set_title(f"Variance in MIA AUC explained by predictor sets (n = {int(keep_y.sum())})",
# # #                   fontsize=12, fontweight="bold")
# # #     fig.tight_layout()
# # #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# # #     fig.savefig(str(out_path) + ".png")
# # #     plt.close(fig)
# # #     print(f"→ {out_path}.pdf / .png")


# # # # ═══════════════════════════════════════════════════════════════════════════
# # # #  Main
# # # # ═══════════════════════════════════════════════════════════════════════════

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--dir",     default="variation_all_results",
# # #                         help="Directory containing all_variations_*.csv")
# # #     parser.add_argument("--out-dir", default="variation_all_results/plots",
# # #                         help="Directory to write PDF+PNG plots")
# # #     args = parser.parse_args()

# # #     csv_dir = Path(args.dir)
# # #     out_dir = Path(args.out_dir)
# # #     out_dir.mkdir(parents=True, exist_ok=True)

# # #     setup_style()

# # #     print(f"\n══ Loading data ══")
# # #     df_all = load_all(csv_dir)

# # #     df_partition = load_sweep(csv_dir, "partitioning")
# # #     df_num_clients = load_sweep(csv_dir, "num_clients")
# # #     df_forget_ratio = load_sweep(csv_dir, "forget_ratio")

# # #     print(f"\n══ Generating plots to {out_dir} ══")

# # #     # 1-3. Sensitivity plots (one per axis)
# # #     plot_sensitivity(df_partition,   "partitioning",  "Partitioning method",
# # #                        out_dir / "01_sensitivity_partitioning",
# # #                        is_categorical=True)
# # #     plot_sensitivity(df_num_clients, "num_clients",   "Number of Clients K",
# # #                        out_dir / "02_sensitivity_num_clients",
# # #                        is_categorical=False)
# # #     plot_sensitivity(df_forget_ratio, "forget_ratio", "Forget Ratio",
# # #                        out_dir / "03_sensitivity_forget_ratio",
# # #                        is_categorical=False)

# # #     # 4-7. Aggregate analyses across all rows
# # #     plot_correlation_grid(df_all, out_dir / "04_correlation_scatters")
# # #     plot_headline_gap_vs_auc(df_all, out_dir / "05_headline_gap_vs_auc")
# # #     plot_boxplots(df_all, out_dir / "06_distribution_boxplots")
# # #     plot_regression_r2(df_all, out_dir / "07_regression_r2")

# # #     print(f"\n✅ Done. Plots → {out_dir}")


# # # if __name__ == "__main__":
# # #     main()


# # """
# # plot_variations.py
# # ==================
# # Comprehensive plotting for the Phase 2 variation sweep.

# # Reads all all_variations_*.csv files under --dir and produces:

# #   01_sensitivity_partitioning.{pdf,png}    2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001
# #   02_sensitivity_num_clients.{pdf,png}     Same, sweeping K
# #   03_sensitivity_forget_ratio.{pdf,png}    Same, sweeping forget ratio
# #   04_correlation_scatters.{pdf,png}        Grid of scatter plots for candidate predictors
# #   05_headline_gap_vs_auc.{pdf,png}         Single big scatter — the paper's main finding
# #   06_distribution_boxplots.{pdf,png}       Per-dataset AUC distributions across all configs
# #   07_regression_r2.{pdf,png}               Bar chart of R² per predictor set

# # Usage
# # -----
# #   python plot_variations.py \\
# #       --dir     variation_all_results \\
# #       --out-dir variation_all_results/plots
# # """
# # from __future__ import annotations

# # import argparse, glob
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib as mpl
# # from scipy.stats import spearmanr

# # # Force non-interactive backend so this runs cleanly on a server / in tmux
# # mpl.use("Agg")


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  Style
# # # ═══════════════════════════════════════════════════════════════════════════

# # DATASET_COLORS = {
# #     "Cora":      "#1f77b4",
# #     "Chameleon": "#ff7f0e",
# #     "Squirrel":  "#2ca02c",
# #     "Actor":     "#d62728",
# # }

# # # 12 well-separated colours — enough to give every dataset a distinct hue.
# # # (tab10 only has 10 and its first 4 collided with the hardcoded names,
# # #  so several datasets shared a colour; this fixes that.)
# # DISTINCT_PALETTE = [
# #     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
# #     "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
# # ]

# # def _dataset_color(ds, _cache={}):
# #     """Return a stable, distinct colour per dataset. Colours are assigned in
# #     first-seen order; since callers iterate datasets in sorted order the
# #     mapping is deterministic within a run."""
# #     base = ds.replace("_filtered", "")   # a *_filtered variant shares its base's colour
# #     if base not in _cache:
# #         _cache[base] = DISTINCT_PALETTE[len(_cache) % len(DISTINCT_PALETTE)]
# #     return _cache[base]

# # MUTED_COLOR = "#bbbbbb"   # for datasets with no MIA signal (near-chance AUC)

# # def _datasets_in(df):
# #     """Datasets actually present, in a stable order."""
# #     return sorted(df["dataset"].unique())

# # PARTITIONER_COLORS = {
# #     "louvain":    "#9467bd",
# #     "metis":      "#7f7f7f",
# #     "metis_plus": "#8c564b",
# #     "random":     "#e377c2",
# # }
# # SCENARIO_STYLE = {"client": "-", "node": "--"}
# # METRIC_LABELS = {
# #     "shokri_mia_auc":              "MIA AUC",
# #     "shokri_mia_balanced_acc":     "Balanced Accuracy",
# #     "shokri_mia_tpr_at_fpr_0.01":  "TPR @ FPR ≤ 0.01",
# #     "shokri_mia_tpr_at_fpr_0.001": "TPR @ FPR ≤ 0.001",
# # }
# # FOUR_METRICS = list(METRIC_LABELS.keys())


# # def setup_style():
# #     plt.rcParams.update({
# #         "font.family":       "sans-serif",
# #         "font.size":         10,
# #         "axes.titlesize":    11,
# #         "axes.labelsize":    10,
# #         "axes.spines.top":   False,
# #         "axes.spines.right": False,
# #         "axes.grid":         True,
# #         "grid.alpha":        0.3,
# #         "grid.linestyle":    "-",
# #         "legend.frameon":    False,
# #         "figure.dpi":        120,
# #         "savefig.dpi":       200,
# #         "savefig.bbox":      "tight",
# #     })


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  Data loading
# # # ═══════════════════════════════════════════════════════════════════════════

# # PRIMARY_KEY = ["dataset", "scenario", "seed",
# #                "partitioning", "num_clients", "forget_ratio"]


# # def _read_csv_aligned(path) -> pd.DataFrame:
# #     """Read a CSV whose data rows may carry extra *trailing* fields beyond the
# #     header width (e.g. stray trailing commas from a later writer version).
# #     Keeps only the header's columns; extra trailing empties are dropped and
# #     blank cells become NaN. Verified lossless for header-aligned rows."""
# #     cols = pd.read_csv(path, nrows=0).columns.tolist()
# #     return pd.read_csv(path, header=0, names=cols, usecols=range(len(cols)))


# # def load_all(csv_dir: Path) -> pd.DataFrame:
# #     files = sorted(glob.glob(str(csv_dir / "all_variations_*.csv")))
# #     if not files:
# #         raise FileNotFoundError(f"No CSVs found under {csv_dir}")
# #     dfs = [_read_csv_aligned(f) for f in files]
# #     print(f"Loaded {len(files)} CSV(s):")
# #     for f, df in zip(files, dfs):
# #         print(f"  {Path(f).name}: {len(df)} rows")
# #     df = pd.concat(dfs, ignore_index=True)
# #     df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
# #     print(f"After dedup: {len(df)} rows")
# #     return df


# # def load_sweep(csv_dir: Path, sweep_name: str) -> pd.DataFrame:
# #     path = csv_dir / f"all_variations_{sweep_name}.csv"
# #     if not path.exists():
# #         print(f"⚠ Missing sweep CSV: {path.name}")
# #         return pd.DataFrame()
# #     return _read_csv_aligned(path)


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  1-3. Sensitivity plots — one axis per figure, 2×2 grid of metrics
# # # ═══════════════════════════════════════════════════════════════════════════

# # def plot_sensitivity(df: pd.DataFrame, axis: str, axis_label: str,
# #                       out_path: Path, scenario: str,
# #                       muted: set = frozenset(), is_categorical: bool = False):
# #     """2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001 vs. `axis`, for a single
# #     scenario (one figure per scenario). Datasets in `muted` (near-chance MIA,
# #     no signal) are drawn as thin grey context lines with no error bars and no
# #     legend entry; all others get a distinct colour and error bars."""
# #     if df.empty:
# #         print(f"⚠ empty df for {axis} sensitivity, skipping")
# #         return

# #     df = df[df.scenario == scenario]
# #     if df.empty:
# #         print(f"⚠ no '{scenario}' rows for {axis} sensitivity, skipping")
# #         return

# #     metrics = [m for m in FOUR_METRICS if m in df.columns]
# #     if not metrics:
# #         print(f"⚠ no metric columns present for {axis}")
# #         return

# #     fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
# #     axes = axes.flatten()

# #     axis_vals = sorted(df[axis].unique())
# #     # Draw muted datasets first (underneath), highlighted ones on top.
# #     datasets = sorted(_datasets_in(df), key=lambda d: (d not in muted, d))

# #     for i, metric in enumerate(metrics):
# #         ax = axes[i]
# #         for ds in datasets:
# #             sub = df[df.dataset == ds]
# #             if sub.empty: continue
# #             agg = sub.groupby(axis)[metric].agg(["mean", "std"]).reindex(axis_vals)
# #             x = np.arange(len(axis_vals)) if is_categorical else np.asarray(axis_vals, dtype=float)
# #             if ds in muted:
# #                 ax.plot(x, agg["mean"], color=MUTED_COLOR, linewidth=1.0,
# #                         alpha=0.6, zorder=1, label="_nolegend_")
# #             else:
# #                 ax.errorbar(
# #                     x, agg["mean"], yerr=agg["std"],
# #                     color=_dataset_color(ds), marker="o", markersize=4,
# #                     capsize=3, linewidth=1.6, zorder=3, label=ds,
# #                 )
# #         ax.set_title(METRIC_LABELS[metric])
# #         ax.set_xlabel(axis_label)
# #         if is_categorical:
# #             ax.set_xticks(np.arange(len(axis_vals)))
# #             ax.set_xticklabels(axis_vals, rotation=15, ha="right")
# #         # Reference lines
# #         if metric == "shokri_mia_auc" or metric == "shokri_mia_balanced_acc":
# #             ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
# #         elif metric.startswith("shokri_mia_tpr_at_fpr"):
# #             # A random attacker's TPR at FPR α is α itself
# #             α = float(metric.rsplit("_", 1)[-1])
# #             ax.axhline(α, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
# #                         label="_random baseline")

# #     # Legend outside on the right (highlighted datasets only)
# #     handles, labels = axes[0].get_legend_handles_labels()
# #     if muted:
# #         handles.append(plt.Line2D([0], [0], color=MUTED_COLOR, linewidth=1.0))
# #         labels.append("others (≈chance)")
# #     fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.13, 0.5),
# #                fontsize=9, title="Dataset")
# #     fig.suptitle(f"Sensitivity to {axis_label}  —  {scenario} scenario",
# #                  fontsize=13, fontweight="bold")
# #     fig.tight_layout(rect=[0, 0, 0.98, 0.97])
# #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# #     fig.savefig(str(out_path) + ".png")
# #     plt.close(fig)
# #     print(f"→ {out_path}.pdf / .png")


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  4. Correlation scatter grid — MIA AUC vs each candidate predictor
# # # ═══════════════════════════════════════════════════════════════════════════

# # CANDIDATE_PREDICTORS = [
# #     ("train_test_gap", "Train/Test Accuracy Gap"),
# #     ("modularity",     "Modularity Q"),
# #     ("homophily",      "Homophily"),
# #     ("n_nodes",        "Total Nodes"),
# #     ("num_clients",    "Number of Clients K"),
# #     ("forget_ratio",   "Forget Ratio"),
# # ]


# # def plot_correlation_grid(df: pd.DataFrame, out_path: Path):
# #     """3×2 grid: MIA AUC vs each predictor, one point per (dataset, config)."""
# #     preds = [(c, l) for c, l in CANDIDATE_PREDICTORS if c in df.columns]
# #     if not preds:
# #         print("⚠ no predictor columns present"); return

# #     rows, cols = 3, 2
# #     fig, axes = plt.subplots(rows, cols, figsize=(11, 12))
# #     axes = axes.flatten()

# #     y = df["shokri_mia_auc"].values

# #     for i, (col, label) in enumerate(preds):
# #         ax = axes[i]
# #         x = df[col].values
# #         keep = ~np.isnan(x) & ~np.isnan(y)

# #         # Colour by partitioning if available
# #         if "partitioning" in df.columns:
# #             for p, colour in PARTITIONER_COLORS.items():
# #                 m = keep & (df["partitioning"].values == p)
# #                 if m.any():
# #                     ax.scatter(x[m], y[m], color=colour, alpha=0.55, s=22,
# #                                 edgecolors="none", label=p)
# #         else:
# #             ax.scatter(x[keep], y[keep], alpha=0.55, s=22, edgecolors="none")

# #         # Linear fit
# #         if keep.sum() >= 5 and np.std(x[keep]) > 1e-8:
# #             slope, intercept = np.polyfit(x[keep], y[keep], 1)
# #             xs = np.linspace(x[keep].min(), x[keep].max(), 50)
# #             ax.plot(xs, slope * xs + intercept, color="black",
# #                      linewidth=1.5, alpha=0.8, linestyle="--")
# #             rho, p = spearmanr(x[keep], y[keep])
# #             ax.text(0.02, 0.95,
# #                      f"Spearman ρ = {rho:+.3f}\np = {p:.4f}\nn = {keep.sum()}",
# #                      transform=ax.transAxes, va="top", ha="left", fontsize=9,
# #                      bbox=dict(boxstyle="round,pad=0.35",
# #                                facecolor="white", edgecolor="lightgray", alpha=0.9))

# #         ax.set_xlabel(label)
# #         ax.set_ylabel("MIA AUC")
# #         ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)

# #     # Hide unused
# #     for j in range(len(preds), rows * cols):
# #         axes[j].set_visible(False)

# #     # Common partitioner legend
# #     if "partitioning" in df.columns:
# #         handles = [mpl.patches.Patch(color=c, label=p)
# #                     for p, c in PARTITIONER_COLORS.items()
# #                     if p in df["partitioning"].unique()]
# #         fig.legend(handles=handles, loc="lower right",
# #                     bbox_to_anchor=(0.98, 0.02), title="Partitioner",
# #                     fontsize=9, ncol=len(handles))

# #     fig.suptitle("MIA AUC vs. Candidate Predictors (n = {} across sweep configs)".format(len(df)),
# #                   fontsize=13, fontweight="bold")
# #     fig.tight_layout(rect=[0, 0.04, 1, 0.97])
# #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# #     fig.savefig(str(out_path) + ".png")
# #     plt.close(fig)
# #     print(f"→ {out_path}.pdf / .png")


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  5. Headline scatter — the paper's main finding in one figure
# # # ═══════════════════════════════════════════════════════════════════════════

# # def plot_headline_gap_vs_auc(df: pd.DataFrame, out_path: Path):
# #     """Train/test gap vs MIA AUC — colored by partitioning, marker per dataset.
# #     The paper's headline figure."""
# #     if "train_test_gap" not in df.columns:
# #         print("⚠ train_test_gap missing"); return

# #     fig, ax = plt.subplots(figsize=(9, 6.5))

# #     _marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*"]
# #     markers = {ds: _marker_pool[i % len(_marker_pool)]
# #                for i, ds in enumerate(_datasets_in(df))}

# #     for ds in _datasets_in(df):
# #         for p in ["louvain", "metis", "metis_plus", "random"]:
# #             sub = df[(df.dataset == ds) & (df.partitioning == p)]
# #             if sub.empty: continue
# #             ax.scatter(
# #                 sub["train_test_gap"], sub["shokri_mia_auc"],
# #                 marker=markers[ds], color=PARTITIONER_COLORS.get(p, "gray"),
# #                 alpha=0.65, s=45, edgecolors="none",
# #             )

# #     # Global regression line
# #     x = df["train_test_gap"].values; y = df["shokri_mia_auc"].values
# #     keep = ~np.isnan(x) & ~np.isnan(y)
# #     slope, intercept = np.polyfit(x[keep], y[keep], 1)
# #     xs = np.linspace(x[keep].min(), x[keep].max(), 50)
# #     ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9,
# #              label=f"OLS fit (slope = {slope:+.2f})")

# #     rho, p = spearmanr(x[keep], y[keep])
# #     from sklearn.linear_model import LinearRegression
# #     R2 = LinearRegression().fit(x[keep].reshape(-1, 1), y[keep]).score(
# #         x[keep].reshape(-1, 1), y[keep])
# #     ax.text(0.02, 0.98,
# #              f"Spearman ρ = {rho:+.3f}\np < {max(p, 1e-16):.1e}\nR² = {R2:.3f}\nn = {keep.sum()}",
# #              transform=ax.transAxes, va="top", ha="left", fontsize=10,
# #              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
# #                        edgecolor="lightgray", alpha=0.92))

# #     ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
# #     ax.set_xlabel("Train / Test Accuracy Gap  (retrain baseline)")
# #     ax.set_ylabel("MIA AUC")
# #     ax.set_title("MIA AUC tracks overfitting, not unlearning quality",
# #                   fontsize=12, fontweight="bold")

# #     # Custom legends: partitioner (colour) + dataset (marker)
# #     p_handles = [plt.Line2D([0], [0], marker="s", color="w",
# #                               markerfacecolor=c, markersize=9, label=p)
# #                   for p, c in PARTITIONER_COLORS.items() if p in df.partitioning.unique()]
# #     ds_handles = [plt.Line2D([0], [0], marker=m, color="gray",
# #                                markersize=9, linestyle="", label=ds)
# #                    for ds, m in markers.items()]

# #     leg1 = ax.legend(handles=p_handles, title="Partitioner",
# #                        loc="lower right", bbox_to_anchor=(1.0, 0.0),
# #                        fontsize=9)
# #     ax.add_artist(leg1)
# #     ax.legend(handles=ds_handles, title="Dataset",
# #                 loc="lower right", bbox_to_anchor=(0.82, 0.0),
# #                 fontsize=9)

# #     fig.tight_layout()
# #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# #     fig.savefig(str(out_path) + ".png")
# #     plt.close(fig)
# #     print(f"→ {out_path}.pdf / .png")


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  6. Distribution boxplots — how much MIA AUC varies within each dataset
# # # ═══════════════════════════════════════════════════════════════════════════

# # def plot_boxplots(df: pd.DataFrame, out_path: Path):
# #     """Per-dataset MIA AUC boxplot across all sweep configs. Shows the range
# #     of MIA behavior on the retrained baseline — the paper's variability story."""
# #     fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

# #     for i, sc in enumerate(["client", "node"]):
# #         ax = axes[i]
# #         data, colors = [], []
# #         labels = []
# #         for ds in _datasets_in(df):
# #             sub = df[(df.dataset == ds) & (df.scenario == sc)]
# #             if sub.empty: continue
# #             data.append(sub["shokri_mia_auc"].values)
# #             colors.append(_dataset_color(ds))
# #             labels.append(ds)

# #         bp = ax.boxplot(data, patch_artist=True, showmeans=True,
# #                           meanprops={"marker": "o", "markerfacecolor": "white",
# #                                     "markeredgecolor": "black", "markersize": 5},
# #                           medianprops={"color": "black", "linewidth": 1.4},
# #                           widths=0.55)
# #         for patch, colour in zip(bp["boxes"], colors):
# #             patch.set_facecolor(colour); patch.set_alpha(0.65)

# #         # Scatter the individual points on top
# #         for j, arr in enumerate(data, 1):
# #             jitter = np.random.default_rng(0).normal(0, 0.06, size=len(arr))
# #             ax.scatter(j + jitter, arr, color=colors[j - 1],
# #                         alpha=0.35, s=14, edgecolors="none", zorder=3)

# #         ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
# #         ax.set_xticks(range(1, len(labels) + 1))
# #         ax.set_xticklabels(labels)
# #         ax.set_ylabel("MIA AUC" if i == 0 else "")
# #         ax.set_title(f"Scenario: {sc}")

# #     fig.suptitle("MIA AUC distribution across all sweep configurations",
# #                   fontsize=13, fontweight="bold")
# #     fig.tight_layout(rect=[0, 0, 1, 0.96])
# #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# #     fig.savefig(str(out_path) + ".png")
# #     plt.close(fig)
# #     print(f"→ {out_path}.pdf / .png")


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  7. Regression R² bar chart — how much variance each predictor set explains
# # # ═══════════════════════════════════════════════════════════════════════════

# # def plot_regression_r2(df: pd.DataFrame, out_path: Path):
# #     """Bar chart of R² for different predictor combinations."""
# #     try:
# #         from sklearn.linear_model import LinearRegression
# #     except ImportError:
# #         print("⚠ sklearn missing, skipping regression plot"); return

# #     y = df["shokri_mia_auc"].values
# #     keep_y = ~np.isnan(y); y = y[keep_y]

# #     def _r2(cols):
# #         if not all(c in df.columns for c in cols): return None
# #         X = df[cols].values[keep_y]
# #         m = ~np.isnan(X).any(axis=1)
# #         if m.sum() < 5: return None
# #         return LinearRegression().fit(X[m], y[m]).score(X[m], y[m])

# #     models = [
# #         ("Train/test gap only",     ["train_test_gap"]),
# #         ("Homophily only",          ["homophily"]),
# #         ("Dataset size only",       ["n_nodes"]),
# #         ("Modularity only",         ["modularity"]),
# #         ("Gap + homophily",         ["train_test_gap", "homophily"]),
# #         ("Gap + size",              ["train_test_gap", "n_nodes"]),
# #         ("Gap + modularity",        ["train_test_gap", "modularity"]),
# #         ("All predictors",          ["train_test_gap", "homophily", "n_nodes",
# #                                       "modularity", "forget_ratio", "num_clients"]),
# #     ]
# #     labels, r2s = [], []
# #     for name, cols in models:
# #         r2 = _r2(cols)
# #         if r2 is not None:
# #             labels.append(name); r2s.append(r2)

# #     fig, ax = plt.subplots(figsize=(9, 5.5))
# #     y_pos = np.arange(len(labels))
# #     colors = ["#1f77b4" if "gap" in l.lower() else "#c0c0c0" for l in labels]
# #     colors[-1] = "#2ca02c"     # highlight "all predictors"

# #     bars = ax.barh(y_pos, r2s, color=colors, edgecolor="black",
# #                      linewidth=0.6, alpha=0.85)
# #     ax.set_yticks(y_pos); ax.set_yticklabels(labels)
# #     ax.invert_yaxis()
# #     ax.set_xlabel("R² for predicting MIA AUC")
# #     ax.set_xlim(0, max(r2s) * 1.15)
# #     for bar, r2 in zip(bars, r2s):
# #         ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
# #                  f"{r2:.3f}", va="center", fontsize=9)
# #     ax.set_title(f"Variance in MIA AUC explained by predictor sets (n = {int(keep_y.sum())})",
# #                   fontsize=12, fontweight="bold")
# #     fig.tight_layout()
# #     fig.savefig(str(out_path) + ".pdf", format="pdf")
# #     fig.savefig(str(out_path) + ".png")
# #     plt.close(fig)
# #     print(f"→ {out_path}.pdf / .png")


# # # ═══════════════════════════════════════════════════════════════════════════
# # #  Main
# # # ═══════════════════════════════════════════════════════════════════════════

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--dir",     default="variation_all_results",
# #                         help="Directory containing all_variations_*.csv")
# #     parser.add_argument("--out-dir", default="variation_all_results/plots",
# #                         help="Directory to write PDF+PNG plots")
# #     args = parser.parse_args()

# #     csv_dir = Path(args.dir)
# #     out_dir = Path(args.out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     setup_style()

# #     print(f"\n══ Loading data ══")
# #     df_all = load_all(csv_dir)

# #     df_partition = load_sweep(csv_dir, "partitioning")
# #     df_num_clients = load_sweep(csv_dir, "num_clients")
# #     df_forget_ratio = load_sweep(csv_dir, "forget_ratio")

# #     # Datasets with no MIA signal (pooled mean AUC within noise of chance) are
# #     # muted in the sensitivity plots. Threshold chosen from the data: a clear
# #     # gap separates the near-chance datasets (mean ≈ 0.505) from the rest (≥0.54).
# #     MUTE_AUC_BELOW = 0.52
# #     ds_mean_auc = df_all.groupby("dataset")["shokri_mia_auc"].mean()
# #     muted = set(ds_mean_auc[ds_mean_auc < MUTE_AUC_BELOW].index)
# #     print(f"\nMuted (near-chance) datasets: {sorted(muted) or 'none'}")

# #     print(f"\n══ Generating plots to {out_dir} ══")

# #     # 1-3. Sensitivity plots — one figure per (axis, scenario)
# #     sensitivity_specs = [
# #         (df_partition,    "partitioning", "Partitioning method",   "01_sensitivity_partitioning",  True),
# #         (df_num_clients,  "num_clients",  "Number of Clients K",   "02_sensitivity_num_clients",   False),
# #         (df_forget_ratio, "forget_ratio", "Forget Ratio",          "03_sensitivity_forget_ratio",  False),
# #     ]
# #     for sdf, axis, label, stem, is_cat in sensitivity_specs:
# #         for sc in ["client", "node"]:
# #             plot_sensitivity(sdf, axis, label, out_dir / f"{stem}_{sc}",
# #                              scenario=sc, muted=muted, is_categorical=is_cat)

# #     # 4-7. Aggregate analyses across all rows
# #     plot_correlation_grid(df_all, out_dir / "04_correlation_scatters")
# #     plot_headline_gap_vs_auc(df_all, out_dir / "05_headline_gap_vs_auc")
# #     plot_boxplots(df_all, out_dir / "06_distribution_boxplots")
# #     plot_regression_r2(df_all, out_dir / "07_regression_r2")

# #     print(f"\n✅ Done. Plots → {out_dir}")


# # if __name__ == "__main__":
# #     main()

# """
# plot_variations.py
# ==================
# Comprehensive plotting for the Phase 2 variation sweep.

# Reads all all_variations_*.csv files under --dir and produces:

#   01_sensitivity_partitioning_{client,node}.{pdf,png}   2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001
#   02_sensitivity_num_clients_{client,node}.{pdf,png}    Same, sweeping K
#   03_sensitivity_forget_ratio_{client,node}.{pdf,png}   Same, sweeping forget ratio
#   04_correlation_scatters.{pdf,png}                     Grid of scatter plots for candidate predictors
#   05_headline_gap_vs_auc.{pdf,png}                      Single big scatter — the paper's main finding
#   06_distribution_boxplots.{pdf,png}                    Per-dataset AUC distributions across all configs
#   07_regression_r2.{pdf,png}                            Bar chart of R² per predictor set

# The default font sizes are tuned so the figures stay readable when included
# in a two-column paper at roughly half textwidth.  If you need still larger
# text, pass e.g. --font-scale 1.3.

# Usage
# -----
#   python plot_variations.py \\
#       --dir     variation_all_results \\
#       --out-dir variation_all_results/plots
# """
# from __future__ import annotations

# import argparse, glob
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from scipy.stats import spearmanr

# # Force non-interactive backend so this runs cleanly on a server / in tmux
# mpl.use("Agg")


# # ═══════════════════════════════════════════════════════════════════════════
# #  Style
# # ═══════════════════════════════════════════════════════════════════════════

# DATASET_COLORS = {
#     "Cora":      "#1f77b4",
#     "Chameleon": "#ff7f0e",
#     "Squirrel":  "#2ca02c",
#     "Actor":     "#d62728",
# }

# # 12 well-separated colours — enough to give every dataset a distinct hue.
# DISTINCT_PALETTE = [
#     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
#     "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
# ]

# def _dataset_color(ds, _cache={}):
#     """Return a stable, distinct colour per dataset. Colours are assigned in
#     first-seen order; since callers iterate datasets in sorted order the
#     mapping is deterministic within a run."""
#     base = ds.replace("_filtered", "")   # a *_filtered variant shares its base's colour
#     if base not in _cache:
#         _cache[base] = DISTINCT_PALETTE[len(_cache) % len(DISTINCT_PALETTE)]
#     return _cache[base]

# MUTED_COLOR = "#bbbbbb"   # for datasets with no MIA signal (near-chance AUC)

# def _datasets_in(df):
#     """Datasets actually present, in a stable order."""
#     return sorted(df["dataset"].unique())

# PARTITIONER_COLORS = {
#     "louvain":    "#9467bd",
#     "metis":      "#7f7f7f",
#     "metis_plus": "#8c564b",
#     "random":     "#e377c2",
# }
# SCENARIO_STYLE = {"client": "-", "node": "--"}
# METRIC_LABELS = {
#     "shokri_mia_auc":              "MIA AUC",
#     "shokri_mia_balanced_acc":     "Balanced Accuracy",
#     "shokri_mia_tpr_at_fpr_0.01":  "TPR @ FPR ≤ 0.01",
#     "shokri_mia_tpr_at_fpr_0.001": "TPR @ FPR ≤ 0.001",
# }
# FOUR_METRICS = list(METRIC_LABELS.keys())


# # Module-level font scale, overridden by --font-scale in main().
# FONT_SCALE = 1.0


# def setup_style(font_scale: float = 1.0):
#     """Set rcParams.  `font_scale` multiplies all text sizes so that the
#     whole figure can be made more readable without touching geometry."""
#     global FONT_SCALE
#     FONT_SCALE = font_scale

#     def S(base: float) -> float:
#         return base * font_scale

#     plt.rcParams.update({
#         "font.family":       "sans-serif",
#         "font.size":         S(12),
#         "axes.titlesize":    S(14),
#         "axes.labelsize":    S(13),
#         "xtick.labelsize":   S(12),
#         "ytick.labelsize":   S(12),
#         "axes.spines.top":   False,
#         "axes.spines.right": False,
#         "axes.grid":         True,
#         "grid.alpha":        0.3,
#         "grid.linestyle":    "-",
#         "legend.frameon":    False,
#         "legend.fontsize":       S(13),
#         "legend.title_fontsize": S(14),
#         "figure.dpi":        120,
#         "savefig.dpi":       200,
#         "savefig.bbox":      "tight",
#     })


# # ═══════════════════════════════════════════════════════════════════════════
# #  Data loading
# # ═══════════════════════════════════════════════════════════════════════════

# PRIMARY_KEY = ["dataset", "scenario", "seed",
#                "partitioning", "num_clients", "forget_ratio"]


# def _read_csv_aligned(path) -> pd.DataFrame:
#     """Read a CSV whose data rows may carry extra *trailing* fields beyond the
#     header width (e.g. stray trailing commas from a later writer version).
#     Keeps only the header's columns; extra trailing empties are dropped and
#     blank cells become NaN. Verified lossless for header-aligned rows."""
#     cols = pd.read_csv(path, nrows=0).columns.tolist()
#     return pd.read_csv(path, header=0, names=cols, usecols=range(len(cols)))


# def load_all(csv_dir: Path) -> pd.DataFrame:
#     files = sorted(glob.glob(str(csv_dir / "all_variations_*.csv")))
#     if not files:
#         raise FileNotFoundError(f"No CSVs found under {csv_dir}")
#     dfs = [_read_csv_aligned(f) for f in files]
#     print(f"Loaded {len(files)} CSV(s):")
#     for f, df in zip(files, dfs):
#         print(f"  {Path(f).name}: {len(df)} rows")
#     df = pd.concat(dfs, ignore_index=True)
#     df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
#     print(f"After dedup: {len(df)} rows")
#     return df


# def load_sweep(csv_dir: Path, sweep_name: str) -> pd.DataFrame:
#     path = csv_dir / f"all_variations_{sweep_name}.csv"
#     if not path.exists():
#         print(f"⚠ Missing sweep CSV: {path.name}")
#         return pd.DataFrame()
#     return _read_csv_aligned(path)


# # ═══════════════════════════════════════════════════════════════════════════
# #  1-3. Sensitivity plots — one axis per figure, 2×2 grid of metrics
# # ═══════════════════════════════════════════════════════════════════════════

# def plot_sensitivity(df: pd.DataFrame, axis: str, axis_label: str,
#                       out_path: Path, scenario: str,
#                       muted: set = frozenset(), is_categorical: bool = False):
#     """2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001 vs. `axis`, for a single
#     scenario (one figure per scenario). Datasets in `muted` (near-chance MIA,
#     no signal) are drawn as thin grey context lines with no error bars and no
#     legend entry; all others get a distinct colour and error bars."""
#     if df.empty:
#         print(f"⚠ empty df for {axis} sensitivity, skipping")
#         return

#     df = df[df.scenario == scenario]
#     if df.empty:
#         print(f"⚠ no '{scenario}' rows for {axis} sensitivity, skipping")
#         return

#     metrics = [m for m in FOUR_METRICS if m in df.columns]
#     if not metrics:
#         print(f"⚠ no metric columns present for {axis}")
#         return

#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     # Figure sized so it survives being shrunk to ~0.48\linewidth.
#     fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True)
#     axes = axes.flatten()

#     axis_vals = sorted(df[axis].unique())
#     # Draw muted datasets first (underneath), highlighted ones on top.
#     datasets = sorted(_datasets_in(df), key=lambda d: (d not in muted, d))

#     for i, metric in enumerate(metrics):
#         ax = axes[i]
#         for ds in datasets:
#             sub = df[df.dataset == ds]
#             if sub.empty: continue
#             agg = sub.groupby(axis)[metric].agg(["mean", "std"]).reindex(axis_vals)
#             x = np.arange(len(axis_vals)) if is_categorical else np.asarray(axis_vals, dtype=float)
#             if ds in muted:
#                 ax.plot(x, agg["mean"], color=MUTED_COLOR, linewidth=1.0,
#                         alpha=0.6, zorder=1, label="_nolegend_")
#             else:
#                 ax.errorbar(
#                     x, agg["mean"], yerr=agg["std"],
#                     color=_dataset_color(ds), marker="o", markersize=5,
#                     capsize=3, linewidth=1.8, zorder=3, label=ds,
#                 )
#         ax.set_title(METRIC_LABELS[metric], fontsize=S(15))
#         ax.set_xlabel(axis_label, fontsize=S(13))
#         ax.tick_params(axis="both", labelsize=S(12))
#         if is_categorical:
#             ax.set_xticks(np.arange(len(axis_vals)))
#             ax.set_xticklabels(axis_vals, rotation=20, ha="right",
#                                fontsize=S(12))
#         # Reference lines
#         if metric == "shokri_mia_auc" or metric == "shokri_mia_balanced_acc":
#             ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
#         elif metric.startswith("shokri_mia_tpr_at_fpr"):
#             # A random attacker's TPR at FPR α is α itself
#             α = float(metric.rsplit("_", 1)[-1])
#             ax.axhline(α, color="black", linestyle=":", linewidth=0.9, alpha=0.5,
#                         label="_random baseline")

#     # Legend outside on the right (highlighted datasets only)
#     handles, labels = axes[0].get_legend_handles_labels()
#     if muted:
#         handles.append(plt.Line2D([0], [0], color=MUTED_COLOR, linewidth=1.5))
#         labels.append("others (≈chance)")

#     fig.legend(handles, labels,
#                loc="center right", bbox_to_anchor=(1.18, 0.5),
#                fontsize=S(14), title="Dataset", title_fontsize=S(15),
#                labelspacing=0.6, handlelength=1.6, borderaxespad=0.4)

#     fig.suptitle(f"Sensitivity to {axis_label}  —  {scenario} scenario",
#                  fontsize=S(16), fontweight="bold")
#     fig.tight_layout(rect=[0, 0, 0.97, 0.96])
#     fig.savefig(str(out_path) + ".pdf", format="pdf")
#     fig.savefig(str(out_path) + ".png")
#     plt.close(fig)
#     print(f"→ {out_path}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  4. Correlation scatter grid — MIA AUC vs each candidate predictor
# # ═══════════════════════════════════════════════════════════════════════════

# CANDIDATE_PREDICTORS = [
#     ("train_test_gap", "Train/Test Accuracy Gap"),
#     ("modularity",     "Modularity Q"),
#     ("homophily",      "Homophily"),
#     ("n_nodes",        "Total Nodes"),
#     ("num_clients",    "Number of Clients K"),
#     ("forget_ratio",   "Forget Ratio"),
# ]


# def plot_correlation_grid(df: pd.DataFrame, out_path: Path):
#     """3×2 grid: MIA AUC vs each predictor, one point per (dataset, config)."""
#     preds = [(c, l) for c, l in CANDIDATE_PREDICTORS if c in df.columns]
#     if not preds:
#         print("⚠ no predictor columns present"); return

#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     rows, cols = 3, 2
#     fig, axes = plt.subplots(rows, cols, figsize=(11, 12))
#     axes = axes.flatten()

#     y = df["shokri_mia_auc"].values

#     for i, (col, label) in enumerate(preds):
#         ax = axes[i]
#         x = df[col].values
#         keep = ~np.isnan(x) & ~np.isnan(y)

#         # Colour by partitioning if available
#         if "partitioning" in df.columns:
#             for p, colour in PARTITIONER_COLORS.items():
#                 m = keep & (df["partitioning"].values == p)
#                 if m.any():
#                     ax.scatter(x[m], y[m], color=colour, alpha=0.55, s=28,
#                                 edgecolors="none", label=p)
#         else:
#             ax.scatter(x[keep], y[keep], alpha=0.55, s=28, edgecolors="none")

#         # Linear fit
#         if keep.sum() >= 5 and np.std(x[keep]) > 1e-8:
#             slope, intercept = np.polyfit(x[keep], y[keep], 1)
#             xs = np.linspace(x[keep].min(), x[keep].max(), 50)
#             ax.plot(xs, slope * xs + intercept, color="black",
#                      linewidth=1.6, alpha=0.8, linestyle="--")
#             rho, p = spearmanr(x[keep], y[keep])
#             ax.text(0.02, 0.95,
#                      f"Spearman ρ = {rho:+.3f}\np = {p:.4f}\nn = {keep.sum()}",
#                      transform=ax.transAxes, va="top", ha="left", fontsize=S(11),
#                      bbox=dict(boxstyle="round,pad=0.35",
#                                facecolor="white", edgecolor="lightgray", alpha=0.9))

#         ax.set_xlabel(label, fontsize=S(13))
#         ax.set_ylabel("MIA AUC", fontsize=S(13))
#         ax.tick_params(axis="both", labelsize=S(12))
#         ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

#     # Hide unused
#     for j in range(len(preds), rows * cols):
#         axes[j].set_visible(False)

#     # Common partitioner legend
#     if "partitioning" in df.columns:
#         handles = [mpl.patches.Patch(color=c, label=p)
#                     for p, c in PARTITIONER_COLORS.items()
#                     if p in df["partitioning"].unique()]
#         fig.legend(handles=handles, loc="lower right",
#                     bbox_to_anchor=(0.98, 0.02), title="Partitioner",
#                     fontsize=S(13), title_fontsize=S(14),
#                     ncol=len(handles))

#     fig.suptitle("MIA AUC vs. Candidate Predictors (n = {} across sweep configs)".format(len(df)),
#                   fontsize=S(16), fontweight="bold")
#     fig.tight_layout(rect=[0, 0.04, 1, 0.97])
#     fig.savefig(str(out_path) + ".pdf", format="pdf")
#     fig.savefig(str(out_path) + ".png")
#     plt.close(fig)
#     print(f"→ {out_path}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  5. Headline scatter — the paper's main finding in one figure
# # ═══════════════════════════════════════════════════════════════════════════

# def plot_headline_gap_vs_auc(df: pd.DataFrame, out_path: Path):
#     """Train/test gap vs MIA AUC — colored by partitioning, marker per dataset.
#     The paper's headline figure."""
#     if "train_test_gap" not in df.columns:
#         print("⚠ train_test_gap missing"); return

#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     fig, ax = plt.subplots(figsize=(9, 6.5))

#     _marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*"]
#     markers = {ds: _marker_pool[i % len(_marker_pool)]
#                for i, ds in enumerate(_datasets_in(df))}

#     for ds in _datasets_in(df):
#         for p in ["louvain", "metis", "metis_plus", "random"]:
#             sub = df[(df.dataset == ds) & (df.partitioning == p)]
#             if sub.empty: continue
#             ax.scatter(
#                 sub["train_test_gap"], sub["shokri_mia_auc"],
#                 marker=markers[ds], color=PARTITIONER_COLORS.get(p, "gray"),
#                 alpha=0.65, s=55, edgecolors="none",
#             )

#     # Global regression line
#     x = df["train_test_gap"].values; y = df["shokri_mia_auc"].values
#     keep = ~np.isnan(x) & ~np.isnan(y)
#     slope, intercept = np.polyfit(x[keep], y[keep], 1)
#     xs = np.linspace(x[keep].min(), x[keep].max(), 50)
#     ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9,
#              label=f"OLS fit (slope = {slope:+.2f})")

#     rho, p = spearmanr(x[keep], y[keep])
#     from sklearn.linear_model import LinearRegression
#     R2 = LinearRegression().fit(x[keep].reshape(-1, 1), y[keep]).score(
#         x[keep].reshape(-1, 1), y[keep])
#     ax.text(0.02, 0.98,
#              f"Spearman ρ = {rho:+.3f}\np < {max(p, 1e-16):.1e}\nR² = {R2:.3f}\nn = {keep.sum()}",
#              transform=ax.transAxes, va="top", ha="left", fontsize=S(13),
#              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
#                        edgecolor="lightgray", alpha=0.92))

#     ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
#     ax.set_xlabel("Train / Test Accuracy Gap  (retrain baseline)", fontsize=S(14))
#     ax.set_ylabel("MIA AUC", fontsize=S(14))
#     ax.tick_params(axis="both", labelsize=S(13))
#     ax.set_title("MIA AUC tracks overfitting, not unlearning quality",
#                   fontsize=S(15), fontweight="bold")

#     # Custom legends: partitioner (colour) + dataset (marker)
#     p_handles = [plt.Line2D([0], [0], marker="s", color="w",
#                               markerfacecolor=c, markersize=11, label=p)
#                   for p, c in PARTITIONER_COLORS.items() if p in df.partitioning.unique()]
#     ds_handles = [plt.Line2D([0], [0], marker=m, color="gray",
#                                markersize=11, linestyle="", label=ds)
#                    for ds, m in markers.items()]

#     leg1 = ax.legend(handles=p_handles, title="Partitioner",
#                        loc="lower right", bbox_to_anchor=(1.0, 0.0),
#                        fontsize=S(13), title_fontsize=S(14))
#     ax.add_artist(leg1)
#     ax.legend(handles=ds_handles, title="Dataset",
#                 loc="lower right", bbox_to_anchor=(0.78, 0.0),
#                 fontsize=S(13), title_fontsize=S(14))

#     fig.tight_layout()
#     fig.savefig(str(out_path) + ".pdf", format="pdf")
#     fig.savefig(str(out_path) + ".png")
#     plt.close(fig)
#     print(f"→ {out_path}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  6. Distribution boxplots — how much MIA AUC varies within each dataset
# # ═══════════════════════════════════════════════════════════════════════════

# def plot_boxplots(df: pd.DataFrame, out_path: Path):
#     """Per-dataset MIA AUC boxplot across all sweep configs. Shows the range
#     of MIA behavior on the retrained baseline — the paper's variability story."""
#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

#     for i, sc in enumerate(["client", "node"]):
#         ax = axes[i]
#         data, colors = [], []
#         labels = []
#         for ds in _datasets_in(df):
#             sub = df[(df.dataset == ds) & (df.scenario == sc)]
#             if sub.empty: continue
#             data.append(sub["shokri_mia_auc"].values)
#             colors.append(_dataset_color(ds))
#             labels.append(ds)

#         bp = ax.boxplot(data, patch_artist=True, showmeans=True,
#                           meanprops={"marker": "o", "markerfacecolor": "white",
#                                     "markeredgecolor": "black", "markersize": 6},
#                           medianprops={"color": "black", "linewidth": 1.6},
#                           widths=0.55)
#         for patch, colour in zip(bp["boxes"], colors):
#             patch.set_facecolor(colour); patch.set_alpha(0.65)

#         # Scatter the individual points on top
#         for j, arr in enumerate(data, 1):
#             jitter = np.random.default_rng(0).normal(0, 0.06, size=len(arr))
#             ax.scatter(j + jitter, arr, color=colors[j - 1],
#                         alpha=0.35, s=18, edgecolors="none", zorder=3)

#         ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
#         ax.set_xticks(range(1, len(labels) + 1))
#         ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=S(12))
#         ax.set_ylabel("MIA AUC" if i == 0 else "", fontsize=S(14))
#         ax.tick_params(axis="y", labelsize=S(12))
#         ax.set_title(f"Scenario: {sc}", fontsize=S(15))

#     fig.suptitle("MIA AUC distribution across all sweep configurations",
#                   fontsize=S(16), fontweight="bold")
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     fig.savefig(str(out_path) + ".pdf", format="pdf")
#     fig.savefig(str(out_path) + ".png")
#     plt.close(fig)
#     print(f"→ {out_path}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  7. Regression R² bar chart — how much variance each predictor set explains
# # ═══════════════════════════════════════════════════════════════════════════

# def plot_regression_r2(df: pd.DataFrame, out_path: Path):
#     """Bar chart of R² for different predictor combinations."""
#     try:
#         from sklearn.linear_model import LinearRegression
#     except ImportError:
#         print("⚠ sklearn missing, skipping regression plot"); return

#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     y = df["shokri_mia_auc"].values
#     keep_y = ~np.isnan(y); y = y[keep_y]

#     def _r2(cols):
#         if not all(c in df.columns for c in cols): return None
#         X = df[cols].values[keep_y]
#         m = ~np.isnan(X).any(axis=1)
#         if m.sum() < 5: return None
#         return LinearRegression().fit(X[m], y[m]).score(X[m], y[m])

#     models = [
#         ("Train/test gap only",     ["train_test_gap"]),
#         ("Homophily only",          ["homophily"]),
#         ("Dataset size only",       ["n_nodes"]),
#         ("Modularity only",         ["modularity"]),
#         ("Gap + homophily",         ["train_test_gap", "homophily"]),
#         ("Gap + size",              ["train_test_gap", "n_nodes"]),
#         ("Gap + modularity",        ["train_test_gap", "modularity"]),
#         ("All predictors",          ["train_test_gap", "homophily", "n_nodes",
#                                       "modularity", "forget_ratio", "num_clients"]),
#     ]
#     labels, r2s = [], []
#     for name, cols in models:
#         r2 = _r2(cols)
#         if r2 is not None:
#             labels.append(name); r2s.append(r2)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     y_pos = np.arange(len(labels))
#     colors = ["#1f77b4" if "gap" in l.lower() else "#c0c0c0" for l in labels]
#     colors[-1] = "#2ca02c"     # highlight "all predictors"

#     bars = ax.barh(y_pos, r2s, color=colors, edgecolor="black",
#                      linewidth=0.6, alpha=0.85)
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(labels, fontsize=S(13))
#     ax.invert_yaxis()
#     ax.set_xlabel("R² for predicting MIA AUC", fontsize=S(14))
#     ax.set_xlim(0, max(r2s) * 1.15)
#     ax.tick_params(axis="x", labelsize=S(12))
#     for bar, r2 in zip(bars, r2s):
#         ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
#                  f"{r2:.3f}", va="center", fontsize=S(12))
#     ax.set_title(f"Variance in MIA AUC explained by predictor sets (n = {int(keep_y.sum())})",
#                   fontsize=S(15), fontweight="bold")
#     fig.tight_layout()
#     fig.savefig(str(out_path) + ".pdf", format="pdf")
#     fig.savefig(str(out_path) + ".png")
#     plt.close(fig)
#     print(f"→ {out_path}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  Main
# # ═══════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dir",     default="variation_all_results",
#                         help="Directory containing all_variations_*.csv")
#     parser.add_argument("--out-dir", default="variation_all_results/plots",
#                         help="Directory to write PDF+PNG plots")
#     parser.add_argument("--font-scale", type=float, default=1.0,
#                         help="Multiplier applied to all text sizes. "
#                              "Use ~1.3 for the 0.48\\linewidth paper inset.")
#     args = parser.parse_args()

#     csv_dir = Path(args.dir)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     setup_style(font_scale=args.font_scale)
#     print(f"Font scale: {args.font_scale}×")

#     print(f"\n══ Loading data ══")
#     df_all = load_all(csv_dir)

#     df_partition = load_sweep(csv_dir, "partitioning")
#     df_num_clients = load_sweep(csv_dir, "num_clients")
#     df_forget_ratio = load_sweep(csv_dir, "forget_ratio")

#     # Datasets with no MIA signal (pooled mean AUC within noise of chance) are
#     # muted in the sensitivity plots. Threshold chosen from the data: a clear
#     # gap separates the near-chance datasets (mean ≈ 0.505) from the rest (≥0.54).
#     MUTE_AUC_BELOW = 0.52
#     ds_mean_auc = df_all.groupby("dataset")["shokri_mia_auc"].mean()
#     muted = set(ds_mean_auc[ds_mean_auc < MUTE_AUC_BELOW].index)
#     print(f"\nMuted (near-chance) datasets: {sorted(muted) or 'none'}")

#     print(f"\n══ Generating plots to {out_dir} ══")

#     # 1-3. Sensitivity plots — one figure per (axis, scenario)
#     sensitivity_specs = [
#         (df_partition,    "partitioning", "Partitioning method",   "01_sensitivity_partitioning",  True),
#         (df_num_clients,  "num_clients",  "Number of Clients K",   "02_sensitivity_num_clients",   False),
#         (df_forget_ratio, "forget_ratio", "Forget Ratio",          "03_sensitivity_forget_ratio",  False),
#     ]
#     for sdf, axis, label, stem, is_cat in sensitivity_specs:
#         for sc in ["client", "node"]:
#             plot_sensitivity(sdf, axis, label, out_dir / f"{stem}_{sc}",
#                              scenario=sc, muted=muted, is_categorical=is_cat)

#     # 4-7. Aggregate analyses across all rows
#     plot_correlation_grid(df_all, out_dir / "04_correlation_scatters")
#     plot_headline_gap_vs_auc(df_all, out_dir / "05_headline_gap_vs_auc")
#     plot_boxplots(df_all, out_dir / "06_distribution_boxplots")
#     plot_regression_r2(df_all, out_dir / "07_regression_r2")

#     print(f"\n✅ Done. Plots → {out_dir}")


# if __name__ == "__main__":
#     main()

"""
plot_variations.py
==================
Comprehensive plotting for the Phase 2 variation sweep.

Reads all all_variations_*.csv files under --dir and produces:

  01_sensitivity_partitioning_{client,node}.{pdf,png}   2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001
  02_sensitivity_num_clients_{client,node}.{pdf,png}    Same, sweeping K
  03_sensitivity_forget_ratio_{client,node}.{pdf,png}   Same, sweeping forget ratio
  04_correlation_scatters.{pdf,png}                     Grid of scatter plots for candidate predictors
  05_headline_gap_vs_auc.{pdf,png}                      Single big scatter — the paper's main finding
  06_distribution_boxplots.{pdf,png}                    Per-dataset AUC distributions across all configs
  07_regression_r2.{pdf,png}                            Bar chart of R² per predictor set

Font sizes are tuned so the figures stay readable when included in a
two-column paper at roughly half textwidth.  Increase further via
--font-scale (e.g. --font-scale 1.8).

Legends are placed outside the axes (bottom strip or right column) so they
never overlap data, with modest padding so the layout stays compact.

Usage
-----
  python plot_variations.py \\
      --dir     variation_all_results \\
      --out-dir variation_all_results/plots \\
      --font-scale 1.6
"""
from __future__ import annotations

import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

# Force non-interactive backend so this runs cleanly on a server / in tmux
mpl.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════════

DATASET_COLORS = {
    "Cora":      "#1f77b4",
    "Chameleon": "#ff7f0e",
    "Squirrel":  "#2ca02c",
    "Actor":     "#d62728",
}

# 12 well-separated colours — enough to give every dataset a distinct hue.
DISTINCT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
]

def _dataset_color(ds, _cache={}):
    """Return a stable, distinct colour per dataset. Colours are assigned in
    first-seen order; since callers iterate datasets in sorted order the
    mapping is deterministic within a run."""
    base = ds.replace("_filtered", "")   # a *_filtered variant shares its base's colour
    if base not in _cache:
        _cache[base] = DISTINCT_PALETTE[len(_cache) % len(DISTINCT_PALETTE)]
    return _cache[base]

MUTED_COLOR = "#bbbbbb"   # for datasets with no MIA signal (near-chance AUC)

def _datasets_in(df):
    """Datasets actually present, in a stable order."""
    return sorted(df["dataset"].unique())

PARTITIONER_COLORS = {
    "louvain":    "#9467bd",
    "metis":      "#7f7f7f",
    "metis_plus": "#8c564b",
    "random":     "#e377c2",
}
SCENARIO_STYLE = {"client": "-", "node": "--"}
METRIC_LABELS = {
    "shokri_mia_auc":              "MIA AUC",
    "shokri_mia_balanced_acc":     "Balanced Accuracy",
    "shokri_mia_tpr_at_fpr_0.01":  "TPR @ FPR ≤ 0.01",
    "shokri_mia_tpr_at_fpr_0.001": "TPR @ FPR ≤ 0.001",
}
FOUR_METRICS = list(METRIC_LABELS.keys())


# Module-level font scale, set by --font-scale in main().
FONT_SCALE = 1.0


def setup_style(font_scale: float = 1.0):
    """Set rcParams.  `font_scale` multiplies every text size so the whole
    figure can be made more readable without touching geometry."""
    global FONT_SCALE
    FONT_SCALE = font_scale

    def S(base: float) -> float:
        return base * font_scale

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         S(12),
        "axes.titlesize":    S(14),
        "axes.labelsize":    S(13),
        "xtick.labelsize":   S(12),
        "ytick.labelsize":   S(12),
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "-",
        "legend.frameon":    False,
        "legend.fontsize":       S(13),
        "legend.title_fontsize": S(14),
        "figure.dpi":        120,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

PRIMARY_KEY = ["dataset", "scenario", "seed",
               "partitioning", "num_clients", "forget_ratio"]


def _read_csv_aligned(path) -> pd.DataFrame:
    """Read a CSV whose data rows may carry extra *trailing* fields beyond the
    header width (e.g. stray trailing commas from a later writer version).
    Keeps only the header's columns; extra trailing empties are dropped and
    blank cells become NaN. Verified lossless for header-aligned rows."""
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return pd.read_csv(path, header=0, names=cols, usecols=range(len(cols)))


def load_all(csv_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(csv_dir / "all_variations_*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {csv_dir}")
    dfs = [_read_csv_aligned(f) for f in files]
    print(f"Loaded {len(files)} CSV(s):")
    for f, df in zip(files, dfs):
        print(f"  {Path(f).name}: {len(df)} rows")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=PRIMARY_KEY, keep="last")
    print(f"After dedup: {len(df)} rows")
    return df


def load_sweep(csv_dir: Path, sweep_name: str) -> pd.DataFrame:
    path = csv_dir / f"all_variations_{sweep_name}.csv"
    if not path.exists():
        print(f"⚠ Missing sweep CSV: {path.name}")
        return pd.DataFrame()
    return _read_csv_aligned(path)


# ═══════════════════════════════════════════════════════════════════════════
#  1-3. Sensitivity plots — one axis per figure, 2×2 grid of metrics
# ═══════════════════════════════════════════════════════════════════════════

def plot_sensitivity(df: pd.DataFrame, axis: str, axis_label: str,
                      out_path: Path, scenario: str,
                      muted: set = frozenset(), is_categorical: bool = False):
    """2×2 grid: AUC / BalAcc / TPR@0.01 / TPR@0.001 vs. `axis`, for a single
    scenario (one figure per scenario).

    Datasets in `muted` (near-chance MIA, no signal) are drawn as thin grey
    context lines with no error bars and no legend entry; all others get a
    distinct colour and error bars.

    Legend is a single horizontal strip *below* the axes, offset well clear of
    the x-axis labels — safe at any font size."""
    if df.empty:
        print(f"⚠ empty df for {axis} sensitivity, skipping")
        return

    df = df[df.scenario == scenario]
    if df.empty:
        print(f"⚠ no '{scenario}' rows for {axis} sensitivity, skipping")
        return

    metrics = [m for m in FOUR_METRICS if m in df.columns]
    if not metrics:
        print(f"⚠ no metric columns present for {axis}")
        return

    def S(base: float) -> float:
        return base * FONT_SCALE

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True)
    axes = axes.flatten()

    axis_vals = sorted(df[axis].unique())
    # Draw muted datasets first (underneath), highlighted ones on top.
    datasets = sorted(_datasets_in(df), key=lambda d: (d not in muted, d))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for ds in datasets:
            sub = df[df.dataset == ds]
            if sub.empty:
                continue
            agg = sub.groupby(axis)[metric].agg(["mean", "std"]).reindex(axis_vals)
            x = (np.arange(len(axis_vals)) if is_categorical
                 else np.asarray(axis_vals, dtype=float))
            if ds in muted:
                ax.plot(x, agg["mean"], color=MUTED_COLOR, linewidth=1.0,
                        alpha=0.6, zorder=1, label="_nolegend_")
            else:
                ax.errorbar(
                    x, agg["mean"], yerr=agg["std"],
                    color=_dataset_color(ds), marker="o", markersize=5,
                    capsize=3, linewidth=1.8, zorder=3, label=ds,
                )
        ax.set_title(METRIC_LABELS[metric], fontsize=S(15))
        ax.set_xlabel(axis_label, fontsize=S(13))
        ax.tick_params(axis="both", labelsize=S(12))
        if is_categorical:
            ax.set_xticks(np.arange(len(axis_vals)))
            ax.set_xticklabels(axis_vals, rotation=20, ha="right",
                               fontsize=S(12))
        # Reference lines
        if metric in ("shokri_mia_auc", "shokri_mia_balanced_acc"):
            ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9,
                       alpha=0.5)
        elif metric.startswith("shokri_mia_tpr_at_fpr"):
            α = float(metric.rsplit("_", 1)[-1])
            ax.axhline(α, color="black", linestyle=":", linewidth=0.9,
                       alpha=0.5)

    # ── Single horizontal legend strip, just below the axes ───────────────
    handles, labels = axes[0].get_legend_handles_labels()
    if muted:
        handles.append(plt.Line2D([0], [0], color=MUTED_COLOR, linewidth=1.5))
        labels.append("others (≈chance)")

    ncol = min(len(labels), 4)
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=ncol, fontsize=S(13), frameon=False,
               columnspacing=1.4, handletextpad=0.6, handlelength=1.6,
               borderaxespad=0.0)

    fig.suptitle(f"Sensitivity to {axis_label}  —  {scenario} scenario",
                 fontsize=S(16), fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(str(out_path) + ".pdf", format="pdf")
    fig.savefig(str(out_path) + ".png")
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Correlation scatter grid — MIA AUC vs each candidate predictor
# ═══════════════════════════════════════════════════════════════════════════

CANDIDATE_PREDICTORS = [
    ("train_test_gap", "Train/Test Accuracy Gap"),
    ("modularity",     "Modularity Q"),
    ("homophily",      "Homophily"),
    ("n_nodes",        "Total Nodes"),
    ("num_clients",    "Number of Clients K"),
    ("forget_ratio",   "Forget Ratio"),
]


def plot_correlation_grid(df: pd.DataFrame, out_path: Path):
    """3×2 grid: MIA AUC vs each predictor, one point per (dataset, config).

    Partitioner legend is placed below the grid, outside the axes."""
    preds = [(c, l) for c, l in CANDIDATE_PREDICTORS if c in df.columns]
    if not preds:
        print("⚠ no predictor columns present")
        return

    def S(base: float) -> float:
        return base * FONT_SCALE

    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 12))
    axes = axes.flatten()

    y = df["shokri_mia_auc"].values

    for i, (col, label) in enumerate(preds):
        ax = axes[i]
        x = df[col].values
        keep = ~np.isnan(x) & ~np.isnan(y)

        if "partitioning" in df.columns:
            for p, colour in PARTITIONER_COLORS.items():
                m = keep & (df["partitioning"].values == p)
                if m.any():
                    ax.scatter(x[m], y[m], color=colour, alpha=0.55, s=28,
                               edgecolors="none", label=p)
        else:
            ax.scatter(x[keep], y[keep], alpha=0.55, s=28, edgecolors="none")

        if keep.sum() >= 5 and np.std(x[keep]) > 1e-8:
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            xs = np.linspace(x[keep].min(), x[keep].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="black",
                    linewidth=1.6, alpha=0.8, linestyle="--")
            rho, p = spearmanr(x[keep], y[keep])
            # ax.text(0.02, 0.95,
            #         f"Spearman ρ = {rho:+.3f}\np = {p:.4f}\nn = {keep.sum()}",
            #         transform=ax.transAxes, va="top", ha="left",
            #         fontsize=S(11),
            #         bbox=dict(boxstyle="round,pad=0.35",
            #                   facecolor="white", edgecolor="lightgray",
            #                   alpha=0.9))

        ax.set_xlabel(label, fontsize=S(13))
        ax.set_ylabel("MIA AUC", fontsize=S(13))
        ax.tick_params(axis="both", labelsize=S(12))
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    for j in range(len(preds), rows * cols):
        axes[j].set_visible(False)

    fig.suptitle("MIA AUC vs. Candidate Predictors "
                 "(n = {} across sweep configs)".format(len(df)),
                 fontsize=S(16), fontweight="bold")

    # ── Partitioner legend, just below the grid ──────────────────────────
    if "partitioning" in df.columns:
        handles = [mpl.patches.Patch(color=c, label=p)
                   for p, c in PARTITIONER_COLORS.items()
                   if p in df["partitioning"].unique()]
        fig.legend(handles=handles,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.015),
                   title="Partitioner", ncol=len(handles),
                   fontsize=S(13), title_fontsize=S(14),
                   frameon=False, columnspacing=1.8, borderaxespad=0.0)
        fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(str(out_path) + ".pdf", format="pdf")
    fig.savefig(str(out_path) + ".png")
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  5. Headline scatter — the paper's main finding in one figure
# ═══════════════════════════════════════════════════════════════════════════

def plot_headline_gap_vs_auc(df: pd.DataFrame, out_path: Path):
    """Train/test gap vs MIA AUC — coloured AND shaped by dataset.

    A single dataset legend is placed *outside* the axes on the right, so it
    never overlaps the scatter regardless of font size."""
    if "train_test_gap" not in df.columns:
        print("⚠ train_test_gap missing")
        return

    def S(base: float) -> float:
        return base * FONT_SCALE

    fig, ax = plt.subplots(figsize=(9, 6))

    _marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*"]
    datasets = _datasets_in(df)
    markers = {ds: _marker_pool[i % len(_marker_pool)]
               for i, ds in enumerate(datasets)}

    for ds in datasets:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        ax.scatter(sub["train_test_gap"], sub["shokri_mia_auc"],
                   marker=markers[ds], color=_dataset_color(ds),
                   alpha=0.7, s=55, edgecolors="none", label=ds)

    # Global regression line (hidden from the legend)
    x = df["train_test_gap"].values
    y = df["shokri_mia_auc"].values
    keep = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    xs = np.linspace(x[keep].min(), x[keep].max(), 50)
    ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9,
            label="_nolegend_")

    rho, p = spearmanr(x[keep], y[keep])
    from sklearn.linear_model import LinearRegression
    R2 = LinearRegression().fit(x[keep].reshape(-1, 1), y[keep]).score(
        x[keep].reshape(-1, 1), y[keep])
    # ax.text(0.02, 0.98,
    #         f"Spearman ρ = {rho:+.3f}\np < {max(p, 1e-16):.1e}\n"
    #         f"R² = {R2:.3f}\nn = {keep.sum()}",
    #         transform=ax.transAxes, va="top", ha="left", fontsize=S(13),
    #         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
    #                   edgecolor="lightgray", alpha=0.92))

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
    ax.set_xlabel("Train / Test Accuracy Gap  (retrain baseline)",
                  fontsize=S(14))
    ax.set_ylabel("MIA AUC", fontsize=S(14))
    ax.tick_params(axis="both", labelsize=S(13))
    # ax.set_title("MIA AUC tracks overfitting, not unlearning quality",
    #              fontsize=S(15), fontweight="bold")

    # ── Dataset legend, outside on the right ─────────────────────────────
    ax.legend(title="Dataset",
              loc="center left",
              bbox_to_anchor=(1.03, 0.5),
              fontsize=S(13), title_fontsize=S(14), frameon=False,
              borderaxespad=0.0, labelspacing=0.4)

    fig.tight_layout()
    fig.savefig(str(out_path) + ".pdf", format="pdf")
    fig.savefig(str(out_path) + ".png")
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  6. Distribution boxplots — how much MIA AUC varies within each dataset
# ═══════════════════════════════════════════════════════════════════════════

def plot_boxplots(df: pd.DataFrame, out_path: Path):
    """Per-dataset MIA AUC boxplot across all sweep configs."""
    def S(base: float) -> float:
        return base * FONT_SCALE

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

    for i, sc in enumerate(["client", "node"]):
        ax = axes[i]
        data, colors, labels = [], [], []
        for ds in _datasets_in(df):
            sub = df[(df.dataset == ds) & (df.scenario == sc)]
            if sub.empty:
                continue
            data.append(sub["shokri_mia_auc"].values)
            colors.append(_dataset_color(ds))
            labels.append(ds)

        bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 6},
                        medianprops={"color": "black", "linewidth": 1.6},
                        widths=0.55)
        for patch, colour in zip(bp["boxes"], colors):
            patch.set_facecolor(colour)
            patch.set_alpha(0.65)

        for j, arr in enumerate(data, 1):
            jitter = np.random.default_rng(0).normal(0, 0.06, size=len(arr))
            ax.scatter(j + jitter, arr, color=colors[j - 1],
                       alpha=0.35, s=18, edgecolors="none", zorder=3)

        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=S(12))
        ax.set_ylabel("MIA AUC" if i == 0 else "", fontsize=S(14))
        ax.tick_params(axis="y", labelsize=S(12))
        ax.set_title(f"Scenario: {sc}", fontsize=S(15))

    fig.suptitle("MIA AUC distribution across all sweep configurations",
                 fontsize=S(16), fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path) + ".pdf", format="pdf")
    fig.savefig(str(out_path) + ".png")
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  7. Regression R² bar chart — how much variance each predictor set explains
# ═══════════════════════════════════════════════════════════════════════════

def plot_regression_r2(df: pd.DataFrame, out_path: Path):
    """Bar chart of R² for different predictor combinations."""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("⚠ sklearn missing, skipping regression plot")
        return

    def S(base: float) -> float:
        return base * FONT_SCALE

    y = df["shokri_mia_auc"].values
    keep_y = ~np.isnan(y)
    y = y[keep_y]

    def _r2(cols):
        if not all(c in df.columns for c in cols):
            return None
        X = df[cols].values[keep_y]
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 5:
            return None
        return LinearRegression().fit(X[m], y[m]).score(X[m], y[m])

    models = [
        ("Train/test gap only", ["train_test_gap"]),
        ("Homophily only",      ["homophily"]),
        ("Dataset size only",   ["n_nodes"]),
        ("Modularity only",     ["modularity"]),
        ("Gap + homophily",     ["train_test_gap", "homophily"]),
        ("Gap + size",          ["train_test_gap", "n_nodes"]),
        ("Gap + modularity",    ["train_test_gap", "modularity"]),
        ("All predictors",      ["train_test_gap", "homophily", "n_nodes",
                                 "modularity", "forget_ratio", "num_clients"]),
    ]
    labels, r2s = [], []
    for name, cols in models:
        r2 = _r2(cols)
        if r2 is not None:
            labels.append(name)
            r2s.append(r2)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    colors = ["#1f77b4" if "gap" in l.lower() else "#c0c0c0" for l in labels]
    colors[-1] = "#2ca02c"

    bars = ax.barh(y_pos, r2s, color=colors, edgecolor="black",
                   linewidth=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=S(13))
    ax.invert_yaxis()
    ax.set_xlabel("R² for predicting MIA AUC", fontsize=S(14))
    ax.set_xlim(0, max(r2s) * 1.15)
    ax.tick_params(axis="x", labelsize=S(12))
    for bar, r2 in zip(bars, r2s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{r2:.3f}", va="center", fontsize=S(12))
    ax.set_title(
        f"Variance in MIA AUC explained by predictor sets "
        f"(n = {int(keep_y.sum())})",
        fontsize=S(15), fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path) + ".pdf", format="pdf")
    fig.savefig(str(out_path) + ".png")
    plt.close(fig)
    print(f"→ {out_path}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="variation_all_results",
                        help="Directory containing all_variations_*.csv")
    parser.add_argument("--out-dir", default="variation_all_results/plots",
                        help="Directory to write PDF+PNG plots")
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="Multiplier applied to all text sizes. "
                             "Try 1.4–2.2 for a 0.48\\linewidth paper inset.")
    args = parser.parse_args()

    csv_dir = Path(args.dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style(font_scale=args.font_scale)
    print(f"Font scale: {args.font_scale}×")

    print("\n══ Loading data ══")
    df_all = load_all(csv_dir)

    df_partition = load_sweep(csv_dir, "partitioning")
    df_num_clients = load_sweep(csv_dir, "num_clients")
    df_forget_ratio = load_sweep(csv_dir, "forget_ratio")

    MUTE_AUC_BELOW = 0.52
    ds_mean_auc = df_all.groupby("dataset")["shokri_mia_auc"].mean()
    muted = set(ds_mean_auc[ds_mean_auc < MUTE_AUC_BELOW].index)
    print(f"\nMuted (near-chance) datasets: {sorted(muted) or 'none'}")

    print(f"\n══ Generating plots to {out_dir} ══")

    sensitivity_specs = [
        (df_partition,    "partitioning", "Partitioning method",
         "01_sensitivity_partitioning",  True),
        (df_num_clients,  "num_clients",  "Number of Clients K",
         "02_sensitivity_num_clients",   False),
        (df_forget_ratio, "forget_ratio", "Forget Ratio",
         "03_sensitivity_forget_ratio",  False),
    ]
    for sdf, axis, label, stem, is_cat in sensitivity_specs:
        for sc in ["client", "node"]:
            plot_sensitivity(sdf, axis, label, out_dir / f"{stem}_{sc}",
                             scenario=sc, muted=muted, is_categorical=is_cat)

    plot_correlation_grid(df_all, out_dir / "04_correlation_scatters")
    plot_headline_gap_vs_auc(df_all, out_dir / "05_headline_gap_vs_auc")
    plot_boxplots(df_all, out_dir / "06_distribution_boxplots")
    plot_regression_r2(df_all, out_dir / "07_regression_r2")

    print(f"\n✅ Done. Plots → {out_dir}")


if __name__ == "__main__":
    main()