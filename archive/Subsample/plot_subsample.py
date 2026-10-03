# # """
# # plot_subsample.py
# # =================
# # Figures for the subsampling study.

# # Produces:
# #   subsample_gap_auc.{pdf,png}   Pooled scatter: train/test gap (x) vs MIA AUC (y),
# #                                 one point per subsampled configuration, coloured
# #                                 by dataset. The controlled analogue of the main
# #                                 gap-vs-AUC figure.
# #   subsample_clean_tier.{pdf,png} Twin-axis line plots for the datasets whose
# #                                 topology was held approximately constant
# #                                 (roman_empire, amazon_ratings), showing gap and
# #                                 AUC rising together as size falls, alongside the
# #                                 dissociation datasets (minesweeper, tolokers)
# #                                 where a flat gap keeps AUC at chance.

# # Usage
# # -----
# #   python plot_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv \\
# #       --out-dir mia_subsample_results/plots
# # """
# # from __future__ import annotations
# # import argparse
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib as mpl
# # from scipy.stats import spearmanr

# # mpl.use("Agg")

# # GAP = "train_test_gap"
# # AUC = "shokri_mia_auc"

# # DATASET_COLORS = {
# #     "roman_empire":   "#bcbd22",
# #     "amazon_ratings": "#17becf",
# #     "minesweeper":    "#7f7f7f",
# #     "tolokers":       "#aec7e8",
# #     "Computers":      "#2ca02c",
# #     "Photo":          "#9467bd",
# #     "WikiCS":         "#e377c2",
# # }


# # def setup_style():
# #     plt.rcParams.update({
# #         "font.family": "sans-serif", "font.size": 11,
# #         "axes.spines.top": False, "axes.spines.right": False,
# #         "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
# #     })


# # def load(csv):
# #     df = pd.read_csv(csv)
# #     if "error" in df.columns:
# #         df = df[df["error"].isna() | (df["error"].astype(str) == "nan")]
# #     return df


# # # ── Figure A: pooled gap-vs-AUC scatter ─────────────────────────────────────
# # def fig_gap_auc(df, out):
# #     fig, ax = plt.subplots(figsize=(7, 5.5))
# #     x = df[GAP].values.astype(float)
# #     y = df[AUC].values.astype(float)

# #     for ds in sorted(df.dataset.unique()):
# #         m = df.dataset.values == ds
# #         ax.scatter(df[GAP].values[m], df[AUC].values[m], s=26, alpha=0.6,
# #                    color=DATASET_COLORS.get(ds, "#3b6fb0"),
# #                    edgecolors="none", label=ds)

# #     # pooled fit + chance line
# #     keep = ~np.isnan(x) & ~np.isnan(y)
# #     slope, intercept = np.polyfit(x[keep], y[keep], 1)
# #     xs = np.linspace(x[keep].min(), x[keep].max(), 100)
# #     ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9)
# #     ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

# #     rho, p = spearmanr(x[keep], y[keep])
# #     ax.text(0.03, 0.97,
# #             f"Spearman $\\rho$ = {rho:.2f}\np < 0.001\nn = {int(keep.sum())}",
# #             transform=ax.transAxes, va="top", ha="left", fontsize=11,
# #             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
# #                       edgecolor="lightgray", alpha=0.9))

# #     ax.set_xlabel("Train / test accuracy gap")
# #     ax.set_ylabel("MIA AUC")
# #     ax.legend(fontsize=8, loc="lower right", frameon=False, title="Dataset")
# #     fig.tight_layout()
# #     for fmt in ("pdf", "png"):
# #         fig.savefig(f"{out}.{fmt}", format=fmt)
# #     plt.close(fig)
# #     print(f"→ {out}.pdf / .png")


# # # ── Figure B: per-dataset gap & AUC vs fraction (twin axis) ─────────────────
# # # def fig_clean_tier(df, out,
# # #                    datasets=("roman_empire", "amazon_ratings",
# # #                              "minesweeper", "tolokers")):
# # #     present = [d for d in datasets if d in df.dataset.unique()]
# # #     n = len(present)

# # def fig_clean_tier(df, out, datasets=None):
# #     if datasets is None:
# #         datasets = sorted(df.dataset.unique())
# #     present = [d for d in datasets if d in df.dataset.unique()]
# #     n = len(present)

# #     fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharex=True)
# #     if n == 1:
# #         axes = [axes]

# #     fracs = sorted(df.fraction.unique(), reverse=True)  # 1.0 → 0.25

# #     for ax, ds in zip(axes, present):
# #         sub = df[df.dataset == ds]
# #         g = (sub.groupby("fraction")
# #                 .agg(gap=(GAP, "mean"), auc=(AUC, "mean"),
# #                      gap_s=(GAP, "std"), auc_s=(AUC, "std"))
# #                 .reindex(fracs))
# #         xpos = np.arange(len(fracs))

# #         # left axis: gap
# #         c_gap = "#d62728"
# #         ax.errorbar(xpos, g["gap"], yerr=g["gap_s"], color=c_gap,
# #                     marker="o", markersize=4, capsize=3, linewidth=1.6,
# #                     label="gap")
# #         ax.set_ylabel("Train / test gap", color=c_gap)
# #         ax.tick_params(axis="y", labelcolor=c_gap)
# #         ax.axhline(0, color=c_gap, linestyle=":", linewidth=0.7, alpha=0.5)

# #         # right axis: AUC
# #         ax2 = ax.twinx()
# #         c_auc = "#1f77b4"
# #         ax2.errorbar(xpos, g["auc"], yerr=g["auc_s"], color=c_auc,
# #                      marker="s", markersize=4, capsize=3, linewidth=1.6,
# #                      label="MIA AUC")
# #         ax2.set_ylabel("MIA AUC", color=c_auc)
# #         ax2.tick_params(axis="y", labelcolor=c_auc)
# #         ax2.axhline(0.5, color=c_auc, linestyle=":", linewidth=0.7, alpha=0.5)
# #         ax2.spines["top"].set_visible(False)

# #         ax.set_xticks(xpos)
# #         ax.set_xticklabels([f"{f:.2f}" for f in fracs])
# #         ax.set_xlabel("Subsample fraction")
# #         ax.set_title(ds)

# #     fig.tight_layout()
# #     for fmt in ("pdf", "png"):
# #         fig.savefig(f"{out}.{fmt}", format=fmt)
# #     plt.close(fig)
# #     print(f"→ {out}.pdf / .png")


# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--csv", default="mia_subsample_results/raw_data/mia_subsample_raw.csv")
# #     ap.add_argument("--out-dir", default="mia_subsample_results/plots")
# #     args = ap.parse_args()
# #     out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
# #     setup_style()
# #     df = load(args.csv)
# #     fig_gap_auc(df, out / "subsample_gap_auc")
# #     fig_clean_tier(df, out / "subsample_clean_tier")
# #     print("✅ Done.")


# # if __name__ == "__main__":
# #     main()

# """
# plot_subsample.py
# =================
# Figures for the subsampling study.

# Produces:
#   subsample_gap_auc.{pdf,png}    Pooled scatter: train/test gap (x) vs MIA AUC (y),
#                                  one point per subsampled configuration, coloured
#                                  by dataset. The controlled analogue of the main
#                                  gap-vs-AUC figure.
#   subsample_clean_tier.{pdf,png} Grouped twin-axis line plots. Top row: the
#                                  'coupled' datasets where gap and AUC rise
#                                  together as the subsample fraction falls.
#                                  Bottom row: the 'decoupled' datasets where a
#                                  flat gap keeps AUC at chance. y-scales are
#                                  locked within each row so shapes are directly
#                                  comparable across panels.

# Usage
# -----
#   python plot_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv \\
#       --out-dir mia_subsample_results/plots \\
#       --font-scale 1.4
# """
# from __future__ import annotations
# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.gridspec import GridSpec
# from matplotlib.lines import Line2D
# from scipy.stats import spearmanr

# mpl.use("Agg")

# GAP = "train_test_gap"
# AUC = "shokri_mia_auc"

# # Module-level font scale, set from --font-scale in main().
# FONT_SCALE = 1.0

# # Shared colour coding for the twin-axis convention.
# C_GAP = "#d62728"     # red:  train/test gap
# C_AUC = "#1f77b4"     # blue: MIA AUC

# DATASET_COLORS = {
#     "roman_empire":   "#bcbd22",
#     "amazon_ratings": "#17becf",
#     "minesweeper":    "#7f7f7f",
#     "tolokers":       "#aec7e8",
#     "Computers":      "#2ca02c",
#     "Photo":          "#9467bd",
#     "WikiCS":         "#e377c2",
# }

# COUPLED_DATASETS   = ["Computers", "Photo", "WikiCS",
#                       "amazon_ratings", "roman_empire"]
# DECOUPLED_DATASETS = ["minesweeper", "tolokers"]


# def setup_style(font_scale: float = 1.0):
#     global FONT_SCALE
#     FONT_SCALE = font_scale

#     def S(base: float) -> float:
#         return base * font_scale

#     plt.rcParams.update({
#         "font.family":       "sans-serif",
#         "font.size":         S(11),
#         "axes.titlesize":    S(12),
#         "axes.labelsize":    S(11),
#         "xtick.labelsize":   S(10),
#         "ytick.labelsize":   S(10),
#         "axes.spines.top":   False,
#         "axes.spines.right": False,
#         "legend.frameon":    False,
#         "legend.fontsize":       S(11),
#         "legend.title_fontsize": S(12),
#         "figure.dpi":        120,
#         "savefig.dpi":       220,
#         "savefig.bbox":      "tight",
#     })


# def load(csv):
#     df = pd.read_csv(csv)
#     if "error" in df.columns:
#         df = df[df["error"].isna() | (df["error"].astype(str) == "nan")]
#     return df


# # ═══════════════════════════════════════════════════════════════════════════
# #  Figure A: pooled gap-vs-AUC scatter
# # ═══════════════════════════════════════════════════════════════════════════

# def fig_gap_auc(df, out):
#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     fig, ax = plt.subplots(figsize=(7, 5.5))
#     x = df[GAP].values.astype(float)
#     y = df[AUC].values.astype(float)

#     for ds in sorted(df.dataset.unique()):
#         m = df.dataset.values == ds
#         ax.scatter(df[GAP].values[m], df[AUC].values[m], s=32, alpha=0.65,
#                    color=DATASET_COLORS.get(ds, "#3b6fb0"),
#                    edgecolors="none", label=ds)

#     keep = ~np.isnan(x) & ~np.isnan(y)
#     slope, intercept = np.polyfit(x[keep], y[keep], 1)
#     xs = np.linspace(x[keep].min(), x[keep].max(), 100)
#     ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9)
#     ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

#     rho, p = spearmanr(x[keep], y[keep])
#     ax.text(0.03, 0.97,
#             f"Spearman $\\rho$ = {rho:.2f}\np < 0.001\nn = {int(keep.sum())}",
#             transform=ax.transAxes, va="top", ha="left", fontsize=S(11),
#             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
#                       edgecolor="lightgray", alpha=0.9))

#     ax.set_xlabel("Train / test accuracy gap", fontsize=S(12))
#     ax.set_ylabel("MIA AUC", fontsize=S(12))
#     ax.tick_params(axis="both", labelsize=S(11))

#     ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
#               title="Dataset", fontsize=S(10), title_fontsize=S(11),
#               borderaxespad=0.0, labelspacing=0.4)

#     fig.tight_layout()
#     for fmt in ("pdf", "png"):
#         fig.savefig(f"{out}.{fmt}", format=fmt)
#     plt.close(fig)
#     print(f"→ {out}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  Figure B: grouped twin-axis panels — coupled on top, decoupled below
# # ═══════════════════════════════════════════════════════════════════════════

# def _draw_panel(ax, df, ds, fracs,
#                 show_gap_label=True, show_auc_label=True):
#     """One twin-axis panel. Red = gap (left), blue = AUC (right).

#     `show_gap_label` / `show_auc_label` control whether the *axis title* is
#     drawn — tick numbers always stay.  Use them to label only the outermost
#     panels of a row, so inner panels don't collide with their neighbours.
#     """
#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     sub = df[df.dataset == ds]
#     if sub.empty:
#         ax.set_visible(False)
#         return None, None

#     g = (sub.groupby("fraction")
#             .agg(gap=(GAP, "mean"), auc=(AUC, "mean"),
#                  gap_s=(GAP, "std"), auc_s=(AUC, "std"))
#             .reindex(fracs))
#     xpos = np.arange(len(fracs))

#     # Left axis: gap
#     ax.errorbar(xpos, g["gap"], yerr=g["gap_s"], color=C_GAP,
#                 marker="o", markersize=4, capsize=3, linewidth=1.6)
#     if show_gap_label:
#         ax.set_ylabel("Train / test gap", color=C_GAP, fontsize=S(11))
#     ax.tick_params(axis="y", labelcolor=C_GAP, labelsize=S(10))
#     ax.axhline(0, color=C_GAP, linestyle=":", linewidth=0.7, alpha=0.5)
#     ax.spines["top"].set_visible(False)

#     # Right axis: AUC
#     ax2 = ax.twinx()
#     ax2.errorbar(xpos, g["auc"], yerr=g["auc_s"], color=C_AUC,
#                  marker="s", markersize=4, capsize=3, linewidth=1.6)
#     if show_auc_label:
#         ax2.set_ylabel("MIA AUC", color=C_AUC, fontsize=S(11))
#     ax2.tick_params(axis="y", labelcolor=C_AUC, labelsize=S(10))
#     ax2.axhline(0.5, color=C_AUC, linestyle=":", linewidth=0.7, alpha=0.5)
#     ax2.spines["top"].set_visible(False)

#     ax.set_xticks(xpos)
#     ax.set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=S(10))
#     ax.set_xlabel("Subsample fraction", fontsize=S(11))
#     ax.set_title(ds, fontsize=S(12))

#     return ax, ax2


# def fig_clean_tier(df, out, coupled=None, decoupled=None):
#     """Two rows of twin-axis panels.

#     Top    = 'coupled'   datasets: as size falls, gap and AUC rise together.
#     Bottom = 'decoupled' datasets: flat gap, AUC pinned at chance.

#     y-scales are locked *within each row* (gap row and AUC row separately),
#     so curve shapes are directly comparable across panels."""
#     if coupled is None:
#         coupled = COUPLED_DATASETS
#     if decoupled is None:
#         decoupled = DECOUPLED_DATASETS

#     coupled   = [d for d in coupled   if d in df.dataset.unique()]
#     decoupled = [d for d in decoupled if d in df.dataset.unique()]

#     n_top, n_bot = len(coupled), len(decoupled)
#     if n_top == 0 and n_bot == 0:
#         print("⚠ no datasets to plot in fig_clean_tier")
#         return

#     def S(base: float) -> float:
#         return base * FONT_SCALE

#     # ── Figure geometry ─────────────────────────────────────────────────
#     # Wider figure so 5 top panels aren't cramped; extra left margin for
#     # the row labels ("coupled" / "decoupled").
#     width = 3.0 * n_top + 1.5
#     fig = plt.figure(figsize=(width, 7.4))

#     # Top row: spans almost the full figure width.
#     top_gs = fig.add_gridspec(1, n_top,
#                               left=0.055, right=0.965,
#                               top=0.93, bottom=0.56,
#                               wspace=0.85)   # generous: twin-axis labels

#     # Bottom row: narrower + centred, so its panels are ~the same width
#     # as the top ones (avoids the "stretched" look of the previous version).
#     bot_gs = fig.add_gridspec(1, n_bot,
#                               left=0.16, right=0.86,
#                               top=0.385, bottom=0.07,
#                               wspace=0.35)

#     top_axes, top_twins = [], []
#     bot_axes, bot_twins = [], []

#     # ── Top row ─────────────────────────────────────────────────────────
#     for j, ds in enumerate(coupled):
#         ax = fig.add_subplot(top_gs[0, j])
#         fracs = sorted(df[df.dataset == ds].fraction.unique(), reverse=True)
#         ax, ax2 = _draw_panel(
#             ax, df, ds, fracs,
#             show_gap_label=(j == 0),          # leftmost panel only
#             show_auc_label=(j == n_top - 1),  # rightmost panel only
#         )
#         if ax is not None:
#             top_axes.append(ax)
#             top_twins.append(ax2)

#     # ── Bottom row ──────────────────────────────────────────────────────
#     for j, ds in enumerate(decoupled):
#         ax = fig.add_subplot(bot_gs[0, j])
#         fracs = sorted(df[df.dataset == ds].fraction.unique(), reverse=True)
#         ax, ax2 = _draw_panel(
#             ax, df, ds, fracs,
#             show_gap_label=(j == 0),
#             show_auc_label=(j == n_bot - 1),
#         )
#         if ax is not None:
#             bot_axes.append(ax)
#             bot_twins.append(ax2)

#     # ── Lock y-scales within each row ───────────────────────────────────
#     def _lock(axes_list):
#         axes_list = [a for a in axes_list if a is not None]
#         if len(axes_list) < 2:
#             return
#         lo = min(a.get_ylim()[0] for a in axes_list)
#         hi = max(a.get_ylim()[1] for a in axes_list)
#         pad = 0.05 * (hi - lo) if hi > lo else 0.01
#         for a in axes_list:
#             a.set_ylim(lo - pad, hi + pad)

#     _lock(top_axes);   _lock(top_twins)
#     _lock(bot_axes);   _lock(bot_twins)

#     # ── Row labels at the figure edge (left margin was reserved above) ──
#     def _row_label(y_fig, text):
#         fig.text(0.012, y_fig, text,
#                  rotation=90, va="center", ha="left",
#                  fontsize=S(12), fontweight="bold", color="#444444")

#     # _row_label((0.56 + 0.93) / 2, "coupled")
#     # _row_label((0.07 + 0.385) / 2, "decoupled")

#     # ── Shared key: red = gap, blue = AUC ───────────────────────────────
#     key_handles = [
#         Line2D([0], [0], color=C_GAP, marker="o", markersize=6,
#                linewidth=1.6, label="Train / test gap  (left axis)"),
#         Line2D([0], [0], color=C_AUC, marker="s", markersize=6,
#                linewidth=1.6, label="MIA AUC  (right axis)"),
#     ]
#     # fig.legend(handles=key_handles, loc="upper center",
#     #            bbox_to_anchor=(0.5, 0.995), ncol=2,
#     #            fontsize=S(11), frameon=False)

#     fig.savefig(f"{out}.pdf", format="pdf")
#     fig.savefig(f"{out}.png")
#     plt.close(fig)
#     print(f"→ {out}.pdf / .png")


# # ═══════════════════════════════════════════════════════════════════════════
# #  Main
# # ═══════════════════════════════════════════════════════════════════════════

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv",
#                     default="mia_subsample_results/raw_data/mia_subsample_raw.csv")
#     ap.add_argument("--out-dir",
#                     default="mia_subsample_results/plots")
#     ap.add_argument("--font-scale", type=float, default=1.0,
#                     help="Multiplier applied to all text sizes. "
#                          "Try 1.2–1.8 for a paper figure.")
#     args = ap.parse_args()

#     out = Path(args.out_dir)
#     out.mkdir(parents=True, exist_ok=True)

#     setup_style(font_scale=args.font_scale)
#     print(f"Font scale: {args.font_scale}×")

#     df = load(args.csv)
#     print(f"Loaded {len(df)} rows, datasets: {sorted(df.dataset.unique())}")

#     fig_gap_auc(df, out / "subsample_gap_auc")
#     fig_clean_tier(df, out / "subsample_clean_tier")
#     print("✅ Done.")


# if __name__ == "__main__":
#     main()

"""
plot_subsample.py
=================
Figures for the subsampling study.

Produces:
  subsample_gap_auc.{pdf,png}    Pooled scatter: train/test gap (x) vs MIA AUC (y),
                                 one point per subsampled configuration, coloured
                                 by dataset.
  subsample_clean_tier.{pdf,png} Grouped twin-axis line plots. Top row: 'coupled'
                                 datasets. Bottom row: 'decoupled' datasets.
                                 y-scales locked within each row.

Usage
-----
  python plot_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv \\
      --out-dir mia_subsample_results/plots \\
      --font-scale 1.4 --legend-boost 1.8
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

mpl.use("Agg")

GAP = "train_test_gap"
AUC = "shokri_mia_auc"

# ── Scaling globals, set by setup_style() from the CLI flags ────────────────
# Base text multiplier        : linear  in --font-scale
# Legend text multiplier      : power   FONT_SCALE ** LEGEND_BOOST
#   - legend_boost = 1.0  ->  identical to base text (old behaviour)
#   - legend_boost = 1.8  ->  legends grow *exponentially* with font-scale
FONT_SCALE   = 1.0
LEGEND_BOOST = 1.0


def S(base: float) -> float:
    """Base text scale (linear in --font-scale)."""
    return base * FONT_SCALE


def L(base: float) -> float:
    """Legend text scale: base * FONT_SCALE ** LEGEND_BOOST.

    LEGEND_BOOST == 1.0 collapses this to the plain linear S().
    """
    return base * (FONT_SCALE ** LEGEND_BOOST)


# Shared colour coding for the twin-axis convention.
C_GAP = "#d62728"     # red:  train/test gap
C_AUC = "#1f77b4"     # blue: MIA AUC

DATASET_COLORS = {
    "roman_empire":   "#bcbd22",
    "amazon_ratings": "#17becf",
    "minesweeper":    "#7f7f7f",
    "tolokers":       "#aec7e8",
    "Computers":      "#2ca02c",
    "Photo":          "#9467bd",
    "WikiCS":         "#e377c2",
}

COUPLED_DATASETS   = ["Computers", "Photo", "WikiCS",
                      "amazon_ratings", "roman_empire"]
DECOUPLED_DATASETS = ["minesweeper", "tolokers"]


def setup_style(font_scale: float = 1.0, legend_boost: float = 1.0):
    global FONT_SCALE, LEGEND_BOOST
    FONT_SCALE = font_scale
    LEGEND_BOOST = legend_boost

    s = S   # local aliases, purely for readability in the dict below
    l = L

    plt.rcParams.update({
        "font.family":           "sans-serif",
        "font.size":             s(11),
        "axes.titlesize":        s(12),
        "axes.labelsize":        s(11),
        "xtick.labelsize":       s(10),
        "ytick.labelsize":       s(10),
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "legend.frameon":        False,
        "legend.fontsize":       l(11),   # legend grows exponentially
        "legend.title_fontsize": l(12),
        "figure.dpi":            120,
        "savefig.dpi":           220,
        "savefig.bbox":          "tight",
    })


def load(csv):
    df = pd.read_csv(csv)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str) == "nan")]
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Figure A: pooled gap-vs-AUC scatter
# ═══════════════════════════════════════════════════════════════════════════

def fig_gap_auc(df, out):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = df[GAP].values.astype(float)
    y = df[AUC].values.astype(float)

    for ds in sorted(df.dataset.unique()):
        m = df.dataset.values == ds
        ax.scatter(df[GAP].values[m], df[AUC].values[m], s=32, alpha=0.65,
                   color=DATASET_COLORS.get(ds, "#3b6fb0"),
                   edgecolors="none", label=ds)

    keep = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    xs = np.linspace(x[keep].min(), x[keep].max(), 100)
    ax.plot(xs, slope * xs + intercept, "k--", linewidth=2, alpha=0.9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    rho, p = spearmanr(x[keep], y[keep])
    ax.text(0.03, 0.97,
            f"Spearman $\\rho$ = {rho:.2f}\np < 0.001\nn = {int(keep.sum())}",
            transform=ax.transAxes, va="top", ha="left", fontsize=S(11),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))

    ax.set_xlabel("Train / test accuracy gap", fontsize=S(12))
    ax.set_ylabel("MIA AUC", fontsize=S(12))
    ax.tick_params(axis="both", labelsize=S(11))

    # Legend: sizes go through L(); spacing scales with the same boost so the
    # entries breathe as they grow.
    boost = FONT_SCALE ** LEGEND_BOOST / FONT_SCALE  # extra factor vs base
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="Dataset",
              fontsize=L(10), title_fontsize=L(11),
              borderaxespad=0.0,
              labelspacing=0.4 * boost,
              handletextpad=0.6 * boost)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(f"{out}.{fmt}", format=fmt)
    plt.close(fig)
    print(f"→ {out}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure B: grouped twin-axis panels — coupled on top, decoupled below
# ═══════════════════════════════════════════════════════════════════════════

def _draw_panel(ax, df, ds, fracs,
                show_gap_label=True, show_auc_label=True):
    """One twin-axis panel. Red = gap (left), blue = AUC (right)."""
    sub = df[df.dataset == ds]
    if sub.empty:
        ax.set_visible(False)
        return None, None

    g = (sub.groupby("fraction")
            .agg(gap=(GAP, "mean"), auc=(AUC, "mean"),
                 gap_s=(GAP, "std"), auc_s=(AUC, "std"))
            .reindex(fracs))
    xpos = np.arange(len(fracs))

    ax.errorbar(xpos, g["gap"], yerr=g["gap_s"], color=C_GAP,
                marker="o", markersize=4, capsize=3, linewidth=1.6)
    if show_gap_label:
        ax.set_ylabel("Train / test gap", color=C_GAP, fontsize=S(11))
    ax.tick_params(axis="y", labelcolor=C_GAP, labelsize=S(10))
    ax.axhline(0, color=C_GAP, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.errorbar(xpos, g["auc"], yerr=g["auc_s"], color=C_AUC,
                 marker="s", markersize=4, capsize=3, linewidth=1.6)
    if show_auc_label:
        ax2.set_ylabel("MIA AUC", color=C_AUC, fontsize=S(11))
    ax2.tick_params(axis="y", labelcolor=C_AUC, labelsize=S(10))
    ax2.axhline(0.5, color=C_AUC, linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.spines["top"].set_visible(False)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=S(10))
    ax.set_xlabel("Subsample fraction", fontsize=S(11))
    ax.set_title(ds, fontsize=S(12))

    return ax, ax2


def fig_clean_tier(df, out, coupled=None, decoupled=None):
    if coupled is None:
        coupled = COUPLED_DATASETS
    if decoupled is None:
        decoupled = DECOUPLED_DATASETS

    coupled   = [d for d in coupled   if d in df.dataset.unique()]
    decoupled = [d for d in decoupled if d in df.dataset.unique()]

    n_top, n_bot = len(coupled), len(decoupled)
    if n_top == 0 and n_bot == 0:
        print("⚠ no datasets to plot in fig_clean_tier")
        return

    width = 3.0 * n_top + 1.5
    fig = plt.figure(figsize=(width, 7.4))

    top_gs = fig.add_gridspec(1, n_top,
                              left=0.055, right=0.965,
                              top=0.93, bottom=0.56,
                              wspace=0.85)
    bot_gs = fig.add_gridspec(1, n_bot,
                              left=0.16, right=0.86,
                              top=0.385, bottom=0.07,
                              wspace=0.35)

    top_axes, top_twins = [], []
    bot_axes, bot_twins = [], []

    for j, ds in enumerate(coupled):
        ax = fig.add_subplot(top_gs[0, j])
        fracs = sorted(df[df.dataset == ds].fraction.unique(), reverse=True)
        ax, ax2 = _draw_panel(ax, df, ds, fracs,
                              show_gap_label=(j == 0),
                              show_auc_label=(j == n_top - 1))
        if ax is not None:
            top_axes.append(ax); top_twins.append(ax2)

    for j, ds in enumerate(decoupled):
        ax = fig.add_subplot(bot_gs[0, j])
        fracs = sorted(df[df.dataset == ds].fraction.unique(), reverse=True)
        ax, ax2 = _draw_panel(ax, df, ds, fracs,
                              show_gap_label=(j == 0),
                              show_auc_label=(j == n_bot - 1))
        if ax is not None:
            bot_axes.append(ax); bot_twins.append(ax2)

    def _lock(axes_list):
        axes_list = [a for a in axes_list if a is not None]
        if len(axes_list) < 2:
            return
        lo = min(a.get_ylim()[0] for a in axes_list)
        hi = max(a.get_ylim()[1] for a in axes_list)
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        for a in axes_list:
            a.set_ylim(lo - pad, hi + pad)

    _lock(top_axes);   _lock(top_twins)
    _lock(bot_axes);   _lock(bot_twins)

    # ── Shared key: red = gap, blue = AUC (now visible, sizes via L()) ──
    boost = FONT_SCALE ** LEGEND_BOOST / FONT_SCALE
    key_handles = [
        Line2D([0], [0], color=C_GAP, marker="o",
               markersize=6 * boost, linewidth=1.6,
               label="Train / test gap  (left axis)"),
        Line2D([0], [0], color=C_AUC, marker="s",
               markersize=6 * boost, linewidth=1.6,
               label="MIA AUC  (right axis)"),
    ]
    # fig.legend(handles=key_handles, loc="upper center",
    #            bbox_to_anchor=(0.5, 0.995), ncol=2,
    #            fontsize=L(11), frameon=False)

    fig.savefig(f"{out}.pdf", format="pdf")
    fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"→ {out}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="mia_subsample_results/raw_data/mia_subsample_raw.csv")
    ap.add_argument("--out-dir",
                    default="mia_subsample_results/plots")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="Linear multiplier for all text sizes. "
                         "Try 1.2–1.8 for a paper figure.")
    ap.add_argument("--legend-boost", type=float, default=1.0,
                    help="Exponent applied to --font-scale for LEGEND text only: "
                         "legend_scale = font_scale ** legend_boost. "
                         "1.0 = same as base text (old behaviour); "
                         "1.5–2.0 = legends grow exponentially and stand out.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    setup_style(font_scale=args.font_scale, legend_boost=args.legend_boost)

    base_mult   = args.font_scale
    legend_mult = args.font_scale ** args.legend_boost
    print(f"font-scale   = {args.font_scale}")
    print(f"legend-boost = {args.legend_boost}")
    print(f"→ base text multiplier   = {base_mult:.3f}×")
    print(f"→ legend text multiplier = {legend_mult:.3f}×  "
          f"({legend_mult / base_mult:.3f}× larger than base)")

    df = load(args.csv)
    print(f"Loaded {len(df)} rows, datasets: {sorted(df.dataset.unique())}")

    fig_gap_auc(df, out / "subsample_gap_auc")
    fig_clean_tier(df, out / "subsample_clean_tier")
    print("✅ Done.")


if __name__ == "__main__":
    main()