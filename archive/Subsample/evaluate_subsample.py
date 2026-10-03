# """
# evaluate_subsample.py
# =====================
# Analyse the subsampling study. Two things to show:

#   1. Topology held roughly fixed as size fell (the validity check). For each
#      dataset we print avg degree, homophily, and label-modularity at each
#      fraction; if these are flat, the subsample changed size but not topology.

#   2. As size fell, did the train/test gap widen and did MIA follow it? We
#      report, per dataset, the gap and MIA AUC / TPR@5% at each fraction, and
#      the overall correlation between gap and MIA across all subsample rows.

# Usage
# -----
#   python evaluate_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv
# """
# from __future__ import annotations

# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from scipy.stats import spearmanr

# AUC = "shokri_mia_auc"
# TPR5 = "shokri_mia_tpr_at_fpr_0.05"
# GAP = "train_test_gap"


# def load(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     if "error" in df.columns:
#         df = df[df["error"].isna()] if df["error"].notna().any() else df
#     return df


# def topology_check(df: pd.DataFrame):
#     print("\n" + "=" * 78)
#     print(" TOPOLOGY CHECK — did the sampler hold topology while shrinking?")
#     print(" (avg degree / homophily / label-modularity should stay ~flat down each block)")
#     print("=" * 78)
#     for ds in sorted(df.dataset.unique()):
#         sub = df[df.dataset == ds]
#         print(f"\n  {ds}")
#         print(f"    {'frac':>5} {'nodes':>7} {'avg_deg':>8} {'homophily':>10} {'Q_label':>9}")
#         g = (sub.groupby("fraction")
#                 .agg(nodes=("sub_nodes", "mean"),
#                      avg_deg=("sub_avg_degree", "mean"),
#                      homophily=("sub_homophily", "mean"),
#                      Q=("sub_modularity_labels", "mean"))
#                 .reset_index().sort_values("fraction", ascending=False))
#         for _, r in g.iterrows():
#             print(f"    {r.fraction:>5.2f} {int(r.nodes):>7} {r.avg_deg:>8.2f} "
#                   f"{r.homophily:>10.3f} {r.Q:>9.3f}")


# def size_gap_mia(df: pd.DataFrame):
#     print("\n" + "=" * 78)
#     print(" SIZE -> GAP -> MIA — per dataset, per fraction (client + node pooled)")
#     print("=" * 78)
#     for ds in sorted(df.dataset.unique()):
#         sub = df[df.dataset == ds]
#         print(f"\n  {ds}")
#         print(f"    {'frac':>5} {'gap (μ±σ)':>16} {'MIA AUC (μ±σ)':>18} {'TPR@5% (μ±σ)':>16}")
#         g = (sub.groupby("fraction")
#                 .agg(gap_m=(GAP, "mean"), gap_s=(GAP, "std"),
#                      auc_m=(AUC, "mean"), auc_s=(AUC, "std"),
#                      tpr_m=(TPR5, "mean"), tpr_s=(TPR5, "std"))
#                 .reset_index().sort_values("fraction", ascending=False))
#         for _, r in g.iterrows():
#             print(f"    {r.fraction:>5.2f} "
#                   f"{r.gap_m:>7.3f} ± {r.gap_s:<6.3f} "
#                   f"{r.auc_m:>7.3f} ± {r.auc_s:<6.3f} "
#                   f"{r.tpr_m:>6.3f} ± {r.tpr_s:<6.3f}")


# def correlations(df: pd.DataFrame):
#     print("\n" + "=" * 78)
#     print(" DOES MIA FOLLOW THE GAP ACROSS ALL SUBSAMPLE ROWS?")
#     print("=" * 78)
#     for col, label in [(AUC, "MIA AUC"), (TPR5, "TPR@5%")]:
#         d = df[[GAP, col]].dropna()
#         if len(d) >= 5:
#             rho, p = spearmanr(d[GAP], d[col])
#             print(f"    gap vs {label:10s}:  Spearman ρ = {rho:+.3f}  p = {p:.4f}  n = {len(d)}")

#     # Does MIA follow SIZE directly, and is that weaker than the gap link?
#     d = df[["fraction", AUC]].dropna()
#     rho, p = spearmanr(d["fraction"], d[AUC])
#     print(f"    fraction vs MIA AUC:  Spearman ρ = {rho:+.3f}  p = {p:.4f}  n = {len(d)}")
#     print("\n  Reading: if gap↔MIA is strong and topology stayed flat (above),")
#     print("  then shrinking raised the gap and MIA followed — size acts through the gap,")
#     print("  not through topology.")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", default="mia_subsample_results/raw_data/mia_subsample_raw.csv")
#     args = ap.parse_args()
#     df = load(Path(args.csv))
#     print(f"Loaded {len(df)} rows | datasets: {sorted(df.dataset.unique())}")
#     topology_check(df)
#     size_gap_mia(df)
#     correlations(df)
#     print("\n✅ Done.")


# if __name__ == "__main__":
#     main()

"""
evaluate_subsample.py
=====================
Analyse the subsampling study. Two things to show:

  1. Topology held roughly fixed as size fell (the validity check). For each
     dataset we print avg degree, homophily, and label-modularity at each
     fraction; if these are flat, the subsample changed size but not topology.

  2. As size fell, did the train/test gap widen and did MIA follow it? We
     report, per dataset, the gap and MIA AUC / TPR@5% at each fraction, and
     the overall correlation between gap and MIA across all subsample rows.

Usage
-----
  python evaluate_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AUC = "shokri_mia_auc"
TPR5 = "shokri_mia_tpr_at_fpr_0.05"
GAP = "train_test_gap"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    return df


def topology_check(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print(" TOPOLOGY CHECK — did the sampler hold topology while shrinking?")
    print(" (avg degree / homophily / label-modularity should stay ~flat down each block)")
    print("=" * 78)
    for ds in sorted(df.dataset.unique()):
        sub = df[df.dataset == ds]
        print(f"\n  {ds}")
        print(f"    {'frac':>5} {'nodes':>7} {'avg_deg':>8} {'homophily':>10} {'Q_label':>9}")
        g = (sub.groupby("fraction")
                .agg(nodes=("sub_nodes", "mean"),
                     avg_deg=("sub_avg_degree", "mean"),
                     homophily=("sub_homophily", "mean"),
                     Q=("sub_modularity_labels", "mean"))
                .reset_index().sort_values("fraction", ascending=False))
        for _, r in g.iterrows():
            print(f"    {r.fraction:>5.2f} {int(r.nodes):>7} {r.avg_deg:>8.2f} "
                  f"{r.homophily:>10.3f} {r.Q:>9.3f}")


def size_gap_mia(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print(" SIZE -> GAP -> MIA — per dataset, per fraction (client + node pooled)")
    print("=" * 78)
    for ds in sorted(df.dataset.unique()):
        sub = df[df.dataset == ds]
        print(f"\n  {ds}")
        print(f"    {'frac':>5} {'gap (μ±σ)':>16} {'MIA AUC (μ±σ)':>18} {'TPR@5% (μ±σ)':>16}")
        have_tpr = TPR5 in sub.columns
        agg_kwargs = dict(gap_m=(GAP, "mean"), gap_s=(GAP, "std"),
                          auc_m=(AUC, "mean"), auc_s=(AUC, "std"))
        if have_tpr:
            agg_kwargs.update(tpr_m=(TPR5, "mean"), tpr_s=(TPR5, "std"))
        g = (sub.groupby("fraction").agg(**agg_kwargs)
                .reset_index().sort_values("fraction", ascending=False))
        for _, r in g.iterrows():
            line = (f"    {r.fraction:>5.2f} "
                    f"{r.gap_m:>7.3f} ± {r.gap_s:<6.3f} "
                    f"{r.auc_m:>7.3f} ± {r.auc_s:<6.3f}")
            if have_tpr:
                line += f" {r.tpr_m:>6.3f} ± {r.tpr_s:<6.3f}"
            print(line)


def correlations(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print(" DOES MIA FOLLOW THE GAP ACROSS ALL SUBSAMPLE ROWS?")
    print("=" * 78)
    metric_cols = [(AUC, "MIA AUC")]
    if TPR5 in df.columns:
        metric_cols.append((TPR5, "TPR@5%"))
    for col, label in metric_cols:
        d = df[[GAP, col]].dropna()
        if len(d) >= 5:
            rho, p = spearmanr(d[GAP], d[col])
            print(f"    gap vs {label:10s}:  Spearman ρ = {rho:+.3f}  p = {p:.4f}  n = {len(d)}")

    # Does MIA follow SIZE directly, and is that weaker than the gap link?
    d = df[["fraction", AUC]].dropna()
    rho, p = spearmanr(d["fraction"], d[AUC])
    print(f"    fraction vs MIA AUC:  Spearman ρ = {rho:+.3f}  p = {p:.4f}  n = {len(d)}")
    print("\n  Reading: if gap↔MIA is strong and topology stayed flat (above),")
    print("  then shrinking raised the gap and MIA followed — size acts through the gap,")
    print("  not through topology.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mia_subsample_results/raw_data/mia_subsample_raw.csv")
    args = ap.parse_args()
    df = load(Path(args.csv))
    print(f"Loaded {len(df)} rows | datasets: {sorted(df.dataset.unique())}")
    topology_check(df)
    size_gap_mia(df)
    correlations(df)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()