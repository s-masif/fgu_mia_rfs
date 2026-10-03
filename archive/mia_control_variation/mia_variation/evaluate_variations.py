"""
evaluate_variations.py
======================
Correlation analysis for the Phase 2 variation sweep.

Reads all mia_variations_raw_*.csv files under `--dir` and computes:
  1. Per-axis sensitivity: how MIA AUC varies with partitioning / K /
     forget_ratio, plotted as tables per dataset.
  2. Univariate Spearman correlations between MIA AUC and each candidate
     predictor: homophily, modularity, n_nodes, train_test_gap, forget_ratio,
     num_clients.
  3. Multivariate OLS regression using train_test_gap as the primary
     control. Reports how much variance is explained by (a) train_test_gap
     alone, (b) train_test_gap + partitioning-specific effects, (c) all
     predictors combined.

Usage
-----
  python evaluate_variations.py \\
      --dir mia_variations_results/raw_data
"""
from __future__ import annotations

import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CANDIDATE_PREDICTORS = [
    ("homophily",       "Homophily (raw)"),
    ("modularity",      "Modularity Q"),
    ("n_nodes",         "Total nodes"),
    ("n_classes",       "Number of classes"),
    ("num_clients",     "Number of clients K"),
    ("forget_ratio",    "Forget ratio"),
    ("train_test_gap",  "Train/Test acc gap"),
]


def load_all_csvs(directory: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "mia_variations_raw*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {directory}")
    dfs = [pd.read_csv(f) for f in files]
    print(f"Loaded {len(files)} CSV(s):")
    for f, df in zip(files, dfs):
        print(f"  {Path(f).name}: {len(df)} rows")
    return pd.concat(dfs, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-axis sensitivity tables
# ═══════════════════════════════════════════════════════════════════════════

def per_axis_sensitivity(df: pd.DataFrame, axis: str, out_dir: Path):
    """Aggregate multiple MIA metrics by axis value, per (dataset, scenario)."""
    print(f"\n{'═' * 80}\n  Sensitivity to '{axis}'\n{'═' * 80}")

    metric_cols = [
        ("shokri_mia_auc",           "AUC"),
        ("shokri_mia_balanced_acc",  "BalAcc"),
        ("shokri_mia_tpr_at_fpr_0.01", "TPR@0.01"),
        ("shokri_mia_tpr_at_fpr_0.001","TPR@0.001"),
    ]
    for col, label in metric_cols:
        if col not in df.columns:
            continue
        piv = df.pivot_table(
            index=["dataset", "scenario"],
            columns=axis,
            values=col,
            aggfunc=lambda x: f"{np.mean(x):.3f} ± {np.std(x):.3f}"
        )
        print(f"\n  ── {label} ──")
        print(piv.to_string())
        piv.to_csv(out_dir / f"sensitivity_{axis}_{label.replace('@', '_at_').replace('.', 'p')}.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  Univariate correlations
# ═══════════════════════════════════════════════════════════════════════════

def univariate_correlations(df: pd.DataFrame, out_dir: Path):
    print(f"\n{'═' * 80}\n  Univariate Spearman correlations with MIA AUC\n{'═' * 80}")
    print(f"  (n={len(df)} rows across all sweep configurations)\n")

    rows = []
    y = df["shokri_mia_auc"].values
    header = f"  {'Predictor':<22} | {'Spearman ρ':>12} | {'p-value':>10} | {'Interpretation'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for col, label in CANDIDATE_PREDICTORS:
        if col not in df.columns:
            continue
        x = df[col].values
        keep = ~np.isnan(x) & ~np.isnan(y)
        if keep.sum() < 5:
            continue
        rho, p = spearmanr(x[keep], y[keep])
        interp = "strong" if abs(rho) > 0.5 else ("moderate" if abs(rho) > 0.3 else "weak/none")
        print(f"  {label:<22} | {rho:+12.3f} | {p:10.4f} | {interp}")
        rows.append({"predictor": label, "spearman_rho": rho, "p_value": p,
                     "n": int(keep.sum())})

    pd.DataFrame(rows).to_csv(out_dir / "univariate_correlations.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Multivariate regression with train/test gap as primary control
# ═══════════════════════════════════════════════════════════════════════════

def multivariate_regression(df: pd.DataFrame, out_dir: Path):
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("\n⚠ sklearn not available — skipping multivariate regression")
        return

    print(f"\n{'═' * 80}\n  Multivariate regression: how much variance in MIA AUC")
    print(f"  is explained by each set of predictors?\n{'═' * 80}")

    y = df["shokri_mia_auc"].values
    keep_row = ~np.isnan(y)
    y = y[keep_row]

    def _fit(cols, name):
        X_arr = []
        for c in cols:
            if c not in df.columns:
                print(f"    ⚠ predictor {c} missing, skipping this model")
                return
            X_arr.append(df[c].values[keep_row])
        X = np.column_stack(X_arr)
        keep = ~np.isnan(X).any(axis=1)
        if keep.sum() < 5:
            return
        R2 = LinearRegression().fit(X[keep], y[keep]).score(X[keep], y[keep])
        print(f"  {name:<48}  R² = {R2:.3f}  (n = {keep.sum()})")
        return R2

    _fit(["train_test_gap"],                                    "Train/test gap only")
    _fit(["homophily"],                                          "Homophily only")
    _fit(["n_nodes"],                                            "Dataset size only")
    _fit(["modularity"],                                         "Modularity only")
    _fit(["train_test_gap", "homophily"],                        "Gap + homophily")
    _fit(["train_test_gap", "n_nodes"],                          "Gap + size")
    _fit(["train_test_gap", "modularity"],                       "Gap + modularity")
    _fit(["train_test_gap", "homophily", "n_nodes",
          "modularity", "forget_ratio", "num_clients"],          "All predictors")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="mia_variations_results/raw_data",
                        help="Directory containing mia_variations_raw*.csv")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    df = load_all_csvs(dir_path)

    out_dir = dir_path.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = int(df.seed.nunique())
    print(f"\n{'═' * 80}")
    print(f"  DATASET SUMMARY (Phase 2 variation sweep)")
    print(f"{'═' * 80}")
    print(f"  Total rows:      {len(df)}")
    print(f"  Datasets:        {sorted(df.dataset.unique())}")
    print(f"  Partitionings:   {sorted(df.partitioning.unique())}")
    print(f"  K values:        {sorted(df.num_clients.unique())}")
    print(f"  Forget ratios:   {sorted(df.forget_ratio.unique())}")
    print(f"  Seeds:           {sorted(df.seed.unique())}  (n_seeds = {n_seeds})")
    print(f"  Reported μ ± σ are computed across {n_seeds} seeds per (dataset, scenario, config) cell.")

    # Report on primary MIA metrics available in the CSV
    metric_cols = [c for c in df.columns if c.startswith("shokri_mia_")]
    print(f"\n  MIA metrics present in CSV: {metric_cols}")

    # Per-axis sensitivity tables
    for axis in ["partitioning", "num_clients", "forget_ratio"]:
        if df[axis].nunique() > 1:
            per_axis_sensitivity(df, axis, out_dir)

    # Correlations
    univariate_correlations(df, out_dir)
    multivariate_regression(df, out_dir)

    print(f"\n✅ Evaluation outputs → {out_dir}")


if __name__ == "__main__":
    main()
