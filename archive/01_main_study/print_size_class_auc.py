from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Total node counts (n_nodes is not in the raw CSV). Filtered counts for the
# two Wikipedia-network datasets.
DATASET_NODES = {
    "Cora": 2708, "PubMed": 19717, "CS": 18333, "Photo": 7487,
    "Computers": 13381, "WikiCS": 11701,
    "Chameleon": 890, "chameleon_filtered": 890,
    "Squirrel": 2223, "squirrel_filtered": 2223,
    "Actor": 7600,
    "tolokers": 11758, "minesweeper": 10000,
    "amazon_ratings": 24492, "roman_empire": 22662,
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # classes come from the CSV (n_classes column)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, sc), g in df.groupby(["dataset", "scenario"]):
        auc = g["shokri_mia_auc"]
        bal = g["shokri_mia_balanced_acc"] if "shokri_mia_balanced_acc" in g else pd.Series([np.nan])
        rows.append({
            "dataset":  ds,
            "scenario": sc,
            "nodes":    DATASET_NODES.get(ds, np.nan),
            "classes":  int(g["n_classes"].iloc[0]),
            "auc_mean": float(auc.mean()),
            "auc_std":  float(auc.std()) if len(auc) > 1 else 0.0,
            "bal_mean": float(bal.mean()),
            "bal_std":  float(bal.std()) if len(bal) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def print_scenario_table(summary: pd.DataFrame, scenario: str) -> None:
    sub = summary[summary.scenario == scenario].copy()
    sub = sub.sort_values("auc_mean", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 66)
    print(f"  SCENARIO: {scenario}   (sorted by MIA AUC, descending)")
    print("=" * 66)
    print(f"  {'Dataset':18s} {'Nodes':>7s} {'Classes':>8s} "
          f"{'MIA BalAcc':>16s} {'MIA AUC':>16s}")
    print("  " + "-" * 78)
    for _, r in sub.iterrows():
        nodes = f"{int(r.nodes):,}" if not np.isnan(r.nodes) else "?"
        print(f"  {r.dataset:18s} {nodes:>7s} {int(r.classes):>8d} "
              f"{r.bal_mean:.3f} +- {r.bal_std:.3f}   "
              f"{r.auc_mean:.3f} +- {r.auc_std:.3f}")

    # Correlations within this scenario
    x_nodes = sub["nodes"].values.astype(float)
    x_cls   = sub["classes"].values.astype(float)
    x_npc   = x_nodes / x_cls
    y       = sub["auc_mean"].values
    keep    = ~np.isnan(x_nodes)
    print("\n  Spearman correlations with MIA AUC (this scenario):")
    for name, x in [("nodes", x_nodes), ("classes", x_cls), ("nodes/class", x_npc)]:
        rho, p = spearmanr(x[keep], y[keep])
        print(f"    {name:12s}  rho = {rho:+.3f}   p = {p:.3f}")


def emit_latex_rows(summary: pd.DataFrame, scenario: str) -> None:
    """Print ready-to-paste LaTeX rows for a slide table."""
    sub = summary[summary.scenario == scenario].copy()
    sub = sub.sort_values("auc_mean", ascending=False).reset_index(drop=True)
    print(f"\n  % --- LaTeX rows for scenario = {scenario} ---")
    for _, r in sub.iterrows():
        ds = r.dataset.replace("_", r"\_")
        nodes = f"{int(r.nodes):,}".replace(",", "{,}") if not np.isnan(r.nodes) else "?"
        print(f"  {ds} & {nodes} & {int(r.classes)} & "
              f"${r.bal_mean:.3f} \\pm {r.bal_std:.3f}$ & "
              f"${r.auc_mean:.3f} \\pm {r.auc_std:.3f}$ \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mia_consistency_results/raw_data/mia_consistency_raw.csv")
    ap.add_argument("--latex", action="store_true",
                    help="also print ready-to-paste LaTeX rows")
    args = ap.parse_args()

    df = load(Path(args.csv))
    summary = build_summary(df)

    for scenario in ["client", "node"]:
        print_scenario_table(summary, scenario)
    if args.latex:
        for scenario in ["client", "node"]:
            emit_latex_rows(summary, scenario)


if __name__ == "__main__":
    main()
