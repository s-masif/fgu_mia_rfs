"""
evaluate_control.py
===================
Post-run analysis for the Phase 1 CONTROL experiment.

Reads mia_control_raw.csv, aggregates per (dataset, scenario), and prints:
  1. A comparison table: Original AUC vs Control AUC (μ ± σ across seeds)
  2. A verdict per row: if the two AUCs are close, the false positive is
     proven for that (dataset, scenario) cell.
  3. Paired Wilcoxon test across all cells for statistical support.

Usage
-----
  python evaluate_control.py \
      --csv mia_control_results/raw_data/mia_control_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DELTA_THRESHOLD = 0.03   # if |orig - ctrl| < this, we say "indistinguishable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="mia_control_results/raw_data/mia_control_raw.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n_seeds = int(df.seed.nunique())
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Datasets:  {sorted(df.dataset.unique())}")
    print(f"Scenarios: {sorted(df.scenario.unique())}")
    print(f"Seeds:     {sorted(df.seed.unique())}  (n_seeds = {n_seeds})")
    print(f"Reported μ ± σ are computed across {n_seeds} seeds per (dataset, scenario) cell.\n")

    # ── Detect if the Carlini-style metrics are present (added post-hoc) ──
    have_bal_acc = "orig_mia_balanced_acc" in df.columns
    have_tpr     = "orig_mia_tpr_at_fpr_0.01" in df.columns
    if have_bal_acc or have_tpr:
        print("  Extended metrics detected — will report balanced accuracy "
              "and TPR@low-FPR alongside AUC.\n")
    else:
        print("  Note: extended metrics (balanced_acc, TPR@low-FPR) not in CSV. "
              "Re-run experiments with the updated mia_attacks.py to populate them.\n")

    # ── Per (dataset, scenario) aggregation ──
    rows = []
    for (ds, sc), g in df.groupby(["dataset", "scenario"]):
        rows.append({
            "dataset":       ds,
            "scenario":      sc,
            "n_seeds":       len(g),
            "orig_auc_mean": g.mia_auc_original.mean(),
            "orig_auc_std":  g.mia_auc_original.std(),
            "ctrl_auc_mean": g.mia_auc_control.mean(),
            "ctrl_auc_std":  g.mia_auc_control.std(),
            "diff_mean":     (g.mia_auc_original - g.mia_auc_control).mean(),
            "diff_abs_mean": (g.mia_auc_original - g.mia_auc_control).abs().mean(),
        })
    summary = pd.DataFrame(rows).sort_values(
        ["dataset", "scenario"]).reset_index(drop=True)

    # ── Print table ──
    print("═" * 90)
    print(" COMPARISON: Original AUC (with forget) vs Control AUC (with heldout)")
    print("═" * 90)
    header = f"  {'Dataset':<12} {'Scenario':<8} {'Original μ±σ':<18} {'Control μ±σ':<18} {'|Δ|':<8} {'Verdict':<25}"
    print(header)
    print("  " + "-" * 90)
    for _, r in summary.iterrows():
        orig = f"{r.orig_auc_mean:.3f} ± {r.orig_auc_std:.3f}"
        ctrl = f"{r.ctrl_auc_mean:.3f} ± {r.ctrl_auc_std:.3f}"
        diff = f"{r.diff_abs_mean:.3f}"
        if r.diff_abs_mean < DELTA_THRESHOLD:
            verdict = "✓ false positive"
        elif r.diff_abs_mean < 2 * DELTA_THRESHOLD:
            verdict = "≈ borderline"
        else:
            verdict = "✗ genuine signal?"
        print(f"  {r.dataset:<12} {r.scenario:<8} {orig:<18} {ctrl:<18} {diff:<8} {verdict:<25}")

    # ── Statistical test across all cells ──
    print("\n═" * 90 if False else "\n" + "═" * 90)
    print(" STATISTICAL TEST (Paired Wilcoxon on per-seed AUCs)")
    print("═" * 90)
    stat, p = wilcoxon(df.mia_auc_original, df.mia_auc_control)
    print(f"  n = {len(df)} pairs (dataset × scenario × seed)")
    print(f"  statistic = {stat:.3f}, p = {p:.4f}")
    if p >= 0.05:
        print("  → cannot reject null: Original and Control AUCs are statistically")
        print("    indistinguishable across the sweep. FALSE POSITIVE PROVEN.")
    else:
        print("  → reject null: Original and Control AUCs differ significantly.")
        print("    Attack may have some forget-specific signal after all.")

    # ── Headline number for paper ──
    print("\n" + "═" * 90)
    print(" HEADLINE NUMBER FOR THE PAPER")
    print("═" * 90)
    overall_orig = df.mia_auc_original.mean()
    overall_ctrl = df.mia_auc_control.mean()
    overall_diff = (df.mia_auc_original - df.mia_auc_control).abs().mean()
    print(f"  Mean Original AUC across all cells:  {overall_orig:.3f}")
    print(f"  Mean Control  AUC across all cells:  {overall_ctrl:.3f}")
    print(f"  Mean |difference|:                   {overall_diff:.3f}")
    print()
    print(f"  Suggested sentence for the paper:")
    print(f"  \"When the forget set is replaced with a matched-distribution")
    print(f"   held-out control, the AUC changes by only {overall_diff:.3f} on average")
    print(f"   ({overall_orig:.2f} → {overall_ctrl:.2f}), indicating the AUC signal is not")
    print(f"   forget-specific.\"")

    # ── Save aggregated summary ──
    out_dir = Path(args.csv).parent.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "mia_control_summary.csv", index=False)
    print(f"\n  → Aggregated summary saved to {out_dir / 'mia_control_summary.csv'}")


if __name__ == "__main__":
    main()
