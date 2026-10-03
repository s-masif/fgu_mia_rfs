"""
aggregate_extension.py
Roll the 65 per-cell outputs of the entropy + RMIA extension into two summary CSVs
(summary_entropy.csv, summary_rmia.csv) with the SAME columns as summary_core.csv,
so the plotting pipeline (make_multiattack_figures.py) can consume all three attacks.

For each cell it reads:
  <ext_dir>/scores/nodescores_<tag>.csv   (per-node entropy_decision / rmia_decision)
  <ext_dir>/scores/attackmeta_<tag>.json  (common-pool AUC)

and computes, per attack, the same quantities as the Shokri summary:
  attack_strength_auc, C_cal, C_F,
  forget_member_rate_M0/MR, heldout_member_rate_M0/MR   (-> C_H derivable)

Rates are the fraction of a subset predicted as member (mean of the *_decision column)
at the fixed operating point, per (stage, subset). This matches how C_cal/C_F/C_H are
defined in the paper.

Usage:
  python aggregate_extension.py --ext-dir extension_results_YYYYMMDD_HHMMSS \
      --alpha 0.10 --out-dir .
Produces: summary_entropy.csv, summary_rmia.csv
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import pandas as pd

def rate(df, attack_decision_col, stage, subset):
    """Fraction predicted member for one (stage, subset)."""
    m = df[(df.stage == stage) & (df.subset == subset)]
    if len(m) == 0:
        return float("nan")
    return float(m[attack_decision_col].mean())

def summarize_attack(nodescores, meta, attack, alpha):
    """Return one summary row (dict) for a given attack ('entropy' or 'rmia')."""
    dec = f"{attack}_decision"
    # per-set rates
    F_M0  = rate(nodescores, dec, "M0", "F")
    F_MR  = rate(nodescores, dec, "MR", "F")
    H_M0  = rate(nodescores, dec, "M0", "H")
    H_MR  = rate(nodescores, dec, "MR", "H")
    # AUC from the common-pool auc in attackmeta
    auc_key = f"{attack}_MR"
    auc = meta.get("common_pool_auc", {}).get(auc_key, float("nan"))
    return dict(
        dataset=meta["dataset"],
        scenario="client",
        seed=meta["seed"],
        forget_member_rate_M0=F_M0,
        forget_member_rate_MR=F_MR,
        heldout_member_rate_M0=H_M0,
        heldout_member_rate_MR=H_MR,
        C_cal=H_MR - alpha,          # FPR(H;MR) - alpha
        C_F=F_MR - H_MR,             # FPR(F;MR) - FPR(H;MR)
        attack_strength_auc=auc,
        alpha=alpha,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-dir", required=True,
                    help="the extension results folder (contains scores/)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    scores_dir = Path(a.ext_dir) / "scores"
    meta_files = sorted(glob.glob(str(scores_dir / "attackmeta_*.json")))
    if not meta_files:
        raise SystemExit(f"No attackmeta_*.json found in {scores_dir}")

    rows_ent, rows_rmi = [], []
    n_ok = 0
    for mf in meta_files:
        meta = json.load(open(mf))
        tag = meta["tag"]
        nf = scores_dir / f"nodescores_{tag}.csv"
        if not nf.exists():
            print(f"  skip {tag}: no nodescores file")
            continue
        ns = pd.read_csv(nf)
        rows_ent.append(summarize_attack(ns, meta, "entropy", a.alpha))
        rows_rmi.append(summarize_attack(ns, meta, "rmia",    a.alpha))
        n_ok += 1

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ent = pd.DataFrame(rows_ent); rmi = pd.DataFrame(rows_rmi)
    ent.to_csv(out / "summary_entropy.csv", index=False)
    rmi.to_csv(out / "summary_rmia.csv", index=False)
    print(f"aggregated {n_ok} cells")
    print(f"  wrote {out/'summary_entropy.csv'}  ({len(ent)} rows)")
    print(f"  wrote {out/'summary_rmia.csv'}     ({len(rmi)} rows)")
    # quick sanity summary
    for name, d in [("entropy", ent), ("rmia", rmi)]:
        if len(d):
            print(f"  {name}: AUC mean={d.attack_strength_auc.mean():.3f}, "
                  f"C_cal range=[{d.C_cal.min():+.3f},{d.C_cal.max():+.3f}], "
                  f"C_F mean={d.C_F.mean():+.4f}")

if __name__ == "__main__":
    main()