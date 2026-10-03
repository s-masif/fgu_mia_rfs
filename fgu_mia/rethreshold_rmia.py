"""
rethreshold_rmia.py  —  OFFLINE RMIA re-thresholding (item 4) + common-pool AUC
alignment (item 5), in one pass. No new scripts.

Item 4: RMIA threshold is computed on M0 only, FROZEN, and applied to both M0 and
        MR. Offline: rmia_score is already saved; we only re-derive rmia_threshold
        and rmia_decision. Scores, ratios, reference models untouched.

Item 5: all three attacks must report the COMMON-POOL AUC (evaluated on E+ ∪ E-),
        which is saved per cell in attackmeta['common_pool_auc']. AUC is
        threshold-free, so it is unchanged by re-thresholding — we carry it forward:
          - RMIA   AUC  <- common_pool_auc['rmia_MR']   (into the corrected rmia summary)
          - Shadow AUC  <- common_pool_auc['shokri_MR'] (rewritten into summary_core.csv)
          - Entropy AUC <- common_pool_auc['entropy_MR'](rewritten into summary_entropy.csv)
        so Table 3 and the AUC>=0.60 analyses use the common pool for all three.

Writes a corrected copy of the scores to <out-dir>/scores/ and:
  - <out-dir>/summary_rmia_rethresholded.csv   (RMIA, frozen tau, common-pool AUC)
  - <out-dir>/summary_core_commonpool.csv       (Shadow, common-pool AUC)        [if --core-csv]
  - <out-dir>/summary_entropy_commonpool.csv    (Entropy, common-pool AUC)       [if --entropy-csv]
The original <ext-dir> is never modified.

Usage:
  python rethreshold_rmia.py \
      --ext-dir extension_results_YYYYMMDD_HHMMSS \
      --out-dir extension_results_YYYYMMDD_HHMMSS_rmiafix \
      --core-csv summary_core.csv --entropy-csv summary_entropy.csv \
      --alpha 0.10
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd


def frozen_tau_from_M0(df, alpha):
    """(1-alpha) quantile of RMIA scores on M0 final-eval non-members."""
    nm = df[(df.stage == "M0") & (df.subset == "attack_nonmember")]["rmia_score"].dropna().values
    if len(nm) == 0:
        return np.inf
    return float(np.quantile(nm, 1.0 - alpha))


def rethreshold_one(ns_path, out_scores_dir, alpha):
    df = pd.read_csv(ns_path)
    tag = Path(ns_path).stem.replace("nodescores_", "")
    has_rmia = df["rmia_score"].notna()

    tau = frozen_tau_from_M0(df, alpha)          # the fix: one tau, from M0
    df.loc[has_rmia, "rmia_threshold"] = tau
    df.loc[has_rmia, "rmia_decision"] = (df.loc[has_rmia, "rmia_score"] > tau).astype(int)
    # non-RMIA rows (NaN) stay NaN

    df.to_csv(out_scores_dir / f"nodescores_{tag}.csv", index=False)
    return tag, tau, df


def recompute_rmia_rates(df, alpha):
    """RMIA endpoint rates per stage, from the re-thresholded decisions."""
    def rate(stage, subset):
        m = df[(df.stage == stage) & (df.subset == subset) & df.rmia_decision.notna()]
        return float(m.rmia_decision.mean()) if len(m) else float("nan")
    F_M0 = rate("M0", "F");  F_MR = rate("MR", "F")
    H_M0 = rate("M0", "H");  H_MR = rate("MR", "H")
    return dict(
        forget_member_rate_M0=F_M0, forget_member_rate_MR=F_MR,
        heldout_member_rate_M0=H_M0, heldout_member_rate_MR=H_MR,
        C_cal=H_MR - alpha,
        C_F=F_MR - H_MR,
        C_H=H_MR - H_M0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--core-csv", default=None,
                    help="existing summary_core.csv (Shadow) to rewrite with common-pool AUC")
    ap.add_argument("--entropy-csv", default=None,
                    help="existing summary_entropy.csv to rewrite with common-pool AUC")
    a = ap.parse_args()

    src_scores = Path(a.ext_dir) / "scores"
    out_dir = Path(a.out_dir)
    out_scores = out_dir / "scores"
    out_scores.mkdir(parents=True, exist_ok=True)

    ns_files = sorted(glob.glob(str(src_scores / "nodescores_*.csv")))
    if not ns_files:
        raise SystemExit(f"No nodescores_*.csv in {src_scores}")

    # common-pool AUCs harvested per cell, keyed by (dataset, seed)
    cp_auc = {}   # (dataset, seed) -> {"shokri":..., "entropy":..., "rmia":...}

    summary_rows = []
    print(f"re-thresholding {len(ns_files)} cells (frozen M0 tau, alpha={a.alpha})")
    for ns in ns_files:
        tag, tau, df = rethreshold_one(ns, out_scores, a.alpha)
        ds = df.dataset.iloc[0]; seed = int(df.seed.iloc[0])

        # read meta: grab common-pool AUCs (threshold-free) and update RMIA threshold
        meta_in = src_scores / f"attackmeta_{tag}.json"
        rmia_auc = None
        if meta_in.exists():
            meta = json.load(open(meta_in))
            cp = meta.get("common_pool_auc", {})
            cp_auc[(ds, seed)] = {
                "shokri":  cp.get("shokri_MR"),
                "entropy": cp.get("entropy_MR"),
                "rmia":    cp.get("rmia_MR"),
            }
            rmia_auc = cp.get("rmia_MR")
            for st in ("M0", "MR"):
                if "rmia_meta" in meta and st in meta["rmia_meta"]:
                    meta["rmia_meta"][st]["threshold"] = tau
                    meta["rmia_meta"][st]["threshold_source"] = "frozen_M0"
            json.dump(meta, open(out_scores / f"attackmeta_{tag}.json", "w"), indent=2)

        rates = recompute_rmia_rates(df, a.alpha)
        summary_rows.append(dict(dataset=ds, scenario="client", seed=seed,
                                 alpha=a.alpha, rmia_threshold=tau,
                                 attack_strength_auc=rmia_auc,   # common-pool, threshold-free
                                 **rates))

    # ---- corrected RMIA summary ----
    out = pd.DataFrame(summary_rows)
    out_csv = out_dir / "summary_rmia_rethresholded.csv"
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(out)} cells)")
    print(f"  C_F   mean={out.C_F.mean():+.4f}  range[{out.C_F.min():+.3f},{out.C_F.max():+.3f}]  "
          f"|C_F|<0.02 in {(out.C_F.abs()<0.02).sum()}/{len(out)}")
    print(f"  C_cal mean={out.C_cal.mean():+.4f}  range[{out.C_cal.min():+.3f},{out.C_cal.max():+.3f}]  "
          f"|C_cal|>0.05 in {(out.C_cal.abs()>0.05).sum()}/{len(out)}")
    print(f"  C_H   mean={out.C_H.mean():+.4f}  range[{out.C_H.min():+.3f},{out.C_H.max():+.3f}]  "
          f"|C_H|>0.05 in {(out.C_H.abs()>0.05).sum()}/{len(out)}")
    if out.attack_strength_auc.notna().any():
        print(f"  AUC   mean={out.attack_strength_auc.mean():.3f} (common pool)")

    # ---- item 5: rewrite Shadow / Entropy summaries with common-pool AUC ----
    def rewrite_auc(csv_path, attack_key, label):
        if not csv_path:
            return
        p = Path(csv_path)
        if not p.exists():
            print(f"  [skip] {label}: {csv_path} not found")
            return
        d = pd.read_csv(p)
        d["attack_strength_auc"] = [
            cp_auc.get((row.dataset, int(row.seed)), {}).get(attack_key)
            for row in d.itertuples(index=False)
        ]
        outp = out_dir / (p.stem + "_commonpool.csv")
        d.to_csv(outp, index=False)
        n_set = d["attack_strength_auc"].notna().sum()
        print(f"  wrote {outp}  ({n_set}/{len(d)} cells got common-pool {label} AUC, "
              f"mean={d['attack_strength_auc'].mean():.3f})")

    rewrite_auc(a.core_csv,    "shokri",  "Shadow")
    rewrite_auc(a.entropy_csv, "entropy", "Entropy")


if __name__ == "__main__":
    main()