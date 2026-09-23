"""Acceptance tests for the corrected MIA-unlearning results.

Checks the actual result files (summary_core.csv, per-node logs, manifests)
against the criteria agreed in the email thread, the PRD, and the pool design.
Run after any batch of cells; prints PASS/FAIL per criterion per cell plus a
summary. Reproduces the reported endpoints from the raw per-node log, so a FAIL
means the summary and the log disagree (a real problem), not just a threshold.

Usage:
    python test_acceptance.py --dir results
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ALPHA = 0.10
NODE_LOG_COLUMNS = [
    "dataset","scenario","seed","global_node_id","client_id","stage","subset",
    "pool_role","true_class","membership_label","mia_score","decision",
    "threshold","fallback_status","local_degree","local_label_agreement",
    "correct_prediction","prob_vector",
]
TOL = 1e-6          # numerical tolerance for recomputed endpoints
CAL_TOL = 0.10      # acceptable |C_cal| drift of the frozen threshold on target


class Results:
    def __init__(self, n): self.n=n; self.passed=0; self.failed=0; self.rows=[]
    def check(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        if ok: self.passed+=1
        else:  self.failed+=1
    def report(self):
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {name}" + (f"  — {detail}" if detail and not ok else
                                            (f"  ({detail})" if detail else "")))
        return self.failed == 0


def test_cell(nodelog_path, summary_row, manifest):
    """Integrity checks are PASS/FAIL. Scientific quantities are DIAGNOSTICS:
    reported for inspection, never judged (Point 6)."""
    df = pd.read_csv(nodelog_path)
    tag = Path(nodelog_path).stem.replace("nodelog_","")
    R = Results(tag)
    diag = []

    # ═══════════ INTEGRITY (PASS/FAIL) ═══════════

    R.check("schema: log has all 18 columns",
            list(df.columns) == NODE_LOG_COLUMNS, f"cols={list(df.columns)}")
    R.check("no NaN scores", not df.mia_score.isna().any(),
            f"{int(df.mia_score.isna().sum())} NaN")

    # Stage-aware labels (Point 1): F=1 under M0, F=0 under MR; H=0 both;
    # attack_member=1; attack_nonmember=0.
    def lab(stage, subset):
        v = df[(df.stage==stage)&(df.subset==subset)].membership_label
        return set(v.unique().tolist()) if len(v) else set()
    R.check("labels: F=1 under M0", lab("M0","F") == {1}, str(lab("M0","F")))
    R.check("labels: F=0 under MR", lab("MR","F") == {0}, str(lab("MR","F")))
    R.check("labels: H=0 under M0 and MR",
            lab("M0","H") == {0} and lab("MR","H") == {0},
            f"M0={lab('M0','H')} MR={lab('MR','H')}")
    R.check("labels: attack_member=1, attack_nonmember=0",
            lab("MR","attack_member") == {1} and lab("MR","attack_nonmember") == {0})

    # Frozen attack: thresholds identical M0 vs MR per class (R4)
    frozen = True
    for cls in df.true_class.unique():
        t0 = set(np.round(df[(df.stage=="M0")&(df.true_class==cls)].threshold, 6))
        t1 = set(np.round(df[(df.stage=="MR")&(df.true_class==cls)].threshold, 6))
        if t0 != t1: frozen = False
    R.check("attack frozen (M0 thresholds == MR)", frozen)

    # Paired F and H across stages
    for sub in ("F","H"):
        a = set(df[(df.stage=="M0")&(df.subset==sub)].global_node_id)
        b = set(df[(df.stage=="MR")&(df.subset==sub)].global_node_id)
        R.check(f"{sub} paired across M0 and MR", a == b and len(a) > 0,
                f"M0={len(a)} MR={len(b)}")

    # All four eval subsets present
    mr_counts = df[df.stage=="MR"].subset.value_counts().to_dict()
    R.check("all four eval subsets non-empty",
            all(mr_counts.get(k,0)>0 for k in ["F","H","attack_member","attack_nonmember"]),
            str(mr_counts))

    # Point 3: forgotten client's nodes must not appear in attack-strength pools
    if manifest is not None and "removed_client_ids" in manifest:
        forgot = set(manifest["removed_client_ids"])
        leak = df[df.subset.isin(["attack_member","attack_nonmember"]) & df.client_id.isin(forgot)]
        R.check("forgotten client absent from attack-strength pools",
                len(leak) == 0, f"{len(leak)} leaked rows")
        # and F/H come only from the forgotten client
        fh = df[df.subset.isin(["F","H"])]
        R.check("F and H come only from the forgotten client",
                set(fh.client_id.unique()) <= forgot, str(set(fh.client_id.unique())))

    # Raw/summary consistency (no-rerun principle)
    fpr_H = df[(df.stage=="MR")&(df.subset=="H")].decision.mean()
    fpr_F = df[(df.stage=="MR")&(df.subset=="F")].decision.mean()
    if summary_row is not None:
        R.check("C_F(log) == C_F(summary)",
                abs((fpr_F - fpr_H) - summary_row["C_F"]) < 1e-4,
                f"log={fpr_F-fpr_H:.4f} sum={summary_row['C_F']:.4f}")
        R.check("C_cal(log) == C_cal(summary)",
                abs((fpr_H - ALPHA) - summary_row["C_cal"]) < 1e-4,
                f"log={fpr_H-ALPHA:.4f} sum={summary_row['C_cal']:.4f}")

    # Disjoint pools from manifest (R2)
    if manifest is not None and "fit_global_ids" in manifest:
        fit_ids   = set(manifest.get("fit_global_ids", []))
        calib_ids = set(manifest.get("calib_global_ids", []))
        eval_ids  = set(manifest.get("eval_global_ids", []))
        R.check("eval ∩ fit == ∅", not (eval_ids & fit_ids), f"{len(eval_ids & fit_ids)} leaked")
        R.check("eval ∩ calibration == ∅", not (eval_ids & calib_ids), f"{len(eval_ids & calib_ids)} leaked")
        R.check("fit ∩ calibration == ∅", not (fit_ids & calib_ids), f"{len(fit_ids & calib_ids)} shared")
        R.check("fit & calibration non-empty", len(fit_ids)>0 and len(calib_ids)>0)

    # Provenance
    if manifest is not None:
        R.check("manifest commit_hash populated", manifest.get("commit_hash") not in (None,"null"))
        R.check("manifest split_ratios populated", manifest.get("split_ratios") not in (None,"null"))

    # ═══════════ DIAGNOSTICS (reported, not judged) ═══════════
    fr_m0 = df[(df.stage=="M0")&(df.subset=="F")].decision.mean()
    fr_mr = df[(df.stage=="MR")&(df.subset=="F")].decision.mean()
    diag.append(f"F member rate: M0={fr_m0:.3f} -> MR={fr_mr:.3f}  (change={fr_mr-fr_m0:+.3f})")
    diag.append(f"H member rate under MR = {fpr_H:.3f}   C_cal = {fpr_H-ALPHA:+.3f}   C_F = {fpr_F-fpr_H:+.3f}")
    mr = df[df.stage=="MR"].groupby("subset").mia_score.mean().to_dict()
    diag.append("MR mean score by subset: " + ", ".join(f"{k}={v:.3f}" for k,v in mr.items()))
    if summary_row is not None:
        diag.append(f"gap: acc_train={summary_row['acc_train']:.3f} acc_test={summary_row['acc_test']:.3f} "
                    f"gap={summary_row['train_test_gap']:.3f}")
        if "attack_strength_auc" in summary_row:
            diag.append(f"attack strength (retained pool, MR): AUC={summary_row['attack_strength_auc']:.3f} "
                        f"TPR@alpha={summary_row.get('attack_tpr_at_alpha', float('nan')):.3f} "
                        f"FPR@alpha={summary_row.get('attack_fpr_at_alpha', float('nan')):.3f}")
    nfb = int(df.fallback_status.astype(str).str.lower().eq("true").sum())
    diag.append(f"rows scored with pooled fallback attack: {nfb}")
    # structural / difficulty diagnostics for F vs H under MR
    mr_df = df[df.stage=="MR"]
    for sub in ["F","H"]:
        d = mr_df[mr_df.subset==sub]
        if len(d):
            diag.append(f"{sub}: local_degree(mean)={d.local_degree.mean():.2f}  "
                        f"homophily(mean)={d.local_label_agreement.mean():.3f}  "
                        f"correct(frac)={d.correct_prediction.mean():.3f}")

    return R, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()
    root = Path(args.dir)

    summary = None
    sp = root / "summary_core.csv"
    if sp.exists():
        summary = pd.read_csv(sp)

    logs = sorted(glob.glob(str(root / "raw" / "nodelog_*.csv")))
    if not logs:
        logs = sorted(glob.glob(str(root / "nodelog_*.csv")))
    if not logs:
        print("No node logs found under", root); return

    # E7: one summary row per cell
    if summary is not None:
        dup = summary.duplicated(subset=["dataset","scenario","seed"]).sum()
        print(f"E7 statistical unit (one row per dataset×seed): "
              f"{'PASS' if dup==0 else f'FAIL ({dup} duplicates)'}\n")

    all_pass = True
    for lg in logs:
        stem = Path(lg).stem.replace("nodelog_","")
        # match summary row + manifest
        srow = None
        if summary is not None:
            parts = stem.rsplit("_", 2)  # dataset_scenario_sSEED
            ds = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
            sc = parts[-2]; sd = int(parts[-1].lstrip("s"))
            m = summary[(summary.dataset==ds)&(summary.scenario==sc)&(summary.seed==sd)]
            srow = m.iloc[0] if len(m) else None
        mpath = Path(lg).parent / f"manifest_{stem}.json"
        manifest = json.load(open(mpath)) if mpath.exists() else None

        print(f"── {stem} ──")
        R, diag = test_cell(lg, srow, manifest)
        ok = R.report()
        print(f"    {R.passed} passed, {R.failed} failed")
        print("    diagnostics (not judged):")
        for d in diag:
            print(f"      · {d}")
        print()
        all_pass = all_pass and ok

    print("=" * 50)
    print("OVERALL:", "ALL CHECKS PASS ✅" if all_pass else "SOME CHECKS FAILED ❌")


if __name__ == "__main__":
    main()
