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
    "threshold","fallback_status","prob_vector",
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
    df = pd.read_csv(nodelog_path)
    tag = Path(nodelog_path).stem.replace("nodelog_","")
    R = Results(tag)

    # ── Structural / schema ──────────────────────────────────────────────
    R.check("Sec4 log has all 15 columns",
            list(df.columns) == NODE_LOG_COLUMNS,
            f"cols={list(df.columns)}")

    R.check("S1 no NaN scores", not df.mia_score.isna().any(),
            f"{int(df.mia_score.isna().sum())} NaN")

    # ── E4/R4: frozen attack (thresholds identical M0 vs MR per class) ────
    frozen = True
    for cls in df.true_class.unique():
        t0 = set(np.round(df[(df.stage=="M0")&(df.true_class==cls)].threshold, 6))
        t1 = set(np.round(df[(df.stage=="MR")&(df.true_class==cls)].threshold, 6))
        if t0 != t1: frozen = False
    R.check("E4/R4 attack frozen (M0 thresholds == MR)", frozen)

    # ── D2: F pairing (same nodes scored under both stages) ──────────────
    f0 = set(df[(df.stage=="M0")&(df.subset=="F")].global_node_id)
    f1 = set(df[(df.stage=="MR")&(df.subset=="F")].global_node_id)
    R.check("D2 F-set paired across M0 and MR", f0 == f1 and len(f0) > 0,
            f"M0={len(f0)} MR={len(f1)}")

    # ── membership labels correct per subset ─────────────────────────────
    ml = df.groupby("subset").membership_label.mean().to_dict()
    labels_ok = (ml.get("attack_member")==1.0 and
                 all(ml.get(k,0.0)==0.0 for k in ["F","H","attack_nonmember"]))
    R.check("labels: member=1, F/H/nonmember=0", labels_ok, str(ml))

    # ── D1: evaluation subsets non-empty ─────────────────────────────────
    mr_counts = df[df.stage=="MR"].subset.value_counts().to_dict()
    R.check("D1 all four eval subsets non-empty",
            all(mr_counts.get(k,0)>0 for k in ["F","H","attack_member","attack_nonmember"]),
            str(mr_counts))

    # ── E2: transition direction (F member-rate drops M0 -> MR) ──────────
    fr_m0 = df[(df.stage=="M0")&(df.subset=="F")].decision.mean()
    fr_mr = df[(df.stage=="MR")&(df.subset=="F")].decision.mean()
    R.check("E2 transition: forget member-rate drops M0->MR",
            fr_mr <= fr_m0 + TOL, f"M0={fr_m0:.3f} MR={fr_mr:.3f}")

    # ── S2: attack works (member scores above forget under MR) ───────────
    mr = df[df.stage=="MR"].groupby("subset").mia_score.mean().to_dict()
    R.check("S2 MR ordering: attack_member > F",
            mr.get("attack_member",0) > mr.get("F",1),
            f"member={mr.get('attack_member',0):.3f} F={mr.get('F',1):.3f}")

    # ── S3/S4: endpoints recomputed from the log match the summary ───────
    fpr_H = df[(df.stage=="MR")&(df.subset=="H")].decision.mean()
    fpr_F = df[(df.stage=="MR")&(df.subset=="F")].decision.mean()
    C_F_log   = fpr_F - fpr_H
    C_cal_log = fpr_H - ALPHA
    if summary_row is not None:
        R.check("S3 C_F(log) == C_F(summary)",
                abs(C_F_log - summary_row["C_F"]) < 1e-4,
                f"log={C_F_log:.4f} sum={summary_row['C_F']:.4f}")
        R.check("S4 C_cal(log) == C_cal(summary)",
                abs(C_cal_log - summary_row["C_cal"]) < 1e-4,
                f"log={C_cal_log:.4f} sum={summary_row['C_cal']:.4f}")
        R.check("E2 forget_rate_MR(log)==summary",
                abs(fpr_F - summary_row["forget_member_rate_MR"]) < 1e-4,
                f"log={fpr_F:.4f} sum={summary_row['forget_member_rate_MR']:.4f}")

    # ── R3: calibration transports (|C_cal| within tolerance) ────────────
    R.check(f"R3 calibration drift |C_cal| <= {CAL_TOL}",
            abs(C_cal_log) <= CAL_TOL,
            f"C_cal={C_cal_log:+.3f}")

    # ── R5: corrected gap (train acc > test acc, gap plausible) ──────────
    if summary_row is not None:
        gap_ok = (summary_row["acc_train"] >= summary_row["acc_test"]
                  and 0 <= summary_row["train_test_gap"] < 1.0)
        R.check("R5 gap: acc_train>=acc_test, 0<=gap<1", gap_ok,
                f"train={summary_row['acc_train']:.3f} test={summary_row['acc_test']:.3f} "
                f"gap={summary_row['train_test_gap']:.3f}")

    # ── E5/R2: identity-disjoint pools (verified from manifest IDs) ──────
    if manifest is not None and "fit_global_ids" in manifest:
        fit_ids   = set(manifest.get("fit_global_ids", []))
        calib_ids = set(manifest.get("calib_global_ids", []))
        eval_ids  = set(manifest.get("eval_global_ids", []))
        leak_fit   = eval_ids & fit_ids
        leak_calib = eval_ids & calib_ids
        leak_fc    = fit_ids & calib_ids
        R.check("E5/R2 eval ∩ fit == ∅", len(leak_fit) == 0,
                f"{len(leak_fit)} eval IDs in fit pool")
        R.check("E5/R2 eval ∩ calibration == ∅", len(leak_calib) == 0,
                f"{len(leak_calib)} eval IDs in calibration pool")
        R.check("E5/R2 fit ∩ calibration == ∅", len(leak_fc) == 0,
                f"{len(leak_fc)} IDs shared by fit and calibration")
        R.check("E5/R2 fit & calibration pools non-empty",
                len(fit_ids) > 0 and len(calib_ids) > 0,
                f"fit={len(fit_ids)} calib={len(calib_ids)}")

    # ── Sec4 provenance: manifest has commit_hash + split_ratios ─────────
    if manifest is not None:
        R.check("Sec4 manifest commit_hash populated",
                manifest.get("commit_hash") not in (None, "null"),
                f"={manifest.get('commit_hash')}")
        R.check("Sec4 manifest split_ratios populated",
                manifest.get("split_ratios") not in (None, "null"),
                f"={manifest.get('split_ratios')}")

    return R


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
        R = test_cell(lg, srow, manifest)
        ok = R.report()
        print(f"    {R.passed} passed, {R.failed} failed\n")
        all_pass = all_pass and ok

    print("=" * 50)
    print("OVERALL:", "ALL CHECKS PASS ✅" if all_pass else "SOME CHECKS FAILED ❌")


if __name__ == "__main__":
    main()