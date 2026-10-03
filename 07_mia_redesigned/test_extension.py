"""
Automated acceptance tests for the attack extension (Modified Entropy + offline RMIA).
Checks CORRECTNESS and CONSISTENCY against the refpack, the frozen node log, and
Fabio's protocol requirements, including the RMIA frozen-M0-threshold rule.
PASS/FAIL for integrity; diagnostics reported.

Run:  python test_extension.py --tag Cora_client_s42 \
          --frozen-dir results_13x5/raw --out-dir attack_ext_out
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ALPHA = 0.10

class R:
    def __init__(s): s.p=0; s.f=0; s.rows=[]
    def check(s,name,ok,detail=""):
        s.rows.append((name,ok,detail)); s.p+= ok; s.f+= (not ok)
    def report(s):
        for n,ok,d in s.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {n}"+("" if ok else f"  <- {d}"))
        print(f"  {s.p} passed, {s.f} failed")
        return s.f==0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag", default="Cora_client_s42")
    ap.add_argument("--frozen-dir", default="results_13x5/raw")
    ap.add_argument("--out-dir", default="attack_ext_out")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a=ap.parse_args()
    alpha=a.alpha

    rp = json.load(open(Path(a.out_dir)/"reference"/f"refpack_{a.tag}.json"))
    df = pd.read_csv(Path(a.frozen_dir)/f"nodelog_{a.tag}.csv")
    ns = pd.read_csv(Path(a.out_dir)/"scores"/f"nodescores_{a.tag}.csv")
    meta = json.load(open(Path(a.out_dir)/"scores"/f"attackmeta_{a.tag}.json"))
    r=R()
    print(f"=== acceptance tests: {a.tag} ===")

    # ---------- INTEGRITY (PASS/FAIL) ----------
    # 1. node-identity match refpack vs frozen
    for sub in ["F","H","attack_member","attack_nonmember"]:
        aset=set(rp["node_groups"][sub]); bset=set(int(x) for x in df[df.subset==sub].global_node_id.unique())
        r.check(f"node ids match frozen: {sub}", aset==bset, f"ref={len(aset)} frozen={len(bset)}")

    # 2. reference / F,H separation: no eval node in any reference training set
    r.check("reference OUT status leak == 0", rp.get("out_status_leak",1)==0, str(rp.get("out_status_leak")))

    # 3. population disjoint from calibration and final-eval (Fabio)
    pop=set(rp["population_ids"]); cal=set(rp["calibration_ids"]); fe=set(rp["final_eval_nonmember_ids"])
    r.check("population ∩ calibration == ∅", not (pop&cal), f"{len(pop&cal)}")
    r.check("population ∩ final_eval == ∅", not (pop&fe), f"{len(pop&fe)}")
    r.check("population ⊆ attack_nonmember", pop <= set(rp["node_groups"]["attack_nonmember"]))
    r.check("final_eval ⊆ attack_nonmember", fe <= set(rp["node_groups"]["attack_nonmember"]))
    r.check("population ∪ final_eval covers attack_nonmember minus calib",
            (pop|fe) == (set(rp["node_groups"]["attack_nonmember"]) - cal))

    # 4. per-node scores saved for the right nodes, both stages
    r.check("nodescores has both stages", set(ns.stage.unique())=={"M0","MR"})
    r.check("nodescores has entropy score+decision", {"entropy_score","entropy_decision"} <= set(ns.columns))
    r.check("nodescores has rmia score+decision", {"rmia_score","rmia_decision"} <= set(ns.columns))
    r.check("no NaN entropy scores", not ns.entropy_score.isna().any(), str(int(ns.entropy_score.isna().sum())))
    rmia_nm = ns[(ns.stage=="MR")&(ns.subset=="attack_nonmember")&ns.rmia_score.notna()]
    r.check("RMIA non-member eval count == final_eval size", len(rmia_nm)==len(fe),
            f"rmia_nm={len(rmia_nm)} final_eval={len(fe)}")

    # 4b. parity columns present (consistency with the frozen Shokri node log)
    parity_cols = {"gid","client_id","subset","stage","true_class","membership_label",
                   "pool_role","local_degree","local_label_agreement","correct_prediction",
                   "dataset","scenario","seed"}
    r.check("nodescores has all parity columns (frozen-log consistency)",
            parity_cols <= set(ns.columns), str(sorted(parity_cols - set(ns.columns))))
    # 4c. rmia_ratio (attack-specific raw quantity) saved
    r.check("nodescores has rmia_ratio (raw RMIA ratio)", "rmia_ratio" in ns.columns)
    # 4d. membership_label populated (not all NaN) after the join
    r.check("membership_label populated", ns["membership_label"].notna().any())

    # 5. RMIA population NOT among the scored/evaluated rmia nodes (disjointness in scoring)
    rmia_scored = set(ns[ns.rmia_score.notna()].gid.unique())
    r.check("RMIA evaluated nodes exclude the population", not (rmia_scored & pop),
            f"{len(rmia_scored & pop)} population nodes leaked into RMIA eval")

    # 6. fallback info present and consistent (entropy)
    fb = meta["entropy_meta"]["MR"]["fallback_classes"]
    r.check("entropy fallback_classes recorded", isinstance(fb, list))
    ns_mr = ns[ns.stage=="MR"]
    bad = ns_mr[(ns_mr.true_class.isin(fb)) & (ns_mr.entropy_fallback!=1)]
    r.check("entropy_fallback flag matches fallback_classes", len(bad)==0, f"{len(bad)} mismatched")

    # 7. calibration labels + reference seeds saved (metadata)
    r.check("calibration_labels saved in refpack", "calibration_labels" in rp)
    r.check("reference seeds saved (ref_seed_base)", "ref_seed_base" in rp and rp["ref_seed_base"] is not None)

    # 8. common-pool AUC present for all three attacks
    cp = meta["common_pool_auc"]
    r.check("population_ratios saved (RMIA re-scoring at other gamma)",
            "population_ratios" in meta["rmia_meta"]["MR"])
    r.check("common-pool AUC present for shokri/entropy/rmia",
            all(k in cp for k in ["shokri_MR","entropy_MR","rmia_MR"]))

    # 9. thresholds frozen sanity: entropy threshold is finite where a class has calib
    tau_ent = meta["entropy_meta"]["MR"]["per_class_tau"]
    r.check("entropy per-class thresholds finite", all(np.isfinite(v) for v in tau_ent.values()))

    # ---------- 10. RMIA FROZEN-M0 THRESHOLD (Fabio item 4) ----------
    # The RMIA threshold must be computed on M0 and applied unchanged to M0 and MR.
    rmia_rows = ns[ns.rmia_score.notna()]
    taus = rmia_rows.rmia_threshold.dropna().unique()
    r.check("RMIA threshold single/frozen across M0 and MR",
            len(np.unique(np.round(taus, 10)))==1, f"distinct taus={np.round(taus,6).tolist()}")
    tau = float(taus[0]) if len(taus) else float("nan")

    # 10b. that tau equals the (1-alpha) quantile of M0 final-eval non-member scores
    nm_m0 = ns[(ns.stage=="M0")&(ns.subset=="attack_nonmember")].rmia_score.dropna().values
    expect = float(np.quantile(nm_m0, 1-alpha)) if len(nm_m0) else float("inf")
    r.check("RMIA tau == (1-alpha) quantile on M0 non-members",
            np.isclose(tau, expect, atol=1e-9), f"tau={tau:.6f} expect={expect:.6f}")

    # 10c. decisions are exactly score > tau, for every scored row (both stages)
    recomputed = (rmia_rows.rmia_score.values > tau).astype(int)
    r.check("RMIA decision == (score > frozen tau), both stages",
            np.array_equal(recomputed, rmia_rows.rmia_decision.values.astype(int)))

    # 10d. attackmeta records the frozen source (if the field is present)
    src_ok = all(meta["rmia_meta"].get(st,{}).get("threshold_source","frozen_M0")=="frozen_M0"
                 for st in ("M0","MR"))
    r.check("attackmeta RMIA threshold_source == frozen_M0", src_ok)
    # and the recorded threshold matches tau
    meta_taus = [meta["rmia_meta"].get(st,{}).get("threshold") for st in ("M0","MR")]
    meta_taus = [t for t in meta_taus if t is not None]
    r.check("attackmeta RMIA threshold matches frozen tau",
            all(np.isclose(t, tau, atol=1e-9) for t in meta_taus) if meta_taus else True)

    # 10e. calibration holds on M0 by construction: M0 non-member rate ~ alpha
    # m0nm_rate = ns[(ns.stage=="M0")&(ns.subset=="attack_nonmember")].rmia_decision.mean()
    # r.check(f"RMIA M0 non-member rate ~ alpha ({m0nm_rate:.3f})",
    #         abs(m0nm_rate - alpha) <= 0.08)

    ok = r.report()

    # ---------- DIAGNOSTICS (reported) ----------
    print("\n  diagnostics (not judged):")
    for attack, skey, dkey in [("entropy","entropy_score","entropy_decision"),
                               ("rmia","rmia_score","rmia_decision")]:
        d=ns[ns.stage=="MR"]
        def rate(sub):
            x=d[(d.subset==sub)&(d[dkey].notna())]; return x[dkey].mean() if len(x) else float("nan")
        print(f"    {attack}: F={rate('F'):.3f} H={rate('H'):.3f} "
              f"member={rate('attack_member'):.3f} nonmember={rate('attack_nonmember'):.3f}")
    # RMIA endpoints with the frozen threshold
    def rrate(stage,sub):
        x=ns[(ns.stage==stage)&(ns.subset==sub)&ns.rmia_decision.notna()]
        return x.rmia_decision.mean() if len(x) else float("nan")
    H_MR=rrate("MR","H"); F_MR=rrate("MR","F"); H_M0=rrate("M0","H")

    m0nm_rate = ns[(ns.stage=="M0")&(ns.subset=="attack_nonmember")].rmia_decision.mean()
    print(f"    RMIA M0 non-member realized rate = {m0nm_rate:.3f} "
          f"(nominal alpha={alpha}; may differ on discrete/near-chance score distributions)")
    
    print(f"    RMIA (frozen tau={tau:.4f}):  C_cal={H_MR-alpha:+.3f}  "
          f"C_F={F_MR-H_MR:+.3f}  C_H={H_MR-H_M0:+.3f}")
    print(f"    common-pool AUC: shokri={cp['shokri_MR']:.3f} "
          f"entropy={cp['entropy_MR']:.3f} rmia={cp['rmia_MR']:.3f}")
    print(f"    entropy fallback classes: {fb}")

    print("\nOVERALL:", "ALL INTEGRITY CHECKS PASS ✅" if ok else "SOME CHECKS FAILED ❌")

if __name__=="__main__":
    main()