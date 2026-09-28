"""
Score Modified-Entropy and offline-RMIA for one cell; save per-node scores and
decisions; and produce a fair cross-attack AUC comparison on a common final pool.

Inputs (no model training here — pure offline scoring):
  - frozen node log:  <frozen_dir>/nodelog_<tag>.csv   (target M0/MR softmax + labels
                      + Shokri per-node mia_score/decision)
  - frozen summary:   <frozen_dir>/../summary_core.csv  (original Shokri endpoints)
  - refpack:          <out_dir>/reference/refpack_<tag>.json  (reference models,
                      calibration ids+labels, population / final-eval split, seeds)

Outputs written under <out_dir>/scores/:
  - nodescores_<tag>.csv : per-node entropy & RMIA scores + decisions, both stages
  - attackmeta_<tag>.json: per-attack thresholds, fallback-class info, AUCs
                           (full-pool and common-final-pool), calibration labels
                           reference, reference seeds.

Cross-attack AUC: all three attacks evaluated on the SAME final pool
  = attack_member + final_eval_nonmember  (RMIA population nodes excluded).
The frozen Shokri full-pool AUC is preserved separately (from summary_core.csv).

Run:  python score_and_match.py --tag Cora_client_s42
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ALPHA     = 0.10
GAMMA     = 2.0
EPS       = 1e-12
MIN_CALIB = 10


# ─────────────────────────── primitives ─────────────────────────────────────
def mentr(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    py = p[y]
    return -(1 - py) * np.log(py) - np.sum(np.delete(p, y) * np.log(1 - np.delete(p, y)))

def q_below(neg, a):   # decision "member if score < tau" -> tau = a-quantile
    return float(np.quantile(neg, a)) if len(neg) else np.inf

def q_above(neg, a):   # decision "member if score > tau" -> tau = (1-a)-quantile
    return float(np.quantile(neg, 1 - a)) if len(neg) else np.inf

def auc_on(subsets, scores, member_ids, nonmember_ids, gids, lower_is_member=False):
    """AUC on a chosen member/non-member id set (for the common-pool comparison)."""
    from sklearn.metrics import roc_auc_score
    mset, nset = set(member_ids), set(nonmember_ids)
    keep = np.array([ (g in mset) or (g in nset) for g in gids ])
    if keep.sum() == 0:
        return float("nan")
    y = np.array([1 if g in mset else 0 for g in gids[keep]])
    sc = -scores[keep] if lower_is_member else scores[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, sc))
    except Exception:
        return float("nan")


def load(tag, frozen_dir, out_dir):
    df = pd.read_csv(Path(frozen_dir) / f"nodelog_{tag}.csv")
    df["prob_vector"] = df["prob_vector"].apply(json.loads)
    rp = json.load(open(Path(out_dir) / "reference" / f"refpack_{tag}.json"))
    summ = None
    sp = Path(frozen_dir).parent / "summary_core.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        row = s[(s.dataset == rp["dataset"]) & (s.seed == rp["seed"])]
        summ = row.iloc[0] if len(row) else None
    return df, rp, summ


def target_lookup(df, stage):
    d = df[df.stage == stage]
    return {int(r.global_node_id): (r.prob_vector, int(r.true_class)) for _, r in d.iterrows()}


# ─────────────────────────── Modified entropy ───────────────────────────────
def entropy_attack(df, rp, stage):
    """Returns per-node records + threshold/fallback metadata for one stage.
    Score = target-model modified entropy. Per-class threshold from the saved
    calibration outputs (reference-model softmax on calibration nodes), with the
    sparse-class pooled fallback (MIN_CALIB), consistent with the Shokri protocol."""
    tgt = target_lookup(df, stage)

    # calibration entropy per class from the saved reference outputs + calib labels
    cal_labels = {int(k): int(v) for k, v in rp.get("calibration_labels", {}).items()}
    cal_by_class = {}
    for gid, y in cal_labels.items():
        refs = [refd.get(str(gid)) for refd in rp["reference_probs"]]
        refs = [r for r in refs if r is not None]
        if not refs:
            continue
        cal_by_class.setdefault(y, []).append(mentr(np.mean(refs, axis=0), y))

    pooled_cal = np.array([v for vals in cal_by_class.values() for v in vals])
    pooled_tau = q_below(pooled_cal, ALPHA) if len(pooled_cal) else np.inf
    tau, fallback_classes = {}, []
    all_classes = set(cal_by_class) | {int(tgt[g][1]) for g in tgt}
    for c in sorted(all_classes):
        vals = np.array(cal_by_class.get(c, []))
        if len(vals) >= MIN_CALIB:
            tau[c] = q_below(vals, ALPHA)
        else:
            tau[c] = pooled_tau
            fallback_classes.append(int(c))

    recs = []
    for sub in ["F", "H", "attack_member", "attack_nonmember"]:
        for gid in rp["node_groups"][sub]:
            if gid not in tgt:
                continue
            p, y = tgt[gid]
            s = mentr(p, y)
            t = tau.get(y, pooled_tau)
            recs.append(dict(gid=int(gid), subset=sub, stage=stage, true_class=int(y),
                             entropy_score=float(s), entropy_threshold=float(t),
                             entropy_decision=int(s < t),
                             entropy_fallback=int(y in fallback_classes)))
    meta = dict(pooled_tau=float(pooled_tau),
                per_class_tau={int(k): float(v) for k, v in tau.items()},
                fallback_classes=fallback_classes,
                min_calib=MIN_CALIB)
    return recs, meta


# ─────────────────────────── Offline RMIA ───────────────────────────────────
def rmia_attack(df, rp, stage):
    """Returns per-node records + metadata. Score = target-vs-reference likelihood
    ratio compared against the population (gamma). Non-member evaluation uses the
    disjoint final_eval set (population excluded)."""
    tgt = target_lookup(df, stage)

    def ratio(gid):
        if gid not in tgt:
            return np.nan
        p, y = tgt[gid]
        num = np.clip(p[y], EPS, 1 - EPS)
        refs = [refd.get(str(gid)) for refd in rp["reference_probs"]]
        refs = [np.clip(r[y], EPS, 1 - EPS) for r in refs if r is not None]
        if not refs:
            return np.nan
        return num / float(np.mean(refs))

    z = np.array([ratio(g) for g in rp["population_ids"]]); z = z[~np.isnan(z)]

    def score(gid):
        rx = ratio(gid)
        if np.isnan(rx) or len(z) == 0:
            return np.nan
        return float(np.mean((rx / z) >= GAMMA))

    eval_groups = {
        "F": rp["node_groups"]["F"],
        "H": rp["node_groups"]["H"],
        "attack_member": rp["node_groups"]["attack_member"],
        "attack_nonmember": rp["final_eval_nonmember_ids"],   # disjoint from population
    }
    # threshold on the disjoint non-member eval set
    nm_scores = np.array([score(g) for g in eval_groups["attack_nonmember"]])
    nm_scores = nm_scores[~np.isnan(nm_scores)]
    tau = q_above(nm_scores, ALPHA)

    recs = []
    for sub, ids in eval_groups.items():
        for gid in ids:
            s = score(gid)
            if np.isnan(s):
                continue
            recs.append(dict(gid=int(gid), subset=sub, stage=stage,
                             true_class=int(tgt[gid][1]),
                             rmia_ratio=float(ratio(gid)),   # raw Pr(x|target)/Pr(x)
                             rmia_score=float(s), rmia_threshold=float(tau),
                             rmia_decision=int(s > tau)))
    meta = dict(threshold=float(tau), gamma=GAMMA,
                n_population=int(len(z)),
                n_final_eval_nonmember=int(len(eval_groups["attack_nonmember"])),
                population_ratios=[float(v) for v in z])   # for offline re-scoring at other gamma
    return recs, meta


# ─────────────────────────── endpoints from records ─────────────────────────
def rates_from_records(recs, score_key, decision_key):
    df = pd.DataFrame(recs)
    out = {}
    for sub in ["F", "H", "attack_member", "attack_nonmember"]:
        d = df[df.subset == sub]
        out[sub] = float(d[decision_key].mean()) if len(d) else float("nan")
    return out


# ─────────────────────────── main ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="Cora_client_s42")
    ap.add_argument("--frozen-dir", default="results_13x5/raw")
    ap.add_argument("--out-dir", default="attack_ext_out")
    a = ap.parse_args()
    df, rp, summ = load(a.tag, a.frozen_dir, a.out_dir)

    # ---- node-identity MATCH against frozen ----
    print(f"=== {a.tag} : MATCH against frozen outputs ===")
    all_ok = True
    for sub in ["F", "H", "attack_member", "attack_nonmember"]:
        aset = set(rp["node_groups"][sub])
        bset = set(int(x) for x in df[df.subset == sub].global_node_id.unique())
        ok = aset == bset; all_ok &= ok
        print(f"  {sub:16s} refpack={len(aset):3d} frozen={len(bset):3d}  match={ok}")
    print(f"  node-identity match: {'ALL MATCH ✓' if all_ok else 'MISMATCH ✗'}\n")

    # ---- score both attacks, both stages; collect per-node records ----
    ent_recs = {st: entropy_attack(df, rp, st) for st in ["M0", "MR"]}
    rmi_recs = {st: rmia_attack(df, rp, st)    for st in ["M0", "MR"]}

    # merge per-node into one long table
    all_node = []
    for st in ["M0", "MR"]:
        er, _ = ent_recs[st]; rr, _ = rmi_recs[st]
        rmi_map = {(d["gid"], d["subset"]): d for d in rr}
        for d in er:
            key = (d["gid"], d["subset"])
            row = dict(d)
            if key in rmi_map:
                row.update({k: rmi_map[key][k] for k in
                            ("rmia_ratio", "rmia_score", "rmia_threshold", "rmia_decision")})
            all_node.append(row)
    node_df = pd.DataFrame(all_node)

    # ---- PARITY: join the frozen log's identifying / ground-truth / diagnostic
    #      columns onto the per-node table, so nodescores is self-contained and
    #      structurally consistent with the frozen Shokri node log. ----
    frozen_cols = ["global_node_id", "stage", "client_id", "membership_label",
                   "pool_role", "local_degree", "local_label_agreement",
                   "correct_prediction"]
    fj = df[frozen_cols].rename(columns={"global_node_id": "gid"})
    node_df = node_df.merge(fj, on=["gid", "stage"], how="left")
    # constant per-cell descriptors, for a fully self-describing file
    node_df["dataset"] = rp["dataset"]
    node_df["scenario"] = "client"
    node_df["seed"] = rp["seed"]
    # order columns: identity -> ground truth -> diagnostics -> per-attack
    col_order = ["dataset", "scenario", "seed", "gid", "client_id", "stage", "subset",
                 "pool_role", "true_class", "membership_label",
                 "local_degree", "local_label_agreement", "correct_prediction",
                 "entropy_score", "entropy_threshold", "entropy_decision", "entropy_fallback",
                 "rmia_ratio", "rmia_score", "rmia_threshold", "rmia_decision"]
    node_df = node_df[[c for c in col_order if c in node_df.columns]]

    # ---- COMMON-POOL cross-attack AUC (attack_member + final_eval_nonmember) ----
    member_ids = list(rp["node_groups"]["attack_member"])
    finaleval_ids = list(rp["final_eval_nonmember_ids"])

    def common_auc_entropy(stage):
        er, _ = ent_recs[stage]
        d = pd.DataFrame(er)
        return auc_on(d.subset.values, d.entropy_score.values, member_ids, finaleval_ids,
                      d.gid.values, lower_is_member=True)
    def common_auc_rmia(stage):
        rr, _ = rmi_recs[stage]
        d = pd.DataFrame(rr)
        return auc_on(d.subset.values, d.rmia_score.values, member_ids, finaleval_ids,
                      d.gid.values, lower_is_member=False)
    def common_auc_shokri(stage):
        d = df[df.stage == stage]
        gids = d.global_node_id.values.astype(int)
        return auc_on(d.subset.values, d.mia_score.values, member_ids, finaleval_ids,
                      gids, lower_is_member=False)

    common = {
        "entropy_MR": common_auc_entropy("MR"),
        "rmia_MR":    common_auc_rmia("MR"),
        "shokri_MR":  common_auc_shokri("MR"),
    }

    # ---- endpoints for the console summary ----
    ent_mr = rates_from_records(ent_recs["MR"][0], "entropy_score", "entropy_decision")
    ent_m0 = rates_from_records(ent_recs["M0"][0], "entropy_score", "entropy_decision")
    rmi_mr = rates_from_records(rmi_recs["MR"][0], "rmia_score", "rmia_decision")
    rmi_m0 = rates_from_records(rmi_recs["M0"][0], "rmia_score", "rmia_decision")

    print("MODIFIED ENTROPY")
    print(f"  common-pool AUC(MR)={common['entropy_MR']:.3f}  "
          f"F:M0={ent_m0['F']:.3f}->MR={ent_mr['F']:.3f}  H:M0={ent_m0['H']:.3f}->MR={ent_mr['H']:.3f}")
    print(f"  C_cal={ent_mr['H']-ALPHA:+.3f}  C_F={ent_mr['F']-ent_mr['H']:+.3f}  "
          f"fallback_classes={ent_recs['MR'][1]['fallback_classes']}\n")

    print("OFFLINE RMIA")
    print(f"  common-pool AUC(MR)={common['rmia_MR']:.3f}  "
          f"F:M0={rmi_m0['F']:.3f}->MR={rmi_mr['F']:.3f}  H:M0={rmi_m0['H']:.3f}->MR={rmi_mr['H']:.3f}")
    print(f"  C_cal={rmi_mr['H']-ALPHA:+.3f}  C_F={rmi_mr['F']-rmi_mr['H']:+.3f}\n")

    print("CROSS-ATTACK AUC on the SAME final pool (attack_member + final_eval_nonmember):")
    print(f"  Shokri={common['shokri_MR']:.3f}  Entropy={common['entropy_MR']:.3f}  RMIA={common['rmia_MR']:.3f}")
    if summ is not None:
        print(f"\nFROZEN SHOKRI full-pool AUC (original core result, kept separately): "
              f"{summ['attack_strength_auc']:.3f}")

    # ---- SAVE per-node scores/decisions + attack metadata ----
    outdir = Path(a.out_dir) / "scores"; outdir.mkdir(parents=True, exist_ok=True)
    node_df.to_csv(outdir / f"nodescores_{a.tag}.csv", index=False)
    meta = dict(
        tag=a.tag, dataset=rp["dataset"], seed=rp["seed"], alpha=ALPHA, gamma=GAMMA,
        ref_seed_base=rp.get("ref_seed_base"),
        n_reference_models=rp.get("n_reference_models"),
        calibration_labels_ref="saved in refpack (calibration_labels)",
        entropy_meta={st: ent_recs[st][1] for st in ["M0", "MR"]},
        rmia_meta={st: rmi_recs[st][1] for st in ["M0", "MR"]},
        common_pool_auc=common,
        frozen_shokri_full_pool_auc=(float(summ["attack_strength_auc"]) if summ is not None else None),
    )
    json.dump(meta, open(outdir / f"attackmeta_{a.tag}.json", "w"), indent=2)
    print(f"\n[SAVED] {outdir/('nodescores_'+a.tag+'.csv')}")
    print(f"[SAVED] {outdir/('attackmeta_'+a.tag+'.json')}")


if __name__ == "__main__":
    main()