"""Client-level core experiment (PRD Phase A).

End-to-end pipeline for one (dataset, scenario, seed) cell:

  1. Load graph, partition into clients, apply per-client train/val/test split.
  2. Train the ORIGINAL model M0 on all clients (val-only selection, R1).
  3. Define the forget set F and a matched held-out reference H.
  4. Train the RETRAINED reference MR on the retain set (R1).
  5. Train shadow models; build IDENTITY-DISJOINT attack pools (R2):
       - fit pool         : shadow member/non-member outputs
       - calibration pool : held-out shadow non-members (disjoint from fit)
       - evaluation pool  : target nodes (attack-strength set excluding F, plus F, plus H)
     F, H, and evaluation node identities are removed from fit/calibration.
  6. Fit the attack on the fit pool; calibrate the threshold at alpha=0.10 on the
     calibration pool; FREEZE (R3).
  7. Apply the frozen attack to M0 and MR with the same query context (R4),
     logging one row per queried node (Section 4).
  8. Record the transition endpoints: s(F;M0) vs s(F;MR), C_cal, C_F (Section 3).

The global member-vs-non-member AUC is recorded as attack strength only (C1).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))

from models import (PlainGCN, DATASET_CONFIGS, DATASET_SPLIT_RATIOS,
                    load_dataset, set_seed, get_metrics)  # noqa: E402
from partition import optimized_balanced_partitioning, apply_per_client_split       # noqa: E402
from train import (make_model, federated_train, train_client,
                   retrain_gold_client, define_forget_set_client)                    # noqa: E402
from attack import (ShokriShadowMIA, compute_detailed_mia_metrics,
                    assert_pools_disjoint, ALPHA)                                     # noqa: E402
from metrics import generalization_gap                                               # noqa: E402
from nodelog import NodeLogger, write_manifest, get_commit_hash                                       # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _labels_np(cl):
    y = cl["labels"]
    return y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)


@torch.no_grad()
def _forward_probs(model, cl, cid):
    model.eval()
    out = model(cl["features"], cl["adj"], cid)
    return torch.softmax(out, dim=1).cpu().numpy()


def _global_ids(cl, local_idx):
    """Map local node positions to ORIGINAL-graph global IDs via node_indices."""
    ni = np.asarray(cl["node_indices"])
    return ni[local_idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Held-out reference H (client-level): a whole retain client set aside as a
#  never-seen, matched non-member reference. It is excluded from MR training and
#  from the attack fit/calibration pools, and used only as a reference under MR.
# ─────────────────────────────────────────────────────────────────────────────
def define_heldout_client(n_clients, forget_clients, seed, size):
    """Pick `size` retain clients (disjoint from forget) as the held-out set H."""
    rng = np.random.default_rng(seed + 777)
    candidates = [c for c in range(n_clients) if c not in set(forget_clients)]
    size = min(size, len(candidates))
    return sorted(rng.choice(candidates, size=size, replace=False).tolist())


# ─────────────────────────────────────────────────────────────────────────────
#  Attack-pool construction (R2) — client scenario
#
#  Evaluation pool (target nodes, scored under M0 and MR):
#    - attack-strength members    : retain-client train nodes  (member=1), EXCLUDING F and H
#    - attack-strength non-members: retain-client test  nodes  (member=0)
#    - forget set F               : forget-client train nodes  (member=0 under MR)
#    - held-out reference H        : held-out-client train nodes (member=0)
#
#  Fit / calibration pools come from the SHADOW models. We additionally guarantee
#  that no evaluation global ID (F, H, or attack-strength nodes) appears in the
#  fit or calibration pools, by excluding those identities.
# ─────────────────────────────────────────────────────────────────────────────
def build_eval_pool_client(model, clients, forget_clients, heldout_clients,
                           eval_member_ids=None, eval_nonmember_ids=None):
    """Return dict with arrays for each evaluation subset, carrying global IDs.
    `model` is the target being queried (M0 or MR)."""
    forget_set  = set(forget_clients)
    heldout_set = set(heldout_clients)

    rows = {k: dict(probs=[], gid=[], cid=[], cls=[], member=[])
            for k in ("attack_member", "attack_nonmember", "F", "H")}

    def push(bucket, probs, gid, cid, cls, member):
        if len(probs) == 0:
            return
        rows[bucket]["probs"].append(probs)
        rows[bucket]["gid"].append(gid)
        rows[bucket]["cid"].append(np.full(len(probs), cid, dtype=int))
        rows[bucket]["cls"].append(cls)
        rows[bucket]["member"].append(np.full(len(probs), member, dtype=int))

    for cid, cl in enumerate(clients):
        labels = _labels_np(cl)
        probs  = _forward_probs(model, cl, cid)
        tr = np.where(np.asarray(cl["train_mask"]))[0]
        te = np.where(np.asarray(cl["test_mask"]))[0]
        g_tr, g_te = _global_ids(cl, tr), _global_ids(cl, te)

        if cid in forget_set:
            push("F", probs[tr], g_tr, cid, labels[tr], 0)
        elif cid in heldout_set:
            push("H", probs[tr], g_tr, cid, labels[tr], 0)
        else:
            # only the designated evaluation-member slice becomes attack_member;
            # the remaining retain-train nodes are reserved for shadow fitting.
            if eval_member_ids is not None:
                emask = np.array([gg in eval_member_ids for gg in g_tr])
                if emask.any():
                    push("attack_member", probs[tr][emask], g_tr[emask], cid,
                         labels[tr][emask], 1)
            else:
                push("attack_member", probs[tr], g_tr, cid, labels[tr], 1)
        # only the designated evaluation-nonmember slice becomes attack_nonmember;
        # the remaining test nodes are reserved for shadow fitting/calibration.
        if eval_nonmember_ids is not None:
            nmask = np.array([gg in eval_nonmember_ids for gg in g_te])
            if nmask.any():
                push("attack_nonmember", probs[te][nmask], g_te[nmask], cid,
                     labels[te][nmask], 0)
        else:
            push("attack_nonmember", probs[te], g_te, cid, labels[te], 0)

    out = {}
    for k, r in rows.items():
        if r["probs"]:
            out[k] = dict(
                probs=np.concatenate(r["probs"]),
                gid=np.concatenate(r["gid"]),
                cid=np.concatenate(r["cid"]),
                cls=np.concatenate(r["cls"]),
                member=np.concatenate(r["member"]),
            )
        else:
            out[k] = dict(probs=np.empty((0,)), gid=np.array([]), cid=np.array([]),
                          cls=np.array([]), member=np.array([]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Shadow fit + calibration pools (R2/R3), excluding evaluation identities.
# ─────────────────────────────────────────────────────────────────────────────
def build_shadow_pools(cfg, data, clients, seed, exclude_global_ids,
                       n_shadow=5, shadow_frac=0.5, calib_frac=0.3):
    """Train shadow models and return (fit, calib) pools of (probs, member, cls),
    with all rows whose global ID is in `exclude_global_ids` removed. The calib
    pool is a held-out slice (by identity) of the shadow NON-members.
    """
    exclude = set(int(g) for g in exclude_global_ids)
    rng = np.random.default_rng(seed + 12345)
    n_clients = len(clients)
    n_use = max(2, int(round(shadow_frac * n_clients)))

    P, M, C, G = [], [], [], []   # probs, member, class, global-id
    for i in range(n_shadow):
        ids = rng.choice(n_clients, n_use, replace=False).tolist()
        shadow_clients = [copy.deepcopy(clients[j]) for j in ids]
        model = make_model(cfg, data)
        model, _ = federated_train(cfg, shadow_clients, model, desc=f"Shadow {i+1}/{n_shadow}")
        for cl in shadow_clients:
            cid = int(cl["id"]) if "id" in cl else 0
            labels = _labels_np(cl)
            probs = _forward_probs(model, cl, cid)
            tr = np.where(np.asarray(cl["train_mask"]))[0]
            te = np.where(np.asarray(cl["test_mask"]))[0]
            for idx, mem in ((tr, 1), (te, 0)):
                g = _global_ids(cl, idx)
                keep = np.array([gg not in exclude for gg in g])
                if keep.sum() == 0:
                    continue
                P.append(probs[idx][keep]); M.append(np.full(keep.sum(), mem, dtype=int))
                C.append(labels[idx][keep]); G.append(g[keep])

    if not P:
        raise RuntimeError(
            "Shadow fit pool is empty after excluding evaluation identities. "
            "Reduce eval_member_frac or check the retain-client node counts.")
    P = np.concatenate(P); M = np.concatenate(M); C = np.concatenate(C); G = np.concatenate(G)

    # Calibration pool = held-out slice of NON-members, split by identity.
    nonmember_ids = np.unique(G[M == 0])
    rng.shuffle(nonmember_ids)
    n_cal = max(1, int(round(calib_frac * len(nonmember_ids)))) if len(nonmember_ids) else 0
    calib_ids = set(int(x) for x in nonmember_ids[:n_cal])

    is_calib = np.array([(m == 0 and int(g) in calib_ids) for m, g in zip(M, G)])
    fit = dict(probs=P[~is_calib], member=M[~is_calib], cls=C[~is_calib], gid=G[~is_calib])
    cal = dict(probs=P[is_calib],  member=M[is_calib],  cls=C[is_calib],  gid=G[is_calib])
    return fit, cal


# ─────────────────────────────────────────────────────────────────────────────
#  Main cell
# ─────────────────────────────────────────────────────────────────────────────
def run_cell(dataset, seed, scenario="client", out_dir="results", n_shadow=5):
    set_seed(seed)
    cfg = DATASET_CONFIGS[dataset]
    K = cfg["num_clients"]

    data = load_dataset(dataset, seed)
    clients = optimized_balanced_partitioning(data, K, seed)
    clients = apply_per_client_split(clients, dataset, seed=seed)
    for cid, cl in enumerate(clients):
        cl["id"] = cid

    # 2. Original model M0 (trained on ALL clients)
    m0 = make_model(cfg, data)
    m0, _ = federated_train(cfg, clients, m0, desc="M0")

    # 3. Forget set F (client-level) and matched held-out H
    forget_clients = define_forget_set_client(cfg, K, seed)
    n_forget = len(forget_clients)
    heldout_clients = define_heldout_client(K, forget_clients, seed, size=n_forget)

    # 4. Retrained reference MR (retain clients only; forget AND heldout excluded
    #    from training so H is a genuine never-seen reference)
    retain_clients = [c for i, c in enumerate(clients)
                      if i not in set(forget_clients) | set(heldout_clients)]
    set_seed(seed)
    mr = make_model(cfg, data)
    mr, _ = federated_train(cfg, retain_clients, mr, desc="MR")

    # 5. Designate a disjoint evaluation-member slice of the retain-client train
    #    nodes (a fraction per retain client). These become attack_member
    #    evaluation nodes; the remaining retain-train nodes are used for shadow
    #    fitting, so evaluation and fitting members never overlap (R2).
    eval_member_frac = 0.30
    rng_em = np.random.default_rng(seed + 999)
    forget_set  = set(forget_clients)
    heldout_set = set(heldout_clients)
    eval_nonmember_frac = 0.30
    eval_member_ids = set()
    eval_nonmember_ids = set()
    for cid, cl in enumerate(clients):
        # evaluation MEMBERS come from retain clients' train nodes
        if cid not in forget_set and cid not in heldout_set:
            tr = np.where(np.asarray(cl["train_mask"]))[0]
            g_tr = _global_ids(cl, tr)
            if len(g_tr) > 0:
                k = max(1, int(round(eval_member_frac * len(g_tr))))
                chosen = rng_em.choice(g_tr, size=min(k, len(g_tr)), replace=False)
                eval_member_ids.update(int(x) for x in chosen)
        # evaluation NON-MEMBERS come from every client's test nodes
        te = np.where(np.asarray(cl["test_mask"]))[0]
        g_te = _global_ids(cl, te)
        if len(g_te) > 0:
            k = max(1, int(round(eval_nonmember_frac * len(g_te))))
            chosen = rng_em.choice(g_te, size=min(k, len(g_te)), replace=False)
            eval_nonmember_ids.update(int(x) for x in chosen)

    # Evaluation pools under MR, using only the designated eval-member slice.
    eval_mr = build_eval_pool_client(mr, clients, forget_clients, heldout_clients,
                                     eval_member_ids=eval_member_ids,
                                     eval_nonmember_ids=eval_nonmember_ids)
    eval_ids = np.concatenate([eval_mr[k]["gid"] for k in eval_mr if len(eval_mr[k]["gid"])])

    # Shadow fit + calibration pools, excluding all evaluation identities
    # (F, H, and the designated evaluation members) so shadows never fit on them.
    fit, cal = build_shadow_pools(cfg, data, retain_clients, seed,
                                  exclude_global_ids=eval_ids, n_shadow=n_shadow)

    # Disjointness assertion (R2)
    assert_pools_disjoint(fit["gid"], cal["gid"], eval_ids)

    # 6. Fit attack on fit pool; calibrate threshold at alpha on calib pool; freeze
    shokri = ShokriShadowMIA(alpha=ALPHA)
    shokri.fit(fit["probs"], fit["member"], fit["cls"],
               calib_probs=cal["probs"], calib_member=cal["member"], calib_cls=cal["cls"])

    # 7. Apply frozen attack to M0 and MR, log per node
    logger = NodeLogger()
    eval_m0 = build_eval_pool_client(m0, clients, forget_clients, heldout_clients,
                                    eval_member_ids=eval_member_ids,
                                    eval_nonmember_ids=eval_nonmember_ids)

    def score_and_log(eval_pool, stage):
        for subset in ("attack_member", "attack_nonmember", "F", "H"):
            e = eval_pool[subset]
            if len(e["probs"]) == 0:
                continue
            pred, score, thr = shokri.predict(e["probs"], e["cls"])
            for k in range(len(e["probs"])):
                logger.add(dataset=dataset, scenario=scenario, seed=seed,
                           global_node_id=e["gid"][k], client_id=e["cid"][k],
                           stage=stage, subset=subset, pool_role="evaluation",
                           true_class=e["cls"][k], membership_label=e["member"][k],
                           mia_score=score[k], decision=pred[k],
                           threshold=thr[k] if hasattr(thr, "__len__") else thr,
                           fallback_status=shokri.fallback_used.get(int(e["cls"][k]), False),
                           prob_vector=e["probs"][k])

    score_and_log(eval_m0, "M0")
    score_and_log(eval_mr, "MR")

    # 8. Transition endpoints (Section 3), from the logged rows
    df = logger.to_frame()
    def rate(stage, subset):
        s = df[(df.stage == stage) & (df.subset == subset)]
        return float(s.decision.mean()) if len(s) else float("nan")
    def meanscore(stage, subset):
        s = df[(df.stage == stage) & (df.subset == subset)]
        return float(s.mia_score.mean()) if len(s) else float("nan")

    fpr_H_MR = rate("MR", "H")           # FPR on known non-members under MR
    fpr_F_MR = rate("MR", "F")
    C_cal = fpr_H_MR - ALPHA
    C_F   = fpr_F_MR - fpr_H_MR

    # Corrected generalization gap on MR (R5)
    acc_tr, acc_te, gap = generalization_gap(mr, retain_clients, "client", None)

    # attack strength (AUC) on the clean member/non-member evaluation set (C1)
    clean = df[df.subset.isin(["attack_member", "attack_nonmember"]) & (df.stage == "MR")]
    from sklearn.metrics import roc_auc_score
    try:
        auc_strength = float(roc_auc_score(clean.membership_label, clean.mia_score))
    except ValueError:
        auc_strength = float("nan")

    summary = dict(
        dataset=dataset, scenario=scenario, seed=seed,
        forget_member_rate_M0=rate("M0", "F"),
        forget_member_rate_MR=fpr_F_MR,
        forget_mean_score_M0=meanscore("M0", "F"),
        forget_mean_score_MR=meanscore("MR", "F"),
        heldout_member_rate_MR=fpr_H_MR,
        C_cal=C_cal, C_F=C_F,
        attack_strength_auc=auc_strength,
        acc_train=acc_tr, acc_test=acc_te, train_test_gap=gap,
        alpha=ALPHA,
    )

    # Write per-node log + manifest
    out = Path(out_dir) / "raw"
    tag = f"{dataset}_{scenario}_s{seed}"
    logger.save(out / f"nodelog_{tag}.csv")
    write_manifest(out / f"manifest_{tag}.json",
                   dataset=dataset, scenario=scenario, seed=seed, config=cfg,
                   removed_client_ids=list(forget_clients) + list(heldout_clients),
                   forget_global_ids=eval_mr["F"]["gid"].tolist(),
                   heldout_global_ids=eval_mr["H"]["gid"].tolist(),
                   alpha=ALPHA, thresholds=shokri.class_thresholds,
                   commit_hash=get_commit_hash(),
                   split_ratios=DATASET_SPLIT_RATIOS.get(dataset, (0.20, 0.40, 0.40)),
                   counts=dict(n_forget_clients=len(forget_clients),
                               n_heldout_clients=len(heldout_clients)),
                   fit_global_ids=fit["gid"].tolist(),
                   calib_global_ids=cal["gid"].tolist(),
                   eval_global_ids=[int(x) for x in eval_ids])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--scenario", default="client")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-shadow", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run cells already present in summary_core.csv instead of skipping them.")
    args = ap.parse_args()

    import pandas as pd
    summary_path = Path(args.out_dir) / "summary_core.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Load any existing summary so runs accumulate instead of overwriting.
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        rows = existing.to_dict("records")
        done = {(r["dataset"], r["scenario"], int(r["seed"])) for _, r in existing.iterrows()}
    else:
        rows, done = [], set()

    key = lambda ds, sc, sd: (ds, sc, int(sd))
    for ds in args.datasets:
        for sd in args.seeds:
            if not args.overwrite and key(ds, args.scenario, sd) in done:
                print(f"\n=== {ds} | {args.scenario} | seed={sd} — already done, skipping "
                      f"(use --overwrite to force) ===")
                continue
            print(f"\n=== {ds} | {args.scenario} | seed={sd} ===")
            row = run_cell(ds, sd, args.scenario, args.out_dir, args.n_shadow)
            # drop any stale row for this exact cell, then append the fresh one
            rows = [r for r in rows
                    if key(r["dataset"], r["scenario"], r["seed"]) != key(ds, args.scenario, sd)]
            rows.append(row)
            done.add(key(ds, args.scenario, sd))
            pd.DataFrame(rows).to_csv(summary_path, index=False)
    print("\n✅ done ->", summary_path, f"({len(rows)} cells total)")


if __name__ == "__main__":
    main()