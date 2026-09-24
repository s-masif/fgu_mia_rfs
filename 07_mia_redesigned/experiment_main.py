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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (PlainGCN, DATASET_CONFIGS, DATASET_SPLIT_RATIOS,
                    CORE_DATASETS, CORE_SEEDS,
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
# Attack hyperparameters (frozen for the core experiment)
SHADOW_FRAC = 0.5
CALIB_FRAC  = 0.3
MIN_CALIB   = 10

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
#  matched supervised non-member, matched non-member reference. It is excluded from MR training and
#  from the attack fit/calibration pools, and used only as a reference under MR.
# ─────────────────────────────────────────────────────────────────────────────
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

def _structural_and_difficulty(cl, cid, probs):
    """Per local-node arrays for the client `cl`:
      local_degree           : degree within THIS client's subgraph (drives
                               message passing in the transductive setting).
      local_label_agreement  : fraction of client-local neighbours sharing the
                               node's label (neighbourhood homophily).
      correct                : whether argmax(probs) == true label under the
                               model whose softmax is `probs`.
    `probs` is (n_local, C) softmax from the model being scored.
    """
    adj = cl["adj"]
    A = adj.cpu().numpy() if hasattr(adj, "cpu") else np.asarray(adj)
    binary = (A > 0).astype(np.float32)
    np.fill_diagonal(binary, 0.0)                 # drop self-loops for degree
    deg = binary.sum(axis=1).astype(int)

    y = _labels_np(cl)
    same = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        nb = np.where(binary[i] > 0)[0]
        same[i] = float(np.mean(y[nb] == y[i])) if len(nb) else float("nan")

    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(int)
    return deg, same, correct


def build_eval_pool_client(model, clients, forget_clients, stage,
                           eval_member_ids=None, eval_nonmember_ids=None):
    """Evaluation pool for one stage ("M0" or "MR"), carrying global IDs.

    Roles (Points 1 and 3):
      F                : forgotten client's TRAIN nodes.
                         label 1 under M0 (they were trained on), 0 under MR.
      H                : forgotten client's TEST nodes (never trained on).
                         label 0 under both stages. Used only as the matched
                         matched supervised non-member reference.
      attack_member    : retained clients' train nodes in the designated
                         evaluation-member slice (label 1).
      attack_nonmember : retained clients' test nodes in the designated
                         evaluation-nonmember slice (label 0). The forgotten
                         client's test nodes are NOT included here; they are H.
    F and H are scored by running `model` over the forgotten client's own
    subgraph (its adjacency and features); the client is excluded from MR's
    training and selection but remains in the graph.
    """
    assert stage in ("M0", "MR")
    forget_set = set(forget_clients)
    f_label = 1 if stage == "M0" else 0

    rows = {k: dict(probs=[], gid=[], cid=[], cls=[], member=[],
                    ldeg=[], lla=[], corr=[])
            for k in ("attack_member", "attack_nonmember", "F", "H")}

    def push(bucket, probs, gid, cid, cls, member, ldeg, lla, corr):
        if len(probs) == 0:
            return
        rows[bucket]["probs"].append(probs)
        rows[bucket]["gid"].append(gid)
        rows[bucket]["cid"].append(np.full(len(probs), cid, dtype=int))
        rows[bucket]["cls"].append(cls)
        rows[bucket]["member"].append(np.full(len(probs), member, dtype=int))
        rows[bucket]["ldeg"].append(ldeg)
        rows[bucket]["lla"].append(lla)
        rows[bucket]["corr"].append(corr)

    for cid, cl in enumerate(clients):
        labels = _labels_np(cl)
        probs  = _forward_probs(model, cl, cid)
        ldeg, lla, corr = _structural_and_difficulty(cl, cid, probs)
        tr = np.where(np.asarray(cl["train_mask"]))[0]
        te = np.where(np.asarray(cl["test_mask"]))[0]
        g_tr, g_te = _global_ids(cl, tr), _global_ids(cl, te)

        if cid in forget_set:
            # Forgotten client: F = its train nodes, H = its test nodes.
            push("F", probs[tr], g_tr, cid, labels[tr], f_label, ldeg[tr], lla[tr], corr[tr])
            push("H", probs[te], g_te, cid, labels[te], 0, ldeg[te], lla[te], corr[te])
            continue   # forgotten client contributes nothing to attack-strength pools

        # Retained client: attack-strength members from the eval-member slice
        if eval_member_ids is not None:
            emask = np.array([gg in eval_member_ids for gg in g_tr])
            if emask.any():
                push("attack_member", probs[tr][emask], g_tr[emask], cid,
                     labels[tr][emask], 1, ldeg[tr][emask], lla[tr][emask], corr[tr][emask])
        else:
            push("attack_member", probs[tr], g_tr, cid, labels[tr], 1, ldeg[tr], lla[tr], corr[tr])

        # Retained client: attack-strength non-members from the eval-nonmember slice
        if eval_nonmember_ids is not None:
            nmask = np.array([gg in eval_nonmember_ids for gg in g_te])
            if nmask.any():
                push("attack_nonmember", probs[te][nmask], g_te[nmask], cid,
                     labels[te][nmask], 0, ldeg[te][nmask], lla[te][nmask], corr[te][nmask])
        else:
            push("attack_nonmember", probs[te], g_te, cid, labels[te], 0, ldeg[te], lla[te], corr[te])

    out = {}
    for k, r in rows.items():
        if r["probs"]:
            out[k] = dict(
                probs=np.concatenate(r["probs"]),
                gid=np.concatenate(r["gid"]),
                cid=np.concatenate(r["cid"]),
                cls=np.concatenate(r["cls"]),
                member=np.concatenate(r["member"]),
                ldeg=np.concatenate(r["ldeg"]),
                lla=np.concatenate(r["lla"]),
                corr=np.concatenate(r["corr"]),
            )
        else:
            out[k] = dict(probs=np.empty((0,)), gid=np.array([]), cid=np.array([]),
                          cls=np.array([]), member=np.array([]),
                          ldeg=np.array([]), lla=np.array([]), corr=np.array([]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Shadow fit + calibration pools (R2/R3), excluding evaluation identities.
# ─────────────────────────────────────────────────────────────────────────────
def build_shadow_pools(cfg, data, clients, seed, exclude_global_ids,
                       n_shadow=5, shadow_frac=SHADOW_FRAC, calib_frac=CALIB_FRAC):
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

        # Point 4: identity separation BEFORE shadow training. Remove every
        # evaluation identity (eval members, F, H) from the shadow clients'
        # supervised train_mask so the shadow models never learn from them.
        # The nodes remain structurally present in the graph.
        for cl in shadow_clients:
            tm = np.asarray(cl["train_mask"]).copy()
            g_all = np.asarray(cl["node_indices"])
            drop = np.array([int(g) in exclude for g in g_all])
            cl["train_mask"] = tm & ~drop

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

    # ── 1. Original model M0 on ALL clients (val-only selection, R1) ──────
    m0 = make_model(cfg, data)
    m0, _ = federated_train(cfg, clients, m0, desc="M0")

    # ── 2. Forget set = whole client(s). H = the SAME client's test nodes ──
    # Louvain may not produce exactly K clients, so use the ACTUAL count.
    n_clients_actual = len(clients)
    forget_clients = define_forget_set_client(cfg, n_clients_actual, seed)
    assert all(0 <= c < n_clients_actual for c in forget_clients), \
        f"forget client index out of range: {forget_clients} vs {n_clients_actual} clients"
    forget_set = set(forget_clients)

    # ── 3. Retrained reference MR on RETAINED clients only (Point 1) ─────
    #   The forgotten client takes no part in MR training or in MR checkpoint
    #   selection: it is simply absent from the client list passed in, so the
    #   selection metric inside federated_train is computed on retained
    #   clients only.
    retain_clients = [c for i, c in enumerate(clients) if i not in forget_set]
    assert all(int(c["id"]) not in forget_set for c in retain_clients), \
        "forgotten client leaked into MR training/selection set"
    set_seed(seed)
    mr = make_model(cfg, data)
    mr, _ = federated_train(cfg, retain_clients, mr, desc="MR")

    # ── 4. Designate evaluation slices on RETAINED clients (Point 3) ─────
    eval_member_frac = 0.30
    eval_nonmember_frac = 0.30
    rng_em = np.random.default_rng(seed + 999)
    eval_member_ids, eval_nonmember_ids = set(), set()
    for cid, cl in enumerate(clients):
        if cid in forget_set:
            continue
        tr = np.where(np.asarray(cl["train_mask"]))[0]
        g_tr = _global_ids(cl, tr)
        if len(g_tr) > 0:
            k = max(1, int(round(eval_member_frac * len(g_tr))))
            eval_member_ids.update(int(x) for x in
                                   rng_em.choice(g_tr, size=min(k, len(g_tr)), replace=False))
        te = np.where(np.asarray(cl["test_mask"]))[0]
        g_te = _global_ids(cl, te)
        if len(g_te) > 0:
            k = max(1, int(round(eval_nonmember_frac * len(g_te))))
            eval_nonmember_ids.update(int(x) for x in
                                      rng_em.choice(g_te, size=min(k, len(g_te)), replace=False))

    # F and H global IDs (forgotten client's train / test nodes)
    F_ids, H_ids = set(), set()
    for cid in forget_set:
        cl = clients[cid]
        F_ids.update(int(x) for x in _global_ids(cl, np.where(np.asarray(cl["train_mask"]))[0]))
        H_ids.update(int(x) for x in _global_ids(cl, np.where(np.asarray(cl["test_mask"]))[0]))

    # ── 5. Shadow fit/calib pools; ALL evaluation identities removed from the
    #      shadow train masks BEFORE training (Point 4) ─────────────────────
    eval_ids_all = eval_member_ids | eval_nonmember_ids | F_ids | H_ids
    fit, cal = build_shadow_pools(cfg, data, retain_clients, seed,
                                  exclude_global_ids=eval_ids_all, n_shadow=n_shadow)
    assert_pools_disjoint(fit["gid"], cal["gid"], list(eval_ids_all))

    # ── 6. Fit + calibrate ONCE, then freeze (R3/R4, Point 5) ───────────
    shokri = ShokriShadowMIA(alpha=ALPHA)
    shokri.fit(fit["probs"], fit["member"], fit["cls"],
               calib_probs=cal["probs"], calib_member=cal["member"],
               calib_cls=cal["cls"], calib_gid=cal["gid"], min_calib=MIN_CALIB)

    # ── 7. Evaluation pools per stage (stage-aware labels) and scoring ──
    eval_m0 = build_eval_pool_client(m0, clients, forget_clients, "M0",
                                     eval_member_ids, eval_nonmember_ids)
    eval_mr = build_eval_pool_client(mr, clients, forget_clients, "MR",
                                     eval_member_ids, eval_nonmember_ids)

    logger = NodeLogger()

    def score_and_log(eval_pool, stage):
        for subset in ("attack_member", "attack_nonmember", "F", "H"):
            e = eval_pool[subset]
            if len(e["probs"]) == 0:
                continue
            pred, score, thr, fb = shokri.predict(e["probs"], e["cls"])
            for k in range(len(e["probs"])):
                logger.add(dataset=dataset, scenario=scenario, seed=seed,
                           global_node_id=e["gid"][k], client_id=e["cid"][k],
                           stage=stage, subset=subset, pool_role="evaluation",
                           true_class=e["cls"][k], membership_label=e["member"][k],
                           mia_score=score[k], decision=pred[k],
                           threshold=thr[k], fallback_status=bool(fb[k]),
                           local_degree=e["ldeg"][k],
                           local_label_agreement=e["lla"][k],
                           correct_prediction=e["corr"][k],
                           prob_vector=e["probs"][k])

    score_and_log(eval_m0, "M0")
    score_and_log(eval_mr, "MR")

    # ── 8. Endpoints from the logged rows ───────────────────────────────
    df = logger.to_frame()
    def rate(stage, subset):
        s_ = df[(df.stage == stage) & (df.subset == subset)]
        return float(s_.decision.mean()) if len(s_) else float("nan")
    def meanscore(stage, subset):
        s_ = df[(df.stage == stage) & (df.subset == subset)]
        return float(s_.mia_score.mean()) if len(s_) else float("nan")

    fpr_H_MR = rate("MR", "H")
    fpr_F_MR = rate("MR", "F")
    C_cal = fpr_H_MR - ALPHA
    C_F   = fpr_F_MR - fpr_H_MR

    # Clean attack strength on RETAINED-client member/nonmember pool under MR:
    # AUC, plus TPR/FPR at the fixed operating point (alpha).
    clean = df[df.subset.isin(["attack_member", "attack_nonmember"]) & (df.stage == "MR")]
    from sklearn.metrics import roc_auc_score
    try:
        auc_strength = float(roc_auc_score(clean.membership_label, clean.mia_score))
    except ValueError:
        auc_strength = float("nan")
    mem = clean[clean.membership_label == 1]
    non = clean[clean.membership_label == 0]
    tpr_alpha = float(mem.decision.mean()) if len(mem) else float("nan")
    fpr_alpha = float(non.decision.mean()) if len(non) else float("nan")

    # Corrected generalization gap on MR over RETAINED clients (R5)
    acc_tr, acc_te, gap = generalization_gap(mr, retain_clients, "client", None)

    n_fallback_classes = int(sum(1 for v in shokri.fallback_used.values() if v))

    summary = dict(
        dataset=dataset, scenario=scenario, seed=seed,
        forget_member_rate_M0=rate("M0", "F"),
        forget_member_rate_MR=fpr_F_MR,
        forget_mean_score_M0=meanscore("M0", "F"),
        forget_mean_score_MR=meanscore("MR", "F"),
        heldout_member_rate_M0=rate("M0", "H"),
        heldout_member_rate_MR=fpr_H_MR,
        heldout_mean_score_M0=meanscore("M0", "H"),
        heldout_mean_score_MR=meanscore("MR", "H"),
        C_cal=C_cal, C_F=C_F,
        attack_strength_auc=auc_strength,
        attack_tpr_at_alpha=tpr_alpha,
        attack_fpr_at_alpha=fpr_alpha,
        acc_train=acc_tr, acc_test=acc_te, train_test_gap=gap,
        alpha=ALPHA,
        n_fallback_classes=n_fallback_classes,
    )

    out = Path(out_dir) / "raw"
    tag = f"{dataset}_{scenario}_s{seed}"
    logger.save(out / f"nodelog_{tag}.csv")
    write_manifest(out / f"manifest_{tag}.json",
                   dataset=dataset, scenario=scenario, seed=seed, config=cfg,
                   removed_client_ids=list(forget_clients),
                   forget_global_ids=sorted(F_ids),
                   heldout_global_ids=sorted(H_ids),
                   alpha=ALPHA,
                   thresholds={str(k): float(v) for k, v in shokri.class_thresholds.items()},
                   commit_hash=get_commit_hash(),
                   split_ratios=DATASET_SPLIT_RATIOS.get(dataset, (0.20, 0.40, 0.40)),
                   counts=dict(n_forget_clients=len(forget_clients),
                               n_retain_clients=len(retain_clients),
                               n_clients_configured=K,
                               n_clients_realized=n_clients_actual,
                               fallback_classes=[int(c) for c, v in shokri.fallback_used.items() if v],
                               calib_support={str(k): int(v) for k, v in shokri.calib_support.items()}),
                   attack_settings=dict(
                       n_shadow=n_shadow,
                       shadow_frac=SHADOW_FRAC,
                       calib_frac=CALIB_FRAC,
                       eval_member_frac=eval_member_frac,
                       eval_nonmember_frac=eval_nonmember_frac,
                       min_calib=MIN_CALIB,
                       alpha=ALPHA),
                   fit_global_ids=fit["gid"].tolist(),
                   calib_global_ids=cal["gid"].tolist(),
                   eval_global_ids=sorted(eval_ids_all))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Datasets to run. Omit if using --core.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--scenario", default="client")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-shadow", type=int, default=5)
    ap.add_argument("--core", action="store_true",
                    help="Use the frozen 13 datasets x 5 seeds core set (fix 4).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run cells already present in summary_core.csv instead of skipping them.")
    args = ap.parse_args()
    if args.core:
        args.datasets = CORE_DATASETS
        args.seeds = CORE_SEEDS
        print(f"Core set: {len(CORE_DATASETS)} datasets x {len(CORE_SEEDS)} seeds = "
              f"{len(CORE_DATASETS)*len(CORE_SEEDS)} cells")
    if not args.datasets:
        ap.error("provide --datasets ... or use --core")

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