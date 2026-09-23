from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from all_client_unlearning import (
    DATASET_CONFIGS, GatedOptimizedGCN, PlainGCN,
    load_dataset, optimized_balanced_partitioning,
    set_seed, train_client, get_metrics,
)

from mia_config import (
    SEEDS, DATASETS_ALL, SCENARIOS,
    CLIENT_FORGET_RATIO, NODE_FORGET_RATIO,
    DATASET_SPLIT_RATIOS,
    MIA_N_SHADOW, MIA_SHADOW_FRAC, MIA_ATTACK_DEFAULTS,
    DIRS, make_dirs,
)
from mia_attacks import (
    ShokriShadowMIA, LRConfidenceMIA, compute_detailed_mia_metrics,
)


# ════════════════════════════════════════════════════════════════════════════
#  Federated infrastructure adapters (partition-first protocol)
# ════════════════════════════════════════════════════════════════════════════

def apply_per_client_split(clients: list, dataset_name: str,
                           seed: int = 0) -> list:
    """Apply per-client train/val/test split with dataset-specific ratios.

    Partitioning has already happened; this re-assigns each client's
    train_mask, val_mask, test_mask using a random permutation of its OWN
    local nodes at the ratio specified for that dataset.
    """
    train_r, val_r, _ = DATASET_SPLIT_RATIOS.get(dataset_name, (0.20, 0.40, 0.40))
    rng = np.random.default_rng(seed)
    for cl in clients:
        n    = cl["features"].shape[0]
        perm = rng.permutation(n)
        t1   = int(n * train_r)
        t2   = int(n * (train_r + val_r))
        train_mask = np.zeros(n, dtype=bool); train_mask[perm[:t1]]   = True
        val_mask   = np.zeros(n, dtype=bool); val_mask[perm[t1:t2]]   = True
        test_mask  = np.zeros(n, dtype=bool); test_mask[perm[t2:]]    = True
        cl["train_mask"], cl["val_mask"], cl["test_mask"] = train_mask, val_mask, test_mask
    return clients


def make_model(cfg: dict, data: dict, num_clients: int | None = None
               ) -> torch.nn.Module:
    """Build a fresh PlainGCN sized for the given (cfg, data).

    Plain 2-layer GCN — no per-client gates. Used for the retrain-from-scratch
    baseline (original, retrained, and shadow models) so the MIA-consistency
    experiment runs against a standard architecture with no custom
    personalization mechanism.

    The `num_clients` argument is accepted for API compatibility with existing
    callers (e.g. shadow training) but is ignored — PlainGCN has no gates
    indexed by client id.
    """
    return PlainGCN(
        in_features  = data["num_features"],
        hidden_dim   = cfg["params"]["hid_dim"],
        out_features = data["num_classes"],
        dropout      = cfg["params"]["dropout"],
    )


def federated_train(cfg: dict, clients: list, model: torch.nn.Module,
                    desc: str = "FedAvg") -> tuple[torch.nn.Module, float]:
    """Standard FedAvg loop with best-checkpoint selection by global val/test
    accuracy (via get_metrics). Returns (best_model, best_acc).
    """
    n_rounds    = cfg["params"]["num_rounds"]
    check_every = max(1, n_rounds // 10)

    best_acc, best_state = -1.0, None
    for rnd in range(n_rounds):
        client_models = []
        for cl in clients:
            cm = copy.deepcopy(model)
            train_client(cm, cl, cfg, use_momentum=True)
            client_models.append(cm)

        # FedAvg
        sd = model.state_dict()
        for key in sd:
            sd[key] = torch.stack(
                [cm.state_dict()[key].float() for cm in client_models]
            ).mean(0)
        model.load_state_dict(sd)

        if (rnd + 1) % check_every == 0 or rnd == n_rounds - 1:
            _, acc = get_metrics(model, clients)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_acc)


# ════════════════════════════════════════════════════════════════════════════
#  Forget set generation  (scenario-specific)
# ════════════════════════════════════════════════════════════════════════════

def define_forget_set_client(cfg: dict, n_clients: int, seed: int,
                              forget_ratio: float = CLIENT_FORGET_RATIO
                              ) -> list[int]:
    """Pick `forget_ratio` of clients at random."""
    rng = np.random.default_rng(seed)
    n_forget = max(1, int(round(forget_ratio * n_clients)))
    return sorted(rng.choice(n_clients, n_forget, replace=False).tolist())


def define_forget_set_node(clients: list, seed: int,
                            forget_ratio: float = NODE_FORGET_RATIO):
    """Pick one target client; sample `forget_ratio` of its training nodes.

    Returns (target_idx, forget_mask_local, retain_mask_local) where masks
    are bool arrays over the target client's LOCAL node indices.
    """
    rng = np.random.default_rng(seed)
    target_idx = int(rng.integers(0, len(clients)))
    tgt = clients[target_idx]
    train_idx = np.where(np.asarray(tgt["train_mask"]))[0]

    if len(train_idx) < 2:
        raise RuntimeError(
            f"Target client {target_idx} has only {len(train_idx)} train nodes; "
            "cannot sample forget set."
        )

    n_forget = max(1, int(round(forget_ratio * len(train_idx))))
    forget_local = rng.choice(train_idx, n_forget, replace=False)

    n_local = tgt["features"].shape[0]
    forget_mask_local = np.zeros(n_local, dtype=bool); forget_mask_local[forget_local] = True
    retain_mask_local = np.asarray(tgt["train_mask"]).copy() & ~forget_mask_local
    return target_idx, forget_mask_local, retain_mask_local


# ════════════════════════════════════════════════════════════════════════════
#  Gold-standard retrain from scratch
# ════════════════════════════════════════════════════════════════════════════

def retrain_gold_client(cfg, data, clients, forget_clients, seed):
    """Retrain FedAvg on retain clients only (forget clients excluded)."""
    set_seed(seed)
    retain = [c for i, c in enumerate(clients) if i not in forget_clients]
    # Keep cfg["num_clients"] the same so gate indexing stays aligned with
    # the original client IDs (forget clients' gates simply stay at init).
    model = make_model(cfg, data)
    return federated_train(cfg, retain, model, desc="Retrain[client]")


def retrain_gold_node(cfg, data, clients, target_idx, forget_mask_local, seed):
    """Retrain FedAvg from scratch with forget nodes excluded from the
    target client's train_mask. All clients participate; only target's
    train_mask is shrunk.
    """
    set_seed(seed)
    mod_clients = [copy.deepcopy(c) for c in clients]
    mod_clients[target_idx]["train_mask"] = (
        np.asarray(mod_clients[target_idx]["train_mask"]) & ~forget_mask_local
    )
    model = make_model(cfg, data)
    return federated_train(cfg, mod_clients, model, desc="Retrain[node]")


# ════════════════════════════════════════════════════════════════════════════
#  Forward helpers
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_probs(model, cl, cid: int) -> np.ndarray:
    """Return softmax probability matrix (n_local, n_classes) for one client."""
    model.eval()
    log_probs = model(cl["features"], cl["adj"], cid)
    return torch.exp(log_probs).cpu().numpy()


def _acc_on_mask(probs: np.ndarray, labels: np.ndarray,
                 mask: np.ndarray) -> tuple[int, int]:
    """Return (#correct, #total) on the given bool mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0, 0
    pred = probs[idx].argmax(axis=1)
    return int((pred == labels[idx]).sum()), len(idx)


def _to_np_labels(cl) -> np.ndarray:
    lab = cl["labels"]
    return lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)


# ════════════════════════════════════════════════════════════════════════════
#  Basic gold-standard metrics  (scenario-aware evaluation)
# ════════════════════════════════════════════════════════════════════════════

def compute_basic_metrics(model_orig, model_ret, clients,
                           scenario: str, forget_info) -> dict:
    """Compute retain/forget/test accuracy and forgetting score on M_ret.

    Evaluation protocol:
      client : retain_acc = mean over RETAIN clients of test_mask accuracy
               (whole clients are removed, so generalization on the kept
               clients is the right utility signal)
               forget_acc = mean over FORGET clients of train_mask accuracy
               (the data that was removed)
               test_acc   = mean over ALL clients of test_mask accuracy
      node   : retain_acc = target client's test_mask accuracy
               (only the target's local view changed; its remaining test
               nodes are the right utility signal)
               forget_acc = target client's forget (removed train) nodes
               test_acc   = target client's test_mask accuracy

    Forgetting score = JSD( P_orig || P_ret ) on forget nodes (using the
    same forward(cid=...) convention as accuracy above).
    """
    out: dict = {}
    eps = 1e-12

    if scenario == "client":
        forget_clients = forget_info
        # Per-client accuracies under each client's own gate
        retain_accs, forget_accs, test_accs = [], [], []
        forget_probs_orig, forget_probs_ret = [], []

        for cid, cl in enumerate(clients):
            labels  = _to_np_labels(cl)
            probs_r = _forward_probs(model_ret, cl, cid)
            probs_o = _forward_probs(model_orig, cl, cid)

            tr_mask, te_mask = np.asarray(cl["train_mask"]), np.asarray(cl["test_mask"])

            # Test accuracy contribution (every client contributes, including forget clients)
            c, t = _acc_on_mask(probs_r, labels, te_mask)
            if t > 0:
                test_accs.append(c / t)

            # Retain accuracy = retain clients' TEST nodes (whole clients removed,
            # so generalization on the kept clients is the right utility signal).
            # Forget accuracy = forget clients' TRAIN nodes (the data that was removed).
            if cid in forget_clients:
                c, t = _acc_on_mask(probs_r, labels, tr_mask)
                if t > 0:
                    forget_accs.append(c / t)
            else:
                c, t = _acc_on_mask(probs_r, labels, te_mask)
                if t > 0:
                    retain_accs.append(c / t)

            # JSD on forget clients' train nodes
            if cid in forget_clients:
                f_idx = np.where(tr_mask)[0]
                if len(f_idx) > 0:
                    forget_probs_orig.append(probs_o[f_idx])
                    forget_probs_ret.append(probs_r[f_idx])

        out["retain_acc"] = float(np.mean(retain_accs)) if retain_accs else float("nan")
        out["forget_acc"] = float(np.mean(forget_accs)) if forget_accs else float("nan")
        out["test_acc"]   = float(np.mean(test_accs))   if test_accs   else float("nan")
        out["n_forget_clients"] = len(forget_clients)
        out["n_retain_clients"] = len(clients) - len(forget_clients)

    else:  # node scenario — TARGET CLIENT'S LOCAL VIEW ONLY
        target_idx, forget_mask_local, retain_mask_local = forget_info
        tgt = clients[target_idx]
        labels  = _to_np_labels(tgt)
        probs_r = _forward_probs(model_ret, tgt, target_idx)
        probs_o = _forward_probs(model_orig, tgt, target_idx)

        # Retain accuracy = target client's TEST nodes (only the target's local
        # view changed; its remaining test nodes are the right utility signal).
        # Forget accuracy = the removed train nodes.
        c_r, t_r = _acc_on_mask(probs_r, labels, np.asarray(tgt["test_mask"]))
        c_f, t_f = _acc_on_mask(probs_r, labels, forget_mask_local)
        c_t, t_t = _acc_on_mask(probs_r, labels, np.asarray(tgt["test_mask"]))

        out["retain_acc"] = c_r / t_r if t_r > 0 else float("nan")
        out["forget_acc"] = c_f / t_f if t_f > 0 else float("nan")
        out["test_acc"]   = c_t / t_t if t_t > 0 else float("nan")
        out["target_client"] = target_idx
        out["n_target_forget"] = int(t_f)
        out["n_target_retain"] = int(np.asarray(retain_mask_local).sum())

        f_idx = np.where(forget_mask_local)[0]
        forget_probs_orig = [probs_o[f_idx]] if len(f_idx) > 0 else []
        forget_probs_ret  = [probs_r[f_idx]] if len(f_idx) > 0 else []

    # Forgetting score (JSD on forget nodes)
    if forget_probs_orig:
        p = np.concatenate(forget_probs_orig)
        q = np.concatenate(forget_probs_ret)
        m = 0.5 * (p + q)
        kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum(axis=1)
        kl_qm = (q * (np.log(q + eps) - np.log(m + eps))).sum(axis=1)
        out["forgetting_score"] = float(np.mean(0.5 * (kl_pm + kl_qm)))
    else:
        out["forgetting_score"] = float("nan")

    return out


# ════════════════════════════════════════════════════════════════════════════
#  Shadow model training  (Shokri attack.py:train_shadow_models)
# ════════════════════════════════════════════════════════════════════════════

def train_shadow_models(cfg, data, clients,
                         n_shadow: int = MIA_N_SHADOW,
                         shadow_frac: float = MIA_SHADOW_FRAC,
                         base_seed: int = 12345):
    """Train `n_shadow` FedAvg models on disjoint random client subsets.

    For each shadow:
      members      = train_mask nodes in its clients   (label=1)
      non-members  = test_mask  nodes in its clients   (label=0)

    Returns
    -------
    shadow_probs    : (M, n_classes)  softmax probs
    shadow_member   : (M,)            {0, 1}
    shadow_classes  : (M,)            true class labels
    """
    n_clients = len(clients)
    n_use     = max(2, int(round(shadow_frac * n_clients)))

    all_probs, all_member, all_classes = [], [], []

    for i in range(n_shadow):
        rng = np.random.default_rng(base_seed + i)
        ids = rng.choice(n_clients, n_use, replace=False).tolist()

        # Keep clients' ORIGINAL ids — `train_client` indexes the gate via
        # `client_data['id']`, so the shadow model must keep the same gate
        # count as the target. Only the gates of selected clients receive
        # updates; the rest stay at initialization (σ(2.0) ≈ 0.88).
        shadow_clients = [copy.deepcopy(clients[j]) for j in ids]

        model = make_model(cfg, data)  # full-size gates (num_clients = cfg["num_clients"])
        model, _ = federated_train(cfg, shadow_clients, model,
                                   desc=f"Shadow {i + 1}/{n_shadow}")

        # Collect softmax outputs using each client's OWN gate (its `id`)
        for cl in shadow_clients:
            cid    = int(cl["id"])
            labels = _to_np_labels(cl)
            probs  = _forward_probs(model, cl, cid)

            tr_idx = np.where(np.asarray(cl["train_mask"]))[0]
            te_idx = np.where(np.asarray(cl["test_mask"]))[0]
            if len(tr_idx) > 0:
                all_probs.append(probs[tr_idx])
                all_member.append(np.ones(len(tr_idx), dtype=int))
                all_classes.append(labels[tr_idx])
            if len(te_idx) > 0:
                all_probs.append(probs[te_idx])
                all_member.append(np.zeros(len(te_idx), dtype=int))
                all_classes.append(labels[te_idx])

    return (
        np.concatenate(all_probs),
        np.concatenate(all_member),
        np.concatenate(all_classes),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Build target attack data  (scenario-aware, matches eval convention)
# ════════════════════════════════════════════════════════════════════════════

def collect_target_attack_data(target_model, clients,
                                scenario: str, forget_info):
    """Build the MIA *test* set from the gold-standard target model.

    Membership labels w.r.t. M_ret's training set:
      • retain (member=1) — nodes M_ret was trained on
      • test   (member=0) — held-out test nodes (never trained on)
      • forget (member=0) — nodes excluded from M_ret's training by construction

    Forward client_id convention follows the scenario-specific eval protocol:
      • client scenario : each client uses its own cid
      • node scenario   : ALL queries use the target client's view (cid=target_idx)
                          since the target client is the only one whose local
                          view changed; this is consistent with the utility
                          eval above.

    Returns
    -------
    probs           : (N, C) softmax
    member          : (N,)   ground-truth membership labels
    true_classes    : (N,)   true class labels
    subset_masks    : dict — bool masks over [0..N) marking each row's bucket
    """
    probs_buf, member_buf, class_buf, subset_buf = [], [], [], []

    def _push(p, m, c, s):
        if len(p) == 0:
            return
        probs_buf.append(p)
        member_buf.append(np.full(len(p), m, dtype=int))
        class_buf.append(c)
        subset_buf.extend([s] * len(p))

    if scenario == "client":
        forget_clients = set(forget_info)
        for cid, cl in enumerate(clients):
            labels = _to_np_labels(cl)
            probs  = _forward_probs(target_model, cl, cid)
            tr_idx = np.where(np.asarray(cl["train_mask"]))[0]
            te_idx = np.where(np.asarray(cl["test_mask"]))[0]

            if cid in forget_clients:
                _push(probs[tr_idx], 0, labels[tr_idx], "forget")
            else:
                _push(probs[tr_idx], 1, labels[tr_idx], "retain")
            _push(probs[te_idx], 0, labels[te_idx], "test")

    else:  # node scenario — target client's local view only (cid=target_idx)
        target_idx, forget_mask_local, retain_mask_local = forget_info

        # Target client: forget + retain + test partitions, all via target's gate
        tgt    = clients[target_idx]
        labels = _to_np_labels(tgt)
        probs  = _forward_probs(target_model, tgt, target_idx)

        f_idx = np.where(forget_mask_local)[0]
        r_idx = np.where(retain_mask_local)[0]
        t_idx = np.where(np.asarray(tgt["test_mask"]))[0]
        _push(probs[f_idx], 0, labels[f_idx], "forget")
        _push(probs[r_idx], 1, labels[r_idx], "retain")
        _push(probs[t_idx], 0, labels[t_idx], "test")

        # Other clients: their training nodes are still members of the federated
        # retrained model (only target's nodes were excluded). Evaluate them
        # under each client's own gate, since the target's gate is meaningless
        # for them. This grows the member pool so MIA has enough positives.
        for cid, cl in enumerate(clients):
            if cid == target_idx:
                continue
            cl_labels = _to_np_labels(cl)
            cl_probs  = _forward_probs(target_model, cl, cid)
            cl_tr     = np.where(np.asarray(cl["train_mask"]))[0]
            cl_te     = np.where(np.asarray(cl["test_mask"]))[0]
            _push(cl_probs[cl_tr], 1, cl_labels[cl_tr], "retain")
            _push(cl_probs[cl_te], 0, cl_labels[cl_te], "test")

    probs        = np.concatenate(probs_buf)
    member       = np.concatenate(member_buf)
    true_classes = np.concatenate(class_buf)
    subset_arr   = np.array(subset_buf)
    subset_masks = {
        "forget": subset_arr == "forget",
        "retain": subset_arr == "retain",
        "test":   subset_arr == "test",
    }
    return probs, member, true_classes, subset_masks


# ════════════════════════════════════════════════════════════════════════════
#  Run one (dataset, scenario, seed) cell
# ════════════════════════════════════════════════════════════════════════════

def run_one(dataset: str, scenario: str, seed: int,
            attack_hp: dict | None = None) -> dict:
    attack_hp = attack_hp or MIA_ATTACK_DEFAULTS
    print(f"\n── {dataset} | {scenario} | seed={seed} ─────────────────────────")

    cfg = DATASET_CONFIGS[dataset]
    set_seed(seed)

    # 1–2. Data
    data    = load_dataset(dataset, seed)
    clients = optimized_balanced_partitioning(data, cfg["num_clients"], seed)
    clients = apply_per_client_split(clients, dataset, seed=seed)

    # 3. Original FedAvg
    t = time.perf_counter()
    model_orig = make_model(cfg, data)
    model_orig, orig_acc = federated_train(cfg, clients, model_orig, desc="Original")
    t_orig = time.perf_counter() - t
    print(f"   original FedAvg done: acc={orig_acc:.4f}  ({t_orig:.1f}s)")

    # 4. Forget set
    if scenario == "client":
        forget_clients = define_forget_set_client(cfg, len(clients), seed)
        forget_info    = forget_clients
        print(f"   forget clients: {forget_clients}")
    else:
        target_idx, forget_mask_local, retain_mask_local = define_forget_set_node(clients, seed)
        forget_info = (target_idx, forget_mask_local, retain_mask_local)
        print(f"   target client: {target_idx}  "
              f"forget nodes: {int(forget_mask_local.sum())} / "
              f"{int(np.asarray(clients[target_idx]['train_mask']).sum())}")

    # 5. Gold-standard retrain
    t = time.perf_counter()
    if scenario == "client":
        model_ret, ret_acc = retrain_gold_client(cfg, data, clients, forget_clients, seed)
    else:
        model_ret, ret_acc = retrain_gold_node(cfg, data, clients, target_idx,
                                                forget_mask_local, seed)
    t_ret = time.perf_counter() - t
    print(f"   gold-standard retrain done: acc={ret_acc:.4f}  ({t_ret:.1f}s)")

    # 6. Basic gold-standard metrics
    basic = compute_basic_metrics(model_orig, model_ret, clients, scenario, forget_info)
    print(f"   retain_acc={basic['retain_acc']:.4f}  "
          f"forget_acc={basic['forget_acc']:.4f}  "
          f"test_acc={basic['test_acc']:.4f}  "
          f"FS={basic['forgetting_score']:.4f}")

    # 7. Shadow models (shared by both attacks)
    t = time.perf_counter()
    shadow_probs, shadow_member, shadow_classes = train_shadow_models(
        cfg, data, clients,
        n_shadow=MIA_N_SHADOW, shadow_frac=MIA_SHADOW_FRAC,
        base_seed=12345 + seed,
    )
    t_shadow = time.perf_counter() - t
    print(f"   shadow models trained ({MIA_N_SHADOW}): "
          f"{len(shadow_probs)} attack samples  ({t_shadow:.1f}s)")

    # 8. Target attack data
    target_probs, target_member, target_classes, subset_masks = \
        collect_target_attack_data(model_ret, clients, scenario, forget_info)

    # 9. Shokri MIA
    shokri = ShokriShadowMIA(**attack_hp)
    shokri.fit(shadow_probs, shadow_member, shadow_classes,
               n_classes=int(data["num_classes"]))
    sh_pred, sh_score, _ = shokri.predict(target_probs, target_classes)
    shokri_metrics = compute_detailed_mia_metrics(
        target_member, sh_pred, sh_score, target_classes, subset_masks)
    print(f"   Shokri MIA: acc={shokri_metrics['mia_acc']:.4f}  "
          f"AUC={shokri_metrics['mia_auc']:.4f}  "
          f"forget_pred_member_rate="
          f"{shokri_metrics.get('forget_predicted_member_rate', float('nan')):.4f}")

    # 10. LR confidence MIA (train on retain+test from target, query everything)
    lr_train_mask = subset_masks["retain"] | subset_masks["test"]
    if lr_train_mask.sum() > 20 and len(np.unique(target_member[lr_train_mask])) == 2:
        lr = LRConfidenceMIA()
        lr.fit(target_probs[lr_train_mask], target_member[lr_train_mask],
               target_classes[lr_train_mask])
        lr_pred, lr_score = lr.predict(target_probs, target_classes)
        lr_metrics = compute_detailed_mia_metrics(
            target_member, lr_pred, lr_score, target_classes, subset_masks)
        print(f"   LR MIA:     acc={lr_metrics['mia_acc']:.4f}  "
              f"AUC={lr_metrics['mia_auc']:.4f}  "
              f"forget_pred_member_rate="
              f"{lr_metrics.get('forget_predicted_member_rate', float('nan')):.4f}")
    else:
        lr_metrics = {}

    # 11. Row
    row = {
        "dataset": dataset, "scenario": scenario, "seed": seed,
        "n_clients": cfg["num_clients"], "n_classes": int(data["num_classes"]),
        "orig_acc": float(orig_acc), "ret_acc": float(ret_acc),
        "t_orig_sec": float(t_orig), "t_ret_sec": float(t_ret),
        "t_shadow_sec": float(t_shadow),
        **basic,
        "n_attack_total":     int(len(target_probs)),
        "n_attack_member":    int((target_member == 1).sum()),
        "n_attack_nonmember": int((target_member == 0).sum()),
        # Shokri
        **{f"shokri_{k}": v for k, v in shokri_metrics.items()
           if not isinstance(v, dict)},
        "shokri_per_data_class": shokri_metrics.get("per_data_class", {}),
        # LR
        **{f"lr_{k}": v for k, v in lr_metrics.items()
           if not isinstance(v, dict)},
        "lr_per_data_class": lr_metrics.get("per_data_class", {}),
        # Attack HP used
        "attack_hp": dict(attack_hp),
    }
    return row


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", default=DATASETS_ALL,
                        choices=DATASETS_ALL)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        choices=SCENARIOS)
    parser.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    parser.add_argument("--hp-path",   type=str, default=None,
                        help="Optional path to best_attack_configs.json")
    parser.add_argument("--out-tag",   type=str, default="",
                        help="Optional suffix for output files")
    args = parser.parse_args()

    make_dirs()

    best_hp: dict = {}
    if args.hp_path and Path(args.hp_path).exists():
        with open(args.hp_path) as f:
            best_hp = json.load(f)
        print(f"Loaded per-dataset attack HPs from {args.hp_path}")

    suffix   = f"_{args.out_tag}" if args.out_tag else ""
    csv_path = DIRS["raw"] / f"mia_consistency_raw{suffix}.csv"
    json_path = DIRS["raw"] / f"mia_consistency_raw{suffix}.json"

    rows: list[dict] = []
    for ds in args.datasets:
        for scenario in args.scenarios:
            hp = best_hp.get(f"{ds}_{scenario}", MIA_ATTACK_DEFAULTS)
            for seed in args.seeds:
                row = run_one(ds, scenario, seed, attack_hp=hp)
                rows.append(row)
                # Incremental save (resume-friendly)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                with open(json_path, "w") as f:
                    json.dump(rows, f, indent=2, default=str)

    print(f"\n✅ Done. Results → {csv_path}")


if __name__ == "__main__":
    main()