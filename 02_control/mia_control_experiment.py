from __future__ import annotations

import argparse, copy, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from all_client_unlearning import (
    DATASET_CONFIGS, PlainGCN,
    load_dataset, optimized_balanced_partitioning,
    set_seed, train_client, get_metrics,
)
from mia_attacks import ShokriShadowMIA, compute_detailed_mia_metrics

from mia_control_config import (
    SEEDS, DATASETS_ALL, SCENARIOS,
    CLIENT_FORGET_RATIO, CLIENT_HELDOUT_RATIO,
    NODE_FORGET_RATIO,   NODE_HELDOUT_RATIO,
    DATASET_SPLIT_RATIOS,
    MIA_N_SHADOW, MIA_SHADOW_FRAC, MIA_ATTACK_DEFAULTS,
    DIRS, make_dirs,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Federated helpers  (mirrors mia_experiment.py, adapted for PlainGCN)
# ═══════════════════════════════════════════════════════════════════════════

def apply_per_client_split(clients: list, dataset_name: str,
                           seed: int = 0) -> list:
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


def make_model(cfg: dict, data: dict) -> torch.nn.Module:
    return PlainGCN(
        in_features  = data["num_features"],
        hidden_dim   = cfg["params"]["hid_dim"],
        out_features = data["num_classes"],
        dropout      = cfg["params"]["dropout"],
    )


def federated_train(cfg: dict, clients: list, model: torch.nn.Module,
                    desc: str = "FedAvg") -> tuple[torch.nn.Module, float]:
    n_rounds    = cfg["params"]["num_rounds"]
    check_every = max(1, n_rounds // 10)
    best_acc, best_state = -1.0, None

    for rnd in range(n_rounds):
        client_models = []
        for cl in clients:
            cm = copy.deepcopy(model)
            train_client(cm, cl, cfg, use_momentum=True)
            client_models.append(cm)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Three-way split at partition time  (the CONTROL EXPERIMENT change)
# ═══════════════════════════════════════════════════════════════════════════

def define_forget_and_heldout_client(n_clients: int, seed: int,
                                      forget_ratio: float, heldout_ratio: float
                                      ) -> tuple[list[int], list[int]]:
    """Draw forget clients AND matched held-out clients from the same pool
    using the same rng. Non-overlapping.

    Both groups are then equally unseen by the trained model, ensuring the
    control is 'matched' in the statistical sense.
    """
    rng = np.random.default_rng(seed)
    n_forget  = max(1, int(round(forget_ratio  * n_clients)))
    n_heldout = max(1, int(round(heldout_ratio * n_clients)))

    if n_forget + n_heldout > n_clients - 1:
        raise ValueError(
            f"n_clients={n_clients} too small for forget={n_forget} + "
            f"heldout={n_heldout}; would leave <1 retain client."
        )

    perm = rng.permutation(n_clients).tolist()
    forget_clients  = sorted(perm[:n_forget])
    heldout_clients = sorted(perm[n_forget : n_forget + n_heldout])
    return forget_clients, heldout_clients


def define_forget_and_heldout_node(clients: list, seed: int,
                                    forget_ratio: float, heldout_ratio: float):
    """Pick target client; draw forget nodes AND matched held-out nodes
    from its training-mask pool. Non-overlapping.

    Returns (target_idx, forget_mask_local, heldout_mask_local, retain_mask_local).
    """
    rng = np.random.default_rng(seed)
    target_idx = int(rng.integers(0, len(clients)))
    tgt        = clients[target_idx]
    train_idx  = np.where(np.asarray(tgt["train_mask"]))[0]

    n_forget  = max(1, int(round(forget_ratio  * len(train_idx))))
    n_heldout = max(1, int(round(heldout_ratio * len(train_idx))))
    if n_forget + n_heldout > len(train_idx) - 1:
        raise ValueError(
            f"target client has only {len(train_idx)} train nodes; "
            f"forget={n_forget} + heldout={n_heldout} too many."
        )

    perm       = rng.permutation(train_idx)
    forget_gi  = perm[:n_forget]
    heldout_gi = perm[n_forget : n_forget + n_heldout]

    n_local = tgt["features"].shape[0]
    forget_mask_local  = np.zeros(n_local, dtype=bool); forget_mask_local[forget_gi]   = True
    heldout_mask_local = np.zeros(n_local, dtype=bool); heldout_mask_local[heldout_gi] = True
    retain_mask_local  = np.asarray(tgt["train_mask"]).copy() & ~forget_mask_local & ~heldout_mask_local
    return target_idx, forget_mask_local, heldout_mask_local, retain_mask_local


# ═══════════════════════════════════════════════════════════════════════════
#  Retrain the control model  (excludes BOTH forget and heldout)
# ═══════════════════════════════════════════════════════════════════════════

def retrain_control_client(cfg, data, clients, forget_clients, heldout_clients, seed):
    """Retrain FedAvg on clients minus (forget ∪ heldout)."""
    set_seed(seed)
    excluded = set(forget_clients) | set(heldout_clients)
    retain   = [c for i, c in enumerate(clients) if i not in excluded]
    model    = make_model(cfg, data)
    return federated_train(cfg, retain, model, desc="Retrain-control[client]")


def retrain_control_node(cfg, data, clients, target_idx,
                         forget_mask_local, heldout_mask_local, seed):
    """Retrain FedAvg with target client's training mask reduced by
    forget ∪ heldout (both groups excluded from local loss)."""
    set_seed(seed)
    mod_clients = [copy.deepcopy(c) for c in clients]
    original    = np.asarray(mod_clients[target_idx]["train_mask"])
    mod_clients[target_idx]["train_mask"] = original & ~forget_mask_local & ~heldout_mask_local
    model = make_model(cfg, data)
    return federated_train(cfg, mod_clients, model, desc="Retrain-control[node]")


# ═══════════════════════════════════════════════════════════════════════════
#  Attack data collection  (returns TWO query pools: original + control)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_probs(model, cl, cid: int) -> np.ndarray:
    model.eval()
    log_probs = model(cl["features"], cl["adj"], cid)
    return torch.exp(log_probs).cpu().numpy()


def _to_np_labels(cl) -> np.ndarray:
    lab = cl["labels"]
    return lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)


def collect_target_attack_data_control(target_model, clients,
                                        scenario: str, split_info):
    """Build TWO query pools (original / control) from the same trained model.

    Returns
    -------
    Two dicts, each with keys probs, member, classes, subset_masks.
    They share the same 'retain' and 'test' entries; only the last
    non-member bucket differs.
    """
    probs_buf, class_buf                  = [], []
    m_orig_buf, m_ctrl_buf                = [], []
    subset_orig_buf, subset_ctrl_buf      = [], []

    def _push(p, c, m_orig, m_ctrl, s_orig, s_ctrl):
        if len(p) == 0:
            return
        probs_buf.append(p)
        class_buf.append(c)
        m_orig_buf.append(np.full(len(p), m_orig, dtype=int))
        m_ctrl_buf.append(np.full(len(p), m_ctrl, dtype=int))
        subset_orig_buf.extend([s_orig] * len(p))
        subset_ctrl_buf.extend([s_ctrl] * len(p))

    if scenario == "client":
        forget_clients  = set(split_info["forget_clients"])
        heldout_clients = set(split_info["heldout_clients"])

        for cid, cl in enumerate(clients):
            labels = _to_np_labels(cl)
            probs  = _forward_probs(target_model, cl, int(cl["id"]))
            tr_idx = np.where(np.asarray(cl["train_mask"]))[0]
            te_idx = np.where(np.asarray(cl["test_mask"]))[0]

            if cid in forget_clients:
                # Forget cluster's train nodes: labelled 'forget' in the original
                # pool, and simply excluded from the control pool.
                _push(probs[tr_idx], labels[tr_idx],
                      m_orig=0, m_ctrl=-1,       # -1 = drop from control pool
                      s_orig="forget", s_ctrl="drop")
            elif cid in heldout_clients:
                # Heldout cluster's train nodes: excluded from original,
                # labelled 'heldout' in control.
                _push(probs[tr_idx], labels[tr_idx],
                      m_orig=-1, m_ctrl=0,
                      s_orig="drop", s_ctrl="heldout")
            else:
                # Retain cluster: member in both pools.
                _push(probs[tr_idx], labels[tr_idx],
                      m_orig=1, m_ctrl=1,
                      s_orig="retain", s_ctrl="retain")

            # Test nodes: non-member in both pools, from every client.
            _push(probs[te_idx], labels[te_idx],
                  m_orig=0, m_ctrl=0,
                  s_orig="test", s_ctrl="test")

    else:  # node scenario
        target_idx         = split_info["target_idx"]
        forget_mask_local  = split_info["forget_mask_local"]
        heldout_mask_local = split_info["heldout_mask_local"]
        retain_mask_local  = split_info["retain_mask_local"]

        # Target client: 4 buckets — retain (member), test (nonmember), forget (nonmember), heldout (nonmember)
        tgt    = clients[target_idx]
        labels = _to_np_labels(tgt)
        probs  = _forward_probs(target_model, tgt, int(tgt["id"]))

        r_idx = np.where(retain_mask_local)[0]
        t_idx = np.where(np.asarray(tgt["test_mask"]))[0]
        f_idx = np.where(forget_mask_local)[0]
        h_idx = np.where(heldout_mask_local)[0]

        _push(probs[r_idx], labels[r_idx],
              m_orig=1, m_ctrl=1, s_orig="retain", s_ctrl="retain")
        _push(probs[t_idx], labels[t_idx],
              m_orig=0, m_ctrl=0, s_orig="test",   s_ctrl="test")
        _push(probs[f_idx], labels[f_idx],
              m_orig=0, m_ctrl=-1, s_orig="forget", s_ctrl="drop")
        _push(probs[h_idx], labels[h_idx],
              m_orig=-1, m_ctrl=0, s_orig="drop",  s_ctrl="heldout")

        # Other clients: train nodes are members (M_ret_control trained on them),
        #                test nodes are non-members. Both in both pools.
        for cid, cl in enumerate(clients):
            if cid == target_idx:
                continue
            cl_labels = _to_np_labels(cl)
            cl_probs  = _forward_probs(target_model, cl, int(cl["id"]))
            cl_tr     = np.where(np.asarray(cl["train_mask"]))[0]
            cl_te     = np.where(np.asarray(cl["test_mask"]))[0]
            _push(cl_probs[cl_tr], cl_labels[cl_tr],
                  m_orig=1, m_ctrl=1, s_orig="retain", s_ctrl="retain")
            _push(cl_probs[cl_te], cl_labels[cl_te],
                  m_orig=0, m_ctrl=0, s_orig="test",   s_ctrl="test")

    probs   = np.concatenate(probs_buf)
    classes = np.concatenate(class_buf)
    m_orig  = np.concatenate(m_orig_buf)
    m_ctrl  = np.concatenate(m_ctrl_buf)
    subset_orig_arr = np.array(subset_orig_buf)
    subset_ctrl_arr = np.array(subset_ctrl_buf)

    keep_orig = m_orig >= 0
    keep_ctrl = m_ctrl >= 0

    pool_orig = {
        "probs":       probs[keep_orig],
        "member":      m_orig[keep_orig],
        "classes":     classes[keep_orig],
        "subset_masks": {
            "forget": subset_orig_arr[keep_orig] == "forget",
            "retain": subset_orig_arr[keep_orig] == "retain",
            "test":   subset_orig_arr[keep_orig] == "test",
        },
    }
    pool_ctrl = {
        "probs":       probs[keep_ctrl],
        "member":      m_ctrl[keep_ctrl],
        "classes":     classes[keep_ctrl],
        "subset_masks": {
            "heldout": subset_ctrl_arr[keep_ctrl] == "heldout",
            "retain":  subset_ctrl_arr[keep_ctrl] == "retain",
            "test":    subset_ctrl_arr[keep_ctrl] == "test",
        },
    }
    return pool_orig, pool_ctrl


# ═══════════════════════════════════════════════════════════════════════════
#  Shadow training (identical to main study — trains on random subsets of
#  ALL K clients, not tied to retain/forget/heldout split).
# ═══════════════════════════════════════════════════════════════════════════

def train_shadow_models(cfg, data, clients,
                         n_shadow: int = MIA_N_SHADOW,
                         shadow_frac: float = MIA_SHADOW_FRAC,
                         base_seed: int = 12345):
    n_clients = len(clients)
    n_use     = max(2, int(round(shadow_frac * n_clients)))
    all_probs, all_member, all_classes = [], [], []

    for i in range(n_shadow):
        rng = np.random.default_rng(base_seed + i)
        ids = rng.choice(n_clients, n_use, replace=False).tolist()
        shadow_clients = [copy.deepcopy(clients[j]) for j in ids]

        model = make_model(cfg, data)
        model, _ = federated_train(cfg, shadow_clients, model,
                                    desc=f"Shadow {i + 1}/{n_shadow}")

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

    return (np.concatenate(all_probs),
            np.concatenate(all_member),
            np.concatenate(all_classes))


# ═══════════════════════════════════════════════════════════════════════════
#  Main per-cell runner
# ═══════════════════════════════════════════════════════════════════════════

def run_one(dataset: str, scenario: str, seed: int,
            attack_hp: dict | None = None) -> dict:
    attack_hp = attack_hp or MIA_ATTACK_DEFAULTS
    print(f"\n── {dataset} | {scenario} | seed={seed} ─────────────────────────")

    cfg = DATASET_CONFIGS[dataset]
    set_seed(seed)

    # 1-2. Data + partition + per-client split
    data    = load_dataset(dataset, seed)
    clients = optimized_balanced_partitioning(data, cfg["num_clients"], seed)
    clients = apply_per_client_split(clients, dataset, seed=seed)

    # 3. THREE-WAY split
    if scenario == "client":
        forget_clients, heldout_clients = define_forget_and_heldout_client(
            len(clients), seed,
            CLIENT_FORGET_RATIO, CLIENT_HELDOUT_RATIO)
        print(f"   forget clients:  {forget_clients}")
        print(f"   heldout clients: {heldout_clients}")
        split_info = {
            "forget_clients":  forget_clients,
            "heldout_clients": heldout_clients,
        }
    else:
        target_idx, f_mask, h_mask, r_mask = define_forget_and_heldout_node(
            clients, seed,
            NODE_FORGET_RATIO, NODE_HELDOUT_RATIO)
        print(f"   target client: {target_idx}")
        print(f"   forget nodes:  {int(f_mask.sum())} / "
              f"heldout nodes: {int(h_mask.sum())} / "
              f"train pool: {int(np.asarray(clients[target_idx]['train_mask']).sum())}")
        split_info = {
            "target_idx":         target_idx,
            "forget_mask_local":  f_mask,
            "heldout_mask_local": h_mask,
            "retain_mask_local":  r_mask,
        }

    # 4. Retrain control model (excludes BOTH forget and heldout)
    t = time.perf_counter()
    if scenario == "client":
        model_ret, ret_acc = retrain_control_client(
            cfg, data, clients, forget_clients, heldout_clients, seed)
    else:
        model_ret, ret_acc = retrain_control_node(
            cfg, data, clients, target_idx, f_mask, h_mask, seed)
    t_ret = time.perf_counter() - t
    print(f"   M_ret_control trained: acc={ret_acc:.4f}  ({t_ret:.1f}s)")

    # 5. Shadow models + attack classifier
    t = time.perf_counter()
    sh_probs, sh_member, sh_classes = train_shadow_models(
        cfg, data, clients,
        n_shadow=MIA_N_SHADOW, shadow_frac=MIA_SHADOW_FRAC,
        base_seed=12345 + seed)
    t_sh = time.perf_counter() - t
    print(f"   shadows trained ({MIA_N_SHADOW}, {t_sh:.1f}s)")

    attack = ShokriShadowMIA(**attack_hp)
    attack.fit(sh_probs, sh_member, sh_classes, n_classes=int(data["num_classes"]))

    # 6. Build TWO query pools and score both
    pool_orig, pool_ctrl = collect_target_attack_data_control(
        model_ret, clients, scenario, split_info)

    def _evaluate(pool):
        pred, score, _ = attack.predict(pool["probs"], pool["classes"])
        return compute_detailed_mia_metrics(
            pool["member"], pred, score, pool["classes"], pool["subset_masks"])

    metrics_orig = _evaluate(pool_orig)
    metrics_ctrl = _evaluate(pool_ctrl)

    print(f"   Original AUC (with forget):  {metrics_orig['mia_auc']:.4f}")
    print(f"   Control  AUC (with heldout): {metrics_ctrl['mia_auc']:.4f}")
    print(f"   → diff = {abs(metrics_orig['mia_auc'] - metrics_ctrl['mia_auc']):.4f}")

    # 7. Row
    row = {
        "dataset": dataset, "scenario": scenario, "seed": seed,
        "n_clients": cfg["num_clients"], "n_classes": int(data["num_classes"]),
        "ret_acc_control": float(ret_acc),
        "t_ret_sec": float(t_ret), "t_shadow_sec": float(t_sh),
        "n_pool_orig":       int(len(pool_orig["probs"])),
        "n_pool_ctrl":       int(len(pool_ctrl["probs"])),
        "n_forget":          int(pool_orig["subset_masks"]["forget"].sum()),
        "n_heldout":         int(pool_ctrl["subset_masks"]["heldout"].sum()),
        # AUC comparison (the headline)
        "mia_auc_original":  float(metrics_orig["mia_auc"]),
        "mia_auc_control":   float(metrics_ctrl["mia_auc"]),
        "mia_auc_diff":      float(abs(metrics_orig["mia_auc"]
                                        - metrics_ctrl["mia_auc"])),
        # Full detail for both
        **{f"orig_{k}": v for k, v in metrics_orig.items()
           if not isinstance(v, dict)},
        **{f"ctrl_{k}": v for k, v in metrics_ctrl.items()
           if not isinstance(v, dict)},
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", default=DATASETS_ALL,
                        choices=DATASETS_ALL)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        choices=SCENARIOS)
    parser.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out-tag",   type=str, default="")
    args = parser.parse_args()

    make_dirs()

    suffix    = f"_{args.out_tag}" if args.out_tag else ""
    csv_path  = DIRS["raw"] / f"mia_control_raw{suffix}.csv"
    json_path = DIRS["raw"] / f"mia_control_raw{suffix}.json"

    rows: list[dict] = []
    for ds in args.datasets:
        for scenario in args.scenarios:
            for seed in args.seeds:
                row = run_one(ds, scenario, seed)
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                with open(json_path, "w") as f:
                    json.dump(rows, f, indent=2, default=str)

    print(f"\n✅ Done. Results → {csv_path}")


if __name__ == "__main__":
    main()
