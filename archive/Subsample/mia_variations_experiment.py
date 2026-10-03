"""
mia_variations_experiment.py
============================
Runner for Phase 2 — the variation sweep.

Compared with the main study, this pipeline adds three axes of variation:
  - --partitionings : which graph-partitioning algorithm to use
  - --num-clients   : list of K values (None → per-dataset default)
  - --forget-ratios : list of forget ratios

Runs the Cartesian product across all axes AND (dataset, scenario, seed).
Every row in the output CSV records exactly which config was used, so a
downstream analysis can slice by any knob and correlate against dataset
properties (size, homophily, modularity, train/test gap).

Usage
-----
  # Single-cell debug run:
  python mia_variations_experiment.py \\
      --datasets Cora --scenarios client --seeds 7 \\
      --partitionings metis --num-clients 5 --forget-ratios 0.20

  # Sweep the partitioning axis (others at default):
  python mia_variations_experiment.py \\
      --partitionings louvain metis metis_plus random \\
      --out-tag sweep_partitioning

  # See run_sweeps.sh for the three sweep invocations in one script.
"""
from __future__ import annotations

import argparse, copy, json, time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from all_client_unlearning import (
    DATASET_CONFIGS, PlainGCN,
    load_dataset, set_seed, train_client, get_metrics,
)
from mia_attacks import ShokriShadowMIA, compute_detailed_mia_metrics
from partitioning import partition_graph, partition_modularity

from mia_variations_config import (
    SEEDS, DATASETS_ALL, SCENARIOS,
    DATASET_SPLIT_RATIOS, DATASET_HOMOPHILY,
    MIA_N_SHADOW, MIA_SHADOW_FRAC, MIA_ATTACK_DEFAULTS,
    DEFAULT_PARTITIONING, DEFAULT_NUM_CLIENTS,
    DEFAULT_CLIENT_FORGET_R, DEFAULT_NODE_FORGET_R,
    DIRS, make_dirs,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Federated helpers  (identical to the control experiment)
# ═══════════════════════════════════════════════════════════════════════════

def apply_per_client_split(clients: list, dataset_name: str, seed: int = 0):
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


def federated_train(cfg, clients, model, desc="FedAvg"):
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
#  Forget-set generation  (standard two-way split, per scenario)
# ═══════════════════════════════════════════════════════════════════════════

def define_forget_set_client(n_clients, seed, forget_ratio):
    rng      = np.random.default_rng(seed)
    n_forget = max(1, int(round(forget_ratio * n_clients)))
    return sorted(rng.choice(n_clients, n_forget, replace=False).tolist())


def define_forget_set_node(clients, seed, forget_ratio):
    rng        = np.random.default_rng(seed)
    target_idx = int(rng.integers(0, len(clients)))
    tgt        = clients[target_idx]
    train_idx  = np.where(np.asarray(tgt["train_mask"]))[0]
    n_forget   = max(1, int(round(forget_ratio * len(train_idx))))
    forget_gi  = rng.choice(train_idx, n_forget, replace=False)

    n_local = tgt["features"].shape[0]
    forget_mask_local = np.zeros(n_local, dtype=bool); forget_mask_local[forget_gi] = True
    retain_mask_local = np.asarray(tgt["train_mask"]).copy() & ~forget_mask_local
    return target_idx, forget_mask_local, retain_mask_local


def retrain_gold_client(cfg, data, clients, forget_clients, seed):
    set_seed(seed)
    retain = [c for i, c in enumerate(clients) if i not in forget_clients]
    return federated_train(cfg, retain, make_model(cfg, data),
                            desc="Retrain[client]")


def retrain_gold_node(cfg, data, clients, target_idx, forget_mask_local, seed):
    set_seed(seed)
    mod_clients = [copy.deepcopy(c) for c in clients]
    mod_clients[target_idx]["train_mask"] = (
        np.asarray(mod_clients[target_idx]["train_mask"]) & ~forget_mask_local
    )
    return federated_train(cfg, mod_clients, make_model(cfg, data),
                            desc="Retrain[node]")


# ═══════════════════════════════════════════════════════════════════════════
#  Forward + attack data collection  (standard two-way, retain vs test+forget)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_probs(model, cl, cid):
    model.eval()
    return torch.exp(model(cl["features"], cl["adj"], cid)).cpu().numpy()


def _to_np_labels(cl):
    lab = cl["labels"]
    return lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)


def _acc_on_mask(probs, labels, mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0, 0
    pred = probs[idx].argmax(axis=1)
    return int((pred == labels[idx]).sum()), len(idx)


def compute_train_test_gap(model, clients, scenario, forget_info):
    """train/test accuracy gap of M_ret — the prof-requested control variable."""
    retain_c, retain_t = 0, 0
    test_c,   test_t   = 0, 0
    for cid, cl in enumerate(clients):
        labels = _to_np_labels(cl)
        probs  = _forward_probs(model, cl, int(cl["id"]))
        tr_mask, te_mask = np.asarray(cl["train_mask"]), np.asarray(cl["test_mask"])
        if scenario == "client":
            if cid in forget_info:
                continue                                  # skip forget clients
        c, t = _acc_on_mask(probs, labels, tr_mask)
        retain_c += c; retain_t += t
        c, t = _acc_on_mask(probs, labels, te_mask)
        test_c += c; test_t += t

    retain_acc = retain_c / retain_t if retain_t > 0 else float("nan")
    test_acc   = test_c   / test_t   if test_t   > 0 else float("nan")
    return retain_acc, test_acc, retain_acc - test_acc


def collect_target_attack_data(target_model, clients, scenario, forget_info):
    probs_buf, mem_buf, cls_buf, sub_buf = [], [], [], []

    def _push(p, m, c, s):
        if len(p) == 0: return
        probs_buf.append(p)
        mem_buf.append(np.full(len(p), m, dtype=int))
        cls_buf.append(c)
        sub_buf.extend([s] * len(p))

    if scenario == "client":
        forget_clients = set(forget_info)
        for cid, cl in enumerate(clients):
            labels = _to_np_labels(cl)
            probs  = _forward_probs(target_model, cl, int(cl["id"]))
            tr_idx = np.where(np.asarray(cl["train_mask"]))[0]
            te_idx = np.where(np.asarray(cl["test_mask"]))[0]
            if cid in forget_clients:
                _push(probs[tr_idx], 0, labels[tr_idx], "forget")
            else:
                _push(probs[tr_idx], 1, labels[tr_idx], "retain")
            _push(probs[te_idx], 0, labels[te_idx], "test")

    else:                                                # node scenario
        target_idx, forget_mask_local, retain_mask_local = forget_info
        tgt    = clients[target_idx]
        labels = _to_np_labels(tgt)
        probs  = _forward_probs(target_model, tgt, int(tgt["id"]))
        f_idx = np.where(forget_mask_local)[0]
        r_idx = np.where(retain_mask_local)[0]
        t_idx = np.where(np.asarray(tgt["test_mask"]))[0]
        _push(probs[f_idx], 0, labels[f_idx], "forget")
        _push(probs[r_idx], 1, labels[r_idx], "retain")
        _push(probs[t_idx], 0, labels[t_idx], "test")

        for cid, cl in enumerate(clients):
            if cid == target_idx: continue
            cl_labels = _to_np_labels(cl)
            cl_probs  = _forward_probs(target_model, cl, int(cl["id"]))
            cl_tr = np.where(np.asarray(cl["train_mask"]))[0]
            cl_te = np.where(np.asarray(cl["test_mask"]))[0]
            _push(cl_probs[cl_tr], 1, cl_labels[cl_tr], "retain")
            _push(cl_probs[cl_te], 0, cl_labels[cl_te], "test")

    probs   = np.concatenate(probs_buf)
    member  = np.concatenate(mem_buf)
    classes = np.concatenate(cls_buf)
    sub_arr = np.array(sub_buf)
    subset_masks = {
        "forget": sub_arr == "forget",
        "retain": sub_arr == "retain",
        "test":   sub_arr == "test",
    }
    return probs, member, classes, subset_masks


# ═══════════════════════════════════════════════════════════════════════════
#  Shadow training  (same as main study)
# ═══════════════════════════════════════════════════════════════════════════

def train_shadow_models(cfg, data, clients, n_shadow=MIA_N_SHADOW,
                         shadow_frac=MIA_SHADOW_FRAC, base_seed=12345):
    n_clients = len(clients)
    n_use     = max(2, int(round(shadow_frac * n_clients)))
    all_probs, all_mem, all_cls = [], [], []
    for i in range(n_shadow):
        rng = np.random.default_rng(base_seed + i)
        ids = rng.choice(n_clients, n_use, replace=False).tolist()
        shadow_clients = [copy.deepcopy(clients[j]) for j in ids]
        model = make_model(cfg, data)
        model, _ = federated_train(cfg, shadow_clients, model,
                                    desc=f"Shadow {i+1}/{n_shadow}")
        for cl in shadow_clients:
            cid    = int(cl["id"])
            labels = _to_np_labels(cl)
            probs  = _forward_probs(model, cl, cid)
            tr_idx = np.where(np.asarray(cl["train_mask"]))[0]
            te_idx = np.where(np.asarray(cl["test_mask"]))[0]
            if len(tr_idx) > 0:
                all_probs.append(probs[tr_idx])
                all_mem.append(np.ones(len(tr_idx), dtype=int))
                all_cls.append(labels[tr_idx])
            if len(te_idx) > 0:
                all_probs.append(probs[te_idx])
                all_mem.append(np.zeros(len(te_idx), dtype=int))
                all_cls.append(labels[te_idx])
    return (np.concatenate(all_probs),
            np.concatenate(all_mem),
            np.concatenate(all_cls))


# ═══════════════════════════════════════════════════════════════════════════
#  Per-cell runner  (aware of the axis knobs)
# ═══════════════════════════════════════════════════════════════════════════

def run_one(dataset, scenario, seed, partitioning, num_clients_override,
             forget_ratio, attack_hp=None):
    attack_hp = attack_hp or MIA_ATTACK_DEFAULTS

    cfg = copy.deepcopy(DATASET_CONFIGS[dataset])
    K   = num_clients_override or cfg["num_clients"]
    cfg["num_clients"] = K

    print(f"\n── {dataset} | {scenario} | seed={seed} | partitioning={partitioning} "
          f"| K={K} | forget_ratio={forget_ratio} ──")
    set_seed(seed)

    # 1. Data + partition + per-client split
    data    = load_dataset(dataset, seed)
    clients = partition_graph(data, K, method=partitioning, seed=seed)
    if len(clients) < K:
        print(f"   ⚠ partitioner returned {len(clients)} clients (< K={K}). Continuing.")
    clients = apply_per_client_split(clients, dataset, seed=seed)

    # Partitioning quality (modularity)
    Q = partition_modularity(clients, data)

    # 2. Forget set
    if scenario == "client":
        forget_info = define_forget_set_client(len(clients), seed, forget_ratio)
    else:
        forget_info = define_forget_set_node(clients, seed, forget_ratio)

    # 3. Gold-standard retrain
    t = time.perf_counter()
    if scenario == "client":
        model_ret, ret_acc = retrain_gold_client(cfg, data, clients, forget_info, seed)
    else:
        target_idx, f_mask, r_mask = forget_info
        model_ret, ret_acc = retrain_gold_node(cfg, data, clients, target_idx, f_mask, seed)
    t_ret = time.perf_counter() - t
    print(f"   M_ret trained: acc={ret_acc:.4f} ({t_ret:.1f}s)")

    # 4. Train/test gap (control variable)
    retain_acc, test_acc, tt_gap = compute_train_test_gap(
        model_ret, clients, scenario,
        forget_info if scenario == "client" else forget_info[0])

    # 5. Shadow models
    t = time.perf_counter()
    sh_probs, sh_mem, sh_cls = train_shadow_models(
        cfg, data, clients,
        n_shadow=MIA_N_SHADOW, shadow_frac=MIA_SHADOW_FRAC,
        base_seed=12345 + seed)
    t_sh = time.perf_counter() - t

    # 6. Attack classifier + evaluate
    attack = ShokriShadowMIA(**attack_hp)
    attack.fit(sh_probs, sh_mem, sh_cls, n_classes=int(data["num_classes"]))
    probs, member, classes, subset_masks = collect_target_attack_data(
        model_ret, clients, scenario, forget_info)
    pred, score, _ = attack.predict(probs, classes)
    metrics = compute_detailed_mia_metrics(
        member, pred, score, classes, subset_masks)

    print(f"   MIA AUC={metrics['mia_auc']:.4f}  Acc={metrics['mia_acc']:.4f}  "
          f"gap={tt_gap:+.3f}  Q={Q:.3f}")

    # 7. Row (includes config + controls + MIA)
    return {
        "dataset":              dataset,
        "scenario":             scenario,
        "seed":                 seed,
        # config axes
        "partitioning":         partitioning,
        "num_clients":          K,
        "forget_ratio":         forget_ratio,
        # partition quality
        "modularity":           Q,
        "n_clients_actual":     len(clients),
        # dataset properties (control variables)
        "n_nodes":              int(data["num_nodes"]),
        "n_features":           int(data["num_features"]),
        "n_classes":            int(data["num_classes"]),
        "homophily":            DATASET_HOMOPHILY.get(dataset, float("nan")),
        # model behavior (control variables)
        "retain_acc":           float(retain_acc),
        "test_acc":             float(test_acc),
        "train_test_gap":       float(tt_gap),
        # timings
        "t_ret_sec":            float(t_ret),
        "t_shadow_sec":         float(t_sh),
        # MIA outcomes
        **{f"shokri_{k}": v for k, v in metrics.items() if not isinstance(v, dict)},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main — Cartesian product of all axes
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",       nargs="+", default=DATASETS_ALL,
                        choices=DATASETS_ALL)
    parser.add_argument("--scenarios",      nargs="+", default=SCENARIOS,
                        choices=SCENARIOS)
    parser.add_argument("--seeds",          nargs="+", type=int, default=SEEDS)
    parser.add_argument("--partitionings",  nargs="+",
                        default=[DEFAULT_PARTITIONING],
                        choices=["louvain", "metis", "metis_plus", "random"])
    parser.add_argument("--num-clients",    nargs="+", type=int, default=[0],
                        help="0 → use DATASET_CONFIGS default per dataset")
    parser.add_argument("--forget-ratios",  nargs="+", type=str,
                        default=["None"],
                        help="Float, or 'None' → per-scenario default (0.20 client, 0.10 node)")
    parser.add_argument("--out-tag",        type=str, default="")
    parser.add_argument("--skip-existing",  action="store_true",
                        help="If the output CSV already has a row for a given "
                             "(ds,sc,seed,part,K,fr) combo, skip it. Lets you "
                             "resume a crashed sweep without redoing work.")
    args = parser.parse_args()

    # Parse forget-ratios: allow string 'None' or a float
    parsed_ratios = []
    for x in args.forget_ratios:
        if str(x).lower() == "none":
            parsed_ratios.append(None)
        else:
            try:
                parsed_ratios.append(float(x))
            except ValueError:
                parser.error(f"invalid --forget-ratios value: {x!r}")
    args.forget_ratios = parsed_ratios

    make_dirs()

    suffix    = f"_{args.out_tag}" if args.out_tag else ""
    csv_path  = DIRS["raw"] / f"mia_variations_raw{suffix}.csv"
    json_path = DIRS["raw"] / f"mia_variations_raw{suffix}.json"

    # ── Resume support: load already-completed configs ─────────────────────
    rows: list[dict] = []
    done: set = set()
    if args.skip_existing and csv_path.exists():
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        for _, r in old.iterrows():
            key = (r["dataset"], r["scenario"], int(r["seed"]),
                   r["partitioning"], int(r["num_clients"]),
                   float(r["forget_ratio"]))
            done.add(key)
        print(f"↺  Resume mode: loaded {len(rows)} completed cells from {csv_path}")

    for ds, sc, seed, part, nc, fr in product(
            args.datasets, args.scenarios, args.seeds,
            args.partitionings, args.num_clients, args.forget_ratios):

        # Resolve defaults
        K            = nc or DATASET_CONFIGS[ds]["num_clients"]
        forget_ratio = fr if fr is not None else (
            DEFAULT_CLIENT_FORGET_R if sc == "client" else DEFAULT_NODE_FORGET_R)

        key = (ds, sc, seed, part, K, forget_ratio)
        if key in done:
            print(f"↺  skipping already-done: {ds} | {sc} | seed={seed} | {part} | K={K} | fr={forget_ratio}")
            continue

        try:
            row = run_one(ds, sc, seed, part, K, forget_ratio)
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            row = {"dataset": ds, "scenario": sc, "seed": seed,
                   "partitioning": part, "num_clients": K,
                   "forget_ratio": forget_ratio, "error": str(e)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    print(f"\n✅ Done. Results → {csv_path}")


if __name__ == "__main__":
    main()