"""
keep the validated M0/MR frozen (do NOT re-run them), train
new reference models with explicit fixed seeds, and save their outputs on
F, H, attack_member, attack_nonmember, the calibration nodes, and the RMIA
population pool.

We never train M0/MR here. The target-side probabilities are read from the frozen
node logs. Only the reference models are trained, each with its own explicit seed,
so they are reproducible without depending on the M0/MR RNG history.

Run from inside 07_mia_redesigned. First run Cora seed 42, inspect, then extend.
"""
from __future__ import annotations
import argparse, json, sys, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "core"))

from models import DATASET_CONFIGS, load_dataset, set_seed                     # noqa: E402
from partition import optimized_balanced_partitioning, apply_per_client_split  # noqa: E402
from train import make_model, federated_train, define_forget_set_client        # noqa: E402

ALPHA        = 0.10
N_REF        = 5          # use all five reference models
SHADOW_FRAC  = 0.5        # each reference model trains on half the retained clients (as before)
POP_FRAC     = 0.5        # split retained non-members into population (this) / final-eval (rest)


# --- helpers identical to the pipeline's own (same forward path) ---------------
def _labels_np(cl):
    y = cl["labels"]
    return y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

@torch.no_grad()
def _forward_probs(model, cl, cid):
    model.eval()
    out = model(cl["features"], cl["adj"], cid)
    return torch.softmax(out, dim=1).cpu().numpy()

def _global_ids(cl, local_idx):
    return np.asarray(cl["node_indices"])[local_idx]

def _emit_probs_for_ids(model, clients, wanted_ids):
    """{global_id: softmax} for the requested ids, via the pipeline forward path."""
    wanted = set(int(g) for g in wanted_ids)
    out = {}
    for cid, cl in enumerate(clients):
        g_all = np.asarray(cl["node_indices"])
        mask = np.array([int(g) in wanted for g in g_all])
        if not mask.any():
            continue
        probs = _forward_probs(model, cl, cid)
        for pos in np.where(mask)[0]:
            out[int(g_all[pos])] = [round(float(p), 6) for p in probs[pos]]
    return out


def build_for_cell(dataset, seed, frozen_dir, out_dir, ref_seed_base):
    cfg = DATASET_CONFIGS[dataset]
    K   = cfg["num_clients"]
    tag = f"{dataset}_client_s{seed}"
    out = Path(out_dir) / "reference"; out.mkdir(parents=True, exist_ok=True)

    # --- reproduce the SAME partition/split as the frozen run (seeded) ----------
    # (this does NOT train M0/MR; it only rebuilds the client graphs/splits so the
    #  reference models see the same node identities as the frozen targets.)
    set_seed(seed)
    data = load_dataset(dataset, seed)
    clients = optimized_balanced_partitioning(data, K, seed)
    clients = apply_per_client_split(clients, dataset, seed=seed)
    for cid, cl in enumerate(clients):
        cl["id"] = cid
    n_clients_actual = len(clients)

    # --- forget set + retained clients (same rule as the pipeline) -------------
    forget_clients = define_forget_set_client(cfg, n_clients_actual, seed)
    forget_set = set(forget_clients)
    retain_clients = [c for i, c in enumerate(clients) if i not in forget_set]

    # --- read the frozen node identities + calibration IDs (targets stay frozen)-
    frozen = pd.read_csv(Path(frozen_dir) / f"nodelog_{tag}.csv")
    groups = {sub: sorted(set(int(x) for x in
                 frozen[frozen.subset == sub].global_node_id.unique()))
              for sub in ["F", "H", "attack_member", "attack_nonmember"]}
    man = json.load(open(Path(frozen_dir) / f"manifest_{tag}.json"))
    calib_ids = set(int(x) for x in man.get("calib_global_ids", []))

    # --- RMIA population: deterministic split of retained non-members,
    #     disjoint from calibration and from the final-evaluation non-members ----
    nonmember_pool = np.array(sorted(g for g in groups["attack_nonmember"]
                                     if g not in calib_ids))
    rng = np.random.default_rng(seed + 777)
    perm = rng.permutation(len(nonmember_pool))
    n_pop = int(round(POP_FRAC * len(nonmember_pool)))
    population_ids = sorted(int(x) for x in nonmember_pool[perm[:n_pop]])
    final_eval_ids = sorted(int(x) for x in nonmember_pool[perm[n_pop:]])
    assert not (set(population_ids) & set(final_eval_ids))
    assert not (set(population_ids) & calib_ids)

    # true labels for calibration nodes (needed so entropy calibrates on the
    # designated calibration pool, consistent with the other attacks).
    gid_to_label = {}
    for cl in clients:
        gl = np.asarray(cl["node_indices"]); yl = _labels_np(cl)
        for g, y in zip(gl, yl):
            gid_to_label[int(g)] = int(y)
    calibration_labels = {int(g): gid_to_label[int(g)] for g in calib_ids if int(g) in gid_to_label}

    # nodes we need reference outputs on (everything listed):
    want_ids = sorted(set(groups["F"]) | set(groups["H"]) | set(groups["attack_member"])
                      | set(groups["attack_nonmember"]) | calib_ids | set(population_ids))

    # reference models must be OUT of every evaluation identity we score:
    exclude = set(groups["F"]) | set(groups["H"]) | set(groups["attack_member"]) \
              | set(groups["attack_nonmember"])

    # --- train N_REF reference models, each with an EXPLICIT fixed seed ---------
    sel_rng = np.random.default_rng(seed + 12345)         # same client-selection stream as pipeline
    n_use = max(2, int(round(SHADOW_FRAC * len(retain_clients))))
    reference_probs, reference_train_ids = [], []
    for i in range(N_REF):
        ids = sel_rng.choice(len(retain_clients), n_use, replace=False).tolist()
        ref_clients = [copy.deepcopy(retain_clients[j]) for j in ids]
        trained = set()
        for cl in ref_clients:
            tm = np.asarray(cl["train_mask"]).copy()
            g_all = np.asarray(cl["node_indices"])
            drop = np.array([int(g) in exclude for g in g_all])
            cl["train_mask"] = tm & ~drop
            trained.update(int(g) for g in g_all[np.asarray(cl["train_mask"])])
        set_seed(ref_seed_base + i)                       # <-- explicit fixed seed (weights)
        rm = make_model(cfg, data)
        rm, _ = federated_train(cfg, ref_clients, rm, desc=f"Ref {i+1}/{N_REF}")
        reference_probs.append(_emit_probs_for_ids(rm, clients, want_ids))
        reference_train_ids.append(sorted(trained))

    # OUT-status check: no scored eval node in any reference training set
    leak = sum(len(set(tr) & exclude) for tr in reference_train_ids)

    # --- save everything asked for ---------------------------------------
    payload = dict(
        dataset=dataset, seed=seed, alpha=ALPHA,
        n_reference_models=N_REF, ref_seed_base=ref_seed_base,
        node_groups=groups,
        calibration_ids=sorted(calib_ids),
        calibration_labels=calibration_labels,
        population_ids=population_ids,
        final_eval_nonmember_ids=final_eval_ids,
        reference_probs=reference_probs,            # list[dict gid->softmax] per ref model
        reference_train_ids=reference_train_ids,    # provenance / OUT evidence
        out_status_leak=leak,
    )
    fp = out / f"refpack_{tag}.json"
    json.dump(payload, open(fp, "w"))
    print(f"[{tag}] refs={N_REF}  want_nodes={len(want_ids)}  population={len(population_ids)}  "
          f"final_eval_nonmembers={len(final_eval_ids)}  OUT_leak={leak} "
          f"{'(clean)' if leak==0 else '(LEAK!)'}")
    print(f"        saved -> {fp}")
    return dict(tag=tag, leak=leak, n_pop=len(population_ids), n_fe=len(final_eval_ids),
                n_want=len(want_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--frozen-dir", default="results_13x5/raw")
    ap.add_argument("--out-dir", default="attack_ext_out")
    ap.add_argument("--ref-seed-base", type=int, default=90000,
                    help="explicit base seed for reference-model weights; model i uses base+i")
    args = ap.parse_args()
    rows = []
    for d in args.datasets:
        for s in args.seeds:
            print(f"\n===== {d} | seed {s} =====")
            rows.append(build_for_cell(d, s, args.frozen_dir, args.out_dir, args.ref_seed_base))
    print("\n===== SUMMARY =====")
    for r in rows:
        print(f"  {r['tag']}: OUT_leak={r['leak']}  population={r['n_pop']}  "
              f"final_eval={r['n_fe']}  scored_nodes={r['n_want']}")
    print("\nNote: M0/MR were NOT retrained. Target probabilities remain the frozen ones "
          "in results_13x5/raw. This pass only adds reference-model outputs.")


if __name__ == "__main__":
    main()