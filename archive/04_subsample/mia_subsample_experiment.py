from __future__ import annotations

import argparse, sys, copy
from pathlib import Path
from itertools import product

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import all_client_unlearning as ACU            # noqa: E402
import mia_variations_experiment as V          # noqa: E402
from subsample import mhrw_sample_nodes, topology_report  # noqa: E402
from mia_subsample_config import (             # noqa: E402
    DATASETS, SCENARIOS, FRACTIONS, SEEDS,
    DEFAULT_PARTITIONING, DEFAULT_CLIENT_FORGET_R, DEFAULT_NODE_FORGET_R,
)

RAW_DIR = Path("mia_subsample_results") / "raw_data"


def _labels_np(data):
    y = data["labels"]
    return y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)


def _row_renormalize_adj(edge_index: np.ndarray, n: int) -> torch.Tensor:
    ei = np.asarray(edge_index)
    if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.T
    A = np.zeros((n, n), dtype=np.float32)
    for s, d in zip(ei[0], ei[1]):
        s, d = int(s), int(d)
        A[s, d] = 1.0; A[d, s] = 1.0
    A = A + np.eye(n, dtype=np.float32)
    deg = A.sum(1)
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    return torch.FloatTensor((A * dinv).T * dinv)


def _subsample(data: dict, frac: float, seed: int):
    """Return (new_data, topo_report). frac>=1.0 returns data unchanged."""
    n = data["num_nodes"]
    if frac >= 1.0:
        keep = np.arange(n)
        return data, topology_report(data["edge_index"], _labels_np(data), keep)

    target = max(50, int(round(frac * n)))
    keep = mhrw_sample_nodes(data["edge_index"], n, target, seed=seed)
    topo = topology_report(data["edge_index"], _labels_np(data), keep)

    keep_map = {int(x): i for i, x in enumerate(keep)}
    ei = np.asarray(data["edge_index"])
    if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.T
    src, dst = [], []
    for s, d in zip(ei[0], ei[1]):
        s, d = int(s), int(d)
        if s in keep_map and d in keep_map and s != d:
            src.append(keep_map[s]); dst.append(keep_map[d])
    new_ei = np.array([src, dst]) if src else np.zeros((2, 0), dtype=int)

    n_sub = len(keep)
    new = dict(data)
    new["num_nodes"] = n_sub
    new["features"] = data["features"][keep]
    new["labels"]   = data["labels"][keep]
    new["edge_index"] = new_ei
    new["adj"] = _row_renormalize_adj(new_ei, n_sub)
    for mk in ("train_mask", "val_mask", "test_mask"):
        new[mk] = np.asarray(data[mk])[keep]
    return new, topo


def run_one(dataset, scenario, frac, seed):
    forget_ratio = (DEFAULT_CLIENT_FORGET_R if scenario == "client"
                    else DEFAULT_NODE_FORGET_R)

    # Monkeypatch load_dataset so the variations pipeline sees the subsample.
    topo_holder = {}
    original_load = ACU.load_dataset

    def patched_load(name, s):
        full = original_load(name, s)
        sub, topo = _subsample(full, frac, s)
        topo_holder.update(topo)
        print(f"   subsample frac={frac}: {topo['sub_nodes']} nodes, "
              f"avg_deg={topo['sub_avg_degree']:.2f}, "
              f"homophily={topo['sub_homophily']:.3f}, "
              f"Q_lbl={topo['sub_modularity_labels']:.3f}")
        return sub

    # V imports load_dataset by name, so patch it on the V module too.
    ACU.load_dataset = patched_load
    V.load_dataset = patched_load
    try:
        row = V.run_one(dataset, scenario, seed,
                        DEFAULT_PARTITIONING, None, forget_ratio)
    finally:
        ACU.load_dataset = original_load
        V.load_dataset = original_load

    row["fraction"] = frac
    row.update(topo_holder)   # attach subsample topology metrics
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets",  nargs="+", default=DATASETS)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    ap.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out-tag",   type=str, default="")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    csv_path = RAW_DIR / f"mia_subsample_raw{suffix}.csv"

    rows, done = [], set()
    if args.skip_existing and csv_path.exists():
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        for _, r in old.iterrows():
            done.add((r["dataset"], r["scenario"], float(r["fraction"]), int(r["seed"])))
        print(f"↺ resume: {len(rows)} rows")

    for ds, sc, fr, sd in product(args.datasets, args.scenarios,
                                  args.fractions, args.seeds):
        if (ds, sc, fr, sd) in done:
            print(f"↺ skip {ds}|{sc}|{fr}|{sd}"); continue
        try:
            rows.append(run_one(ds, sc, fr, sd))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   ✗ FAILED: {e}")
            rows.append({"dataset": ds, "scenario": sc, "fraction": fr,
                         "seed": sd, "error": str(e)})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n✅ Done → {csv_path}")


if __name__ == "__main__":
    main()
