"""
mia_hp_search.py
================
Per-dataset hyperparameter search for the Shokri shadow MIA attack model.

Goal
----
For each (dataset, scenario), find the strongest Shokri attack — best
(attack_type, n_hidden, epochs, lr) — using ONLY shadow data. The target
gold-standard model is held out and never seen during HP selection, which
preserves the integrity of the final MIA-consistency numbers.

Procedure
---------
For each (dataset, scenario) and each HP-search seed:
  1. Load data, partition, per-client split
  2. Train N shadow FedAvg models (shared across HP combos)
  3. 80/20 stratified split of the shadow attack-data
  4. For each HP combo: fit ShokriShadowMIA on the 80%, score the 20% via AUC
  5. Aggregate AUC over seeds; pick the combo with highest mean AUC

Output: best_attack_configs.json  (per-dataset, per-scenario HPs)
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from all_client_unlearning import (
    DATASET_CONFIGS, load_dataset, optimized_balanced_partitioning, set_seed,
)
from mia_config import (
    DATASETS_ALL, SCENARIOS, HP_SEEDS,
    MIA_N_SHADOW, MIA_SHADOW_FRAC, MIA_HP_GRID, MIA_ATTACK_DEFAULTS,
    DIRS, make_dirs,
)
from mia_attacks import ShokriShadowMIA
from mia_experiment import apply_per_client_split, train_shadow_models


def hp_search_one(dataset: str, scenario: str,
                  hp_seeds: list[int] | None = None,
                  hp_grid: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Search HPs for one (dataset, scenario).

    Note: The shadow training procedure is identical between scenarios — only
    the *target* differs. We still parameterize by scenario so per-dataset/
    per-scenario tuning is possible if attack difficulty varies.
    """
    hp_seeds = hp_seeds or HP_SEEDS
    hp_grid  = hp_grid  or MIA_HP_GRID
    cfg      = DATASET_CONFIGS[dataset]

    combos = list(itertools.product(
        hp_grid["attack_type"], hp_grid["n_hidden"],
        hp_grid["epochs"],      hp_grid["lr"],
    ))
    print(f"\n══ HP search: {dataset} | {scenario} "
          f"| {len(combos)} combos × {len(hp_seeds)} seeds ══")

    rows: list[dict] = []
    for seed in hp_seeds:
        set_seed(seed)
        data    = load_dataset(dataset, seed)
        clients = optimized_balanced_partitioning(data, cfg["num_clients"], seed)
        clients = apply_per_client_split(clients, dataset, seed=seed)

        # Shadow models — shared across HP combos for this seed
        shadow_probs, shadow_member, shadow_classes = train_shadow_models(
            cfg, data, clients,
            n_shadow=MIA_N_SHADOW, shadow_frac=MIA_SHADOW_FRAC,
            base_seed=12345 + seed,
        )

        # 80/20 stratified split by membership
        idx_tr, idx_va = train_test_split(
            np.arange(len(shadow_probs)),
            test_size=0.2, stratify=shadow_member, random_state=seed,
        )
        n_classes = int(data["num_classes"])

        for (atype, nh, ep, lr) in combos:
            hp = dict(attack_type=atype, n_hidden=nh, epochs=ep, lr=lr,
                      batch_size=100, l2_ratio=1e-7)
            atk = ShokriShadowMIA(**hp).fit(
                shadow_probs[idx_tr], shadow_member[idx_tr],
                shadow_classes[idx_tr], n_classes=n_classes,
            )
            pred, score, _ = atk.predict(shadow_probs[idx_va],
                                          shadow_classes[idx_va])
            val_y = shadow_member[idx_va]
            try:
                auc = float(roc_auc_score(val_y, score)) \
                       if len(np.unique(val_y)) == 2 else 0.5
            except ValueError:
                auc = 0.5
            acc = float(accuracy_score(val_y, pred))
            rows.append(dict(seed=seed, attack_type=atype, n_hidden=nh,
                             epochs=ep, lr=lr, val_auc=auc, val_acc=acc))
            print(f"   {atype:8s} h={nh:3d} ep={ep:3d} lr={lr:.4f} "
                  f"→ AUC={auc:.4f} acc={acc:.4f}  (seed={seed})")

    # Aggregate across seeds
    df = pd.DataFrame(rows)
    agg = (df.groupby(["attack_type", "n_hidden", "epochs", "lr"])
             .agg(val_auc_mean=("val_auc", "mean"),
                  val_auc_std =("val_auc", "std"),
                  val_acc_mean=("val_acc", "mean"))
             .reset_index()
             .sort_values("val_auc_mean", ascending=False))

    best_row = agg.iloc[0].to_dict()
    best_cfg = {
        "attack_type":  str(best_row["attack_type"]),
        "n_hidden":     int(best_row["n_hidden"]),
        "epochs":       int(best_row["epochs"]),
        "lr":           float(best_row["lr"]),
        "batch_size":   100,
        "l2_ratio":     1e-7,
        "_val_auc_mean":float(best_row["val_auc_mean"]),
        "_val_auc_std": float(best_row["val_auc_std"]) if not np.isnan(best_row["val_auc_std"]) else 0.0,
        "_val_acc_mean":float(best_row["val_acc_mean"]),
    }
    print(f"\n   ✅ Best: {best_cfg['attack_type']:8s} h={best_cfg['n_hidden']} "
          f"ep={best_cfg['epochs']} lr={best_cfg['lr']}  "
          f"AUC={best_cfg['_val_auc_mean']:.4f}±{best_cfg['_val_auc_std']:.4f}")

    return best_cfg, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", default=DATASETS_ALL,
                        choices=DATASETS_ALL)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        choices=SCENARIOS)
    parser.add_argument("--seeds",     nargs="+", type=int, default=HP_SEEDS)
    args = parser.parse_args()

    make_dirs()

    best_configs: dict = {}
    for ds in args.datasets:
        for sc in args.scenarios:
            best, df = hp_search_one(ds, sc, hp_seeds=args.seeds)
            best_configs[f"{ds}_{sc}"] = best
            df.to_csv(DIRS["hp"] / f"hp_search_{ds}_{sc}.csv", index=False)

    out = DIRS["hp"] / "best_attack_configs.json"
    with open(out, "w") as f:
        json.dump(best_configs, f, indent=2)
    print(f"\n✅ Best configs saved → {out}")


if __name__ == "__main__":
    main()
