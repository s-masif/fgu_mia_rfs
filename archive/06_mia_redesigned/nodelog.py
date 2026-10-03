"""Per-node logging (Minimal Plan Section 1.5 / PRD Section 4).

One row per queried node, written for both stages (M0 and MR), so that every
downstream endpoint -- forget-set transition, C_cal, C_F, member rates, FPR at
the fixed operating point -- can be recomputed offline without rerunning.

The statistical unit for analysis is the (dataset, seed) cell; these per-node
rows support the paired analyses within a cell.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def get_commit_hash():
    """Return the current git commit hash, or 'nogit' if not a git repo.
    Never raises: provenance logging must not break a run."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        h = out.stdout.strip()
        return h if h else "nogit"
    except Exception:
        return "nogit"


NODE_LOG_COLUMNS = [
    "dataset", "scenario", "seed",
    "global_node_id", "client_id",
    "stage",            # M0 | MR
    "subset",           # F | H | attack_member | attack_nonmember
    "pool_role",        # fit | calibration | evaluation
    "true_class",
    "membership_label", # supervised membership under this stage (1/0)
    "mia_score",
    "decision",
    "threshold",
    "fallback_status",
    "prob_vector",      # JSON list of the softmax used by the attack
]


class NodeLogger:
    """Accumulates per-node rows and writes them as a long-format table."""

    def __init__(self):
        self._rows = []

    def add(self, *, dataset, scenario, seed, global_node_id, client_id,
            stage, subset, pool_role, true_class, membership_label,
            mia_score, decision, threshold, fallback_status, prob_vector):
        self._rows.append({
            "dataset": dataset, "scenario": scenario, "seed": int(seed),
            "global_node_id": int(global_node_id), "client_id": int(client_id),
            "stage": stage, "subset": subset, "pool_role": pool_role,
            "true_class": int(true_class),
            "membership_label": int(membership_label),
            "mia_score": float(mia_score),
            "decision": int(decision),
            "threshold": float(threshold),
            "fallback_status": str(fallback_status),
            "prob_vector": json.dumps([round(float(p), 6) for p in prob_vector]),
        })

    def add_batch(self, *, global_ids, client_ids, stage, subset, pool_role,
                  true_classes, membership_labels, scores, decisions,
                  thresholds, fallback_statuses, prob_vectors,
                  dataset, scenario, seed):
        for k in range(len(global_ids)):
            self.add(dataset=dataset, scenario=scenario, seed=seed,
                     global_node_id=global_ids[k], client_id=client_ids[k],
                     stage=stage, subset=subset[k] if isinstance(subset, (list, np.ndarray)) else subset,
                     pool_role=pool_role,
                     true_class=true_classes[k], membership_label=membership_labels[k],
                     mia_score=scores[k], decision=decisions[k],
                     threshold=thresholds[k] if hasattr(thresholds, "__len__") else thresholds,
                     fallback_status=fallback_statuses[k] if isinstance(fallback_statuses, (list, np.ndarray)) else fallback_statuses,
                     prob_vector=prob_vectors[k])

    def to_frame(self):
        return pd.DataFrame(self._rows, columns=NODE_LOG_COLUMNS)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)


def write_manifest(path, *, dataset, scenario, seed, config,
                   removed_client_ids=None, forget_global_ids=None,
                   heldout_global_ids=None, alpha=0.10, thresholds=None,
                   commit_hash=None, split_ratios=None, counts=None,
                   fit_global_ids=None, calib_global_ids=None,
                   eval_global_ids=None):
    """One run-level manifest per (dataset, scenario, seed) cell."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset, "scenario": scenario, "seed": int(seed),
        "alpha": alpha,
        "removed_client_ids": list(map(int, removed_client_ids or [])),
        "forget_global_ids": list(map(int, forget_global_ids or [])),
        "heldout_global_ids": list(map(int, heldout_global_ids or [])),
        "frozen_thresholds": thresholds or {},
        "commit_hash": commit_hash,
        "split_ratios": split_ratios,
        "counts": counts or {},
        # pool identities for offline disjointness verification (R2/E5)
        "fit_global_ids": sorted(int(x) for x in (fit_global_ids or [])),
        "calib_global_ids": sorted(int(x) for x in (calib_global_ids or [])),
        "eval_global_ids": sorted(int(x) for x in (eval_global_ids or [])),
        "config": config,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest