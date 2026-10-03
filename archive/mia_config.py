"""
mia_config.py
=============
Configuration for the *MIA-is-not-all-you-need* consistency experiment.

Goal
----
Test whether membership-inference attacks (MIA) give consistent answers on
the gold-standard retrain-from-scratch model across seeds. If MIA varies
substantially on a model that is **by construction perfectly unlearned**,
then MIA cannot serve as the gold-standard evaluation for machine unlearning.

Scenarios
---------
- client : forget 20% of clients (gold = FedAvg retrain without them)
- node   : forget 10% of one target client's training nodes (gold = FedAvg
           retrain with those nodes excluded from target's train_mask)

Evaluation (matches existing pipelines)
---------------------------------------
- client : mean across all clients of forward(x, adj, client_id=c) on local
           test_mask  (== eval_mean_local_acc from the edge/feature pipeline)
- node   : forward(x, adj, client_id=target_idx) on TARGET CLIENT'S local
           test_mask only — the only client whose local view changed
"""
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
SEEDS    = [7, 42, 99, 100, 999]                # 5 seeds — central claim is variance
HP_SEEDS = [2024, 2025, 2026]                   # 3 seeds for attack-model HP search

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS_ALL = [
    "Cora", "PubMed", "CS", "Photo",
    "Tolokers", "minesweeper", "Amazon-ratings",
    "Computers", "WikiCS", "Chameleon", "Squirrel", "Actor", "roman_empire",
]

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS           = ["client", "node"]
CLIENT_FORGET_RATIO = 0.20                       # matches GFGU protocol
NODE_FORGET_RATIO   = 0.10                       # matches node-UL protocol

# ── Per-client train/val/test split ratios (partition-first protocol) ────────
DATASET_SPLIT_RATIOS = {
    "Cora":           (0.20, 0.40, 0.40),
    "PubMed":         (0.20, 0.40, 0.40),
    "CS":             (0.20, 0.40, 0.40),
    "Photo":          (0.20, 0.40, 0.40),
    "Tolokers":       (0.50, 0.25, 0.25),
    "minesweeper":    (0.50, 0.25, 0.25),
    "Amazon-ratings": (0.50, 0.25, 0.25),
    "Computers":    (0.20, 0.40, 0.40),
    "WikiCS":       (0.20, 0.40, 0.40),
    "Chameleon":    (0.48, 0.32, 0.20),
    "Squirrel":     (0.48, 0.32, 0.20),
    "Actor":        (0.50, 0.25, 0.25),
    "roman_empire": (0.50, 0.25, 0.25),
}

# ── Shokri shadow MIA ─────────────────────────────────────────────────────────
# Following csong27/membership-inference (Shokri et al., S&P 2017):
#   - N shadow FedAvg models, each on a random subset of clients
#   - For each shadow: train nodes -> member=1, test nodes -> non-member=0
#   - ONE attack model per data class (per-class design)
#   - Attack input = softmax probability vector
#   - Attack labels = {0: non-member, 1: member}
MIA_N_SHADOW    = 5     # Shokri default is 20 but FedAvg is expensive here
MIA_SHADOW_FRAC = 0.5   # fraction of clients each shadow trains on

# Default attack architecture (porting classifier.py)
#   - "nn"      : 1 hidden layer + tanh + softmax output + Adam + L2
#   - "softmax" : pure logistic regression (default in attack.py)
MIA_ATTACK_DEFAULTS = {
    "attack_type": "nn",
    "n_hidden":    50,
    "epochs":      100,
    "batch_size":  100,
    "lr":          0.01,
    "l2_ratio":    1e-7,
}

# Per-dataset HP grid for the attack model
#   2 (type) × 2 (hidden) × 2 (epochs) × 2 (lr) = 16 combos per dataset
MIA_HP_GRID = {
    "attack_type": ["softmax", "nn"],
    "n_hidden":    [50, 128],
    "epochs":      [50, 100],
    "lr":          [0.001, 0.01],
}

# ── Output layout ─────────────────────────────────────────────────────────────
ROOT = Path("mia_consistency_results")
DIRS = {
    "raw":    ROOT / "raw_data",
    "plots":  ROOT / "plots",
    "tables": ROOT / "tables",
    "hp":     ROOT / "hp_search",
}


def make_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
