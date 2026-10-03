"""
mia_variations_config.py
========================
Configuration for Phase 2 — the variation sweep.

The prof asked us to build up more data points by varying:
  - Partitioning
  - Number of clients
  - Forget ratio

And to include as controls (measured, not varied):
  - Dataset size
  - Train-to-test accuracy gap

Sweep strategy: vary ONE axis at a time, holding others at default.
Grid values below are our choice — no specific values were prescribed.
"""
from pathlib import Path

# ── Base setup (matches the control experiment scope) ──────────────────────
SEEDS        = [7, 42, 99, 100, 999]
# DATASETS_ALL = ["Cora", "Chameleon", "Squirrel", "Actor"]

# DATASETS_ALL = [
#     "Cora", "PubMed", "CS", "Photo",
#     "tolokers", "minesweeper", "amazon_ratings",
#     # ── NEW (must match keys in all_client_unlearning.py DATASET_CONFIGS) ──
#     "Computers", "WikiCS",
#     "Chameleon", "Squirrel", "Actor", "roman_empire",
# ]

# DATASETS_ALL = [
#     "Cora", "PubMed", "Photo",
#     "tolokers", "minesweeper",
#     # ── NEW (must match keys in all_client_unlearning.py DATASET_CONFIGS) ──
#     "Chameleon", "Squirrel", "Actor", "roman_empire", "amazon_ratings"
# ]

DATASETS_ALL = [
    "Cora",
    "PubMed",
    "Photo",
    "tolokers",
    "minesweeper",
    "amazon_ratings",
    "Computers",
    "WikiCS",
    "Chameleon",
    "Squirrel",
    "Actor",
    "roman_empire",
]


SCENARIOS    = ["client", "node"]

# ── Per-client split ratios (from the main study) ──────────────────────────
DATASET_SPLIT_RATIOS = {
    "Cora":      (0.20, 0.40, 0.40),
    "Chameleon": (0.48, 0.32, 0.20),
    "Squirrel":  (0.48, 0.32, 0.20),
    "Actor":     (0.50, 0.25, 0.25),
}

# ── Shokri shadow MIA (same as main study) ─────────────────────────────────
MIA_N_SHADOW    = 5
MIA_SHADOW_FRAC = 0.5

MIA_ATTACK_DEFAULTS = {
    "attack_type": "nn",
    "n_hidden":    50,
    "epochs":      100,
    "batch_size":  100,
    "lr":          0.01,
    "l2_ratio":    1e-7,
}

# ═══════════════════════════════════════════════════════════════════════════
#  Sweep grids  (one axis varied at a time, others at default)
# ═══════════════════════════════════════════════════════════════════════════

# ── Defaults (used when an axis is not being swept) ─────────────────────────
DEFAULT_PARTITIONING     = "louvain"
DEFAULT_NUM_CLIENTS      = None            # None → use DATASET_CONFIGS[ds]["num_clients"]
DEFAULT_CLIENT_FORGET_R  = 0.20
DEFAULT_NODE_FORGET_R    = 0.10

# ── Sweep 1: Partitioning ───────────────────────────────────────────────────
# 4 partitioners × 4 datasets × 2 scenarios × 3 seeds = 96 cells
PARTITIONING_GRID = ["louvain", "metis", "metis_plus", "random"]

# ── Sweep 2: Number of clients ─────────────────────────────────────────────
# 4 K values × 4 datasets × 2 scenarios × 3 seeds = 96 cells
NUM_CLIENTS_GRID = [3, 5, 10, 15]

# ── Sweep 3: Forget ratio ──────────────────────────────────────────────────
# Client scenario (forget_ratio applies to fraction of clients):
CLIENT_FORGET_RATIO_GRID = [0.10, 0.20, 0.30, 0.40]
# Node scenario (forget_ratio applies to fraction of target's training nodes):
NODE_FORGET_RATIO_GRID   = [0.05, 0.10, 0.20, 0.30]
# 4 ratios × 4 datasets × 2 scenarios × 3 seeds = 96 cells

# ── Total: 96 × 3 = ~288 cells (some overlap across sweeps at default) ────


# ═══════════════════════════════════════════════════════════════════════════
#  Homophily (raw edge homophily) — control variable for correlation analysis
# ═══════════════════════════════════════════════════════════════════════════

DATASET_HOMOPHILY = {
    "Cora":      0.81,
    "Chameleon": 0.23,
    "Squirrel":  0.22,
    "Actor":     0.22,
}

# ── Output layout ──────────────────────────────────────────────────────────
ROOT = Path("mia_variations_results")
DIRS = {
    "raw":   ROOT / "raw_data",
    "eval":  ROOT / "evaluation",
    "plots": ROOT / "plots",
}


def make_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
