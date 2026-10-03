"""
mia_control_config.py
=====================
Configuration for the Phase 1 CONTROL experiment.

Purpose (following the professor's request verbatim):

    "Start with the control: same attack, but replace the forget set with
     nodes the model never saw, from the same distribution as the retain
     set. If the AUC stays high, the false positive is proven."

Design:
  • Three-way split at partition time: retain, forget, held-out (H).
  • Train M_ret_control on retain only (both forget and H are unseen).
  • Same Shokri MIA attack applied to M_ret_control.
  • Two AUCs computed on the same trained model:
        - Original AUC : query pool = {retain, test, forget}
        - Control AUC  : query pool = {retain, test, H}
  • If Original ≈ Control (both high), the false positive is proven.

Scope:
  4 datasets × 2 scenarios × 3 seeds = 24 experimental cells.
"""
from pathlib import Path

# ── Reproducibility ─────────────────────────────────────────────────────────
SEEDS = [7, 42, 99]

# ── The 4 interesting datasets (high MIA AUC in the main study) ─────────────
# DATASETS_ALL = ["Cora", "Chameleon", "Squirrel", "Actor"]

DATASETS_ALL = [
    "Cora", "PubMed", "CS", "Photo",
    "tolokers", "minesweeper", "amazon_ratings",
    # ── NEW (must match keys in all_client_unlearning.py DATASET_CONFIGS) ──
    "Computers", "WikiCS",
    "Chameleon", "Squirrel", "Actor", "roman_empire",
]

# ── Unlearning scenarios ────────────────────────────────────────────────────
SCENARIOS = ["client", "node"]

# ── Ratios ──────────────────────────────────────────────────────────────────
# Forget size stays what it was in the main study.
# Held-out size = forget size (matched control).
CLIENT_FORGET_RATIO   = 0.20
CLIENT_HELDOUT_RATIO  = 0.20        # matched
NODE_FORGET_RATIO     = 0.10
NODE_HELDOUT_RATIO    = 0.10        # matched

# ── Per-client split ratios (matches main study) ────────────────────────────
DATASET_SPLIT_RATIOS = {
    "Cora":      (0.20, 0.40, 0.40),
    "Chameleon": (0.48, 0.32, 0.20),
    "Squirrel":  (0.48, 0.32, 0.20),
    "Actor":     (0.50, 0.25, 0.25),
}

# ── Shokri shadow MIA (same as main study) ──────────────────────────────────
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

# ── Output layout ───────────────────────────────────────────────────────────
ROOT = Path("mia_control_results")
DIRS = {
    "raw":    ROOT / "raw_data",
    "eval":   ROOT / "evaluation",
    "plots":  ROOT / "plots",
}


def make_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
