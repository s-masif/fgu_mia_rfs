from pathlib import Path

# ── Reproducibility ─────────────────────────────────────────────────────────
SEEDS = [7, 42, 99]

# ── The 4 interesting datasets (high MIA AUC in the main study) ─────────────
DATASETS_ALL = ["Cora", "Chameleon", "Squirrel", "Actor"]

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
