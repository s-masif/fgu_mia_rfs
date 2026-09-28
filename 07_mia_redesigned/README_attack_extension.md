# Attack Extension: Modified Entropy + Offline RMIA

This adds two membership attacks (modified prediction entropy, offline RMIA) on top
of the frozen client-level results, to test the paper's claim under attacks other
than the original Shokri shadow attack. It follows the plan: keep the validated M0/MR runs frozen, train new reference models, and
score the new attacks offline.

## Design principle

- **M0/MR are never retrained.** The target-model probabilities come from the
  frozen node logs of the validated 65-cell run. Nothing on the target side changes.
- **New reference models** are trained fresh, with explicit fixed seeds, to serve as
  the "not trained on it" baseline the new attacks need. They are trained without
  F/H (and the other evaluation nodes), so they are valid OUT models.
- **Scoring is offline.** Entropy and RMIA are computed from the frozen target
  outputs plus the reference outputs. No training happens at scoring time.

The extension is two stages: build the reference outputs once (needs GPU), then
score as many times as needed (pure CPU/offline).

## Files

```
attack_extension/
├── build_references.py   Stage 1: train reference models, save their outputs (per cell)
├── score_and_match.py    Stage 2: compute entropy + RMIA, save per-node scores, AUCs
├── test_extension.py     Automated acceptance tests (integrity + diagnostics)
├── check_refpack.py      Lighter refpack consistency checker
└── README.md
```

## Stage 1 — build_references.py

For each cell it: rebuilds the same seeded partition (no M0/MR training), reads the
frozen node identities and calibration ids, defines the RMIA population, trains 5
reference models with explicit fixed seeds (`ref_seed_base + i`) excluding F/H from
their training, and saves everything to `refpack_<tag>.json`.

The refpack contains:
- `reference_probs` — the 5 reference models' softmax on F, H, attack_member,
  attack_nonmember, the calibration nodes, and the population pool.
- `node_groups` — the node-id list of each evaluation group.
- `calibration_ids`, `calibration_labels` — calibration pool ids and their labels.
- `population_ids`, `final_eval_nonmember_ids` — the deterministic disjoint split of
  the retained non-members (population for RMIA; final-eval for evaluation).
- `ref_seed_base`, `reference_train_ids`, `out_status_leak` — for regenerating the
  reference models and verifying their OUT status.

Population definition (reviewer's requirement): a deterministic subset of the
retained non-members, disjoint from the calibration nodes and from the final
evaluation nodes.

Run:
```bash
python build_references.py --datasets Cora --seeds 42 \
    --frozen-dir results_13x5/raw --out-dir attack_ext_out
```

## Stage 2 — score_and_match.py

Reads the frozen node log (target M0/MR softmax, true labels, Shokri per-node
scores) and the refpack, and computes both attacks under M0 and MR.

**Modified entropy (Song & Mittal 2021).**
- Score: `Mentr(p, y) = -(1 - p_y) log(p_y) - Σ_{i≠y} p_i log(1 - p_i)` from the
  target-model softmax. Lower = more member-like.
- Threshold: per class at the α = 0.10 operating point, computed from the saved
  calibration outputs (mean reference softmax on the calibration nodes, with their
  labels). Classes with fewer than `MIN_CALIB = 10` calibration nodes use a pooled
  threshold — the same sparse-class fallback used for the Shokri attack. The
  fallback classes are recorded.

**Offline RMIA (Zarifzadeh et al. 2024).**
- Ratio: `Pr(x|target) / mean_over_reference_models Pr(x|reference)`, at the true
  label.
- Score: fraction of the population `z` with `ratio_x / ratio_z ≥ γ` (γ = 2).
- Non-member evaluation and thresholding use the `final_eval_nonmember_ids` set,
  which is disjoint from the population `z`.

**Cross-attack AUC (comparison only).** All three attacks (Shokri, entropy, RMIA)
are also evaluated on one common final pool = `attack_member + final_eval_nonmember`
(the RMIA population excluded from evaluation), so their AUCs are directly
comparable. The frozen Shokri full-pool AUC from the core results is kept separately
(reported from `summary_core.csv`), not overwritten.

Outputs written under `attack_ext_out/scores/`:
- `nodescores_<tag>.csv` — one row per node per stage: entropy score/threshold/
  decision/fallback-flag, and RMIA score/threshold/decision.
- `attackmeta_<tag>.json` — per-attack thresholds, entropy fallback classes, the
  RMIA threshold/gamma/population sizes, the common-pool AUCs, and the frozen Shokri
  full-pool AUC reference.

Run:
```bash
python score_and_match.py --tag Cora_client_s42 \
    --frozen-dir results_13x5/raw --out-dir attack_ext_out
```

## Testing — test_extension.py

Automated acceptance tests, PASS/FAIL for integrity and diagnostics reported:

Integrity checks:
- node identities match the frozen node log for all four groups;
- reference OUT-status leak is 0 (no evaluation node in any reference training set);
- population is disjoint from calibration and from final-eval, ⊆ attack_nonmember,
  and population ∪ final_eval covers attack_nonmember minus calibration;
- per-node scores saved for both stages, both attacks, no NaN entropy scores;
- RMIA non-member evaluation count equals the final-eval size, and the population is
  excluded from RMIA evaluation (disjointness in scoring);
- entropy fallback classes recorded, and the per-node fallback flag matches them;
- calibration labels and reference seeds saved;
- common-pool AUC present for all three attacks;
- entropy per-class thresholds finite.

Run:
```bash
python test_extension.py --tag Cora_client_s42 \
    --frozen-dir results_13x5/raw --out-dir attack_ext_out
```

## Reproducibility and consistency notes

- The reference models are reproducible from `ref_seed_base` (model i uses
  `ref_seed_base + i`), independent of the M0/MR training history, because an
  explicit seed is set immediately before each reference model.
- Scoring is deterministic (verified: identical AUCs across re-runs).
- The common-pool AUC for all three attacks is computed on exactly the same node
  set (member + final_eval), so the comparison is fair; Shokri's comparison AUC is
  recomputed offline from its saved per-node scores in the frozen node log, while
  its original full-pool AUC is left untouched in the core results.
- Small datasets: after splitting attack_nonmember into population + final-eval, the
  final-eval pool can be small (e.g. ~20 nodes on Chameleon), so per-cell AUC/threshold
  there are noisier; report the per-cell sizes.

## Extending to all cells

After the Cora seed 42 check is approved, run the two stages across all 13 datasets
× 5 seeds. Stage 1 trains 5 reference models per cell (the only training; M0/MR stay
frozen); Stage 2 scores offline. The test suite should be run per cell.
