# Attack Extension: Modified Entropy + Offline RMIA

This adds two membership attacks (modified prediction entropy, offline RMIA) on top
of the frozen client-level results, to test the paper's claim under attacks other
than the original Shokri shadow attack. It follows the plan agreed with the
reviewer: keep the validated M0/MR runs frozen, train new reference models, and
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

### nodescores_<tag>.csv — one row per node per stage
Structurally consistent with the frozen Shokri node log (same identity, ground-truth
and diagnostic columns), with the single Shokri attack columns replaced by per-attack
entropy and RMIA columns.

Identity / provenance:
- `dataset`, `scenario`, `seed` — the cell this row belongs to.
- `gid` — global node id (same as `global_node_id` in the frozen log).
- `client_id` — the client the node belongs to.
- `stage` — M0 (original model) or MR (retrained model).
- `subset` — F, H, attack_member, or attack_nonmember.
- `pool_role` — evaluation (all scored nodes are evaluation nodes).

Ground truth:
- `true_class` — the node's true label.
- `membership_label` — supervised membership under this stage (F=1 at M0 and 0 at MR;
  H=0 at both; attack_member=1; attack_nonmember=0).

Structural / difficulty diagnostics (same as the frozen log, for confound checks):
- `local_degree` — degree within the node's client subgraph.
- `local_label_agreement` — fraction of client-local neighbours sharing the label.
- `correct_prediction` — 1 if the target model classifies the node correctly.

Modified-entropy attack:
- `entropy_score` — modified prediction entropy from the target-model softmax
  (lower = more member-like).
- `entropy_threshold` — the per-class decision threshold applied to this node.
- `entropy_decision` — 1 if predicted member (entropy_score < threshold).
- `entropy_fallback` — 1 if this node's class used the pooled fallback threshold
  (too few calibration nodes for its own).

Offline-RMIA attack (only on F, H, attack_member, and the final-eval non-members —
the RMIA population is excluded from evaluation):
- `rmia_ratio` — the raw likelihood ratio Pr(x|target)/mean_ref Pr(x|reference) at
  the true label (the core RMIA quantity, before the population comparison).
- `rmia_score` — fraction of the population with ratio_x/ratio_z >= gamma.
- `rmia_threshold` — the decision threshold (set at the operating point on the
  final-eval non-members).
- `rmia_decision` — 1 if predicted member (rmia_score > threshold).
  (RMIA columns are blank for nodes RMIA does not evaluate.)

### attackmeta_<tag>.json — per-cell attack metadata
- `tag`, `dataset`, `seed`, `alpha`, `gamma` — cell + operating-point settings.
- `ref_seed_base`, `n_reference_models` — for regenerating the reference models.
- `calibration_labels_ref` — note that calibration labels are stored in the refpack.
- `entropy_meta` (per stage): `pooled_tau`, `per_class_tau`, `fallback_classes`,
  `min_calib`.
- `rmia_meta` (per stage): `threshold`, `gamma`, `n_population`,
  `n_final_eval_nonmember`, and `population_ratios` (the population's raw ratios, so
  RMIA can be re-scored at a different gamma offline without re-running).
- `common_pool_auc` — the cross-attack AUC of shokri / entropy / rmia on the common
  final pool (attack_member + final_eval_nonmember).
- `frozen_shokri_full_pool_auc` — the original Shokri full-pool AUC from the core
  results, kept separately for reference (not overwritten).

### refpack_<tag>.json — the reference-model outputs (produced by build_references.py)
- `reference_probs` — list of the 5 reference models' softmax vectors on F, H,
  attack_member, attack_nonmember, calibration, and population nodes.
- `node_groups` — the node-id list of each evaluation group.
- `calibration_ids`, `calibration_labels` — calibration pool ids and their true labels.
- `population_ids`, `final_eval_nonmember_ids` — the deterministic disjoint split of
  the retained non-members.
- `ref_seed_base` — base seed; reference model i uses ref_seed_base + i.
- `reference_train_ids` — the nodes each reference model trained on (to verify OUT
  status).
- `out_status_leak` — number of evaluation nodes that leaked into any reference
  training set (must be 0).
- `dataset`, `seed`, `alpha`, `n_reference_models` — cell + settings.

The frozen node log (`results_13x5/raw/nodelog_<tag>.csv`) remains the source of the
target M0/MR softmax, true labels, and the Shokri per-node scores; it is not
modified by this extension.

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


## Reference tables — every score/field, what it means, why it is needed

### nodescores_<tag>.csv  (one row per node per stage)

| Field | What it is | Why it is needed |
|---|---|---|
| dataset, scenario, seed | which cell this row belongs to | makes each file self-describing; lets rows be pooled across cells |
| gid | global node id (= global_node_id in the frozen log) | joins to the frozen node log and across attacks |
| client_id | the client the node belongs to | per-client analysis; confirms F/H come from the forgotten client |
| stage | M0 (original) or MR (retrained) | the two models the frozen attack was run on; the transition is M0 -> MR |
| subset | F / H / attack_member / attack_nonmember | the role of the node in the audit |
| pool_role | evaluation | confirms only evaluation nodes are scored (not fit/calibration) |
| true_class | the node's true label | needed by both attacks (entropy and RMIA use the true-label probability) |
| membership_label | supervised membership under this stage (F=1@M0,0@MR; H=0 both; member=1; nonmember=0) | ground truth for computing rates and AUC |
| local_degree | degree within the node's client subgraph | check whether an F/H difference is structural, not membership |
| local_label_agreement | fraction of client-local neighbours sharing the label (homophily) | sharper structural confound check than degree |
| correct_prediction | 1 if the target model classifies the node correctly | check whether an F/H difference is a difficulty effect, not membership |
| entropy_score | target-model modified prediction entropy (lower = more member-like) | the entropy attack's per-node signal |
| entropy_threshold | per-class decision threshold applied to this node | records the operating point actually used |
| entropy_decision | 1 if predicted member (entropy_score < threshold) | the entropy attack's per-node call |
| entropy_fallback | 1 if this node's class used the pooled fallback threshold | flags sparse-class calibration, for honest reporting |
| rmia_ratio | raw Pr(x|target)/mean_ref Pr(x|reference) at the true label | the core RMIA quantity; lets RMIA be re-scored at any gamma offline |
| rmia_score | fraction of the population with ratio_x/ratio_z >= gamma | the RMIA attack's per-node signal |
| rmia_threshold | RMIA decision threshold at the operating point | records the operating point used |
| rmia_decision | 1 if predicted member (rmia_score > threshold) | the RMIA attack's per-node call |

Note: RMIA columns are populated only for nodes RMIA evaluates (F, H, attack_member,
final-eval non-members); RMIA population nodes are excluded from evaluation.

### attackmeta_<tag>.json  (per-cell attack metadata)

| Field | What it is | Why it is needed |
|---|---|---|
| tag, dataset, seed | cell identity | bookkeeping |
| alpha | operating point (false-positive target = 0.10) | the shared operating point of all three attacks |
| gamma | RMIA domination threshold (= 2, paper default) | the RMIA hyperparameter; recorded for reproducibility |
| ref_seed_base | base seed for the reference models (model i uses base+i) | regenerate the exact reference models |
| n_reference_models | number of reference models (= 5) | reproducibility; RMIA uses all five |
| entropy_meta[stage].pooled_tau | pooled entropy threshold | the fallback threshold value used |
| entropy_meta[stage].per_class_tau | per-class entropy thresholds | audit the calibration per class |
| entropy_meta[stage].fallback_classes | classes that used the pooled fallback | honest record of sparse-class calibration |
| entropy_meta[stage].min_calib | minimum calibration nodes for own threshold (= 10) | the sparse-class rule, same as Shokri |
| rmia_meta[stage].threshold | RMIA decision threshold at alpha | operating point used |
| rmia_meta[stage].gamma | RMIA gamma (= 2) | reproducibility |
| rmia_meta[stage].n_population | number of population nodes (z) | audit the population size (small on some datasets) |
| rmia_meta[stage].n_final_eval_nonmember | number of final-eval non-members | audit the evaluation pool size |
| rmia_meta[stage].population_ratios | the population's raw ratios | re-score RMIA at any gamma offline without re-running |
| common_pool_auc | AUC of shokri/entropy/rmia on the common final pool | fair cross-attack strength comparison |
| frozen_shokri_full_pool_auc | Shokri's original full-pool AUC from the core results | kept separately; the extension never recomputes/overwrites it |

### Derived endpoints (computed from the above; reported per cell)

| Quantity | Definition | Why it matters |
|---|---|---|
| forget member rate (M0, MR) | fraction of F predicted member under each model | the known membership transition (member at M0 -> non-member at MR) |
| held-out member rate (M0, MR) | fraction of H predicted member under each model | H is a non-member in both; a change here is signal moving without a membership change |
| C_cal = FPR(H;MR) - alpha | how far the calibrated operating point transfers to the removed client | measures whether the absolute signal is calibrated across settings |
| C_F = FPR(F;MR) - FPR(H;MR) | how far the forget set departs from a matched non-member reference under MR | the main endpoint: ~0 means F is indistinguishable from a matched non-member after retraining |
| attack strength AUC (common pool) | member vs non-member AUC on attack_member + final_eval_nonmember | how strong each attack is, on one identical pool for all three |

Note on Shokri and C_cal/C_F: C_cal and C_F are defined on the F and H sets and come
from each attack's own F/H scoring (Shokri's are from the frozen core run; entropy and
RMIA are computed by this extension on the same F/H nodes). The common final pool is
only for the AUC strength comparison (member vs non-member), so C_cal/C_F are not
defined on it.


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