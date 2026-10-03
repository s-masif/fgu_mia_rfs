# Attacks and Logs: Shokri (core) + Modified Entropy + Offline RMIA

This is the deep technical reference for the three membership inference attacks used in
the paper and every log they write. The **Shokri shadow-model attack** is the original
core attack, run inside the main experiment; the **modified prediction-entropy** and
**offline RMIA** attacks are added as an offline extension on top of the frozen core run,
to test the paper's claim under attacks of different construction.

For the project overview, headline results, figures, and quick reproduce commands, see the
root `README.md`.

- [Core attack: Shokri shadow-model](#core-attack-shokri-shadow-model-stage-1)
- [Extension design principle](#extension-design-principle)
- [Extension files](#extension-files-active-pipeline)
- [Stage 1 — build_references.py](#stage-1--build_referencespy)
- [Stage 2 — score_and_match.py](#stage-2--score_and_matchpy)
- [Stage 2b — rethreshold_rmia.py](#stage-2b--rethreshold_rmiapy-review-correction)
- [Testing](#testing--test_extensionpy)
- [Full run](#full-run-over-all-13x5-cells--run_full_extensionsh)
- [Aggregation](#aggregation--aggregate_extensionpy)
- [Figures](#figures--make_combined_figurespy)
- [Results summary](#results-summary-full-13x5-three-attacks)
- [Reference tables (every field)](#reference-tables--every-scorefield-what-it-means-why-it-is-needed)
- [Reproducibility notes](#reproducibility-and-consistency-notes)
- [End-to-end reproduction](#end-to-end-reproduction)

---

## Core attack: Shokri shadow-model (Stage 1)

The Shokri attack (`attack.py`, driven by `experiment_main.py`) is the original,
learned membership inference attack. Shadow models are trained with the same federated
procedure as the target, on subsets of the retained clients, to imitate the target
model; a per-class classifier is then trained to distinguish members from non-members
from their prediction vectors. The decision threshold is set out-of-sample on a held-out
pool of shadow non-members at the operating point α = 0.10 and then **frozen**, so M0 and
MR are read with the same threshold. Classes with too few calibration nodes
(`min_calib = 10`) use a pooled fallback threshold, recorded per cell.

The core run (`experiment_main.py`, Stage 1 of the pipeline) is the source of truth for
everything downstream: it trains M0 and MR, runs the frozen Shokri attack on
F, H, attack_member, attack_nonmember under both models, and writes the per-node log and
run manifest to `results_13x5/raw/`, plus the per-cell Shokri summary `summary_core.csv`.
The extension attacks (entropy, RMIA) read the target-model probabilities and node
identities straight from these frozen logs and never retrain M0/MR.

### results_13x5/raw/nodelog_&lt;tag&gt;.csv — core Shokri per-node log

Column order:
```
dataset, scenario, seed, global_node_id, client_id, stage, subset, pool_role,
true_class, membership_label, mia_score, decision, threshold, fallback_status,
local_degree, local_label_agreement, correct_prediction, prob_vector
```

| Field | What it is | Why it is needed |
|---|---|---|
| dataset, scenario, seed | which cell this row belongs to | makes each file self-describing; lets rows be pooled across cells |
| global_node_id | global node id (= `gid` in the extension logs) | the join key across logs and attacks |
| client_id | the client the node belongs to | per-client analysis; confirms F/H come from the forgotten client |
| stage | M0 (original) or MR (retrained) | the two models the frozen attack was run on |
| subset | F / H / attack_member / attack_nonmember | the role of the node in the audit |
| pool_role | evaluation / fit / calibration | separates scored evaluation nodes from the attack's own fit/calibration nodes |
| true_class | the node's true label | ground truth; also used by the extension attacks |
| membership_label | supervised membership under this stage (F=1@M0,0@MR; H=0 both; member=1; nonmember=0) | ground truth for rates and AUC |
| mia_score | the Shokri attack's per-node membership score | the core attack's signal |
| decision | 1 if predicted member at the frozen threshold | the core attack's per-node call |
| threshold | the (per-class) decision threshold applied to this node | records the frozen operating point used |
| fallback_status | whether this node's class used the pooled fallback threshold | flags sparse-class calibration, for honest reporting |
| local_degree | degree within the node's client subgraph | structural confound check |
| local_label_agreement | fraction of client-local neighbours sharing the label | sharper structural confound check |
| correct_prediction | 1 if the target model classifies the node correctly | difficulty confound check |
| prob_vector | the target model's full softmax vector for the node | lets the entropy/RMIA attacks be computed offline from the frozen outputs |

The `prob_vector` column is what makes the offline extension possible: the entropy and
RMIA attacks recompute their scores from these stored target probabilities, so M0/MR are
never retrained.

### results_13x5/raw/manifest_&lt;tag&gt;.json — per-cell run manifest

Keys: `dataset`, `scenario`, `seed`, `alpha`, `removed_client_ids`, `forget_global_ids`,
`heldout_global_ids`, `frozen_thresholds`, `commit_hash`, `split_ratios`, `counts`,
`attack_settings`, `fit_global_ids`, `calib_global_ids`, `eval_global_ids`, `config`.

| Field | What it is | Why it is needed |
|---|---|---|
| alpha | operating point (= 0.10) | the shared false-positive target |
| removed_client_ids | the clients in the forget set D | records which clients were removed |
| forget_global_ids / heldout_global_ids | the node ids of F and H | exact definition of the forget set and matched reference |
| frozen_thresholds | the frozen per-class Shokri thresholds | the operating point actually applied to M0 and MR |
| commit_hash | git SHA of the code that produced the run | provenance / reproducibility |
| split_ratios | train/val/test split ratios | records the data split |
| counts | client and calibration counts (see below) | audit of configured vs realized clients and calibration support |
| attack_settings | the Shokri hyper-parameters (see below) | records the attack configuration |
| fit_global_ids / calib_global_ids / eval_global_ids | node ids used to fit, calibrate, and evaluate the attack | identity-disjointness audit (no evaluation node in fit/calibration) |
| config | the per-dataset model/training config | records hyper-parameters for the cell |

`counts` records, for example:
`{n_forget_clients, n_retain_clients, n_clients_configured, n_clients_realized,
fallback_classes, calib_support}` — so configured vs realized client counts and per-class
calibration support are auditable (this is where the Squirrel/Tolokers realized-count
difference is recorded).

`attack_settings` records the Shokri configuration:
`{n_shadow: 5, shadow_frac: 0.5, calib_frac: 0.3, eval_member_frac: 0.3,
eval_nonmember_frac: 0.3, min_calib: 10, alpha: 0.1}`.

### summary_core.csv — per-cell Shokri summary

Column order:
```
dataset, scenario, seed,
forget_member_rate_M0, forget_member_rate_MR, forget_mean_score_M0, forget_mean_score_MR,
heldout_member_rate_M0, heldout_member_rate_MR, heldout_mean_score_M0, heldout_mean_score_MR,
C_cal, C_F, attack_strength_auc, attack_tpr_at_alpha, attack_fpr_at_alpha,
acc_train, acc_test, train_test_gap, alpha, n_fallback_classes
```

| Field | What it is |
|---|---|
| forget_member_rate_M0/MR | fraction of F predicted member under each model |
| forget_mean_score_M0/MR | mean Shokri score on F under each model |
| heldout_member_rate_M0/MR | fraction of H predicted member under each model |
| heldout_mean_score_M0/MR | mean Shokri score on H under each model |
| C_cal | FPR(H;MR) − α |
| C_F | FPR(F;MR) − FPR(H;MR) |
| attack_strength_auc | Shokri attack strength (reported on the common pool after the extension's harmonisation) |
| attack_tpr_at_alpha / attack_fpr_at_alpha | realized TPR/FPR at the operating point |
| acc_train / acc_test / train_test_gap | retrained-model accuracies and their gap |
| alpha | operating point (0.10) |
| n_fallback_classes | number of classes that used the pooled fallback threshold |

Note: the Shokri `attack_strength_auc` reported in the paper is the common-pool AUC
(E⁺ ∪ E⁻), harmonised with entropy and RMIA by the extension; the original full-pool
Shokri AUC is preserved separately in each cell's `attackmeta` as
`frozen_shokri_full_pool_auc`.

---

## Extension design principle

- **M0/MR are never retrained.** The target-model probabilities come from the frozen node
  logs above (`prob_vector`). Nothing on the target side changes.
- **New reference models** are trained fresh, with explicit fixed seeds, to serve as the
  "not trained on it" baseline the new attacks need. They exclude F/H (and the other
  evaluation nodes), so they are valid OUT models.
- **Scoring is offline.** Entropy and RMIA are computed from the frozen target outputs
  plus the reference outputs. No training happens at scoring time.

The extension is two stages: build the reference outputs once (needs GPU), then score as
many times as needed (offline). A subsequent offline re-thresholding step
(`rethreshold_rmia.py`) applies the review correction to RMIA without any rescoring.

## Extension files (active pipeline)

```
fgu_mia/
├── build_references.py       Stage 1: train reference models, save their outputs (per cell)
├── score_and_match.py        Stage 2: compute entropy + RMIA, save per-node scores, AUCs
├── rethreshold_rmia.py       Stage 2b: freeze RMIA threshold on M0, apply to M0 and MR
├── test_extension.py         Automated acceptance tests (integrity + diagnostics)
├── run_full_extension.sh     Driver: runs Stage 1 + Stage 2 over all 13x5 cells, logs, spot-checks
├── aggregate_extension.py    Rolls the 65 per-cell outputs into summary_entropy.csv / summary_rmia.csv
├── make_combined_figures.py  Builds the paper's combined figures from the summaries
└── README_attack_extension.md
```

Superseded helper scripts (`check_refpack.py`, `make_impact_figures.py`,
`make_multi_attack_figures.py`, `impact_with_real_homophily.py`, `impact_analysis_stats.py`,
`print_all_stats.py`) are kept under `legacy/` for provenance and are not part of the
current pipeline.

## Stage 1 — build_references.py

For each cell it: rebuilds the same seeded partition (no M0/MR training), reads the frozen
node identities and calibration ids, defines the RMIA population, trains 5 reference
models with explicit fixed seeds (`ref_seed_base + i`) excluding F/H from their training,
and saves everything to `refpack_<tag>.json`.

The refpack contains: `reference_probs` (the 5 reference models' softmax on F, H,
attack_member, attack_nonmember, calibration, and population nodes), `node_groups`,
`calibration_ids`, `calibration_labels`, `population_ids`, `final_eval_nonmember_ids`
(the deterministic disjoint split of the retained non-members), `ref_seed_base`,
`reference_train_ids`, and `out_status_leak`.

Population definition (reviewer's requirement): a deterministic subset of the retained
non-members, disjoint from the calibration nodes and from the final evaluation nodes.

Run (single cell):
```bash
python build_references.py --datasets Cora --seeds 42 \
    --frozen-dir results_13x5/raw --out-dir attack_ext_out --ref-seed-base 90000
```

## Stage 2 — score_and_match.py

Reads the frozen node log (target M0/MR softmax via `prob_vector`, true labels, Shokri
per-node scores) and the refpack, and computes both attacks under M0 and MR.

**Modified entropy (Song & Mittal 2021).**
- Score: `Mentr(p, y) = -(1 - p_y) log(p_y) - Σ_{i≠y} p_i log(1 - p_i)` from the
  target-model softmax. Lower = more member-like.
- Threshold: per class at α = 0.10, from the saved calibration outputs (mean reference
  softmax on the calibration nodes, with their labels). Classes with fewer than
  `MIN_CALIB = 10` calibration nodes use a pooled threshold — the same sparse-class
  fallback as Shokri. Fallback classes are recorded.

**Offline RMIA (Zarifzadeh et al. 2024).**
- Ratio: `Pr(x|target) / mean_over_reference_models Pr(x|reference)`, at the true label.
- Score: fraction of the population `z` with `ratio_x / ratio_z ≥ γ` (γ = 2).
- Non-member evaluation and thresholding use the `final_eval_nonmember_ids` set, disjoint
  from the population `z`.

**Common-pool AUC (attack-strength comparison).** All three attacks (Shokri, entropy,
RMIA) are evaluated on one common final pool, `attack_member + final_eval_nonmember` (the
RMIA population excluded), so their AUCs are directly comparable on an identical set of
nodes. This common-pool AUC is what the paper reports as attack strength for all three —
including Shadow, replacing the earlier full-N Shadow AUC. The original frozen Shokri
full-pool AUC is still saved separately (`frozen_shokri_full_pool_auc`) and never
overwritten.

Outputs under `<out-dir>/scores/`:

### nodescores_&lt;tag&gt;.csv — extension per-node log (one row per node per stage)
Mirrors the core log's identity/ground-truth/diagnostic columns, with the Shokri attack
columns replaced by per-attack entropy and RMIA columns. Column order:
```
dataset, scenario, seed, gid, client_id, stage, subset, pool_role,
true_class, membership_label, local_degree, local_label_agreement, correct_prediction,
entropy_score, entropy_threshold, entropy_decision, entropy_fallback,
rmia_ratio, rmia_score, rmia_threshold, rmia_decision
```
(`gid` here equals `global_node_id` in the core log.) Full field descriptions are in the
reference tables below. RMIA columns are populated only for nodes RMIA evaluates (F, H,
attack_member, final-eval non-members); the RMIA population is excluded from evaluation.

### attackmeta_&lt;tag&gt;.json — per-cell attack metadata
Keys: `tag`, `dataset`, `seed`, `alpha`, `gamma`, `ref_seed_base`, `n_reference_models`,
`calibration_labels_ref`, `entropy_meta`, `rmia_meta`, `common_pool_auc`,
`frozen_shokri_full_pool_auc`. Field descriptions are in the reference tables below.

Run (single cell):
```bash
python score_and_match.py --tag Cora_client_s42 \
    --frozen-dir results_13x5/raw --out-dir attack_ext_out
```

## Stage 2b — rethreshold_rmia.py (review correction)

RMIA originally calibrated its threshold separately on M0 and MR. The reviewer asked for a
single threshold: compute it on M0, freeze it, and apply the same threshold to both M0 and
MR. This is a pure offline re-thresholding of the saved RMIA scores — no rescoring, no
reference-model retraining.

It reads the saved `scores/nodescores_<tag>.csv` and `attackmeta_<tag>.json`, recomputes
the RMIA decision at the frozen M0 threshold for both stages, and writes corrected
`scores/` plus `summary_rmia_rethresholded.csv`. With `--core-csv` / `--entropy-csv` it
also writes the common-pool versions of the Shadow and Entropy summaries so all three
attacks are reported on the same pool.

```bash
python rethreshold_rmia.py \
    --ext-dir extension_results_<timestamp> \
    --out-dir extension_results_<timestamp>_rmiafix \
    --core-csv summary_core.csv --entropy-csv summary_entropy.csv
```

Output directories ending in `_rmiafix` are regenerable and git-ignored. The corrected
`summary_rmia_rethresholded.csv` and the common-pool Shadow/Entropy CSVs are adopted as the
canonical `summary_rmia.csv` / `summary_core.csv` / `summary_entropy.csv`.

## Testing — test_extension.py

Automated acceptance tests, PASS/FAIL for integrity and diagnostics reported:
node identities match the frozen log for all four groups; reference OUT-status leak is 0;
population is disjoint from calibration and final-eval, ⊆ attack_nonmember, and
population ∪ final_eval covers attack_nonmember minus calibration; per-node scores saved
for both stages and both attacks with no NaN entropy; RMIA non-member evaluation count
equals the final-eval size and the population is excluded; entropy fallback classes
recorded and the per-node flag matches; calibration labels and reference seeds saved;
common-pool AUC present for all three attacks; entropy per-class thresholds finite.

```bash
python test_extension.py --tag Cora_client_s42 \
    --frozen-dir results_13x5/raw --out-dir attack_ext_out
```

The single-cell validation (Cora, seed 42) was approved before the full run: entropy
common-pool AUC 0.658 (C_cal −0.082, C_F +0.009); RMIA common-pool AUC 0.622
(C_cal −0.019, C_F +0.019).

---

## Full run over all 13x5 cells — run_full_extension.sh

```bash
bash run_full_extension.sh
```
Stage 1 builds 5 reference models for every cell (325 total; M0/MR stay frozen). Stage 2
scores every cell for entropy + RMIA, one cell at a time, logging each; a cell that errors
is recorded in `failed_cells.txt` and the loop continues. Stage 3 runs the acceptance test
on a few spot-check cells. Uses `--ref-seed-base 90000`.

Output folder (`extension_results_<timestamp>/`): `reference/refpack_<tag>.json`,
`scores/nodescores_<tag>.csv` + `attackmeta_<tag>.json`, `logs/`, `failed_cells.txt`,
`RUN_INFO.txt`. The frozen core (`results_13x5/raw/`, `summary_core.csv`) is read-only
throughout. The 13x5 run completed 65/65 cells with 0 failures and reproduced the
validated Cora s42 scores exactly.

---

## Aggregation — aggregate_extension.py

```bash
python aggregate_extension.py --ext-dir extension_results_<timestamp> \
    --alpha 0.10 --out-dir .
```
Produces `summary_entropy.csv` and `summary_rmia.csv` (65 rows each): `attack_strength_auc`
(common-pool, from `attackmeta`), `C_cal`, `C_F`, and the forget/held-out member rates
under M0 and MR (so `C_H` is derivable). RMIA's summary is then re-thresholded by Stage 2b
and carries explicit `rmia_threshold` and `C_H` columns.

---

## Figures — make_combined_figures.py

```bash
python make_combined_figures.py \
    --attacks shokri=summary_core.csv entropy=summary_entropy.csv rmia=summary_rmia.csv \
    --out figs
```
Produces (PDF + PNG, 400 dpi):
- `fig_rq2rq3` — (a) C_cal per dataset; (b) distribution of C_F.
- `fig_rq4_overview` — (a) membership rate on F and H under θ₀ and θ_R (one panel per
  attack, shared y-axis); (b) C_cal vs C_F scatter.
- `fig_impact` — (a) pre/post forget-versus-reference on discriminative cells (AUC ≥ 0.60);
  (b) per-dataset reference shift C_H for the three attacks.

(The earlier homophily-correlation figure was removed during review.)

---

## Results summary (full 13x5, three attacks)

At α = 0.10 on the retrained model, over 65 cells per attack (after the review
corrections: RMIA threshold frozen on M0; common-pool AUC for all three).

| Quantity | Shadow | Entropy | RMIA |
|---|---|---|---|
| Mean AUC (common pool) | 0.544 | 0.563 | 0.581 |
| C_cal range | [−0.10, +0.17] | [−0.10, +0.53] | [−0.10, +0.79] |
| C_cal \|·\|>0.05 (of 65) | 41 | 36 | 40 |
| C_F mean (all cells) | +0.002 | −0.0004 | −0.001 |
| C_F mean (AUC ≥ 0.60) | +0.006 | −0.008 | −0.006 |
| C_H mean | −0.065 | −0.084 | −0.103 |
| std(C_cal) / std(C_F) | 4.4× | 6.6× | 3.6× |

Headline findings:
- **Calibration does not transfer.** C_cal lies outside ±0.05 in 41 / 36 / 40 of 65 cells;
  the entropy attack is biased upward (mean C_cal +0.087).
- **Forget ≈ reference after retraining.** C_F is centred near zero for all three attacks,
  including where they are strong (Chameleon, Squirrel, AUC ≈ 0.70); std(C_cal) exceeds
  std(C_F) by 4.4 / 6.6 / 3.6×.
- **Not a weak-attack artifact.** On discriminative cells (AUC ≥ 0.60), F is clearly more
  member-like than the reference before retraining (gaps +0.10 / +0.22 / +0.16) and the gap
  collapses to within ±0.01 after.
- **The reference shifts across models, consistently.** Per-dataset C_H correlates between
  attacks (Pearson r = 0.91 shadow–entropy, 0.81 shadow–RMIA, 0.73 entropy–RMIA), with the
  same datasets (Cora, CS, Computers, Photo) shifting most.

---

## Reference tables — every score/field, what it means, why it is needed

### nodescores_&lt;tag&gt;.csv  (extension; one row per node per stage)

| Field | What it is | Why it is needed |
|---|---|---|
| dataset, scenario, seed | which cell this row belongs to | self-describing; pooling across cells |
| gid | global node id (= global_node_id in the core log) | joins to the core log and across attacks |
| client_id | the client the node belongs to | per-client analysis; confirms F/H source |
| stage | M0 or MR | the transition M0 → MR |
| subset | F / H / attack_member / attack_nonmember | role of the node in the audit |
| pool_role | evaluation | only evaluation nodes are scored |
| true_class | the node's true label | both attacks use the true-label probability |
| membership_label | supervised membership under this stage | ground truth for rates and AUC |
| local_degree | degree within the client subgraph | structural confound check |
| local_label_agreement | fraction of client-local neighbours sharing the label | structural confound check |
| correct_prediction | 1 if the target model is correct | difficulty confound check |
| entropy_score | modified prediction entropy (lower = more member-like) | the entropy attack's per-node signal |
| entropy_threshold | per-class decision threshold applied | records the operating point used |
| entropy_decision | 1 if entropy_score < threshold | the entropy attack's per-node call |
| entropy_fallback | 1 if the class used the pooled fallback | flags sparse-class calibration |
| rmia_ratio | Pr(x\|target)/mean_ref Pr(x\|reference) at the true label | core RMIA quantity; re-score at any γ offline |
| rmia_score | fraction of population with ratio_x/ratio_z ≥ γ | the RMIA attack's per-node signal |
| rmia_threshold | RMIA decision threshold at the operating point | records the operating point used |
| rmia_decision | 1 if rmia_score > threshold | the RMIA attack's per-node call |

### attackmeta_&lt;tag&gt;.json  (per-cell attack metadata)

| Field | What it is | Why it is needed |
|---|---|---|
| tag, dataset, seed | cell identity | bookkeeping |
| alpha | operating point (0.10) | the shared operating point |
| gamma | RMIA domination threshold (2) | RMIA hyper-parameter |
| ref_seed_base | base seed (model i uses base+i) | regenerate the reference models |
| n_reference_models | 5 | reproducibility; RMIA uses all five |
| calibration_labels_ref | note that labels live in the refpack | pointer |
| entropy_meta[stage] | pooled_tau, per_class_tau, fallback_classes, min_calib | audit entropy calibration |
| rmia_meta[stage] | threshold, gamma, n_population, n_final_eval_nonmember, population_ratios | audit RMIA + re-score at any γ offline |
| common_pool_auc | AUC of shokri/entropy/rmia on the common final pool | fair cross-attack strength comparison |
| frozen_shokri_full_pool_auc | Shokri's original full-pool AUC | kept separately; never overwritten |

### Derived endpoints (reported per cell)

| Quantity | Definition | Why it matters |
|---|---|---|
| forget / held-out member rate (M0, MR) | fraction of F / H predicted member under each model | the known membership transition; signal moving without a membership change |
| C_cal = FPR(H;MR) − α | operating-point transfer to the removed client | is the absolute signal calibrated across settings |
| C_F = FPR(F;MR) − FPR(H;MR) | forget set vs matched non-member reference under MR | the main endpoint: ≈ 0 ⇒ F behaves like a matched non-member |
| C_H = FPR(H;MR) − FPR(H;M0) | shift of the same non-member reference between models | the reference moves though its membership is unchanged |
| attack strength AUC (common pool) | member vs non-member AUC on attack_member + final_eval_nonmember | strength on one identical pool for all three attacks |

---

## Reproducibility and consistency notes

- Reference models are reproducible from `ref_seed_base` (model i uses `ref_seed_base + i`),
  independent of the M0/MR history, because an explicit seed is set before each.
- Scoring is deterministic (identical AUCs across re-runs).
- The common-pool AUC for all three attacks is on exactly the same node set
  (attack_member + final_eval_nonmember); Shokri's comparison AUC is computed offline from
  its saved per-node scores in the frozen log, while its original full-pool AUC is left
  untouched in the core results.
- RMIA's decision threshold is frozen on M0 and applied to both stages (Stage 2b).
- Small datasets: after splitting attack_nonmember into population + final-eval, the
  final-eval pool can be small (e.g. ~20 nodes on Chameleon), so per-cell AUC/threshold
  there are noisier; per-cell sizes are recorded in `rmia_meta`.
- The full 13x5 run reproduced the validated Cora s42 scores exactly.

## End-to-end reproduction

```bash
# 0. (if starting from scratch) core run -> results_13x5/raw/ + summary_core.csv
python experiment_main.py \
    --datasets Cora PubMed CS Photo Computers WikiCS Chameleon Squirrel Actor \
               tolokers minesweeper amazon_ratings roman_empire \
    --seeds 7 42 99 123 456 --scenario client --n-shadow 5 --core --out-dir results_13x5

# 1. extension: build references + score all 65 cells -> extension_results_<timestamp>/
bash run_full_extension.sh

# 2. aggregate into per-attack summaries
python aggregate_extension.py --ext-dir extension_results_<timestamp> --out-dir .

# 3. freeze the RMIA threshold on M0; write common-pool Shadow/Entropy summaries
python rethreshold_rmia.py \
    --ext-dir extension_results_<timestamp> \
    --out-dir extension_results_<timestamp>_rmiafix \
    --core-csv summary_core.csv --entropy-csv summary_entropy.csv
cp extension_results_<timestamp>_rmiafix/summary_rmia_rethresholded.csv summary_rmia.csv
cp extension_results_<timestamp>_rmiafix/summary_core_commonpool.csv    summary_core.csv
cp extension_results_<timestamp>_rmiafix/summary_entropy_commonpool.csv summary_entropy.csv

# 4. build the combined figures
python make_combined_figures.py \
    --attacks shokri=summary_core.csv entropy=summary_entropy.csv rmia=summary_rmia.csv \
    --out figs
```