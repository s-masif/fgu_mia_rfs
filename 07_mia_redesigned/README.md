# Is Membership Inference a Reliable Check for Unlearning in Federated Graphs?

This repository studies whether a **membership inference attack (MIA)** can be
used to verify unlearning in **federated graph learning**. Rather than attacking
an unlearned model, we apply the attack to the **retrain-from-scratch model**
(`MR`) — the model trained without the forgotten data, which is the reference for
"perfect" forgetting — and ask whether the attack behaves as a faithful measure
of forgetting.

The central idea is the **membership transition of the forget set**. The same
forget nodes are members of the original model `M0` and supervised non-members of
the retrained model `MR`. One frozen attack scores those nodes under both models,
and we check whether the attack tracks that known change, and whether its output
stays meaningful when compared against a matched supervised non-member reference.

---

## Pipeline, for one (dataset, seed) cell

1. Load the graph, split it across clients, train the **original model `M0`** on
   all clients.
2. Choose a **forget set**: one whole client (client-level forgetting). Within
   that same client, its **train nodes** become `F` and its **test nodes** become
   the matched reference `H`.
3. Train the **retrained model `MR`** on the retained clients only. The forgotten
   client takes no part in `MR` training or checkpoint selection.
4. Train shadow models and build the attack, keeping the nodes used to **fit and
   calibrate** the attack strictly separate (by node identity) from the nodes
   finally **scored**.
5. Freeze the attack and apply it to `M0` and `MR`, scoring `F` and `H` through
   the forgotten client's own subgraph (which stays in the graph, only left out
   of `MR` training).
6. Record, for the forget set, how often it is predicted as a member before
   (`M0`) and after (`MR`) retraining, and compare `F` to the reference `H`.

---

## Key quantities (α = 0.10, the fixed operating point)

- **Forget predicted-member rate (M0 vs MR)** — fraction of forget nodes
  predicted as members before and after retraining. Expected to drop after
  retraining.
- **`C_cal = FPR(H; MR) − α`** — how far the decision threshold drifts on the
  forgotten client (a transportability diagnostic). Near 0 = the threshold
  transfers well.
- **`C_F = FPR(F; MR) − FPR(H; MR)`** — how much the forget set stands out
  compared to the matched reference. `C_F ≈ 0` = the forget set looks like a
  normal non-member; `C_F > 0` = still member-like after removal.
- **Attack strength** — AUC (plus TPR/FPR at α) on the retained-client
  member/non-member pool. Reported as attack strength only, not as a forgetting
  measure.

---

## Design decisions (why the numbers mean what they mean)

- **Transductive setting.** Each client trains on the full local adjacency (all
  nodes participate in message passing) and the loss is masked to training nodes.
  A node removed from the loss can still influence the model through message
  passing, so membership is defined at the level of the **training loss**, kept
  distinct from structural presence.
- **Client-level forgetting.** The forget set is a whole client. `MR` excludes
  only that client; the forgotten client is used only after `MR` is trained, to
  score `F` and `H`.
- **`H` is the forgotten client's test nodes.** `F` (its train nodes) and `H`
  (its test nodes) come from the same client, so `H` is a matched supervised
  non-member reference for `F`. A number on `F` is read relative to `H`.
- **One frozen attack for both models.** Fit and calibrated once, applied
  unchanged to `M0` and `MR`.
- **Identity-disjoint pools.** Within each retained client, a 30% slice of the
  train/test nodes is reserved as evaluation members/non-members; the rest fit
  and calibrate the attack. `F`, `H`, and all evaluation nodes are removed from
  the shadow training masks **before** the shadow models are trained.
- **Fixed operating point with out-of-sample calibration.** The threshold is set
  at α = 0.10 on a separate calibration pool, then frozen. ~10% predicted-members
  on genuine non-members is expected, so `F` is always read against `H`.
- **Corrected generalization gap.** `acc_train(MR) − acc_test(MR)`, with training
  accuracy on the retained training nodes (forget nodes excluded) and test
  accuracy on the untouched test nodes.

---

## Changes in this version (review corrections)

Seven corrections from review, plus two extra diagnostic columns:

1. **H and MR isolation.** `MR` excludes only the forgotten client. `H` is that
   client's test nodes. Labels are stage-aware: `F` is a member under `M0` and a
   non-member under `MR`; `H` is a non-member under both.
2. **Validation-only model selection.** Checkpoint selection uses validation
   nodes only; a client with no validation nodes is skipped. Test is never used
   during training or selection.
3. **Attack pool from retained clients only.** The forgotten client's test nodes
   belong to `H` only, never to the attack pool.
4. **Identity separation before shadow training.** Evaluation nodes (`F`, `H`,
   the evaluation slice) are removed from the shadow training masks before
   training; the nodes remain in the graph.
5. **Sparse-class fallback.** A class uses its own attack and threshold only with
   enough calibration support (counted as **unique node IDs**, `min_calib = 10`).
   Otherwise it uses the **pooled fallback attack together with its pooled
   threshold**, recorded per class.
6. **Test suite: integrity vs diagnostics.** PASS/FAIL only for implementation
   integrity; scientific quantities are reported as diagnostics, not judged.
7. **Provenance and settings.** The manifest records a real git commit SHA, the
   split ratios, and the attack settings (`n_shadow`, split fractions,
   `min_calib`, α).

Two extra per-node diagnostic columns (to check whether an `F`/`H` difference is
structural or difficulty-related, without a rerun):

- **`local_degree`** — degree within the node's client subgraph.
- **`local_label_agreement`** — fraction of client-local neighbours sharing the
  node's label (homophily).
- **`correct_prediction`** — whether the model classifies the node correctly.

**Known open point (interpretation, not a bug).** The threshold is calibrated on
the retained clients but applied to `F`/`H` in the forgotten client. When the
forgotten client's score distribution differs from the retained ones,
the threshold may not transfer, and `C_cal` can be far from 0 on some datasets.
`C_F` remains a valid relative comparison (`F` and `H` are scored under the same
threshold on the same client); `C_cal` is reported as the transportability
diagnostic.

---

## Repository structure

```
mia-unlearning/
├── src/
│   ├── core/
│   │   ├── models.py      GNN model, dataset loaders, per-dataset config,
│   │   │                  get_metrics (validation-only), CORE_DATASETS/CORE_SEEDS
│   │   ├── partition.py   graph partitioning across clients + train/val/test split
│   │   ├── train.py       federated training and the retrain-from-scratch models
│   │   ├── attack.py      Shokri shadow-model MIA, out-of-sample calibration at α,
│   │   │                  pooled-fallback attack for sparse classes
│   │   ├── metrics.py     corrected train/test generalization gap
│   │   └── nodelog.py     per-node logging (18 columns) + run manifest
│   └── experiment_main.py client-level experiment runner
├── analysis/
│   └── test_acceptance.py integrity checks (PASS/FAIL) + diagnostics (reported)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Datasets download automatically on first use. The heterophilous datasets and the
filtered Chameleon/Squirrel versions come from the Yandex `heterophilous-graphs`
repository; the others come through PyTorch Geometric. Filtered Chameleon/Squirrel
are used to avoid duplicated-node leakage.

---

## How to run

One dataset, one seed:

```bash
cd src
python experiment_main.py --datasets Cora --seeds 42 --scenario client --out-dir results
```

The full frozen core (13 datasets × 5 seeds = 65 cells):

```bash
python experiment_main.py --core --scenario client --out-dir results
```

Runs accumulate into `results/summary_core.csv`; a cell already present is
skipped, and `--overwrite` forces a re-run.

The frozen core set is defined in `src/core/models.py` as `CORE_DATASETS`
(13 datasets) and `CORE_SEEDS` (5 seeds).

---

## Outputs

Under `--out-dir` (default `results/`):

- `summary_core.csv` — one row per (dataset, seed). Columns include:
  `forget_member_rate_M0/MR`, `forget_mean_score_M0/MR`,
  `heldout_member_rate_M0/MR`, `heldout_mean_score_M0/MR`, `C_cal`, `C_F`,
  `attack_strength_auc`, `attack_tpr_at_alpha`, `attack_fpr_at_alpha`,
  `acc_train`, `acc_test`, `train_test_gap`, `alpha`, `n_fallback_classes`.
- `raw/nodelog_<dataset>_<scenario>_s<seed>.csv` — one row per scored node
  (18 columns): `dataset`, `scenario`, `seed`, `global_node_id`, `client_id`,
  `stage`, `subset`, `pool_role`, `true_class`, `membership_label`, `mia_score`,
  `decision`, `threshold`, `fallback_status`, `local_degree`,
  `local_label_agreement`, `correct_prediction`, `prob_vector`. All summary
  numbers can be recomputed from these logs, so analysis never requires a rerun.
- `raw/manifest_<dataset>_<scenario>_s<seed>.json` — run settings: removed
  client(s), `F`/`H` node identities, the fit/calibration/evaluation pool
  identities, frozen thresholds, which classes used the pooled fallback and their
  calibration support, the attack settings, the split ratios, and the git SHA.

---

## Checking a run

```bash
cd analysis
python test_acceptance.py --dir ../src/results
```

PASS/FAIL (implementation integrity): 18-column schema; no NaN scores;
stage-aware labels; frozen thresholds; `F`/`H` paired across stages and drawn
only from the forgotten client; forgotten client absent from the attack pools;
fit/calibration/evaluation pools pairwise disjoint; summary matching the raw
logs; provenance present.

Diagnostics (reported, not judged): forget rate M0 → MR; `H` rate under MR,
`C_cal`, `C_F`; MR mean score by subset; generalization gap; attack strength;
fallback usage; and F/H structural summaries (`local_degree`,
`local_label_agreement`, `correct_prediction`).

---

## Notes

- The 30% evaluation / 70% fit split and `min_calib = 10` are implementation
  choices to keep all pools non-empty and reliable; both are adjustable in
  `experiment_main.py` and `core/attack.py`.
- A cell whose forgotten client has too few test nodes will produce an empty `H`
  and therefore NaN `C_cal`/`C_F`; the acceptance suite flags this (its "four
  subsets non-empty" check), and such cells should be noted rather than treated
  as results.
- Node-level forgetting, additional attacks, and sensitivity sweeps are out of
  scope for this client-level core.
