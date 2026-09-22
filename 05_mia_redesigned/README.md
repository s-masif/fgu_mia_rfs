# Is Membership Inference a Reliable Check for Unlearning in Federated Graphs?

This repository studies whether a **membership inference attack (MIA)** can be
used to verify unlearning in **federated graph learning**. Rather than attacking
an unlearned model, we apply the attack to the **retrain-from-scratch model** —
the model trained without the forgotten data, which is the reference for
"perfect" forgetting — and ask whether the attack behaves as a faithful measure
of forgetting.

The central idea is the **membership transition of the forget set**. The same
forget nodes are members of the original model `M0` and non-members of the
retrained model `MR`. One frozen attack scores those nodes under both models,
and we check whether the attack tracks that known change, and whether its
output stays meaningful when compared against a matched never-seen reference set.

This README documents the **corrected pipeline** (client-level core), including
the design decisions and the six protocol corrections applied after review.

---

## What the pipeline does, for one (dataset, seed) cell

1. Load the graph, split it across clients, train the **original model `M0`** on
   all clients.
2. Choose a **forget set**: one whole client (client-level forgetting). Within
   that same client, its **train nodes** become `F` and its **test nodes**
   become the held-out reference `H`.
3. Train the **retrained model `MR`** on the retained clients only. The forgotten
   client takes no part in `MR` training or checkpoint selection.
4. Train shadow models and build the attack, keeping the nodes used to **fit and
   calibrate** the attack strictly separate (by node identity) from the nodes
   finally **scored**.
5. Freeze the attack and apply it to `M0` and `MR`, scoring `F` and `H` through
   the forgotten client's own subgraph (which remains in the graph, only left
   out of `MR` training).
6. Record, for the forget set, how often it is predicted as a member before
   (`M0`) and after (`MR`) retraining, and compare `F` to the reference `H`.

---

## Key quantities

Let `α = 0.10` be the fixed operating point (target false-positive rate).

- **Forget predicted-member rate (M0 vs MR)** — fraction of forget nodes
  predicted as members before and after retraining. It is expected to drop after
  retraining.
- **`C_cal = FPR(H; MR) − α`** — how far the decision threshold drifts on the
  forgotten client. Near 0 means the threshold transfers well; far from 0 means
  it does not.
- **`C_F = FPR(F; MR) − FPR(H; MR)`** — how much the forget set stands out
  compared to the matched never-seen reference. `C_F ≈ 0` means the forget set
  looks like a normal non-member; `C_F > 0` means it is still flagged as
  member-like after removal.
- **Attack strength** — AUC (plus TPR/FPR at `α`) on the retained-client
  member/non-member pool. This measures how strong the attack is, and is
  reported separately from any forgetting claim.

---

## Design decisions (and why)

These choices define what the numbers mean; they were agreed during review.

**Setting is transductive.** Each client trains on the full local adjacency
(all nodes participate in message passing) and the loss is masked to the
training nodes. A node removed from the supervised loss can still influence the
model through message passing, so "membership" is defined at the level of the
**training loss**, kept distinct from structural presence in the graph.

**The forget set is a whole client (client-level).** `MR` excludes only that
client. The forgotten client is used **only** after `MR` is trained, to score
`F` and `H`; it never participates in `MR` training or in `MR` checkpoint
selection.

**`H` is the forgotten client's test nodes.** `F` (its train nodes) and `H`
(its test nodes) come from the same client, so `H` is a matched, never-seen
reference for `F`. A membership number on `F` is read relative to `H`, never on
its own.

**One frozen attack for both models.** The attack is fit and calibrated once,
then applied unchanged to `M0` and `MR`. Only the model differs between the two.

**Identity-disjoint pools.** The nodes used to fit and calibrate the attack are
kept separate, by node identity, from the nodes finally scored. Within each
retained client the supervised (train) and held-out (test) nodes are split: a
30% slice is reserved as the evaluation members/non-members, and the rest are
used to fit and calibrate. `F`, `H`, and all evaluation nodes are removed from
the shadow training masks **before** the shadow models are trained, so the
attack never learns from any node it later scores.

**Fixed operating point with out-of-sample calibration.** The decision threshold
is set at `α = 0.10` on a separate calibration pool (out-of-sample non-members),
then frozen. At this point, roughly 10% predicted-members on genuine
non-members is expected, so the forget-set number is always read against `H`.

**Corrected generalization gap.** The gap is
`acc_train(MR) − acc_test(MR)`, where training accuracy is measured on the
retained training nodes (forget nodes excluded) and test accuracy on the
untouched test nodes.

---

## The six protocol corrections in this version

This version applies six corrections from review. Each is summarised with where
it lives.

1. **Held-out set and MR isolation.** `MR` excludes only the forgotten client
   (not an extra held-out client). `H` is the forgotten client's test nodes.
   Membership labels are stage-aware: `F` is a member under `M0` and a
   non-member under `MR`; `H` is a non-member under both.
   *(`experiment_main.py`: `retain_clients`, `build_eval_pool_client`.)*

2. **Validation-only model selection.** Checkpoint selection uses validation
   nodes only; a client with no validation nodes is skipped for that metric. The
   test set is never used during training or selection.
   *(`core/models.py`: `get_metrics`.)*

3. **Attack pool from retained clients only.** The attack-strength members come
   from retained clients' train nodes and non-members from retained clients' test
   nodes. The forgotten client's test nodes belong to `H` only.
   *(`experiment_main.py`: `build_eval_pool_client`.)*

4. **Identity separation before shadow training.** Evaluation nodes (`F`, `H`,
   and the evaluation slice) are removed from the shadow clients' training masks
   before `federated_train`, so the shadow models never learn from them. The
   nodes remain structurally present in the graph.
   *(`experiment_main.py`: `build_shadow_pools`.)*

5. **Sparse-class fallback.** A class uses its own attack and threshold only if
   it has enough calibration support (at least `min_calib = 10` calibration
   non-members). Otherwise it is routed to the **pooled fallback attack together
   with its pooled threshold**, and this is recorded per class. Applying a pooled
   threshold to an under-trained class-specific attack is avoided.
   *(`core/attack.py`: `ShokriShadowMIA.fit`, `predict`.)*

6. **Test suite: integrity vs diagnostics.** The acceptance script marks
   PASS/FAIL only for implementation integrity (stage-aware labels, frozen
   thresholds, disjoint pools, summary matching the raw logs, provenance).
   Scientific quantities (whether `F` decreases, the size of `C_cal`, train
   vs test accuracy) are reported as diagnostics, not judged.
   *(`analysis/test_acceptance.py`.)*

**Known open point (interpretation, not a bug).** The threshold is calibrated on
the retained clients but applied to `F` and `H` in the forgotten client. When
the forgotten client's score distribution differs from the retained ones (which
community-based partitioning tends to cause), the threshold does not transfer
cleanly, and `C_cal` can be far from 0 on some datasets. `C_F` remains a valid
relative comparison (`F` and `H` are scored under the same threshold on the same
client), but the absolute rates and `C_cal` depend on this transfer. `C_cal` is
therefore reported as a transportability diagnostic.

---

## Repository structure

```
mia-unlearning/
├── src/
│   ├── core/
│   │   ├── models.py      GNN model, dataset loaders, per-dataset config,
│   │   │                  get_metrics (validation-only selection)
│   │   ├── partition.py   graph partitioning across clients + train/val/test split
│   │   ├── train.py       federated training and the retrain-from-scratch models
│   │   ├── attack.py      Shokri shadow-model MIA, out-of-sample calibration at α,
│   │   │                  pooled-fallback attack for sparse classes
│   │   ├── metrics.py     corrected train/test generalization gap
│   │   └── nodelog.py     per-node logging + run manifest (with pool identities)
│   └── experiment_main.py client-level experiment runner
├── analysis/
│   └── test_acceptance.py integrity checks (PASS/FAIL) + diagnostics (reported)
├── requirements.txt
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
filtered Chameleon/Squirrel versions come from the Yandex
`heterophilous-graphs` repository; the others come through PyTorch Geometric.
The filtered Chameleon/Squirrel are used to avoid duplicated-node leakage.

---

## How to run

One dataset, one seed:

```bash
cd src
python experiment_main.py --datasets Cora --seeds 42 --scenario client --out-dir results
```

Several datasets and seeds:

```bash
python experiment_main.py \
    --datasets Cora Chameleon Squirrel Actor \
    --seeds 42 \
    --scenario client --out-dir results
```

Runs accumulate into `results/summary_core.csv`; a cell already present is
skipped, and `--overwrite` forces a re-run.

The full client-level core is 13 datasets × 5 seeds = 65 cells.

---

## Outputs

Under `--out-dir` (default `results/`):

- `summary_core.csv` — one row per (dataset, seed) cell. Columns include:
  `forget_member_rate_M0`, `forget_member_rate_MR`,
  `forget_mean_score_M0`, `forget_mean_score_MR`,
  `heldout_member_rate_MR`, `C_cal`, `C_F`,
  `attack_strength_auc`, `attack_tpr_at_alpha`, `attack_fpr_at_alpha`,
  `acc_train`, `acc_test`, `train_test_gap`, `alpha`, `n_fallback_classes`.
- `raw/nodelog_<dataset>_<scenario>_s<seed>.csv` — one row per scored node
  (`global_node_id`, `client_id`, `stage` M0/MR, `subset` F/H/attack_member/
  attack_nonmember, `membership_label`, `mia_score`, `decision`, `threshold`,
  `fallback_status`, `prob_vector`). All summary numbers can be recomputed from
  these logs, so analysis never requires a re-run.
- `raw/manifest_<dataset>_<scenario>_s<seed>.json` — the run's settings: the
  removed client(s), the `F`/`H` node identities, the fit/calibration/evaluation
  pool identities, the frozen thresholds, which classes used the pooled fallback
  and their calibration support, the split ratios, and the code commit hash.

---

## Checking a run

After a batch, verify against the agreed criteria:

```bash
cd analysis
python test_acceptance.py --dir ../src/results
```

PASS/FAIL (implementation integrity):
- schema and no NaN scores;
- stage-aware labels (`F`=1 under M0, `F`=0 under MR, `H`=0 under both;
  attack_member=1, attack_nonmember=0);
- frozen thresholds (identical for `M0` and `MR`);
- `F` and `H` paired across stages, and drawn only from the forgotten client;
- forgotten client absent from the attack-strength pools;
- fit / calibration / evaluation pools pairwise disjoint (from the manifest);
- summary values match the values recomputed from the raw logs;
- provenance (commit hash, split ratios) present.

Diagnostics (reported, never judged):
- forget member rate M0 → MR and its change;
- `H` member rate under MR, `C_cal`, `C_F`;
- MR mean score by subset;
- generalization gap;
- attack strength (AUC, TPR/FPR at α);
- number of rows scored with the pooled fallback attack.

---

## Notes

- The 30% evaluation / 70% fit split within each client, and the
  `min_calib = 10` threshold for the sparse-class fallback, are implementation
  choices to keep all pools non-empty and reliable; both are easily adjusted in
  `experiment_main.py` and `core/attack.py`.
- Node-level forgetting, additional attacks, and any sensitivity sweeps are out
  of scope for this client-level core; they would be added only if a final claim
  requires them.
