# Is Membership Inference a Reliable Check for Unlearning in Federated Graphs?

This folder contains the corrected pipeline for studying whether a **membership
inference attack (MIA)** can verify unlearning in **federated graph learning**.
Rather than attacking an unlearned model, we apply the attack to the
**retrain-from-scratch model** (`MR`) — the model trained without the forgotten
data, which is the reference for "perfect" forgetting — and ask whether the
attack behaves as a faithful measure of forgetting.

The central idea is the **membership transition of the forget set**. The same
forget nodes are members of the original model `M0` and supervised non-members of
the retrained model `MR`. One frozen attack scores those nodes under both models,
and we check whether the attack tracks that known change, and whether its output
stays meaningful when compared against a matched supervised non-member reference.

---

git status                          # see what changed
git add 07_mia_redesigned/          # stage only your code folder (not stray files)
git status                          # confirm what's staged
git commit -m "what you changed"
git push

## Files (flat layout)

```
07_mia_redesigned/
├── models.py            GNN model, dataset loaders, per-dataset config,
│                        get_metrics (validation-only), CORE_DATASETS / CORE_SEEDS
├── partition.py         graph partitioning across clients + train/val/test split
├── train.py             federated training and the retrain-from-scratch models
├── attack.py            Shokri shadow-model MIA, out-of-sample calibration at α,
│                        pooled-fallback attack for sparse classes
├── metrics.py           corrected train/test generalization gap
├── nodelog.py           per-node logging (18 columns) + run manifest
├── experiment_main.py   client-level experiment runner
├── test_acceptance.py   integrity checks (PASS/FAIL) + diagnostics (reported)
├── requirements.txt
├── .gitignore
└── README.md
```

All modules import each other as flat siblings (e.g. `experiment_main.py` does
`from models import ...`, `from attack import ...`), so run scripts from inside
this folder.

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
  forgotten client (a transportability diagnostic). Near 0 = transfers well.
- **`C_F = FPR(F; MR) − FPR(H; MR)`** — how much the forget set stands out
  compared to the matched reference. `C_F ≈ 0` = looks like a normal non-member;
  `C_F > 0` = still member-like after removal.
- **Attack strength** — AUC (plus TPR/FPR at α) on the retained-client
  member/non-member pool. Reported as attack strength only, not as a forgetting
  measure.

---

## Design decisions

- **Transductive setting.** Each client trains on the full local adjacency (all
  nodes participate in message passing); the loss is masked to training nodes.
  Membership is defined at the level of the **training loss**, kept distinct from
  structural presence in the graph.
- **Client-level forgetting.** The forget set is a whole client. `MR` excludes
  only that client; the forgotten client is used only after `MR` is trained, to
  score `F` and `H`.
- **`H` is the forgotten client's test nodes** — a matched supervised non-member
  reference for `F` (its train nodes). A number on `F` is read relative to `H`.
- **One frozen attack for both models.** Fit and calibrated once, applied
  unchanged to `M0` and `MR`.
- **Identity-disjoint pools.** Within each retained client, a 30% slice of the
  train/test nodes is reserved as evaluation members/non-members; the rest fit
  and calibrate the attack. `F`, `H`, and all evaluation nodes are removed from
  the shadow training masks **before** the shadow models are trained.
- **Fixed operating point with out-of-sample calibration.** The threshold is set
  at α = 0.10 on a separate calibration pool, then frozen. `F` is always read
  against `H`.
- **Corrected generalization gap.** `acc_train(MR) − acc_test(MR)`, with training
  accuracy on the retained training nodes (forget nodes excluded) and test
  accuracy on the untouched test nodes.

---

## Changes in this version (review corrections)

1. **H and MR isolation.** `MR` excludes only the forgotten client. `H` is that
   client's test nodes. Labels are stage-aware: `F` is a member under `M0` and a
   non-member under `MR`; `H` is a non-member under both.
2. **Validation-only model selection.** A client with no validation nodes is
   skipped; the test set is never used during training or selection.
3. **Attack pool from retained clients only.** The forgotten client's test nodes
   belong to `H` only.
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
forgotten client's score distribution differs from the retained ones, the
threshold may not transfer, and `C_cal` can be far from 0 on some datasets. `C_F`
remains a valid relative comparison; `C_cal` is reported as the transportability
diagnostic.

---

## Per-node log: columns and the analyses they enable

Each row of `raw/nodelog_*.csv` is one scored node under one model stage. The 18
columns are below, with what each one lets you check later — without rerunning,
since everything is saved.

| Column | What it is | What analysis it enables |
|---|---|---|
| `dataset` | dataset name | group/compare results by dataset |
| `scenario` | client or node level | separate the two unlearning settings |
| `seed` | random seed | check stability across seeds; pair rows within a cell |
| `global_node_id` | node's index in the original graph | pair the same node across M0 and MR; verify pool disjointness; join to structure |
| `client_id` | the node's client | confirm F/H come only from the forgotten client; group by client |
| `stage` | `M0` or `MR` | measure the membership transition (before vs after retraining) |
| `subset` | `F` / `H` / `attack_member` / `attack_nonmember` | split nodes into their roles for every comparison |
| `pool_role` | evaluation (all logged nodes) | confirm only evaluation nodes are scored |
| `true_class` | node's label | stratify by class; needed for per-class analysis |
| `membership_label` | supervised membership (1/0) under this stage | ground truth for the transition (F=1 at M0, 0 at MR; H=0 both) |
| `mia_score` | the attack's membership score | the raw signal; recompute rates at any threshold |
| `decision` | member/non-member at the frozen threshold | compute member rates, C_F, C_cal |
| `threshold` | the frozen threshold applied | confirm the attack is frozen (same M0 vs MR); audit calibration |
| `fallback_status` | whether the pooled fallback attack scored this node | check how much of a result rests on the fallback |
| `local_degree` | degree within the node's client subgraph | check if an F/H difference is driven by connectivity |
| `local_label_agreement` | fraction of client-local neighbours sharing the label (homophily) | check if an F/H difference is driven by neighbourhood homophily |
| `correct_prediction` | 1 if the model classifies the node correctly | check if an F/H difference is driven by classification difficulty |
| `prob_vector` | the softmax vector used by the attack | recompute confidence/entropy and any score-based quantity |

### The confound checks in detail

The three structural/difficulty columns exist to answer the question a reviewer
will ask: *when F and H score differently (or when a score changes M0 to MR),
how do you know it is about membership and not some other property the two groups
happen to differ on?* Each column supports a concrete two-step check.

**`correct_prediction` — is it classification difficulty?**
MIA scores track how well the model fits a node: a correctly-classified node is
usually fit confidently and looks member-like, a misclassified one looks
non-member-like. So difficulty can drive the score independently of membership.
- First pass: compare the fraction of correctly-classified nodes in F vs H. If F
  is, say, 80% correct and H is 60%, that difference in difficulty could explain a
  score gap.
- Rigorous pass: look only at the correctly-classified F and H nodes and see if
  the score difference still holds. If it vanishes once you match on correctness,
  the difference was really about difficulty; if it survives, the membership
  signal is real.

**`local_label_agreement` (homophily) — is it neighbourhood structure?**
A node surrounded by same-label neighbours is predicted confidently (the GNN
agrees with its neighbours), so it looks member-like for a structural reason.
- First pass: compare average homophily of F vs H. A large gap flags a possible
  structural confound.
- Rigorous pass: compare F and H nodes at the same homophily level (stratify into
  low/medium/high homophily bins) and check whether the score difference persists
  within bins. If it disappears, homophily explained it; if it persists, it did
  not.

**`local_degree` — is it raw connectivity?**
More neighbours means more message passing, which can shift the score regardless
of membership.
- First pass: compare average degree of F vs H.
- Rigorous pass: match F and H nodes on degree (or add degree as a control) and
  see whether the score difference remains. This is the coarsest of the three
  structural checks; homophily usually explains MIA confidence better than raw
  degree, so treat degree as a supporting, not primary, control.

**`prob_vector` — confidence and entropy, for free.**
The full softmax is saved, so you can compute the model's confidence (max
probability) or the prediction entropy for any node at analysis time, and repeat
the difficulty check above with a continuous measure rather than the binary
`correct_prediction`.

### Important caveat when using these checks

All three columns are **correlated with membership** — members tend to be
classified correctly, confidently, and in well-fit neighbourhoods, *because* they
were trained on. So a raw difference-in-averages can mislead in both directions:
it can invent a confound that is really the membership effect wearing another
label, or it can hide a real effect. Use these columns as flags that a comparison
may be unclean, and rely on the stratified/matched analysis (the "rigorous pass"
above) rather than a blunt average comparison. This is why they are reported as
diagnostics, never as pass/fail criteria.

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
python experiment_main.py --datasets Cora --seeds 42 --scenario client --out-dir results
```

The full frozen core (13 datasets × 5 seeds = 65 cells):

```bash
python experiment_main.py --core --scenario client --out-dir results
```

Runs accumulate into `results/summary_core.csv`; a cell already present is
skipped, and `--overwrite` forces a re-run. The frozen core set is defined in
`models.py` as `CORE_DATASETS` (13) and `CORE_SEEDS` (5).

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
python test_acceptance.py --dir results
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
  choices to keep all pools non-empty and reliable; adjustable in
  `experiment_main.py` and `attack.py`.
- A cell whose forgotten client has too few test nodes will produce an empty `H`
  and therefore NaN `C_cal`/`C_F`; the acceptance suite flags this, and such
  cells should be noted rather than treated as results.
- Node-level forgetting, additional attacks, and sensitivity sweeps are out of
  scope for this client-level core.