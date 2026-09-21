# Is Membership Inference a Reliable Check for Unlearning in Federated Graphs?

This repository studies whether a **membership inference attack (MIA)** can be
used to verify unlearning in **federated graph learning**. Instead of attacking
an unlearned model, we apply the attack to the **retrain-from-scratch model** —
the model that is trained without the forget set and is therefore the reference
for "perfect" forgetting — and ask whether the attack behaves as a faithful
measure of forgetting.

The key idea is the **membership transition of the forget set**. The same forget
nodes are members of the original model `M0` and non-members of the retrained
model `MR`. We use one frozen attack to score those nodes under both models and
check whether the attack tracks that known change, and whether its output stays
meaningful when compared against a matched never-seen reference set.

## What the code does, in one run

For one `(dataset, seed)` cell, the pipeline:

1. Loads the graph, splits it across clients, trains the **original model `M0`**.
2. Picks a **forget set `F`** (a client) and a matched, never-seen **held-out
   set `H`** of the same size.
3. Trains the **retrained model `MR`** on the remaining data (both `F` and `H`
   excluded).
4. Trains shadow models and builds the attack, keeping the nodes used to
   **fit/calibrate** the attack separate from the nodes finally **scored**.
5. Freezes the attack and applies it to `M0` and `MR`.
6. Records, for the forget set, how often it is predicted as a member before
   (`M0`) and after (`MR`) retraining, and compares `F` to the held-out
   reference `H`.

## Main quantities

- **Forget predicted-member rate (M0 vs MR)** — the fraction of forget nodes
  predicted as members before and after retraining. A faithful signal should
  drop after retraining.
- **`C_cal = FPR(H; MR) − α`** — how far the decision threshold drifts on the
  real target (small is good; the threshold is set at a false-positive rate of
  `α = 0.10`).
- **`C_F = FPR(F; MR) − FPR(H; MR)`** — how much the forget set stands out
  compared to a matched never-seen set. `C_F ≈ 0` means the forget set looks like
  a normal non-member; `C_F > 0` means it is still flagged as member-like after
  it was removed.

## Repository structure

```
mia-unlearning/
├── src/
│   ├── core/
│   │   ├── models.py      GNN model, dataset loaders, per-dataset config
│   │   ├── partition.py   graph partitioning across clients + train/val/test split
│   │   ├── train.py       federated training and the retrain-from-scratch models
│   │   ├── attack.py      Shokri shadow-model MIA + calibrated decision threshold
│   │   ├── metrics.py     train/test generalization gap
│   │   └── nodelog.py     per-node logging and the run manifest
│   └── experiment_main.py client-level experiment runner
├── analysis/
│   └── test_acceptance.py checks each run against the agreed criteria (pass/fail)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Datasets download automatically on first use. The heterophilous datasets and the
filtered Chameleon/Squirrel come from the Yandex `heterophilous-graphs`
repository; the others come through PyTorch Geometric.

## How to run

Run one or more datasets for one or more seeds:

```bash
cd src
python experiment_main.py --datasets Cora --seeds 7 --scenario client --out-dir results
```

Several datasets and seeds at once:

```bash
python experiment_main.py \
    --datasets Cora Chameleon Squirrel Actor \
    --seeds 7 42 \
    --scenario client --out-dir results
```

Runs accumulate: results are added to `results/summary_core.csv`, and a cell that
is already done is skipped (use `--overwrite` to force a re-run).

## Outputs

Under `--out-dir` (default `results/`):

- `summary_core.csv` — one row per `(dataset, seed)` cell with the main
  quantities (forget rates, `C_cal`, `C_F`, attack strength, gap).
- `raw/nodelog_<dataset>_<scenario>_s<seed>.csv` — one row per scored node
  (ID, client, stage `M0`/`MR`, which set it belongs to, label, score,
  prediction, threshold). All the summary numbers can be recomputed from these
  files.
- `raw/manifest_<dataset>_<scenario>_s<seed>.json` — the run's settings: the
  node identities of each group, the frozen thresholds, the split ratios, and the
  code commit hash.

## Checking a run

After a batch of runs, verify them against the agreed criteria:

```bash
cd analysis
python test_acceptance.py --dir ../src/results
```

It prints a PASS/FAIL for each check per cell: the attack is frozen (same
thresholds for `M0` and `MR`), the fit/calibration/evaluation node groups do not
overlap, the forget set changes in the expected direction, the summary numbers
match the raw logs, the threshold drift is within tolerance, and the gap is
computed correctly.

## Notes

- **Model selection uses the validation set only**; the test set is never used
  during training, since it is also used in the attack and the gap.
- **One frozen attack** is used for both `M0` and `MR`; only the model changes.
- The nodes used to **build** the attack and the nodes finally **scored** are
  kept separate by node identity; the graph itself is shared (transductive
  setting).
- The **generalization gap** is training accuracy (on the retained training
  nodes, forget set excluded) minus test accuracy.
- A couple of split fractions (how many nodes per client are held out for
  evaluation, and how many non-members are used to set the threshold) are
  implementation choices and can be adjusted in `experiment_main.py`.
