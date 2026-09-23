# Reliability of Membership Inference for Verifying Unlearning in Federated Graphs

This repository contains the code for a study of whether a membership inference
attack (MIA) is a reliable way to verify unlearning in federated graph learning.
Instead of attacking an unlearned model, we apply the attack to the
**retrain-from-scratch gold-standard model** — the one model whose forgetting is
certain — and ask whether its MIA score behaves as a faithful measure of
forgetting or is instead driven by the model's generalization gap.

The repository is organized as four self-contained experiments. Each folder runs
on its own with no cross-imports; shared backbone files (model, attack,
partitioning) are duplicated per folder on purpose so that every experiment is
reproducible in isolation.

## Repository layout

```
fgu-mia-audit/
├── 01_main_study/     MIA on the retrained model across 13 datasets
├── 02_control/        matched held-out control (is the signal forget-specific?)
├── 03_variations/     sweeps over partitioning, #clients, forget ratio
├── 04_subsample/      shrink datasets with topology held fixed (size vs topology)
└── docs/              one detailed guide per experiment
```

## The four experiments at a glance

| # | Folder | Question it answers | Scale |
|---|--------|---------------------|-------|
| 1 | `01_main_study`  | How does MIA AUC on the retrained model behave across datasets and scenarios? | 13 datasets x 2 scenarios x 5 seeds = 130 runs |
| 2 | `02_control`     | Is the score specific to the forget set, or would any matched unseen set give the same score? | 13 x 2 x 3 = 78 runs |
| 3 | `03_variations`  | Does the score depend on partitioning, number of clients, or forget ratio? | 7 datasets x 2 x 3 seeds x 4 values x 3 sweeps = 504 runs |
| 4 | `04_subsample`   | Is the "small dataset -> high MIA" effect due to size or to topology? | 7 datasets x 2 x 3 seeds x 4 fractions = 168 runs |

Each run trains one gold-standard model and five shadow models for the attack.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The code was developed with Python 3.11 and PyTorch 2.7 (CUDA 11.8). A GPU is
recommended; the sweeps are long (order of hours) on CPU.

`pymetis` is only needed for the Metis and Metis-Plus partitioners in
experiment 3. If it is not installed, use the other partitioners.

## How to run

Each experiment folder has its own guide in `docs/`. In brief:

```bash
# 1. Main study
cd 01_main_study
python mia_experiment.py
python evaluate_mia_consistency.py --csv mia_consistency_results/raw_data/mia_consistency_raw.csv

# 2. Control
cd 02_control
python mia_control_experiment.py
python evaluate_control.py

# 3. Variations (three sweeps; supports --skip-existing to resume)
cd 03_variations
bash run_sweeps.sh all
python evaluate_variations.py --dir mia_variations_results/raw_data
python plot_figure1_gap.py --dir mia_variations_results/raw_data --out-dir mia_variations_results/plots

# 4. Subsampling
cd 04_subsample
python mia_subsample_experiment.py
python evaluate_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv
python plot_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv --out-dir mia_subsample_results/plots
```

Every experiment writes results under `<experiment>_results/raw_data/` as CSV,
and the evaluators/plotters read from there. Runners support resuming: re-running
with results already on disk skips completed configurations.

## Setting

The models are trained **transductively**: each client holds an induced subgraph,
the forward pass runs on the full local adjacency, and the loss is masked to the
client's training nodes. Membership is defined at the level of the training loss.
Client-level unlearning removes whole clients (nodes and edges); node-level
unlearning removes nodes from the training loss while the nodes remain in the
graph. See `docs/` for the exact definitions per experiment.

## Datasets

Datasets are downloaded automatically on first use (Planetoid, Amazon, Coauthor,
WikiCS, Actor via PyTorch Geometric; the heterophilous datasets from the Yandex
`heterophilous-graphs` repository). For Chameleon and Squirrel we use the
**filtered** versions of Platonov et al. (2023), which remove duplicated nodes
that otherwise leak information between train and test.

## Notes on reproducibility

- Results are averaged over multiple seeds; per-run numbers vary slightly with
  hardware and library versions.
- The heterophilous datasets are loaded from `.npz` files whose edges are stored
  as `(E, 2)`; the loader transposes them to `(2, E)` before building the
  adjacency.
