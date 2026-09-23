# Experiment 3 — Sensitivity to Federated Configuration

## Question
Does the membership score depend on modelling choices that do not change the fact
that the forget set is excluded — the graph partitioning method, the number of
clients, and the forget ratio?

## What it does
Three sweeps, each varying one axis while holding the others at their defaults,
then retraining the gold standard and running the attack for every setting.

## Configuration (`mia_variations_config.py`, `run_sweeps.sh`)
- Datasets: 7 representative datasets.
- Scenarios: client and node.
- Seeds: 3.
- Sweep 1 — partitioning: {louvain, metis, metis_plus, random}; K and forget
  ratio at per-dataset / per-scenario defaults.
- Sweep 2 — number of clients K: {3, 5, 10, 15}; partitioning = louvain, forget
  ratio default.
- Sweep 3 — forget ratio: {0.05, 0.10, 0.20, 0.30}; partitioning = louvain, K
  default.
- Each sweep: 7 x 2 x 3 x 4 = 168 runs; three sweeps = 504 runs.

## Files
- `partitioning.py` — Louvain / Metis / Metis-Plus / random partitioners and
  Newman–Girvan modularity. Metis inputs are sanitized (self-loops removed,
  deduplicated, symmetrized, isolated nodes patched) to avoid pymetis crashes.
- `mia_variations_config.py` — sweep grids and defaults.
- `mia_variations_experiment.py` — runner; supports `--skip-existing` to resume.
- `run_sweeps.sh` — orchestrates the three sweeps.
- `evaluate_variations.py` — sensitivity tables, correlations, regression.
- `plot_variations.py` — sensitivity plots (AUC / BalAcc / TPR panels).
- `plot_figure1_gap.py` — the pooled train/test gap vs MIA AUC scatter.
- `plot_property_analysis.py`, `plot_predictor_analysis.py` — property and
  predictor analyses.

## Run
```bash
bash run_sweeps.sh all            # or: partitioning | num_clients | forget_ratio
python evaluate_variations.py --dir mia_variations_results/raw_data
python plot_figure1_gap.py --dir mia_variations_results/raw_data --out-dir mia_variations_results/plots
python plot_variations.py  --dir mia_variations_results/raw_data --out-dir mia_variations_results/plots
```

## Output
`mia_variations_results/raw_data/mia_variations_raw_sweep_*.csv` — one file per
sweep, one row per run, with the attack metrics, the train/test gap, and the
partition modularity.

## Reading it
The sensitivity tables show whether each metric moves with the swept axis. The
gap-vs-AUC scatter pools all runs; a tight relationship there indicates the score
tracks the generalization gap rather than the configuration.
