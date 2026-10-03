# Experiment 4 — Isolating Size from Topology

## Question
Smaller datasets tend to show higher MIA AUC, but small datasets also differ in
graph structure. Is the effect due to size, or to topology?

## What it does
Each dataset is shrunk to a fraction of its nodes with a Metropolis–Hastings
random walk (MHRW), which samples nodes approximately uniformly and so preserves
the degree distribution far better than uniform node removal. The topology of the
subsample (average degree, homophily, label modularity) is measured and recorded,
then the gold standard is retrained on the subsample and attacked.

## Configuration (`mia_subsample_config.py`)
- Datasets: 7 (the medium/large ones with room to shrink).
- Scenarios: client and node.
- Fractions: {1.0, 0.75, 0.50, 0.25}.
- Seeds: 3.
- Combinations: 7 x 2 x 4 x 3 = 168 runs.

## Files
- `subsample.py` — MHRW sampler and the topology-report function.
- `mia_subsample_config.py` — datasets, fractions, seeds.
- `mia_subsample_experiment.py` — runs the subsample, retrains, attacks. It reuses
  the training/attack pipeline from `mia_variations_experiment.py` (present in
  this folder) by substituting the loaded dataset with the subsample, so the
  training and attack are identical to experiment 3.
- `evaluate_subsample.py` — topology-check table and size -> gap -> MIA tables.
- `plot_subsample.py` — pooled gap-vs-AUC scatter and per-dataset twin-axis plots
  of gap and AUC against the subsample fraction.

## Run
```bash
python mia_subsample_experiment.py
python evaluate_subsample.py --csv mia_subsample_results/raw_data/mia_subsample_raw.csv
python plot_subsample.py     --csv mia_subsample_results/raw_data/mia_subsample_raw.csv --out-dir mia_subsample_results/plots
```

## Output
`mia_subsample_results/raw_data/mia_subsample_raw.csv` — per run, the subsample's
topology metrics, the train/test gap, and the attack metrics.

## Reading it
First check the topology table: on datasets where average degree, homophily, and
modularity stay roughly constant across fractions, size is isolated cleanly. On
those datasets, compare how the gap and the AUC change as the fraction falls. The
degree drops on dense graphs when they are shrunk, so those are less clean tests
and should be read with that caveat.
