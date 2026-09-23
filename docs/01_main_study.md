# Experiment 1 — MIA on the Gold-Standard Retrained Model

## Question
On the retrain-from-scratch model, which never sees the forget set, how does the
membership inference score behave across datasets and unlearning granularities?
If the score measured forgetting it should sit near chance (AUC 0.5); we record
how far it departs from chance and how it varies.

## What it does
For each (dataset, scenario, seed) it partitions the graph across clients, trains
the gold-standard model on the retain set with FedAvg, runs the Shokri
shadow-model attack, and records the attack metrics together with the model's
train/test accuracy gap.

## Configuration
- Datasets: the whitelist in `mia_config.py` (`DATASETS_ALL`) — 13 datasets
  spanning homophilic (Cora, PubMed, CS, Photo, Computers, WikiCS) and
  heterophilous (Chameleon-filtered, Squirrel-filtered, Actor, tolokers,
  minesweeper, amazon_ratings, roman_empire) graphs.
- Scenarios: `client` (remove whole clients) and `node` (remove nodes within a
  client).
- Seeds: 5.
- Combinations: 13 x 2 x 5 = 130 runs, each training 1 gold-standard model + 5
  shadow models.

## Files
- `all_client_unlearning.py` — PlainGCN model and dataset loaders.
- `mia_attacks.py` — Shokri shadow attack and metric computation.
- `mia_config.py` — dataset whitelist and split ratios.
- `mia_experiment.py` — main runner.
- `mia_hp_search.py` — optional hyper-parameter search.
- `evaluate_mia_consistency.py` — aggregation, summary tables, headline figure.
- `print_size_class_auc.py` — per-scenario table of nodes/classes/BalAcc/AUC.

## Run
```bash
python mia_experiment.py
python evaluate_mia_consistency.py --csv mia_consistency_results/raw_data/mia_consistency_raw.csv
python print_size_class_auc.py --csv mia_consistency_results/raw_data/mia_consistency_raw.csv
```

## Output
`mia_consistency_results/raw_data/mia_consistency_raw.csv` — one row per run with
the attack metrics (accuracy, balanced accuracy, AUC, TPR at low FPR) and the
train/test gap. The evaluator writes summary CSVs and a figure to
`mia_consistency_results/evaluation/`.

## Reading it
Report AUC and balanced accuracy per (dataset, scenario). AUC well above 0.5 on
a model that never saw the forget set is the phenomenon the paper investigates.
Sorting datasets by AUC shows the relationship to dataset size and number of
classes.
