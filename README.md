# Reference-Relative MIA for Federated Graph Unlearning

Code and results for the paper **"Rethinking Membership Inference for Unlearning:
A Reference-Relative View on Federated Graphs."**

We study whether a membership inference attack (MIA) is a reliable way to verify
unlearning in federated graph learning. Instead of attacking an approximately
unlearned model, we apply a frozen, out-of-sample-calibrated attack to the
**retrain-from-scratch (RTS)** reference — the model whose forgetting state is known
by construction — and read the forget set against a **matched supervised non-member
reference** rather than treating the absolute attack score as a standalone forgetting
measure. We evaluate three attacks (shadow-model, modified prediction-entropy, and
offline RMIA) across 13 datasets and 5 seeds (65 cells).

The paper's code and results live in **`07_mia_redesigned/`**. Earlier exploratory
work and superseded iterations are preserved in **`archive/`**.

---

## Repository layout

```
.
├── 07_mia_redesigned/        Canonical pipeline and results for the paper
│   ├── models.py             GNN model, dataset loaders, per-dataset config
│   ├── partition.py          Louvain community-aware partitioning + splits
│   ├── train.py              federated training; original (M0) and RTS (MR) models
│   ├── attack.py             shadow-model MIA, out-of-sample calibration at alpha
│   ├── metrics.py            train/test generalization gap
│   ├── nodelog.py            per-node logging + run manifest
│   ├── experiment_main.py    Stage 1: core run -> results_13x5/raw/
│   ├── build_references.py   Stage 2: reference-model pack per cell
│   ├── score_and_match.py    Stage 3: entropy + RMIA scores, common-pool AUC
│   ├── rethreshold_rmia.py   Stage 4: RMIA threshold frozen on M0 (review fix)
│   ├── aggregate_extension.py Stage 5: roll up -> summary CSVs
│   ├── make_combined_figures.py Stage 6: paper figures
│   ├── run_full_extension.sh  driver for Stages 2-3 over all 65 cells
│   ├── test_acceptance.py    core-run integrity checks
│   ├── test_extension.py     extension integrity checks
│   ├── summary_core.csv      shadow-model attack, per cell (common pool)
│   ├── summary_entropy.csv   prediction-entropy attack, per cell
│   ├── summary_rmia.csv      offline RMIA, per cell (frozen-M0 threshold)
│   ├── figs/                 fig_rq2rq3.pdf, fig_rq4_overview.pdf, fig_impact.pdf
│   ├── results_13x5/         frozen core run (per-node logs + manifests)
│   ├── extension_results_20260930_174043/  the extension run behind the paper
│   ├── legacy/               superseded scripts/summaries (see its README)
│   ├── README.md             short pointer to this file
│   ├── README_attack_extension.md  detailed extension notes
│   └── requirements.txt
├── archive/                  Prior / superseded / unrelated material (see below)
├── requirements.txt
└── README.md                 (this file)
```

### archive/

Kept for provenance; not part of the paper's pipeline.

- `01_main_study/`–`04_subsample/` — an earlier four-experiment study (main study,
  matched control, partitioning/clients/forget-ratio sweeps, subsampling) with its own
  README (`archive/README_original_study.md`) and per-experiment guides
  (`archive/docs/`).
- `05_mia_redesigned/`, `06_mia_redesigned/` — earlier iterations of the redesigned
  pipeline, superseded by `07_mia_redesigned/`.
- `MIA_2/`, `Subsample/`, `files (8)/`, `mia_control_variation/`, loose `mia_*.py` —
  prototype/scratch code and outputs.
- `plot/` — unrelated reinforcement-learning plots that were in the working tree.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Developed with Python 3.11 and PyTorch 2.x. A GPU is recommended (training all cells is
expensive on CPU). Datasets download automatically on first use via PyTorch Geometric;
the heterophilous datasets come from the Yandex heterophilous-graphs release. For
Chameleon and Squirrel the filtered versions of Platonov et al. (2023) are used
(duplicate nodes removed to prevent train/test leakage).

---

## Pipeline

A **cell** is one (dataset, seed) pair: 13 datasets x 5 seeds = 65 cells. Each cell has
a `<tag>` of the form `<dataset>_client_s<seed>` (e.g. `Cora_client_s42`).

```
experiment_main.py          results_13x5/raw/            (Stage 1: ROOT source)
        |
        v
run_full_extension.sh       extension_results_<stamp>/   (Stages 2-3)
   |- build_references.py        reference/refpack_<tag>.json
   |- score_and_match.py         scores/nodescores_<tag>.csv, scores/attackmeta_<tag>.json
        |
        v
aggregate_extension.py      summary_entropy.csv, summary_rmia.csv   (Stage 5, initial)
        |
        v
rethreshold_rmia.py         summary_rmia_rethresholded.csv + *_commonpool.csv  (Stage 4 fix)
        |
        v
make_combined_figures.py    figs/*.pdf                              (Stage 6)
```

### Stage 1 — core run (`experiment_main.py`)

Partitions each graph, trains the original model **M0** (all clients) and the
retrain-from-scratch reference **MR** (retained clients only), fits and freezes the
shadow-model attack, and logs every evaluated node under both models.

- **Writes:** `results_13x5/raw/nodelog_<tag>.csv` (per-node scores/decisions) and
  `results_13x5/raw/manifest_<tag>.json` (provenance: client counts, split ratios,
  frozen thresholds, attack settings, git SHA). Also produces the shadow-model summary
  used as `summary_core.csv`.

### Stages 2-3 — extension attacks (`run_full_extension.sh`)

- **`build_references.py`** reads `results_13x5/raw/` and writes
  `<out>/reference/refpack_<tag>.json` (reference-model outputs, calibration labels,
  population/evaluation pools).
- **`score_and_match.py`** reads the frozen logs + refpack and writes
  `<out>/scores/nodescores_<tag>.csv` (entropy + RMIA scores/decisions) and
  `<out>/scores/attackmeta_<tag>.json` (per-attack metadata incl. common-pool AUC on
  E+ union E-).
- **`test_extension.py`** spot-checks a few cells.

### Stage 4 — RMIA re-thresholding (`rethreshold_rmia.py`)

Computes the RMIA decision threshold on M0, freezes it, and applies the same threshold
to both M0 and MR (offline; no rescoring or retraining). With `--core-csv` and
`--entropy-csv` it also writes the common-pool versions of the Shadow and Entropy
summaries.

- **Writes:** `<out>/summary_rmia_rethresholded.csv`, `<out>/summary_core_commonpool.csv`,
  `<out>/summary_entropy_commonpool.csv`, and corrected `scores/`. (Output dirs ending
  in `_rmiafix` are regenerable and git-ignored.)

### Stage 5 — aggregate (`aggregate_extension.py`)

Reads all cells' `attackmeta`/`nodescores` and writes `summary_entropy.csv` and
`summary_rmia.csv`. Each row is one cell: `attack_strength_auc`, the forget/held-out
member rates under M0 and MR, and C_cal, C_F, C_H.

### Stage 6 — figures (`make_combined_figures.py`)

Reads the three summaries and writes:
`figs/fig_rq2rq3.pdf` (C_cal per dataset; C_F distribution),
`figs/fig_rq4_overview.pdf` (F/H transition, shared y-axis; C_cal vs C_F),
`figs/fig_impact.pdf` (pre/post F-H on discriminative cells; per-dataset C_H).

---

## Reproduce

All commands run from inside `07_mia_redesigned/`.

### A. Figures only (from the committed summaries) — fastest

```bash
cd 07_mia_redesigned
python make_combined_figures.py \
  --attacks shokri=summary_core.csv entropy=summary_entropy.csv rmia=summary_rmia.csv \
  --out figs
```

### B. Summaries + figures (from the committed core run)

```bash
cd 07_mia_redesigned

# Stages 2-3: build references + score all 65 cells (reads results_13x5/raw/)
bash run_full_extension.sh                      # writes extension_results_<stamp>/

# Stage 5: initial aggregate
python aggregate_extension.py \
  --ext-dir extension_results_<stamp> --alpha 0.10 --out-dir .

# Stage 4: freeze RMIA threshold on M0 and write common-pool summaries
python rethreshold_rmia.py \
  --ext-dir extension_results_<stamp> \
  --out-dir extension_results_<stamp>_rmiafix \
  --core-csv summary_core.csv --entropy-csv summary_entropy.csv

# Adopt the corrected summaries as canonical:
cp extension_results_<stamp>_rmiafix/summary_rmia_rethresholded.csv summary_rmia.csv
cp extension_results_<stamp>_rmiafix/summary_core_commonpool.csv    summary_core.csv
cp extension_results_<stamp>_rmiafix/summary_entropy_commonpool.csv summary_entropy.csv

# Stage 6: figures
python make_combined_figures.py \
  --attacks shokri=summary_core.csv entropy=summary_entropy.csv rmia=summary_rmia.csv \
  --out figs
```

To reproduce the paper's numbers exactly, use the committed run
`extension_results_20260930_174043` as `--ext-dir`.

### C. Everything from scratch (retrains all cells — expensive)

```bash
cd 07_mia_redesigned

# Stage 1: core run — all 13 datasets x 5 seeds, frozen M0/MR logs
python experiment_main.py \
  --datasets Cora PubMed CS Photo Computers WikiCS Chameleon Squirrel Actor \
             tolokers minesweeper amazon_ratings roman_empire \
  --seeds 7 42 99 123 456 \
  --scenario client --n-shadow 5 --core --out-dir results_13x5

# then Stages 2-6 as in B above
```

`experiment_main.py` flags: `--datasets`, `--seeds`, `--scenario` (default `client`),
`--out-dir` (default `results`), `--n-shadow` (default 5), `--core`, `--overwrite`.
The paper's core run used `--out-dir results_13x5`.

---

## Configuration

13 datasets x 5 seeds (7, 42, 99, 123, 456) = 65 cells. Operating point alpha = 0.10
(nominal; threshold from empirical quantiles, so the realized rate is not always exactly
0.10). Client-level removal with forget ratio rho = 0.20. Offline RMIA with a = 1,
5 OUT reference models, gamma = 2. Per-dataset hyper-parameters (clients, split ratios,
LR, epochs, hidden dim, rounds, optimizer) are in the paper's configuration tables.

### Key quantities (per cell, in the summaries)

- `attack_strength_auc` — AUC on the common pool E+ union E- under MR (comparable across
  attacks).
- `C_cal = FPR(H; MR) - alpha` — drift of the calibrated operating point on the matched
  reference.
- `C_F = FPR(F; MR) - FPR(H; MR)` — forget set vs matched reference under MR
  (approx. 0 means it behaves like a non-member).
- `C_H = FPR(H; MR) - FPR(H; M0)` — shift of the same non-member reference between models.
- `forget_member_rate_M0/MR`, `heldout_member_rate_M0/MR` — realized member rates.

---

## Tests

```bash
cd 07_mia_redesigned
python test_acceptance.py      # core-run integrity (reads results_13x5/)
python test_extension.py --tag Cora_client_s42 \
  --frozen-dir results_13x5/raw --out-dir extension_results_<stamp>
```

See `07_mia_redesigned/README_attack_extension.md` for detailed extension notes and
`07_mia_redesigned/legacy/README.md` for the archived scripts.