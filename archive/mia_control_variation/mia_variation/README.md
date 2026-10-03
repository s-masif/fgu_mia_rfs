# MIA Control Experiment

**Phase 1 of the MIA-consistency study.**

Following the professor's request:

> *"Start with the control: same attack, but replace the forget set with
> nodes the model never saw, from the same distribution as the retain set.
> If the AUC stays high, the false positive is proven."*

## Design

For each `(dataset, scenario, seed)`:

1. **Three-way split at partition time:**
   - `retain` — used for training
   - `forget` — the "true" forget set (matched-size, same rule)
   - `heldout H` — matched control (same size, same rule, non-overlapping with forget)

2. **Train `M_ret_control` on retain only.** Both `forget` and `H` are unseen by this model — they are indistinguishable from its perspective.

3. **Run the same Shokri MIA attack on `M_ret_control`.** Same shadow models, same per-class classifier as in the main study.

4. **Evaluate the attack twice on the same trained model:**
   - **Original AUC** — query pool = `{retain (member=1), test (0), forget (0)}`
   - **Control AUC**  — query pool = `{retain (member=1), test (0), H (0)}`

5. **Compare.** If the two AUCs are close (mean absolute diff < 0.03), the "high AUC" reported in the main study is not forget-specific — you get the same number with any matched unseen group. False positive proven.

## Scope

- **Datasets:** Cora, Chameleon, Squirrel, Actor (the 4 with highest MIA AUC in the main study)
- **Scenarios:** client (20% forget clients + 20% heldout clients), node (10% forget nodes + 10% heldout nodes)
- **Seeds:** 7, 42, 99 — **n_seeds = 3 per (dataset, scenario) cell** — all reported μ ± σ are across these three seeds
- **Cells:** 4 × 2 × 3 = 24 experimental cells

## Metrics reported per cell

Both the original and control query pools report:

| Metric | Purpose |
|---|---|
| `mia_acc`, `mia_f1`, `mia_auc` | Standard MIA metrics |
| `mia_balanced_acc` | `(TPR + TNR) / 2` — robust to pool imbalance (addresses "Acc < 0.5 with AUC > 0.5" issue caused by unbalanced query pools) |
| `mia_tpr_at_fpr_0.001` | Carlini et al. 2022: TPR at FPR ≤ 0.001 — the low-FPR regime that matters for real attackers |
| `mia_tpr_at_fpr_0.01` | Same, FPR ≤ 0.01 |
| `mia_tpr_at_fpr_0.1` | Same, FPR ≤ 0.1 |

## Files

| File | Purpose |
|---|---|
| `all_client_unlearning.py` | Shared federated + partitioning code, including `PlainGCN` |
| `mia_attacks.py` | Shokri shadow MIA + detailed metrics (unchanged from main study) |
| `mia_control_config.py` | Config: datasets, seeds, ratios |
| `mia_control_experiment.py` | Main runner — three-way split, two AUC evaluation |
| `evaluate_control.py` | Post-run analysis + comparison table + Wilcoxon test |
| `README.md` | This file |

## Usage

```bash
# 1. Smoke test on one small dataset (~5 min)
python mia_control_experiment.py \
    --datasets Chameleon --scenarios client --seeds 7 --out-tag smoke

# 2. Full sweep (4 datasets × 2 scenarios × 3 seeds ≈ 2–3 hours)
python mia_control_experiment.py

# 3. Evaluate the collected results
python evaluate_control.py \
    --csv mia_control_results/raw_data/mia_control_raw.csv
```

## Output

- `mia_control_results/raw_data/mia_control_raw.csv` — one row per (dataset, scenario, seed) with both AUCs
- `mia_control_results/evaluation/mia_control_summary.csv` — aggregated μ ± σ table
- Terminal output includes the paired Wilcoxon test result and a headline sentence for the paper

## Expected Result

Original AUC ≈ Control AUC (paired Wilcoxon p > 0.05), i.e. the two are statistically indistinguishable. This proves the "high AUC" is not forget-specific and validates the central claim of the paper: MIA on the retrained baseline picks up structural signal (retain-vs-anything-else), not forget-specific leakage.

## What's Independent from the Main Study

This folder does not share state with the main `mia_consistency` experiment. Even the trained models are new (they exclude both forget and heldout, so they're strictly smaller-retain models than the main study's M_ret). This is by design — the control needs its own training run to have `forget` and `H` in matched positions.
