# MIA Consistency Experiment

Empirical test of the central claim: **MIA is not all you need** for unlearning
evaluation. Even on the gold-standard retrain-from-scratch model — by
construction perfectly unlearned — MIA predictions vary substantially across
seeds, making MIA an unreliable verification metric.

## Design

For each `(dataset, scenario, seed)`:

1. Load + Louvain-partition + per-client train/val/test split (partition-first)
2. Train original FedAvg model `M_orig`
3. Define forget set:
   - **client**: 20% of clients (random per seed)
   - **node**: one target client + 10% of its training nodes
4. Retrain gold standard `M_ret` on retain set (FedAvg from scratch)
5. Compute utility / unlearning metrics on `M_ret` (see *Evaluation*)
6. Train `N=5` shadow FedAvg models on disjoint random client subsets
7. **Shokri shadow MIA** — per-class attack models (PyTorch port of
   [csong27/membership-inference](https://github.com/csong27/membership-inference))
8. **LR confidence MIA** — comparison attack on `[max_conf, entropy, margin, true_conf]`
9. Save row

## Evaluation protocol

| Scenario | `test_acc` / `retain_acc` / `forget_acc` |
|----------|------------------------------------------|
| client   | mean over clients of `M_ret(cid=c, ...)` on local mask |
| node     | `M_ret(cid=target_idx, ...)` on **target client's** local mask only |

Forgetting score = `JSD(P_orig ‖ P_ret)` on forget nodes, using the same
`forward(cid=...)` convention as accuracy.

## Attack architecture (faithful Shokri port)

PyTorch port of `csong27/membership-inference/classifier.py`:

- `nn`: 1 hidden layer + `tanh` + softmax (Adam, L2 weight decay)
- `softmax`: logistic regression (the default in the reference)
- One attack model **per data class** (Shokri's design)
- Features: softmax probability vector
- Fallback: pooled-class attack model for sparse classes (`< 10` samples or
  only one membership label present)

## Files

```
mia_config.py        seeds, datasets, scenarios, MIA defaults & HP grid
mia_attacks.py       ShokriShadowMIA, LRConfidenceMIA, detailed metrics
mia_experiment.py    federated adapters, gold retrain, main runner
mia_hp_search.py     per-(dataset, scenario) attack-model HP search
```

## Usage

```bash
# 1. (Optional) Find best attack HPs per (dataset, scenario)
python mia_hp_search.py --datasets Cora PubMed --scenarios client node

# 2. Run full consistency experiment with default HPs
python mia_experiment.py \
    --datasets  Cora PubMed CS Photo Tolokers minesweeper Amazon-ratings \
    --scenarios client node \
    --seeds     7 42 99 100 999

# 3. Run with the searched HPs
python mia_experiment.py \
    --hp-path mia_consistency_results/hp_search/best_attack_configs.json
```

## Output

- `mia_consistency_results/raw_data/mia_consistency_raw.csv` — one row per
  `(dataset, scenario, seed)` with all metrics flattened
- `.json` — same data, with the nested `per_data_class` breakdowns preserved
- `mia_consistency_results/hp_search/best_attack_configs.json` — per-dataset HPs

## Headline metrics for the paper

The argument hinges on **variance of MIA outputs across seeds on the gold
standard**. Key columns to plot/tabulate:

- `shokri_mia_acc`, `shokri_mia_auc`, `shokri_mia_f1`
- `shokri_forget_predicted_member_rate` — should be ≈ 0 if MIA is reliable
  (forget nodes are not in `M_ret`'s training); large variance ⇒ inconsistent
- `shokri_per_data_class` — per-class breakdown
- `lr_*` versions for comparison

Std across the 5 seeds is the signal. Large σ on a gold-standard model is the
core "MIA is not all you need" evidence.
