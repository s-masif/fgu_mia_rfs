#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run_sweeps.sh — one-shot script for Phase 2 (variation sweeps)
#
#  Runs three sweeps sequentially:
#    1. Partitioning:  4 methods × 4 datasets × 2 scenarios × 3 seeds =  96 cells
#    2. Num clients K: 4 values  × 4 datasets × 2 scenarios × 3 seeds =  96 cells
#    3. Forget ratio:  4 values  × 4 datasets × 2 scenarios × 3 seeds =  96 cells
#
#  Total ~288 cells. Each writes to a separate CSV so you can inspect
#  results per sweep or aggregate with evaluate_variations.py.
#
#  Estimated runtime on 4 small datasets: ~8-12 hours. Use tmux.
#
#  Usage:
#    bash run_sweeps.sh partitioning   # just sweep 1
#    bash run_sweeps.sh num_clients    # just sweep 2
#    bash run_sweeps.sh forget_ratio   # just sweep 3
#    bash run_sweeps.sh all            # all three (default)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(Cora Chameleon Squirrel Actor)
SCENARIOS=(client node)
SEEDS=(7 42 99)

sweep_partitioning() {
  echo "══ SWEEP 1: Partitioning ═════════════════════════════════════════════"
  python mia_variations_experiment.py \
    --datasets      "${DATASETS[@]}" \
    --scenarios     "${SCENARIOS[@]}" \
    --seeds         "${SEEDS[@]}" \
    --partitionings louvain metis metis_plus random \
    --num-clients   0 \
    --forget-ratios None \
    --out-tag       sweep_partitioning \
    --skip-existing
}

sweep_num_clients() {
  echo "══ SWEEP 2: Number of clients K ═════════════════════════════════════"
  # Note: default (K=0 → per-dataset default) is included via louvain baseline
  # in sweep_partitioning. Here we vary K explicitly.
  python mia_variations_experiment.py \
    --datasets      "${DATASETS[@]}" \
    --scenarios     "${SCENARIOS[@]}" \
    --seeds         "${SEEDS[@]}" \
    --partitionings louvain \
    --num-clients   3 5 10 15 \
    --forget-ratios None \
    --out-tag       sweep_num_clients \
    --skip-existing
}

sweep_forget_ratio() {
  echo "══ SWEEP 3: Forget ratio ═════════════════════════════════════════════"
  # Note: scenario handling — the runner treats forget_ratio the same for
  # both scenarios (fraction of applicable pool). We sweep a common range;
  # analyse by scenario after the fact.
  python mia_variations_experiment.py \
    --datasets      "${DATASETS[@]}" \
    --scenarios     "${SCENARIOS[@]}" \
    --seeds         "${SEEDS[@]}" \
    --partitionings louvain \
    --num-clients   0 \
    --forget-ratios 0.05 0.10 0.20 0.30 \
    --out-tag       sweep_forget_ratio \
    --skip-existing
}

case "${1:-all}" in
  partitioning) sweep_partitioning ;;
  num_clients)  sweep_num_clients  ;;
  forget_ratio) sweep_forget_ratio ;;
  all)          sweep_partitioning && sweep_num_clients && sweep_forget_ratio ;;
  *) echo "Usage: $0 {partitioning|num_clients|forget_ratio|all}"; exit 1 ;;
esac

echo "══ Done. Aggregate results and analyse: ══════════════════════════════"
echo "  python evaluate_variations.py --dir mia_variations_results/raw_data"
