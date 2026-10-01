#!/usr/bin/env bash
# =====================================================================================
# run_full_extension.sh
# Runs the full entropy + RMIA extension on all 13 datasets x 5 seeds (65 cells).
#   Stage 1: build 5 reference models per cell   (build_references.py)
#   Stage 2: score each cell (entropy + RMIA)    (score_and_match.py)
#   Stage 3: spot-check a few cells              (test_extension.py)
# All outputs + logs go into a separate timestamped results folder.
#
# Usage:   bash run_full_extension.sh
# Run this from the directory that contains build_references.py etc.
# (…/07_mia_redesigned/). Adjust FROZEN_DIR below if your frozen M0/MR live elsewhere.
# =====================================================================================
set -euo pipefail

# ---- configuration ----
DATASETS=(Cora PubMed CS Photo Computers WikiCS Chameleon Squirrel Actor tolokers minesweeper amazon_ratings roman_empire)
SEEDS=(7 42 99 123 456)
FROZEN_DIR="results_13x5/raw"                       # frozen M0/MR from the 13x5 run
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="extension_results_${STAMP}"                # everything lands here
LOG_DIR="${OUT_DIR}/logs"
REF_SEED_BASE=90000                                 # matches the validated Cora s42 run

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
echo "Results folder: ${OUT_DIR}"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Frozen:   ${FROZEN_DIR}"
echo "Started:  $(date)" | tee "${OUT_DIR}/RUN_INFO.txt"

# ---- sanity check: frozen dir exists ----
if [ ! -d "${FROZEN_DIR}" ]; then
  echo "ERROR: frozen dir '${FROZEN_DIR}' not found. Fix FROZEN_DIR and re-run." >&2
  exit 1
fi

# =====================================================================================
# STAGE 1 — build reference models for all cells (the expensive step)
# =====================================================================================
echo ""
echo "================ STAGE 1: building reference models (all 65 cells) ================"
python build_references.py \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --frozen-dir "${FROZEN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --ref-seed-base "${REF_SEED_BASE}" \
  2>&1 | tee "${LOG_DIR}/stage1_build_references.log"
echo "Stage 1 done: $(date)"

# =====================================================================================
# STAGE 2 — score every cell (entropy + RMIA)
# =====================================================================================
echo ""
echo "================ STAGE 2: scoring all 65 cells ================"
n_ok=0; n_fail=0
: > "${OUT_DIR}/failed_cells.txt"
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="${ds}_client_s${seed}"
    echo "  scoring ${tag} ..."
    if python score_and_match.py \
         --tag "${tag}" \
         --frozen-dir "${FROZEN_DIR}" \
         --out-dir "${OUT_DIR}" \
         > "${LOG_DIR}/score_${tag}.log" 2>&1; then
      n_ok=$((n_ok+1))
    else
      n_fail=$((n_fail+1))
      echo "${tag}" >> "${OUT_DIR}/failed_cells.txt"
      echo "    !! FAILED (see ${LOG_DIR}/score_${tag}.log)"
    fi
  done
done
echo "Stage 2 done: ${n_ok} ok, ${n_fail} failed. $(date)"
if [ "${n_fail}" -gt 0 ]; then
  echo "Failed cells listed in ${OUT_DIR}/failed_cells.txt"
fi

# =====================================================================================
# STAGE 3 — spot-check a few cells with the acceptance test
# =====================================================================================
echo ""
echo "================ STAGE 3: spot-check acceptance tests ================"
for tag in Cora_client_s42 roman_empire_client_s7 Chameleon_client_s99; do
  echo "  testing ${tag} ..."
  python test_extension.py \
    --tag "${tag}" \
    --frozen-dir "${FROZEN_DIR}" \
    --out-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/test_${tag}.log" || echo "    (test reported issues for ${tag})"
done

# =====================================================================================
# summary
# =====================================================================================
echo ""
echo "================ DONE ================"
echo "Finished: $(date)" | tee -a "${OUT_DIR}/RUN_INFO.txt"
echo "Scored cells: ${n_ok}/65 ok, ${n_fail} failed" | tee -a "${OUT_DIR}/RUN_INFO.txt"
echo "All outputs in: ${OUT_DIR}/"
echo "  - reference models + scores + attackmeta"
echo "  - logs/            (per-cell logs)"
echo "  - failed_cells.txt (if any)"
echo "  - RUN_INFO.txt     (start/finish, counts)"