# Legacy / superseded files

Kept for provenance only. These are **not** part of the canonical pipeline that
produced the paper's results (see ../README.md). Nothing here is deleted; it is
retained for reference.

Scripts
- `make_impact_figures.py`, `make_multi_attack_figures.py` — earlier figure scripts,
  superseded by `../make_combined_figures.py`.
- `impact_analysis_stats.py`, `impact_with_real_homophily.py` — homophily-correlation
  analysis, removed from the paper during review.
- `print_all_stats.py` — ad-hoc stats dump.
- `check_refpack.py` — one-off reference-pack validation.
- `cora42_attacks.txt` — scratch output.

Superseded summaries
- `summary_rmia_old.csv`, `summary_rmia_old_2.csv` — RMIA summaries before the
  frozen-threshold (M0) re-thresholding correction.
- `summary_core_fullN_backup.csv` — Shadow AUC on the full-N pool, before switching
  to the common evaluation pool (E+ union E-).

Folders (retained locally, excluded from version control due to size)
- `extension_results_YYYYMMDD_HHMMSS_rmiafix/` — created by running a usage-example
  command verbatim; superseded.
- `dryrun_test/` — scratch test output.
