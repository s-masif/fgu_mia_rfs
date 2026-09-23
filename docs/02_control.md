# Experiment 2 — Matched Held-Out Control

## Question
Is the membership score specific to the forget set, or would any matched group
the model also never saw produce the same score?

## What it does
At partition time each client's nodes are split three ways: retain, forget, and a
held-out set H of the same size drawn by the same rule. The gold-standard model
is trained on the retain set only, so both forget and H are excluded from
training in the same way. The attack is then computed twice on the same model:
once scoring the forget set (original AUC) and once scoring H (control AUC).

## Configuration
- Datasets and scenarios: as in `mia_control_config.py` (the 13-dataset set,
  both scenarios).
- Seeds: 3.
- Combinations: 13 x 2 x 3 = 78 runs.
- H is matched to the forget set in size and sampling rule, and is excluded from
  training, so forget and H are exchangeable from the model's point of view.

## Files
- `mia_control_config.py` — datasets, seeds, matched-control settings.
- `mia_control_experiment.py` — three-way split, retrain, two-way scoring.
- `evaluate_control.py` — paired comparison of original vs control AUC, with a
  Wilcoxon test and effect-size interpretation.

## Run
```bash
python mia_control_experiment.py
python evaluate_control.py
```

## Output
`mia_control_results/raw_data/mia_control_raw.csv` — per run, the original AUC
(with forget), the control AUC (with H), and their absolute difference.

## Reading it
If original and control AUC are close (small mean absolute difference), the score
is not specific to the forget set: any matched unseen group yields the same
value. The evaluator reports the mean absolute difference and a significance test;
note that a small effect size can be statistically detectable yet negligible in
magnitude, so both are reported.
