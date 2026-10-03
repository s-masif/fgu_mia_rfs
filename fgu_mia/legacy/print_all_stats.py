"""
print_all_stats.py — complete text summary of the three-attack results.
Run: python print_all_stats.py
Expects summary_core.csv, summary_entropy.csv, summary_rmia.csv in the current dir.
"""
import pandas as pd, numpy as np

ATTACKS = {"Shokri":"summary_core.csv", "Entropy":"summary_entropy.csv", "RMIA":"summary_rmia.csv"}
ALPHA = 0.10

def load(f):
    d = pd.read_csv(f)
    if "C_H" not in d:
        d["C_H"] = d["heldout_member_rate_MR"] - d["heldout_member_rate_M0"]
    return d

data = {name: load(f) for name, f in ATTACKS.items()}

print("="*78)
print("OVERALL PER-ATTACK SUMMARY (65 cells each)")
print("="*78)
print(f"{'Attack':10s} {'AUC':>14s} {'C_cal':>18s} {'C_F':>18s} {'C_H':>18s}")
for name, d in data.items():
    print(f"{name:10s} "
          f"{d.attack_strength_auc.mean():.3f}±{d.attack_strength_auc.std():.3f}   "
          f"{d.C_cal.mean():+.3f}[{d.C_cal.min():+.2f},{d.C_cal.max():+.2f}] "
          f"{d.C_F.mean():+.4f}[{d.C_F.min():+.2f},{d.C_F.max():+.2f}] "
          f"{d.C_H.mean():+.3f}[{d.C_H.min():+.2f},{d.C_H.max():+.2f}]")

print("\n"+"="*78)
print("RQ2: CALIBRATION TRANSFER — C_cal spread (should be WIDE)")
print("="*78)
for name, d in data.items():
    out = (d.C_cal.abs() > 0.05).sum()
    print(f"{name:10s} |C_cal|>0.05 in {out}/65 cells   range [{d.C_cal.min():+.3f}, {d.C_cal.max():+.3f}]")

print("\n"+"="*78)
print("RQ3: FORGET vs REFERENCE — C_F (should be NEAR ZERO)")
print("="*78)
for name, d in data.items():
    within = (d.C_F.abs() < 0.02).sum()
    strong = d[d.attack_strength_auc >= 0.60]
    print(f"{name:10s} C_F mean={d.C_F.mean():+.4f}  |C_F|<0.02 in {within}/65   "
          f"| strong cells(AUC>=.60): n={len(strong)}, "
          f"C_F mean={strong.C_F.mean():+.4f}, max|C_F|={strong.C_F.abs().max() if len(strong) else float('nan'):.3f}")

print("\n"+"="*78)
print("RQ4: REFERENCE STABILITY — C_H (H shifts across models)")
print("="*78)
for name, d in data.items():
    big = (d.C_H.abs() > 0.05).sum()
    print(f"{name:10s} C_H mean={d.C_H.mean():+.3f}  |C_H|>0.05 in {big}/65   "
          f"range [{d.C_H.min():+.3f}, {d.C_H.max():+.3f}]")

print("\n"+"="*78)
print("THE CENTRAL CONTRAST: C_cal spread vs C_F spread (per attack)")
print("="*78)
for name, d in data.items():
    print(f"{name:10s} C_cal std={d.C_cal.std():.3f}  vs  C_F std={d.C_F.std():.3f}   "
          f"(ratio {d.C_cal.std()/d.C_F.std():.1f}x wider)")

print("\n"+"="*78)
print("PER-DATASET AUC (all three attacks)")
print("="*78)
order = data["Shokri"].groupby("dataset").attack_strength_auc.mean().sort_values(ascending=False).index
print(f"{'Dataset':16s} {'Shokri':>8s} {'Entropy':>8s} {'RMIA':>8s}")
for ds in order:
    row = f"{ds:16s}"
    for name in ATTACKS:
        v = data[name][data[name].dataset==ds].attack_strength_auc.mean()
        row += f" {v:8.3f}"
    print(row)

print("\n"+"="*78)
print("PER-DATASET C_cal (all three attacks)")
print("="*78)
print(f"{'Dataset':16s} {'Shokri':>8s} {'Entropy':>8s} {'RMIA':>8s}")
for ds in order:
    row = f"{ds:16s}"
    for name in ATTACKS:
        v = data[name][data[name].dataset==ds].C_cal.mean()
        row += f" {v:+8.3f}"
    print(row)

print("\n"+"="*78)
print("PER-DATASET C_F (all three attacks)")
print("="*78)
print(f"{'Dataset':16s} {'Shokri':>8s} {'Entropy':>8s} {'RMIA':>8s}")
for ds in order:
    row = f"{ds:16s}"
    for name in ATTACKS:
        v = data[name][data[name].dataset==ds].C_F.mean()
        row += f" {v:+8.3f}"
    print(row)

print("\n"+"="*78)
print("PER-DATASET C_H (all three attacks)")
print("="*78)
print(f"{'Dataset':16s} {'Shokri':>8s} {'Entropy':>8s} {'RMIA':>8s}")
for ds in order:
    row = f"{ds:16s}"
    for name in ATTACKS:
        v = data[name][data[name].dataset==ds].C_H.mean()
        row += f" {v:+8.3f}"
    print(row)

print("\n"+"="*78)
print("CROSS-ATTACK AGREEMENT CHECK (does the pattern hold for all 3?)")
print("="*78)
print(f"C_F near zero for ALL three attacks:  "
      f"{all(abs(data[n].C_F.mean())<0.01 for n in ATTACKS)}")
print(f"C_cal spreads (std>0.05) for ALL three: "
      f"{all(data[n].C_cal.std()>0.05 for n in ATTACKS)}")