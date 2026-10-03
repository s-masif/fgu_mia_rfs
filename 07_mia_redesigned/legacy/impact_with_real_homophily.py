"""
impact_with_real_homophily.py
Uses YOUR measured per-node homophily (local_label_agreement from nodescores) aggregated
to a per-dataset value, instead of literature homophily. Re-tests Plot B and prints the
per-dataset homophily table so you can verify it.

Run from 07_mia_redesigned/ :
  python impact_with_real_homophily.py --ext-dir extension_results_YYYYMMDD_HHMMSS
Needs summary_core/entropy/rmia.csv in cwd and the extension scores/ for homophily.
"""
import argparse, glob, json
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("--ext-dir", required=True, help="extension results folder (has scores/)")
a = ap.parse_args()

# ---- 1. per-dataset homophily from YOUR nodescores (local_label_agreement) ----
# average over all nodes across all cells of each dataset (H nodes are the reference;
# use all nodes for a stable per-dataset homophily estimate).
rows=[]
for f in glob.glob(str(Path(a.ext_dir)/"scores"/"nodescores_*.csv")):
    df=pd.read_csv(f, usecols=["dataset","local_label_agreement"])
    rows.append(df)
allnodes=pd.concat(rows, ignore_index=True)
homo = allnodes.groupby("dataset").local_label_agreement.mean()
print("="*70)
print("PER-DATASET HOMOPHILY (your measured local_label_agreement, mean over nodes)")
print("="*70)
for ds,v in homo.sort_values(ascending=False).items():
    print(f"  {ds:16s} {v:.3f}")

# ---- 2. attach to summaries and re-test Plot B ----
ATTACKS={"Shokri":"summary_core.csv","Entropy":"summary_entropy.csv","RMIA":"summary_rmia.csv"}
def load(f):
    d=pd.read_csv(f)
    if "C_H" not in d: d["C_H"]=d.heldout_member_rate_MR-d.heldout_member_rate_M0
    d["homophily"]=d.dataset.map(homo)
    return d
data={n:load(f) for n,f in ATTACKS.items()}

def corr(x,y):
    m=~(x.isna()|y.isna())
    return pearsonr(x[m],y[m]) if m.sum()>=4 else (float('nan'),float('nan'))

print("\n"+"="*70)
print("PLOT B (real homophily): correlation with C_cal spread and C_F")
print("="*70)
for n,d in data.items():
    rc,pc=corr(d.homophily, d.C_cal.abs())
    rf,pf=corr(d.homophily, d.C_F.abs())
    ra,pa=corr(d.homophily, d.attack_strength_auc)
    print(f"{n:9s} corr(homo,|C_cal|)={rc:+.2f}(p={pc:.2f})  "
          f"corr(homo,|C_F|)={rf:+.2f}(p={pf:.2f})  corr(homo,AUC)={ra:+.2f}(p={pa:.2f})")

print("\n  C_cal std by homophily group (your values):")
for n,d in data.items():
    hi=d[d.homophily>=d.homophily.median()]; lo=d[d.homophily<d.homophily.median()]
    print(f"    {n:9s} high-homo C_cal std={hi.C_cal.std():.3f}  low-homo C_cal std={lo.C_cal.std():.3f}")

# ---- 3. confounding check: is homophily just a proxy for attack strength? ----
print("\n"+"="*70)
print("CONFOUNDING CHECK: homophily vs attack strength (AUC)")
print("="*70)
for n,d in data.items():
    ra,pa=corr(d.homophily, d.attack_strength_auc)
    print(f"  {n:9s} corr(homophily, AUC)={ra:+.2f} (p={pa:.2f})"
          + ("  <- strong: homophily & AUC entangled, interpret with care" if abs(ra)>0.5 else ""))