"""
impact_analysis_stats.py — text-only exploration of candidate impact analyses.
Tells you which additional robustness plots have a real signal, before you commit to figures.
Run: python impact_analysis_stats.py   (needs summary_core/entropy/rmia.csv)
"""
import pandas as pd, numpy as np
from scipy.stats import pearsonr

ATTACKS={"Shokri":"summary_core.csv","Entropy":"summary_entropy.csv","RMIA":"summary_rmia.csv"}
# standard homophily values (edge homophily) for these datasets (from Platonov et al. 2023 /
# standard sources). VERIFY against your own graph stats before using in the paper.
HOMOPHILY={"Cora":0.81,"PubMed":0.80,"CS":0.81,"Photo":0.83,"Computers":0.78,"WikiCS":0.65,
           "Chameleon":0.23,"Squirrel":0.22,"Actor":0.22,"tolokers":0.59,"minesweeper":0.68,
           "amazon_ratings":0.38,"roman_empire":0.05}

def load(f):
    d=pd.read_csv(f)
    if "C_H" not in d: d["C_H"]=d.heldout_member_rate_MR-d.heldout_member_rate_M0
    d["homophily"]=d.dataset.map(HOMOPHILY)
    return d
data={n:load(f) for n,f in ATTACKS.items()}

def corr(x,y):
    m=~(x.isna()|y.isna())
    if m.sum()<4: return (float('nan'),float('nan'))
    return pearsonr(x[m],y[m])

print("="*78)
print("PLOT A CANDIDATE: does C_F stay flat vs AUC, while C_cal does not?")
print("(the robustness 'money plot' — relative is stable, absolute is not)")
print("="*78)
for n,d in data.items():
    r_cal,p_cal=corr(d.attack_strength_auc, d.C_cal.abs())
    r_cf ,p_cf =corr(d.attack_strength_auc, d.C_F.abs())
    print(f"{n:9s} corr(AUC,|C_cal|)={r_cal:+.2f}(p={p_cal:.2f})  "
          f"corr(AUC,|C_F|)={r_cf:+.2f}(p={p_cf:.2f})")
print("  -> interpret: |C_cal| may rise/fall with AUC; |C_F| should be ~uncorrelated & tiny")

print("\n"+"="*78)
print("PLOT B CANDIDATE: homophily impact on C_cal spread and C_F")
print("="*78)
for n,d in data.items():
    r_cal,p_cal=corr(d.homophily, d.C_cal.abs())
    r_cf ,p_cf =corr(d.homophily, d.C_F.abs())
    r_auc,p_auc=corr(d.homophily, d.attack_strength_auc)
    print(f"{n:9s} corr(homophily,|C_cal|)={r_cal:+.2f}(p={p_cal:.2f})  "
          f"corr(homophily,|C_F|)={r_cf:+.2f}(p={p_cf:.2f})  "
          f"corr(homophily,AUC)={r_auc:+.2f}(p={p_auc:.2f})")
# per-homophily-group means
print("\n  C_cal spread (std) by homophily group:")
for n,d in data.items():
    hi=d[d.homophily>=0.6]; lo=d[d.homophily<0.6]
    print(f"    {n:9s} homophilous(>=.6) C_cal std={hi.C_cal.std():.3f}  "
          f"heterophilous(<.6) C_cal std={lo.C_cal.std():.3f}")

print("\n"+"="*78)
print("PLOT C CANDIDATE: cross-attack consistency of C_H (same datasets move?)")
print("="*78)
# correlation of per-dataset mean C_H between attacks
cH={n:data[n].groupby('dataset').C_H.mean() for n in ATTACKS}
import itertools
for a,b in itertools.combinations(ATTACKS,2):
    r,p=pearsonr(cH[a],cH[b])
    print(f"  corr(C_H_{a}, C_H_{b}) = {r:+.2f} (p={p:.3f})  -> high = same datasets move under both")
print("\n  datasets with |mean C_H|>0.1 under ALL three attacks:")
big=[ds for ds in cH['Shokri'].index if all(abs(cH[n][ds])>0.1 for n in ATTACKS)]
print("   ", big)

print("\n"+"="*78)
print("EXTRA: per-attack C_cal bias (does an attack systematically over/under-flag H?)")
print("="*78)
for n,d in data.items():
    print(f"  {n:9s} C_cal mean={d.C_cal.mean():+.3f}  (>0 = over-flags H under MR)")

print("\n"+"="*78)
print("VERDICT GUIDE")
print("="*78)
print("  Plot A worth it IF: corr(AUC,|C_F|) is weak/near zero AND |C_F| small -> relative robust")
print("  Plot B worth it IF: corr(homophily, |C_cal|) or AUC is clear (|r|>0.4) -> graph-property effect")
print("  Plot C worth it IF: corr(C_H) between attacks is high (>0.6) -> H-transition is setting-driven")