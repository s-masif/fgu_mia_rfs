"""
Reproducible analysis of the 13x5 client-level MIA-unlearning results.

Run:  python analyze_results.py
Reads summary_core.csv (one row per dataset x seed = 65 cells) and produces
publication-quality figures + statistical tables under analysis/.

The dataset is 13 datasets x 5 seeds. "Datasets" are the units of interest;
"seeds" are repetitions (NOT variables). The variables of interest are the
membership-audit endpoints, not generic columns, so the analysis is built around
them rather than around generic 13x5 EDA.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─────────────────────────── CONFIG (edit here) ─────────────────────────────
DATA_FILE   = "summary_core.csv"      # or use pd.read_excel(...) below
ID_COLUMN   = "dataset"
SEED_COLUMN = "seed"
OUT_DIR     = Path("analysis")
DPI         = 400
VECTOR_EXT  = "pdf"                    # "pdf" or "svg"

# The audit endpoints (the real "variables of interest")
ENDPOINTS = {
    "attack_strength_auc":     "Attack strength (AUC)",
    "C_cal":                   "C_cal  (FPR(H;MR) - alpha)",
    "C_F":                     "C_F  (FPR(F;MR) - FPR(H;MR))",
    "train_test_gap":          "Train/test gap",
    "forget_member_rate_M0":   "Forget member rate (M0)",
    "forget_member_rate_MR":   "Forget member rate (MR)",
    "heldout_member_rate_M0":  "Held-out member rate (M0)",
    "heldout_member_rate_MR":  "Held-out member rate (MR)",
}
ALPHA = 0.10
AUC_STRONG = 0.60                      # informative-cell threshold

# colourblind-friendly palette (Okabe-Ito)
CB = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#F0E442","#000000"]

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.family": "sans-serif",
})

FIG = OUT_DIR/"figures"; TAB = OUT_DIR/"tables"; RES = OUT_DIR/"results"
for d in (FIG,TAB,RES): d.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG/f"{name}.png"); fig.savefig(FIG/f"{name}.{VECTOR_EXT}")
    plt.close(fig)


# ─────────────────────────── 1. LOAD + VALIDATE ─────────────────────────────
def load():
    df = pd.read_csv(DATA_FILE)          # pd.read_excel(DATA_FILE) for xlsx
    log = []
    log.append(f"shape: {df.shape}")
    log.append(f"datasets: {df[ID_COLUMN].nunique()}  seeds: {sorted(df[SEED_COLUMN].unique())}")
    log.append(f"cells: {len(df)}  (expected 13x5=65)")
    log.append(f"missing values: {int(df.isna().sum().sum())}")
    log.append(f"duplicate cells: {int(df.duplicated([ID_COLUMN,'scenario',SEED_COLUMN]).sum())}")
    # per-dataset seed completeness
    inc = df.groupby(ID_COLUMN)[SEED_COLUMN].nunique()
    incomplete = inc[inc != 5]
    if len(incomplete): log.append(f"INCOMPLETE datasets: {incomplete.to_dict()}")
    # value-range sanity for rates (should be in [0,1])
    for c in df.columns:
        if "rate" in c or c in ("attack_strength_auc","acc_train","acc_test"):
            bad = df[(df[c]<-1e-9)|(df[c]>1+1e-9)]
            if len(bad): log.append(f"OUT-OF-RANGE {c}: {len(bad)} rows")
    (RES/"validation.txt").write_text("\n".join(log))
    print("\n".join(log))
    return df


# ─────────────────────────── 2. DESCRIPTIVES ────────────────────────────────
def descriptives(df):
    rows=[]
    for col,label in ENDPOINTS.items():
        x=df[col].dropna().values
        ci=stats.t.interval(0.95,len(x)-1,loc=x.mean(),scale=stats.sem(x)) if len(x)>1 else (np.nan,np.nan)
        cv = x.std(ddof=1)/abs(x.mean()) if x.mean()!=0 else np.nan
        rows.append(dict(variable=col,label=label,mean=x.mean(),median=np.median(x),
                         std=x.std(ddof=1),var=x.var(ddof=1),min=x.min(),max=x.max(),
                         range=x.max()-x.min(),iqr=np.subtract(*np.percentile(x,[75,25])),
                         cv=cv,ci95_lo=ci[0],ci95_hi=ci[1]))
    d=pd.DataFrame(rows).round(4); d.to_csv(TAB/"descriptive_statistics.csv",index=False)
    print("\n=== descriptive statistics ==="); print(d.to_string(index=False))
    return d


# ─────────────────── 3. PER-DATASET AGGREGATE (mean over seeds) ──────────────
def per_dataset(df):
    agg = df.groupby(ID_COLUMN).agg(
        AUC=("attack_strength_auc","mean"),
        F_M0=("forget_member_rate_M0","mean"), F_MR=("forget_member_rate_MR","mean"),
        H_M0=("heldout_member_rate_M0","mean"), H_MR=("heldout_member_rate_MR","mean"),
        C_cal=("C_cal","mean"), C_cal_sd=("C_cal","std"),
        C_F=("C_F","mean"), C_F_sd=("C_F","std"),
        gap=("train_test_gap","mean"),
    )
    agg["H_transition"]=agg.H_MR-agg.H_M0
    agg["F_transition"]=agg.F_MR-agg.F_M0
    agg=agg.sort_values("AUC",ascending=False).round(4)
    agg.to_csv(TAB/"per_dataset_summary.csv")
    print("\n=== per-dataset (mean over seeds) ==="); print(agg.to_string())
    return agg


# ─────────────────────────── 4. CORRELATIONS ────────────────────────────────
def correlations(df):
    cols=list(ENDPOINTS.keys())
    pear=df[cols].corr(method="pearson").round(3)
    spear=df[cols].corr(method="spearman").round(3)
    pear.to_csv(TAB/"correlations_pearson.csv"); spear.to_csv(TAB/"correlations_spearman.csv")
    # targeted pairs with CIs (bootstrap) + p
    pairs=[("train_test_gap","attack_strength_auc"),
           ("train_test_gap","C_F"),
           ("heldout_member_rate_M0","heldout_member_rate_MR"),
           ("attack_strength_auc","C_cal")]
    rows=[]
    rng=np.random.default_rng(0)
    for a,b in pairs:
        x,y=df[a].values,df[b].values
        r,p=stats.pearsonr(x,y); rho,pp=stats.spearmanr(x,y)
        boot=[stats.pearsonr(*(lambda i:(x[i],y[i]))(rng.integers(0,len(x),len(x))))[0] for _ in range(2000)]
        lo,hi=np.percentile(boot,[2.5,97.5])
        rows.append(dict(pair=f"{a} ~ {b}",pearson_r=round(r,3),p=round(p,4),
                         ci95=f"[{lo:+.2f},{hi:+.2f}]",spearman_rho=round(rho,3)))
    c=pd.DataFrame(rows); c.to_csv(TAB/"correlation_targeted.csv",index=False)
    print("\n=== targeted correlations (n=65, bootstrap CI) ==="); print(c.to_string(index=False))
    return pear,spear


# ─────────────────────────── 5. OUTLIERS ────────────────────────────────────
def outliers(df):
    cols=list(ENDPOINTS.keys())
    z=(df[cols]-df[cols].mean())/df[cols].std(ddof=1)
    rows=[]
    for c in cols:
        q1,q3=df[c].quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        mask=(df[c]<lo)|(df[c]>hi)
        for i in df[mask].index:
            rows.append(dict(cell=f"{df.loc[i,ID_COLUMN]}_s{df.loc[i,SEED_COLUMN]}",
                             variable=c,value=round(df.loc[i,c],4),
                             z=round(z.loc[i,c],2),method="IQR"))
    o=pd.DataFrame(rows); o.to_csv(TAB/"outlier_analysis.csv",index=False)
    print("\n=== IQR outliers ==="); print(o.to_string(index=False) if len(o) else "  none")
    return o


# ─────────────────────────── 6. FIGURES ─────────────────────────────────────
def fig_overview(agg):
    fig,ax=plt.subplots(figsize=(11,5))
    cols=["AUC","C_cal","C_F","H_transition","gap"]
    x=np.arange(len(agg)); w=0.16
    for k,c in enumerate(cols):
        ax.bar(x+(k-2)*w,agg[c],w,label=c,color=CB[k])
    ax.axhline(0,color="k",lw=.6); ax.set_xticks(x)
    ax.set_xticklabels(agg.index,rotation=45,ha="right")
    ax.set_ylabel("value (mean over 5 seeds)")
    ax.set_title("Audit endpoints by dataset")
    ax.legend(ncol=5,fontsize=9,frameon=False)
    save(fig,"figure_01_overview")

def fig_distributions(df):
    cols=["attack_strength_auc","C_cal","C_F","train_test_gap"]
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,c in zip(axes,cols):
        ax.violinplot(df[c],showmedians=True)
        ax.scatter(np.random.normal(1,0.04,len(df)),df[c],s=12,alpha=0.5,color=CB[0])
        ax.set_title(ENDPOINTS[c],fontsize=10); ax.set_xticks([])
        if c=="C_cal": ax.axhline(0,color="grey",ls=":")
    fig.suptitle("Distribution of key endpoints across 65 cells")
    save(fig,"figure_02_distributions")

def fig_transition(agg):
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(len(agg))
    ax.plot(x,agg.H_M0,"o-",color=CB[0],label="H @ M0")
    ax.plot(x,agg.H_MR,"s--",color=CB[1],label="H @ MR")
    ax.axhline(ALPHA,color="grey",ls=":",label=f"alpha={ALPHA}")
    ax.set_xticks(x); ax.set_xticklabels(agg.index,rotation=45,ha="right")
    ax.set_ylabel("held-out predicted-member rate")
    ax.set_title("Held-out reference H: member rate M0 vs MR (membership unchanged)")
    ax.legend(frameon=False)
    save(fig,"figure_03_H_transition")

def fig_heatmap(df):
    cols=list(ENDPOINTS.keys())
    piv=df.groupby(ID_COLUMN)[cols].mean()
    z=(piv-piv.mean())/piv.std(ddof=1)
    fig,ax=plt.subplots(figsize=(10,7))
    im=ax.imshow(z.values,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index,fontsize=9)
    fig.colorbar(im,label="z-score (across datasets)")
    ax.set_title("Standardised endpoints (per-dataset means)")
    save(fig,"figure_04_heatmap")

def fig_corr(df):
    cols=list(ENDPOINTS.keys())
    c=df[cols].corr(method="spearman")
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(c.values,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols,fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j,i,f"{c.values[i,j]:.2f}",ha="center",va="center",
                    fontsize=7,color="white" if abs(c.values[i,j])>0.5 else "black")
    fig.colorbar(im,label="Spearman rho")
    ax.set_title("Endpoint correlations (Spearman, n=65)")
    save(fig,"figure_05_correlation")

def fig_gap_auc(df):
    fig,ax=plt.subplots(figsize=(7,6))
    for i,d in enumerate(sorted(df[ID_COLUMN].unique())):
        sub=df[df[ID_COLUMN]==d]
        ax.scatter(sub.train_test_gap,sub.attack_strength_auc,label=d,
                   color=CB[i%len(CB)],s=40,alpha=0.8)
    r,p=stats.pearsonr(df.train_test_gap,df.attack_strength_auc)
    m,b=np.polyfit(df.train_test_gap,df.attack_strength_auc,1)
    xs=np.linspace(df.train_test_gap.min(),df.train_test_gap.max(),50)
    ax.plot(xs,m*xs+b,"k--",lw=1.5)
    ax.set_xlabel("train/test gap"); ax.set_ylabel("attack strength (AUC)")
    ax.set_title(f"Attack strength vs generalization gap  (r={r:+.2f}, p={p:.3f}, n=65)")
    ax.legend(fontsize=7,ncol=2,frameon=False)
    save(fig,"figure_06_gap_vs_auc")

def fig_ccal_spread(df):
    fig,ax=plt.subplots(figsize=(11,5))
    order=df.groupby(ID_COLUMN).C_cal.mean().sort_values().index
    data=[df[df[ID_COLUMN]==d].C_cal.values for d in order]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.axhline(0,color=CB[2],lw=1); ax.axhspan(-0.05,0.05,color=CB[2],alpha=0.1)
    ax.set_ylabel("C_cal = FPR(H;MR) - alpha")
    ax.set_title("Calibration transfer to the removed client, by dataset (0 = perfect)")
    save(fig,"figure_07_ccal_spread")

def fig_informative(agg):
    fig,ax=plt.subplots(figsize=(8,6))
    colors=[CB[1] if a>=AUC_STRONG else CB[7] for a in agg.AUC]
    ax.scatter(agg.AUC,agg.C_F.abs(),c=colors,s=60)
    for d,r in agg.iterrows():
        ax.annotate(d,(r.AUC,abs(r.C_F)),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.axvline(AUC_STRONG,color="grey",ls=":")
    ax.set_xlabel("attack strength (AUC)"); ax.set_ylabel("|C_F|")
    ax.set_title("Informative datasets (AUC>=0.60, orange) vs weak-attack datasets")
    save(fig,"figure_08_informative")


# ─────────────────────────── MAIN ───────────────────────────────────────────
def main():
    df=load()
    descriptives(df)
    agg=per_dataset(df)
    correlations(df)
    outliers(df)
    fig_overview(agg); fig_distributions(df); fig_transition(agg)
    fig_heatmap(df); fig_corr(df); fig_gap_auc(df); fig_ccal_spread(df); fig_informative(agg)
    print(f"\n✅ figures -> {FIG}\n✅ tables -> {TAB}\n✅ results -> {RES}")

if __name__=="__main__":
    main()