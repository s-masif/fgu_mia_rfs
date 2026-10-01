"""
Consolidated publication-quality figure pipeline for the four research questions,
from the Shokri 13x5 results (summary_core.csv). Produces the chosen mix:

  RQ1 attack strength       -> ranked horizontal bar + error bars   fig_rq1_auc
  RQ2 calibration transfer  -> caterpillar/dot plot + alpha band    fig_rq2_ccal
  RQ3 forget-vs-reference    -> histogram/KDE of all C_F values      fig_rq3_cf
  RQ4 F & H transition       -> slope plot F and H, M0 -> MR         fig_rq4_FH
  overview                  -> scatter C_cal vs C_F (colour = AUC)   fig_overview
  summary table             -> per-dataset mean +/- std (csv + tex)

Vector (PDF) + raster (PNG, 400 dpi). Colourblind-safe. Serif math.
Run: python make_all_figures.py --summary summary_core.csv --out figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":400,"savefig.bbox":"tight","font.size":10,
    "font.family":"serif","mathtext.fontset":"cm","axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "grid.linewidth":0.5,"axes.axisbelow":True})
BLUE,ORANGE,GREEN,RED,PURPLE,GREY="#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#555555"
ALPHA=0.10

def load(summary):
    df=pd.read_csv(summary)
    df["C_H"]=df["heldout_member_rate_MR"]-df["heldout_member_rate_M0"]
    return df

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{name}.pdf");fig.savefig(out/f"{name}.png");plt.close(fig)
    print(f"  saved {name}")

# RQ1: ranked horizontal bar + error bars
def fig_rq1(df,out):
    g=df.groupby("dataset").attack_strength_auc.agg(["mean","std"]).sort_values("mean")
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    y=np.arange(len(g))
    colors=[BLUE if m>=0.60 else GREY for m in g["mean"]]
    ax.barh(y,g["mean"],xerr=g["std"],color=colors,alpha=0.85,
            error_kw=dict(ecolor="#333",lw=0.8,capsize=2))
    ax.axvline(0.5,color=RED,ls="--",lw=1,label="chance ($0.5$)")
    ax.set_yticks(y);ax.set_yticklabels(g.index,fontsize=8)
    ax.set_xlabel("Attack strength (AUC)");ax.set_xlim(0.4,0.75)
    # ax.set_title("RQ1: Attack strength, ranked by dataset",fontsize=10)
    ax.legend(frameon=False,fontsize=8,loc="lower right")
    _save(fig,out,"fig_rq1_auc")

# RQ2: caterpillar / dot plot
def fig_rq2(df,out):
    g=df.groupby("dataset").C_cal.agg(["mean","min","max"]).sort_values("mean")
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    y=np.arange(len(g))
    ax.axvspan(-0.05,0.05,color=GREEN,alpha=0.08)
    ax.axvline(0,color=GREEN,lw=1.2,label="perfect transfer")
    ax.hlines(y,g["min"],g["max"],color=ORANGE,lw=1.0,alpha=0.6)
    ax.scatter(g["mean"],y,color=ORANGE,s=32,zorder=3,edgecolor="#7a4b00",linewidth=0.5)
    ax.set_yticks(y);ax.set_yticklabels(g.index,fontsize=8)
    ax.set_xlabel("$C_{\\mathrm{cal}} = \\mathrm{FPR}(H;\\theta_R)-\\alpha$")
    # ax.set_title("RQ2: Calibration transfer varies across datasets",fontsize=10)
    ax.legend(frameon=False,fontsize=8,loc="lower right")
    _save(fig,out,"fig_rq2_ccal")

# RQ3: histogram / KDE of all C_F
def fig_rq3(df,out):
    from scipy.stats import gaussian_kde
    fig,ax=plt.subplots(figsize=(6.2,3.4))
    vals=df.C_F.values
    ax.hist(vals,bins=20,color=GREEN,alpha=0.55,edgecolor="white",linewidth=0.5,density=True)
    xs=np.linspace(vals.min()-0.01,vals.max()+0.01,200)
    ax.plot(xs,gaussian_kde(vals)(xs),color="#00614a",lw=1.5)
    ax.axvline(0,color=GREY,ls="--",lw=1.2,label="forget $=$ reference")
    ax.axvline(vals.mean(),color=RED,ls=":",lw=1.2,label=f"mean $={vals.mean():+.3f}$")
    ax.set_xlabel("$C_F = \\mathrm{FPR}(F;\\theta_R)-\\mathrm{FPR}(H;\\theta_R)$")
    ax.set_ylabel("density")
    # ax.set_title("RQ3: $C_F$ concentrates near zero across all 65 cells",fontsize=10)
    ax.legend(frameon=False,fontsize=8)
    _save(fig,out,"fig_rq3_cf")

# RQ4: F and H transition slope
def fig_rq4(df,out):
    g=df.groupby("dataset")[["forget_member_rate_M0","forget_member_rate_MR",
                             "heldout_member_rate_M0","heldout_member_rate_MR"]].mean()
    g=g.sort_values("forget_member_rate_M0",ascending=False)
    fig,ax=plt.subplots(figsize=(6.2,4.4))
    for d,r in g.iterrows():
        ax.plot([0,1],[r.forget_member_rate_M0,r.forget_member_rate_MR],
                "-o",color=RED,alpha=0.55,ms=3,lw=1.0)
        ax.plot([0,1],[r.heldout_member_rate_M0,r.heldout_member_rate_MR],
                "-s",color=BLUE,alpha=0.45,ms=3,lw=1.0)
    ax.axhline(ALPHA,color=GREY,ls="--",lw=1,label="$\\alpha=0.10$")
    ax.plot([],[],"-o",color=RED,label="forget set $F$")
    ax.plot([],[],"-s",color=BLUE,label="matched reference $H$")
    ax.set_xticks([0,1]);ax.set_xticklabels(["$\\theta_0$ (original)","$\\theta_R$ (retrained)"])
    ax.set_xlim(-0.15,1.15);ax.set_ylabel("predicted-member rate")
    # ax.set_title("RQ4: Membership rate on $F$ and $H$, original vs.\\ retrained",fontsize=10)
    ax.legend(frameon=False,fontsize=8,loc="upper right")
    _save(fig,out,"fig_rq4_FH")

# overview scatter
def fig_overview(df,out):
    fig,ax=plt.subplots(figsize=(5.6,4.6))
    sc=ax.scatter(df.C_cal,df.C_F,c=df.attack_strength_auc,cmap="viridis",
                  s=36,edgecolor="white",linewidth=0.4)
    ax.axhline(0,color=GREY,lw=0.8);ax.axvline(0,color=GREY,lw=0.8)
    ax.axhspan(-0.02,0.02,color=GREEN,alpha=0.08)
    cb=fig.colorbar(sc,ax=ax,shrink=0.85);cb.set_label("attack strength (AUC)",fontsize=8)
    ax.set_xlabel("$C_{\\mathrm{cal}}$ (absolute rate deviates from $\\alpha$)")
    ax.set_ylabel("$C_F$ (forget vs.\\ matched reference)")
    # ax.set_title("Absolute calibration spreads horizontally;\n$C_F$ stays in a narrow band",fontsize=10)
    _save(fig,out,"fig_overview")

def summary_table(df,out):
    rows=[]
    for d in df.groupby("dataset").attack_strength_auc.mean().sort_values(ascending=False).index:
        s=df[df.dataset==d]
        rows.append(dict(dataset=d,
            AUC=f"{s.attack_strength_auc.mean():.3f} $\\pm$ {s.attack_strength_auc.std():.3f}",
            C_cal=f"{s.C_cal.mean():+.3f} $\\pm$ {s.C_cal.std():.3f}",
            C_F=f"{s.C_F.mean():+.3f} $\\pm$ {s.C_F.std():.3f}",
            C_H=f"{s.C_H.mean():+.3f} $\\pm$ {s.C_H.std():.3f}",
            fallback=f"{s.n_fallback_classes.mean():.1f}"))
    t=pd.DataFrame(rows)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    t.to_csv(out/"rq_summary_table.csv",index=False)
    with open(out/"rq_summary_table.tex","w") as f:
        f.write(t.to_latex(index=False,escape=False,column_format="lccccc",
            caption="Per-dataset attack strength and reference-relative quantities "
                    "(mean $\\pm$ std over 5 seeds). The last column reports the mean "
                    "number of classes using the pooled-threshold fallback.",
            label="tab:rq_summary"))
    print("  saved rq_summary_table.csv / .tex")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",default="summary_core.csv")
    ap.add_argument("--out",default="figures")
    a=ap.parse_args()
    df=load(a.summary);print(f"loaded {len(df)} cells")
    fig_rq1(df,a.out);fig_rq2(df,a.out);fig_rq3(df,a.out);fig_rq4(df,a.out);fig_overview(df,a.out)
    summary_table(df,a.out)
    print(f"done -> {a.out}/")

if __name__=="__main__":
    main()