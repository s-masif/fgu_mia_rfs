"""
Multi-attack-ready figure pipeline.
Pairing guidance:
  - fig_rq2_ccal + fig_rq3_cf  -> side-by-side (identical aspect, they align)
  - fig_rq1_auc  + any single-panel -> side-by-side (identical aspect)
  - fig_overview -> single-panel (pair with another single-panel or standalone)
  - fig_rq4_FH   -> STANDALONE full-width (figure*), it has 3 panels

Usage:
  python make_multiattack_figures.py --attacks shokri=summary_core.csv \
      entropy=summary_entropy.csv rmia=summary_rmia.csv --out figs
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="serif")
except ImportError:
    pass

mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":400,"savefig.bbox":"tight",
    "font.size":18,
    "axes.labelsize":21, "axes.labelweight":"bold",
    "xtick.labelsize":15, "ytick.labelsize":15,
    "legend.fontsize":16,
    "font.family":"serif","mathtext.fontset":"cm",
    # very bold axes
    "axes.linewidth":2.4, "axes.edgecolor":"#111111",
    "xtick.major.width":2.0, "ytick.major.width":2.0,
    "xtick.major.size":7, "ytick.major.size":7,
    "xtick.color":"#111111", "ytick.color":"#111111",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"grid.linewidth":0.8,"axes.axisbelow":True,
    "lines.linewidth":2.2,"lines.markersize":9,
})
ALPHA=0.10
PANEL=(7.5,5.5)   # ONE size for every single-panel figure -> all pairs align

ATTACK_STYLE = {
    "shokri":  dict(color="#4C72B0", marker="o", label="Shadow"),
    "entropy": dict(color="#DD8452", marker="s", label="Entropy"),
    "rmia":    dict(color="#55A868", marker="^", label="RMIA"),
}
GREY, RED, GREEN = "#333333", "#C44E52", "#55A868"
FORGET_C, REF_C = "#C44E52", "#4C72B0"

def _bold_ticks(ax):
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontweight("bold")

def load_attacks(specs):
    out={}
    for spec in specs:
        name,path=spec.split("=",1)
        df=pd.read_csv(path)
        df["C_H"]=df["heldout_member_rate_MR"]-df["heldout_member_rate_M0"]
        out[name]=df
    return out

def _dataset_order(attacks):
    first=next(iter(attacks.values()))
    return first.groupby("dataset").attack_strength_auc.mean().sort_values(ascending=False).index.tolist()

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{name}.pdf");fig.savefig(out/f"{name}.png");plt.close(fig)
    print(f"  saved {name}")

def _grouped_points(ax, attacks, col, order, dodge=0.18):
    n=len(attacks)
    offs=np.linspace(-dodge*(n-1)/2, dodge*(n-1)/2, n)
    for (name,df),off in zip(attacks.items(),offs):
        st=ATTACK_STYLE.get(name,dict(color=GREY,marker="o",label=name))
        g=df.groupby("dataset")[col].agg(["mean","std"]).reindex(order)
        x=np.arange(len(order))+off
        ax.errorbar(x,g["mean"],yerr=g["std"],fmt=st["marker"],color=st["color"],
                    ms=9,lw=0,elinewidth=1.8,capsize=4,capthick=1.8,
                    label=st["label"],alpha=0.95)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order,rotation=45,ha="right")

def fig_rq1(attacks,order,out):
    fig,ax=plt.subplots(figsize=PANEL)
    _grouped_points(ax,attacks,"attack_strength_auc",order)
    ax.axhline(0.5,color=RED,ls="--",lw=1.8,label="chance")
    ax.set_ylabel("Attack strength (AUC)")
    ax.legend(frameon=False,ncol=len(attacks)+1,loc="lower center",bbox_to_anchor=(0.5,1.02))
    _bold_ticks(ax)
    _save(fig,out,"fig_rq1_auc")

def fig_rq2(attacks,order,out):
    fig,ax=plt.subplots(figsize=PANEL)
    ax.axhspan(-0.05,0.05,color=GREEN,alpha=0.12)
    ax.axhline(0,color=GREEN,lw=2.0,label="perfect transfer")
    _grouped_points(ax,attacks,"C_cal",order)
    ax.set_ylabel("$C_{\\mathrm{cal}}=\\mathrm{FPR}(H;\\theta_R)-\\alpha$")
    ax.legend(frameon=False,ncol=len(attacks)+1,loc="lower center",bbox_to_anchor=(0.5,1.02))
    _bold_ticks(ax)
    _save(fig,out,"fig_rq2_ccal")

def fig_rq3(attacks,order,out):
    fig,ax=plt.subplots(figsize=PANEL)
    for name,df in attacks.items():
        st=ATTACK_STYLE.get(name,dict(color=GREY,label=name))
        ax.hist(df.C_F.values,bins=18,density=True,alpha=0.55,color=st["color"],
                label=st["label"],edgecolor="white",linewidth=0.8)
    ax.axvline(0,color=GREY,ls="--",lw=2.0,label="forget $=$ reference")
    ax.set_xlabel("$C_F=\\mathrm{FPR}(F;\\theta_R)-\\mathrm{FPR}(H;\\theta_R)$")
    ax.set_ylabel("density")
    ax.legend(frameon=False,loc="upper right")
    _bold_ticks(ax)
    _save(fig,out,"fig_rq3_cf")

def fig_rq4(attacks,order,out):
    # 3-panel (one per attack) -> intended as STANDALONE full-width figure*
    n=len(attacks)
    fig,axes=plt.subplots(1,n,figsize=(5.0*n,5.6),sharey=True,squeeze=False)
    for ax,(name,df) in zip(axes[0],attacks.items()):
        st=ATTACK_STYLE.get(name,dict(color=GREY,label=name))
        g=df.groupby("dataset")[["forget_member_rate_M0","forget_member_rate_MR",
                                 "heldout_member_rate_M0","heldout_member_rate_MR"]].mean()
        for _,r in g.iterrows():
            ax.plot([0,1],[r.forget_member_rate_M0,r.forget_member_rate_MR],
                    "-o",color=FORGET_C,alpha=0.6,ms=6,lw=1.6)
            ax.plot([0,1],[r.heldout_member_rate_M0,r.heldout_member_rate_MR],
                    "-s",color=REF_C,alpha=0.5,ms=6,lw=1.6)
        ax.axhline(ALPHA,color=GREY,ls="--",lw=1.8)
        ax.set_xticks([0,1]);ax.set_xticklabels(["$\\theta_0$","$\\theta_R$"])
        ax.set_xlim(-0.25,1.25)
        ax.text(0.5,1.02,st["label"],transform=ax.transAxes,ha="center",
                fontsize=19,fontweight="bold")
        _bold_ticks(ax)
    axes[0][0].set_ylabel("predicted-member rate")
    axes[0][0].plot([],[],"-o",color=FORGET_C,label="forget $F$")
    axes[0][0].plot([],[],"-s",color=REF_C,label="reference $H$")
    axes[0][0].legend(frameon=False,loc="upper right")
    _save(fig,out,"fig_rq4_FH")

def fig_overview(attacks,order,out):
    fig,ax=plt.subplots(figsize=PANEL)
    for name,df in attacks.items():
        st=ATTACK_STYLE.get(name,dict(color=GREY,marker="o",label=name))
        ax.scatter(df.C_cal,df.C_F,color=st["color"],marker=st["marker"],
                   s=80,alpha=0.8,edgecolor="white",linewidth=0.7,label=st["label"])
    ax.axhline(0,color=GREY,lw=1.4);ax.axvline(0,color=GREY,lw=1.4)
    ax.axhspan(-0.02,0.02,color=GREEN,alpha=0.12)
    ax.set_xlabel("$C_{\\mathrm{cal}}$ (absolute rate deviates from $\\alpha$)")
    ax.set_ylabel("$C_F$ (forget vs.\\ matched reference)")
    ax.legend(frameon=False)
    _bold_ticks(ax)
    _save(fig,out,"fig_overview")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attacks",nargs="+",required=True)
    ap.add_argument("--out",default="figs_multiattack")
    a=ap.parse_args()
    attacks=load_attacks(a.attacks)
    order=_dataset_order(attacks)
    print(f"attacks: {list(attacks)} | {len(order)} datasets")
    fig_rq1(attacks,order,a.out);fig_rq2(attacks,order,a.out)
    fig_rq3(attacks,order,a.out);fig_rq4(attacks,order,a.out)
    fig_overview(attacks,order,a.out)
    print(f"done -> {a.out}/")

if __name__=="__main__":
    main()