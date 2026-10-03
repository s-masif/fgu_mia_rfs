"""
make_combined_figures.py — combined two-panel figures (built in matplotlib).
Insert each as a single full-width figure* in LaTeX.

Produces:
  fig_rq2rq3.pdf       : (a) C_cal per dataset   (b) C_F distribution   [shared top-center legend]
  fig_rq4_overview.pdf : (a) F/H transition (3 attack panels, shared y-axis)  (b) C_cal vs C_F scatter
  fig_impact.pdf       : (a) F and H rates, theta_0 vs theta_R   (b) per-dataset C_H

Run:
  python make_combined_figures.py \
    --attacks shokri=summary_core.csv entropy=summary_entropy.csv rmia=summary_rmia.csv \
    --out figs
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="serif")
except ImportError:
    pass

mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":400,"savefig.bbox":"tight",
    "font.size":18,
    "axes.labelsize":22, "axes.labelweight":"bold",
    "xtick.labelsize":15, "ytick.labelsize":15,
    "legend.fontsize":16,
    "font.family":"serif","mathtext.fontset":"cm",
    "axes.linewidth":2.4, "axes.edgecolor":"#111111",
    "xtick.major.width":2.0, "ytick.major.width":2.0,
    "xtick.major.size":7, "ytick.major.size":7,
    "xtick.color":"#111111", "ytick.color":"#111111",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"grid.linewidth":0.8,"axes.axisbelow":True,
    "lines.linewidth":2.2,"lines.markersize":9,
})
ALPHA=0.10
AUC_STRONG=0.60   # "informative" cells for the pre/post F–H panel

# sequential Blues (ColorBrewer), dark -> medium -> light; distinct markers as backup channel
ATTACK_STYLE={
    "shokri":  dict(color="#08519c", marker="o", label="Shadow"),   # dark blue
    "entropy": dict(color="#4292c6", marker="s", label="Entropy"),  # medium blue
    "rmia":    dict(color="#6baed6", marker="^", label="RMIA"),     # light-medium blue (visible on white)
}
GREY,RED,GREEN="#333333","#C44E52","#2ca25f"
BLUE_DARK, BLUE_MED, BLUE_LIGHT = "#08519c", "#4292c6", "#9ecae1"
F_COLOR, H_COLOR = "#08519c", "#9ecae1"   # F = dark blue, H = light blue
EDGE = "#083b6f"
FORGET_C, REF_C = "#08519c", "#6baed6"    # for the fig_rq4 transition lines

def _bold_ticks(ax):
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontweight("bold")

def _bold_legend(leg):
    for t in leg.get_texts(): t.set_fontweight("bold")

def _panel_label_below(ax, s, y=-0.42):
    ax.text(0.5, y, s, transform=ax.transAxes, fontsize=22, fontweight="bold",
            va="top", ha="center")

def load_attacks(specs):
    out={}
    for spec in specs:
        name,path=spec.split("=",1)
        d=pd.read_csv(path); d["C_H"]=d.heldout_member_rate_MR-d.heldout_member_rate_M0
        out[name]=d
    return out

def order_by_auc(attacks):
    first=next(iter(attacks.values()))
    return first.groupby("dataset").attack_strength_auc.mean().sort_values(ascending=False).index.tolist()

def _grouped(ax, attacks, col, order, dodge=0.18):
    n=len(attacks); offs=np.linspace(-dodge*(n-1)/2,dodge*(n-1)/2,n)
    for (name,df),off in zip(attacks.items(),offs):
        st=ATTACK_STYLE[name]
        g=df.groupby("dataset")[col].agg(["mean","std"]).reindex(order)
        ax.errorbar(np.arange(len(order))+off,g["mean"],yerr=g["std"],fmt=st["marker"],
                    color=st["color"],ms=9,lw=0,elinewidth=1.8,capsize=4,capthick=1.8,
                    markeredgecolor=EDGE, markeredgewidth=0.8,
                    label=st["label"],alpha=0.95)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=45,ha="right")

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{name}.pdf");fig.savefig(out/f"{name}.png");plt.close(fig)
    print(f"  saved {name}")

# ===== FIGURE 1: RQ2 (C_cal) + RQ3 (C_F dist) — ONE shared legend, top-center =====
def fig_rq2rq3(attacks,order,out):
    fig,(axL,axR)=plt.subplots(1,2,figsize=(16,6.4))
    plt.subplots_adjust(wspace=0.28, bottom=0.30, top=0.82)

    # (a) C_cal per dataset
    axL.axhspan(-0.05,0.05,color=GREEN,alpha=0.12)
    axL.axhline(0,color=GREEN,lw=2.0)
    _grouped(axL,attacks,"C_cal",order)
    axL.set_ylabel("$\\mathbf{C_{\\mathrm{cal}}=\\mathrm{FPR}(H;\\theta_R)-\\alpha}$")
    _bold_ticks(axL); _panel_label_below(axL,"(a)")

    # (b) C_F distribution
    for name,df in attacks.items():
        st=ATTACK_STYLE[name]
        axR.hist(df.C_F.values,bins=18,density=True,alpha=0.55,color=st["color"],
                 edgecolor="white",linewidth=0.8)
    axR.axvline(0,color=GREY,ls="--",lw=2.0)
    axR.set_xlabel("$\\mathbf{C_F=\\mathrm{FPR}(F;\\theta_R)-\\mathrm{FPR}(H;\\theta_R)}$")
    axR.set_ylabel("density")
    _bold_ticks(axR); _panel_label_below(axR,"(b)")

    # ---- ONE shared figure legend, top-center ----
    handles=[mlines.Line2D([],[],color=ATTACK_STYLE[n]["color"],marker=ATTACK_STYLE[n]["marker"],
                           linestyle="none",markersize=11,markeredgecolor=EDGE,
                           label=ATTACK_STYLE[n]["label"]) for n in attacks]
    handles+=[mlines.Line2D([],[],color=GREEN,lw=2.0,label="perfect transfer"),
              mlines.Line2D([],[],color=GREY,lw=2.0,ls="--",label="forget $=$ reference")]
    leg=fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5,1.0), fontsize=16)
    _bold_legend(leg)
    _save(fig,out,"fig_rq2rq3")

# ===== FIGURE 2: RQ4 (F/H transition, shared y-axis) + overview scatter =====
def fig_rq4_overview(attacks,order,out):
    fig=plt.figure(figsize=(17,6.4))
    nA=len(attacks)
    gs=fig.add_gridspec(1, nA+2, width_ratios=[1]*nA+[0.3,1.6], wspace=0.30)
    fig.subplots_adjust(bottom=0.26, top=0.88)

    # --- global y-range across ALL attacks (shared y-axis) ---
    rate_cols=["forget_member_rate_M0","forget_member_rate_MR",
               "heldout_member_rate_M0","heldout_member_rate_MR"]
    allv=np.concatenate([
        df.groupby("dataset")[rate_cols].mean().values.ravel()
        for df in attacks.values()
    ])
    ymin=min(0.0,float(allv.min())); ymax=float(allv.max())
    pad=0.05*(ymax-ymin if ymax>ymin else 1.0)
    ylo, yhi = ymin-pad, ymax+pad

    axes=[fig.add_subplot(gs[0,i]) for i in range(nA)]
    for ax,(name,df) in zip(axes,attacks.items()):
        g=df.groupby("dataset")[rate_cols].mean()
        for _,r in g.iterrows():
            ax.plot([0,1],[r.forget_member_rate_M0,r.forget_member_rate_MR],
                    "-o",color=FORGET_C,alpha=0.6,ms=6,lw=1.6)
            ax.plot([0,1],[r.heldout_member_rate_M0,r.heldout_member_rate_MR],
                    "-s",color=H_COLOR,alpha=0.6,ms=6,lw=1.6)
        ax.axhline(ALPHA,color=GREY,ls="--",lw=1.8)
        ax.set_xticks([0,1])
        ax.set_xticklabels(["$\\boldsymbol{\\theta_0}$","$\\boldsymbol{\\theta_R}$"],fontsize=20)
        ax.set_xlim(-0.35,1.35)
        ax.set_ylim(ylo, yhi)                      # SHARED y-axis for all three
        ax.text(0.5,1.04,ATTACK_STYLE[name]["label"],transform=ax.transAxes,
                ha="center",fontsize=18,fontweight="bold")
        _bold_ticks(ax)
    axes[0].set_ylabel("predicted-member rate")
    for ax in axes[1:]: ax.tick_params(labelleft=False)
    axes[0].plot([],[],"-o",color=FORGET_C,label="forget $F$")
    axes[0].plot([],[],"-s",color=H_COLOR,label="reference $H$")
    legA=axes[0].legend(frameon=False,loc="upper right",fontsize=13); _bold_legend(legA)
    axes[nA//2].text(0.5,-0.28,"(a)",transform=axes[nA//2].transAxes,
                     fontsize=22,fontweight="bold",va="top",ha="center")
    # scatter panel
    axS=fig.add_subplot(gs[0,nA+1])
    for name,df in attacks.items():
        st=ATTACK_STYLE[name]
        axS.scatter(df.C_cal,df.C_F,color=st["color"],marker=st["marker"],
                    s=90,alpha=0.85,edgecolor=EDGE,linewidth=0.6,label=st["label"])
    axS.axhline(0,color=GREY,lw=1.4);axS.axvline(0,color=GREY,lw=1.4)
    axS.axhspan(-0.02,0.02,color=GREEN,alpha=0.12)
    axS.set_xlabel("$\\mathbf{C_{\\mathrm{cal}}}$"); axS.set_ylabel("$\\mathbf{C_F}$")
    legS=axS.legend(frameon=False); _bold_legend(legS); _bold_ticks(axS)
    axS.text(0.5,-0.28,"(b)",transform=axS.transAxes,fontsize=22,fontweight="bold",
             va="top",ha="center")
    _save(fig,out,"fig_rq4_overview")

# ===== FIGURE 3: (a) F and H rates, theta_0 vs theta_R  (b) per-dataset C_H =====
def fig_impact(attacks,order,out):
    fig,(axL,axR)=plt.subplots(1,2,figsize=(16,6.6))
    plt.subplots_adjust(wspace=0.30, bottom=0.32, top=0.88)

    # ---------- (a) F and H member rates at theta_0 and theta_R, informative cells ----------
    names=list(attacks.keys()); nA=len(names)
    F0,H0,F1,H1,ns=[],[],[],[],[]
    for n in names:
        s=attacks[n][attacks[n].attack_strength_auc>=AUC_STRONG]; ns.append(len(s))
        F0.append(s.forget_member_rate_M0.mean()); H0.append(s.heldout_member_rate_M0.mean())
        F1.append(s.forget_member_rate_MR.mean()); H1.append(s.heldout_member_rate_MR.mean())

    w=0.38
    centers=[]; ticklabels=[]; xcur=0.0
    for i,name in enumerate(names):
        for st,(Fv,Hv) in zip([r"$\theta_0$",r"$\theta_R$"],
                               [(F0[i],H0[i]),(F1[i],H1[i])]):
            axL.bar(xcur-w/2, Fv, width=w, color=F_COLOR, edgecolor=EDGE, linewidth=1.1, zorder=3)
            axL.bar(xcur+w/2, Hv, width=w, color=H_COLOR, edgecolor=EDGE, linewidth=1.1, zorder=3)
            centers.append(xcur); ticklabels.append(st)
            xcur+=1.3
        xcur+=0.7   # gap between attacks
    axL.axhline(ALPHA, color="#111111", ls="--", lw=1.4, zorder=2)
    axL.set_xticks(centers); axL.set_xticklabels(ticklabels, fontsize=15)
    for i,name in enumerate(names):
        c=(centers[2*i]+centers[2*i+1])/2
        axL.text(c, -0.16, ATTACK_STYLE[name]["label"], transform=axL.get_xaxis_transform(),
                 ha="center", va="top", fontsize=15, fontweight="bold")
    axL.set_ylabel("mean predicted-member rate")
    leg_handles=[
        mpatches.Patch(facecolor=F_COLOR, edgecolor=EDGE, label="forget $F$"),
        mpatches.Patch(facecolor=H_COLOR, edgecolor=EDGE, label="reference $H$"),
    ]
    legL=axL.legend(handles=leg_handles, frameon=False, fontsize=13, loc="upper right")
    _bold_legend(legL); _bold_ticks(axL)
    axL.text(0.5,-0.30,"(a)",transform=axL.transAxes,fontsize=22,fontweight="bold",va="top",ha="center")

    # ---------- (b) per-dataset C_H bars (same blue family) ----------
    SEQ3=[BLUE_DARK, BLUE_MED, BLUE_LIGHT]
    cH_order=attacks["shokri"].groupby("dataset").C_H.mean().sort_values().index
    xb=np.arange(len(cH_order)); wb=0.26
    for i,(name,df) in enumerate(attacks.items()):
        g=df.groupby("dataset").C_H.mean().reindex(cH_order)
        axR.bar(xb+(i-1)*wb, g.values, width=wb, color=SEQ3[i], alpha=0.95,
                edgecolor=EDGE, linewidth=1.0, label=ATTACK_STYLE[name]["label"], zorder=3)
    axR.axhline(0,color="#111111",lw=1.4, zorder=2)
    axR.set_xticks(xb); axR.set_xticklabels(cH_order, rotation=45, ha="right")
    axR.set_ylabel("$\\mathbf{C_H=\\mathrm{FPR}(H;\\theta_R)-\\mathrm{FPR}(H;\\theta_0)}$")
    legR=axR.legend(frameon=False); _bold_legend(legR)
    _bold_ticks(axR)
    axR.text(0.5,-0.42,"(b)",transform=axR.transAxes,fontsize=22,fontweight="bold",va="top",ha="center")
    _save(fig,out,"fig_impact")

    # text-verifiable numbers
    print("\n--- F and H member rates on informative cells (AUC>=%.2f) ---" % AUC_STRONG)
    for n,f0,h0,f1,h1,k in zip(names,F0,H0,F1,H1,ns):
        print(f"  {n:8s} n={k:2d}  M0: F={f0:.3f} H={h0:.3f} (gap {f0-h0:+.3f})  "
              f"MR: F={f1:.3f} H={h1:.3f} (gap {f1-h1:+.3f})")
    print("--- C_H cross-attack correlation ---")
    cH={n:attacks[n].groupby('dataset').C_H.mean() for n in attacks}
    for a,b in itertools.combinations(attacks,2):
        r,_=pearsonr(cH[a],cH[b]); print(f"  {a}-{b}: {r:+.2f}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attacks",nargs="+",required=True)
    ap.add_argument("--out",default="figs")
    a=ap.parse_args()
    attacks=load_attacks(a.attacks); order=order_by_auc(attacks)
    print(f"attacks: {list(attacks)} | {len(order)} datasets")
    fig_rq2rq3(attacks,order,a.out)
    fig_rq4_overview(attacks,order,a.out)
    fig_impact(attacks,order,a.out)
    print(f"done -> {a.out}/")

if __name__=="__main__":
    main()