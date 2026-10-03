"""
make_impact_figures.py — two robustness/impact figures.
  Plot B: homophily vs C_cal (rises) and C_F (flat), all three attacks.
  Plot C: per-dataset C_H for all three attacks.
Run:
  python make_impact_figures.py --ext-dir extension_results_YYYYMMDD_HHMMSS --out figs
"""
import argparse, glob, itertools
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import pearsonr
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="serif")
except ImportError:
    pass

mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":400,"savefig.bbox":"tight",
    "font.size":14,"axes.labelsize":15,"axes.titlesize":15,
    "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
    "font.family":"serif","mathtext.fontset":"cm",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"grid.linewidth":0.6,"axes.axisbelow":True})

# seaborn 'deep' palette — matches make_multiattack_figures.py
STYLE={"Shokri":dict(color="#4C72B0",marker="o"),
       "Entropy":dict(color="#DD8452",marker="s"),
       "RMIA":dict(color="#55A868",marker="^")}
GREY="#4C4C4C"
ATTACKS={"Shokri":"summary_core.csv","Entropy":"summary_entropy.csv","RMIA":"summary_rmia.csv"}

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{name}.pdf");fig.savefig(out/f"{name}.png");plt.close(fig)
    print(f"  saved {name}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ext-dir",required=True)
    ap.add_argument("--out",default="figs")
    a=ap.parse_args()

    rows=[pd.read_csv(f,usecols=["dataset","local_label_agreement"])
          for f in glob.glob(str(Path(a.ext_dir)/"scores"/"nodescores_*.csv"))]
    homo=pd.concat(rows,ignore_index=True).groupby("dataset").local_label_agreement.mean()

    data={}
    for n,f in ATTACKS.items():
        d=pd.read_csv(f)
        if "C_H" not in d: d["C_H"]=d.heldout_member_rate_MR-d.heldout_member_rate_M0
        d["homophily"]=d.dataset.map(homo)
        data[n]=d

    # ---- Plot B: homophily vs |C_cal| (left) and |C_F| (right) ----
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for n,d in data.items():
        st=STYLE[n]
        g=d.groupby("dataset").agg(homophily=("homophily","first"),
                                   C_cal=("C_cal","mean"), C_F=("C_F","mean"))
        axes[0].scatter(g.homophily, g.C_cal.abs(), color=st["color"], marker=st["marker"],
                        s=60, alpha=0.85, edgecolor="white", linewidth=0.5, label=n)
        axes[1].scatter(g.homophily, g.C_F.abs(), color=st["color"], marker=st["marker"],
                        s=60, alpha=0.85, edgecolor="white", linewidth=0.5, label=n)
    axes[0].set_xlabel("dataset homophily"); axes[0].set_ylabel("$|C_{\\mathrm{cal}}|$")
    axes[1].set_xlabel("dataset homophily"); axes[1].set_ylabel("$|C_F|$")
    ymax=max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes: ax.set_ylim(0,ymax); ax.legend(frameon=False)
    _save(fig,a.out,"fig_impact_homophily")

    # ---- Plot C: per-dataset C_H for the three attacks ----
    order=data["Shokri"].groupby("dataset").C_H.mean().sort_values().index
    fig,ax=plt.subplots(figsize=(9.2,4.6))
    x=np.arange(len(order)); w=0.26
    for i,(n,d) in enumerate(data.items()):
        st=STYLE[n]
        g=d.groupby("dataset").C_H.mean().reindex(order)
        ax.bar(x+(i-1)*w, g.values, width=w, color=st["color"], alpha=0.88, label=n)
    ax.axhline(0,color=GREY,lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(order,rotation=45,ha="right",fontsize=12)
    ax.set_ylabel("$C_H=\\mathrm{FPR}(H;\\theta_R)-\\mathrm{FPR}(H;\\theta_0)$")
    ax.legend(frameon=False)
    _save(fig,a.out,"fig_impact_CH_consistency")

    # ---- text-verifiable numbers ----
    print("\n--- Plot B correlations (real homophily) ---")
    for n,d in data.items():
        g=d.groupby("dataset").agg(h=("homophily","first"),cc=("C_cal","mean"),cf=("C_F","mean"))
        rc,_=pearsonr(g.h,g.cc.abs()); rf,_=pearsonr(g.h,g.cf.abs())
        print(f"  {n:8s} corr(homo,|C_cal|)={rc:+.2f}  corr(homo,|C_F|)={rf:+.2f}")
    print("\n--- Plot C cross-attack C_H correlation ---")
    cH={n:data[n].groupby('dataset').C_H.mean() for n in ATTACKS}
    for aa,bb in itertools.combinations(ATTACKS,2):
        r,_=pearsonr(cH[aa],cH[bb]); print(f"  corr(C_H {aa},{bb})={r:+.2f}")

if __name__=="__main__":
    main()