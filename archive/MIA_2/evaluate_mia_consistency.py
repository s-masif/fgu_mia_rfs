"""
evaluate_mia_consistency.py
===========================
Aggregate raw results from mia_experiment.py and produce the artefacts needed
to make the "MIA is not all you need" argument:

  • mia_summary.csv             one row per (dataset, scenario) with mean / std /
                                min / max / range across seeds for every metric
  • mia_per_class.csv           per-(dataset, scenario, attack, data-class)
                                breakdown — exposes systematic failure modes
  • mia_inconsistency.png/.pdf  headline figure: variance of
                                forget_predicted_member_rate across seeds
  • mia_table.tex               LaTeX table for the paper
  • prints headline summary to stdout

Usage
-----
  python evaluate_mia_consistency.py \
         --csv     mia_consistency_results/raw_data/mia_consistency_raw.csv \
         --out-dir mia_consistency_results/evaluation
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  Column groups
# ─────────────────────────────────────────────────────────────────────────────

UTILITY_COLS = [
    "orig_acc", "ret_acc", "retain_acc", "forget_acc", "test_acc",
    "forgetting_score",
]

def _attack_cols(prefix: str) -> list[str]:
    return [
        f"{prefix}_mia_acc",  f"{prefix}_mia_precision", f"{prefix}_mia_recall",
        f"{prefix}_mia_f1",   f"{prefix}_mia_auc",
        f"{prefix}_mia_balanced_acc",
        f"{prefix}_member_precision",    f"{prefix}_member_recall",    f"{prefix}_member_f1",
        f"{prefix}_nonmember_precision", f"{prefix}_nonmember_recall", f"{prefix}_nonmember_f1",
        f"{prefix}_forget_predicted_member_rate",
        f"{prefix}_retain_predicted_member_rate",
        f"{prefix}_test_predicted_member_rate",
        f"{prefix}_forget_mean_score",
        f"{prefix}_retain_mean_score",
        f"{prefix}_test_mean_score",
    ]

ALL_METRICS = UTILITY_COLS + _attack_cols("shokri") + _attack_cols("lr")


# Total node counts per dataset (n_nodes is not written to the raw CSV).
DATASET_NODES = {
    "Cora": 2708, "PubMed": 19717, "CS": 18333, "Photo": 7487,
    "Computers": 13381, "WikiCS": 11701,
    "Chameleon": 890,      # filtered
    "Squirrel": 2223,      # filtered
    "Actor": 7600,
    "tolokers": 11758, "minesweeper": 10000,
    "amazon_ratings": 24492, "roman_empire": 22662,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Load + aggregate
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_cells = len(df.groupby(["dataset", "scenario", "seed"]))
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Coverage: {df['dataset'].nunique()} datasets × "
          f"{df['scenario'].nunique()} scenarios × "
          f"{df['seed'].nunique()} seeds = {n_cells} cells")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, scenario): mean / std / min / max / range across seeds."""
    rows = []
    for (ds, sc), g in df.groupby(["dataset", "scenario"]):
        row = {"dataset": ds, "scenario": sc, "n_seeds": len(g)}
        for col in ALL_METRICS:
            if col not in g.columns:
                continue
            vals = g[col].dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"]  = float(vals.mean())
            row[f"{col}_std"]   = float(vals.std()) if len(vals) > 1 else 0.0
            row[f"{col}_min"]   = float(vals.min())
            row[f"{col}_max"]   = float(vals.max())
            row[f"{col}_range"] = float(vals.max() - vals.min())
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Headline printout
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [headers] + rows)
              for i in range(len(headers))]
    fmt    = "  " + " | ".join(f"{{:<{w}}}" for w in widths)
    sep    = "  " + "-+-".join("-" * w for w in widths)
    out    = [fmt.format(*headers), sep]
    for r in rows:
        out.append(fmt.format(*[str(x) for x in r]))
    return "\n".join(out)


def print_overview(summary: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Compact overview: dataset, scenario, nodes, classes,
    MIA Acc / Balanced Acc / AUC (Shokri)."""
    print("\n📌 Overview — Shokri MIA (Acc / Balanced Acc / AUC):\n")

    classes_map = (df_raw.groupby("dataset")["n_classes"]
                          .first().to_dict())

    rows = []
    for _, r in summary.iterrows():
        ds = r["dataset"]
        rows.append([
            ds, r["scenario"],
            str(DATASET_NODES.get(ds, "?")),
            str(int(classes_map.get(ds, 0))),
            f"{r.get('shokri_mia_acc_mean', np.nan):.3f} ± "
            f"{r.get('shokri_mia_acc_std', np.nan):.3f}",
            f"{r.get('shokri_mia_balanced_acc_mean', np.nan):.3f} ± "
            f"{r.get('shokri_mia_balanced_acc_std', np.nan):.3f}",
            f"{r.get('shokri_mia_auc_mean', np.nan):.3f} ± "
            f"{r.get('shokri_mia_auc_std', np.nan):.3f}",
        ])
    print(_fmt_table(
        ["Dataset", "Scenario", "Nodes", "Classes",
         "MIA Acc μ±σ", "MIA BalAcc μ±σ", "MIA AUC μ±σ"], rows))


def print_headline(summary: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    print()
    print("═" * 80)
    print(" HEADLINE: MIA inconsistency on the gold-standard retrained model")
    print("═" * 80)

    # ── Forget→Member rate (THE headline) ────────────────────────────────
    print("\n📌 Forget-set predicted-member rate (should be ≈ 0 if MIA reliable):\n")
    rows = []
    for _, r in summary.iterrows():
        s_m   = r.get("shokri_forget_predicted_member_rate_mean",  np.nan)
        s_s   = r.get("shokri_forget_predicted_member_rate_std",   np.nan)
        s_rng = r.get("shokri_forget_predicted_member_rate_range", np.nan)
        l_m   = r.get("lr_forget_predicted_member_rate_mean",      np.nan)
        l_s   = r.get("lr_forget_predicted_member_rate_std",       np.nan)
        l_rng = r.get("lr_forget_predicted_member_rate_range",     np.nan)
        rows.append([
            r["dataset"], r["scenario"],
            f"{s_m:.3f} ± {s_s:.3f}", f"{s_rng:.3f}",
            f"{l_m:.3f} ± {l_s:.3f}", f"{l_rng:.3f}",
        ])
    print(_fmt_table(
        ["Dataset", "Scenario", "Shokri μ±σ", "Range", "LR μ±σ", "Range"], rows))

    # ── MIA accuracy across seeds ────────────────────────────────────────
    print("\n📌 Shokri MIA accuracy across seeds (variance ⇒ unreliability):\n")
    rows = []
    for _, r in summary.iterrows():
        rows.append([
            r["dataset"], r["scenario"],
            f"{r.get('shokri_mia_acc_mean', np.nan):.3f} ± "
            f"{r.get('shokri_mia_acc_std', np.nan):.3f}",
            f"{r.get('shokri_mia_acc_range', np.nan):.3f}",
            f"{r.get('shokri_mia_f1_mean', np.nan):.3f} ± "
            f"{r.get('shokri_mia_f1_std', np.nan):.3f}",
            f"{r.get('shokri_mia_auc_mean', np.nan):.3f} ± "
            f"{r.get('shokri_mia_auc_std', np.nan):.3f}",
        ])
    print(_fmt_table(
        ["Dataset", "Scenario", "MIA Acc μ±σ", "Acc Range",
         "MIA F1 μ±σ", "MIA AUC μ±σ"], rows))

    # ── Utility (sanity check on the retrained model) ────────────────────
    print("\n📌 Gold-standard utility & forgetting (sanity check):\n")
    rows = []
    for _, r in summary.iterrows():
        rows.append([
            r["dataset"], r["scenario"],
            f"{r.get('test_acc_mean', np.nan):.3f} ± "
            f"{r.get('test_acc_std', np.nan):.3f}",
            f"{r.get('retain_acc_mean', np.nan):.3f} ± "
            f"{r.get('retain_acc_std', np.nan):.3f}",
            f"{r.get('forget_acc_mean', np.nan):.3f} ± "
            f"{r.get('forget_acc_std', np.nan):.3f}",
            f"{r.get('forgetting_score_mean', np.nan):.4f} ± "
            f"{r.get('forgetting_score_std', np.nan):.4f}",
        ])
    print(_fmt_table(
        ["Dataset", "Scenario", "Test μ±σ", "Retain μ±σ",
         "Forget μ±σ", "FS (JSD) μ±σ"], rows))

    # ── Verdicts ─────────────────────────────────────────────────────────
    print("\n📌 Verdict per attack:\n")
    for attack in ["shokri", "lr"]:
        # Degenerate if recall == 0 (i.e. predicts non-member for every sample)
        rec = df_raw[f"{attack}_mia_recall"]
        if (rec < 1e-6).all():
            print(f"  • {attack.upper():6s}: ⚠ DEGENERATE — predicts non-member "
                  f"for every sample across all {len(df_raw)} rows")
        else:
            print(f"  • {attack.upper():6s}: ✓ active attack — recall "
                  f"mean={rec.mean():.3f}, max={rec.max():.3f}")

    # ── Worst-case inconsistency (the headline number for the abstract) ──
    print("\n📌 Worst-case Shokri inconsistency (for the abstract):\n")
    s = summary.copy()
    s["forget_range"] = s["shokri_forget_predicted_member_rate_range"]
    worst = s.sort_values("forget_range", ascending=False).iloc[0]
    print(f"  ⇒ {worst['dataset']} / {worst['scenario']}: "
          f"forget→member rate varies from "
          f"{worst['shokri_forget_predicted_member_rate_min']:.3f} to "
          f"{worst['shokri_forget_predicted_member_rate_max']:.3f} "
          f"across {worst['n_seeds']} seeds "
          f"(range = {worst['forget_range']:.3f}).")


# ─────────────────────────────────────────────────────────────────────────────
#  Headline figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_inconsistency(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plot: forget→member rate across seeds per (dataset, scenario)."""
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, attack, title in zip(
        axes, ["shokri", "lr"],
        ["Shokri shadow MIA (per-class)", "LR confidence MIA (degenerate)"],
    ):
        col   = f"{attack}_forget_predicted_member_rate"
        data, labels, positions = [], [], []
        pos = 0
        for ds in sorted(df["dataset"].unique()):
            for sc in sorted(df["scenario"].unique()):
                vals = df[(df["dataset"] == ds) & (df["scenario"] == sc)][col].dropna()
                if len(vals) == 0:
                    continue
                data.append(vals.values)
                labels.append(f"{ds}\n{sc}")
                positions.append(pos)
                pos += 1
            pos += 0.5   # gap between datasets

        bp = ax.boxplot(data, positions=positions, widths=0.55,
                        patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "#D55E00",
                                   "markeredgecolor": "#D55E00", "markersize": 4})
        for patch in bp["boxes"]:
            patch.set_facecolor("#0173B2"); patch.set_alpha(0.35)

        # Overlay the actual seed values
        for d, p in zip(data, positions):
            ax.scatter([p] * len(d), d, s=10, color="#0173B2",
                       alpha=0.6, zorder=3)

        ax.axhline(0, color="gray", lw=0.5, linestyle="--",
                   label="Truth (forget = non-member)")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Forget → predicted member rate")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, max(1.0, max((d.max() for d in data), default=0.5) + 0.05))
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.4)

    fig.suptitle("MIA predictions on the gold-standard retrained model "
                 "— wide spread across seeds = inconsistency",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    for ext, dpi in [(".png", 150), (".pdf", 300)]:
        fig.savefig(out_dir / f"mia_inconsistency{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"   → {out_dir / 'mia_inconsistency.png'}")
    print(f"   → {out_dir / 'mia_inconsistency.pdf'}")


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def emit_latex(summary: pd.DataFrame, out_path: Path) -> None:
    def _ms(r, col):
        return (f"{r.get(f'{col}_mean', np.nan):.3f}\\,$\\pm$\\,"
                f"{r.get(f'{col}_std', np.nan):.3f}")
    def _rng(r, col):
        return f"{r.get(f'{col}_range', np.nan):.3f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{MIA inconsistency on the gold-standard retrain-from-scratch "
        r"model. ``Forget$\to$Mem'' is the rate at which forget-set nodes are "
        r"predicted as members; by construction this should be 0 since forget "
        r"nodes were excluded from $M_{\text{ret}}$'s training set. The variance "
        r"across 5 seeds ($\pm$ column and Range) is direct evidence that MIA "
        r"is not a reliable verification metric for unlearning.}",
        r"\label{tab:mia-consistency}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Shokri shadow MIA} & "
        r"\multicolumn{4}{c}{LR confidence MIA} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"Dataset & Scenario & MIA Acc & MIA F1 & "
        r"Forget$\to$Mem & Range & MIA Acc & MIA F1 & "
        r"Forget$\to$Mem & Range \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['dataset']} & {r['scenario']} & "
            f"{_ms(r, 'shokri_mia_acc')} & {_ms(r, 'shokri_mia_f1')} & "
            f"{_ms(r, 'shokri_forget_predicted_member_rate')} & "
            f"{_rng(r, 'shokri_forget_predicted_member_rate')} & "
            f"{_ms(r, 'lr_mia_acc')} & {_ms(r, 'lr_mia_f1')} & "
            f"{_ms(r, 'lr_forget_predicted_member_rate')} & "
            f"{_rng(r, 'lr_forget_predicted_member_rate')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out_path.write_text("\n".join(lines))
    print(f"   → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-data-class analysis
# ─────────────────────────────────────────────────────────────────────────────

def per_class_analysis(df: pd.DataFrame, out_path: Path) -> pd.DataFrame | None:
    """Parse the per_data_class dict columns; aggregate per (dataset, scenario,
    attack, class) across seeds. Exposes systematic per-class failures."""
    rows = []
    for _, r in df.iterrows():
        for attack in ["shokri", "lr"]:
            col = f"{attack}_per_data_class"
            if col not in r or pd.isna(r[col]):
                continue
            try:
                pc = ast.literal_eval(r[col])
            except (ValueError, SyntaxError):
                continue
            for c, m in pc.items():
                rows.append({
                    "dataset":     r["dataset"],
                    "scenario":    r["scenario"],
                    "seed":        r["seed"],
                    "attack":      attack,
                    "class":       int(c),
                    "n":           m.get("n", 0),
                    "n_member":    m.get("n_member", 0),
                    "n_nonmember": m.get("n_nonmember", 0),
                    "acc":         m.get("acc", np.nan),
                    "member_acc":  m.get("member_acc", np.nan),
                    "member_f1":   m.get("member_f1", np.nan),
                })
    if not rows:
        print("   ⚠ no per-class data found")
        return None

    pc_df = pd.DataFrame(rows)
    agg = (pc_df.groupby(["dataset", "scenario", "attack", "class"])
                .agg(n_seeds         = ("seed",       "count"),
                     n_mean          = ("n",          "mean"),
                     n_member_mean   = ("n_member",   "mean"),
                     acc_mean        = ("acc",        "mean"),
                     acc_std         = ("acc",        "std"),
                     member_acc_mean = ("member_acc", "mean"),
                     member_acc_std  = ("member_acc", "std"),
                     member_acc_min  = ("member_acc", "min"),
                     member_acc_max  = ("member_acc", "max"),
                     member_f1_mean  = ("member_f1",  "mean"),
                     member_f1_std   = ("member_f1",  "std"))
                .reset_index())
    agg.to_csv(out_path, index=False)
    print(f"   → {out_path}")

    # Print classes where MIA completely fails (member_acc_mean = 0 across all seeds)
    shokri = agg[(agg["attack"] == "shokri") & (agg["n_member_mean"] >= 5)]
    failed = shokri[shokri["member_acc_max"] < 0.05]
    if len(failed):
        print(f"\n📌 Classes where Shokri MIA fails entirely "
              f"(member_acc=0 across all seeds, n_member≥5):  "
              f"{len(failed)}/{len(shokri)} class-rows")
        cols = ["dataset", "scenario", "class", "n_member_mean",
                "member_acc_max"]
        print(failed[cols].head(20).to_string(index=False))
    return agg


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="mia_consistency_results/raw_data/mia_consistency_raw.csv")
    parser.add_argument("--out-dir", default="mia_consistency_results/evaluation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_results(Path(args.csv))

    summary = aggregate(df_raw)
    summary.to_csv(out_dir / "mia_summary.csv", index=False)
    print(f"   → {out_dir / 'mia_summary.csv'}")

    print_headline(summary, df_raw)
    print_overview(summary, df_raw)
    plot_inconsistency(df_raw, out_dir)
    emit_latex(summary, out_dir / "mia_table.tex")
    per_class_analysis(df_raw, out_dir / "mia_per_class.csv")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()