"""Regenerates every data figure of the paper from files under runs/ (and the
leaderboard numbers in docs/LEADERBOARD.md). Run from the repository root:
    python paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"classical": "#4C78A8", "deep": "#9AA5B1", "ensemble": "#54A24B", "proposed": "#D62728",
       "hybrid": "#F58518"}


# ---- 1. leakage before/after (docs/evaluation_protocol.md section 4) -------------
def fig_leak():
    names = ["PPG only", "ECG only", "Concat.", "Cross-attn."]
    before = [0.660, 0.736, 0.750, 0.774]
    after = [0.602, 0.642, 0.693, 0.699]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(x - 0.2, before, 0.38, label="Original (leaky)", color="#BBBBBB")
    ax.bar(x + 0.2, after, 0.38, label="Leak-free", color=COL["classical"])
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - 0.2, b + 0.005, f"{b:.3f}", ha="center", fontsize=6)
        ax.text(i + 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x, names)
    ax.set_ylim(0.5, 0.82)
    ax.set_ylabel("F1 (Challenge 2015)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.savefig(OUT / "leakage.pdf")
    plt.close(fig)


# ---- 2. leaderboards (docs/LEADERBOARD.md) --------------------------------------
CINC = [  # name, group, F1, AUC
    ("Ensemble (10 models)", "ensemble", 0.827, 0.948),
    ("XGBoost (feat. v2)", "classical", 0.818, 0.939),
    ("Extra Trees (feat. v2)", "classical", 0.802, 0.947),
    ("MLP (feat. v2)", "classical", 0.802, 0.927),
    ("Logistic reg. (feat. v2)", "classical", 0.801, 0.928),
    ("Random Forest (feat. v2)", "classical", 0.795, 0.938),
    ("SVM (feat. v2)", "classical", 0.791, 0.928),
    ("ResNet1D", "deep", 0.760, 0.884),
    ("TCN", "deep", 0.751, 0.887),
    ("InceptionTime", "deep", 0.732, 0.876),
    ("CNN cross-attention (proposed)", "proposed", 0.699, 0.841),
    ("CNN concatenation", "deep", 0.693, 0.836),
]
VTAC = [
    ("Hybrid deep recipe", "hybrid", 0.858, 0.966),
    ("Ensemble incl. hybrid", "ensemble", 0.844, 0.957),
    ("Ensemble (10 models)", "ensemble", 0.839, 0.955),
    ("MLP (feat. v2)", "classical", 0.821, 0.949),
    ("XGBoost (feat. v2)", "classical", 0.814, 0.942),
    ("SVM (feat. v2)", "classical", 0.806, 0.948),
    ("CNN bidir. cross-attn.", "deep", 0.775, 0.933),
    ("TCN", "deep", 0.760, 0.919),
    ("ResNet1D", "deep", 0.750, 0.915),
    ("CNN cross-attention (proposed)", "proposed", 0.746, 0.915),
    ("CNN concatenation", "deep", 0.744, 0.909),
]


def fig_leaderboard():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, rows, title, lo in [(axes[0], CINC, "(a) Challenge 2015 (10 seeds)", 0.65),
                                (axes[1], VTAC, "(b) VTaC (official split, 5 seeds)", 0.70)]:
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [r[2] for r in rows], color=[COL[r[1]] for r in rows], height=0.7)
        for yi, r in zip(y, rows):
            ax.text(r[2] + 0.002, yi, f"{r[2]:.3f}", va="center", fontsize=6)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=6.5)
        ax.set_xlim(lo, 0.89)
        ax.set_xlabel("F1 (threshold 0.5)")
        ax.set_title(title, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ["ensemble", "hybrid", "classical", "deep", "proposed"]]
    fig.legend(handles, ["Ensemble", "Hybrid deep", "Classical (features)", "Deep", "Proposed model"],
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "leaderboard.pdf")
    plt.close(fig)


# ---- 3. paired comparison forest plot (runs/challenge2015_ppg/significance.csv) ---
def fig_forest():
    s = pd.read_csv(ROOT / "runs/challenge2015_ppg/significance.csv")
    lab = {"concat->cross_attention": "Concat. $\\to$ cross-attn.",
           "ecg_only->cross_attention": "ECG-only $\\to$ cross-attn.",
           "ppg_only->cross_attention": "PPG-only $\\to$ cross-attn.",
           "cross_attention->bi_cross_attention": "Cross-attn. $\\to$ bidirectional",
           "cross_attention->joint_transformer": "Cross-attn. $\\to$ joint transf.",
           "cross_attention->bottleneck": "Cross-attn. $\\to$ bottleneck",
           "cross_attention->gated_cross_attention": "Cross-attn. $\\to$ gated",
           "ecg_only->resnet1d": "ECG-only $\\to$ ResNet1D",
           "resnet1d->cross_attention": "ResNet1D $\\to$ cross-attn.",
           "tcn->cross_attention": "TCN $\\to$ cross-attn.",
           "gbm_features->cross_attention": "GBM (feat. v1) $\\to$ cross-attn."}
    s["key"] = s.baseline + "->" + s.challenger
    s = s[s.key.isin(lab)].reset_index(drop=True)
    ci = s["boot_ci[f1@t50]"].str.replace("+", "", regex=False).str.strip("[]").str.split(",", expand=True).astype(float)
    y = np.arange(len(s))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    d = s["boot_diff[f1@t50]"]
    ax.errorbar(d, y, xerr=[d - ci[0], ci[1] - d], fmt="o", ms=3, lw=1, color="k", capsize=2)
    ax.axvline(0, color="r", lw=0.6, ls="--")
    ax.set_yticks(y, [lab[k] for k in s.key], fontsize=6.5)
    ax.set_xlabel("$\\Delta$F1 (challenger $-$ baseline), 95% bootstrap CI")
    fig.savefig(OUT / "forest.pdf")
    plt.close(fig)


# ---- 4. counterfactual PPG perturbations (runs/.../counterfactual_auc.csv) -------
def fig_counterfactual():
    c = pd.read_csv(ROOT / "runs/challenge2015_ppg/counterfactual/counterfactual_auc.csv")
    c = c[c.perturbed == "ppg"].set_index("variant")
    models = ["concat", "cross_attention", "resnet1d", "tcn"]
    names = ["Concat.", "Cross-attn.", "ResNet1D", "TCN"]
    perts = [("identity", "Unperturbed"), ("swap_same_label", "Other patient, same label"),
             ("shift_2s", "Shift 2 s"), ("swap_other_label", "Other patient, opposite label"),
             ("noise", "Noise")]
    cols = ["#4C78A8", "#72B7B2", "#9ECAE9", "#E45756", "#BAB0AC"]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.16
    for i, (k, nm) in enumerate(perts):
        ax.bar(x + (i - 2) * w, [c.loc[m, k] for m in models], w, label=nm, color=cols[i])
    ax.set_xticks(x, names)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC (ECG intact)")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.savefig(OUT / "counterfactual.pdf")
    plt.close(fig)


# ---- 5. noise robustness (runs/.../noise_auc_f1_vs_snr.csv) ----------------------
def fig_noise():
    n = pd.read_csv(ROOT / "runs/challenge2015_ppg/stress/noise_auc_f1_vs_snr.csv")
    b = n[n.family == "both_noise"].groupby(["variant", "level"]).auc.mean().unstack()
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for v, nm, col in [("resnet1d", "ResNet1D", "#4C78A8"), ("cross_attention", "Cross-attn.", "#D62728"),
                       ("concat", "Concat.", "#F58518"), ("ecg_only", "ECG-only", "#54A24B"),
                       ("ppg_only", "PPG-only", "#9467BD")]:
        ax.plot(b.columns, b.loc[v], "o-", ms=3, lw=1, label=nm, color=col)
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB), NSTDB ECG noise + synthetic PPG motion")
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3, lw=0.4)
    fig.savefig(OUT / "noise.pdf")
    plt.close(fig)


# ---- 6. cross-dataset transfer (runs/cross_dataset) ------------------------------
def fig_transfer():
    models = ["ecg_only", "ppg_only", "concat", "cross_attention", "resnet1d", "tcn"]
    names = ["ECG-only", "PPG-only", "Concat.", "Cross-attn.", "ResNet1D", "TCN"]
    a = json.load(open(ROOT / "runs/cross_dataset/cinc_to_vtac/results.json"))
    b = json.load(open(ROOT / "runs/cross_dataset/vtac_to_cinc_vt/results.json"))
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(x - 0.2, [a[m]["f1_mean"] for m in models], 0.38, label="Challenge 2015 $\\to$ VTaC", color="#E45756")
    ax.bar(x + 0.2, [b[m]["f1_mean"] for m in models], 0.38, label="VTaC $\\to$ Challenge 2015 (VT alarms)", color="#4C78A8")
    ax.set_xticks(x, names, rotation=20, fontsize=6.5)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "transfer.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_leak, fig_leaderboard, fig_forest, fig_counterfactual, fig_noise, fig_transfer):
        f()
    print("figures written to", OUT)
