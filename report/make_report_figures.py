"""Figures for the Word report. Every data figure is computed from files under runs/.
Run from the repository root:  python report/make_report_figures.py
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(exist_ok=True)

# ---- 1. re-use the paper's data figures, saved as PNG for Word -------------------
sys.path.insert(0, str(ROOT / "paper"))
import make_figures as mf
mf.OUT = OUT
_orig = plt.Figure.savefig
def _png(self, fname, *a, **k):
    k["dpi"] = 220
    return _orig(self, str(fname).replace(".pdf", ".png"), *a, **k)
plt.Figure.savefig = _png
for f in (mf.fig_leak, mf.fig_leaderboard, mf.fig_forest, mf.fig_counterfactual, mf.fig_noise, mf.fig_transfer):
    f()
plt.Figure.savefig = _orig
plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.bbox": "tight", "figure.dpi": 150})

# ---- 2. confusion matrices from results.json fold counts --------------------------
def counts(path, key):
    fm = json.load(open(ROOT / path))[key]["fold_metrics"]
    t = {k: sum(f[k] for f in fm) for k in ("tp", "fn", "fp", "tn")}
    seeds = len({f["seed"] for f in fm})
    return t, seeds

CM = {
    "Challenge 2015: cross-attention (ladder)": counts("runs/challenge2015_ppg/results.json", "cross_attention"),
    "Challenge 2015: attention CNN-GRU recipe": counts("runs/deep_plus_cinc/results.json", "deepplus"),
    "VTaC: cross-attention (ladder)": counts("runs/vtac_official/results.json", "cross_attention"),
    "VTaC: attention CNN-GRU recipe": counts("runs/deep_plus_vtac/results.json", "deepplus"),
}
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6))
report = {}
for ax, (title, (t, seeds)) in zip(axes.ravel(), CM.items()):
    m = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]]) / seeds
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > m.max() * 0.55 else "black")
    ax.set_xticks([0, 1], ["False alarm", "True alarm"])
    ax.set_yticks([0, 1], ["False alarm", "True alarm"], rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=8.5)
    report[title] = {**t, "seeds": seeds}
fig.tight_layout()
fig.savefig(OUT / "confusion.png", dpi=220)
plt.close(fig)
json.dump(report, open(OUT / "confusion_counts.json", "w"), indent=1)

# ---- 3. best-epoch distribution of the recipe (from member_info) -------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
for ax, (p, label, cap) in zip(axes, [("runs/deep_plus_cinc/results.json", "Challenge 2015", 30),
                                      ("runs/deep_plus_vtac/results.json", "VTaC", 20)]):
    fm = json.load(open(ROOT / p))["deepplus"]["fold_metrics"]
    ep = [m["epoch_ema"] for f in fm for m in f["member_info"]]
    ax.hist(ep, bins=np.arange(0.5, cap + 1.5, 1), color="#4C78A8")
    ax.axvline(cap, color="r", ls="--", lw=0.8)
    ax.set_xlabel("Best epoch (EMA, inner-val. AUC)")
    ax.set_ylabel("Ensemble members")
    ax.set_title(f"{label} ({len(ep)} members, cap {cap})", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "best_epoch.png", dpi=220)
plt.close(fig)

# ---- 4. workflow diagram ------------------------------------------------------------
def boxflow(steps, fname, w=7.4, h=2.3, cols=4, color="#DCE9F5"):
    rows = -(-len(steps) // cols)
    fig, ax = plt.subplots(figsize=(w, h * rows / 1.0 * 0.75))
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    pos = []
    for i, s in enumerate(steps):
        r, c = divmod(i, cols)
        if r % 2 == 1: c = cols - 1 - c
        x, y = c + 0.08, rows - 1 - r + 0.18
        ax.add_patch(mp.FancyBboxPatch((x, y), 0.84, 0.64, boxstyle="round,pad=0.02", fc=color, ec="#2F4B6E", lw=1))
        ax.text(x + 0.42, y + 0.32, s, ha="center", va="center", fontsize=6.3)
        pos.append((x, y, r, c))
    for (x1, y1, r1, c1), (x2, y2, r2, c2) in zip(pos[:-1], pos[1:]):
        if r1 == r2:
            d = 1 if c2 > c1 else -1
            ax.annotate("", xy=(x2 + (0.0 if d > 0 else 0.84), y2 + 0.32), xytext=(x1 + (0.84 if d > 0 else 0.0), y1 + 0.32),
                        arrowprops=dict(arrowstyle="-|>", color="#2F4B6E"))
        else:
            ax.annotate("", xy=(x2 + 0.42, y2 + 0.64), xytext=(x1 + 0.42, y1),
                        arrowprops=dict(arrowstyle="-|>", color="#2F4B6E"))
    fig.savefig(OUT / fname, dpi=220)
    plt.close(fig)

boxflow(["Problem definition:\nverify an ECG-triggered\nICU alarm",
         "Data collection:\nCinC 2015 and VTaC\n(PhysioNet)",
         "Inspection and\npreprocessing: 10 s window,\nfilter, z-score, quality",
         "Record-wise splits\n+ inner validation set",
         "Model design:\nladder, zoo, features,\nCNN-GRU recipe",
         "Training and\nvalidation: early stopping,\nmodel selection",
         "Testing: outer fold\n(or official test split)\nonce",
         "Results, statistics,\ncounterfactual and\nrobustness analysis"], "workflow.png")

# ---- 5. architecture diagrams ---------------------------------------------------------
def arch(blocks, fname, w=9.2):
    fig, ax = plt.subplots(figsize=(w, 3.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    def box(x, y, wd, ht, txt, fc="#DCE9F5"):
        ax.add_patch(mp.FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.03", fc=fc, ec="#2F4B6E", lw=1))
        ax.text(x + wd / 2, y + ht / 2, txt, ha="center", va="center", fontsize=6.0)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#2F4B6E"))
    for b in blocks:
        if b[0] == "box": box(*b[1:])
        else: arrow(*b[1:])
    fig.savefig(OUT / fname, dpi=220)
    plt.close(fig)

arch([
 ("box", 0.05, 3.3, 1.3, 1.1, "ECG\nwindow\n(2500)", "#FFF2CC"),
 ("box", 0.05, 0.6, 1.3, 1.1, "PPG / ABP\nwindow\n(2500)", "#FFF2CC"),
 ("box", 1.7, 2.9, 2.9, 1.9, "CNN encoder (ECG)\n3 x [Conv1d, BN, ReLU,\nMaxPool 2, Dropout 0.3]\nk = 15, 9, 7\nch = 32, 64, 128\n-> (312 x 128)"),
 ("box", 1.7, 0.2, 2.9, 1.9, "CNN encoder (PPG)\nsame layers,\nown weights\n-> (312 x 128)"),
 ("box", 5.1, 0.9, 2.9, 3.2, "Cross-attention fusion\nLinear 128->128\n+ positional encoding\nECG = queries\nPPG = keys, values\n4 heads\nresidual + LayerNorm\n-> (312 x 128)", "#E2F0D9"),
 ("box", 8.4, 1.8, 1.4, 1.4, "Mean pool\nover time\n-> (128)"),
 ("box", 10.1, 1.1, 1.85, 2.8, "Head\nLinear 128->64\nReLU, Dropout\nLinear 64->1\nsigmoid\n-> P(true alarm)", "#FCE4D6"),
 ("arrow", 1.35, 3.85, 1.7, 3.85), ("arrow", 1.35, 1.15, 1.7, 1.15),
 ("arrow", 4.6, 3.85, 5.1, 3.4), ("arrow", 4.6, 1.15, 5.1, 1.7),
 ("arrow", 8.0, 2.5, 8.4, 2.5), ("arrow", 9.8, 2.5, 10.1, 2.5),
], "arch_ladder.png")

arch([
 ("box", 0.05, 3.3, 1.3, 1.1, "ECG\nwindow\n(2500)", "#FFF2CC"),
 ("box", 0.05, 0.6, 1.3, 1.1, "PPG / ABP\nwindow\n(2500)", "#FFF2CC"),
 ("box", 1.7, 2.7, 3.9, 2.2, "ECG branch\n3 x [Conv1d, BN, ReLU, MaxPool 4]\nch = 21, 41, 82; k = 15, 7, 5 -> (39 x 82)\nBiGRU, 82 units per direction -> (39 x 164)\nadditive attention pooling -> (164)"),
 ("box", 1.7, 0.1, 3.9, 2.2, "PPG branch\nsame structure, own weights\n-> (164)"),
 ("box", 6.0, 1.5, 1.5, 2.0, "Concatenate\n-> (328)", "#E2F0D9"),
 ("box", 7.9, 1.2, 2.0, 2.6, "Head\nLinear 328->64\nReLU, Dropout 0.3\nLinear 64->1", "#FCE4D6"),
 ("box", 10.2, 1.2, 1.75, 2.6, "Training/test\nrecipe\nx3 members\nEMA weights\nTTA (5 shifts)\nmean probability"),
 ("arrow", 1.35, 3.85, 1.7, 3.85), ("arrow", 1.35, 1.15, 1.7, 1.15),
 ("arrow", 5.6, 3.8, 6.0, 3.1), ("arrow", 5.6, 1.2, 6.0, 1.9),
 ("arrow", 7.5, 2.5, 7.9, 2.5), ("arrow", 9.9, 2.5, 10.2, 2.5),
], "arch_recipe.png")

# ---- 6. equation images -----------------------------------------------------------------
EQ = {
 "eq_relu": r"$f(x)=\max(0,\,x)$",
 "eq_attn": r"$\mathrm{Attn}(E,P)=\mathrm{softmax}\!\left(\dfrac{(EW_Q)(PW_K)^{\top}}{\sqrt{d_k}}\right)PW_V$",
 "eq_gru1": r"$z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z),\;\; r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)$",
 "eq_gru2": r"$\tilde h_t=\tanh\!\left(Wx_t+U(r_t\odot h_{t-1})+b\right),\;\; h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$",
 "eq_addattn": r"$e_t=v^{\top}\tanh(Wh_t+b),\;\; a_t=\dfrac{\exp(e_t)}{\sum_{s}\exp(e_s)},\;\; c=\sum_{t}a_t\,h_t$",
 "eq_sigmoid": r"$p=\sigma(z)=\dfrac{1}{1+e^{-z}}$",
 "eq_focal": r"$\mathcal{L}=\dfrac{1}{N}\sum_{i=1}^{N}\left(1-p_{t,i}\right)^{\gamma}\,\ell_i,\;\;\ell_i=-\left[w\,y_i\log p_i+(1-y_i)\log(1-p_i)\right],\;\; \gamma=2$",
 "eq_adamw": r"$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\;\; v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{2},\;\; \theta_t=\theta_{t-1}-\eta_t\left(\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\right)$",
 "eq_ema": r"$\bar\theta_t=\beta\,\bar\theta_{t-1}+(1-\beta)\,\theta_t$",
 "eq_metrics": r"$\mathrm{Acc}=\dfrac{TP+TN}{TP+TN+FP+FN},\;\; \mathrm{Prec}=\dfrac{TP}{TP+FP},\;\; \mathrm{Rec}=\dfrac{TP}{TP+FN},\;\; F_1=\dfrac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}}$",
 "eq_score": r"$S=\dfrac{TP+TN}{TP+TN+FP+5\,FN}$",
 "eq_nb": r"$t=\dfrac{\bar d}{\sqrt{\left(\frac{1}{J}+\frac{n_{\mathrm{test}}}{n_{\mathrm{train}}}\right)s_d^{2}}}$",
}
for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=13)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    plt.close(fig)
print("done")
