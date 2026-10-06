"""Charts used by docs/REPORT.md (reads existing outputs only)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
COL = {"classical": "#4C78A8", "deep": "#9AA5B1", "ensemble": "#54A24B"}


def leaderboard(dataset_prefix, fname, title):
    df = pd.read_csv(ROOT / "runs" / "final_leaderboard.csv")
    d = df[df.dataset.str.startswith(dataset_prefix)].sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(d) + 1.2))
    colors = ["#D62728" if "proposed" in m else COL[g] for m, g in zip(d.model, d.group)]
    ax.barh(d.model, d.accuracy, color=colors)
    for y, (a, f) in enumerate(zip(d.accuracy, d.f1)):
        ax.text(a + 0.002, y, f"{a:.3f}  (F1 {f:.3f})", va="center", fontsize=7)
    ax.set_xlim(0.7, 0.95)
    ax.set_xlabel("accuracy (threshold 0.5; axis starts at 0.7)")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


leaderboard("Challenge 2015 (PPG cohort, 5-fold", "leaderboard_cinc.png", "Challenge 2015, 10 seeds (red = originally proposed model)")
leaderboard("VTaC", "leaderboard_vtac.png", "VTaC official split, 5 seeds (red = originally proposed model)")

# before / after leakage fix
old = pd.read_csv(ROOT / "runs" / "legacy_pre_fix" / "comparison_table.csv")
new = pd.read_csv(ROOT / "runs" / "challenge2015_ppg" / "comparison_table.csv")
f = lambda s: float(str(s).split("+/-")[0])  # noqa: E731
names = ["ppg_only", "ecg_only", "concat", "cross_attention"]
o = [f(old.set_index("variant").loc[n, "F1"]) for n in names]
n_ = [f(new.set_index("variant").loc[n, "F1"]) for n in names]
fig, ax = plt.subplots(figsize=(6, 3.4))
x = range(len(names))
ax.bar([i - 0.2 for i in x], o, 0.4, label="before fix (leaky, 3 seeds)", color="#E45756")
ax.bar([i + 0.2 for i in x], n_, 0.4, label="after fix (inner validation, 10 seeds)", color="#4C78A8")
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("F1"); ax.set_ylim(0.5, 0.85)
for i, (a, b) in enumerate(zip(o, n_)):
    ax.text(i - 0.2, a + 0.004, f"{a:.2f}", ha="center", fontsize=7); ax.text(i + 0.2, b + 0.004, f"{b:.2f}", ha="center", fontsize=7)
ax.legend(fontsize=7, frameon=False); ax.set_title("Effect of removing the validation leak", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "leakage_before_after.png", dpi=160); plt.close(fig)

# counterfactual AUC
t = pd.read_csv(ROOT / "runs" / "challenge2015_ppg" / "counterfactual" / "counterfactual_auc.csv")
t = t[t.perturbed == "ppg"].set_index("variant")
vs = [v for v in ["ppg_only", "concat", "cross_attention", "resnet1d", "tcn"] if v in t.index]
kinds = ["identity", "swap_same_label", "shift_2s", "phase_scramble", "swap_other_label", "noise"]
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.8 / len(kinds)
for k, kd in enumerate(kinds):
    ax.bar([i + k * w for i in range(len(vs))], [t.loc[v, kd] for v in vs], w, label=kd)
ax.set_xticks([i + 0.4 - w / 2 for i in range(len(vs))]); ax.set_xticklabels(vs, fontsize=8)
ax.set_ylabel("AUC"); ax.set_ylim(0.2, 1.0); ax.legend(fontsize=6, ncol=3, frameon=False)
ax.set_title("AUC when the PPG is perturbed (ECG intact)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "counterfactual_ppg.png", dpi=160); plt.close(fig)
print("figures written", sorted(p.name for p in OUT.glob("*.png")))
