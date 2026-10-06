"""Generate results.ipynb: the project's honest, leak-free results report.

The notebook only READS finished artefacts (runs/*.csv|json|png); it retrains
nothing and every code cell is fast. Missing files print a note instead of
failing, so re-running this script after more runs finish picks them up.

Usage:
    python scripts/build_results_notebook.py            # build + execute
    python scripts/build_results_notebook.py --no-exec  # build only
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


SETUP = r'''
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

# Locate the repo root whether the notebook is run from the root or elsewhere.
ROOT = Path.cwd()
for _p in [ROOT, *ROOT.parents]:
    if (_p / "runs").is_dir():
        ROOT = _p
        break
RUNS = ROOT / "runs"
P15 = RUNS / "challenge2015_ppg"

GROUP_COLORS = {"classical": "#4C78A8", "deep": "#F58518", "ensemble": "#54A24B"}
PROPOSED_COLOR = "#D62728"


def note(msg):
    display(Markdown(f"> *Note: {msg}*"))


def read_csv(path, **kw):
    path = Path(path)
    if not path.exists():
        note(f"`{path.relative_to(ROOT)}` not found (run may still be in progress).")
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        note(f"could not read `{path.name}`: {e}")
        return None


def read_json(path):
    path = Path(path)
    if not path.exists():
        note(f"`{path.relative_to(ROOT)}` not found (run may still be in progress).")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        note(f"could not read `{path.name}`: {e}")
        return None


def show_png(path, title=None, figsize=(10, 5)):
    path = Path(path)
    if not path.exists():
        note(f"`{path.relative_to(ROOT)}` not found.")
        return
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    ax.grid(False)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def is_proposed(name):
    return "proposed" in str(name).lower()

print("repo root:", ROOT)
'''

PROTOCOL = r'''
for f in ["docs/LEADERBOARD.md", "docs/evaluation_protocol.md", "runs/final_leaderboard.csv"]:
    print(f"{f:35s}", "ok" if (ROOT / f).exists() else "MISSING")
'''

LEAK = r'''
old = read_csv(RUNS / "legacy_pre_fix" / "comparison_table.csv")
new = read_csv(P15 / "comparison_table.csv")
if old is not None and new is not None:
    num = lambda s: s.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0].astype(float)
    keys = [v for v in ["ecg_only", "ppg_only", "concat", "cross_attention"] if v in set(old.variant) & set(new.variant)]
    o = old.set_index("variant").loc[keys]
    n = new.set_index("variant").loc[keys]
    tab = pd.DataFrame({
        "F1 old (leaky)": o["F1"], "F1 new (leak-free)": n["F1"],
        "dF1": (num(n["F1"]) - num(o["F1"])).round(3).values,
        "AUC old": o["AUC"], "AUC new": n["AUC"],
        "dAUC": (num(n["AUC"]) - num(o["AUC"])).round(3).values,
    })
    display(tab)

    x = np.arange(len(keys)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["F1", "AUC"]):
        ax.bar(x - w/2, num(o[col]), w, label="old (best epoch picked on test fold)", color="#BBBBBB")
        ax.bar(x + w/2, num(n[col]), w, label="new (inner validation split, 10 seeds)", color="#4C78A8")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15); ax.set_ylabel(col)
        ax.set_ylim(0.5, 0.9); ax.set_title(f"{col}: before vs after the leakage fix")
    axes[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.show()
    if {"concat", "cross_attention"} <= set(keys):
        d_old = num(o["F1"])["cross_attention"] - num(o["F1"])["concat"]
        d_new = num(n["F1"])["cross_attention"] - num(n["F1"])["concat"]
        print(f"cross_attention - concat (F1): old {d_old:+.3f} -> new {d_new:+.3f}")
'''

LEADER = r'''
lb = read_csv(RUNS / "final_leaderboard.csv")


def leaderboard_chart(df, title):
    df = df.copy()
    order = {"classical": 0, "deep": 1, "ensemble": 2}
    df["_o"] = df["group"].map(order)
    df = df.sort_values(["_o", "accuracy"], ascending=[True, True]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.38 * len(df) + 1.6), sharey=True)
    colors = [PROPOSED_COLOR if is_proposed(m) else GROUP_COLORS.get(g, "grey")
              for m, g in zip(df.model, df.group)]
    for ax, (col, lab) in zip(axes, [("accuracy", "Accuracy"), ("f1", "F1 (true-alarm class)"), ("auc", "AUC")]):
        bars = ax.barh(df.model, df[col], color=colors, edgecolor=["black" if is_proposed(m) else "none" for m in df.model],
                       linewidth=1.5)
        for b, v in zip(bars, df[col]):
            ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7)
        ax.set_xlim(max(0.5, df[col].min() - 0.08), min(1.0, df[col].max() + 0.06))
        ax.set_title(lab); ax.grid(axis="y", alpha=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [*GROUP_COLORS.values(), PROPOSED_COLOR]]
    fig.legend(handles, [*GROUP_COLORS.keys(), "proposed cross-attention"], loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.04, fontsize=12)
    plt.tight_layout(); plt.show()


if lb is not None:
    for ds in lb["dataset"].unique():
        sub = lb[lb["dataset"] == ds]
        leaderboard_chart(sub, f"{ds}  (n_seeds={int(sub.n_seeds.iloc[0])})")
        cols = ["model", "group", "accuracy", "f1", "auc", "sens", "spec", "challenge", "bootstrap_f1_best_minus_model"]
        display(sub[cols].sort_values("accuracy", ascending=False).round(3).reset_index(drop=True))
    # headline: best of each group per dataset
    best = lb.sort_values("accuracy", ascending=False).groupby(["dataset", "group"]).head(1)
    display(best[["dataset", "group", "model", "accuracy", "f1", "auc"]].round(3).sort_values(["dataset", "accuracy"], ascending=[True, False]))
'''

SIG = r'''
sig = read_csv(P15 / "significance.csv")
if sig is not None:
    cols = {
        "baseline": "baseline", "challenger": "challenger", "f1_mean_diff": "dF1 (mean)",
        "nb_p": "Nadeau-Bengio p", "nb_p_holm": "NB p (Holm)",
        "boot_ci[f1@t50]": "bootstrap dF1 95% CI", "boot_p[f1@t50]": "boot p",
        "boot_p[f1@t50]_holm": "boot p (Holm)",
        "boot_ci[specificity@s95]": "spec@sens95 dSpec CI", "boot_p[specificity@s95]_holm": "spec@s95 p (Holm)",
    }
    t = sig[[c for c in cols if c in sig.columns]].rename(columns=cols)
    # robust = Holm-bootstrap significant AND NB test agrees in sign (per protocol)
    if {"boot_p (Holm)", "dF1 (mean)"} <= set(t.columns):
        t["Holm-significant (boot)"] = t["boot p (Holm)"] < 0.05
        t["NB p(Holm) < 0.05"] = t["NB p (Holm)"] < 0.05
        t["verdict"] = np.where(t["Holm-significant (boot)"] & t["NB p(Holm) < 0.05"], "robust",
                         np.where(t["Holm-significant (boot)"], "bootstrap only: not robust", "n.s."))
    display(t.round(4))
    print("Key row: concat -> cross_attention:")
    display(t[(t.baseline == "concat") & (t.challenger == "cross_attention")].round(4))
    print("Protocol: significant only if it survives Holm on the bootstrap AND the NB test agrees; "
          "otherwise 'not robust'. With 50 (seed, fold) pairs the NB test is conservative.")
'''

CF = r'''
cfd = RUNS / "challenge2015_ppg" / "counterfactual"
cf_auc = read_csv(cfd / "counterfactual_auc.csv")
cf_idx = read_csv(cfd / "alignment_indices.csv")
if cf_idx is not None:
    display(cf_idx.round(3))
    piv = cf_idx.pivot(index="variant", columns="perturbed")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ai = piv["alignment_index(identity-swap_same)"]; ti = piv["timing_index(identity-phase_scramble)"]
    ai.plot.bar(ax=axes[0], color=["#4C78A8", "#F58518"]); axes[0].set_title("alignment index (AUC drop when the partner window is swapped for another same-label alarm)")
    axes[0].title.set_fontsize(8); axes[0].set_ylabel("AUC drop")
    ms = piv["max_shift_drop"]; ms.plot.bar(ax=axes[1], color=["#4C78A8", "#F58518"])
    axes[1].set_title("max AUC drop under 0.25-2 s time shift of one channel"); axes[1].title.set_fontsize(8)
    for a in axes: a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.show()
if cf_auc is not None:
    display(cf_auc.round(3))
    ppg = cf_auc[cf_auc.perturbed == "ppg"].set_index("variant")
    cols = [c for c in ["identity", "swap_same_label", "swap_other_label", "shift_1s", "phase_scramble", "noise"] if c in ppg.columns]
    ppg[cols].plot.bar(figsize=(10, 3.8)); plt.ylabel("AUC"); plt.title("Perturbing the PPG channel"); plt.xticks(rotation=25)
    plt.tight_layout(); plt.show()
'''

ATT = r'''
s = read_json(P15 / "attention_ptt" / "summary.json")
if s is not None:
    keys = ["n_windows", "n_windows_with_ptt", "token_resolution_ms",
            "window_level_spearman_rho(attn_lag, measured_PTT)", "spearman_p", "permutation_p",
            "beat_level_spearman_rho", "attention_mass_in_physiological_lag_range",
            "mass_if_attention_were_uniform", "median_attn_lag_s", "median_measured_ptt_s"]
    display(pd.Series({k: s.get(k) for k in keys}, name="value").to_frame())
    rho = s.get("window_level_spearman_rho(attn_lag, measured_PTT)")
    m, u = s.get("attention_mass_in_physiological_lag_range"), s.get("mass_if_attention_were_uniform")
    if rho is not None and m is not None:
        print(f"window-level rho = {rho:+.3f} (permutation p = {s.get('permutation_p')}); "
              f"attention mass in physiological lag range = {m:.3f} vs {u:.3f} if uniform.")
show_png(P15 / "attention_ptt" / "attention_vs_ptt.png", "Cross-attention lag vs measured PTT", figsize=(10, 5))
'''

STRESS = r'''
sd = P15 / "stress"
noise = read_csv(sd / "noise_auc_f1_vs_snr.csv")
if noise is not None:
    show_variants = [v for v in ["ecg_only", "ppg_only", "concat", "cross_attention", "resnet1d", "tcn", "inceptiontime"]
                     if v in set(noise.variant)]
    d = noise[noise.family == "both_noise"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, metric in zip(axes, ["auc", "f1"]):
        for v in show_variants:
            g = d[d.variant == v].groupby("level")[metric].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", ms=3, lw=2.2 if v == "cross_attention" else 1.2,
                    color=PROPOSED_COLOR if v == "cross_attention" else None, label=v)
        ax.set_xlabel("SNR (dB), both channels, mean over em/ma/bw noise"); ax.set_ylabel(metric.upper()); ax.set_title(f"{metric.upper()} vs SNR")
    axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.show()
deg = read_csv(sd / "degradation_table.csv")
if deg is not None:
    display(deg.round(3))
mm = read_csv(sd / "missing_modality.csv")
if mm is not None:
    z = mm[mm.kind == "missing_zero"].pivot_table(index="variant", columns="target", values="auc")
    clean = read_csv(sd / "degradation_table.csv")
    if clean is not None and "clean_auc" in clean.columns:
        z.insert(0, "clean", clean.set_index("variant")["clean_auc"].reindex(z.index))
    z = z.sort_values("clean" if "clean" in z.columns else z.columns[0], ascending=False)
    z.plot.bar(figsize=(11, 4), color=["#999999", "#4C78A8", "#F58518"][: z.shape[1]])
    plt.ylabel("AUC"); plt.title("Missing modality (channel zeroed): AUC when ECG / PPG is absent")
    plt.xticks(rotation=30); plt.tight_layout(); plt.show()
    display(z.round(3))
for fn, tt in [("noise_curves_auc.png", "Noise curves (AUC)"), ("ptt_dropout_curves.png", "PTT jitter / dropout curves"),
               ("missing_modality.png", "Missing modality")]:
    show_png(sd / fn, tt, figsize=(11, 5))
sc = read_json(sd / "signature_check.json")
if sc is not None:
    print("Noise-signature leakage check (can a classifier detect the label from the injected-noise signature?):",
          "noise_component_vs_label_auc =", round(sc.get("noise_component_vs_label_auc", float("nan")), 3),
          "(0.5 = no label information in the injected noise)")
'''

SUBSETS = r'''
msd = RUNS / "challenge2015_recovered" / "modality_subsets"
sub = read_csv(msd / "subsets.csv")
rob = read_csv(msd / "robustness.csv")
sh = read_csv(msd / "shapley.csv")
if sub is not None:
    s2 = sub.sort_values("auc")
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(s2) + 1))
    err = None
    if {"auc_ci_low", "auc_ci_high"} <= set(s2.columns):
        err = [s2.auc - s2.auc_ci_low, s2.auc_ci_high - s2.auc]
    cmap = {1: "#BBBBBB", 2: "#8DA0CB", 3: "#4C78A8", 4: "#1F3F73"}
    ax.barh(s2.subset, s2.auc, xerr=err, color=[cmap.get(n, "grey") for n in s2.n_modalities], capsize=2)
    ax.set_xlabel("AUC (95% CI)"); ax.set_xlim(0.5, 0.95); ax.set_title("Sensor subsets (recovered 4-sensor cohort, n_windows = %s)" % sub.n_windows.iloc[0])
    plt.tight_layout(); plt.show()
    display(sub.round(3))
if rob is not None:
    display(rob.round(3))
if sh is not None:
    display(sh.round(3))
    sh.set_index("modality").shapley_auc.plot.bar(figsize=(5, 3), color="#4C78A8"); plt.ylabel("Shapley AUC value")
    plt.title("Per-sensor contribution"); plt.tight_layout(); plt.show()
print("Caveat: this is the small 'recovered' cohort (a few hundred windows, CIs wide) and is NOT comparable to the 592-record PPG cohort.")
'''

LABEL = r'''
led = RUNS / "labeleff"
le = read_csv(led / "summary.csv")
pv = read_csv(led / "paired_vs_scratch.csv")
if le is not None:
    variants = list(le.variant.unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        for (init, mode), g in le[le.variant == v].groupby(["init", "mode"]):
            g = g.sort_values("label_fraction")
            ax.plot(g.label_fraction, g.F1, marker="o", ms=3, ls="-" if mode == "finetune" else "--",
                    lw=2.4 if init == "scratch" else 1.2, color="black" if init == "scratch" else None, label=f"{init}/{mode}")
        ax.set_title(v); ax.set_xlabel("label fraction"); ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1.0"])
    axes[0][0].set_ylabel("F1"); axes[0][-1].legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.show()
if pv is not None:
    p = pv.assign(sig=pv.nb_p < 0.05)
    display(p.pivot_table(index=["variant", "init", "mode"], columns="label_fraction", values="dF1_vs_scratch").round(3))
    print("Paired contrasts with Nadeau-Bengio p < 0.05:", int(p.sig.sum()), "of", len(p))
    display(p[p.sig].round(3))
show_png(led / "label_efficiency.png", "Label efficiency (pipeline figure)", figsize=(10, 5))
'''

XD = r'''
rows = []
for d in sorted((RUNS / "cross_dataset").glob("*")) if (RUNS / "cross_dataset").exists() else []:
    r = read_json(d / "results.json")
    if not r:
        continue
    for variant, m in r.items():
        if isinstance(m, dict) and "auc_mean" in m:
            rows.append(dict(direction=d.name, variant=variant, F1=m.get("f1_mean"), AUC=m.get("auc_mean"),
                             AUC_std=m.get("auc_std"), sens=m.get("sensitivity_mean"), spec=m.get("specificity_mean")))
if rows:
    xd = pd.DataFrame(rows)
    display(xd.round(3))
    piv = xd.pivot(index="variant", columns="direction", values="AUC")
    piv.plot.bar(figsize=(9, 4)); plt.ylabel("AUC"); plt.ylim(0.5, 1); plt.xticks(rotation=20)
    plt.title("Cross-dataset transfer (train on one corpus, test on the other)"); plt.tight_layout(); plt.show()
else:
    note("no cross-dataset results found.")
'''

SAFETY = r'''
sf = P15 / "safety"
ut = read_json(sf / "concat_utility.json")
if ut:
    display(pd.Series(ut, name="value").to_frame())
    print(f"With a certified miss-rate target (alpha = {ut.get('alpha_miss_target')}, confidence {ut.get('confidence')}): "
          f"{ut.get('false_alarms_suppressed_pct', float('nan')):.1f}% of false alarms suppressed at the cost of "
          f"{ut.get('true_alarms_missed_pct', float('nan')):.1f}% true alarms missed; a threshold was certifiable in {ut.get('folds_with_certified_threshold')} folds.")
cal = read_csv(sf / "concat_calibration.csv")
if cal is not None:
    display(cal.drop(columns=["fold"]).mean().round(3).to_frame("mean over folds").T)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["ECE raw", "ECE calibrated"], [cal.ece_raw.mean(), cal.ece_cal.mean()], color=["#BBBBBB", "#4C78A8"])
    ax.set_title("Temperature scaling (fit on validation)"); plt.tight_layout(); plt.show()
dc = read_csv(sf / "concat_decision_curve.csv")
if dc is not None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(dc.threshold, dc.net_benefit_model, label="model")
    ax.plot(dc.threshold, dc.net_benefit_alarm_all, "--", label="treat all alarms as true")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit"); ax.legend()
    ax.set_title("Decision curve (concat model)"); plt.tight_layout(); plt.show()
    gap = (dc.net_benefit_model - dc.net_benefit_alarm_all)[dc.threshold <= 0.5]
    print(f"Max net-benefit gain over 'alarm all' (thresholds <= 0.5): {gap.max():+.3f} at threshold {dc.threshold[gap.idxmax()]:.2f}")
'''


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = []
    c.append(md("""
# ECG + PPG alarm verification: leak-free results

This notebook is generated by `scripts/build_results_notebook.py` and reads only finished
artefacts under `runs/` (nothing is retrained). It **supersedes** the first version of this report,
whose numbers were optimistically biased because the best epoch was selected on the same fold that was reported.
Sections whose files are missing print a note; re-run the builder after more runs finish.
"""))
    c.append(code(SETUP))

    c.append(md("""
## 1. Protocol and the leakage fix

* One alarm event = one 10 s window ending at alarm onset; all windows of a record stay on one side of every split.
* Challenge 2015: record-wise 5-fold CV x 10 seeds. VTaC: official patient-disjoint split (4060/495/482 events), 5 seeds.
* Inside each training fold, 15 % of the records form an **inner validation set** for early stopping, epoch choice,
  temperature scaling and threshold selection. The outer test fold is touched once.
* All models see identical splits; neural variants are parameter-matched (~261k); hyper-parameters fixed a priori. Thresholds are 0.5 in the leaderboard.
* Statistics: pre-registered comparison family, Nadeau-Bengio corrected t-test, paired record-level bootstrap (5000), Holm correction.
  A result is "significant" only if it survives Holm on the bootstrap **and** agrees with the NB test.
* Not controlled: possible patient overlap between corpora, no VTaC device ids, tiny validation sets (15-25 positives), no prospective data.

Full protocol: `docs/evaluation_protocol.md`; full tables: `docs/LEADERBOARD.md`.
"""))
    c.append(code(PROTOCOL))
    c.append(md("""
### Before vs after the leakage fix (Challenge 2015, PPG cohort)
Old numbers come from `runs/legacy_pre_fix/comparison_table.csv` (3 seeds, test-fold epoch selection);
new from `runs/challenge2015_ppg/comparison_table.csv` (10 seeds, inner validation). Every fused/neural number dropped,
and the old cross-attention advantage over concatenation (+0.024 F1) collapsed to about +0.006.
"""))
    c.append(code(LEAK))

    c.append(md("""
## 2. Final leaderboard

Bars are grouped as classical / deep / ensemble; the **proposed CNN cross-attention model is red with a black outline**.
Thresholds fixed at 0.5. The central finding: engineered-feature gradient boosting and ensembles beat all deep models,
and the proposed cross-attention model is among the weakest.
"""))
    c.append(code(LEADER))

    c.append(md("""
## 3. Pre-registered significance (Challenge 2015)

The comparison family was fixed in `configs/config.yaml` before final results existed. "robust" means the result
survives Holm correction on the paired bootstrap **and** the Nadeau-Bengio test agrees; otherwise it is reported as not robust.
"""))
    c.append(code(SIG))

    c.append(md("""
## 4. Counterfactual alignment: do fused models use timing?

Test-time perturbations of one channel: swap with another alarm's channel of the same label (*alignment index*),
phase scrambling (*timing index*), shifts of 0.25-2 s, and replacement by noise (*pulse-presence cue*).

**Interpretation.** Swapping in a different same-label partner window and shifting one channel by up to 2 s barely change AUC
(alignment index and max-shift drop are of the order 0.00-0.06), while replacing the channel with noise costs far more, and scrambling the phase of the ECG also hurts (timing index 0.12-0.32; only 0.03-0.07 for PPG).
The fused models therefore use each channel's *content* (is there a plausible pulse/rhythm) and **not the beat-to-beat ECG-PPG timing**
(the physiological signal cross-attention was designed to exploit). Large "swap_other_label" drops show they are label-sensitive per channel,
not alignment-sensitive.
"""))
    c.append(code(CF))

    c.append(md("""
## 5. Does cross-attention attend to pulse transit time?

Attention lag (where PPG tokens attend relative to the ECG R-peak) is compared to measured PTT on windows where PTT could be measured.
A correlation near zero and attention mass in the physiological lag range no larger than uniform means the learned attention does **not**
recover PTT; attention maps should not be presented as physiological evidence.
"""))
    c.append(code(ATT))

    c.append(md("""
## 6. Stress benchmark and missing modality

Noise is added at test time only (NSTDB electrode-motion / baseline-wander / muscle noise on ECG, synthetic band-limited noise on PPG),
at several SNR levels; models are not trained on it. The signature check confirms the injected noise carries no label information.
Missing-modality = the channel is zeroed or replaced by noise.
"""))
    c.append(code(STRESS))

    c.append(md("""
## 7. Modality subsets (recovered multi-sensor cohort)

AUC for every subset of the four available sensors (two ECG leads, PPG, ABP) on the small *recovered* cohort,
with bootstrap CIs and Shapley attribution. This is a different, much smaller cohort than the 592-record PPG cohort.
"""))
    c.append(code(SUBSETS))

    c.append(md("""
## 8. Label efficiency (self-supervised pre-training)

F1 at 10/25/50/100 % of training labels for scratch training versus masked / contrastive / cross-modal pre-training, either fine-tuned or frozen.
`paired_vs_scratch.csv` holds paired differences with Nadeau-Bengio p-values. Pre-training should only be claimed to help where these contrasts are significant.
"""))
    c.append(code(LABEL))

    c.append(md("""
## 9. Cross-dataset transfer

Train on one corpus, test on the other (Challenge 2015 -> VTaC, and VTaC -> Challenge 2015 VT alarms). Different institutions, monitors and
label definitions, so scores are a generalisation signal rather than a benchmark.
"""))
    c.append(code(XD))

    c.append(md("""
## 10. Safety layer

Temperature-scaling calibration, a certified (high-confidence) miss-rate threshold with a three-way policy
(auto-suppress / defer to a human / high-priority), and a decision-curve analysis. Shown for the concatenation model.
"""))
    c.append(code(SAFETY))

    c.append(md("""
## 11. Limitations

* Small cohort: 592 Challenge 2015 records with PPG; fold-to-fold standard deviations are large and several comparisons are underpowered.
* The best models are classical feature models and ensembles; the deep fusion architectures (including the proposed one) did not earn their complexity,
  and cross-attention is not distinguishable from concatenation.
* Fused models do not use ECG-PPG timing (section 4) and attention does not track PTT (section 5): the physiological motivation is not supported.
* Stress-test noise is synthetic or from a different database (NSTDB), so it approximates but does not equal real ICU artefact.
* Cross-dataset transfer and the multi-sensor "recovered" cohort are small and CIs are wide; the VTaC result is for VT alarms only.
* Possible patient overlap between public corpora cannot be ruled out; validation sets are tiny, so validation-chosen operating points are noisy.
* Retrospective, US ICU data only; no prospective validation. Not a medical device.
* Sections marked with a note above rely on runs that were not yet finished when the notebook was generated.
"""))
    nb["cells"] = c
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return nb


def main() -> None:
    nb = build()
    out = ROOT / "results.ipynb"
    if "--no-exec" not in sys.argv:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=300, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}},
                                allow_errors=True)
        client.execute()
        failed = [i for i, cell in enumerate(nb.cells) if cell.cell_type == "code" and
                  any(o.get("output_type") == "error" for o in cell.get("outputs", []))]
        nbf.write(nb, out)
        print(f"wrote {out}")
        print("failed code cells:", failed if failed else "none")
        sys.exit(1 if failed else 0)
    nbf.write(nb, out)
    print(f"wrote {out} (not executed)")


if __name__ == "__main__":
    main()
