"""Error taxonomy + case studies on PhysioNet/CinC 2015 (updates.md 6.12).

For every variant, each record has exactly one out-of-fold prediction per seed
(10 seeds). The *consensus error rate* is the fraction of seeds in which that
prediction (threshold 0.5) is wrong. Records are then classified as

  persistent FN : true alarm, error rate >= PERSIST (missed true alarms)
  persistent FP : false alarm, error rate >= PERSIST (not suppressed)
  unstable      : PERSIST > error rate > STABLE  (split by label)
  stable-correct: error rate <= STABLE

and profiled by alarm type, signal-quality tercile, beat-detector disagreement,
recording setup / pulse source and heart-rate regime. Subgroup tests are
Fisher exact (one level vs rest), Holm-corrected per (variant, error type)
family; a record-level repeated stratified-CV logistic regression on the
hand-crafted features estimates how predictable persistent errors are.

Failure-mode annotations are RULE-BASED HEURISTICS on the hand-crafted
features, not causal explanations.

Usage:  python -m src.analysis.error_taxonomy [--run-name challenge2015_ppg]
Outputs under runs/<run>/error_taxonomy/ (CSV/JSON + figures/*.png).
CPU only; no existing file is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (FS, detect_ppg_peaks, detect_r_peaks, extract_features,  # noqa: E402
                          ppg_feet)

VARIANTS = ["gbm_features", "resnet1d", "cross_attention", "ecg_only", "ppg_only"]
BEST, PROPOSED = "gbm_features", "cross_attention"
PERSIST, STABLE = 0.8, 0.2
DATA = ROOT / "data" / "processed" / "challenge2015_windows.npz"


# ---------------------------------------------------------------- loading
def load_data() -> dict:
    z = np.load(DATA, allow_pickle=True)
    return {k: z[k] for k in z.files}


def error_matrix(run: Path, variant: str, n: int, y: np.ndarray):
    """(n, n_seeds) wrong-at-0.5 matrix and (n, n_seeds) probabilities."""
    files = sorted((run / "preds").glob(f"{variant}_seed*_fold*.npz"))
    seeds = sorted({int(f.stem.split("_seed")[1].split("_fold")[0]) for f in files})
    P = np.full((n, len(seeds)), np.nan)
    for j, sd in enumerate(seeds):
        for f in sorted((run / "preds").glob(f"{variant}_seed{sd}_fold*.npz")):
            z = np.load(f)
            P[z["test_idx"], j] = z["test_prob"]
            assert np.array_equal(z["test_y"].astype(int), y[z["test_idx"]].astype(int))
    assert not np.isnan(P).any(), f"{variant}: some record lacks an OOF prediction"
    E = ((P >= 0.5).astype(int) != y[:, None].astype(int)).astype(float)
    return E, P


def compute_features(data: dict, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    rows = [extract_features(e, p) for e, p in zip(data["ecg"], data["ppg"])]
    df = pd.DataFrame(rows)
    df.insert(0, "record_id", data["record_id"])
    df = df.set_index("record_id")
    df.to_csv(cache)
    return df


# ---------------------------------------------------------------- subgroups
def tercile(x: np.ndarray, names=("low", "mid", "high")) -> np.ndarray:
    """Tercile by value with ties kept together (quality is ceiling-saturated, so
    bins can be unequal; empty bins are simply absent)."""
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    out = np.where(x < q1, names[0], np.where(x < q2, names[1], names[2]))
    return out.astype(object)


def build_groups(data: dict, F: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame(index=F.index)
    g["alarm_type"] = data["alarm_type"]
    for c in ["quality", "ecg_quality", "pulse_quality"]:
        g[f"{c}_tercile"] = tercile(data[c].astype(float))
    g["quality_degraded"] = np.where(data["quality"] < 0.9, "degraded(<0.9)", "clean(>=0.9)")
    g["detector_disagree"] = np.where(F["hr_abs_diff"].values >= 15, "HR_diff>=15bpm", "HR_diff<15bpm")
    g["pulse_coverage"] = np.where(F["ecg_beats_with_pulse_frac"].values < 0.5,
                                   "ECG_beats_with_pulse<0.5", "ECG_beats_with_pulse>=0.5")
    g["ppg_acf"] = np.where(F["ppg_acf_peak"].values < 0.3, "ppg_acf<0.3", "ppg_acf>=0.3")
    g["pulse_source"] = np.where(data["pulse_is_abp"], "ABP-as-pulse", "PPG")
    g["abp_also_recorded"] = np.where(data["has_abp"], "ABP_available", "no_ABP")
    hr = F["ecg_hr"].values
    n = F["ecg_n"].values
    g["hr_regime"] = np.select(
        [n < 3, hr < 50, hr < 100, hr < 140],
        ["ECG<3_beats", "HR<50", "HR50-100", "HR100-140"], "HR>=140")
    return g


# ---------------------------------------------------------------- heuristics
def make_annotator(F: pd.DataFrame, data: dict):
    thr = {
        "ecg_highf": np.quantile(F["ecg_highf_power"], 0.9),
        "ppg_highf": np.quantile(F["ppg_highf_power"], 0.9),
        "ecg_amp_cv": np.quantile(F["ecg_amp_cv"], 0.9),
    }

    def annotate(rid: str, i: int, label: int) -> str:
        r = F.loc[rid]
        tags = []
        if r["ecg_highf_power"] > thr["ecg_highf"] or r["ecg_tmpl_corr"] < 0.5 and r["ecg_n"] > 3:
            tags.append("ECG artefact: high HF power / poor beat-template match")
        if data["ecg_quality"][i] < 0.9:
            tags.append("low ECG quality index")
        if data["pulse_quality"][i] < 0.9:
            tags.append("low pulse quality index")
        if r["ppg_highf_power"] > thr["ppg_highf"] or r["ppg_acf_peak"] < 0.3:
            tags.append("PPG artefact / non-periodic pulse")
        if r["hr_abs_diff"] >= 15:
            tags.append(f"beat detectors disagree (ECG {r['ecg_hr']:.0f} vs PPG {r['ppg_hr']:.0f} bpm)")
        if r["ecg_beats_with_pulse_frac"] < 0.5 and r["ecg_rr_cv"] < 0.25 and r["ecg_n"] >= 4:
            tags.append("pulse absent/uncoupled but ECG regular")
        if not tags:
            if label == 1:
                tags.append("signals look clean and concordant (no heuristic flag)")
            else:
                tags.append("clean concordant rhythm, but model fires (no heuristic flag)")
        return "; ".join(tags[:3]) + "  [heuristic]"

    return annotate


# ---------------------------------------------------------------- stats
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def subgroup_tests(cat: pd.DataFrame, G: pd.DataFrame, y: np.ndarray, err_flag: dict, minn=5) -> pd.DataFrame:
    """For each variant x error type (FN among positives, FP among negatives) and
    each subgroup level: Fisher exact of 'persistent error' membership vs rest."""
    rows = []
    for v, flag in err_flag.items():
        for etype, pop in (("FN", y == 1), ("FP", y == 0)):
            part = []
            for col in G.columns:
                for lev in sorted(G[col].unique()):
                    inlev = (G[col].values == lev) & pop
                    out = (G[col].values != lev) & pop
                    if inlev.sum() < minn or out.sum() < minn:
                        continue
                    a, b = int((flag[inlev] == 1).sum()), int(inlev.sum())
                    c, d = int((flag[out] == 1).sum()), int(out.sum())
                    odds, p = fisher_exact([[a, b - a], [c, d - c]])
                    part.append({"variant": v, "error_type": etype, "factor": col, "level": lev,
                                 "n_level": b, "persist_err_level": a, "rate_level": a / b,
                                 "n_rest": d, "persist_err_rest": c, "rate_rest": c / d,
                                 "odds_ratio": odds, "p": p})
            if part:
                adj = holm(np.array([r["p"] for r in part]))
                for r, a_ in zip(part, adj):
                    r["p_holm"] = a_
                    r["family_size"] = len(part)
                rows += part
    return pd.DataFrame(rows)


def predictability(F: pd.DataFrame, y: np.ndarray, flag: np.ndarray, pop: np.ndarray, seed=0) -> dict:
    """Repeated stratified 5-fold CV AUC of a logistic regression on the
    hand-crafted features for 'persistent error' within one class."""
    X = np.nan_to_num(F.values[pop].astype(float))
    t = flag[pop].astype(int)
    if t.sum() < 8 or (1 - t).sum() < 8:
        return {"n": int(pop.sum()), "n_err": int(t.sum()), "auc": float("nan"), "auc_sd": float("nan")}
    cv = RepeatedStratifiedKFold(n_splits=min(5, int(t.sum())), n_repeats=10, random_state=seed)
    aucs, rep = [], {}
    for k, (tr, te) in enumerate(cv.split(X, t)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(X[tr], t[tr])
        s = m.predict_proba(X[te])[:, 1]
        if len(set(t[te])) > 1:
            aucs.append(roc_auc_score(t[te], s))
    return {"n": int(pop.sum()), "n_err": int(t.sum()), "auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs))}


# ---------------------------------------------------------------- plotting
def plot_case(data, F, rid_i, rid, label, atype, rates, probs, note, path, kind_title):
    ecg, ppg = data["ecg"][rid_i], data["ppg"][rid_i]
    t = np.arange(len(ecg)) / FS
    r = detect_r_peaks(ecg)
    pk = detect_ppg_peaks(ppg)
    ft = ppg_feet(ppg, pk)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    ax[0].plot(t, ecg, lw=0.7, color="#1f4e79")
    ax[0].plot(t[r], ecg[r], "v", color="#d62728", ms=6, label=f"R peaks (n={len(r)}, HR {F.loc[rid, 'ecg_hr']:.0f})")
    ax[0].set_ylabel("ECG (z)")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(t, ppg, lw=0.8, color="#2a7d4f")
    ax[1].plot(t[pk], ppg[pk], "^", color="#ff7f0e", ms=6, label=f"PPG peaks (n={len(pk)}, HR {F.loc[rid, 'ppg_hr']:.0f})")
    ax[1].plot(t[ft], ppg[ft], "o", mfc="none", color="#9467bd", ms=6, label="pulse feet")
    ax[1].set_ylabel("PPG (z)")
    ax[1].set_xlabel("time (s); alarm at 10 s")
    ax[1].legend(loc="upper left", fontsize=8)
    lab = "TRUE alarm" if label == 1 else "FALSE alarm"
    ptxt = "  ".join(f"{v.replace('cross_attention', 'x-attn').replace('gbm_features', 'gbm')}={probs[v]:.2f}(err {rates[v]:.1f})"
                     for v in VARIANTS)
    fig.suptitle(f"{kind_title}: record {rid} | {lab} | {atype.replace('_', ' ')}\n{ptxt}", fontsize=9.5, y=0.995)
    fig.text(0.01, 0.005, "Auto-annotation: " + note, fontsize=8.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pick_cases(cand: pd.DataFrame, k: int, used: set) -> list:
    """Highest error rate first, diversified over alarm types, preferring records
    not yet used by another figure group."""
    cand = cand.sort_values(["rate", "conf"], ascending=False)
    chosen, types = [], set()
    for pref_unused in (True, False):
        for pass_div in (True, False):
            for rid, row in cand.iterrows():
                if len(chosen) >= k:
                    break
                if rid in chosen or (pref_unused and rid in used):
                    continue
                if pass_div and row["atype"] in types:
                    continue
                chosen.append(rid)
                types.add(row["atype"])
    return chosen


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--cases-per-group", type=int, default=4)
    a = ap.parse_args()
    run = ROOT / "runs" / a.run_name
    out = run / "error_taxonomy"
    figd = out / "figures"
    figd.mkdir(parents=True, exist_ok=True)

    data = load_data()
    y = data["label"].astype(int)
    rid = data["record_id"]
    n = len(y)
    F = compute_features(data, out / "features.csv")
    G = build_groups(data, F)
    annotate = make_annotator(F, data)

    E, P, rate, mprob = {}, {}, {}, {}
    for v in VARIANTS:
        E[v], P[v] = error_matrix(run, v, n, y)
        rate[v] = E[v].mean(1)
        mprob[v] = P[v].mean(1)
    n_seeds = E[BEST].shape[1]

    # ---- per-record table
    rec = pd.DataFrame({"record_id": rid, "label": y, "alarm_type": data["alarm_type"]})
    for v in VARIANTS:
        rec[f"err_rate_{v}"] = rate[v]
        rec[f"mean_prob_{v}"] = mprob[v]
    rec = rec.join(G.reset_index(drop=True).drop(columns=['alarm_type']))
    rec["annotation"] = [annotate(rid[i], i, y[i]) for i in range(n)]

    # ---- categories
    cat_rows, flags = [], {}
    for v in VARIANTS:
        r = rate[v]
        cat = np.where(r <= STABLE, "stable_correct", np.where(r >= PERSIST,
                       np.where(y == 1, "persistent_FN", "persistent_FP"),
                       np.where(y == 1, "unstable_true_alarm", "unstable_false_alarm")))
        rec[f"cat_{v}"] = cat
        flags[v] = (r >= PERSIST).astype(int)
        for c in ["persistent_FN", "persistent_FP", "unstable_true_alarm", "unstable_false_alarm", "stable_correct"]:
            cnt = int((cat == c).sum())
            base = int((y == 1).sum()) if "FN" in c or "true" in c else int((y == 0).sum())
            if c == "stable_correct":
                base = n
            cat_rows.append({"variant": v, "category": c, "n": cnt,
                             "pct_of_class": 100 * cnt / base,
                             "pct_of_all": 100 * cnt / n})
    catdf = pd.DataFrame(cat_rows)
    catdf.to_csv(out / "category_counts.csv", index=False)
    rec.to_csv(out / "per_record_errors.csv", index=False)

    # ---- breakdowns (persistent FN among true alarms, persistent FP among false alarms)
    bd = []
    for v in VARIANTS:
        for col in G.columns:
            for lev in sorted(G[col].unique()):
                m = G[col].values == lev
                for etype, pop in (("FN", y == 1), ("FP", y == 0)):
                    mm = m & pop
                    if mm.sum() == 0:
                        continue
                    bd.append({"variant": v, "factor": col, "level": lev, "error_type": etype,
                               "n_pop": int(mm.sum()), "n_persistent": int(((rate[v] >= PERSIST) & mm).sum()),
                               "pct_persistent": 100 * ((rate[v] >= PERSIST) & mm).sum() / mm.sum(),
                               "mean_err_rate": float(rate[v][mm].mean())})
    pd.DataFrame(bd).to_csv(out / "breakdowns.csv", index=False)

    tests = subgroup_tests(rec, G.reset_index(drop=True), y, flags)
    tests.to_csv(out / "subgroup_tests.csv", index=False)

    pred = {}
    for v in VARIANTS:
        pred[v] = {"FN": predictability(F, y, flags[v], y == 1), "FP": predictability(F, y, flags[v], y == 0)}

    # ---- agreement between variants
    sets = {v: {"FN": set(rid[(flags[v] == 1) & (y == 1)]), "FP": set(rid[(flags[v] == 1) & (y == 0)])} for v in VARIANTS}
    maj = {v: rate[v] >= 0.5 for v in VARIANTS}
    all_wrong = np.all([maj[v] for v in VARIANTS], axis=0)
    others = [v for v in VARIANTS if v != PROPOSED]
    uniq = maj[PROPOSED] & ~np.any([maj[v] for v in others], axis=0)
    uniq_gbm = maj[BEST] & ~np.any([maj[v] for v in VARIANTS if v != BEST], axis=0)
    prop_pers_all = np.all([flags[v] == 1 for v in VARIANTS], axis=0)
    shared = {
        "all5_majority_wrong": int(all_wrong.sum()),
        "all5_majority_wrong_FN": int((all_wrong & (y == 1)).sum()),
        "all5_majority_wrong_FP": int((all_wrong & (y == 0)).sum()),
        "all5_persistent_wrong": int(prop_pers_all.sum()),
        "unique_to_proposed(majority; others<0.5)": int(uniq.sum()),
        "unique_to_proposed_FN": int((uniq & (y == 1)).sum()),
        "unique_to_proposed_FP": int((uniq & (y == 0)).sum()),
        "unique_to_gbm(majority; others<0.5)": int(uniq_gbm.sum()),
        "proposed_majority_wrong": int(maj[PROPOSED].sum()),
        "gbm_majority_wrong": int(maj[BEST].sum()),
        "proposed_wrong_and_gbm_right": int((maj[PROPOSED] & ~maj[BEST]).sum()),
        "gbm_wrong_and_proposed_right": int((maj[BEST] & ~maj[PROPOSED]).sum()),
        "ecgonly_ppgonly_both_majority_wrong": int((maj["ecg_only"] & maj["ppg_only"]).sum()),
    }
    # fusion oracle: proposed wrong while at least one unimodal right
    shared["proposed_wrong_but_a_unimodal_right"] = int((maj[PROPOSED] & (~maj["ecg_only"] | ~maj["ppg_only"])).sum())
    shared["jaccard_persistent_FN_gbm_vs_xattn"] = (
        len(sets[BEST]["FN"] & sets[PROPOSED]["FN"]) / max(1, len(sets[BEST]["FN"] | sets[PROPOSED]["FN"])))
    shared["jaccard_persistent_FP_gbm_vs_xattn"] = (
        len(sets[BEST]["FP"] & sets[PROPOSED]["FP"]) / max(1, len(sets[BEST]["FP"] | sets[PROPOSED]["FP"])))

    # ---- annotation tag frequency per category (proposed + best)
    tagrows = []
    for v in (BEST, PROPOSED):
        for c in ["persistent_FN", "persistent_FP", "stable_correct"]:
            m = rec[f"cat_{v}"].values == c
            ann = rec.loc[m, "annotation"]
            for tag in ["ECG artefact", "low ECG quality", "low pulse quality", "PPG artefact",
                        "beat detectors disagree", "pulse absent", "no heuristic flag"]:
                tagrows.append({"variant": v, "category": c, "tag": tag, "n": int(m.sum()),
                                "n_tag": int(ann.str.contains(tag).sum()),
                                "pct": 100 * float(ann.str.contains(tag).mean()) if m.sum() else float("nan")})
    pd.DataFrame(tagrows).to_csv(out / "annotation_tag_rates.csv", index=False)

    json.dump({"n_records": n, "n_seeds": n_seeds, "persist_threshold": PERSIST, "stable_threshold": STABLE,
               "predictability_cv_auc": pred, "shared_unique": shared,
               "n_persistent_sets": {v: {k: len(s) for k, s in sets[v].items()} for v in VARIANTS}},
              open(out / "summary.json", "w"), indent=2)

    # ================= figures
    def case_group(v, etype, used):
        lab = 1 if etype == "FN" else 0
        m = (rate[v] >= PERSIST) & (y == lab)
        idx = np.where(m)[0]
        cand = pd.DataFrame({"rate": rate[v][idx], "conf": np.abs(mprob[v][idx] - 0.5),
                             "atype": data["alarm_type"][idx]}, index=rid[idx])
        return pick_cases(cand, a.cases_per_group, used)

    used, nfig = set(), 0
    manifest = []
    for v, etype in [(BEST, "FN"), (PROPOSED, "FN"), (BEST, "FP"), (PROPOSED, "FP")]:
        chosen = case_group(v, etype, used)
        for k, r_ in enumerate(chosen):
            used.add(r_)
            i = int(np.where(rid == r_)[0][0])
            path = figd / f"case_{etype}_{v}_{k + 1}_{r_}.png"
            plot_case(data, F, i, r_, y[i], data["alarm_type"][i],
                      {w: rate[w][i] for w in VARIANTS}, {w: mprob[w][i] for w in VARIANTS},
                      rec.loc[i, "annotation"], path,
                      f"Persistent {etype} of {v}")
            manifest.append({"figure": path.name, "record": r_, "group": f"{v}_{etype}"})
            nfig += 1

    # shared vs unique figure (counts + scatter)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    types = sorted(set(data["alarm_type"]))
    masks = {"all 5 variants\n(majority wrong)": all_wrong, "unique to\ncross_attention": uniq,
             "unique to\ngbm_features": uniq_gbm}
    bottom = np.zeros(len(masks))
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for t_, c_ in zip(types, cols):
        vals = np.array([(m & (data["alarm_type"] == t_)).sum() for m in masks.values()])
        ax[0].bar(list(masks), vals, bottom=bottom, color=c_, label=t_.replace("_", " "))
        bottom += vals
    for j, m in enumerate(masks.values()):
        ax[0].text(j, bottom[j] + 0.5, f"{int(m.sum())} (FN {int((m & (y == 1)).sum())}, FP {int((m & (y == 0)).sum())})",
                   ha="center", fontsize=8)
    ax[0].set_ylabel("records"); ax[0].set_title("Shared vs unique errors (error rate >= 0.5)")
    ax[0].legend(fontsize=7)
    sc = ax[1]
    for lab, col, mk in [(1, "#d62728", "o"), (0, "#1f77b4", "s")]:
        m = y == lab
        sc.scatter(mprob[BEST][m], mprob[PROPOSED][m], c=col, marker=mk, s=14, alpha=0.6,
                   label="true alarm" if lab else "false alarm")
    sc.axhline(0.5, color="k", lw=0.5); sc.axvline(0.5, color="k", lw=0.5)
    sc.set_xlabel("mean OOF prob, gbm_features"); sc.set_ylabel("mean OOF prob, cross_attention")
    sc.set_title("Per-record mean probability"); sc.legend(fontsize=8)
    ax[2].scatter(rate[BEST], rate[PROPOSED], c=np.where(y == 1, "#d62728", "#1f77b4"), s=14, alpha=0.5)
    ax[2].set_xlabel("error rate gbm_features"); ax[2].set_ylabel("error rate cross_attention")
    ax[2].set_title("Consensus error rates (10 seeds)")
    fig.tight_layout(); fig.savefig(figd / "shared_vs_unique_errors.png", dpi=110); plt.close(fig)
    nfig += 1

    # gallery: examples of shared and proposed-unique errors
    ex = []
    for name, m in [("ALL variants wrong", all_wrong), ("UNIQUE to cross_attention", uniq)]:
        idx = np.where(m)[0]
        idx = idx[np.argsort(-rate[PROPOSED][idx])][:3]
        ex += [(name, i) for i in idx]
    if ex:
        fig, axes = plt.subplots(len(ex), 2, figsize=(14, 2.3 * len(ex)), squeeze=False)
        for r_, (name, i) in enumerate(ex):
            t = np.arange(2500) / FS
            rp = detect_r_peaks(data["ecg"][i]); pp = detect_ppg_peaks(data["ppg"][i])
            axes[r_, 0].plot(t, data["ecg"][i], lw=0.6, color="#1f4e79"); axes[r_, 0].plot(t[rp], data["ecg"][i][rp], "v", color="#d62728", ms=4)
            axes[r_, 1].plot(t, data["ppg"][i], lw=0.7, color="#2a7d4f"); axes[r_, 1].plot(t[pp], data["ppg"][i][pp], "^", color="#ff7f0e", ms=4)
            lab = "TRUE" if y[i] else "FALSE"
            axes[r_, 0].set_title(f"{name}: {rid[i]} {lab} {data['alarm_type'][i].replace('_', ' ')} | x-attn p={mprob[PROPOSED][i]:.2f}, gbm p={mprob[BEST][i]:.2f}", fontsize=8, loc="left")
            axes[r_, 1].set_title(rec.loc[i, "annotation"], fontsize=7.5, loc="left", style="italic")
        fig.tight_layout(); fig.savefig(figd / "shared_vs_unique_examples.png", dpi=105); plt.close(fig)
        nfig += 1

    # overview: category counts + consensus error-rate histograms
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    w = 0.16
    order = ["persistent_FN", "unstable_true_alarm", "persistent_FP", "unstable_false_alarm"]
    for j, v in enumerate(VARIANTS):
        vals = [int(catdf[(catdf.variant == v) & (catdf.category == c)].n.iloc[0]) for c in order]
        ax[0].bar(np.arange(4) + (j - 2) * w, vals, w, label=v)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax[0].set_ylabel("records"); ax[0].legend(fontsize=7); ax[0].set_title("Error taxonomy counts")
    for v in (BEST, PROPOSED):
        ax[1].hist(rate[v], bins=np.linspace(0, 1, 11), alpha=0.5, label=v)
    ax[1].set_yscale("log"); ax[1].set_xlabel("consensus error rate"); ax[1].set_ylabel("records (log)")
    ax[1].legend(); ax[1].set_title("Distribution of consensus error rate")
    fig.tight_layout(); fig.savefig(figd / "taxonomy_overview.png", dpi=110); plt.close(fig)
    nfig += 1

    json.dump(manifest, open(out / "figure_manifest.json", "w"), indent=1)
    print(f"figures written: {nfig}")
    print(catdf.pivot(index="category", columns="variant", values="n"))
    print(json.dumps(shared, indent=1))
    print(json.dumps(pred, indent=1))
    sig = tests[tests.p_holm < 0.05].sort_values("p_holm")
    print(sig[["variant", "error_type", "factor", "level", "n_level", "persist_err_level", "rate_level",
               "rate_rest", "odds_ratio", "p", "p_holm"]].to_string())


if __name__ == "__main__":
    main()
