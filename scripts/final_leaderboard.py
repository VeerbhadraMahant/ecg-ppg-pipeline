"""Final leak-free leaderboard across runs (updates.md section 10, results).

Collects fold-level predictions from several run directories, restricts every
model to the seeds shared by all of them, and reports accuracy / F1 / AUC /
sensitivity / specificity / Challenge score at the 0.5 threshold, with a paired
record-level bootstrap of the best model's advantage over each other model.

    python scripts/final_leaderboard.py
Writes docs/LEADERBOARD.md and runs/final_leaderboard.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import challenge_score  # noqa: E402
from src.stats import record_bootstrap  # noqa: E402

# dataset -> (n_records, [(run, variant, display name, group)])
SPECS = {
    "Challenge 2015 (PPG cohort, 5-fold CV)": ("cinc", 592, [
        ("final_cinc", "ens_logit", "Ensemble (logit mean, 10 models)", "ensemble"),
        ("final_cinc", "ens_stack", "Ensemble (LR stacker)", "ensemble"),
        ("classical_cinc", "clf_boost", "XGBoost on features v2", "classical"),
        ("classical_cinc", "clf_extratrees", "Extra Trees on features v2", "classical"),
        ("classical_cinc", "clf_svm", "SVM (RBF) on features v2", "classical"),
        ("classical_cinc", "clf_mlp", "MLP on features v2", "classical"),
        ("classical_cinc", "clf_rf", "Random Forest on features v2", "classical"),
        ("classical_cinc", "clf_logreg", "Logistic regression on features v2", "classical"),
        ("challenge2015_ppg", "gbm_features", "HistGB on features v1", "classical"),
        ("challenge2015_ppg", "resnet1d", "ResNet1D (deep)", "deep"),
        ("challenge2015_ppg", "tcn", "TCN (deep)", "deep"),
        ("challenge2015_ppg", "inceptiontime", "InceptionTime (deep)", "deep"),
        ("challenge2015_ppg", "concat", "CNN concat fusion (deep)", "deep"),
        ("challenge2015_ppg", "cross_attention", "CNN cross-attention fusion (proposed)", "deep"),
    ]),
    "Challenge 2015 (PPG cohort, 5 seeds, incl. hybrid deep recipe)": ("cinc", 592, [
        ("final_cinc", "ens_stack_dp", "Ensemble incl. hybrid deep (LR stacker)", "ensemble"),
        ("final_cinc", "ens_mean_dp", "Ensemble incl. hybrid deep (mean)", "ensemble"),
        ("deep_plus_cinc", "deepplus", "Hybrid deep: attn CNN-RNN + features, aug/EMA/TTA/ens", "deep"),
        ("classical_cinc", "clf_boost", "XGBoost on features v2", "classical"),
        ("classical_cinc", "clf_extratrees", "Extra Trees on features v2", "classical"),
        ("classical_cinc", "clf_svm", "SVM (RBF) on features v2", "classical"),
        ("classical_cinc", "clf_mlp", "MLP on features v2", "classical"),
        ("classical_cinc", "clf_rf", "Random Forest on features v2", "classical"),
        ("classical_cinc", "clf_logreg", "Logistic regression on features v2", "classical"),
        ("challenge2015_ppg", "resnet1d", "ResNet1D (deep)", "deep"),
        ("challenge2015_ppg", "cross_attention", "CNN cross-attention fusion (proposed)", "deep"),
    ]),
    "VTaC (official split, 5 seeds)": ("vtac", 4542, [
        ("final_vtac", "ens_stack_dp", "Ensemble incl. hybrid deep (LR stacker)", "ensemble"),
        ("final_vtac", "ens_logit_dp", "Ensemble incl. hybrid deep (logit mean)", "ensemble"),
        ("deep_plus_vtac", "deepplus", "Hybrid deep: attn CNN-RNN + features, aug/EMA/TTA/ens", "deep"),
        ("final_vtac", "ens_logit", "Ensemble (logit mean, 10 models)", "ensemble"),
        ("final_vtac", "ens_stack", "Ensemble (LR stacker)", "ensemble"),
        ("classical_vtac", "clf_boost", "XGBoost on features v2", "classical"),
        ("classical_vtac", "clf_extratrees", "Extra Trees on features v2", "classical"),
        ("classical_vtac", "clf_svm", "SVM (RBF) on features v2", "classical"),
        ("classical_vtac", "clf_mlp", "MLP on features v2", "classical"),
        ("classical_vtac", "clf_rf", "Random Forest on features v2", "classical"),
        ("classical_vtac", "clf_logreg", "Logistic regression on features v2", "classical"),
        ("vtac_official", "gbm_features", "HistGB on features v1", "classical"),
        ("vtac_official", "resnet1d", "ResNet1D (deep)", "deep"),
        ("vtac_official", "tcn", "TCN (deep)", "deep"),
        ("vtac_official", "bi_cross_attention", "CNN bidirectional cross-attention (deep)", "deep"),
        ("vtac_official", "concat", "CNN concat fusion (deep)", "deep"),
        ("vtac_official", "cross_attention", "CNN cross-attention fusion (proposed)", "deep"),
    ]),
}


def load(run: str, variant: str, max_seed: int):
    files = sorted((ROOT / "runs" / run / "preds").glob(f"{variant}_seed*_fold*.npz"))
    out = {}
    for f in files:
        sd = int(f.stem.split("_seed")[1].split("_fold")[0])
        if sd <= max_seed:
            out[(sd, int(f.stem.split("_fold")[1]))] = np.load(f)
    return out


def metrics(zs):
    accs, f1s, aucs, sens, spec, chs = [], [], [], [], [], []
    from sklearn.metrics import roc_auc_score

    for z in zs:
        y, p = z["test_y"].astype(int), z["test_prob"]
        pred = (p >= 0.5).astype(int)
        tp, fp = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum()
        tn, fn = ((pred == 0) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
        accs.append((tp + tn) / len(y))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        aucs.append(roc_auc_score(y, p) if len(set(y)) > 1 else np.nan)
        sens.append(tp / max(tp + fn, 1))
        spec.append(tn / max(tn + fp, 1))
        chs.append(challenge_score(tp, tn, fp, fn))
    m = lambda v: float(np.nanmean(v))  # noqa: E731
    return dict(accuracy=m(accs), f1=m(f1s), auc=m(aucs), sens=m(sens), spec=m(spec), challenge=m(chs))


def tallies(data: dict, n: int, seeds):
    t = np.zeros((len(seeds), n, 4))
    for (sd, fo), z in data.items():
        k = seeds.index(sd)
        pred, y = z["test_prob"] >= 0.5, z["test_y"] == 1
        idx = z["test_idx"]
        t[k, idx, 0], t[k, idx, 1] = pred & y, pred & ~y
        t[k, idx, 2], t[k, idx, 3] = ~pred & ~y, ~pred & y
    return t


def main():
    rows, md = [], ["# Final leak-free leaderboard", "",
                    "All thresholds fixed at 0.5; every model sees identical splits and the seeds shared by all "
                    "models in a table. Hyper-parameters / combiners were fitted on inner validation sets only. "
                    "CI = paired record-level bootstrap of (best - model).", ""]
    for title, (key, n, specs) in SPECS.items():
        loaded = {}
        for run, v, name, grp in specs:
            d = load(run, v, 99)
            if d:
                loaded[(run, v)] = d
        if not loaded:
            continue
        seed_sets = [{s for s, _ in d} for d in loaded.values()]
        common = sorted(set.intersection(*seed_sets))
        print(f"{title}: seeds in common = {common}")
        res = []
        for run, v, name, grp in specs:
            if (run, v) not in loaded:
                continue
            d = {k: z for k, z in loaded[(run, v)].items() if k[0] in common}
            m = metrics(d.values())
            m.update(model=name, group=grp, dataset=title, n_seeds=len(common))
            res.append((m, d))
        best = max(res, key=lambda r: r[0]["accuracy"])
        tb = tallies(best[1], n, common)
        for m, d in res:
            if m is best[0]:
                m["bootstrap_f1_best_minus_model"] = "-"
            else:
                tm = tallies(d, n, common)
                used = (tb.sum((0, 2)) + tm.sum((0, 2))) > 0
                b = record_bootstrap(tm[:, used], tb[:, used], "f1", 2000)
                m["bootstrap_f1_best_minus_model"] = f"{b['obs_diff']:+.3f} [{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}]"
            rows.append(m)
        df = pd.DataFrame([m for m, _ in res]).sort_values("accuracy", ascending=False)
        md += [f"## {title}", f"*seeds: {common}; best by accuracy: **{best[0]['model']}***", "",
               "| Model | Group | Accuracy | F1 | AUC | Sens | Spec | Challenge | best−model ΔF1 [95% CI] |",
               "|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            md.append(f"| {r.model} | {r.group} | {r.accuracy:.3f} | {r.f1:.3f} | {r.auc:.3f} | {r.sens:.3f} | "
                      f"{r.spec:.3f} | {r.challenge:.1f} | {r.bootstrap_f1_best_minus_model} |")
        md.append("")
    (ROOT / "docs" / "LEADERBOARD.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(rows).to_csv(ROOT / "runs" / "final_leaderboard.csv", index=False)
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
