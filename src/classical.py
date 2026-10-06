"""Six classical ML algorithms on engineered features v2, leak-free.

Variants: clf_logreg, clf_svm, clf_rf, clf_extratrees, clf_boost (xgboost),
clf_mlp, plus the ensembles clf_stack (logistic-regression stacker trained on
the inner-validation predictions) and clf_avg (mean of the 6 probabilities).

Protocol (docs/evaluation_protocol.md): splits come from src.train.iter_splits,
so they pair exactly with the neural runs. Inside every fold
  * imputation / percentile clipping / standardisation are fitted on the FIT set only;
  * every model gets `--n-iter` random hyper-parameter configs, each trained on
    the fit set and scored by AUC on the INNER VALIDATION set; the outer test
    fold is predicted once with the selected config;
  * the stacker is fitted on validation predictions (its own validation
    predictions are 5-fold cross-validated), thresholds use validation only.

Usage:
    python -m src.classical --dataset cinc --run-name classical_cinc --seeds 10
    python -m src.classical --dataset vtac --run-name classical_vtac --seeds 10
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.features_v2 import DATASETS, load_features_v2  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.train import iter_splits, run_dir  # noqa: E402

warnings.filterwarnings("ignore")

BASE = ["clf_logreg", "clf_svm", "clf_rf", "clf_extratrees", "clf_boost", "clf_mlp"]
ALL = BASE + ["clf_stack", "clf_avg"]
PROTOCOL = {"cinc": "cv", "cinc_recovered": "cv", "vtac": "official"}


# --------------------------------------------------------------- search spaces
def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_params(name: str, rng: np.random.RandomState) -> dict:
    if name == "clf_logreg":
        return dict(C=_logu(rng, 1e-4, 1e1), class_weight=rng.choice(["none", "balanced"]))
    if name == "clf_svm":
        return dict(C=_logu(rng, 1e-1, 1e3), gamma=_logu(rng, 1e-4, 5e-2), class_weight=rng.choice(["none", "balanced"]))
    if name in ("clf_rf", "clf_extratrees"):
        return dict(max_depth=int(rng.choice([0, 3, 5, 8, 12, 20])), min_samples_leaf=int(rng.choice([1, 2, 3, 5, 8, 12])),
                    max_features=rng.choice(["sqrt", "log2", "0.2", "0.4"]),
                    class_weight=rng.choice(["none", "balanced_subsample"]), n_estimators=300)
    if name == "clf_boost":
        return dict(n_estimators=int(rng.choice([100, 200, 300, 500])), learning_rate=_logu(rng, 0.01, 0.2),
                    max_depth=int(rng.randint(2, 7)), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample_bytree=float(rng.uniform(0.3, 1.0)), min_child_weight=float(rng.choice([1, 2, 5, 10])),
                    reg_lambda=_logu(rng, 0.1, 30), spw=rng.choice(["1", "balanced"]))
    if name == "clf_mlp":
        return dict(hidden=rng.choice(["64", "128", "64-32", "128-64", "256-128"]), alpha=_logu(rng, 1e-4, 3.0),
                    lr=_logu(rng, 3e-4, 1e-2))
    raise KeyError(name)


def build(name: str, p: dict, seed: int, y_fit: np.ndarray, probability: bool = True):
    if name == "clf_logreg":
        cw = None if p["class_weight"] == "none" else "balanced"
        return LogisticRegression(C=p["C"], class_weight=cw, max_iter=3000, random_state=seed)
    if name == "clf_svm":
        cw = None if p["class_weight"] == "none" else "balanced"
        return SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", class_weight=cw, probability=probability,
                   random_state=seed)
    if name in ("clf_rf", "clf_extratrees"):
        mf = p["max_features"]
        mf = float(mf) if mf.replace(".", "").isdigit() else str(mf)
        cw = None if p["class_weight"] == "none" else "balanced_subsample"
        cls = RandomForestClassifier if name == "clf_rf" else ExtraTreesClassifier
        return cls(n_estimators=p["n_estimators"], max_depth=None if p["max_depth"] == 0 else p["max_depth"],
                   min_samples_leaf=p["min_samples_leaf"], max_features=mf, class_weight=cw, n_jobs=1,
                   random_state=seed)
    if name == "clf_boost":
        spw = 1.0 if p["spw"] == "1" else float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                 max_depth=p["max_depth"], subsample=p["subsample"],
                                 colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
                                 reg_lambda=p["reg_lambda"], scale_pos_weight=spw, tree_method="hist", n_jobs=1,
                                 random_state=seed, verbosity=0)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                                                  max_depth=p["max_depth"], l2_regularization=p["reg_lambda"],
                                                  random_state=seed)
    if name == "clf_mlp":
        h = tuple(int(v) for v in str(p["hidden"]).split("-"))
        return MLPClassifier(hidden_layer_sizes=h, alpha=p["alpha"], learning_rate_init=p["lr"], max_iter=300,
                             early_stopping=True, n_iter_no_change=15, validation_fraction=0.15, random_state=seed)
    raise KeyError(name)


def score(model, X, search: bool = False) -> np.ndarray:
    if search and isinstance(model, SVC):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------- preprocessing
class Prep:
    """median-impute -> percentile clip -> standardise, all fitted on the fit set."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xi = self._imp(X)
        self.lo, self.hi = np.percentile(Xi, 0.5, axis=0), np.percentile(Xi, 99.5, axis=0)
        Xc = np.clip(Xi, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(0), Xc.std(0) + 1e-6
        return self

    def _imp(self, X):
        return np.where(np.isnan(X), self.med, X)

    def transform(self, X):
        return (np.clip(self._imp(X), self.lo, self.hi) - self.mu) / self.sd


def val_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5


def search_and_fit(name, Xf, yf, Xv, yv, Xt, seed, n_iter, rng_seed):
    rng = np.random.RandomState(rng_seed)
    best, best_auc, best_p = None, -1.0, None
    trace = []
    for _ in range(n_iter):
        p = sample_params(name, rng)
        try:
            m = build(name, p, seed, yf, probability=False).fit(Xf, yf)
            a = val_auc(yv, score(m, Xv, search=True))
        except Exception as e:  # pragma: no cover
            print(f"{name} cfg failed: {e}")
            continue
        trace.append(a)
        if a > best_auc:
            best_auc, best_p = a, p
    # final model with the selected config (SVM gets Platt probabilities from the fit set)
    if name == "clf_mlp":  # average 5 inits: the MLP is high-variance
        ms = [build(name, best_p, seed * 31 + k, yf).fit(Xf, yf) for k in range(5)]
        pv = np.mean([score(m, Xv) for m in ms], axis=0)
        pt = np.mean([score(m, Xt) for m in ms], axis=0)
        return pv, pt, best_p, best_auc, ms
    m = build(name, best_p, seed, yf, probability=True).fit(Xf, yf)
    return score(m, Xv), score(m, Xt), best_p, best_auc, [m]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack(val_probs, yv, test_probs, seed):
    """Logistic-regression stacker on logits of base-model validation predictions."""
    Zv = np.stack([_logit(v) for v in val_probs], 1)
    Zt = np.stack([_logit(t) for t in test_probs], 1)
    lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
    n_pos, n_neg = int(yv.sum()), int((yv == 0).sum())
    k = min(5, n_pos, n_neg)
    if k >= 2:
        cv = StratifiedKFold(k, shuffle=True, random_state=seed)
        pv = cross_val_predict(lr, Zv, yv.astype(int), cv=cv, method="predict_proba")[:, 1]
    else:
        pv = lr.fit(Zv, yv.astype(int)).predict_proba(Zv)[:, 1]
    lr.fit(Zv, yv.astype(int))
    return pv, lr.predict_proba(Zt)[:, 1], lr.coef_[0].tolist()


def val_threshold_metrics(yv, pv, yt, pt) -> dict:
    """Threshold maximising F1 on the VALIDATION set only, applied to test."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pr = pv >= t
        tp = ((pr == 1) & (yv == 1)).sum()
        f = 2 * tp / max(2 * tp + ((pr == 1) & (yv == 0)).sum() + ((pr == 0) & (yv == 1)).sum(), 1)
        if f > best_f + 1e-12 or (abs(f - best_f) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_f = t, f
    m = binary_metrics(yt, pt, best_t)
    return {"threshold": float(best_t), "f1": m["f1"], "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "accuracy": (m["tp"] + m["tn"]) / m["n"],
            "challenge_score": challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])}


def perm_importance(models, name, Xv, yv, names, seed, n_rep=5):
    """Permutation importance (AUC drop) on the INNER VALIDATION set."""
    rng = np.random.RandomState(seed)
    pred = (lambda X: np.mean([score(m, X) for m in models], axis=0))
    base = val_auc(yv, pred(Xv))
    out = np.zeros(Xv.shape[1])
    for j in range(Xv.shape[1]):
        d = 0.0
        for _ in range(n_rep):
            Xp = Xv.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d += base - val_auc(yv, pred(Xp))
        out[j] = d / n_rep
    return {"variant": name, "base_val_auc": base, "importance": dict(zip(names, out.tolist()))}


# ----------------------------------------------------------------- one fold
def run_fold(seed, fold_i, fit_idx, val_idx, test_idx, X, y, names, out_dir, n_iter, base_seed, variants,
             perm_folds):
    preds_dir = out_dir / "preds"
    if all((preds_dir / f"{v}_seed{seed}_fold{fold_i}.json").exists() for v in variants):
        return [json.loads((preds_dir / f"{v}_seed{seed}_fold{fold_i}.json").read_text()) for v in variants]
    t0 = time.time()
    prep = Prep().fit(X[fit_idx])
    Xf, Xv, Xt = (prep.transform(X[i]) for i in (fit_idx, val_idx, test_idx))
    yf, yv, yt = y[fit_idx].astype(int), y[val_idx].astype(int), y[test_idx].astype(int)
    res = {}
    for k, name in enumerate(BASE):
        if name not in variants and not any(v in variants for v in ("clf_stack", "clf_avg")):
            continue
        pv, pt, bp, ba, models = search_and_fit(name, Xf, yf, Xv, yv, Xt, base_seed + seed, n_iter,
                                                (base_seed + seed) * 1000 + fold_i * 10 + k)
        res[name] = dict(val=pv, test=pt, params=bp, val_auc_sel=ba)
        if fold_i < perm_folds and seed == 0 and name in ("clf_rf", "clf_boost", "clf_logreg"):
            (out_dir / f"importance_{name}_seed{seed}_fold{fold_i}.json").write_text(
                json.dumps(perm_importance(models, name, Xv, yv, names, seed)))
    if all(b in res for b in BASE):
        sv, st, coef = stack([res[b]["val"] for b in BASE], yv, [res[b]["test"] for b in BASE], base_seed + seed)
        res["clf_stack"] = dict(val=sv, test=st, params={"coef": dict(zip(BASE, coef))}, val_auc_sel=val_auc(yv, sv))
        res["clf_avg"] = dict(val=np.mean([res[b]["val"] for b in BASE], 0),
                              test=np.mean([res[b]["test"] for b in BASE], 0), params={}, val_auc_sel=np.nan)
    out = []
    for name in variants:
        r = res[name]
        m = binary_metrics(yt, r["test"])
        m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
        m["accuracy"] = (m["tp"] + m["tn"]) / m["n"]
        m["ops"] = operating_points(yv, r["val"], yt, r["test"])
        m["val_thr"] = val_threshold_metrics(yv, r["val"], yt, r["test"])
        m.update(params=int(X.shape[1]), best_params=r["params"], val_auc_selected=float(r["val_auc_sel"]),
                 val_auc_of_final=float(val_auc(yv, r["val"])), seed=seed, fold=fold_i, n_train=len(fit_idx),
                 n_test=len(test_idx), seconds=time.time() - t0)
        np.savez(preds_dir / f"{name}_seed{seed}_fold{fold_i}.npz", test_idx=test_idx, val_idx=val_idx,
                 val_prob=r["val"], val_y=y[val_idx], test_prob=r["test"], test_y=y[test_idx])
        (preds_dir / f"{name}_seed{seed}_fold{fold_i}.json").write_text(json.dumps(m, default=float))
        out.append(m)
    print(f"seed={seed} fold={fold_i} done in {time.time() - t0:.0f}s  " +
          " ".join(f"{v[4:]}={m['auc']:.3f}" for v, m in zip(variants, out)), flush=True)
    return out


def extended_summary(fm: list[dict]) -> dict:
    s = summarize_across_folds(fm)
    for k, fn in (("accuracy", lambda m: m["accuracy"]), ("challenge_score", lambda m: m["challenge_score"]),
                  ("f1_valthr", lambda m: m["val_thr"]["f1"]), ("accuracy_valthr", lambda m: m["val_thr"]["accuracy"]),
                  ("sensitivity_valthr", lambda m: m["val_thr"]["sensitivity"]),
                  ("specificity_valthr", lambda m: m["val_thr"]["specificity"])):
        v = np.array([fn(m) for m in fm], dtype=float)
        s[f"{k}_mean"], s[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--protocol", default=None, choices=["cv", "official"])
    ap.add_argument("--no-alarm-type", action="store_true", help="exclude the alarm-type one-hot features")
    ap.add_argument("--variants", default=",".join(ALL))
    ap.add_argument("--perm-folds", type=int, default=3)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    a = ap.parse_args()
    if a.run_name in ("challenge2015_ppg", "vtac_official", "challenge2015_recovered"):
        raise SystemExit("refusing to write into a neural run directory")

    cfg = yaml.safe_load(Path(a.config).read_text())
    protocol = a.protocol or PROTOCOL[a.dataset]
    data = load_processed(ROOT / "data" / "processed" / f"{DATASETS[a.dataset]}.npz")
    X, names, oh = load_features_v2(a.dataset, data)
    names = list(names)
    if not a.no_alarm_type and len(np.unique(data["alarm_type"])) > 1:
        X = np.concatenate([X, oh], 1)
        names += [f"alarm_{t}" for t in ["asystole", "brady", "tachy", "vfib", "vt"]]
    y = data["label"]
    variants = a.variants.split(",")
    out_dir = run_dir(cfg, a.run_name)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    print(f"{a.dataset}: X={X.shape} protocol={protocol} seeds={a.seeds} n_iter={a.n_iter} variants={variants}",
          flush=True)

    jobs = []
    for seed in range(a.seeds):
        for fold_i, fit, val, test in iter_splits(data, cfg, seed, protocol):
            jobs.append(delayed(run_fold)(seed, fold_i, fit, val, test, X, y, names, out_dir, a.n_iter, cfg["seed"],
                                          variants, a.perm_folds))
    rows = Parallel(n_jobs=a.n_jobs, verbose=0)(jobs)
    by_variant = {v: [] for v in variants}
    for fold_res in rows:
        for v, m in zip(variants, fold_res):
            by_variant[v].append(m)
    results = {}
    for v, fm in by_variant.items():
        s = extended_summary(fm)
        s.update(variant=v, width_mult=1.0, fold_metrics=fm)
        results[v] = s
        print(f"{v:15s} acc={s['accuracy_mean']:.3f}+/-{s['accuracy_std']:.3f} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} "
              f"AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f} sens={s['sensitivity_mean']:.3f} "
              f"spec={s['specificity_mean']:.3f} CS={s['challenge_score_mean']:.1f}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "feature_names.json").write_text(json.dumps(names))
    print("wrote", out_dir / "results.json")


if __name__ == "__main__":
    main()
