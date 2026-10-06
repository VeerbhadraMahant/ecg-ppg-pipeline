"""AF screening on MIMIC PERform AF (updates.md 3.6 and 7.3).

MIMIC PERform AF is an AF-vs-non-AF task, not alarm verification, so it is its
own task with its own SUBJECT-WISE cross-validation and baselines:

  scratch  : each ladder variant trained from random init on AF
  pretrain : cross_attention initialised from the alarm-verification
             checkpoint (same architecture), then fine-tuned on AF
  gbm_*    : hand-crafted RR / pulse-interval irregularity features + gradient
             boosting under the identical folds

Protocol: 5 outer folds, stratified by AF label, all windows of a subject on
one side; inner validation = ~20% of the training SUBJECTS (stratified) for
early stopping. Every window is a test window exactly once per seed, so
out-of-fold (OOF) window probabilities are stored per seed and used for a
subject-level cluster bootstrap (n = 35 subjects: LOW POWER).

Usage:
    python -m src.af_task --seeds 3
    python -m src.af_task --variants gbm_ecg_rr,gbm_ppg_pp --seeds 5
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset  # noqa: E402
from src.external_validation import build_external_windows  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.train import get_width_mult, set_seed, train_one_fold  # noqa: E402

OUT_DIR = ROOT / "runs" / "af_task"
CACHE = ROOT / "data" / "processed" / "mimic_af_windows.npz"


# ----------------------------------------------------------------- data ----
def original_subject_ids(mimic_dir: Path) -> dict[str, str]:
    """csv stem -> original MIMIC subject id (from the *_fix.txt sidecars), so
    the split groups by true patient even if one patient had two exports."""
    out = {}
    for fx in mimic_dir.rglob("*_fix.txt"):
        stem = fx.name.replace("_fix.txt", "_data")
        for line in fx.read_text().splitlines():
            if line.startswith("Original Subject ID:"):
                out[stem] = line.split(":", 1)[1].strip()
    return out


def load_af_windows(cfg: dict, max_windows: int) -> dict:
    """Windows (ECG/PPG 250 Hz, 10 s, filtered, z-normalised) + group ids."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if int(d["max_windows"]) == max_windows:
            return {k: d[k] for k in ("ecg", "ppg", "label", "subject_id", "group")}
    data = build_external_windows(cfg, max_windows)
    mimic_dir = ROOT / cfg["paths"]["external_dir"]
    omap = original_subject_ids(mimic_dir)
    data["group"] = np.array([omap.get(s, s) for s in data["subject_id"]])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, max_windows=max_windows, **data)
    return data


# ---------------------------------------------------------------- splits ---
def _group_labels(group, y):
    ids = np.unique(group)
    lab = np.array([int(y[group == g][0]) for g in ids])
    return ids, lab


def stratified_group_folds(group, y, n_folds: int, seed: int):
    """Yield (train_idx, test_idx); groups (subjects) dealt round-robin within
    each class after shuffling, so every fold has both classes."""
    ids, lab = _group_labels(group, y)
    rng = np.random.RandomState(seed)
    fold_of, i = {}, 0
    for c in (1, 0):
        gs = ids[lab == c].copy()
        rng.shuffle(gs)
        for g in gs:
            fold_of[g] = i % n_folds
            i += 1
    fa = np.array([fold_of[g] for g in group])
    for f in range(n_folds):
        yield np.where(fa != f)[0], np.where(fa == f)[0]


def stratified_group_holdout(group, y, idx, frac: float, seed: int):
    """Split `idx` into (fit, val) by subject, >=1 subject per class in val."""
    ids, lab = _group_labels(group[idx], y[idx])
    rng = np.random.RandomState(seed)
    val = set()
    for c in (0, 1):
        gs = ids[lab == c].copy()
        rng.shuffle(gs)
        val |= set(gs[: max(1, int(round(frac * len(gs))))])
    in_val = np.array([g in val for g in group[idx]])
    return idx[~in_val], idx[in_val]


def splits(group, y, n_folds, seed):
    for fold, (tr, te) in enumerate(stratified_group_folds(group, y, n_folds, seed)):
        fit, val = stratified_group_holdout(group, y, tr, 0.2, seed + fold)
        yield fold, fit, val, te


# --------------------------------------------------------------- metrics ---
def subject_probs(prob, y, group):
    ids = np.unique(group)
    sp = np.array([prob[group == g].mean() for g in ids])
    sy = np.array([int(y[group == g][0]) for g in ids])
    return ids, sp, sy


def fold_row(prob, y, group, seed, fold):
    w = binary_metrics(y, prob)
    _, sp, sy = subject_probs(prob, y, group)
    s = binary_metrics(sy, sp)
    return {"seed": seed, "fold": fold, "n_subj": len(sy),
            "win_f1": w["f1"], "win_auc": w["auc"], "win_sens": w["sensitivity"], "win_spec": w["specificity"],
            "subj_f1": s["f1"], "subj_auc": s["auc"], "subj_sens": s["sensitivity"], "subj_spec": s["specificity"]}


METRICS = ["win_f1", "win_auc", "win_sens", "win_spec", "subj_f1", "subj_auc", "subj_sens", "subj_spec"]


def pooled_metrics(oof_row, y, group):
    """Metrics for one seed's pooled OOF predictions (all 35 subjects)."""
    w = binary_metrics(y, oof_row)
    _, sp, sy = subject_probs(oof_row, y, group)
    s = binary_metrics(sy, sp)
    return {"win_f1": w["f1"], "win_auc": w["auc"], "win_sens": w["sensitivity"], "win_spec": w["specificity"],
            "subj_f1": s["f1"], "subj_auc": s["auc"], "subj_sens": s["sensitivity"], "subj_spec": s["specificity"]}


def bootstrap_ci(oof, y, group, n_boot=1000, seed=0):
    """Cluster (subject-level) bootstrap, resampling subjects within class.
    oof: (n_seeds, N) window probs. Per replicate the metric is averaged over
    seeds (matching the headline mean over seeds). Returns {metric: [lo, hi]}."""
    rng = np.random.RandomState(seed)
    ids, lab = _group_labels(group, y)
    members = {g: np.where(group == g)[0] for g in ids}
    reps = {m: [] for m in METRICS}
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(ids[lab == c], (lab == c).sum(), replace=True) for c in (0, 1)])
        ix = np.concatenate([members[g] for g in pick])
        gg = np.concatenate([[f"{g}#{k}"] * len(members[g]) for k, g in enumerate(pick)])
        vals = [pooled_metrics(o[ix], y[ix], gg) for o in oof]
        for m in METRICS:
            reps[m].append(np.nanmean([v[m] for v in vals]))
    return {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for m, v in reps.items()}


def summarize(rows, oof, y, group, n_boot):
    fold = {m: [float(np.nanmean([r[m] for r in rows])), float(np.nanstd([r[m] for r in rows]))] for m in METRICS}
    per_seed = [pooled_metrics(o, y, group) for o in oof]
    pooled = {m: [float(np.nanmean([p[m] for p in per_seed])), float(np.nanstd([p[m] for p in per_seed]))] for m in METRICS}
    return {"fold_mean_std": fold, "pooled_oof_mean_std_over_seeds": pooled,
            "pooled_oof_bootstrap95": bootstrap_ci(oof, y, group, n_boot)}


# ------------------------------------------------------------- GBM feats ---
def sample_entropy(x, m=2, r_frac=0.2):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 3:
        return np.nan
    r = r_frac * x.std()
    if r <= 0:
        return 0.0

    def count(mm):
        t = np.array([x[i:i + mm] for i in range(n - m)])
        d = np.max(np.abs(t[:, None, :] - t[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(t)) / 2.0

    b, a = count(m), count(m + 1)
    if a <= 0 or b <= 0:
        return np.nan
    return float(-np.log(a / b))


def rr_features(peaks, fs=250):
    names = ["n", "mean_rr", "cv", "rmssd", "rmssd_norm", "pnn50", "pnn20", "sampen", "mad_diff_norm", "range_norm"]
    if len(peaks) < 4:
        return dict.fromkeys(names, np.nan)
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    if len(rr) < 3:
        return dict.fromkeys(names, np.nan)
    d = np.diff(rr)
    mean = rr.mean()
    return {"n": float(len(rr)), "mean_rr": mean, "cv": rr.std() / mean,
            "rmssd": float(np.sqrt(np.mean(d ** 2))), "rmssd_norm": float(np.sqrt(np.mean(d ** 2)) / mean),
            "pnn50": float(np.mean(np.abs(d) > 0.05)), "pnn20": float(np.mean(np.abs(d) > 0.02)),
            "sampen": sample_entropy(rr), "mad_diff_norm": float(np.median(np.abs(d)) / mean),
            "range_norm": float((rr.max() - rr.min()) / mean)}


def gbm_feature_matrix(sig, kind):
    from src.features import detect_ppg_peaks, detect_r_peaks

    det = detect_r_peaks if kind == "ecg" else detect_ppg_peaks
    rows = [rr_features(det(w)) for w in sig]
    keys = list(rows[0])
    return np.array([[r[k] for k in keys] for r in rows], dtype=np.float64)


def run_gbm(name, data, seeds, n_folds):
    from sklearn.ensemble import HistGradientBoostingClassifier

    kind = "ecg" if name == "gbm_ecg_rr" else "ppg"
    X = gbm_feature_matrix(data["ecg" if kind == "ecg" else "ppg"], kind)
    y, group = data["label"], data["group"]
    oof = np.zeros((seeds, len(y)))
    rows = []
    for seed in range(seeds):
        for fold, fit, val, te in splits(group, y, n_folds, 42 + seed):
            tr = np.concatenate([fit, val])  # GBM uses its own internal early stopping, no subject-level val needed
            clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                 l2_regularization=1.0, random_state=seed)
            clf.fit(X[tr], y[tr])
            oof[seed, te] = clf.predict_proba(X[te])[:, 1]
            rows.append(fold_row(oof[seed, te], y[te], group[te], seed, fold))
    return rows, oof


# ----------------------------------------------------------- NN variants ---
def run_nn(name, cfg, data, seeds, n_folds, device):
    pretrain = name.endswith("+pretrain")
    variant = name.replace("+pretrain", "")
    wm = get_width_mult(variant, ROOT)
    ecg, ppg, y, group = data["ecg"], data["ppg"], data["label"], data["group"]
    ckpt_dir = ROOT / cfg["paths"]["runs_dir"] / "challenge2015_ppg" / "checkpoints"
    oof = np.zeros((seeds, len(y)))
    rows = []
    for seed in range(seeds):
        set_seed(cfg["seed"] + seed)
        for fold, fit, val, te in splits(group, y, n_folds, cfg["seed"] + seed):
            c = copy.deepcopy(cfg)
            if pretrain:
                ck = ckpt_dir / f"{variant}_seed0_fold{fold}.pt"
                if not ck.exists():
                    raise FileNotFoundError(f"{ck}: train the alarm model first")
                c["train"]["init_from"] = str(ck)
            mk = lambda ix: AlarmWindowDataset(ecg[ix], ppg[ix], y[ix])  # noqa: E731
            _, _, preds = train_one_fold(c, variant, wm, mk(fit), mk(val), mk(te), device, seed)
            oof[seed, te] = preds["test_prob"]
            rows.append(fold_row(preds["test_prob"], y[te], group[te], seed, fold))
    return rows, oof


def fmt(summ):
    p, f = summ["pooled_oof_mean_std_over_seeds"], summ["fold_mean_std"]
    ci = summ["pooled_oof_bootstrap95"]
    return (f"win F1={p['win_f1'][0]:.3f} AUC={p['win_auc'][0]:.3f} | subj F1={p['subj_f1'][0]:.3f} "
            f"AUC={p['subj_auc'][0]:.3f} [{ci['subj_auc'][0]:.2f},{ci['subj_auc'][1]:.2f}] "
            f"sens={p['subj_sens'][0]:.2f} spec={p['subj_spec'][0]:.2f} | fold-mean subj AUC={f['subj_auc'][0]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--max-windows", type=int, default=60, help="windows per subject (evenly spaced)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--variants", default="ecg_only,ppg_only,concat,cross_attention,cross_attention+pretrain,gbm_ecg_rr,gbm_ppg_pp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["train"]["epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_af_windows(cfg, args.max_windows)
    y, group = data["label"], data["group"]
    n_folds = cfg["split"]["n_folds"]
    print(f"{len(y)} windows, {len(set(group))} subjects, {y.mean():.1%} AF windows, device={device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = OUT_DIR / "af_task_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in args.variants.split(","):
        safe = name.replace("+", "_")
        if name in results and (OUT_DIR / f"oof_{safe}.npz").exists() and not args.force:
            print(f"{name}: cached"); continue
        t0 = time.time()
        rows, oof = run_gbm(name, data, args.seeds, n_folds) if name.startswith("gbm_") else \
            run_nn(name, cfg, data, args.seeds, n_folds, device)
        np.savez(OUT_DIR / f"oof_{safe}.npz", oof=oof, label=y, group=group)
        results[name] = {**summarize(rows, oof, y, group, args.n_boot), "folds": rows,
                         "seeds": args.seeds, "epochs": args.epochs, "max_windows": args.max_windows,
                         "seconds": time.time() - t0}
        res_path.write_text(json.dumps(results, indent=2, default=float))
        print(f"{name:26s} {fmt(results[name])} ({time.time() - t0:.0f}s)", flush=True)
    print(f"written {res_path}")


if __name__ == "__main__":
    main()
