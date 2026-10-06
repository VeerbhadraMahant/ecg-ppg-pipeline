"""Evaluation metrics: sensitivity/specificity/F1 plus helpers for the
performance-vs-signal-quality analysis called for in proposal.md."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    if len(np.unique(y_true)) < 2:
        tn = fp = fn = tp = 0
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def challenge_score(tp: float, tn: float, fp: float, fn: float) -> float:
    """Official PhysioNet/CinC 2015 score: (TP+TN) / (TP+TN+FP+5*FN) * 100.
    A missed true alarm costs five times a false alarm."""
    tp, tn, fp, fn = (np.asarray(v, dtype=float) for v in (tp, tn, fp, fn))
    denom = tp + tn + fp + 5 * fn
    score = np.where(denom > 0, 100.0 * (tp + tn) / np.where(denom > 0, denom, 1.0), np.nan)
    return float(score) if score.ndim == 0 else score


def threshold_for_sensitivity(y_val: np.ndarray, p_val: np.ndarray, target: float) -> float:
    """Largest threshold whose sensitivity on the (inner validation) set is
    >= target. Chosen on validation data only, then applied unchanged to the
    held-out test fold, so the reported operating point is not tuned on test."""
    pos = np.sort(p_val[y_val.astype(int) == 1])
    if len(pos) == 0:
        return 0.5
    # need at least ceil(target * n_pos) positives at or above the threshold
    k = int(np.ceil(target * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[len(pos) - k])


SENS_TARGETS = (0.95, 0.99, 1.0)


def operating_points(y_val, p_val, y_test, p_test) -> dict:
    """Confusion counts on the test fold at threshold 0.5 and at the
    thresholds that reach each target sensitivity on the validation set."""
    out = {}
    thr = {"t50": 0.5}
    for s in SENS_TARGETS:
        thr[f"s{int(round(s * 100))}"] = threshold_for_sensitivity(y_val, p_val, s)
    y_test = y_test.astype(int)
    for name, t in thr.items():
        pred = (p_test >= t).astype(int)
        out[name] = {
            "threshold": t,
            "tp": int(((pred == 1) & (y_test == 1)).sum()),
            "fp": int(((pred == 1) & (y_test == 0)).sum()),
            "tn": int(((pred == 0) & (y_test == 0)).sum()),
            "fn": int(((pred == 0) & (y_test == 1)).sum()),
        }
    return out


def suppression_rate(tn: float, fp: float) -> float:
    """Fraction of false alarms suppressed (== specificity at that operating point)."""
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def summarize_across_folds(fold_metrics: list[dict]) -> dict:
    """Mean +/- std for each metric across folds/seeds, as called for in
    proposal.md (error bars, not a single run)."""
    keys = ["sensitivity", "specificity", "f1", "auc"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics if not np.isnan(m[k])])
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{k}_std"] = float(vals.std()) if len(vals) else float("nan")
    return out
