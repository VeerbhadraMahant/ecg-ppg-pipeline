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
