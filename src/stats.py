"""Significance testing that respects the dependence between CV folds/seeds.

Replaces the naive paired t-test on 15 (seed, fold) pairs, which treats
overlapping-training-set fold scores as independent and overstates evidence.

- corrected_resampled_ttest: Nadeau & Bengio (2003) variance correction.
- record_bootstrap: paired bootstrap that resamples *records* (the unit of
  independence) and reuses the same resample for both models.
- holm: step-down correction over the comparison family, which is fixed in
  configs/config.yaml (`stats.comparisons`) before any results are seen.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from src.metrics import challenge_score


def corrected_resampled_ttest(diffs: np.ndarray, n_train: float, n_test: float) -> dict:
    """Nadeau-Bengio corrected resampled t-test on per-(seed,fold) score
    differences. variance factor = 1/J + n_test/n_train instead of 1/J."""
    diffs = np.asarray(diffs, dtype=float)
    j = len(diffs)
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    if var <= 0 or j < 2:
        return {"mean_diff": float(mean), "t_stat": float("nan"), "p_value": float("nan"), "n_pairs": j}
    t = mean / np.sqrt((1.0 / j + n_test / n_train) * var)
    p = 2 * sp_stats.t.sf(abs(t), df=j - 1)
    return {"mean_diff": float(mean), "t_stat": float(t), "p_value": float(p), "n_pairs": j}


def _f1(c: np.ndarray) -> np.ndarray:
    tp, fp, _tn, fn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def _specificity(c: np.ndarray) -> np.ndarray:
    tn, fp = c[..., 2], c[..., 1]
    return np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1), np.nan)


def _sensitivity(c: np.ndarray) -> np.ndarray:
    tp, fn = c[..., 0], c[..., 3]
    return np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)


def _score(c: np.ndarray) -> np.ndarray:
    return challenge_score(c[..., 0], c[..., 2], c[..., 1], c[..., 3])


METRIC_FNS = {"f1": _f1, "specificity": _specificity, "sensitivity": _sensitivity, "challenge_score": _score}


def record_bootstrap(
    tallies_a: np.ndarray,
    tallies_b: np.ndarray,
    metric: str = "f1",
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Paired record-level bootstrap of metric(b) - metric(a).

    tallies_*: (n_seeds, n_records, 4) per-record [tp, fp, tn, fn] counts, with
    records in the same order for both models. Each bootstrap draw resamples
    records with replacement (same draw for a and b), computes the metric per
    seed on the resampled counts, and averages over seeds.
    """
    fn = METRIC_FNS[metric]
    n_seeds, n_rec, _ = tallies_a.shape
    rng = np.random.RandomState(seed)
    counts = rng.multinomial(n_rec, np.full(n_rec, 1.0 / n_rec), size=n_boot)  # (B, R)
    # (B,R) x (S,R,4) -> (B,S,4)
    agg_a = np.einsum("br,srk->bsk", counts, tallies_a)
    agg_b = np.einsum("br,srk->bsk", counts, tallies_b)
    diff = np.nanmean(fn(agg_b), axis=1) - np.nanmean(fn(agg_a), axis=1)
    obs = np.nanmean(fn(tallies_b.sum(axis=1)), axis=0) - np.nanmean(fn(tallies_a.sum(axis=1)), axis=0)
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return {
        "metric": metric, "obs_diff": float(obs), "ci95_low": float(lo), "ci95_high": float(hi),
        "p_value": float(min(p, 1.0)), "n_boot": n_boot,
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()
