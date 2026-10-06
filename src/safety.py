"""Safety layer and clinical utility analysis (updates.md section 7.4 / 7.5).

Works from the saved per-fold predictions (runs/<run>/preds/*.npz), where each
fold has validation probabilities (used to FIT thresholds / calibration) and
test probabilities (used only to REPORT):

1. Calibration: temperature scaling fit on validation; ECE / Brier on test.
2. Risk-controlled suppression threshold: the largest threshold t such that
   the exact one-sided Clopper-Pearson UPPER bound of the true-alarm miss rate
   on the validation positives is <= alpha_miss (default 5%) at confidence
   1 - delta. If the validation set has too few true alarms to certify the
   target, no threshold is certified and the layer abstains from suppressing
   anything (reported honestly rather than faked).
3. Deferral band: p < t_low  -> auto-suppress (certified), p >= t_high ->
   high priority, in between -> human review / normal priority.
4. Clinical utility: false alarms removed per 100 alarms and missed true
   alarms per 1000 true alarms (rates, no assumed alarm volume). Optional
   --alarms-per-bed-day converts to per-bed-day numbers with a USER-SUPPLIED
   rate, explicitly labelled as an assumption.
5. Decision-curve analysis: net benefit vs the alarm-on-everything policy.

Usage:
    python -m src.safety --run-name challenge2015_ppg --variant cross_attention
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import run_dir  # noqa: E402


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(p_val: np.ndarray, y_val: np.ndarray) -> float:
    """Single temperature T minimising validation NLL of sigmoid(logit(p)/T)."""
    z = _logit(p_val)

    def nll(logT):
        q = 1 / (1 + np.exp(-z / np.exp(logT)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))

    res = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(p) / T))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def cp_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - delta, k + 1, n - k))


def risk_controlled_threshold(p_val: np.ndarray, y_val: np.ndarray, alpha_miss: float, delta: float):
    """Largest t whose miss-rate upper bound on validation positives is <= alpha_miss.
    Returns (threshold or None, miss_upper_bound at that threshold)."""
    pos = np.sort(p_val[y_val == 1])
    n = len(pos)
    if n == 0:
        return None, 1.0
    # candidate thresholds: just above each positive score (misses = positives below t)
    best = None
    for k in range(0, n + 1):  # k positives allowed below the threshold
        ub = cp_upper(k, n, delta)
        if ub <= alpha_miss:
            t = pos[k] if k < n else 1.0 + 1e-9  # threshold at the (k+1)-th smallest positive misses exactly k
            best = (float(t), ub, k)
        else:
            break
    if best is None:
        return None, cp_upper(0, n, delta)
    return best[0], best[1]


def net_benefit(p: np.ndarray, y: np.ndarray, pts: np.ndarray) -> dict:
    n = len(y)
    prev = y.mean()
    nb_model, nb_all = [], []
    for pt in pts:
        pred = p >= pt
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        w = pt / (1 - pt)
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prev - (1 - prev) * w)
    return {"model": np.array(nb_model), "all": np.array(nb_all)}


def load_folds(run: Path, variant: str):
    for f in sorted((run / "preds").glob(f"{variant}_seed*_fold*.npz")):
        z = np.load(f)
        yield f.stem, z["val_prob"], z["val_y"].astype(int), z["test_prob"], z["test_y"].astype(int)


def analyse(run: Path, variant: str, alpha_miss: float, delta: float, alarms_per_bed_day: float | None):
    rows, cal_rows = [], []
    all_p, all_y = [], []
    agg = {"suppressed_fa": 0, "n_fa": 0, "missed": 0, "n_true": 0, "auto_suppressed": 0, "deferred": 0,
           "high": 0, "n": 0, "high_true": 0, "defer_true": 0, "supp_true": 0, "no_cert": 0, "folds": 0}
    for name, pv, yv, pt, yt in load_folds(run, variant):
        T = fit_temperature(pv, yv)
        pv_c, pt_c = apply_temperature(pv, T), apply_temperature(pt, T)
        cal_rows.append({"fold": name, "T": T, "ece_raw": ece(pt, yt), "ece_cal": ece(pt_c, yt),
                         "brier_raw": float(np.mean((pt - yt) ** 2)), "brier_cal": float(np.mean((pt_c - yt) ** 2))})
        all_p.append(pt_c); all_y.append(yt)

        t_low, ub = risk_controlled_threshold(pv_c, yv, alpha_miss, delta)
        agg["folds"] += 1
        agg["n"] += len(yt); agg["n_fa"] += int((yt == 0).sum()); agg["n_true"] += int((yt == 1).sum())
        if t_low is None:
            agg["no_cert"] += 1
            agg["high"] += len(yt); agg["high_true"] += int(yt.sum())
            continue
        t_high = max(0.5, t_low)
        sup = pt_c < t_low
        high = pt_c >= t_high
        defer = ~sup & ~high
        agg["suppressed_fa"] += int((sup & (yt == 0)).sum())
        agg["missed"] += int((sup & (yt == 1)).sum())
        agg["auto_suppressed"] += int(sup.sum()); agg["deferred"] += int(defer.sum()); agg["high"] += int(high.sum())
        agg["supp_true"] += int((sup & (yt == 1)).sum()); agg["defer_true"] += int((defer & (yt == 1)).sum())
        agg["high_true"] += int((high & (yt == 1)).sum())

    cal = pd.DataFrame(cal_rows)
    n_cert = agg["folds"] - agg["no_cert"]
    util = {
        "variant": variant, "alpha_miss_target": alpha_miss, "confidence": 1 - delta,
        "folds_with_certified_threshold": f"{n_cert}/{agg['folds']}",
        "false_alarms_suppressed_pct": 100 * agg["suppressed_fa"] / max(agg["n_fa"], 1),
        "true_alarms_missed_pct": 100 * agg["missed"] / max(agg["n_true"], 1),
        "auto_suppressed_pct_of_alarms": 100 * agg["auto_suppressed"] / max(agg["n"], 1),
        "deferred_pct_of_alarms": 100 * agg["deferred"] / max(agg["n"], 1),
        "high_priority_pct_of_alarms": 100 * agg["high"] / max(agg["n"], 1),
        "true_rate_in_suppressed_pct": 100 * agg["supp_true"] / max(agg["auto_suppressed"], 1),
        "true_rate_in_deferred_pct": 100 * agg["defer_true"] / max(agg["deferred"], 1),
        "true_rate_in_high_priority_pct": 100 * agg["high_true"] / max(agg["high"], 1),
        "false_alarms_removed_per_100_alarms": 100 * agg["suppressed_fa"] / max(agg["n"], 1),
        "missed_true_alarms_per_1000_true": 1000 * agg["missed"] / max(agg["n_true"], 1),
    }
    if alarms_per_bed_day:
        util["ASSUMPTION_alarms_per_bed_day"] = alarms_per_bed_day
        util["false_alarms_removed_per_bed_day"] = alarms_per_bed_day * util["false_alarms_removed_per_100_alarms"] / 100
        util["missed_true_alarms_per_bed_day"] = (
            alarms_per_bed_day * util["missed_true_alarms_per_1000_true"] / 1000 * (agg["n_true"] / max(agg["n"], 1))
        )
    p_all, y_all = np.concatenate(all_p), np.concatenate(all_y)
    return util, cal, p_all, y_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--variant", default="cross_attention")
    ap.add_argument("--alpha-miss", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--alarms-per-bed-day", type=float, default=None,
                    help="user-supplied alarm volume; turns per-100-alarm rates into per-bed-day numbers")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = run_dir(cfg, args.run_name)

    util, cal, p_all, y_all = analyse(run, args.variant, args.alpha_miss, args.delta, args.alarms_per_bed_day)
    print(json.dumps(util, indent=2))
    print("\ncalibration (mean over folds):")
    print(cal[["T", "ece_raw", "ece_cal", "brier_raw", "brier_cal"]].mean().round(4).to_string())

    pts = np.linspace(0.05, 0.95, 91)
    nb = net_benefit(p_all, y_all, pts)
    dca = pd.DataFrame({"threshold": pts, "net_benefit_model": nb["model"], "net_benefit_alarm_all": nb["all"]})
    out = run / "safety"
    out.mkdir(exist_ok=True)
    (out / f"{args.variant}_utility.json").write_text(json.dumps(util, indent=2))
    cal.to_csv(out / f"{args.variant}_calibration.csv", index=False)
    dca.to_csv(out / f"{args.variant}_decision_curve.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(pts, nb["model"], label=f"{args.variant} verifier")
    ax.plot(pts, nb["all"], "--", label="alarm on everything")
    ax.axhline(0, color="grey", lw=0.8, label="suppress everything")
    ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit"); ax.legend(frameon=False)
    ax.set_title("Decision curve (calibrated, out-of-fold)")
    fig.tight_layout(); fig.savefig(out / f"{args.variant}_decision_curve.png", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
