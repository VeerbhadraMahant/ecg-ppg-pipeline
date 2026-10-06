"""Aggregate runs/classical_* into markdown tables (leaderboard + feature importances).

python -m src.classical_report  -> prints markdown, writes runs/classical_leaderboard.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = [("CinC 2015 (PPG cohort, 592)", "classical_cinc"), ("CinC recovered (724)", "classical_cinc_recovered"),
        ("VTaC (official split)", "classical_vtac")]
GBM = {"classical_cinc": "challenge2015_ppg", "classical_vtac": "vtac_official"}


def ms(s, k):
    return f"{s[k + '_mean']:.3f} ± {s[k + '_std']:.3f}"


def table(run: str) -> str:
    p = ROOT / "runs" / run / "results.json"
    if not p.exists():
        return "_not run yet_\n"
    r = json.loads(p.read_text())
    lines = ["| Model | Accuracy | F1 | AUC | Sensitivity | Specificity | Challenge | F1 (val-thr) | Acc (val-thr) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v, s in r.items():
        lines.append(f"| {v} | {ms(s, 'accuracy')} | {ms(s, 'f1')} | {ms(s, 'auc')} | {ms(s, 'sensitivity')} | "
                     f"{ms(s, 'specificity')} | {s['challenge_score_mean']:.1f} ± {s['challenge_score_std']:.1f} | "
                     f"{ms(s, 'f1_valthr')} | {ms(s, 'accuracy_valthr')} |")
    g = ROOT / "runs" / GBM.get(run, "x") / "results.json"
    if g.exists():
        gr = json.loads(g.read_text()).get("gbm_features")
        if gr:
            fm = gr["fold_metrics"]
            acc = np.mean([(m["tp"] + m["tn"]) / m["n"] for m in fm])
            lines.append(f"| _gbm_features (v1, reference)_ | {acc:.3f} | {ms(gr, 'f1')} | {ms(gr, 'auc')} | "
                         f"{ms(gr, 'sensitivity')} | {ms(gr, 'specificity')} | "
                         f"{np.nanmean([m['challenge_score'] for m in fm]):.1f} | - | - |")
    return "\n".join(lines) + "\n"


def importances(run: str, variant: str, top=15) -> str:
    files = sorted((ROOT / "runs" / run).glob(f"importance_{variant}_seed0_fold*.json"))
    if not files:
        return ""
    acc: dict[str, list[float]] = {}
    for f in files:
        for k, v in json.loads(f.read_text())["importance"].items():
            acc.setdefault(k, []).append(v)
    mean = sorted(((np.mean(v), k) for k, v in acc.items()), reverse=True)[:top]
    lines = [f"Top {top} by mean validation-AUC drop ({variant}, seed 0, {len(files)} folds):", "",
             "| Feature | AUC drop |", "|---|---|"]
    lines += [f"| {k} | {m:.4f} |" for m, k in mean]
    return "\n".join(lines) + "\n"


def main():
    out = []
    for title, run in RUNS:
        out += [f"### {title}", "", table(run)]
        for v in ("clf_boost", "clf_rf"):
            t = importances(run, v)
            if t:
                out += [t]
    txt = "\n".join(out)
    (ROOT / "runs" / "classical_leaderboard.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
