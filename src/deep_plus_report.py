"""Tables for docs/deep_recipe.md from runs/deep_plus_* (analysis only).

    python -m src.deep_plus_report cinc deep_plus_cinc <variant> [<variant> ...]
Prints markdown: headline metrics per variant (mean +/- std over seed x fold),
the ablation views stored in the main variant's preds, and paired differences.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import binary_metrics  # noqa: E402


def fold_metrics(y, p):
    m = binary_metrics(y, p)
    m["acc"] = (m["tp"] + m["tn"]) / m["n"]
    return m


def load(run: str, variant: str):
    files = sorted((ROOT / "runs" / run / "preds").glob(f"{variant}_seed*_fold*.npz"))
    return [(f, np.load(f)) for f in files]


def table_for(run, variant, view=None):
    rows = []
    for f, z in load(run, variant):
        p = z["test_prob"] if view is None else z[f"test_prob__{view}"]
        m = fold_metrics(z["test_y"], p)
        seed = int(f.stem.split("_seed")[1].split("_")[0])
        fold = int(f.stem.split("_fold")[1])
        m.update(seed=seed, fold=fold)
        rows.append(m)
    return rows


def fmt(rows, keys=("acc", "f1", "auc", "sensitivity", "specificity")):
    out = []
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        out.append(f"{v.mean():.3f} ± {v.std():.3f}")
    return " | ".join(out)


def paired(a, b, key="f1"):
    da = {(r["seed"], r["fold"]): r[key] for r in a}
    db = {(r["seed"], r["fold"]): r[key] for r in b}
    ks = sorted(set(da) & set(db))
    d = np.array([da[k] - db[k] for k in ks])
    # seed-level means (folds averaged) to respect dependence between folds
    seeds = sorted({k[0] for k in ks})
    ds = np.array([np.mean([da[k] - db[k] for k in ks if k[0] == s]) for s in seeds])
    se = ds.std(ddof=1) / np.sqrt(len(ds)) if len(ds) > 1 else float("nan")
    return d.mean(), se, len(ds)


def main():
    _, run, *variants = sys.argv[1:]
    print("| variant | n | acc | F1 | AUC | sens | spec |\n|---|---|---|---|---|---|---|")
    tabs = {}
    for v in variants:
        rows = table_for(run, v)
        tabs[v] = rows
        print(f"| {v} | {len(rows)} | {fmt(rows)} |")
    main_v = variants[0]
    files = load(run, main_v)
    if files:
        views = [k[len("test_prob__"):] for k in files[0][1].files if k.startswith("test_prob__")]
        print(f"\nViews stored in {main_v} (same trained models):\n")
        print("| view | acc | F1 | AUC | sens | spec | dF1 vs ens_ema_tta (±SE over seeds) |\n|---|---|---|---|---|---|---|")
        kmax = max(int(v.split("_")[0][3:]) for v in views if v.startswith("ens") and v.endswith("ema_tta"))
        ref = table_for(run, main_v, f"ens{kmax}_ema_tta")
        for v in sorted(views):
            rows = table_for(run, main_v, v)
            d, se, _ = paired(rows, ref)
            print(f"| {v} | {fmt(rows)} | {d:+.3f} ± {se:.3f} |")
    print("\nPaired F1/AUC differences, variant minus first variant (mean, SE over seeds):\n")
    for v in variants[1:]:
        for k in ("f1", "auc"):
            d, se, n = paired(tabs[v], tabs[main_v], k)
            print(f"- {v} - {main_v}: d{k} = {d:+.4f} (SE {se:.4f}, {n} seeds)")


if __name__ == "__main__":
    main()
