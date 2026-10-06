"""Aggregate the label-efficiency sweep (runs/labeleff/<init>_<mode>_f<frac>/results.json)
into one table + plot, with paired differences against training from scratch
on the SAME seeds/folds/records (updates.md analysis 6.7).

    python scripts/aggregate_label_efficiency.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats import corrected_resampled_ttest  # noqa: E402


def main():
    rows, per_fold = [], {}
    # cross_attention at its native width (labeleff/) and single-signal models at
    # width 1.0 (labeleff_single/): width-matched models cannot load the width-1.0
    # pretrained encoders, so single-signal runs use width 1.0 for BOTH scratch and
    # pretrained arms. Stale ecg_only/ppg_only entries in labeleff/ are ignored.
    dirs = [(d, {"cross_attention"}) for d in sorted((ROOT / "runs" / "labeleff").glob("*_f*"))]
    dirs += [(d, {"ecg_only", "ppg_only"}) for d in sorted((ROOT / "runs" / "labeleff_single").glob("*_f*"))]
    for d, allowed in dirs:
        m = re.match(r"(.+)_(finetune|frozen)_f([\d.]+)$", d.name)
        rp = d / "results.json"
        if not m or not rp.exists():
            continue
        init, mode, frac = m.group(1), m.group(2), float(m.group(3))
        res = json.loads(rp.read_text())
        for variant, s in res.items():
            if variant not in allowed:
                continue
            fm = s["fold_metrics"]
            per_fold[(variant, init, mode, frac)] = {(f["seed"], f["fold"]): f for f in fm}
            rows.append({"variant": variant, "init": init, "mode": mode, "label_fraction": frac,
                         "n_folds": len(fm), "F1": np.mean([f["f1"] for f in fm]),
                         "F1_std": np.std([f["f1"] for f in fm]), "AUC": np.nanmean([f["auc"] for f in fm]),
                         "n_train_records_mean": np.mean([f.get("n_train", np.nan) for f in fm])})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no completed runs yet")
        return

    # paired difference to scratch at the same fraction
    diffs = []
    for (variant, init, mode, frac), cur in per_fold.items():
        if init == "scratch":
            continue
        ref = per_fold.get((variant, "scratch", "finetune", frac))
        if not ref:
            continue
        keys = sorted(set(cur) & set(ref))
        if len(keys) < 5:
            continue
        d = np.array([cur[k]["f1"] - ref[k]["f1"] for k in keys])
        n_tr = np.mean([cur[k].get("n_train", 400) for k in keys])
        n_te = np.mean([cur[k].get("n_test", 120) for k in keys])
        t = corrected_resampled_ttest(d, n_tr, n_te)
        diffs.append({"variant": variant, "init": init, "mode": mode, "label_fraction": frac,
                      "dF1_vs_scratch": t["mean_diff"], "nb_p": t["p_value"], "n_pairs": t["n_pairs"]})
    dd = pd.DataFrame(diffs)
    out = ROOT / "runs" / "labeleff"
    out.mkdir(exist_ok=True)
    df.sort_values(["variant", "init", "mode", "label_fraction"]).to_csv(out / "summary.csv", index=False)
    dd.sort_values(["variant", "init", "mode", "label_fraction"]).to_csv(out / "paired_vs_scratch.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.pivot_table(index=["variant", "init", "mode"], columns="label_fraction", values="F1").round(3))
        print("\npaired dF1 vs scratch (same seeds/folds):")
        if len(dd):
            print(dd.pivot_table(index=["variant", "init", "mode"], columns="label_fraction", values="dF1_vs_scratch").round(3))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted(df.variant.unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, variants):
        for (init, mode), g in df[df.variant == v].groupby(["init", "mode"]):
            g = g.sort_values("label_fraction")
            ax.plot(g.label_fraction, g.F1, marker="o", ms=3, ls="-" if init == "scratch" else ("--" if mode == "finetune" else ":"),
                    label=f"{init}/{mode}")
        ax.set_title(v)
        ax.set_xlabel("fraction of training records")
    axes[0].set_ylabel("F1 (test folds)")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
