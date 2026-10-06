"""Modality-subset study and missing-sensor robustness (updates.md 6.8, 6.9).

A single multimodal model (src/models/multimodal.py, trained with modality
dropout) is evaluated with EVERY non-empty subset of {ecg, ecg2, ppg, abp}
switched on at inference by masking the others, on the SAME test windows
(only windows where all four modalities genuinely exist, so subsets are
compared on identical samples).

Outputs
  subsets.csv      AUC / F1 / sensitivity / specificity per subset (mean over
                   folds, seed-0 checkpoints) with a bootstrap CI over windows
  shapley.csv      Shapley-style marginal contribution of each modality to AUC,
                   with v(empty set) = 0.5 (chance)
  robustness.csv   performance as sensors drop out one by one (all 4 -> 3 -> 2 -> 1),
                   averaged over every drop order

Usage:
    python -m src.analysis.modality_subsets --run-name challenge2015_recovered \
        --processed data/processed/challenge2015_windows_recovered.npz
"""
from __future__ import annotations

import argparse
import itertools
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.models.multimodal import MODALITIES, to_multimodal_inputs  # noqa: E402
from src.train import get_width_mult, iter_splits, run_dir  # noqa: E402


@torch.no_grad()
def predict(model, x, m, device, bs=128):
    out = []
    for i in range(0, len(x), bs):
        lo, _ = model(torch.from_numpy(x[i:i + bs]).to(device), torch.from_numpy(m[i:i + bs]).to(device))
        out.append(torch.sigmoid(lo).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_recovered")
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed" / "challenge2015_windows_recovered.npz"))
    ap.add_argument("--protocol", default="cv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data = load_processed(args.processed)
    x, mask = to_multimodal_inputs(data)
    y = data["label"].astype(int)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_path = run_dir(cfg, args.run_name)
    width = get_width_mult("multimodal", ROOT)

    # fixed per-fold test windows with all four modalities present
    full = mask.sum(1) == 4
    print(f"windows with all 4 modalities: {int(full.sum())} / {len(y)}")

    probs = {}  # subset(tuple of ints) -> (idx, prob) concatenated over folds
    subsets = [s for r in range(1, 5) for s in itertools.combinations(range(4), r)]
    pool_idx, pool_prob = {s: [] for s in subsets}, {s: [] for s in subsets}
    for fold, fit, val, test in iter_splits(data, cfg, args.seed, args.protocol):
        test = test[full[test]]
        if len(test) == 0:
            continue
        model = build_model("multimodal", cfg, width).to(device)
        model.load_state_dict(torch.load(run_path / "checkpoints" / f"multimodal_seed{args.seed}_fold{fold}.pt", map_location=device))
        model.eval()
        for s in subsets:
            mk = np.zeros_like(mask[test])
            mk[:, list(s)] = 1.0
            pool_idx[s].append(test)
            pool_prob[s].append(predict(model, x[test], mk, device))
    idx = np.concatenate(pool_idx[subsets[0]])
    yt = y[idx]
    rng = np.random.RandomState(0)

    rows, auc_of = [], {}
    for s in subsets:
        p = np.concatenate(pool_prob[s])
        auc = roc_auc_score(yt, p)
        auc_of[s] = auc
        boots = []
        for _ in range(300):
            b = rng.randint(0, len(yt), len(yt))
            if len(set(yt[b])) > 1:
                boots.append(roc_auc_score(yt[b], p[b]))
        pred = p >= 0.5
        rows.append({"subset": "+".join(MODALITIES[i] for i in s), "n_modalities": len(s), "auc": auc,
                     "auc_ci_low": np.percentile(boots, 2.5), "auc_ci_high": np.percentile(boots, 97.5),
                     "f1@0.5": f1_score(yt, pred), "sensitivity": float((pred & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                     "specificity": float((~pred & (yt == 0)).sum() / max((yt == 0).sum(), 1)), "n_windows": len(yt)})
    out = run_path / "modality_subsets"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["n_modalities", "auc"], ascending=[True, False])
    df.to_csv(out / "subsets.csv", index=False)
    print(df.round(3).to_string(index=False))

    # Shapley value of each modality for AUC, v(empty) = 0.5
    n = 4
    v = {(): 0.5, **auc_of}
    sh = {}
    for i in range(n):
        tot = 0.0
        others = [j for j in range(n) if j != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                w = factorial(r) * factorial(n - r - 1) / factorial(n)
                tot += w * (v[tuple(sorted(S + (i,)))] - v[tuple(sorted(S))])
        sh[MODALITIES[i]] = tot
    shdf = pd.DataFrame({"modality": list(sh), "shapley_auc": list(sh.values())})
    shdf.to_csv(out / "shapley.csv", index=False)
    print("\nShapley contribution to AUC (sums to AUC(all) - 0.5):")
    print(shdf.round(4).to_string(index=False), "| total", round(sum(sh.values()), 4))

    # sensor drop-out: average AUC over all drop orders as sensors disappear
    rob = []
    for k in range(4, 0, -1):
        aucs = [auc_of[tuple(sorted(c))] for c in itertools.combinations(range(4), k)]
        rob.append({"n_sensors_available": k, "mean_auc_over_subsets": float(np.mean(aucs)),
                    "worst_auc": float(np.min(aucs)), "best_auc": float(np.max(aucs))})
    pd.DataFrame(rob).to_csv(out / "robustness.csv", index=False)
    print("\nRobustness as sensors drop out:")
    print(pd.DataFrame(rob).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
