"""Gradient-boosted trees on hand-crafted agreement features (updates.md 3.4).

Uses the SAME record-wise outer folds, inner validation holdout, seeds and
prediction/metric format as src/train.py, so its scores slot into the same
tables and paired significance tests as the neural models.

Usage:
    python -m src.baselines_gbm --run-name challenge2015_ppg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed, record_wise_folds, record_wise_holdout  # noqa: E402
from src.features import feature_matrix  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.train import iter_splits, run_dir  # noqa: E402

VARIANT = "gbm_features"


def cached_features(data: dict, cache_path: Path):
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if len(z["X"]) == len(data["label"]):
            return z["X"], list(z["names"])
    t0 = time.time()
    X, names = feature_matrix(data["ecg"], data["ppg"])
    print(f"features: {X.shape} in {time.time() - t0:.0f}s")
    np.savez(cache_path, X=X, names=np.array(names))
    return X, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--processed", default=None)
    parser.add_argument("--run-name", default="challenge2015_ppg")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--protocol", default="cv", choices=["cv", "official"])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_seeds = args.seeds or cfg["train"].get("n_seeds", 3)
    processed = Path(args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz"))
    data = load_processed(processed)
    X, names = cached_features(data, processed.with_name(processed.stem + "_features.npz"))
    y, rec = data["label"], data["record_id"]

    out_dir = run_dir(cfg, args.run_name)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for seed in range(n_seeds):
        for fold_i, fit_idx, val_idx, test_idx in iter_splits(data, cfg, seed, args.protocol):
            clf = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False, random_state=cfg["seed"] + seed,
            )
            clf.fit(X[fit_idx], y[fit_idx].astype(int))
            val_prob = clf.predict_proba(X[val_idx])[:, 1]
            test_prob = clf.predict_proba(X[test_idx])[:, 1]

            m = binary_metrics(y[test_idx], test_prob)
            m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
            m["ops"] = operating_points(y[val_idx], val_prob, y[test_idx], test_prob)
            m.update(params=0, seed=seed, fold=fold_i, n_train=len(fit_idx), n_test=len(test_idx))
            fold_metrics.append(m)
            np.savez(out_dir / "preds" / f"{VARIANT}_seed{seed}_fold{fold_i}.npz",
                     test_idx=test_idx, val_idx=val_idx, val_prob=val_prob, val_y=y[val_idx],
                     test_prob=test_prob, test_y=y[test_idx])
            (out_dir / "preds" / f"{VARIANT}_seed{seed}_fold{fold_i}.json").write_text(json.dumps(m, default=float))

    summary = summarize_across_folds(fold_metrics)
    summary.update(variant=VARIANT, width_mult=1.0, fold_metrics=fold_metrics)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results[VARIANT] = summary
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"{VARIANT}: F1={summary['f1_mean']:.3f}+/-{summary['f1_std']:.3f} "
          f"Sens={summary['sensitivity_mean']:.3f} Spec={summary['specificity_mean']:.3f} AUC={summary['auc_mean']:.3f}")

    # feature importances via permutation would be costly; report top split-gain free proxy: none for HGB
    print(f"{len(names)} features used: {', '.join(names[:8])}, ...")


if __name__ == "__main__":
    main()
