"""Cross-dataset / cross-site generalisation (updates.md analysis 6.5).

Train on dataset A (inner record-wise validation holdout for early stopping),
test on dataset B, with several seeds. Typical uses:

  CinC 2015 -> VTaC (official test split):
      python -m src.cross_dataset --train data/processed/challenge2015_windows_recovered.npz \
          --test data/processed/vtac_windows.npz --test-split test --name cinc_to_vtac
  VTaC (train+val) -> CinC 2015 VT alarms only:
      python -m src.cross_dataset --train data/processed/vtac_windows.npz --train-splits train,val \
          --test data/processed/challenge2015_windows_recovered.npz \
          --test-filter alarm_type=Ventricular_Tachycardia --name vtac_to_cinc_vt
  Leave-one-device-out inside VTaC, using the lead-signature PROXY for monitor
  manufacturer (VTaC does not release device/hospital ids):
      python -m src.cross_dataset --train data/processed/vtac_windows.npz --train-splits train,val,test \
          --holdout-signature "I|II|III|V" --name vtac_lodo

For a fair comparison the same-task (in-distribution) reference for the test
set is the model trained on B itself, produced by src/train.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset, load_processed, record_wise_holdout  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.models.classifier import ALL_VARIANTS  # noqa: E402
from src.train import get_width_mult, run_dir, set_seed, train_one_fold  # noqa: E402


def select(data: dict, mask: np.ndarray) -> dict:
    n = len(data["label"])
    return {k: v[mask] for k, v in data.items() if hasattr(v, "__len__") and len(v) == n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="cross_dataset")
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--train-splits", default=None, help="comma list of `split` values to keep for training")
    ap.add_argument("--train-filter", default=None, help="key=value filter on the training set")
    ap.add_argument("--test", default=None)
    ap.add_argument("--test-split", default=None)
    ap.add_argument("--test-filter", default=None, help="key=value filter on the test set")
    ap.add_argument("--holdout-signature", default=None,
                    help="leave-one-device-out: hold out windows with this lead_signature from --train data")
    ap.add_argument("--variants", default="ecg_only,ppg_only,concat,cross_attention")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    A = load_processed(args.train)
    if args.train_splits:
        A = select(A, np.isin(A["split"], args.train_splits.split(",")))
    if args.train_filter:
        k, v = args.train_filter.split("=")
        A = select(A, A[k] == v)

    if args.holdout_signature:
        sig = A["lead_signature"]
        B, A = select(A, sig == args.holdout_signature), select(A, sig != args.holdout_signature)
    else:
        B = load_processed(args.test)
        if args.test_split:
            B = select(B, B["split"] == args.test_split)
        if args.test_filter:
            k, v = args.test_filter.split("=")
            B = select(B, B[k] == v)
    print(f"train: {len(A['label'])} windows ({A['label'].mean():.1%} true) | "
          f"test: {len(B['label'])} windows ({B['label'].mean():.1%} true)")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    out_dir = run_dir(cfg, args.run_name) / args.name
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in args.variants.split(","):
        if variant not in ALL_VARIANTS:
            raise SystemExit(f"unknown variant {variant}")
        wm = get_width_mult(variant, ROOT)
        folds = []
        for seed in range(args.seeds):
            set_seed(cfg["seed"] + seed)
            fit, val = record_wise_holdout(A["record_id"], np.arange(len(A["label"])),
                                           cfg["split"].get("val_fraction", 0.15), cfg["seed"] + seed)
            mk = lambda d, ix=None: AlarmWindowDataset(  # noqa: E731
                d["ecg"] if ix is None else d["ecg"][ix], d["ppg"] if ix is None else d["ppg"][ix],
                d["label"] if ix is None else d["label"][ix])
            m, _model, preds = train_one_fold(cfg, variant, wm, mk(A, fit), mk(A, val), mk(B), device, seed)
            m.update(seed=seed, fold=0, n_train=len(fit), n_test=len(B["label"]))
            folds.append(m)
            np.savez(out_dir / "preds" / f"{variant}_seed{seed}.npz", **preds)
            print(f"[{args.name}][{variant}] seed={seed} F1={m['f1']:.3f} AUC={m['auc']:.3f} "
                  f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} chall={m['challenge_score']:.1f}", flush=True)
        s = summarize_across_folds(folds)
        s["fold_metrics"] = folds
        results[variant] = s
        print(f"== {variant}: F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
