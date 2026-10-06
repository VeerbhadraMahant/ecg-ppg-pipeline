"""Protocol audit (updates.md 6.4): how much of a published method's performance
survives this repo's strict evaluation protocol?

Trains the reproduced Mousavi-style attention CNN-RNN (src/models/reproductions.py)
under, with identical seeds / folds count / model / optimiser / epochs:

  original : random window-wise STRATIFIED k-fold, no separate validation set;
             the best epoch and early stopping are chosen on the TEST fold
             (the likely leakage in the original protocol).
  random_val: same random stratified folds, but model selection on a held-out
             inner validation split (isolates the effect of test-fold selection).
  strict   : this repo's protocol - record-wise outer folds (iter_splits), record-wise
             inner validation, test fold touched once.

NOTE: CinC2015 has ONE window per record, so a random window-wise split is
already record-disjoint here; the record-level leakage that inflates published
numbers on multi-window datasets cannot occur. What remains - and is measured -
is (i) test-fold model selection/early stopping (original -> random_val) and
(ii) loss of the validation data from training plus fold-composition differences
(random_val -> strict). The gap here is therefore a LOWER bound on what window
leakage would cause on a multi-window dataset.

Usage:
    python -m src.analysis.protocol_audit --seeds 3 --out runs/protocol_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.train import iter_splits, set_seed  # noqa: E402
from src.train_reproductions import fit_fold, get_repro_width, make_tensors  # noqa: E402

VARIANT = "mousavi_attn_cnn_rnn"
METRICS = ["f1", "auc", "sensitivity", "specificity", "challenge_score"]


def random_splits(data, cfg, seed, with_val):
    y = data["label"].astype(int)
    skf = StratifiedKFold(cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"] + seed)
    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        if with_val:
            fit, val = train_test_split(tr, test_size=cfg["split"].get("val_fraction", 0.15),
                                        stratify=y[tr], random_state=cfg["seed"] + seed + fold_i)
        else:
            fit, val = tr, None
        yield fold_i, np.sort(fit), (None if val is None else np.sort(val)), te


def run_protocol(name, cfg, data, T, device, seeds, width):
    rows = []
    for seed in seeds:
        set_seed(cfg["seed"] + seed)
        if name == "strict":
            gen = iter_splits(data, cfg, seed, "cv")
        else:
            gen = random_splits(data, cfg, seed, with_val=(name == "random_val"))
        for fold_i, fit, val, te in gen:
            sel = "test" if name == "original" else "val"
            m, _ = fit_fold(cfg, VARIANT, width, T, fit, val, te, device, cfg["seed"] + seed + fold_i, selection=sel)
            rows.append({"seed": seed, "fold": fold_i, **{k: float(m[k]) for k in METRICS}})
            print(f"[{name}] seed={seed} fold={fold_i} f1={m['f1']:.3f} auc={m['auc']:.3f}", flush=True)
    return rows


def summarize(rows):
    """Per-seed mean over folds, then mean/std over seeds."""
    seeds = sorted({r["seed"] for r in rows})
    out = {}
    for k in METRICS:
        per_seed = np.array([np.nanmean([r[k] for r in rows if r["seed"] == s]) for s in seeds])
        out[k] = {"mean": float(per_seed.mean()), "std": float(per_seed.std()), "per_seed": per_seed.tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--processed", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "protocol_audit"))
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    data = load_processed(args.processed or (ROOT / cfg["paths"]["processed_dir"] / "challenge2015_windows.npz"))
    T = make_tensors(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width = get_repro_width(VARIANT)
    seeds = list(range(args.seeds))

    res = {"variant": VARIANT, "width_mult": width, "seeds": seeds, "protocols": {}, "rows": {}}
    for name in ("original", "random_val", "strict"):
        rows = run_protocol(name, cfg, data, T, device, seeds, width)
        res["rows"][name] = rows
        res["protocols"][name] = summarize(rows)

    P = res["protocols"]
    res["gap"] = {
        "original_minus_strict": {k: P["original"][k]["mean"] - P["strict"][k]["mean"] for k in METRICS},
        "original_minus_random_val": {k: P["original"][k]["mean"] - P["random_val"][k]["mean"] for k in METRICS},
        "random_val_minus_strict": {k: P["random_val"][k]["mean"] - P["strict"][k]["mean"] for k in METRICS},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol_audit.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'protocol':12s}" + "".join(f"{k:>18s}" for k in METRICS))
    for name, s in P.items():
        print(f"{name:12s}" + "".join(f"{s[k]['mean']:>11.3f}+/-{s[k]['std']:.3f}" for k in METRICS))
    for g, d in res["gap"].items():
        print(f"{g:28s}" + "".join(f"{d[k]:>+9.3f}" for k in METRICS))
    print(f"written to {out / 'protocol_audit.json'}")


if __name__ == "__main__":
    main()
