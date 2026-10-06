"""Leak-free ensembling / stacking of already-trained models.

Every (seed, fold) of every member run has saved INNER-VALIDATION predictions
(val_prob) and TEST predictions (test_prob) over identical splits. A combiner
is fitted on the validation predictions of that fold only and applied to the
test predictions, so the outer test fold is never used for any decision.

Combiners (new variant names written back into the run, comparable in the
same tables / significance tests):
  ens_mean      average of member probabilities
  ens_logit     average of member logits
  ens_stack     logistic-regression stacker on member logits (fit on val)
  ens_greedy    Caruana-style greedy forward selection (with replacement)
                maximising validation AUC, then probability averaging
  ens_topk      mean of the k members with the best validation AUC

Usage:
    python -m src.ensemble --run-name challenge2015_ppg \
        --members gbm_features,resnet1d,tcn,inceptiontime,ssm,cross_attention
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.train import run_dir  # noqa: E402


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def fold_keys(run: Path, member: str):
    return {(int(f.stem.split("_seed")[1].split("_fold")[0]), int(f.stem.split("_fold")[1])): f
            for f in (run / "preds").glob(f"{member}_seed*_fold*.npz")}


def resolve(cfg: dict, default_run: Path, member: str):
    """'variant' (in the default run) or 'run_name:variant' (any run directory)."""
    if ":" in member:
        rn, v = member.split(":", 1)
        return run_dir(cfg, rn), v
    return default_run, member


def combine(name: str, V: np.ndarray, T: np.ndarray, yv: np.ndarray, k: int = 3):
    """V, T: (n_members, n_val/test) probabilities. Returns (val_prob, test_prob)."""
    Lv, Lt = _logit(V), _logit(T)
    if name == "ens_mean":
        return V.mean(0), T.mean(0)
    if name == "ens_logit":
        return _sig(Lv.mean(0)), _sig(Lt.mean(0))
    if name == "ens_stack":
        clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
        clf.fit(Lv.T, yv)
        return clf.predict_proba(Lv.T)[:, 1], clf.predict_proba(Lt.T)[:, 1]
    aucs = np.array([roc_auc_score(yv, v) if len(set(yv)) > 1 else 0.5 for v in V])
    if name == "ens_topk":
        top = np.argsort(-aucs)[:k]
        return V[top].mean(0), T[top].mean(0)
    if name == "ens_greedy":
        chosen, best = [], -1.0
        for _ in range(12):
            cand = [roc_auc_score(yv, V[chosen + [j]].mean(0)) for j in range(len(V))]
            j = int(np.argmax(cand))
            if cand[j] <= best + 1e-9 and chosen:
                break
            chosen.append(j)
            best = cand[j]
        return V[chosen].mean(0), T[chosen].mean(0)
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--members", required=True, help="comma list of variant names with saved preds")
    ap.add_argument("--combiners", default="ens_mean,ens_logit,ens_stack,ens_greedy,ens_topk")
    ap.add_argument("--suffix", default="", help="appended to the combiner name, e.g. _top6")
    ap.add_argument("--out-run", default=None, help="write ensemble preds/results here (default: --run-name)")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = run_dir(cfg, args.run_name)
    out_run = run_dir(cfg, args.out_run) if args.out_run else run
    out_run.mkdir(parents=True, exist_ok=True)
    members = args.members.split(",")

    keysets = [fold_keys(*resolve(cfg, run, m)) for m in members]
    common = sorted(set.intersection(*[set(k) for k in keysets]))
    print(f"{len(members)} members, {len(common)} common (seed, fold) splits")

    results_path = out_run / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for comb in args.combiners.split(","):
        folds = []
        for sd, fo in common:
            zs = [np.load(k[(sd, fo)]) for k in keysets]
            yv, yt = zs[0]["val_y"].astype(int), zs[0]["test_y"].astype(int)
            assert all(np.array_equal(z["test_idx"], zs[0]["test_idx"]) for z in zs), "split mismatch"
            V = np.stack([z["val_prob"] for z in zs])
            T = np.stack([z["test_prob"] for z in zs])
            pv, pt = combine(comb, V, T, yv, args.topk)
            m = binary_metrics(yt, pt)
            m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
            m["ops"] = operating_points(yv, pv, yt, pt)
            meta = json.loads(keysets[0][(sd, fo)].with_suffix(".json").read_text())
            m.update(params=0, seed=sd, fold=fo, n_train=meta.get("n_train"), n_test=meta.get("n_test"))
            folds.append(m)
            name = comb + args.suffix
            (out_run / "preds").mkdir(exist_ok=True)
            np.savez(out_run / "preds" / f"{name}_seed{sd}_fold{fo}.npz", test_idx=zs[0]["test_idx"],
                     val_idx=zs[0]["val_idx"], val_prob=pv, val_y=yv, test_prob=pt, test_y=yt)
            (out_run / "preds" / f"{name}_seed{sd}_fold{fo}.json").write_text(json.dumps(m, default=float))
        s = summarize_across_folds(folds)
        s.update(variant=comb + args.suffix, width_mult=1.0, fold_metrics=folds, members=members)
        results[comb + args.suffix] = s
        acc = np.mean([(f["tp"] + f["tn"]) / f["n"] for f in folds])
        print(f"{comb + args.suffix:16s} acc={acc:.3f} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} "
              f"AUC={s['auc_mean']:.3f} sens={s['sensitivity_mean']:.3f} spec={s['specificity_mean']:.3f}")
    results_path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
