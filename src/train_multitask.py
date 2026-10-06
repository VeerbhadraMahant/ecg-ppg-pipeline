"""Multitask trainer + task-interference guardrail (updates.md 5.5, 5.6, 7.1, 7.2, 13).

Trains the cross_attention backbone with auxiliary heads (alarm type, per-channel
SQI, beat-position heatmaps) under a weighted multi-task loss, and the SAME
model with all auxiliary weights = 0 ("noaux"), over identical seeds/folds/inner
splits and per-fold seeding. Conventions follow src/train.py: record-wise outer
folds + inner record-wise validation (iter_splits), early stopping and
best-epoch selection on inner-val MAIN-task F1 only, per-fold preds
npz/json (test_idx, val_idx, val_prob, val_y, test_prob, test_y), results.json
keyed by variant ("mt_aux", "mt_noaux") with the same schema as src/train.py.

Auxiliary targets (all derived without human labels, label-independent):
  type    : alarm_type string -> 5 classes
  quality : PSEUDO SQI from src.heads.quality.quality_targets
  beat    : binary token map of detected R peaks (ECG) / PPG peaks (PPG), from
            src.features detectors (these are detector outputs, not ground truth)

Usage:
    python -m src.train_multitask --run-name multitask_guardrail --seeds 3
    python -m src.train_multitask --compare-only --run-name multitask_guardrail
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.heads.beat_ptt import beat_heatmap_targets  # noqa: E402
from src.heads.quality import quality_targets  # noqa: E402
from src.features import detect_ppg_peaks, detect_r_peaks  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.models.losses import build_loss  # noqa: E402
from src.models.multitask import build_multitask  # noqa: E402
from src.stats import corrected_resampled_ttest, record_bootstrap  # noqa: E402
from src.train import iter_splits, run_dir, set_seed  # noqa: E402

DEFAULT_WEIGHTS = {"type": 0.3, "quality": 1.0, "beat": 0.3}
ZERO_WEIGHTS = {"type": 0.0, "quality": 0.0, "beat": 0.0}
BEAT_POS_WEIGHT = 4.0  # peaks occupy ~4-5% of tokens


def build_aux_targets(data: dict, cfg: dict, n_tokens: int) -> dict:
    """Cache-friendly aux targets for every window."""
    q = cfg["quality"]
    T = data["ecg"].shape[1]
    types = sorted(np.unique(data["alarm_type"]))
    type_idx = np.array([types.index(t) for t in data["alarm_type"]], dtype=np.int64)
    r = [detect_r_peaks(e) for e in data["ecg"]]
    p = [detect_ppg_peaks(x) for x in data["ppg"]]
    beat = np.stack([beat_heatmap_targets(r, T, n_tokens), beat_heatmap_targets(p, T, n_tokens)], axis=-1)  # (N,T',2)
    return {
        "type": type_idx, "types": types,
        "quality": quality_targets(data["ecg"], data["ppg"], q["flatline_std_threshold"], q["clip_fraction_threshold"]),
        "beat": beat,
    }


def _tensors(data, aux, idx):
    f = lambda a, dt=torch.float32: torch.from_numpy(np.asarray(a)[idx]).to(dt)  # noqa: E731
    return TensorDataset(f(data["ecg"]), f(data["ppg"]), f(data["label"]), f(aux["type"], torch.long),
                         f(aux["quality"]), f(aux["beat"]))


@torch.no_grad()
def _evaluate(model, ds, device, bs):
    model.eval()
    out = {"prob": [], "type": [], "q": [], "beat": []}
    for ecg, ppg, *_ in DataLoader(ds, batch_size=bs, shuffle=False):
        o = model.forward_all(ecg.to(device), ppg.to(device))
        out["prob"].append(torch.sigmoid(o["logit"]).cpu().numpy())
        out["type"].append(o["type_logits"].argmax(-1).cpu().numpy())
        out["q"].append(o["quality"].cpu().numpy())
        out["beat"].append(torch.sigmoid(o["beat_logits"]).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def aux_test_metrics(ev, ds) -> dict:
    """How well each auxiliary head did on the held-out fold (reported even if its weight is 0,
    in which case the head is untrained and these are chance-level reference values)."""
    _, _, _, ty, q, bt = ds.tensors
    ty, q, bt = ty.numpy(), q.numpy(), bt.numpy()
    pred_b = ev["beat"] >= 0.5
    out = {"type_acc": float((ev["type"] == ty).mean()), "quality_mae": float(np.abs(ev["q"] - q).mean())}
    for j, n in enumerate(("ecg", "ppg")):
        yt, pt = q[:, j], ev["q"][:, j]
        out[f"quality_pearson_{n}"] = float(np.corrcoef(yt, pt)[0, 1]) if pt.std() > 0 and yt.std() > 0 else 0.0
        # beat heatmap: token-level F1 and +/-1-token tolerant F1
        b, pb = bt[..., j] > 0.5, pred_b[..., j]
        tp, fp, fn = (b & pb).sum(), (~b & pb).sum(), (b & ~pb).sum()
        out[f"beat_f1_{n}"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def train_fold(cfg, weights, data, aux, fit_idx, val_idx, test_idx, device):
    t = cfg["train"]
    train_ds, val_ds, test_ds = (_tensors(data, aux, i) for i in (fit_idx, val_idx, test_idx))
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True)
    model = build_multitask(cfg).to(device)
    n_pos = float(train_ds.tensors[2].sum())
    pos_weight = (len(train_ds) - n_pos) / n_pos if n_pos > 0 else 1.0
    main_loss = build_loss(t["loss"], pos_weight, t.get("focal_gamma", 2.0), device)
    beat_pw = torch.tensor(BEAT_POS_WEIGHT, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    best_f1, best_state, patience = -1.0, None, t["early_stop_patience"]
    for _ in range(t["epochs"]):
        model.train()
        for ecg, ppg, y, ty, q, bt in loader:
            ecg, ppg, y, ty, q, bt = (v.to(device) for v in (ecg, ppg, y, ty, q, bt))
            o = model.forward_all(ecg, ppg)
            loss = main_loss(o["logit"], y)
            if weights["type"] > 0:
                loss = loss + weights["type"] * F.cross_entropy(o["type_logits"], ty)
            if weights["quality"] > 0:
                loss = loss + weights["quality"] * F.mse_loss(o["quality"], q)
            if weights["beat"] > 0:
                loss = loss + weights["beat"] * F.binary_cross_entropy_with_logits(o["beat_logits"], bt, pos_weight=beat_pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # early stopping / model selection on inner-val MAIN-task F1 only
        ev = _evaluate(model, val_ds, device, t["batch_size"] * 2)
        f1 = binary_metrics(val_ds.tensors[2].numpy(), ev["prob"])["f1"]
        if f1 > best_f1:
            best_f1, patience = f1, t["early_stop_patience"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    bs = t["batch_size"] * 2
    ev_val, ev_test = _evaluate(model, val_ds, device, bs), _evaluate(model, test_ds, device, bs)
    val_y, test_y = val_ds.tensors[2].numpy(), test_ds.tensors[2].numpy()
    m = binary_metrics(test_y, ev_test["prob"])
    m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
    m["ops"] = operating_points(val_y, ev_val["prob"], test_y, ev_test["prob"])
    m["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m["aux"] = aux_test_metrics(ev_test, test_ds)
    preds = {"val_prob": ev_val["prob"], "val_y": val_y, "test_prob": ev_test["prob"], "test_y": test_y}
    return m, preds


def run_variant(name, weights, cfg, data, aux, device, n_seeds, out_dir, resume=True):
    fms = []
    for seed in range(n_seeds):
        for fold_i, fit, val, test in iter_splits(data, cfg, seed, "cv"):
            pred_path = out_dir / "preds" / f"{name}_seed{seed}_fold{fold_i}.npz"
            m_path = pred_path.with_suffix(".json")
            if resume and pred_path.exists() and m_path.exists():
                fms.append(json.loads(m_path.read_text()))
                continue
            set_seed(cfg["seed"] + seed * 100 + fold_i)  # identical init/batch order in both arms
            t0 = time.time()
            m, preds = train_fold(cfg, weights, data, aux, fit, val, test, device)
            m.update(seed=seed, fold=fold_i, train_seconds=time.time() - t0, n_train=len(fit), n_test=len(test),
                     weights=weights)
            fms.append(m)
            print(f"[{name}] seed={seed} fold={fold_i} f1={m['f1']:.3f} auc={m['auc']:.3f} "
                  f"type_acc={m['aux']['type_acc']:.2f} ({m['train_seconds']:.0f}s)", flush=True)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pred_path, test_idx=test, val_idx=val, **preds)
            m_path.write_text(json.dumps(m, default=float))
    s = summarize_across_folds(fms)
    s.update(variant=name, width_mult=1.0, weights=weights, fold_metrics=fms)
    return s


def _tallies(out_dir, name, n, seeds, thr=0.5):
    t = np.zeros((len(seeds), n, 4))
    for k, sd in enumerate(seeds):
        for f in sorted((out_dir / "preds").glob(f"{name}_seed{sd}_fold*.npz")):
            z = np.load(f)
            pred, y = z["test_prob"] >= thr, z["test_y"] == 1
            i = z["test_idx"]
            t[k, i, 0], t[k, i, 1], t[k, i, 2], t[k, i, 3] = pred & y, pred & ~y, ~pred & ~y, ~pred & y
    return t


def compare(out_dir: Path, results: dict, record_id: np.ndarray, a="mt_noaux", b="mt_aux", n_boot=5000) -> dict:
    """Guardrail: paired (aux - noaux) differences over identical (seed, fold) pairs."""
    fa = {(m["seed"], m["fold"]): m for m in results[a]["fold_metrics"]}
    fb = {(m["seed"], m["fold"]): m for m in results[b]["fold_metrics"]}
    keys = sorted(set(fa) & set(fb))
    n_train = np.mean([fa[k]["n_train"] for k in keys])
    n_test = np.mean([fa[k]["n_test"] for k in keys])
    out = {"baseline": a, "challenger": b, "n_pairs": len(keys)}
    for metric in ("f1", "auc", "sensitivity", "specificity", "challenge_score"):
        d = np.array([fb[k][metric] - fa[k][metric] for k in keys], dtype=float)
        d = d[~np.isnan(d)]
        out[metric] = {"mean_noaux": float(np.nanmean([fa[k][metric] for k in keys])),
                       "mean_aux": float(np.nanmean([fb[k][metric] for k in keys])),
                       **corrected_resampled_ttest(d, n_train, n_test)}
    seeds = sorted({k[0] for k in keys})
    # collapse window tallies to per-RECORD tallies (record = unit of independence)
    _, inv = np.unique(record_id, return_inverse=True)
    def per_record(t):
        o = np.zeros((t.shape[0], inv.max() + 1, 4))
        for k in range(t.shape[0]):
            np.add.at(o[k], inv, t[k])
        return o
    ta, tb = (per_record(_tallies(out_dir, n, len(record_id), seeds)) for n in (a, b))
    out["record_bootstrap_f1_t50"] = record_bootstrap(ta, tb, "f1", n_boot)
    out["record_bootstrap_challenge_score_t50"] = record_bootstrap(ta, tb, "challenge_score", n_boot)
    # aux head quality on held-out folds (aux arm only; noaux heads are untrained)
    keys_aux = ["type_acc", "quality_mae", "quality_pearson_ecg", "quality_pearson_ppg", "beat_f1_ecg", "beat_f1_ppg"]
    out["aux_head_test_metrics_mean"] = {
        k: {"aux": float(np.mean([fb[x]["aux"][k] for x in keys])),
            "noaux_untrained": float(np.mean([fa[x]["aux"][k] for x in keys]))} for k in keys_aux}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="multitask_guardrail")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--processed", default=None)
    ap.add_argument("--w-type", type=float, default=DEFAULT_WEIGHTS["type"])
    ap.add_argument("--w-quality", type=float, default=DEFAULT_WEIGHTS["quality"])
    ap.add_argument("--w-beat", type=float, default=DEFAULT_WEIGHTS["beat"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--arms", default="mt_noaux,mt_aux", help="comma list from mt_noaux, mt_aux")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    data = load_processed(args.processed or Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = run_dir(cfg, args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    if not args.compare_only:
        with torch.no_grad():  # token count of the encoder stack
            n_tokens = build_multitask(cfg).ecg_encoder(torch.zeros(1, data["ecg"].shape[1])).shape[1]
        aux = build_aux_targets(data, cfg, n_tokens)
        weights = {"mt_noaux": ZERO_WEIGHTS,
                   "mt_aux": {"type": args.w_type, "quality": args.w_quality, "beat": args.w_beat}}
        print(f"device={device} seeds={args.seeds} n={len(data['label'])} tokens={n_tokens} weights={weights}")
        for name in args.arms.split(","):
            results[name] = run_variant(name, weights[name], cfg, data, aux, device, args.seeds, out_dir,
                                        not args.no_resume)
            results_path.write_text(json.dumps(results, indent=2, default=str))
            s = results[name]
            print(f"== {name:9s} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f} "
                  f"Sens={s['sensitivity_mean']:.3f} Spec={s['specificity_mean']:.3f}")
    if "mt_aux" in results and "mt_noaux" in results:
        g = compare(out_dir, results, data["record_id"])
        (out_dir / "guardrail.json").write_text(json.dumps(g, indent=2, default=float))
        for k in ("f1", "auc", "sensitivity", "specificity"):
            r = g[k]
            print(f"{k:12s} noaux={r['mean_noaux']:.3f} aux={r['mean_aux']:.3f} diff={r['mean_diff']:+.3f} "
                  f"p(NB)={r['p_value']:.3f}")
        print("bootstrap F1@0.5:", g["record_bootstrap_f1_t50"])
        print("aux head test metrics:", json.dumps(g["aux_head_test_metrics_mean"], indent=1))


if __name__ == "__main__":
    main()
