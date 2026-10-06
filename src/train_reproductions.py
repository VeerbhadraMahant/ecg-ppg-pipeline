"""Trainer for the published-method reproductions (updates.md 5.7).

Reuses the conventions of src/train.py: iter_splits (record-wise outer CV with
record-wise inner validation), binary_metrics / challenge_score /
operating_points, and per-fold preds at runs/<run>/preds/<variant>_seed{s}_fold{f}.{npz,json}
with keys test_idx, val_idx, val_prob, val_y, test_prob, test_y, plus a
results.json whose entries are compatible with src/evaluate.py.

Variants:
  mousavi_attn_cnn_rnn    plain (ecg, ppg) model, same loss as the rest of the repo
  alarm_type_contrastive  self-contained: alarm-type embedding as an extra input
                          + supervised-contrastive auxiliary loss. Alarm types are
                          handled here (index tensors), NOT via a model attribute.

Usage:
    python -m src.train_reproductions --variant mousavi_attn_cnn_rnn --seeds 2 --run-name reproductions_sanity
    python -m src.train_reproductions --variant alarm_type_contrastive --seeds 3 --run-name reproductions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.models.losses import build_loss  # noqa: E402
from src.models.reproductions import (  # noqa: E402
    REPRODUCTIONS,
    build_alarm_type_contrastive,
    build_reproduction,
    count_params,
    supcon_loss,
)
from src.train import iter_splits, run_dir, set_seed, subsample_labels  # noqa: E402

REPRO_VARIANTS = list(REPRODUCTIONS) + ["alarm_type_contrastive"]
CONTRASTIVE_DEFAULTS = {"weight": 0.5, "temperature": 0.1, "batch_size": 32}


def get_repro_width(variant: str, root: Path = ROOT) -> float:
    p = root / "configs" / "width_mult_reproductions.yaml"
    if p.exists():
        return float(yaml.safe_load(p.read_text()).get(variant, 1.0))
    return 1.0


def alarm_type_ids(data: dict) -> np.ndarray:
    """Stable integer ids from the sorted unique alarm-type strings."""
    names = sorted(set(map(str, data["alarm_type"])))
    lut = {n: i for i, n in enumerate(names)}
    return np.array([lut[str(a)] for a in data["alarm_type"]], dtype=np.int64)


def build_for(variant: str, cfg: dict, width_mult: float, n_types: int = 5) -> torch.nn.Module:
    if variant == "alarm_type_contrastive":
        m = build_alarm_type_contrastive(cfg, width_mult)
        if n_types != m.n_types:
            raise ValueError(f"data has {n_types} alarm types, model built for {m.n_types}")
        return m
    return build_reproduction(variant, cfg, width_mult)


def _forward(model, ecg, ppg, at, use_type):
    if use_type:
        return model.forward_with_embedding(ecg, ppg, at)
    return model(ecg, ppg)[0], None


@torch.no_grad()
def _predict(model, T, idx, device, use_type, bs=64):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = torch.as_tensor(idx[i:i + bs])
        lg, _ = _forward(model, T["ecg"][b].to(device), T["ppg"][b].to(device), T["at"][b].to(device), use_type)
        out.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(out)


def fit_fold(cfg, variant, width_mult, T, fit_idx, val_idx, test_idx, device, seed, selection="val"):
    """Train one fold.

    selection="val"  : strict repo protocol - best epoch / early stopping on val_idx
                       (record-wise inner holdout), test touched once at the end.
    selection="test" : emulates leaky published protocols - best epoch / early
                       stopping chosen on the TEST fold itself (val_idx ignored;
                       val preds are set equal to test preds).
    """
    t = cfg["train"]
    use_type = variant == "alarm_type_contrastive"
    cc = {**CONTRASTIVE_DEFAULTS, **cfg.get("contrastive", {})}
    bs = cc["batch_size"] if use_type else t["batch_size"]
    sel_idx = val_idx if selection == "val" else test_idx

    model = build_for(variant, cfg, width_mult).to(device)
    n_params = count_params(model)
    y_fit = T["y"][fit_idx]
    n_pos = float(y_fit.sum())
    pos_weight = (len(fit_idx) - n_pos) / n_pos if n_pos > 0 else 1.0
    loss_fn = build_loss(t["loss"], pos_weight, t.get("focal_gamma", 2.0), device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    gen = torch.Generator().manual_seed(seed)

    best_f1, best_state, patience = -1.0, None, t["early_stop_patience"]
    for _ in range(t["epochs"]):
        model.train()
        perm = torch.as_tensor(fit_idx)[torch.randperm(len(fit_idx), generator=gen)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            ecg, ppg, y, at = (T["ecg"][b].to(device), T["ppg"][b].to(device),
                               T["y"][b].to(device), T["at"][b].to(device))
            opt.zero_grad()
            logits, z = _forward(model, ecg, ppg, at, use_type)
            loss = loss_fn(logits, y)
            if use_type:
                loss = loss + cc["weight"] * supcon_loss(z, y, cc["temperature"])
            loss.backward()
            opt.step()
        p_sel = _predict(model, T, sel_idx, device, use_type)
        f1 = binary_metrics(T["y"][sel_idx].numpy(), p_sel)["f1"]
        if f1 > best_f1:
            best_f1, patience = f1, t["early_stop_patience"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break

    model.load_state_dict(best_state)
    test_prob = _predict(model, T, test_idx, device, use_type)
    test_y = T["y"][test_idx].numpy()
    if selection == "val":
        val_prob = _predict(model, T, val_idx, device, use_type)
        val_y = T["y"][val_idx].numpy()
    else:
        val_prob, val_y = test_prob, test_y
    m = binary_metrics(test_y, test_prob)
    m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
    m["ops"] = operating_points(val_y, val_prob, test_y, test_prob)
    m["params"] = n_params
    return m, {"val_prob": val_prob, "val_y": val_y, "test_prob": test_prob, "test_y": test_y}


def make_tensors(data: dict) -> dict:
    return {
        "ecg": torch.from_numpy(data["ecg"].astype(np.float32)),
        "ppg": torch.from_numpy(data["ppg"].astype(np.float32)),
        "y": torch.from_numpy(data["label"].astype(np.float32)),
        "at": torch.from_numpy(alarm_type_ids(data)),
    }


def run_reproduction(variant, cfg, data, device, n_seeds, out_dir, resume=True, protocol="cv") -> dict:
    width_mult = get_repro_width(variant)
    T = make_tensors(data)
    all_m = []
    for seed in range(n_seeds):
        set_seed(cfg["seed"] + seed)
        for fold_i, fit_idx, val_idx, test_idx in iter_splits(data, cfg, seed, protocol):
            fit_idx = subsample_labels(fit_idx, data, cfg["train"].get("label_fraction", 1.0), cfg["seed"] + seed + fold_i)
            pred_path = out_dir / "preds" / f"{variant}_seed{seed}_fold{fold_i}.npz"
            m_path = pred_path.with_suffix(".json")
            if resume and pred_path.exists() and m_path.exists():
                all_m.append(json.loads(m_path.read_text()))
                continue
            t0 = time.time()
            m, preds = fit_fold(cfg, variant, width_mult, T, fit_idx, val_idx, test_idx, device, cfg["seed"] + seed + fold_i)
            m.update(seed=seed, fold=fold_i, train_seconds=time.time() - t0, n_train=len(fit_idx), n_test=len(test_idx))
            all_m.append(m)
            print(f"[{variant}] seed={seed} fold={fold_i} f1={m['f1']:.3f} auc={m['auc']:.3f} "
                  f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} params={m['params']:,} ({m['train_seconds']:.0f}s)")
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pred_path, test_idx=test_idx, val_idx=val_idx, **preds)
            m_path.write_text(json.dumps(m, default=float))
    summary = summarize_across_folds(all_m)
    summary.update(variant=variant, width_mult=width_mult, fold_metrics=all_m)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mousavi_attn_cnn_rnn", help=f"comma-separated from {REPRO_VARIANTS}")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--processed", default=None)
    ap.add_argument("--run-name", default="reproductions")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--protocol", default="cv", choices=["cv", "official"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    n_seeds = args.seeds or cfg["train"].get("n_seeds", 3)
    data = load_processed(args.processed or (ROOT / cfg["paths"]["processed_dir"] / "challenge2015_windows.npz"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} seeds: {n_seeds} n={len(data['label'])}")
    out_dir = run_dir(cfg, args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for v in args.variant.split(","):
        if v not in REPRO_VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {REPRO_VARIANTS}")
        results[v] = run_reproduction(v, cfg, data, device, n_seeds, out_dir, not args.no_resume, args.protocol)
        results_path.write_text(json.dumps(results, indent=2, default=str))
        s = results[v]
        print(f"== {v:24s} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} Sens={s['sensitivity_mean']:.3f} "
              f"Spec={s['specificity_mean']:.3f} AUC={s['auc_mean']:.3f}")
    print(f"results written to {results_path}")


if __name__ == "__main__":
    main()
