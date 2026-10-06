"""Signal-quality estimator head (updates.md 5.6 / 7.1).

!! PSEUDO-LABELS ONLY. No human quality annotations exist for this data. The
per-channel target is a deterministic heuristic:

    sqi_c = signal_quality_score(x_c)  *  clip(template_corr_c, 0, 1)

  * signal_quality_score (src/data/signal_ops): 1 - 0.5*flatline_penalty
    - 0.5*clip_penalty, using the thresholds in configs/config.yaml `quality`.
  * template_corr_c: mean correlation of every beat (R peaks for ECG, PPG
    peaks for PPG, detectors from src/features.py) with the median beat
    (features._template_corr). It is 0 when fewer than 3 full beats are found
    (no usable rhythm), so an arrhythmic-but-clean strip is partly penalised.
    This conflates "noisy" with "irregular morphology / unreliable detector".

The CNN therefore learns to reproduce a heuristic SQI from raw samples (a
cheap, differentiable, detector-free surrogate); it does NOT measure agreement
with clinicians. Evaluation: record-wise CV (src.train.iter_splits) vs. a
trivial train-mean baseline.

Run: python -m src.heads.quality --seeds 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.data.signal_ops import signal_quality_score  # noqa: E402
from src.features import FS, _template_corr, detect_ppg_peaks, detect_r_peaks  # noqa: E402
from src.models.encoders import CNNEncoder  # noqa: E402


def channel_sqi(x: np.ndarray, peaks: np.ndarray, half_s: float, flat_thr: float, clip_thr: float, fs: int = FS) -> float:
    base = signal_quality_score(x, flat_thr, clip_thr)
    tc = _template_corr(x, peaks, fs, half_s, "c")["c_tmpl_corr"]
    return float(base * np.clip(tc, 0.0, 1.0))


def quality_targets(ecg: np.ndarray, ppg: np.ndarray, flat_thr: float = 0.01, clip_thr: float = 0.2,
                    fs: int = FS) -> np.ndarray:
    """(N, 2) float32 pseudo-labels in [0, 1]: columns [ecg, ppg]."""
    out = np.zeros((len(ecg), 2), dtype=np.float32)
    for i, (e, p) in enumerate(zip(ecg, ppg)):
        out[i, 0] = channel_sqi(e, detect_r_peaks(e, fs), 0.25, flat_thr, clip_thr, fs)
        out[i, 1] = channel_sqi(p, detect_ppg_peaks(p, fs), 0.4, flat_thr, clip_thr, fs)
    return out


class QualityNet(nn.Module):
    """One CNNEncoder per channel (reuses src.models.encoders.CNNEncoder),
    time-mean-pooled -> sigmoid scalar. Each channel only sees its own signal."""

    def __init__(self, channels, kernel_sizes, pool, dropout, width_mult=1.0):
        super().__init__()
        self.ecg_encoder = CNNEncoder(channels, kernel_sizes, pool, dropout, width_mult)
        self.ppg_encoder = CNNEncoder(channels, kernel_sizes, pool, dropout, width_mult)
        self.ecg_out = nn.Linear(self.ecg_encoder.out_dim, 1)
        self.ppg_out = nn.Linear(self.ppg_encoder.out_dim, 1)

    def forward(self, ecg, ppg):
        e = self.ecg_out(self.ecg_encoder(ecg).mean(1))
        p = self.ppg_out(self.ppg_encoder(ppg).mean(1))
        return torch.sigmoid(torch.cat([e, p], dim=1))  # (B, 2)


def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(("ecg", "ppg")):
        yt, pt = y[:, j], p[:, j]
        ss = ((yt - yt.mean()) ** 2).sum()
        out[name] = {
            "mae": float(np.abs(yt - pt).mean()),
            "r2": float(1 - ((yt - pt) ** 2).sum() / ss) if ss > 0 else float("nan"),
            "pearson": float(pearsonr(yt, pt)[0]) if np.std(pt) > 0 and np.std(yt) > 0 else 0.0,
            "spearman": float(spearmanr(yt, pt)[0]) if np.std(pt) > 0 and np.std(yt) > 0 else 0.0,
        }
    return out


def _fit_fold(cfg, ecg, ppg, y, fit, val, test, device, epochs, patience=10):
    m = cfg["model"]
    net = QualityNet(m["cnn_channels"], m["cnn_kernel_sizes"], m["cnn_pool"], m["dropout"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    T = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)  # noqa: E731
    E, P, Y = T(ecg), T(ppg), T(y)
    bs = cfg["train"]["batch_size"]

    def pred(idx):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(E[idx[i:i + 128]], P[idx[i:i + 128]]) for i in range(0, len(idx), 128)]).cpu().numpy()

    best, best_state, left = 1e9, None, patience
    fit_t = torch.from_numpy(fit).to(device)
    val_t = torch.from_numpy(val).to(device)
    for _ in range(epochs):
        net.train()
        perm = fit_t[torch.randperm(len(fit_t), device=device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(E[b], P[b]), Y[b])
            loss.backward()
            opt.step()
        v = float(((pred(val_t) - y[val]) ** 2).mean())
        if v < best:
            best, left = v, patience
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                break
    net.load_state_dict(best_state)
    return pred(torch.from_numpy(test).to(device)), pred(val_t)


def run(seeds: int = 1, epochs: int = 60, processed: str | None = None, out: str | None = None) -> dict:
    from src.train import iter_splits, set_seed

    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    data = load_processed(processed or ROOT / cfg["paths"]["processed_dir"] / "challenge2015_windows.npz")
    q = cfg["quality"]
    y = quality_targets(data["ecg"], data["ppg"], q["flatline_std_threshold"], q["clip_fraction_threshold"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for seed in range(seeds):
        set_seed(cfg["seed"] + seed)
        for fold, fit, val, test in iter_splits(data, cfg, seed, "cv"):
            pt, _ = _fit_fold(cfg, data["ecg"], data["ppg"], y, fit, val, test, device, epochs)
            base = np.tile(y[fit].mean(0), (len(test), 1))  # trivial baseline: train-mean
            rows.append({"seed": seed, "fold": fold, "cnn": reg_metrics(y[test], pt),
                         "mean_baseline": reg_metrics(y[test], base)})
            print(f"seed={seed} fold={fold} ecg r={rows[-1]['cnn']['ecg']['pearson']:.2f} "
                  f"ppg r={rows[-1]['cnn']['ppg']['pearson']:.2f}")
    summ = {}
    for model in ("cnn", "mean_baseline"):
        for ch in ("ecg", "ppg"):
            for k in ("mae", "r2", "pearson", "spearman"):
                v = np.array([r[model][ch][k] for r in rows])
                summ[f"{model}.{ch}.{k}"] = [float(np.nanmean(v)), float(np.nanstd(v))]
    res = {
        "label_rule": "sqi = signal_quality_score * clip(template_corr,0,1); PSEUDO-labels, no human annotation",
        "target_stats": {"ecg_mean": float(y[:, 0].mean()), "ppg_mean": float(y[:, 1].mean()),
                         "ecg_std": float(y[:, 0].std()), "ppg_std": float(y[:, 1].std())},
        "corr_with_dataset_quality": {
            "ecg": float(np.corrcoef(y[:, 0], data["ecg_quality"])[0, 1]) if np.std(data["ecg_quality"]) > 0 else None,
            "ppg": float(np.corrcoef(y[:, 1], data["pulse_quality"])[0, 1]) if np.std(data["pulse_quality"]) > 0 else None},
        "summary_mean_std": summ, "folds": rows,
    }
    out = Path(out or ROOT / "runs" / "heads" / "quality" / "results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    for k, v in summ.items():
        print(f"{k:28s} {v[0]:+.3f} +/- {v[1]:.3f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    run(a.seeds, a.epochs)
