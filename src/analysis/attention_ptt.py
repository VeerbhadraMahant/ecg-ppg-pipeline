"""Does cross-attention track measured pulse transit time? (updates.md 6.2)

For each test window (seed-0 cross_attention checkpoints):
  1. measured PTT: detect R peaks on the ECG and pulse feet on the PPG with the
     detectors in src/features.py; per-beat PTT = delay from each R peak to the
     first foot in [0.08, 0.6] s; window PTT = median over beats.
  2. attention-implied lag: the model's ECG->PPG attention (heads averaged)
     is a T'_e x T'_p matrix; with token stride s = pool^n_blocks samples, the
     lag distribution is the attention mass at each token offset j - i.
     Window lag = the offset with the most mass (restricted to 0..1 s).
  3. Compare: Spearman correlation across windows between attention-implied lag
     and measured PTT, plus the share of attention mass inside the
     physiological lag range versus what uniform attention would put there.
A permutation control (shuffle measured PTT across windows) gives the null.

Usage:
    python -m src.analysis.attention_ptt --run-name challenge2015_ppg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.features import FS, detect_r_peaks, detect_ppg_peaks, ppg_feet, ptt_series  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult, iter_splits, run_dir  # noqa: E402


@torch.no_grad()
def attention_for(model, ecg, ppg, device, bs=64):
    out = []
    for i in range(0, len(ecg), bs):
        _, w = model(torch.from_numpy(ecg[i:i + bs]).to(device), torch.from_numpy(ppg[i:i + bs]).to(device))
        out.append(w.cpu().numpy())
    return np.concatenate(out)  # (N, Te, Tp)


def lag_profile(w: np.ndarray, max_lag_tok: int):
    """Mean attention mass at token offsets d = j - i for d in [-max_lag_tok, max_lag_tok]."""
    te, tp = w.shape
    prof = np.zeros(2 * max_lag_tok + 1)
    for d in range(-max_lag_tok, max_lag_tok + 1):
        diag = np.diagonal(w, offset=d)
        prof[d + max_lag_tok] = diag.sum() / te
    return prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--variant", default="cross_attention")
    ap.add_argument("--protocol", default="cv")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    processed = args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    data = load_processed(processed)
    run_path = run_dir(cfg, args.run_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ecg, ppg, y = data["ecg"], data["ppg"], data["label"].astype(int)

    stride = cfg["model"]["cnn_pool"] ** len(cfg["model"]["cnn_channels"])  # samples per token
    tok_s = stride / FS
    max_lag_tok = int(round(1.0 / tok_s))
    phys_lo, phys_hi = int(round(0.08 / tok_s)), int(round(0.6 / tok_s))

    rows = []
    mass_in, chance_in = [], []
    for fold, fit, val, test in iter_splits(data, cfg, 0, args.protocol):
        model = build_model(args.variant, cfg, get_width_mult(args.variant, ROOT)).to(device)
        model.load_state_dict(torch.load(run_path / "checkpoints" / f"{args.variant}_seed0_fold{fold}.pt", map_location=device))
        model.eval()
        W = attention_for(model, ecg[test], ppg[test], device)
        for k, idx in enumerate(test):
            w = W[k]
            prof = lag_profile(w, max_lag_tok)
            lags = np.arange(-max_lag_tok, max_lag_tok + 1)
            pos = lags >= 0
            d_star = lags[pos][np.argmax(prof[pos])]

            r = detect_r_peaks(ecg[idx])
            pk = detect_ppg_peaks(ppg[idx])
            feet = ppg_feet(ppg[idx], pk)
            ptt = ptt_series(r, feet)
            # per-beat attention lag at the ECG token of each R peak (token resolution)
            beat_pairs = []
            for rp in r:
                i = rp // stride
                if i >= w.shape[0]:
                    continue
                seg = w[i, i:i + max_lag_tok + 1]
                if seg.size == 0:
                    continue
                near = (feet - rp)[(feet - rp >= 0.08 * FS) & (feet - rp <= 0.6 * FS)]
                if len(near):
                    beat_pairs.append((np.argmax(seg) * tok_s, near.min() / FS))

            te, tp = w.shape
            in_mass = float(prof[(lags >= phys_lo) & (lags <= phys_hi)].sum())
            uni = np.full_like(w, 1.0 / tp)
            uni_prof = lag_profile(uni, max_lag_tok)
            chance = float(uni_prof[(lags >= phys_lo) & (lags <= phys_hi)].sum())
            mass_in.append(in_mass); chance_in.append(chance)
            rows.append({"idx": int(idx), "fold": fold, "label": int(y[idx]), "attn_lag_s": d_star * tok_s,
                         "ptt_med_s": float(np.median(ptt)) if len(ptt) else np.nan, "n_beats_ptt": len(ptt),
                         "mass_in_phys_range": in_mass, "mass_if_uniform": chance,
                         "beat_pairs": beat_pairs})
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["ptt_med_s"])
    ok = ok[ok.n_beats_ptt >= 3]
    rho, p = stats.spearmanr(ok.attn_lag_s, ok.ptt_med_s)
    rng = np.random.RandomState(0)
    null = [stats.spearmanr(ok.attn_lag_s, rng.permutation(ok.ptt_med_s.values))[0] for _ in range(2000)]
    p_perm = float(np.mean(np.abs(null) >= abs(rho)))

    bp = np.array([b for lst in df.beat_pairs for b in lst])
    rho_b = stats.spearmanr(bp[:, 0], bp[:, 1])[0] if len(bp) > 10 else np.nan

    summary = {
        "variant": args.variant, "n_windows": int(len(df)), "n_windows_with_ptt": int(len(ok)),
        "token_resolution_ms": 1000 * tok_s,
        "window_level_spearman_rho(attn_lag, measured_PTT)": float(rho), "spearman_p": float(p),
        "permutation_p": p_perm, "n_beats_pairs": int(len(bp)), "beat_level_spearman_rho": float(rho_b),
        "attention_mass_in_physiological_lag_range": float(np.mean(mass_in)),
        "mass_if_attention_were_uniform": float(np.mean(chance_in)),
        "median_attn_lag_s": float(df.attn_lag_s.median()), "median_measured_ptt_s": float(ok.ptt_med_s.median()),
    }
    out = run_path / "attention_ptt"
    out.mkdir(exist_ok=True)
    df.drop(columns="beat_pairs").to_csv(out / "per_window.csv", index=False)
    pd.Series(summary).to_json(out / "summary.json", indent=2)
    for k, v in summary.items():
        print(f"{k}: {v}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(ok.ptt_med_s, ok.attn_lag_s, s=8, alpha=0.5)
    ax[0].set_xlabel("measured PTT (s)"); ax[0].set_ylabel("attention-implied lag (s)")
    ax[0].set_title(f"window level, rho={rho:.2f} (perm p={p_perm:.3f})")
    ax[1].hist(df.attn_lag_s, bins=30, alpha=0.6, label="attention lag")
    ax[1].hist(ok.ptt_med_s, bins=30, alpha=0.6, label="measured PTT")
    ax[1].legend(frameon=False); ax[1].set_xlabel("seconds")
    fig.tight_layout(); fig.savefig(out / "attention_vs_ptt.png", dpi=150)


if __name__ == "__main__":
    main()
