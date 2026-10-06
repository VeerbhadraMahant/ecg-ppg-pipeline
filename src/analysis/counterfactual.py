"""Counterfactual alignment tests (updates.md analysis 6.1).

Question: do fusion models use ECG-to-pulse TEMPORAL ALIGNMENT, or merely
treat the pulse as a patient-agnostic quality/presence cue?

For each held-out test fold (seed-0 checkpoint) we perturb ONE modality while
leaving the other intact and measure how much discrimination (AUC) and the
mean predicted probability change:

  swap_same_label   : the pulse (or ECG) from a DIFFERENT patient with the same
                      label and alarm type. Real waveform, same kind of
                      evidence, but its timing is unrelated to this ECG.
                      -> If the model needed alignment this should hurt;
                         if it only used a quality cue it should not.
  swap_other_label  : same, but from the opposite label (content test).
  shift_{x}s        : circular time shift of the pulse by +x seconds.
                      Shifts beyond the physiological PTT range (~0.1-0.6 s)
                      should hurt an alignment-using model.
  reverse           : time reversal (keeps spectrum, breaks causality).
  phase_scramble    : keeps the amplitude spectrum, randomises phase: a
                      pulse-like signal with destroyed timing.
  drop_25 / drop_50 : zero random 1 s chunks covering 25% / 50% of the window.
  noise             : pure Gaussian noise (no pulse content at all).

Derived indices (per model):
  alignment_index = AUC(identity) - AUC(swap_same_label)
  content_index   = AUC(swap_same_label) - AUC(swap_other_label)... see report
  quality_cue     = AUC(identity) - AUC(noise)

Usage:
    python -m src.analysis.counterfactual --run-name challenge2015_ppg \
        --variants ecg_only,ppg_only,concat,cross_attention
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult, iter_splits, run_dir  # noqa: E402

FS = 250


def perturb(x: np.ndarray, kind: str, rng: np.random.RandomState, pool=None) -> np.ndarray:
    """x: (N, T) signals of the modality being perturbed."""
    n, T = x.shape
    if kind == "identity":
        return x.copy()
    if kind.startswith("shift_"):
        k = int(round(float(kind[6:-1]) * FS))
        return np.roll(x, k, axis=1)
    if kind == "reverse":
        return x[:, ::-1].copy()
    if kind == "phase_scramble":
        f = np.fft.rfft(x, axis=1)
        ph = rng.uniform(0, 2 * np.pi, f.shape)
        ph[:, 0] = 0
        out = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=T, axis=1)
        return ((out - out.mean(1, keepdims=True)) / (out.std(1, keepdims=True) + 1e-8)).astype(np.float32)
    if kind in ("drop_25", "drop_50"):
        frac = 0.25 if kind == "drop_25" else 0.5
        out = x.copy()
        n_chunks = int(round(frac * (T // FS)))
        for i in range(n):
            for c in rng.choice(T // FS, size=n_chunks, replace=False):
                out[i, c * FS:(c + 1) * FS] = 0.0
        return out
    if kind == "noise":
        return rng.randn(n, T).astype(np.float32)
    if kind in ("swap_same_label", "swap_other_label"):
        src, y, rec, at = pool
        out = x.copy()
        for i in range(n):
            if kind == "swap_same_label":
                cand = np.where((y == y[i]) & (rec != rec[i]) & (at == at[i]))[0]
                if len(cand) == 0:
                    cand = np.where((y == y[i]) & (rec != rec[i]))[0]
            else:
                cand = np.where(y != y[i])[0]
            if len(cand):
                out[i] = src[rng.choice(cand)]
        return out
    raise ValueError(kind)


KINDS = ["identity", "swap_same_label", "swap_other_label", "shift_0.25s", "shift_0.5s", "shift_1s",
         "shift_2s", "reverse", "phase_scramble", "drop_25", "drop_50", "noise"]
STOCHASTIC = {"swap_same_label", "swap_other_label", "phase_scramble", "drop_25", "drop_50", "noise"}


@torch.no_grad()
def predict(model, ecg, ppg, device, bs=128):
    out = []
    for i in range(0, len(ecg), bs):
        lo, _ = model(torch.from_numpy(ecg[i:i + bs]).to(device), torch.from_numpy(ppg[i:i + bs]).to(device))
        out.append(torch.sigmoid(lo).cpu().numpy())
    return np.concatenate(out)


def run(cfg, run_path: Path, data, variant: str, target: str, device: str, n_draws: int = 5, seed: int = 0):
    ecg, ppg, y = data["ecg"], data["ppg"], data["label"].astype(int)
    rec, at = data["record_id"], data["alarm_type"]
    width = get_width_mult(variant, ROOT)
    rows = []
    for fold, fit, val, test in iter_splits(data, cfg, seed, cfg.get("_protocol", "cv")):
        ck = run_path / "checkpoints" / f"{variant}_seed{seed}_fold{fold}.pt"
        if not ck.exists():
            raise FileNotFoundError(ck)
        model = build_model(variant, cfg, width).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        e, p, yt = ecg[test], ppg[test], y[test]
        base = e if target == "ecg" else p
        pool = (base, yt, rec[test], at[test])
        for kind in KINDS:
            reps = n_draws if kind in STOCHASTIC else 1
            aucs, means = [], []
            for r in range(reps):
                rng = np.random.RandomState(1000 * fold + r)
                pert = perturb(base, kind, rng, pool).astype(np.float32)
                pe, pp = (pert, p) if target == "ecg" else (e, pert)
                prob = predict(model, pe, pp, device)
                aucs.append(roc_auc_score(yt, prob) if len(set(yt)) > 1 else np.nan)
                means.append(prob.mean())
            rows.append({"variant": variant, "perturbed": target, "fold": fold, "kind": kind,
                         "auc": float(np.nanmean(aucs)), "mean_prob": float(np.mean(means))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--variants", default="ecg_only,ppg_only,concat,cross_attention")
    ap.add_argument("--targets", default="ppg,ecg")
    ap.add_argument("--protocol", default="cv")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["_protocol"] = args.protocol
    processed = args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    data = load_processed(processed)
    run_path = run_dir(cfg, args.run_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = []
    for v in args.variants.split(","):
        for tgt in args.targets.split(","):
            if v == "ecg_only" and tgt == "ppg" or v == "ppg_only" and tgt == "ecg":
                continue  # the model never sees that modality
            frames.append(run(cfg, run_path, data, v, tgt, device))
            print(f"done {v} / perturb {tgt}", flush=True)
    df = pd.concat(frames)
    out = run_path / "counterfactual"
    out.mkdir(exist_ok=True)
    prev = out / "counterfactual_by_fold.csv"
    if prev.exists():  # merge with earlier runs; new results win for the same (variant, target)
        old = pd.read_csv(prev)
        keep = old[~old.set_index(["variant", "perturbed"]).index.isin(df.set_index(["variant", "perturbed"]).index)]
        df = pd.concat([keep, df])
    df.to_csv(prev, index=False)

    piv = df.groupby(["variant", "perturbed", "kind"]).auc.agg(["mean", "std"]).reset_index()
    table = piv.pivot_table(index=["variant", "perturbed"], columns="kind", values="mean")[KINDS].round(3)
    table.to_csv(out / "counterfactual_auc.csv")
    print("\nAUC under perturbation (mean over test folds; seed-0 checkpoints):")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string())

    idx = []
    for (v, t), g in df.groupby(["variant", "perturbed"]):
        a = g.groupby("kind").auc.mean()
        idx.append({"variant": v, "perturbed": t, "AUC_identity": a["identity"],
                    "alignment_index(identity-swap_same)": a["identity"] - a["swap_same_label"],
                    "timing_index(identity-phase_scramble)": a["identity"] - a["phase_scramble"],
                    "max_shift_drop": a["identity"] - a[["shift_0.25s", "shift_0.5s", "shift_1s", "shift_2s"]].min(),
                    "pulse_presence_cue(identity-noise)": a["identity"] - a["noise"]})
    idx = pd.DataFrame(idx).round(3)
    idx.to_csv(out / "alignment_indices.csv", index=False)
    print("\nIndices:")
    print(idx.to_string(index=False))


if __name__ == "__main__":
    main()
