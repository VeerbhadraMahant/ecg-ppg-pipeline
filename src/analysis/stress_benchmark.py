"""Synthetic stress benchmark + missing-sensor robustness (updates.md 6.10, 6.9).

Takes CLEAN (high quality-score) held-out test windows and perturbs them in
controlled ways, then evaluates every variant that has seed-0 fold checkpoints
(auto-discovered) on the perturbed test folds. Checkpoints are NOT retrained:
the models were trained on the original data only, so they cannot have learned
the injection signature (see `signature_check`).

Perturbation families (all applied AFTER the pipeline's own filtering and
z-normalisation, then the window is z-normalised again like the pipeline does):

  ecg_noise   REAL noise: MIT-BIH Noise Stress Test Database (nstdb; PhysioNet)
              em = electrode motion, ma = muscle artefact, bw = baseline
              wander; 360 Hz -> resampled to 250 Hz. Added at a controlled SNR
              (signal power / noise power over the window, dB).
  ppg_noise   SYNTHETIC motion artefact (there is no real PPG noise database
              here): Gaussian noise band-limited to 0.5-5 Hz, scaled to the SNR.
  both_noise  ecg_noise (each nstdb type) + synthetic ppg_noise at the same SNR.
  ptt_const   PPG delayed by 0-400 ms relative to the ECG (constant shift).
  ptt_jitter  PPG time axis warped by a random-walk delay whose std at the end
              of the window is the stated level (ms).
  dropout     one modality zeroed ("zero") or held at its last value
              ("flatline") over a contiguous segment covering 0..100% of the
              window (random position).
  missing     whole modality absent at inference: replaced by zeros
              ("missing_zero") or by N(0,1) noise ("missing_noise"). Single-
              signal variants ignore the other input, so ecg_only is invariant
              to a missing PPG (and ppg_only to a missing ECG) by construction;
              the converse gives chance-level AUC.

Usage:
    python -m src.analysis.stress_benchmark --run-name challenge2015_ppg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import signal as sp
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.signal_ops import resample_signal, z_normalize  # noqa: E402

FS = 250
SNR_LEVELS = [24, 18, 12, 6, 0]
ECG_NOISE_TYPES = ["em", "ma", "bw"]
PTT_MS = [0, 50, 100, 200, 300, 400]
JITTER_MS = [25, 50, 100, 200]
DROP_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
NSTDB_DIR = ROOT / "data" / "raw" / "nstdb"

# --------------------------------------------------------------------------
# Noise sources
# --------------------------------------------------------------------------


def load_nstdb(directory: Path = NSTDB_DIR, fs_out: int = FS) -> dict[str, np.ndarray]:
    """Return {type: (2, L) float array at fs_out Hz} for bw / em / ma.

    Downloads via wfdb (pn_dir='nstdb') when files are absent."""
    import wfdb

    directory = Path(directory)
    bank = {}
    for name in ECG_NOISE_TYPES:
        if not (directory / f"{name}.dat").exists():
            directory.mkdir(parents=True, exist_ok=True)
            wfdb.dl_files("nstdb", str(directory), [f"{name}.dat", f"{name}.hea"])
        rec = wfdb.rdrecord(str(directory / name))
        chans = [resample_signal(rec.p_signal[:, c], rec.fs, fs_out) for c in range(rec.p_signal.shape[1])]
        bank[name] = np.stack(chans).astype(np.float32)
    return bank


def _scale_to_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` (n, T) row-wise so that 10log10(P_x/P_noise) = snr_db."""
    noise = noise - noise.mean(axis=1, keepdims=True)
    px = np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1, keepdims=True)
    pn = np.mean(noise ** 2, axis=1, keepdims=True) + 1e-12
    return noise * np.sqrt(px / (pn * 10 ** (snr_db / 10.0)))


def sample_nstdb_noise(bank: dict, kind: str, n: int, T: int, rng: np.random.RandomState) -> np.ndarray:
    src = bank[kind]
    out = np.empty((n, T), dtype=np.float32)
    for i in range(n):
        ch = rng.randint(src.shape[0])
        s = rng.randint(0, src.shape[1] - T)
        out[i] = src[ch, s:s + T] * rng.choice([-1.0, 1.0])
    return out


def synth_ppg_motion(n: int, T: int, rng: np.random.RandomState, lo=0.5, hi=5.0, fs=FS) -> np.ndarray:
    """SYNTHETIC motion artefact: white Gaussian noise band-limited to lo-hi Hz."""
    w = rng.randn(n, T)
    f = np.fft.rfft(w, axis=1)
    fr = np.fft.rfftfreq(T, 1 / fs)
    f[:, (fr < lo) | (fr > hi)] = 0
    return np.fft.irfft(f, n=T, axis=1).astype(np.float32)


def _renorm(x: np.ndarray) -> np.ndarray:
    return np.stack([z_normalize(r) for r in x]).astype(np.float32)


def inject_ecg_noise(ecg, bank, kind, snr_db, rng, return_noise=False):
    noise = _scale_to_snr(ecg, sample_nstdb_noise(bank, kind, *ecg.shape, rng), snr_db)
    out = _renorm(ecg + noise)
    return (out, noise) if return_noise else out


def inject_ppg_noise(ppg, snr_db, rng, return_noise=False):
    noise = _scale_to_snr(ppg, synth_ppg_motion(*ppg.shape, rng), snr_db)
    out = _renorm(ppg + noise)
    return (out, noise) if return_noise else out


def shift_ppg(ppg: np.ndarray, delay_ms: float) -> np.ndarray:
    """Delay the PPG by a constant (PPG lags ECG more). Edge-padded, re-normalised."""
    k = int(round(delay_ms / 1000 * FS))
    if k == 0:
        return ppg.copy()
    out = np.concatenate([np.repeat(ppg[:, :1], k, axis=1), ppg[:, :-k]], axis=1)
    return _renorm(out)


def jitter_ppg(ppg: np.ndarray, end_std_ms: float, rng: np.random.RandomState) -> np.ndarray:
    """Time-warp the PPG with a random-walk delay d(t), std(d(T)) = end_std_ms."""
    n, T = ppg.shape
    step = (end_std_ms / 1000 * FS) / np.sqrt(T)
    d = np.cumsum(rng.randn(n, T) * step, axis=1)  # samples
    d = np.clip(d, -0.8 * FS, 0.8 * FS)
    t = np.arange(T)[None, :] - d
    out = np.stack([np.interp(t[i], np.arange(T), ppg[i]) for i in range(n)])
    return _renorm(out)


def dropout(x: np.ndarray, frac: float, mode: str, rng: np.random.RandomState) -> np.ndarray:
    """Zero (or hold-last-value) a contiguous segment covering `frac` of the window."""
    n, T = x.shape
    L = int(round(frac * T))
    out = x.copy()
    if L == 0:
        return out
    for i in range(n):
        s = rng.randint(0, T - L + 1)
        out[i, s:s + L] = 0.0 if (mode == "zero" or s == 0) else x[i, s - 1]
    return out


# --------------------------------------------------------------------------
# Condition grid
# --------------------------------------------------------------------------


def build_conditions(bank: dict) -> list[dict]:
    """Each condition: family, target, kind, level, fn(e, p, rng) -> (e', p')."""
    C = []

    def add(family, target, kind, level, fn, stochastic=True):
        C.append(dict(family=family, target=target, kind=kind, level=level, fn=fn, stochastic=stochastic))

    add("clean", "none", "clean", np.nan, lambda e, p, r: (e, p), False)
    for snr in SNR_LEVELS:
        for k in ECG_NOISE_TYPES:
            add("ecg_noise", "ecg", k, snr, lambda e, p, r, k=k, s=snr: (inject_ecg_noise(e, bank, k, s, r), p))
            add("both_noise", "both", k, snr,
                lambda e, p, r, k=k, s=snr: (inject_ecg_noise(e, bank, k, s, r), inject_ppg_noise(p, s, r)))
        add("ppg_noise", "ppg", "synthetic_motion", snr, lambda e, p, r, s=snr: (e, inject_ppg_noise(p, s, r)))
    for ms in PTT_MS:
        add("ptt_const", "ppg", "shift", ms, lambda e, p, r, ms=ms: (e, shift_ppg(p, ms)), False)
    for ms in JITTER_MS:
        add("ptt_jitter", "ppg", "random_walk", ms, lambda e, p, r, ms=ms: (e, jitter_ppg(p, ms, r)))
    for mode in ["zero", "flatline"]:
        for fr in DROP_FRACS:
            add("dropout", "ecg", mode, fr, lambda e, p, r, m=mode, f=fr: (dropout(e, f, m, r), p))
            add("dropout", "ppg", mode, fr, lambda e, p, r, m=mode, f=fr: (e, dropout(p, f, m, r)))
    add("missing", "ecg", "missing_zero", 1.0, lambda e, p, r: (np.zeros_like(e), p), False)
    add("missing", "ppg", "missing_zero", 1.0, lambda e, p, r: (e, np.zeros_like(p)), False)
    add("missing", "ecg", "missing_noise", 1.0, lambda e, p, r: (r.randn(*e.shape).astype(np.float32), p))
    add("missing", "ppg", "missing_noise", 1.0, lambda e, p, r: (e, r.randn(*p.shape).astype(np.float32)))
    return C


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def discover_variants(run_path: Path, seed: int = 0, n_folds: int = 5) -> list[str]:
    names = set()
    for f in (run_path / "checkpoints").glob(f"*_seed{seed}_fold0.pt"):
        names.add(f.name[: -len(f"_seed{seed}_fold0.pt")])
    return sorted(v for v in names
                  if all((run_path / "checkpoints" / f"{v}_seed{seed}_fold{k}.pt").exists() for k in range(n_folds)))


def _metrics(y, prob):
    auc = roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan
    return auc, f1_score(y, (prob >= 0.5).astype(int), zero_division=0)


def clean_test_idx(data, test, q_thr):
    return test[data["quality"][test] >= q_thr]


def run(cfg, run_path, data, variants, device, n_draws=3, seed=0, q_thr=0.97, bank=None,
        max_conditions=None, verbose=True):
    import torch

    from src.analysis.counterfactual import predict
    from src.models.classifier import build_model
    from src.train import get_width_mult, iter_splits

    bank = bank or load_nstdb()
    conds = build_conditions(bank)[:max_conditions]
    ecg, ppg, y = data["ecg"], data["ppg"], data["label"].astype(int)
    rows, sigstore = [], {}
    for fold, _fit, _val, test in iter_splits(data, cfg, seed, "cv"):
        idx = clean_test_idx(data, test, q_thr)
        e0, p0, yt = ecg[idx], ppg[idx], y[idx]
        models = {}
        for v in variants:
            try:
                m = build_model(v, cfg, get_width_mult(v, ROOT)).to(device)
                m.load_state_dict(torch.load(run_path / "checkpoints" / f"{v}_seed{seed}_fold{fold}.pt", map_location=device))
                models[v] = m.eval()
            except Exception as ex:  # noqa: BLE001 - e.g. multimodal needs other inputs
                if fold == 0:
                    print(f"[skip] {v}: {type(ex).__name__}: {str(ex)[:80]}", flush=True)
        for ci, c in enumerate(conds):
            for d in range(n_draws if c["stochastic"] else 1):
                rng = np.random.RandomState(100003 * fold + 1009 * ci + d)  # same perturbation for all variants
                e1, p1 = c["fn"](e0, p0, rng)
                e1, p1 = np.ascontiguousarray(e1, dtype=np.float32), np.ascontiguousarray(p1, dtype=np.float32)
                for v, m in models.items():
                    prob = predict(m, e1, p1, device)
                    auc, f1 = _metrics(yt, prob)
                    rows.append(dict(variant=v, fold=fold, family=c["family"], target=c["target"], kind=c["kind"],
                                     level=c["level"], draw=d, n=len(yt), n_pos=int(yt.sum()), auc=auc, f1=f1,
                                     mean_prob=float(prob.mean())))
        if verbose:
            print(f"fold {fold} done ({len(idx)} clean test windows, {len(models)} variants)", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over draws within fold, then mean/std over folds."""
    key = ["variant", "family", "target", "kind", "level"]
    per_fold = df.groupby(key + ["fold"], dropna=False)[["auc", "f1", "mean_prob"]].mean().reset_index()
    g = per_fold.groupby(key, dropna=False)
    out = g.agg(auc=("auc", "mean"), auc_std=("auc", "std"), f1=("f1", "mean"), f1_std=("f1", "std"),
                mean_prob=("mean_prob", "mean"), n_folds=("fold", "nunique")).reset_index()
    return out


def degradation_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Per variant: clean AUC and AUC drop at each headline stress."""
    clean = summ[summ.family == "clean"].set_index("variant").auc
    rows = []
    for v, c in clean.items():
        s = summ[summ.variant == v]

        def auc(fam, tgt, kind=None, lvl=None):
            q = s[(s.family == fam) & (s.target == tgt)]
            if kind is not None:
                q = q[q.kind == kind]
            if lvl is not None:
                q = q[q.level == lvl]
            return float(q.auc.mean()) if len(q) else np.nan

        r = {"variant": v, "clean_auc": c,
             "ecg_noise_0dB(mean em,ma,bw)": auc("ecg_noise", "ecg", lvl=0),
             "ecg_noise_6dB": auc("ecg_noise", "ecg", lvl=6),
             "ppg_noise_0dB": auc("ppg_noise", "ppg", lvl=0),
             "ppg_noise_6dB": auc("ppg_noise", "ppg", lvl=6),
             "both_noise_0dB": auc("both_noise", "both", lvl=0),
             "both_noise_6dB": auc("both_noise", "both", lvl=6),
             "ptt_400ms": auc("ptt_const", "ppg", lvl=400),
             "jitter_200ms": auc("ptt_jitter", "ppg", lvl=200),
             "ppg_dropout50_zero": auc("dropout", "ppg", "zero", 0.5),
             "ecg_dropout50_zero": auc("dropout", "ecg", "zero", 0.5),
             "missing_ppg_zero": auc("missing", "ppg", "missing_zero"),
             "missing_ecg_zero": auc("missing", "ecg", "missing_zero")}
        # graceful-degradation score: mean AUC across the noise grid (all SNRs, ecg/ppg/both noise)
        nz = s[s.family.isin(["ecg_noise", "ppg_noise", "both_noise"])]
        r["mean_auc_all_noise"] = float(nz.auc.mean())
        r["mean_drop_all_noise"] = c - r["mean_auc_all_noise"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("mean_drop_all_noise")


# --------------------------------------------------------------------------
# Injection-signature / sanity check
# --------------------------------------------------------------------------


def spectral_features(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Cheap per-window spectral features: relative band powers, entropy, kurtosis."""
    from scipy.stats import kurtosis

    f, p = sp.welch(x, fs=fs, nperseg=512, axis=1)
    p = p / (p.sum(1, keepdims=True) + 1e-12)
    edges = [(0, 0.5), (0.5, 5), (5, 15), (15, 40), (40, 125.1)]
    bands = np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in edges], axis=1)
    ent = -(p * np.log(p + 1e-12)).sum(1) / np.log(p.shape[1])
    return np.column_stack([bands, ent, kurtosis(x, axis=1)])


def _cv_auc(X, y, groups, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def signature_check(cfg, run_path, data, variants, device, bank, snr_db=12, seed=0, q_thr=0.97):
    """(a) labels unchanged; (b) injected-vs-clean separability from model
    probabilities vs spectral features (grouped-by-record CV logistic
    regression); (c) the injected noise itself carries no label information."""
    import torch

    from src.analysis.counterfactual import predict
    from src.models.classifier import build_model
    from src.train import get_width_mult, iter_splits

    ecg, ppg, y = data["ecg"], data["ppg"], data["label"].astype(int)
    y_before = y.copy()
    E, P, Y, R, noise_e, noise_p = [], [], [], [], [], []
    probs = {v: {"clean": [], "inj": []} for v in variants}
    for fold, _f, _v, test in iter_splits(data, cfg, seed, "cv"):
        idx = clean_test_idx(data, test, q_thr)
        rng = np.random.RandomState(7 + fold)
        e1, ne = inject_ecg_noise(ecg[idx], bank, "em", snr_db, rng, return_noise=True)
        p1, npp = inject_ppg_noise(ppg[idx], snr_db, rng, return_noise=True)
        E.append((ecg[idx], e1)); P.append((ppg[idx], p1)); Y.append(y[idx]); R.append(data["record_id"][idx])
        noise_e.append(ne); noise_p.append(npp)
        for v in variants:
            m = build_model(v, cfg, get_width_mult(v, ROOT)).to(device)
            m.load_state_dict(torch.load(run_path / "checkpoints" / f"{v}_seed{seed}_fold{fold}.pt", map_location=device))
            m.eval()
            probs[v]["clean"].append(predict(m, ecg[idx], ppg[idx], device))
            probs[v]["inj"].append(predict(m, e1, p1, device))
    Yc, Rc = np.concatenate(Y), np.concatenate(R)
    ec, ei = (np.concatenate([a for a, _ in E]), np.concatenate([b for _, b in E]))
    pc, pi_ = (np.concatenate([a for a, _ in P]), np.concatenate([b for _, b in P]))
    out = {"snr_db": snr_db, "n_windows": int(len(Yc)),
           "labels_unchanged": bool(np.array_equal(y_before, data["label"].astype(int))),
           "injection_functions_take_no_labels": True}

    spec_c = np.hstack([spectral_features(ec), spectral_features(pc)])
    spec_i = np.hstack([spectral_features(ei), spectral_features(pi_)])
    z = np.r_[np.zeros(len(Yc)), np.ones(len(Yc))]
    grp = np.r_[Rc, Rc]
    out["spectral_only_auc"] = _cv_auc(np.vstack([spec_c, spec_i]), z, grp)
    for v in variants:
        pcl, pin = np.concatenate(probs[v]["clean"]), np.concatenate(probs[v]["inj"])
        lg = lambda p: np.log((p + 1e-6) / (1 - p + 1e-6))  # noqa: E731
        Xp = np.r_[lg(pcl), lg(pin)][:, None]
        out[f"{v}/prob_only_auc"] = _cv_auc(Xp, z, grp)
        out[f"{v}/prob+spectral_auc"] = _cv_auc(np.hstack([Xp, np.vstack([spec_c, spec_i])]), z, grp)
        out[f"{v}/mean_prob_clean"] = float(pcl.mean())
        out[f"{v}/mean_prob_injected"] = float(pin.mean())
    # (c) does the injected noise component alone predict the label? (should be ~0.5)
    ne, npp = np.concatenate(noise_e), np.concatenate(noise_p)
    Xn = np.hstack([spectral_features(ne), spectral_features(npp)])
    if len(set(Yc)) > 1:
        out["noise_component_vs_label_auc"] = _cv_auc(Xn, Yc, Rc)
    out["provenance"] = ("NSTDB noise = recordings of electrode-motion/baseline-wander/muscle noise made by "
                         "physically recording noise on volunteers with ECG leads, then added to clean MIT-BIH "
                         "Arrhythmia DB records 118/119 (physionet.org/content/nstdb). It is a different database "
                         "with different subjects, acquisition hardware and sampling rate (360 Hz) from the "
                         "PhysioNet/CinC 2015 ICU alarm records, so no CinC record contributes noise segments. "
                         "PPG artefact is synthetic Gaussian noise (0.5-5 Hz) with no source recording at all.")
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def make_plots(summ: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted(summ.variant.unique())
    cmap = plt.get_cmap("tab20")
    col = {v: cmap(i % 20) for i, v in enumerate(variants)}
    clean = summ[summ.family == "clean"].set_index("variant")

    def panel(ax, q, xcol, title, xlabel, metric="auc", invert=False):
        for v in variants:
            g = q[q.variant == v].groupby(xcol)[metric].mean().sort_index()
            if len(g):
                ax.plot(g.index, g.values, marker="o", ms=3, lw=1.4, color=col[v], label=v)
        ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel); ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        if invert:
            ax.invert_xaxis()

    for metric in ["auc", "f1"]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        panels = [("ecg_noise", "ecg", "ecg_noise (NSTDB, mean of em/ma/bw)"),
                  ("ppg_noise", "ppg", "ppg_noise (SYNTHETIC motion)"),
                  ("both_noise", "both", "both: NSTDB ECG + synthetic PPG")]
        for ax, (fam, tgt, t) in zip(axs[0], panels):
            panel(ax, summ[(summ.family == fam) & (summ.target == tgt)], "level", t, "SNR (dB)", metric, True)
        for ax, k in zip(axs[1], ECG_NOISE_TYPES):
            panel(ax, summ[(summ.family == "ecg_noise") & (summ.kind == k)], "level", f"ecg_noise: {k}", "SNR (dB)", metric, True)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="center right", fontsize=8)
        fig.suptitle(f"{metric.upper()} vs injected noise level (seed-0 checkpoints, clean test windows)")
        fig.tight_layout(rect=(0, 0, 0.9, 0.96))
        fig.savefig(out_dir / f"noise_curves_{metric}.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    panel(axs[0, 0], summ[summ.family == "ptt_const"], "level", "PPG delay vs ECG (constant)", "delay (ms)")
    panel(axs[0, 1], summ[summ.family == "ptt_jitter"], "level", "PPG random-walk jitter", "end-of-window std (ms)")
    for ax, (tgt, mode) in zip([axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]],
                               [("ppg", "zero"), ("ecg", "zero"), ("ppg", "flatline"), ("ecg", "flatline")]):
        panel(ax, summ[(summ.family == "dropout") & (summ.target == tgt) & (summ.kind == mode)], "level",
              f"{tgt.upper()} dropout ({mode})", "fraction of window lost")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=8)
    fig.suptitle("AUC under PTT perturbation and sensor dropout")
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    fig.savefig(out_dir / "ptt_dropout_curves.png", dpi=130); plt.close(fig)

    ms = summ[summ.family == "missing"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.2
    conds = [("clean", None), ("missing_zero", "ppg"), ("missing_zero", "ecg"), ("missing_noise", "ppg"), ("missing_noise", "ecg")]
    for i, (k, tgt) in enumerate(conds):
        vals = [clean.auc[v] if k == "clean" else float(ms[(ms.variant == v) & (ms.kind == k) & (ms.target == tgt)].auc.mean())
                for v in variants]
        ax.bar(np.arange(len(variants)) + (i - 2) * w, vals, w, label="clean" if k == "clean" else f"{k} {tgt}")
    ax.set_xticks(np.arange(len(variants))); ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_ylabel("AUC"); ax.legend(fontsize=8)
    ax.set_title("Whole-modality missing at inference")
    fig.tight_layout(); fig.savefig(out_dir / "missing_modality.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--variants", default=None, help="comma list; default = auto-discover seed-0 checkpoints")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--quality-threshold", type=float, default=0.97)
    ap.add_argument("--sig-variants", default="ecg_only,ppg_only,cross_attention")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from src.data.dataset import load_processed
    from src.train import run_dir

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed = args.processed or (ROOT / cfg["paths"]["processed_dir"] / "challenge2015_windows.npz")
    data = load_processed(processed)
    run_path = run_dir(cfg, args.run_name)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    variants = args.variants.split(",") if args.variants else discover_variants(run_path)
    print("variants:", variants, "device:", device, flush=True)
    out = run_path / "stress"
    out.mkdir(parents=True, exist_ok=True)

    bank = load_nstdb()
    df = run(cfg, run_path, data, variants, device, n_draws=args.draws, q_thr=args.quality_threshold, bank=bank)
    df.to_csv(out / "stress_by_fold.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "stress_summary.csv", index=False)
    deg = degradation_table(summ)
    deg.round(4).to_csv(out / "degradation_table.csv", index=False)
    summ[summ.family.isin(["ecg_noise", "ppg_noise", "both_noise"])].round(4).to_csv(out / "noise_auc_f1_vs_snr.csv", index=False)
    summ[summ.family == "missing"].round(4).to_csv(out / "missing_modality.csv", index=False)
    make_plots(summ, out)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(deg.round(3).to_string(index=False))

    sv = [v for v in args.sig_variants.split(",") if v in set(df.variant)]
    sig = signature_check(cfg, run_path, data, sv, device, bank, q_thr=args.quality_threshold)
    (out / "signature_check.json").write_text(json.dumps(sig, indent=2))
    print(json.dumps({k: v for k, v in sig.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
