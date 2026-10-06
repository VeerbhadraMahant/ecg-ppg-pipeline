"""Formal beat-detection + pulse-transit-time (PTT) module (updates.md 5.5/7.2).

Thin, documented wrapper around the detectors in src/features.py:
  analyze_window / analyze_windows -> per-window dicts with R peaks, PPG peaks,
  PPG feet, a per-beat PTT series (aligned to R peaks, NaN where no pulse
  follows) and heart rates from each modality.

Validation against external ground truth (PhysioNet via wfdb streaming):
  * validate_mitdb : ECG R-peak sensitivity / PPV / F1 vs. MIT-BIH beat
                     annotations (150 ms tolerance). ECG only (no PPG there).
  * validate_bidmc : BIDMC has paired ECG+PPG but NO beat annotations (only
                     breath annotations), so we validate against the monitor's
                     1 Hz HR / PULSE numerics plus ECG/PPG consistency.
  * validate_synthetic : offline fallback with known beat times.

Run: python -m src.heads.beat_ptt --out runs/heads/beat_ptt_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.signal_ops import bandpass_filter, resample_signal, z_normalize  # noqa: E402
from src.features import FS, detect_ppg_peaks, detect_r_peaks, ppg_feet  # noqa: E402

PTT_LO, PTT_HI = 0.08, 0.6


# ----------------------------------------------------------------------------
# Per-window analysis
# ----------------------------------------------------------------------------
def per_beat_ptt(r_peaks: np.ndarray, feet: np.ndarray, fs: int = FS, lo=PTT_LO, hi=PTT_HI) -> np.ndarray:
    """PTT (s) for every R peak (same length as r_peaks): delay to the first PPG
    foot within [lo, hi] s after it, NaN if none. Same rule as
    features.ptt_series but keeps beat alignment."""
    out = np.full(len(r_peaks), np.nan)
    for i, r in enumerate(r_peaks):
        d = (feet - r) / fs
        ok = d[(d >= lo) & (d <= hi)]
        if len(ok):
            out[i] = ok.min()
    return out


def _hr(peaks: np.ndarray, fs: int) -> float:
    """Median-RR heart rate (bpm); NaN with <2 peaks."""
    if len(peaks) < 2:
        return float("nan")
    return float(60.0 / np.median(np.diff(peaks) / fs))


def analyze_window(ecg: np.ndarray, ppg: np.ndarray, fs: int = FS) -> dict:
    r = detect_r_peaks(ecg, fs)
    pk = detect_ppg_peaks(ppg, fs)
    feet = ppg_feet(ppg, pk)
    ptt = per_beat_ptt(r, feet, fs)
    valid = ptt[~np.isnan(ptt)]
    return {
        "r_peaks": r, "ppg_peaks": pk, "ppg_feet": feet, "ptt": ptt,
        "hr_ecg": _hr(r, fs), "hr_ppg": _hr(pk, fs),
        "inst_hr_ecg": 60.0 * fs / np.diff(r) if len(r) > 1 else np.array([]),
        "inst_hr_ppg": 60.0 * fs / np.diff(pk) if len(pk) > 1 else np.array([]),
        "ptt_mean": float(valid.mean()) if len(valid) else float("nan"),
        "ptt_std": float(valid.std()) if len(valid) > 1 else float("nan"),
        "pulse_coverage": float(len(valid) / max(len(r), 1)),
    }


def analyze_windows(ecg: np.ndarray, ppg: np.ndarray, fs: int = FS) -> list[dict]:
    """ecg, ppg: (N, T). Returns a list of N per-window dicts."""
    return [analyze_window(e, p, fs) for e, p in zip(ecg, ppg)]


def beat_heatmap_targets(peaks_list: list[np.ndarray], T: int, n_tokens: int, sigma_tokens: float = 0.0) -> np.ndarray:
    """(N, n_tokens) binary (or Gaussian-smoothed, in [0,1] if sigma_tokens>0)
    targets: token k covers samples [k*T/n_tokens, (k+1)*T/n_tokens)."""
    out = np.zeros((len(peaks_list), n_tokens), dtype=np.float32)
    for i, pk in enumerate(peaks_list):
        tok = np.clip((np.asarray(pk, dtype=int) * n_tokens) // T, 0, n_tokens - 1)
        out[i, tok] = 1.0
    if sigma_tokens > 0:
        from scipy.ndimage import gaussian_filter1d

        sm = gaussian_filter1d(out, sigma_tokens, axis=1, mode="constant")
        out = np.clip(sm / (sm.max(axis=1, keepdims=True) + 1e-8), 0, 1).astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Matching / scoring
# ----------------------------------------------------------------------------
def match_beats(ref: np.ndarray, det: np.ndarray, tol: int) -> tuple[int, int, int]:
    """Greedy one-to-one nearest matching within +/-tol samples -> (tp, fp, fn)."""
    ref, det = np.sort(np.asarray(ref)), np.sort(np.asarray(det))
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        if not len(det):
            break
        d = np.abs(det - r).astype(float)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    return tp, int(len(det) - tp), int(len(ref) - tp)


def prf(tp: int, fp: int, fn: int) -> dict:
    se = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "sensitivity": se, "ppv": ppv, "f1": f1}


def _windows(x: np.ndarray, fs: int, seconds: int = 10):
    n = seconds * fs
    for s in range(0, len(x) - n + 1, n):
        yield s, s + n


def _prep(ecg=None, ppg=None, fs_in=250.0):
    """Mirror src/data/preprocess: resample to 250 Hz, bandpass (ECG 0.5-40, PPG 0.5-8)."""
    out = []
    for x, band in ((ecg, (0.5, 40)), (ppg, (0.5, 8))):
        if x is None:
            out.append(None)
            continue
        x = np.nan_to_num(np.asarray(x, dtype=float))
        x = resample_signal(x, fs_in, FS)
        out.append(bandpass_filter(x, FS, *band))
    return out


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
MITDB_RECORDS = ["100", "101", "103", "105", "106", "119", "200", "203", "207", "217"]


def validate_mitdb(records=MITDB_RECORDS, minutes: float = 5.0, tol_ms: float = 150.0) -> dict:
    import wfdb

    tol = int(round(tol_ms / 1000 * FS))
    per, TP = {}, [0, 0, 0]
    for rec in records:
        try:
            sampto = int(minutes * 60 * 360)
            r = wfdb.rdrecord(rec, pn_dir="mitdb", sampto=sampto, channels=[0])
            a = wfdb.rdann(rec, "atr", pn_dir="mitdb", sampto=sampto)
        except Exception as e:  # pragma: no cover - network
            per[rec] = {"error": repr(e)}
            continue
        ecg, = _prep(ecg=r.p_signal[:, 0], fs_in=r.fs)[:1]
        samp = np.array([s for s, sym in zip(a.sample, a.symbol) if sym in BEAT_SYMBOLS])
        samp = np.round(samp * FS / r.fs).astype(int)
        tp = fp = fn = 0
        for lo, hi in _windows(ecg, FS):
            w = z_normalize(ecg[lo:hi])
            det = detect_r_peaks(w, FS)
            ref = samp[(samp >= lo) & (samp < hi)] - lo
            a_, b_, c_ = match_beats(ref, det, tol)
            tp, fp, fn = tp + a_, fp + b_, fn + c_
        per[rec] = prf(tp, fp, fn)
        TP = [TP[0] + tp, TP[1] + fp, TP[2] + fn]
    return {"dataset": "mitdb", "tolerance_ms": tol_ms, "minutes_per_record": minutes,
            "per_record": per, "pooled": prf(*TP)}


def validate_bidmc(records=("01", "02", "03", "04", "05", "06", "07", "08"), minutes: float = 5.0) -> dict:
    """No beat annotations exist in BIDMC. We compare (a) detector HR with the
    monitor's HR numerics (ECG-derived) and PULSE numerics (PPG-derived), median
    per 10 s window, and (b) physiological plausibility of per-beat PTT."""
    import wfdb

    rows, ptt_all, cov = [], [], []
    for rid in records:
        try:
            name = f"bidmc{rid}"
            r = wfdb.rdrecord(name, pn_dir="bidmc")
            nm = wfdb.rdrecord(name + "n", pn_dir="bidmc")
        except Exception as e:  # pragma: no cover
            rows.append({"record": name, "error": repr(e)})
            continue
        names = [n.strip(" ,").upper() for n in r.sig_name]
        ecg_i, ppg_i = names.index("II"), names.index("PLETH")
        nn = [n.strip(" ,").upper() for n in nm.sig_name]
        hr_num, pulse_num = nm.p_signal[:, nn.index("HR")], nm.p_signal[:, nn.index("PULSE")]
        ecg, ppg = _prep(r.p_signal[:, ecg_i], r.p_signal[:, ppg_i], r.fs)
        lim = int(minutes * 60 * FS)
        ecg, ppg = ecg[:lim], ppg[:lim]
        d_ecg, d_ppg = [], []
        for k, (lo, hi) in enumerate(_windows(ecg, FS)):
            res = analyze_window(z_normalize(ecg[lo:hi]), z_normalize(ppg[lo:hi]))
            s0, s1 = lo // FS, hi // FS
            ref_hr, ref_pu = np.nanmedian(hr_num[s0:s1]), np.nanmedian(pulse_num[s0:s1])
            if np.isfinite(res["hr_ecg"]) and np.isfinite(ref_hr):
                d_ecg.append(res["hr_ecg"] - ref_hr)
            if np.isfinite(res["hr_ppg"]) and np.isfinite(ref_pu):
                d_ppg.append(res["hr_ppg"] - ref_pu)
            v = res["ptt"][~np.isnan(res["ptt"])]
            ptt_all.extend(v.tolist())
            cov.append(res["pulse_coverage"])
        d_ecg, d_ppg = np.array(d_ecg), np.array(d_ppg)
        rows.append({"record": name, "n_windows": len(d_ecg),
                     "ecg_hr_mae_bpm": float(np.mean(np.abs(d_ecg))) if len(d_ecg) else None,
                     "ecg_hr_within5pct": float(np.mean(np.abs(d_ecg) < 5)) if len(d_ecg) else None,
                     "ppg_hr_mae_bpm": float(np.mean(np.abs(d_ppg))) if len(d_ppg) else None,
                     "ppg_hr_within5pct": float(np.mean(np.abs(d_ppg) < 5)) if len(d_ppg) else None})
    ok = [r for r in rows if "error" not in r and r["n_windows"]]
    ptt_all = np.array(ptt_all)
    summ = {
        "note": "BIDMC has no beat annotations; validated against monitor HR/PULSE numerics (1 Hz), not beat-level truth.",
        "per_record": rows,
        "mean_ecg_hr_mae_bpm": float(np.mean([r["ecg_hr_mae_bpm"] for r in ok])) if ok else None,
        "mean_ppg_hr_mae_bpm": float(np.mean([r["ppg_hr_mae_bpm"] for r in ok])) if ok else None,
        "mean_pulse_coverage": float(np.mean(cov)) if cov else None,
        "ptt_median_s": float(np.median(ptt_all)) if len(ptt_all) else None,
        "ptt_iqr_s": [float(x) for x in np.percentile(ptt_all, [25, 75])] if len(ptt_all) else None,
    }
    return {"dataset": "bidmc", **summ}


def synthetic_pair(hr_period=0.8, ptt=0.25, fs=FS, T=2500, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    r = np.arange(0.5, T / fs - 0.6, hr_period)
    ecg, ppg = np.zeros(T), np.zeros(T)
    for x in r:
        ecg[int(x * fs)] = 1.0
        ppg[int((x + ptt) * fs)] = 1.0
    ecg = np.convolve(ecg, np.hanning(15), "same") + noise * rng.randn(T)
    ppg = np.convolve(ppg, np.hanning(60), "same") + noise * rng.randn(T)
    return z_normalize(ecg), z_normalize(ppg), (r * fs).astype(int)


def validate_synthetic(tol_ms: float = 150.0) -> dict:
    tol = int(tol_ms / 1000 * FS)
    out = {}
    for noise in (0.0, 0.1, 0.3):
        TP = [0, 0, 0]
        errs = []
        for s in range(20):
            e, p, ref = synthetic_pair(hr_period=0.6 + 0.05 * (s % 8), noise=noise, seed=s)
            res = analyze_window(e, p)
            a, b, c = match_beats(ref, res["r_peaks"], tol)
            TP = [TP[0] + a, TP[1] + b, TP[2] + c]
            errs.append(np.nanmean(res["ptt"]) - 0.25)
        out[f"noise{noise}"] = {**prf(*TP), "ptt_bias_s": float(np.nanmean(errs))}
    return {"dataset": "synthetic", "results": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "heads" / "beat_ptt_validation.json"))
    ap.add_argument("--minutes", type=float, default=5.0)
    args = ap.parse_args()
    res = {"synthetic": validate_synthetic()}
    for name, fn in (("mitdb", validate_mitdb), ("bidmc", validate_bidmc)):
        try:
            res[name] = fn(minutes=args.minutes)
        except Exception as e:
            res[name] = {"unreachable_or_failed": repr(e)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: (v.get("pooled") or {kk: vv for kk, vv in v.items() if kk != "per_record"}) for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
