"""Hand-crafted ECG/PPG agreement features (updates.md 3.4): beat-detector
agreement, pulse-transit-time (PTT) statistics, and signal-quality indices.

Operates on already-filtered, z-normalised windows (as produced by
src/data/preprocess.py), alarm trigger at the END of the window.
These features are also the beat/PTT ground-truth proxy for later analyses
(updates.md 7.2), so detectors are exposed individually.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp
from scipy import stats as sps

FS = 250


def detect_r_peaks(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Pan-Tompkins-style detector: derivative -> square -> moving integration
    -> adaptive-threshold peak picking with a 250 ms refractory period."""
    d = np.diff(ecg, prepend=ecg[0])
    sq = d * d
    win = max(1, int(0.12 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    if integ.max() <= 0:
        return np.array([], dtype=int)
    thr = 0.3 * np.percentile(integ, 99)
    cand, _ = sp.find_peaks(integ, height=thr, distance=int(0.25 * fs))
    # refine each candidate to the local ECG max/min of largest magnitude
    half = int(0.08 * fs)
    out = []
    for c in cand:
        lo, hi = max(0, c - half), min(len(ecg), c + half)
        out.append(lo + int(np.argmax(np.abs(ecg[lo:hi]))))
    return np.unique(np.array(out, dtype=int))


def detect_ppg_peaks(ppg: np.ndarray, fs: int = FS) -> np.ndarray:
    sd = np.std(ppg) + 1e-8
    p, _ = sp.find_peaks(ppg, distance=int(0.3 * fs), prominence=0.4 * sd)
    return p


def ppg_feet(ppg: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Pulse foot = minimum between the previous peak (or window start) and this peak."""
    feet = []
    prev = 0
    for pk in peaks:
        seg = ppg[prev:pk]
        feet.append(prev + int(np.argmin(seg)) if len(seg) else pk)
        prev = pk
    return np.array(feet, dtype=int)


def ptt_series(r_peaks: np.ndarray, feet: np.ndarray, fs: int = FS, lo=0.08, hi=0.6) -> np.ndarray:
    """For each R peak, delay (s) to the first PPG foot within [lo, hi] s after it."""
    out = []
    for r in r_peaks:
        d = (feet - r) / fs
        ok = d[(d >= lo) & (d <= hi)]
        if len(ok):
            out.append(ok.min())
    return np.array(out)


def _rate_stats(peaks: np.ndarray, fs: int, T: int, prefix: str) -> dict:
    f = {}
    f[f"{prefix}_n"] = len(peaks)
    f[f"{prefix}_n_last5s"] = int((peaks >= T - 5 * fs).sum())
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        med = np.median(rr)
        f[f"{prefix}_hr"] = 60.0 / med
        f[f"{prefix}_rr_cv"] = rr.std() / (rr.mean() + 1e-8)
        f[f"{prefix}_rr_max"] = rr.max()
        f[f"{prefix}_rr_min"] = rr.min()
        f[f"{prefix}_rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0
        f[f"{prefix}_irregular_frac"] = float(np.mean(np.abs(rr - med) > 0.2 * med))
    else:
        f.update({f"{prefix}_hr": 0.0, f"{prefix}_rr_cv": 0.0, f"{prefix}_rr_max": T / fs,
                  f"{prefix}_rr_min": T / fs, f"{prefix}_rmssd": 0.0, f"{prefix}_irregular_frac": 1.0})
    last = peaks[-1] if len(peaks) else 0
    f[f"{prefix}_gap_to_end"] = (T - last) / fs
    gaps = np.diff(np.concatenate([[0], peaks, [T]])) / fs
    f[f"{prefix}_longest_gap"] = gaps.max()
    return f


def _sqi(x: np.ndarray, fs: int, prefix: str, band: tuple[float, float]) -> dict:
    f, p = sp.welch(x, fs=fs, nperseg=min(len(x), 512))
    p = p / (p.sum() + 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    inband = float(p[(f >= band[0]) & (f <= band[1])].sum())
    lowf = float(p[f < 0.5].sum())
    hi = float(p[f > 30].sum())
    return {
        f"{prefix}_kurt": float(sps.kurtosis(x)), f"{prefix}_skew": float(sps.skew(x)),
        f"{prefix}_spec_entropy": ent, f"{prefix}_inband_power": inband,
        f"{prefix}_lowf_power": lowf, f"{prefix}_highf_power": hi,
        f"{prefix}_flat_frac": float(np.mean(np.abs(np.diff(x)) < 1e-3)),
    }


def _template_corr(x: np.ndarray, peaks: np.ndarray, fs: int, half_s: float, prefix: str) -> dict:
    """Mean correlation of each beat with the median beat (beat-to-beat morphology SQI)."""
    half = int(half_s * fs)
    beats = [x[p - half:p + half] for p in peaks if p - half >= 0 and p + half <= len(x)]
    if len(beats) < 3:
        return {f"{prefix}_tmpl_corr": 0.0, f"{prefix}_tmpl_corr_min": 0.0}
    b = np.stack(beats)
    tmpl = np.median(b, axis=0)
    c = [np.corrcoef(v, tmpl)[0, 1] for v in b]
    c = np.nan_to_num(c)
    return {f"{prefix}_tmpl_corr": float(np.mean(c)), f"{prefix}_tmpl_corr_min": float(np.min(c))}


def _acf_peak(x: np.ndarray, fs: int, prefix: str) -> dict:
    """Strength of the dominant periodicity at physiological beat lags
    (0.3-1.5 s): near 0 for noise / absent pulse, near 1 for a clean rhythm."""
    x = (x - x.mean()) / (x.std() + 1e-8)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (n * 1.0)
    lo, hi = int(0.3 * fs), int(1.5 * fs)
    return {f"{prefix}_acf_peak": float(acf[lo:hi].max())}


def _coupling(r_peaks: np.ndarray, ppg: np.ndarray, T: int, fs: int) -> dict:
    """Peak of the cross-correlation between a smoothed R-impulse train and the PPG,
    searched over physiological lags (0-0.8 s)."""
    imp = np.zeros(T)
    imp[r_peaks] = 1.0
    imp = np.convolve(imp, sp.windows.gaussian(int(0.1 * fs), int(0.02 * fs)), mode="same")
    imp = (imp - imp.mean()) / (imp.std() + 1e-8)
    pp = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    best, best_lag = -1.0, 0
    for lag in range(0, int(0.8 * fs), 2):
        c = float(np.dot(imp[: T - lag], pp[lag:]) / (T - lag))
        if c > best:
            best, best_lag = c, lag
    return {"xcorr_peak": best, "xcorr_lag_s": best_lag / fs}


def extract_features(ecg: np.ndarray, ppg: np.ndarray, fs: int = FS) -> dict:
    T = len(ecg)
    r = detect_r_peaks(ecg, fs)
    pk = detect_ppg_peaks(ppg, fs)
    feet = ppg_feet(ppg, pk)

    f = {}
    f.update(_rate_stats(r, fs, T, "ecg"))
    f.update(_rate_stats(pk, fs, T, "ppg"))
    f.update(_sqi(ecg, fs, "ecg", (5, 15)))
    f.update(_sqi(ppg, fs, "ppg", (0.5, 4)))
    f.update(_template_corr(ecg, r, fs, 0.25, "ecg"))
    f.update(_template_corr(ppg, pk, fs, 0.4, "ppg"))
    f.update(_acf_peak(ecg, fs, "ecg"))
    f.update(_acf_peak(ppg, fs, "ppg"))
    f["ecg_amp_cv"] = float(np.std(np.abs(ecg[r])) / (np.mean(np.abs(ecg[r])) + 1e-8)) if len(r) else 0.0

    # agreement between the two detectors
    f["hr_abs_diff"] = abs(f["ecg_hr"] - f["ppg_hr"])
    f["beat_count_ratio"] = (len(r) + 1) / (len(pk) + 1)
    f["beat_count_last5s_diff"] = f["ecg_n_last5s"] - f["ppg_n_last5s"]
    ptt = ptt_series(r, feet, fs)
    f["ecg_beats_with_pulse_frac"] = len(ptt) / max(len(r), 1)
    ptt_r = ptt_series(r[-6:], feet, fs) if len(r) else np.array([])
    f["recent_beats_with_pulse_frac"] = len(ptt_r) / max(min(len(r), 6), 1)
    f["ptt_mean"] = float(ptt.mean()) if len(ptt) else -1.0
    f["ptt_std"] = float(ptt.std()) if len(ptt) > 1 else -1.0
    f["ptt_median"] = float(np.median(ptt)) if len(ptt) else -1.0
    f["ptt_cv"] = float(ptt.std() / (ptt.mean() + 1e-8)) if len(ptt) > 1 else -1.0
    # PPG peaks that have a preceding R peak
    matched = 0
    for p in pk:
        d = (p - r) / fs
        matched += int(np.any((d > 0.1) & (d < 0.9)))
    f["ppg_peaks_with_r_frac"] = matched / max(len(pk), 1)
    f.update(_coupling(r, ppg, T, fs))
    return f


def feature_matrix(ecg: np.ndarray, ppg: np.ndarray, fs: int = FS) -> tuple[np.ndarray, list[str]]:
    rows = [extract_features(e, p, fs) for e, p in zip(ecg, ppg)]
    names = sorted(rows[0].keys())
    X = np.array([[r[n] for n in names] for r in rows], dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3), names
