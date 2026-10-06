"""Hand-crafted feature set v2 (v1 in src/features.py is untouched).

~230 features per 10 s / 250 Hz z-normalised window, the alarm fires at the
window END. Groups (name prefixes):
  e_*   ECG rhythm/HRV/morphology        p_*   PPG pulse features
  es_*/ps_*  Welch + wavelet spectral    x_*   ECG-PPG agreement
  eq_*/pq_*  signal-quality indices      d_*   alarm-type detectors (asystole, brady/tachy, VT/VF)
Everything is computed from the window itself (no cross-window statistics), so
extraction cannot leak across folds.

CLI:  python -m src.features_v2 --dataset {cinc,cinc_recovered,vtac}
caches data/processed/<stem>_features_v2.npz (X, names, alarm_onehot, alarm_names).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import signal as sp
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import detect_ppg_peaks, detect_r_peaks, ppg_feet  # noqa: E402

try:
    import pywt
except Exception:  # pragma: no cover
    pywt = None

FS = 250
T = 2500
ALARM_TYPES = ["Asystole", "Bradycardia", "Tachycardia", "Ventricular_Flutter_Fib", "Ventricular_Tachycardia"]
DATASETS = {
    "cinc": "challenge2015_windows",
    "cinc_recovered": "challenge2015_windows_recovered",
    "vtac": "vtac_windows",
}


# ----------------------------------------------------------------- helpers
def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 4:
        return np.nan
    sd = x.std()
    if sd < 1e-8:
        return 0.0
    r = r_frac * sd

    def count(mm):
        tm = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.max(np.abs(tm[:, None, :] - tm[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(tm)) / 2.0

    b, a = count(m), count(m + 1)
    if a <= 0 or b <= 0:
        return np.log(max(b, 1.0) * 1.0) if b > 0 else np.nan
    return float(-np.log(a / b))


def _stats(v, p, f, fn=("mean", "std", "med", "min", "max")):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    nan = np.nan
    if len(v) == 0:
        for n in fn:
            f[f"{p}_{n}"] = nan
        return
    d = {"mean": v.mean(), "std": v.std() if len(v) > 1 else nan, "med": np.median(v), "min": v.min(),
         "max": v.max(), "mad": np.median(np.abs(v - np.median(v))),
         "cv": v.std() / (abs(v.mean()) + 1e-8) if len(v) > 1 else nan}
    for n in fn:
        f[f"{p}_{n}"] = d[n]


def _hr_in(peaks, lo, hi, fs=FS):
    """(count-based HR, median-RR HR, count) of peaks in [lo,hi) samples."""
    pk = peaks[(peaks >= lo) & (peaks < hi)]
    span = (hi - lo) / fs
    cnt_hr = 60.0 * len(pk) / span
    if len(pk) >= 2:
        return cnt_hr, 60.0 / np.median(np.diff(pk) / fs), len(pk)
    return cnt_hr, 0.0, len(pk)


def _hrv(peaks, prefix, f, fs=FS):
    rr = np.diff(peaks) / fs
    f[f"{prefix}_n"] = len(peaks)
    for w in (3, 4, 5):
        c, m, n = _hr_in(peaks, T - w * fs, T, fs)
        f[f"{prefix}_hrc_last{w}"] = c
        f[f"{prefix}_n_last{w}"] = n
        if w in (4, 5):
            f[f"{prefix}_hrm_last{w}"] = m
    f[f"{prefix}_hrc_first5"] = _hr_in(peaks, 0, 5 * fs, fs)[0]
    f[f"{prefix}_hrc_all"] = 60.0 * len(peaks) / (T / fs)
    f[f"{prefix}_gap_to_end"] = (T - peaks[-1]) / fs if len(peaks) else T / fs
    f[f"{prefix}_gap_from_start"] = peaks[0] / fs if len(peaks) else T / fs
    allp = np.concatenate([[0], peaks, [T]])
    f[f"{prefix}_longest_gap"] = np.diff(allp).max() / fs
    f[f"{prefix}_longest_rr"] = rr.max() if len(rr) else np.nan
    if len(rr) < 2:
        for n in ("hr_med", "hr_mean", "sdnn", "rmssd", "pnn50", "pnn20", "rr_cv", "rr_min", "rr_range", "rr_mad",
                  "rr_ent", "rr_sampen", "sd1", "sd2", "sd_ratio", "irreg", "rr_last", "rr_slope", "rr_skew",
                  "rr_tri", "hr_slope", "hr_last_vs_first"):
            f[f"{prefix}_{n}"] = np.nan
        f[f"{prefix}_hr_med"] = 0.0
        return
    med = np.median(rr)
    d = np.diff(rr)
    f[f"{prefix}_hr_med"] = 60.0 / med
    f[f"{prefix}_hr_mean"] = 60.0 / rr.mean()
    f[f"{prefix}_sdnn"] = rr.std()
    f[f"{prefix}_rmssd"] = np.sqrt(np.mean(d ** 2)) if len(d) else np.nan
    f[f"{prefix}_pnn50"] = np.mean(np.abs(d) > 0.05) if len(d) else np.nan
    f[f"{prefix}_pnn20"] = np.mean(np.abs(d) > 0.02) if len(d) else np.nan
    f[f"{prefix}_rr_cv"] = rr.std() / (rr.mean() + 1e-8)
    f[f"{prefix}_rr_min"] = rr.min()
    f[f"{prefix}_rr_range"] = rr.max() - rr.min()
    f[f"{prefix}_rr_mad"] = np.median(np.abs(rr - med))
    h, _ = np.histogram(rr, bins=np.linspace(0.2, 2.5, 13))
    pr = h[h > 0] / max(h.sum(), 1)
    f[f"{prefix}_rr_ent"] = float(-(pr * np.log(pr)).sum()) if len(pr) else 0.0
    f[f"{prefix}_rr_sampen"] = sample_entropy(rr) if len(rr) >= 8 else np.nan
    if len(rr) >= 3:
        a, b = rr[:-1], rr[1:]
        sd1 = np.std((b - a) / np.sqrt(2))
        sd2 = np.std((b + a) / np.sqrt(2))
        f[f"{prefix}_sd1"], f[f"{prefix}_sd2"], f[f"{prefix}_sd_ratio"] = sd1, sd2, sd1 / (sd2 + 1e-8)
    else:
        f[f"{prefix}_sd1"] = f[f"{prefix}_sd2"] = f[f"{prefix}_sd_ratio"] = np.nan
    f[f"{prefix}_irreg"] = np.mean(np.abs(rr - med) > 0.2 * med)
    f[f"{prefix}_rr_last"] = rr[-1]
    f[f"{prefix}_rr_slope"] = np.polyfit(np.arange(len(rr)), rr, 1)[0] if len(rr) >= 3 else np.nan
    f[f"{prefix}_rr_skew"] = sps.skew(rr) if len(rr) >= 3 else np.nan
    f[f"{prefix}_rr_tri"] = len(rr) / (np.histogram(rr, bins=np.arange(rr.min(), rr.max() + 1 / 128 + 1e-9, 1 / 128))[0].max() + 1e-8) \
        if rr.max() > rr.min() else np.nan
    hr_beats = 60.0 / rr
    f[f"{prefix}_hr_slope"] = np.polyfit(np.arange(len(rr)), hr_beats, 1)[0] if len(rr) >= 3 else np.nan
    k = max(1, len(rr) // 3)
    f[f"{prefix}_hr_last_vs_first"] = hr_beats[-k:].mean() - hr_beats[:k].mean()


def _template_corr(x, peaks, half, prefix, f):
    h = int(half * FS)
    beats = [x[p - h:p + h] for p in peaks if p - h >= 0 and p + h <= len(x)]
    if len(beats) < 3:
        for n in ("tmpl_mean", "tmpl_min", "tmpl_std", "tmpl_last"):
            f[f"{prefix}_{n}"] = np.nan
        return None
    b = np.stack(beats)
    tm = np.median(b, axis=0)
    c = np.nan_to_num([np.corrcoef(v, tm)[0, 1] if v.std() > 1e-8 and tm.std() > 1e-8 else 0.0 for v in b])
    f[f"{prefix}_tmpl_mean"], f[f"{prefix}_tmpl_min"], f[f"{prefix}_tmpl_std"] = c.mean(), c.min(), c.std()
    f[f"{prefix}_tmpl_last"] = c[-1]
    return tm


def _acf(x, prefix, f, lo_s=0.25, hi_s=1.5):
    x = (x - x.mean()) / (x.std() + 1e-8)
    n = len(x)
    F = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(F * np.conj(F))[:n] / n
    lo, hi = int(lo_s * FS), int(hi_s * FS)
    seg = acf[lo:hi]
    f[f"{prefix}_acf_peak"] = seg.max()
    f[f"{prefix}_acf_lag"] = (lo + int(np.argmax(seg))) / FS
    # second periodicity peak (regularity at twice the lag)
    l2 = 2 * (lo + int(np.argmax(seg)))
    f[f"{prefix}_acf_2nd"] = acf[l2 - 5:l2 + 5].max() if l2 + 5 < n else np.nan


def _spectral(x, prefix, bands, f, rng_ent=(0.3, 40)):
    fr, p = sp.welch(x, fs=FS, nperseg=1024, noverlap=512)
    tot = p.sum() + 1e-12
    pn = p / tot
    for lo, hi in bands:
        f[f"{prefix}_bp_{lo:g}_{hi:g}"] = pn[(fr >= lo) & (fr < hi)].sum()
    m = (fr >= rng_ent[0]) & (fr <= rng_ent[1])
    q = p[m] / (p[m].sum() + 1e-12)
    f[f"{prefix}_spec_ent"] = float(-(q * np.log(q + 1e-12)).sum() / np.log(m.sum()))
    fm, pm = fr[m], p[m]
    f[f"{prefix}_dom_f"] = fm[np.argmax(pm)]
    f[f"{prefix}_dom_ratio"] = pm.max() / (pm.sum() + 1e-12)
    f[f"{prefix}_centroid"] = (fm * pm).sum() / (pm.sum() + 1e-12)
    f[f"{prefix}_bw"] = np.sqrt((((fm - f[f"{prefix}_centroid"]) ** 2) * pm).sum() / (pm.sum() + 1e-12))
    cs = np.cumsum(pm) / (pm.sum() + 1e-12)
    f[f"{prefix}_edge90"] = fm[np.searchsorted(cs, 0.9)]
    return fr, p


def _wavelet(x, prefix, f, level=6):
    if pywt is None:
        return
    c = pywt.wavedec(x, "db4", level=level)
    e = np.array([np.sum(ci ** 2) for ci in c]) + 1e-12
    r = e / e.sum()
    for i, v in enumerate(r):
        f[f"{prefix}_wav{i}"] = v
    f[f"{prefix}_wav_ent"] = float(-(r * np.log(r)).sum())


def _sqi(x, prefix, f):
    f[f"{prefix}_kurt"] = sps.kurtosis(x)
    f[f"{prefix}_skew"] = sps.skew(x)
    dx = np.diff(x)
    f[f"{prefix}_flat_frac"] = np.mean(np.abs(dx) < 1e-3)
    f[f"{prefix}_clip_frac"] = np.mean(np.abs(x) >= 0.98 * np.abs(x).max())
    seg = x[: (len(x) // FS) * FS].reshape(-1, FS)
    sds = seg.std(axis=1)
    f[f"{prefix}_seg_sd_std"] = sds.std()
    f[f"{prefix}_seg_sd_min"] = sds.min()
    f[f"{prefix}_seg_sd_max"] = sds.max()
    f[f"{prefix}_last2s_sd"] = x[-2 * FS:].std() / (x.std() + 1e-8)
    f[f"{prefix}_last1s_sd"] = x[-FS:].std() / (x.std() + 1e-8)
    f[f"{prefix}_max_absdiff"] = np.abs(dx).max()
    f[f"{prefix}_line_len"] = np.abs(dx).mean()
    f[f"{prefix}_zcr"] = np.mean(np.diff(np.signbit(x - np.median(x))) != 0)
    hj_mob = np.std(dx) / (np.std(x) + 1e-8)
    f[f"{prefix}_hjorth_mob"] = hj_mob
    f[f"{prefix}_hjorth_cmp"] = (np.std(np.diff(dx)) / (np.std(dx) + 1e-8)) / (hj_mob + 1e-8)


def _qrs_morph(ecg, r, f):
    h, w = int(0.12 * FS), int(0.08 * FS)
    widths, ptp, tr, pr_, ramp = [], [], [], [], []
    for i, p in enumerate(r):
        lo, hi = p - h, p + h
        if lo < 0 or hi > T:
            continue
        seg = ecg[lo:hi]
        a = np.abs(seg - np.median(seg))
        ramp.append(abs(ecg[p]))
        ptp.append(ecg[max(0, p - w):p + w].max() - ecg[max(0, p - w):p + w].min())
        above = np.where(a >= 0.5 * a.max())[0]
        widths.append((above.max() - above.min() + 1) / FS * 1000 if len(above) else np.nan)
        # T / P proxies relative to the beat amplitude
        nxt = r[i + 1] if i + 1 < len(r) else T
        t_lo, t_hi = p + int(0.15 * FS), min(p + int(0.45 * FS), nxt - int(0.12 * FS))
        if t_hi - t_lo > 5:
            tr.append(np.abs(ecg[t_lo:t_hi]).max() / (abs(ecg[p]) + 1e-8))
        p_lo, p_hi = max(p - int(0.25 * FS), 0), p - int(0.08 * FS)
        if p_hi - p_lo > 5:
            pr_.append(np.abs(ecg[p_lo:p_hi]).max() / (abs(ecg[p]) + 1e-8))
    _stats(widths, "e_qrs_w", f, ("mean", "std", "med", "max"))
    _stats(ramp, "e_qrs_amp", f, ("mean", "std", "cv", "min", "max"))
    _stats(ptp, "e_qrs_ptp", f, ("mean", "std", "cv"))
    _stats(tr, "e_t_ratio", f, ("mean", "med"))
    _stats(pr_, "e_p_ratio", f, ("mean", "med"))
    if len(ramp) > 1:
        f["e_amp_last_vs_all"] = np.mean(ramp[-3:]) / (np.mean(ramp) + 1e-8)
    else:
        f["e_amp_last_vs_all"] = np.nan


def _ppg_pulses(ppg, pk, feet, f):
    amp, rise, area, dic, upslope, widths = [], [], [], [], [], []
    dp = np.gradient(ppg)
    for i, (p, ft) in enumerate(zip(pk, feet)):
        a = ppg[p] - ppg[ft]
        amp.append(a)
        rise.append((p - ft) / FS)
        upslope.append(dp[ft:p + 1].max() if p > ft else np.nan)
        nxt = feet[i + 1] if i + 1 < len(feet) and feet[i + 1] > p else min(p + int(0.6 * FS), T)
        if nxt > p + 5:
            seg = ppg[p:nxt]
            area.append(np.trapezoid(ppg[ft:nxt] - ppg[ft], dx=1 / FS))
            # dicrotic proxy: secondary local maximum on the descending limb
            sub, _ = sp.find_peaks(seg, prominence=0.03 * (abs(a) + 1e-8))
            sub = sub[sub > 3]
            dic.append((ppg[p + sub[0]] - ppg[ft]) / (a + 1e-8) if len(sub) else 0.0)
            half = ppg[ft] + 0.5 * a
            wdt = np.sum(ppg[ft:nxt] >= half) / FS
            widths.append(wdt)
    _stats(amp, "p_amp", f, ("mean", "std", "cv", "min", "med"))
    _stats(rise, "p_rise", f, ("mean", "std", "med"))
    _stats(area, "p_area", f, ("mean", "std", "cv"))
    _stats(dic, "p_dicrotic", f, ("mean", "med"))
    _stats(upslope, "p_upslope", f, ("mean", "std"))
    _stats(widths, "p_width50", f, ("mean", "std"))
    amp = np.asarray(amp)
    if len(amp) >= 2:
        f["p_amp_last_vs_all"] = amp[-2:].mean() / (amp.mean() + 1e-8)
        f["p_amp_rmssd"] = np.sqrt(np.mean(np.diff(amp) ** 2))
        f["p_perfusion"] = amp.mean() / (np.std(ppg) + 1e-8)
    else:
        f["p_amp_last_vs_all"] = f["p_amp_rmssd"] = f["p_perfusion"] = np.nan
    f["p_amp_last5_mean"] = np.mean([ppg[p] - ppg[ft] for p, ft in zip(pk, feet) if p >= T - 5 * FS]) \
        if np.any(pk >= T - 5 * FS) else 0.0


def _agreement(ecg, ppg, r, pk, feet, f):
    # PTT
    ptt, pulse_for_beat, ptt_t = [], [], []
    for b in r:
        d = (feet - b) / FS
        ok = d[(d >= 0.08) & (d <= 0.6)]
        pulse_for_beat.append(len(ok) > 0)
        if len(ok):
            ptt.append(ok.min())
            ptt_t.append(b)
    ptt = np.array(ptt)
    pfb = np.array(pulse_for_beat, dtype=bool)
    _stats(ptt, "x_ptt", f, ("mean", "std", "med", "mad", "min", "max"))
    f["x_ptt_n"] = len(ptt)
    f["x_ptt_rmssd"] = np.sqrt(np.mean(np.diff(ptt) ** 2)) if len(ptt) > 2 else np.nan
    f["x_beats_pulse_frac"] = pfb.mean() if len(r) else 0.0
    last5 = r >= T - 5 * FS
    f["x_beats_pulse_frac_last5"] = pfb[last5].mean() if last5.any() else np.nan
    f["x_beats_pulse_frac_last6beats"] = pfb[-6:].mean() if len(r) else 0.0
    f["x_beats_nopulse_last5"] = int((~pfb[last5]).sum()) if last5.any() else 0
    # ppg peaks with an R peak before them
    m = 0
    for p in pk:
        d = (p - r) / FS
        m += int(np.any((d > 0.08) & (d < 0.9)))
    f["x_ppk_with_r_frac"] = m / max(len(pk), 1)
    # HR agreement
    for name, lo in (("full", 0), ("last5", T - 5 * FS), ("last4", T - 4 * FS)):
        ce, me, ne = _hr_in(r, lo, T)
        cp, mp, npk = _hr_in(pk, lo, T)
        f[f"x_hr_absdiff_cnt_{name}"] = abs(ce - cp)
        f[f"x_hr_absdiff_med_{name}"] = abs(me - mp)
        f[f"x_hr_ratio_cnt_{name}"] = (ce + 1) / (cp + 1)
        f[f"x_beat_ratio_{name}"] = (ne + 1) / (npk + 1)
        f[f"x_beat_diff_{name}"] = ne - npk
    f["x_beat_ratio_last3"] = (int((r >= T - 3 * FS).sum()) + 1) / (int((pk >= T - 3 * FS).sum()) + 1)
    f["x_ppg_gap_to_end"] = (T - pk[-1]) / FS if len(pk) else T / FS
    f["x_ecg_ppg_gap_diff"] = f["x_ppg_gap_to_end"] - ((T - r[-1]) / FS if len(r) else T / FS)
    # envelope cross-correlation
    env = np.abs(sp.sosfiltfilt(sp.butter(2, 8, "low", fs=FS, output="sos"), np.abs(ecg - np.median(ecg))))
    env = sp.decimate(env, 5)
    pp = sp.decimate(ppg, 5)
    fs2 = FS / 5

    def xc(a, b, max_lag_s=1.0):
        a = (a - a.mean()) / (a.std() + 1e-8)
        b = (b - b.mean()) / (b.std() + 1e-8)
        n = len(a)
        c = np.correlate(b, a, "full") / n
        lags = np.arange(-n + 1, n) / fs2
        mk = (lags >= -0.2) & (lags <= max_lag_s)
        i = np.argmax(c[mk])
        return c[mk][i], lags[mk][i]

    f["x_xc_peak"], f["x_xc_lag"] = xc(env, pp)
    f["x_xc_peak_last5"], f["x_xc_lag_last5"] = xc(env[-int(5 * fs2):], pp[-int(5 * fs2):])
    # R-impulse vs PPG (v1 style)
    imp = np.zeros(T)
    imp[r] = 1.0
    imp = np.convolve(imp, sp.windows.gaussian(int(0.1 * FS) | 1, 0.02 * FS), mode="same")
    ii = sp.decimate(imp, 5)
    f["x_imp_xc_peak"], f["x_imp_xc_lag"] = xc(ii, pp)
    # coherence
    fr, coh = sp.coherence(env, pp, fs=fs2, nperseg=128)
    mk = (fr >= 0.7) & (fr <= 3.5)
    f["x_coh_mean"] = coh[mk].mean()
    f["x_coh_max"] = coh[mk].max()
    f["x_coh_at_hr"] = coh[mk][np.argmax(sp.welch(pp, fs=fs2, nperseg=128)[1][mk])]
    # spectral HR agreement: dominant frequency ecg-envelope vs ppg
    pe = sp.welch(env, fs=fs2, nperseg=256)
    pq = sp.welch(pp, fs=fs2, nperseg=256)
    mk2 = (pe[0] >= 0.7) & (pe[0] <= 3.5)
    f["x_domf_absdiff"] = abs(pe[0][mk2][np.argmax(pe[1][mk2])] - pq[0][mk2][np.argmax(pq[1][mk2])]) * 60


def _detectors(ecg, r, pk, f):
    rr = np.diff(r) / FS
    # asystole
    f["d_asys_longest_gap"] = f["e_longest_gap"]
    f["d_asys_since_last"] = f["e_gap_to_end"]
    f["d_asys_since_last_ppg"] = f["p_gap_to_end"]
    f["d_asys_gap_gt2p5"] = float(f["e_longest_gap"] > 2.5)
    f["d_asys_gap_gt4"] = float(f["e_longest_gap"] > 4.0)
    f["d_flat_last2s"] = float(ecg[-2 * FS:].std())
    f["d_n_rr_gt2"] = int((rr > 2.0).sum())
    # brady / tachy: rate at the end
    f["d_hr_last5"] = f["e_hrc_last5"]
    f["d_brady_5"] = float(f["e_hrc_last5"] < 50)
    f["d_tachy_5"] = float(f["e_hrc_last5"] > 120)
    f["d_hr_min_rolling"], f["d_hr_max_rolling"] = np.nan, np.nan
    if len(r) >= 2:
        # rolling HR over 5 consecutive intervals
        hr = 60.0 / rr
        if len(hr) >= 5:
            roll = np.convolve(hr, np.ones(5) / 5, "valid")
            f["d_hr_min_rolling"], f["d_hr_max_rolling"] = roll.min(), roll.max()
        else:
            f["d_hr_min_rolling"], f["d_hr_max_rolling"] = hr.min(), hr.max()
        f["d_n_beats_hr_gt100"] = int((hr > 100).sum())
        f["d_n_beats_hr_gt150"] = int((hr > 150).sum())
        f["d_n_beats_hr_lt50"] = int((hr < 50).sum())
    else:
        f["d_n_beats_hr_gt100"] = f["d_n_beats_hr_gt150"] = f["d_n_beats_hr_lt50"] = 0
    # VT / VF: regularity, entropy, spectral concentration, leakage
    x50 = sp.decimate(ecg, 5)
    f["d_ecg_sampen"] = sample_entropy(x50[:500])
    f["d_ecg_sampen_last5"] = sample_entropy(x50[-250:], r_frac=0.2)
    fr, p = sp.welch(ecg, fs=FS, nperseg=1024, noverlap=512)
    m = (fr >= 1) & (fr <= 12)
    fd = fr[m][np.argmax(p[m])]
    f["d_vf_dom_f"] = fd
    band = (fr >= fd - 0.5) & (fr <= fd + 0.5)
    f["d_vf_conc"] = p[band].sum() / (p[(fr >= 0.5) & (fr <= 30)].sum() + 1e-12)
    # leakage (Amann): power near the dominant freq and its 2nd harmonic
    harm = ((fr >= 0.5 * fd) & (fr <= 1.4 * fd)) | ((fr >= 1.6 * fd) & (fr <= 2.4 * fd))
    f["d_vf_leak"] = 1.0 - p[harm].sum() / (p[(fr >= 0.5) & (fr <= 30)].sum() + 1e-12)
    f["d_vf_freq_3_7_frac"] = p[(fr >= 3) & (fr <= 7)].sum() / (p[(fr >= 0.5) & (fr <= 30)].sum() + 1e-12)
    # regularity of rhythm last 5 s
    rl = r[r >= T - 5 * FS]
    rrl = np.diff(rl) / FS
    f["d_last5_rr_cv"] = rrl.std() / (rrl.mean() + 1e-8) if len(rrl) >= 2 else np.nan
    f["d_last5_rr_mean"] = rrl.mean() if len(rrl) else np.nan
    f["d_last5_ecg_acf"] = np.nan
    seg = ecg[-5 * FS:]
    seg = (seg - seg.mean()) / (seg.std() + 1e-8)
    n = len(seg)
    F = np.fft.rfft(seg, 2 * n)
    a = np.fft.irfft(F * np.conj(F))[:n] / n
    f["d_last5_ecg_acf"] = a[int(0.2 * FS):int(1.5 * FS)].max()
    f["d_ecg_amp_last5_vs_all"] = ecg[-5 * FS:].std() / (ecg.std() + 1e-8)
    f["d_ppg_last5_pulses"] = int((pk >= T - 5 * FS).sum())


def extract_v2(ecg: np.ndarray, ppg: np.ndarray) -> dict:
    ecg = ecg.astype(np.float64)
    ppg = ppg.astype(np.float64)
    r = detect_r_peaks(ecg, FS)
    pk = detect_ppg_peaks(ppg, FS)
    feet = ppg_feet(ppg, pk) if len(pk) else np.array([], dtype=int)
    f: dict = {}
    _hrv(r, "e", f)
    _hrv(pk, "p", f)
    _qrs_morph(ecg, r, f)
    _template_corr(ecg, r, 0.25, "e", f)
    _template_corr(ppg, pk, 0.4, "p", f)
    _acf(ecg, "e", f)
    _acf(ppg, "p", f)
    f["e_amp_beat_n_ratio"] = len(r) / max(len(pk), 1)
    if len(pk):
        _ppg_pulses(ppg, pk, feet, f)
    else:
        for n in ("amp_mean", "rise_mean", "area_mean"):
            f[f"p_{n}"] = np.nan
    # spectral
    _spectral(ecg, "es", [(0, 0.5), (0.5, 3), (3, 8), (8, 15), (15, 30), (30, 60)], f)
    _spectral(ppg, "ps", [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 20)], f)
    _wavelet(ecg, "es", f)
    _wavelet(ppg, "ps", f)
    # quality
    _sqi(ecg, "eq", f)
    _sqi(ppg, "pq", f)
    # detector self-consistency (bSQI-like): a second, independent ECG detector vs. the main one
    try:
        b = sp.sosfiltfilt(sp.butter(2, [5, 20], "band", fs=FS, output="sos"), ecg)
        ab = np.abs(b)
        alt, _ = sp.find_peaks(ab, height=0.4 * np.percentile(ab, 99.5), distance=int(0.25 * FS))
        f["eq_bsqi"] = np.mean([np.any(np.abs(alt - p) <= 0.05 * FS) for p in r]) if len(r) else 0.0
        f["eq_alt_n_ratio"] = (len(alt) + 1) / (len(r) + 1)
    except Exception:
        f["eq_bsqi"] = f["eq_alt_n_ratio"] = np.nan
    f["eq_ecg_snr"] = float(np.var(sp.sosfiltfilt(sp.butter(2, [5, 20], "band", fs=FS, output="sos"), ecg))
                            / (np.var(sp.sosfiltfilt(sp.butter(2, 30, "high", fs=FS, output="sos"), ecg)) + 1e-8))
    f["pq_ppg_snr"] = float(np.var(sp.sosfiltfilt(sp.butter(2, [0.6, 4], "band", fs=FS, output="sos"), ppg))
                            / (np.var(sp.sosfiltfilt(sp.butter(2, 8, "high", fs=FS, output="sos"), ppg)) + 1e-8))
    _agreement(ecg, ppg, r, pk, feet, f)
    _detectors(ecg, r, pk, f)
    return f


def _safe(e, p):
    try:
        return extract_v2(e, p)
    except Exception as ex:  # a corrupt window must not kill the batch
        print("feature failure:", repr(ex))
        return {}


def feature_matrix_v2(ecg, ppg, n_jobs=-1):
    rows = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_safe)(e, p) for e, p in zip(ecg, ppg))
    names = sorted(set().union(*[r.keys() for r in rows]))
    X = np.array([[float(r.get(n, np.nan)) for n in names] for r in rows], dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    X = np.clip(X, -1e6, 1e6)
    return X, names


def alarm_onehot(alarm_type):
    return np.array([[float(a == t) for t in ALARM_TYPES] for a in alarm_type], dtype=np.float32)


def load_features_v2(dataset: str, data: dict | None = None, n_jobs=-1):
    """Cached feature loader. Returns X (NaN allowed), names, alarm one-hot."""
    stem = DATASETS[dataset]
    path = ROOT / "data" / "processed" / f"{stem}.npz"
    cache = path.with_name(stem + "_features_v2.npz")
    if data is None:
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in ("ecg", "ppg", "alarm_type")}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["X"]) == len(data["ppg"]):
            return z["X"], list(z["names"]), z["alarm_onehot"]
    t0 = time.time()
    X, names = feature_matrix_v2(data["ecg"], data["ppg"], n_jobs)
    oh = alarm_onehot(data["alarm_type"])
    print(f"features_v2 {dataset}: {X.shape} in {time.time() - t0:.0f}s")
    np.savez(cache, X=X, names=np.array(names), alarm_onehot=oh, alarm_names=np.array(ALARM_TYPES))
    return X, names, oh


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--n-jobs", type=int, default=16)
    a = ap.parse_args()
    X, names, _ = load_features_v2(a.dataset, n_jobs=a.n_jobs)
    print(X.shape, "nan frac:", float(np.isnan(X).mean()))
