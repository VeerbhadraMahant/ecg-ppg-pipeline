import numpy as np

from src.features import detect_r_peaks, extract_features, ppg_feet, ptt_series


def _synthetic(hr_period=0.8, ptt=0.25, fs=250, T=2500):
    r = np.arange(0.5, T / fs - 0.5, hr_period)
    ecg = np.zeros(T)
    ppg = np.zeros(T)
    for x in r:
        ecg[int(x * fs)] = 1.0
        ppg[int((x + ptt) * fs)] = 1.0
    ecg = np.convolve(ecg, np.hanning(15), "same")
    ppg = np.convolve(ppg, np.hanning(60), "same")
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    return z(ecg), z(ppg), r


def test_beat_detection_and_heart_rate():
    ecg, ppg, r = _synthetic()
    peaks = detect_r_peaks(ecg)
    assert abs(len(peaks) - len(r)) <= 1
    f = extract_features(ecg, ppg)
    assert abs(f["ecg_hr"] - 75) < 3 and abs(f["ppg_hr"] - 75) < 3
    assert f["hr_abs_diff"] < 3


def test_ptt_is_in_physiological_range_and_pulse_absence_is_visible():
    ecg, ppg, _ = _synthetic()
    f = extract_features(ecg, ppg)
    assert 0.08 <= f["ptt_mean"] <= 0.6
    assert f["ecg_beats_with_pulse_frac"] > 0.8
    # flat PPG (no pulse) => beats have no corroborating pulse
    flat = np.random.RandomState(0).randn(len(ppg)) * 1e-3
    g = extract_features(ecg, flat)
    # periodicity strength separates a real pulse train from noise
    assert f["ppg_acf_peak"] > 0.5 > g["ppg_acf_peak"]
