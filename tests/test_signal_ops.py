import numpy as np

from src.data.signal_ops import (
    bandpass_filter,
    clip_fraction,
    flatline_fraction,
    resample_signal,
    signal_quality_score,
    z_normalize,
)


def test_resample_signal_preserves_duration():
    fs_in, fs_out = 100, 250
    x = np.sin(2 * np.pi * 1.0 * np.arange(fs_in * 5) / fs_in)
    y = resample_signal(x, fs_in, fs_out)
    assert abs(len(y) / fs_out - len(x) / fs_in) < 0.05


def test_bandpass_filter_removes_dc_offset():
    fs = 250
    t = np.arange(fs * 4) / fs
    x = 5.0 + np.sin(2 * np.pi * 5 * t)  # DC offset + 5Hz tone
    y = bandpass_filter(x, fs, 0.5, 40)
    assert abs(np.mean(y)) < 0.5


def test_z_normalize_zero_mean_unit_std():
    x = np.random.RandomState(0).randn(1000) * 3 + 7
    y = z_normalize(x)
    assert abs(np.mean(y)) < 1e-6
    assert abs(np.std(y) - 1.0) < 1e-6


def test_flatline_fraction_detects_constant_signal():
    x = np.ones(1000) * 3.0
    assert flatline_fraction(x) == 1.0


def test_flatline_fraction_low_for_varying_signal():
    x = np.sin(2 * np.pi * np.arange(1000) / 50)
    assert flatline_fraction(x) < 0.5


def test_clip_fraction_detects_saturation():
    x = np.concatenate([np.full(500, 1.0), np.linspace(-1, 1, 500)])
    assert clip_fraction(x) > 0.4


def test_signal_quality_score_clean_signal_high():
    x = np.sin(2 * np.pi * np.arange(2500) / 50)
    q = signal_quality_score(x, flatline_std_threshold=0.01, clip_fraction_threshold=0.2)
    assert q > 0.5


def test_signal_quality_score_flatline_low():
    x = np.zeros(2500)
    q = signal_quality_score(x, flatline_std_threshold=0.01, clip_fraction_threshold=0.2)
    assert q < 0.5
