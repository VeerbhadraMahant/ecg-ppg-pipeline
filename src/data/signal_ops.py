"""Signal-level operations shared by all loaders: resampling, filtering,
normalization, and quality scoring. Pure functions on numpy arrays so they
are trivially unit-testable without touching disk.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample a 1D signal from fs_in to fs_out via polyphase filtering."""
    if fs_in == fs_out:
        return x.astype(np.float64)
    from math import gcd

    g = gcd(int(round(fs_in)), int(round(fs_out)))
    up = int(round(fs_out)) // g
    down = int(round(fs_in)) // g
    return sp_signal.resample_poly(x, up, down).astype(np.float64)


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. `high` is clamped below Nyquist."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    low = max(low, 0.01)
    sos = sp_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, x)


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-window z-normalization so absolute amplitude (sensor gain,
    placement) cannot be exploited by the model."""
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / (sigma + eps)


def flatline_fraction(x: np.ndarray, window: int = 25) -> float:
    """Fraction of the signal where local std is near zero (disconnected
    lead / flatline)."""
    if len(x) < window:
        window = max(1, len(x))
    n_windows = len(x) // window
    if n_windows == 0:
        return 1.0
    trimmed = x[: n_windows * window].reshape(n_windows, window)
    local_std = trimmed.std(axis=1)
    return float(np.mean(local_std < 1e-3 * (np.std(x) + 1e-8)))


def clip_fraction(x: np.ndarray) -> float:
    """Fraction of samples sitting at (or within 0.1%) of the signal's own
    min/max rails — a proxy for ADC saturation."""
    lo, hi = np.min(x), np.max(x)
    span = hi - lo
    if span < 1e-8:
        return 1.0
    tol = 0.001 * span
    at_rail = (x <= lo + tol) | (x >= hi - tol)
    return float(np.mean(at_rail))


def signal_quality_score(x: np.ndarray, flatline_std_threshold: float, clip_fraction_threshold: float) -> float:
    """Composite quality score in [0, 1]; 1 = clean, 0 = unusable.

    Combines flatline detection and clipping/saturation detection. This is a
    lightweight heuristic score, not a clinical SQI — good enough to filter
    obviously bad windows and to stratify the performance-vs-quality analysis.
    """
    flat = flatline_fraction(x)
    clip = clip_fraction(x)
    flat_penalty = min(flat / max(flatline_std_threshold, 1e-6), 1.0)
    clip_penalty = min(clip / max(clip_fraction_threshold, 1e-6), 1.0)
    return float(max(0.0, 1.0 - 0.5 * flat_penalty - 0.5 * clip_penalty))
