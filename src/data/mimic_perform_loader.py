"""Loader for the MIMIC PERform AF dataset (external test only — never train
on this; datasets.md).

Zenodo distributes two zips, one per class, each containing a per-subject CSV
with (at minimum) ECG and PPG columns sampled at a fixed rate. The exact
column names/rate are confirmed by inspecting the first extracted CSV, since
Zenodo's own docs are the only authority on the export format; this loader
introspects a file's header rather than hard-coding assumptions untested
against the real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_signal_columns(columns: list[str]) -> tuple[str, str]:
    """Best-effort match of ECG/PPG column names across export variants."""
    lower = {c.lower(): c for c in columns}
    ecg_candidates = ["ecg", "ekg"]
    ppg_candidates = ["ppg", "pleth"]

    ecg_col = next((lower[c] for c in ecg_candidates if c in lower), None)
    ppg_col = next((lower[c] for c in ppg_candidates if c in lower), None)
    if ecg_col is None or ppg_col is None:
        raise ValueError(f"could not find ECG/PPG columns among {columns}")
    return ecg_col, ppg_col


def infer_fs(df: pd.DataFrame, time_col: str | None) -> float:
    if time_col is not None and time_col in df.columns:
        dt = np.median(np.diff(df[time_col].values[:1000]))
        return 1.0 / dt
    raise ValueError("no time column found to infer sampling rate")


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Sparse sensor dropouts show up as NaN in the raw CSVs. filtfilt
    propagates a single NaN across the entire output via its IIR recursion,
    so these are linearly interpolated (not zero-filled) before filtering."""
    nan_mask = np.isnan(x)
    if not nan_mask.any():
        return x
    idx = np.arange(len(x))
    x = x.copy()
    x[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    return x


def load_subject_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns if c.lower() in ("time", "time_s", "t")), None)
    ecg_col, ppg_col = find_signal_columns(list(df.columns))
    fs = infer_fs(df, time_col)
    ecg = _interpolate_nans(df[ecg_col].to_numpy(dtype=np.float64))
    ppg = _interpolate_nans(df[ppg_col].to_numpy(dtype=np.float64))
    return ecg, ppg, fs


def iter_subjects(mimic_dir: Path):
    """Yields (subject_id, label, csv_path) for both classes.
    label: 1 = AF, 0 = non-AF."""
    af_dir = mimic_dir / "af"
    non_af_dir = mimic_dir / "non_af"
    for label, d in [(1, af_dir), (0, non_af_dir)]:
        for csv_path in sorted(d.rglob("*.csv")):
            yield csv_path.stem, label, csv_path
