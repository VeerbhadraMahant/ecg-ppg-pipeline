"""Loader for the PhysioNet/CinC Challenge 2015 training set.

Verified layout (training/ subdirectory of the extracted training.zip):
    RECORDS        - one record name per line, e.g. "a103l"
    ALARMS         - CSV: record_id,alarm_type,label (label: 1 = true alarm, 0 = false alarm)
    <record>.hea   - WFDB header; signal names among "II", "V" (ECG leads),
                     "PLETH" (PPG) or "ABP" (arterial line, not PPG)
    <record>.mat   - WFDB-format binary signal data (despite the .mat extension,
                     this is native WFDB format 16, not a MATLAB file — wfdb
                     reads it directly via the .hea spec)

Per datasets.md: not every record has PLETH (some carry ABP instead); records
without a PLETH channel are skipped when building ECG+PPG pairs.

Per the Challenge 2015 protocol, the alarm fires at the 5-minute mark
(sample index = 300 * fs) in every record; "long" (l) records simply carry
extra history before that point.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import wfdb

ALARM_TRIGGER_SECONDS = 300.0
ECG_LEAD_PRIORITY = ["II", "V"]


def read_alarms(training_dir: Path) -> dict[str, tuple[str, int]]:
    """record_id -> (alarm_type, label)."""
    alarms = {}
    with open(training_dir / "ALARMS", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            record_id, alarm_type, label = row
            alarms[record_id] = (alarm_type, int(label))
    return alarms


def read_records(training_dir: Path) -> list[str]:
    with open(training_dir / "RECORDS") as f:
        return [line.strip() for line in f if line.strip()]


def load_record(training_dir: Path, record_id: str):
    """Returns (ecg_signal, ecg_fs, ppg_signal, ppg_fs) or None if no usable
    PPG channel is present."""
    rec = wfdb.rdrecord(str(training_dir / record_id))
    sig_names = rec.sig_name

    ecg_idx = None
    for lead in ECG_LEAD_PRIORITY:
        if lead in sig_names:
            ecg_idx = sig_names.index(lead)
            break
    if ecg_idx is None:
        return None

    if "PLETH" not in sig_names:
        return None
    ppg_idx = sig_names.index("PLETH")

    ecg = rec.p_signal[:, ecg_idx].astype(np.float64)
    ppg = rec.p_signal[:, ppg_idx].astype(np.float64)

    ecg = np.nan_to_num(ecg, nan=0.0)
    ppg = np.nan_to_num(ppg, nan=0.0)

    return ecg, rec.fs, ppg, rec.fs
