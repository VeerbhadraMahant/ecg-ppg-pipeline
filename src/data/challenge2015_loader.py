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
ECG_LEAD_NAMES = ["II", "V", "I", "III", "AVR", "AVL", "AVF", "MCL", "MCL1"]


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


def load_record_multi(training_dir: Path, record_id: str) -> dict | None:
    """Loads every usable channel of a record (updates.md 3.3: recover records
    that lack PPG but carry ABP, and expose a second ECG lead).

    Returns a dict with float64 arrays (None where absent): ecg, ecg2, ppg,
    abp, plus fs and the raw signal-name list, or None when there is no ECG.
    The second ECG lead is the next lead in ECG_LEAD_NAMES order that differs
    from the primary one.
    """
    rec = wfdb.rdrecord(str(training_dir / record_id))
    names = [n.upper() for n in rec.sig_name]

    ecg_idx = [names.index(n) for n in ECG_LEAD_NAMES if n in names]
    # fall back to any lead-like channel that is not a pulse/resp channel
    if not ecg_idx:
        ecg_idx = [i for i, n in enumerate(names) if n not in ("PLETH", "ABP", "RESP")]
    if not ecg_idx:
        return None

    def col(i):
        return None if i is None else np.nan_to_num(rec.p_signal[:, i].astype(np.float64), nan=0.0)

    return {
        "ecg": col(ecg_idx[0]),
        "ecg2": col(ecg_idx[1]) if len(ecg_idx) > 1 else None,
        "ppg": col(names.index("PLETH")) if "PLETH" in names else None,
        "abp": col(names.index("ABP")) if "ABP" in names else None,
        "fs": rec.fs,
        "sig_names": list(rec.sig_name),
    }
