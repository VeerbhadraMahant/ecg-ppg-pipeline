"""PyTorch Dataset and record-wise CV splitting over a preprocessed windows
file (produced by src/data/preprocess.py).

The processed file is an .npz with:
    ecg:        (N, T) float32
    ppg:        (N, T) float32
    label:      (N,) float32, 1.0 = true alarm, 0.0 = false alarm
    quality:    (N,) float32 in [0, 1], min(ecg_quality, ppg_quality)
    record_id:  (N,) str, PhysioNet record name (e.g. "a103l")
    alarm_type: (N,) str, one of the five alarm coding strings
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AlarmWindowDataset(Dataset):
    def __init__(self, ecg: np.ndarray, ppg: np.ndarray, label: np.ndarray):
        self.ecg = torch.from_numpy(ecg.astype(np.float32))
        self.ppg = torch.from_numpy(ppg.astype(np.float32))
        self.label = torch.from_numpy(label.astype(np.float32))

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int):
        return self.ecg[idx], self.ppg[idx], self.label[idx]


def load_processed(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def record_wise_holdout(record_id: np.ndarray, idx: np.ndarray, val_fraction: float, seed: int):
    """Split `idx` into (fit_idx, holdout_idx) by record, so a model-selection
    set (early stopping / best-epoch choice) can be carved out of a training
    fold without ever touching the outer test fold."""
    recs = np.unique(record_id[idx])
    rng = np.random.RandomState(seed)
    rng.shuffle(recs)
    n_val = max(1, int(round(len(recs) * val_fraction)))
    val_recs = set(recs[:n_val])
    in_val = np.array([r in val_recs for r in record_id[idx]])
    return idx[~in_val], idx[in_val]


def record_wise_folds(record_id: np.ndarray, n_folds: int, seed: int):
    """Yield (train_idx, val_idx) with all windows from a given record in
    exactly one side of the split — never split a patient's record across
    train and validation. Records (not windows) are shuffled and dealt round
    -robin into folds so each fold gets a comparable number of records."""
    unique_records = np.unique(record_id)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_records)

    fold_of_record = {}
    for i, rec in enumerate(unique_records):
        fold_of_record[rec] = i % n_folds

    fold_assignment = np.array([fold_of_record[r] for r in record_id])
    for fold in range(n_folds):
        val_idx = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]
        yield train_idx, val_idx
