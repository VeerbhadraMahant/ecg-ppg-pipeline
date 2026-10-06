"""VTaC v1.1 -> fixed-length, decision-time windows (updates.md 2.1 / 4.1).

Raw events (scripts/download_vtac.py) are 90 s segments at 250 Hz:
[240 s, 330 s] of the original 6-minute record, so the alarm onset sits at
sample ONSET = 60 s * 250 = 15000.

LEAKAGE CONTROL: the headline task window is the `window_seconds` ending
exactly at the onset (no post-alarm samples). Post-alarm samples are only
read by the time-to-verdict analysis via `post_s`.

Outputs (same schema as the CinC npz, plus extras):
    ecg, ppg (pulse: PLETH, else ABP), ecg2, abp, label, quality,
    record_id (patient), event_id, split (official train/val/test),
    alarm_type ("Ventricular_Tachycardia"), pulse_is_abp, has_ecg2,
    annotator_frac_true / n_annotators (label reliability),
    lead_signature (channel-set string, a PROXY for monitor manufacturer:
    VTaC metadata does not release hospital or device identifiers).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.signal_ops import bandpass_filter, signal_quality_score, z_normalize

FS = 250
ONSET = 60 * FS  # index of the alarm onset inside a stored 90 s segment
ECG_PRIORITY = ["II", "V", "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "MCL"]


def read_labels(vtac_dir: Path) -> dict:
    lab = {}
    with open(vtac_dir / "event_labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            lab[r["event"]] = (r["record"], r["decision"].strip().lower() == "true")
    return lab


def read_split(vtac_dir: Path) -> dict:
    with open(vtac_dir / "benchmark_data_split.csv", newline="") as f:
        return {r["event"]: r["split"] for r in csv.DictReader(f)}


def read_annotators(vtac_dir: Path) -> dict:
    p = vtac_dir / "event_label_per_annotator.csv"
    votes = defaultdict(list)
    if p.exists():
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["is_adjudication"].strip().upper() == "TRUE":
                    continue  # adjudicated rows are the consensus, not an independent vote
                votes[r["event"]].append(r["decision"].strip().upper() == "TRUE")
    return votes


def _col(sig, names, key):
    return sig[:, names.index(key)].astype(np.float64) if key in names else None


def split_channels(sig: np.ndarray, names: list[str]) -> dict:
    ecg_names = [n for n in ECG_PRIORITY if n in names]
    if not ecg_names:
        ecg_names = [n for n in names if n not in ("PLETH", "ABP", "RESP")]
    return {
        "ecg": _col(sig, names, ecg_names[0]) if ecg_names else None,
        "ecg2": _col(sig, names, ecg_names[1]) if len(ecg_names) > 1 else None,
        "ppg": _col(sig, names, "PLETH"),
        "abp": _col(sig, names, "ABP"),
    }


def window_at_onset(x: np.ndarray, window_s: float, post_s: float = 0.0) -> np.ndarray | None:
    """`window_s` seconds ending `post_s` after the onset (post_s = 0 -> decision time)."""
    end = ONSET + int(round(post_s * FS))
    start = end - int(round(window_s * FS))
    if start < 0 or end > len(x):
        return None
    return x[start:end]


def prep(x: np.ndarray | None, band, cfg: dict, window_s: float, post_s: float = 0.0):
    if x is None:
        return None
    w = window_at_onset(x, window_s, post_s)
    if w is None:
        return None
    q = cfg["quality"]
    filt = bandpass_filter(w, FS, *band, order=cfg["signal"]["filter_order"])
    quality = float(signal_quality_score(w, q["flatline_std_threshold"], q["clip_fraction_threshold"]))
    return z_normalize(filt).astype(np.float32), quality


def build_vtac_windows(vtac_dir: Path, cfg: dict, min_quality: float = 0.05, post_s: float = 0.0,
                       window_s: float | None = None) -> dict:
    s = cfg["signal"]
    window_s = window_s or s["window_seconds"]
    labels, splits, votes = read_labels(vtac_dir), read_split(vtac_dir), read_annotators(vtac_dir)
    T = int(window_s * FS)
    zeros = np.zeros(T, np.float32)

    rows = []
    audit = {"events_total": 0, "no_pulse": 0, "no_ecg": 0, "low_quality": 0}
    for ev_file in sorted((vtac_dir / "events").glob("*.npz")):
        ev = ev_file.stem
        if ev not in labels:
            continue
        audit["events_total"] += 1
        z = np.load(ev_file)
        names = [str(n) for n in z["names"]]
        sig = z["sig"].astype(np.float64)
        ch = split_channels(sig, names)
        ecg = prep(ch["ecg"], s["ecg_band"], cfg, window_s, post_s)
        if ecg is None:
            audit["no_ecg"] += 1
            continue
        ppg = prep(ch["ppg"], s["ppg_band"], cfg, window_s, post_s)
        abp = prep(ch["abp"], s["ppg_band"], cfg, window_s, post_s)
        ecg2 = prep(ch["ecg2"], s["ecg_band"], cfg, window_s, post_s)
        pulse, is_abp = (ppg, False) if ppg is not None else (abp, True)
        if pulse is None:
            audit["no_pulse"] += 1
            continue
        quality = min(ecg[1], pulse[1])
        if quality < min_quality:
            audit["low_quality"] += 1
            continue
        v = votes.get(ev, [])
        rows.append({
            "ecg": ecg[0], "ppg": pulse[0], "ecg2": ecg2[0] if ecg2 else zeros, "abp": abp[0] if abp else zeros,
            "label": float(labels[ev][1]), "quality": quality, "record_id": labels[ev][0], "event_id": ev,
            "split": splits.get(ev, "none"), "pulse_is_abp": is_abp, "has_ecg2": ecg2 is not None,
            "frac_true": float(np.mean(v)) if v else np.nan, "n_ann": len(v),
            "lead_sig": "|".join(n for n in names if n not in ("PLETH", "ABP", "RESP")),
        })
    audit["kept"] = len(rows)
    if not rows:
        raise RuntimeError(f"no VTaC events found in {vtac_dir / 'events'}")
    out = {
        "ecg": np.stack([r["ecg"] for r in rows]), "ppg": np.stack([r["ppg"] for r in rows]),
        "ecg2": np.stack([r["ecg2"] for r in rows]), "abp": np.stack([r["abp"] for r in rows]),
        "label": np.array([r["label"] for r in rows], np.float32),
        "quality": np.array([r["quality"] for r in rows], np.float32),
        "record_id": np.array([r["record_id"] for r in rows]), "event_id": np.array([r["event_id"] for r in rows]),
        "split": np.array([r["split"] for r in rows]),
        "alarm_type": np.array(["Ventricular_Tachycardia"] * len(rows)),
        "pulse_is_abp": np.array([r["pulse_is_abp"] for r in rows]),
        "has_ecg2": np.array([r["has_ecg2"] for r in rows]),
        "annotator_frac_true": np.array([r["frac_true"] for r in rows], np.float32),
        "n_annotators": np.array([r["n_ann"] for r in rows]),
        "lead_signature": np.array([r["lead_sig"] for r in rows]),
    }
    out["_audit"] = audit
    return out
