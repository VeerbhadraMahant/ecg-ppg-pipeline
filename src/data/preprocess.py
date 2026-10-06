"""Turn the raw PhysioNet Challenge 2015 training set into fixed-length,
aligned, filtered, z-normalized ECG/PPG window pairs with per-window quality
scores, per architecture.md section 1 and proposal.md's methodology.

Usage:
    python -m src.data.preprocess
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.challenge2015_loader import (  # noqa: E402
    ALARM_TRIGGER_SECONDS,
    load_record,
    load_record_multi,
    read_alarms,
    read_records,
)
from src.data.signal_ops import (  # noqa: E402
    bandpass_filter,
    resample_signal,
    signal_quality_score,
    z_normalize,
)


def extract_window(sig: np.ndarray, fs: float, trigger_s: float, window_s: float) -> np.ndarray | None:
    """Window ends AT the alarm trigger (no post-alarm data used).

    Challenge 2015 "short" records run exactly to the 300s trigger with no
    signal after it; "long" records carry extra history before that point.
    Anchoring the window's end to the trigger (rather than centering on it)
    is the only alignment that works for both without discarding the short
    records outright.
    """
    end = int(round(trigger_s * fs))
    start = end - int(round(window_s * fs))
    if start < 0 or end > len(sig):
        return None
    return sig[start:end]


def prep_channel(sig, fs, band, cfg):
    """Resample, window (ending at the alarm trigger), filter, z-normalize and
    score one channel. Returns (normalized window float32, quality) or None."""
    s = cfg["signal"]
    q = cfg["quality"]
    rs = resample_signal(sig, fs, s["target_fs"])
    win = extract_window(rs, s["target_fs"], ALARM_TRIGGER_SECONDS, s["window_seconds"])
    if win is None:
        return None
    filt = bandpass_filter(win, s["target_fs"], *band, order=s["filter_order"])
    quality = signal_quality_score(win, q["flatline_std_threshold"], q["clip_fraction_threshold"])
    return z_normalize(filt).astype(np.float32), float(quality)


def process_record(ecg, ecg_fs, ppg, ppg_fs, cfg) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Back-compatible ECG+PPG pair processor (min quality over both)."""
    s = cfg["signal"]
    e = prep_channel(ecg, ecg_fs, s["ecg_band"], cfg)
    p = prep_channel(ppg, ppg_fs, s["ppg_band"], cfg)
    if e is None or p is None:
        return None
    return e[0], p[0], min(e[1], p[1])


def _write_cohort(path: Path, rows: list[dict], T: int) -> None:
    def stack(key):
        return np.stack([r[key] for r in rows]) if rows else np.zeros((0, T), np.float32)

    np.savez(
        path,
        ecg=stack("ecg"),
        ppg=stack("pulse"),  # PPG, or ABP where PPG is absent (see pulse_is_abp)
        ecg2=stack("ecg2"),
        abp=stack("abp"),
        label=np.array([r["label"] for r in rows], dtype=np.float32),
        quality=np.array([r["quality"] for r in rows], dtype=np.float32),
        ecg_quality=np.array([r["ecg_q"] for r in rows], dtype=np.float32),
        pulse_quality=np.array([r["pulse_q"] for r in rows], dtype=np.float32),
        record_id=np.array([r["record_id"] for r in rows]),
        alarm_type=np.array([r["alarm_type"] for r in rows]),
        pulse_is_abp=np.array([r["pulse_is_abp"] for r in rows], dtype=bool),
        has_ecg2=np.array([r["has_ecg2"] for r in rows], dtype=bool),
        has_abp=np.array([r["has_abp"] for r in rows], dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--min-quality", type=float, default=0.05,
                         help="drop windows below this quality (near-total flatline/clipping)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    s = cfg["signal"]
    training_dir = ROOT / cfg["paths"]["raw_dir"] / "training"
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    T = int(s["window_seconds"] * s["target_fs"])
    zeros = np.zeros(T, np.float32)

    alarms = read_alarms(training_dir)
    records = read_records(training_dir)

    ppg_rows, recovered_rows, audit = [], [], []

    for record_id in tqdm(records, desc="preprocessing"):
        if record_id not in alarms:
            continue
        alarm_type, label = alarms[record_id]
        entry = {"record_id": record_id, "alarm_type": alarm_type, "label": label,
                 "signals": "", "has_ppg": False, "has_abp": False, "has_ecg2": False,
                 "in_ppg_cohort": False, "in_recovered_cohort": False, "reason": ""}

        loaded = load_record_multi(training_dir, record_id)
        if loaded is None:
            entry["reason"] = "no_ecg"
            audit.append(entry)
            continue
        fs = loaded["fs"]
        entry["signals"] = "|".join(loaded["sig_names"])
        entry["has_ppg"] = loaded["ppg"] is not None
        entry["has_abp"] = loaded["abp"] is not None
        entry["has_ecg2"] = loaded["ecg2"] is not None

        ecg = prep_channel(loaded["ecg"], fs, s["ecg_band"], cfg)
        ecg2 = prep_channel(loaded["ecg2"], fs, s["ecg_band"], cfg) if entry["has_ecg2"] else None
        ppg = prep_channel(loaded["ppg"], fs, s["ppg_band"], cfg) if entry["has_ppg"] else None
        abp = prep_channel(loaded["abp"], fs, s["ppg_band"], cfg) if entry["has_abp"] else None

        if ecg is None:
            entry["reason"] = "window_too_short"
            audit.append(entry)
            continue

        def make_row(pulse, pulse_is_abp):
            quality = min(ecg[1], pulse[1])
            return {
                "record_id": record_id, "alarm_type": alarm_type, "label": float(label),
                "ecg": ecg[0], "pulse": pulse[0],
                "ecg2": ecg2[0] if ecg2 else zeros, "abp": abp[0] if abp else zeros,
                "ecg_q": ecg[1], "pulse_q": pulse[1], "quality": quality,
                "pulse_is_abp": pulse_is_abp, "has_ecg2": ecg2 is not None, "has_abp": abp is not None,
            }

        if ppg is not None and ppg[1] >= args.min_quality and min(ecg[1], ppg[1]) >= args.min_quality:
            row = make_row(ppg, False)
            ppg_rows.append(row)
            recovered_rows.append(row)
            entry["in_ppg_cohort"] = entry["in_recovered_cohort"] = True
        elif abp is not None and min(ecg[1], abp[1]) >= args.min_quality:
            recovered_rows.append(make_row(abp, True))
            entry["in_recovered_cohort"] = True
            entry["reason"] = "recovered_via_abp" if not entry["has_ppg"] else "recovered_via_abp_ppg_bad"
        else:
            entry["reason"] = ("no_ppg_no_abp" if not (entry["has_ppg"] or entry["has_abp"])
                               else "window_or_quality")
        audit.append(entry)

    import pandas as pd

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(out_dir / "cohort_audit.csv", index=False)

    n = len(audit_df)
    flow = {
        "records_in_training_set": n,
        "with_ppg": int(audit_df.has_ppg.sum()),
        "with_abp": int(audit_df.has_abp.sum()),
        "with_second_ecg_lead": int(audit_df.has_ecg2.sum()),
        "ppg_cohort": int(audit_df.in_ppg_cohort.sum()),
        "recovered_cohort_ppg_or_abp": int(audit_df.in_recovered_cohort.sum()),
        "recovered_via_abp": int((audit_df.reason.str.startswith("recovered_via_abp")).sum()),
        "excluded_reasons": audit_df[~audit_df.in_recovered_cohort].reason.value_counts().to_dict(),
    }
    import json

    (out_dir / "cohort_flow.json").write_text(json.dumps(flow, indent=2))
    print("cohort flow:", json.dumps(flow, indent=2))

    _write_cohort(out_dir / "challenge2015_windows.npz", ppg_rows, T)
    _write_cohort(out_dir / "challenge2015_windows_recovered.npz", recovered_rows, T)
    for name, rows in [("challenge2015_windows.npz (PPG cohort)", ppg_rows),
                       ("challenge2015_windows_recovered.npz (PPG or ABP)", recovered_rows)]:
        lab = np.array([r["label"] for r in rows])
        print(f"wrote {name}: {len(rows)} records, {lab.mean():.1%} true alarms ({int(lab.sum())}/{len(lab)})")


if __name__ == "__main__":
    main()
