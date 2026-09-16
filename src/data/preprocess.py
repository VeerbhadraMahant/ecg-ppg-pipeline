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


def process_record(ecg, ecg_fs, ppg, ppg_fs, cfg) -> tuple[np.ndarray, np.ndarray, float] | None:
    s = cfg["signal"]
    target_fs = s["target_fs"]

    ecg_rs = resample_signal(ecg, ecg_fs, target_fs)
    ppg_rs = resample_signal(ppg, ppg_fs, target_fs)

    ecg_win = extract_window(ecg_rs, target_fs, ALARM_TRIGGER_SECONDS, s["window_seconds"])
    ppg_win = extract_window(ppg_rs, target_fs, ALARM_TRIGGER_SECONDS, s["window_seconds"])
    if ecg_win is None or ppg_win is None:
        return None

    ecg_filt = bandpass_filter(ecg_win, target_fs, *s["ecg_band"], order=s["filter_order"])
    ppg_filt = bandpass_filter(ppg_win, target_fs, *s["ppg_band"], order=s["filter_order"])

    ecg_norm = z_normalize(ecg_filt)
    ppg_norm = z_normalize(ppg_filt)

    q = cfg["quality"]
    ecg_q = signal_quality_score(ecg_win, q["flatline_std_threshold"], q["clip_fraction_threshold"])
    ppg_q = signal_quality_score(ppg_win, q["flatline_std_threshold"], q["clip_fraction_threshold"])
    quality = min(ecg_q, ppg_q)

    return ecg_norm.astype(np.float32), ppg_norm.astype(np.float32), quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--min-quality", type=float, default=0.05,
                         help="drop windows below this quality (near-total flatline/clipping)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    training_dir = ROOT / cfg["paths"]["raw_dir"] / "training"
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    alarms = read_alarms(training_dir)
    records = read_records(training_dir)

    ecgs, ppgs, labels, qualities, record_ids, alarm_types = [], [], [], [], [], []
    skipped_no_ppg = 0
    skipped_window = 0
    skipped_quality = 0

    for record_id in tqdm(records, desc="preprocessing"):
        if record_id not in alarms:
            continue
        alarm_type, label = alarms[record_id]

        loaded = load_record(training_dir, record_id)
        if loaded is None:
            skipped_no_ppg += 1
            continue
        ecg, ecg_fs, ppg, ppg_fs = loaded

        result = process_record(ecg, ecg_fs, ppg, ppg_fs, cfg)
        if result is None:
            skipped_window += 1
            continue
        ecg_norm, ppg_norm, quality = result

        if quality < args.min_quality:
            skipped_quality += 1
            continue

        ecgs.append(ecg_norm)
        ppgs.append(ppg_norm)
        labels.append(float(label))
        qualities.append(quality)
        record_ids.append(record_id)
        alarm_types.append(alarm_type)

    print(f"\nkept {len(ecgs)} / {len(records)} records "
          f"(skipped: no_ppg={skipped_no_ppg}, window={skipped_window}, quality={skipped_quality})")

    out_path = out_dir / "challenge2015_windows.npz"
    np.savez(
        out_path,
        ecg=np.stack(ecgs),
        ppg=np.stack(ppgs),
        label=np.array(labels, dtype=np.float32),
        quality=np.array(qualities, dtype=np.float32),
        record_id=np.array(record_ids),
        alarm_type=np.array(alarm_types),
    )
    print(f"wrote {out_path}")

    labels_arr = np.array(labels)
    print(f"label balance: {labels_arr.mean():.1%} true alarms ({int(labels_arr.sum())}/{len(labels_arr)})")


if __name__ == "__main__":
    main()
