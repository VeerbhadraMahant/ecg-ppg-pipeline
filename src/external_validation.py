"""Generalization check on MIMIC PERform AF (datasets.md: external test
only, never trained on). Loads the best cross_attention checkpoint from the
Challenge-2015 training run and evaluates it on MIMIC subjects windowed the
same way as training data.

Caveat: the Challenge-2015 model is trained to verify alarm true/false-ness
(does the ECG abnormality have PPG corroboration), while MIMIC PERform AF's
label is AF/non-AF. These are related but distinct tasks, so this is a
generalization/transfer check on whether the learned ECG-PPG agreement
signal is informative for a nearby task, not an apples-to-apples benchmark.

Usage:
    python -m src.external_validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset  # noqa: E402
from src.data.mimic_perform_loader import iter_subjects, load_subject_csv  # noqa: E402
from src.data.signal_ops import bandpass_filter, resample_signal, z_normalize  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult  # noqa: E402


def build_external_windows(cfg: dict) -> dict:
    s = cfg["signal"]
    mimic_dir = ROOT / cfg["paths"]["external_dir"]

    ecgs, ppgs, labels, subject_ids = [], [], [], []
    for subject_id, label, csv_path in iter_subjects(mimic_dir):
        ecg, ppg, fs = load_subject_csv(csv_path)

        ecg_rs = resample_signal(ecg, fs, s["target_fs"])
        ppg_rs = resample_signal(ppg, fs, s["target_fs"])

        win_len = int(s["window_seconds"] * s["target_fs"])
        n_windows = len(ecg_rs) // win_len
        for w in range(n_windows):
            e = ecg_rs[w * win_len:(w + 1) * win_len]
            p = ppg_rs[w * win_len:(w + 1) * win_len]

            e_filt = bandpass_filter(e, s["target_fs"], *s["ecg_band"], order=s["filter_order"])
            p_filt = bandpass_filter(p, s["target_fs"], *s["ppg_band"], order=s["filter_order"])

            ecgs.append(z_normalize(e_filt).astype(np.float32))
            ppgs.append(z_normalize(p_filt).astype(np.float32))
            labels.append(float(label))
            subject_ids.append(subject_id)

    return {
        "ecg": np.stack(ecgs),
        "ppg": np.stack(ppgs),
        "label": np.array(labels, dtype=np.float32),
        "subject_id": np.array(subject_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--variant", default="cross_attention")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("building external validation windows from MIMIC PERform AF...")
    data = build_external_windows(cfg)
    print(f"{len(data['label'])} windows from {len(set(data['subject_id']))} subjects "
          f"({data['label'].mean():.1%} AF)")

    width_mult = get_width_mult(args.variant, ROOT)
    ckpt_dir = ROOT / cfg["paths"]["runs_dir"] / "checkpoints"
    ckpts = sorted(ckpt_dir.glob(f"{args.variant}_seed0_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints for {args.variant} in {ckpt_dir}; run src/train.py first")

    all_metrics = []
    for ckpt_path in ckpts:
        model = build_model(args.variant, cfg, width_mult).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        ds = AlarmWindowDataset(data["ecg"], data["ppg"], data["label"])
        with torch.no_grad():
            logits, _ = model(ds.ecg.to(device), ds.ppg.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()
        m = binary_metrics(data["label"], prob)
        m["checkpoint"] = ckpt_path.name
        all_metrics.append(m)
        print(f"{ckpt_path.name}: F1={m['f1']:.3f} Sens={m['sensitivity']:.3f} "
              f"Spec={m['specificity']:.3f} AUC={m['auc']:.3f}")

    f1s = [m["f1"] for m in all_metrics]
    print(f"\nmean F1 across folds on external MIMIC AF set: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")


if __name__ == "__main__":
    main()
