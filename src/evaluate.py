"""Turn runs/results.json into the comparison table and a
performance-vs-signal-quality breakdown (proposal.md objectives 3 and 4).

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset, load_processed, record_wise_folds  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.models.classifier import build_model, VARIANTS  # noqa: E402
from src.train import get_width_mult  # noqa: E402


def comparison_table(results_path: Path) -> pd.DataFrame:
    results = json.loads(results_path.read_text())
    rows = []
    for variant in VARIANTS:
        if variant not in results:
            continue
        s = results[variant]
        rows.append(
            {
                "variant": variant,
                "params": s["fold_metrics"][0]["params"] if s["fold_metrics"] else None,
                "F1": f"{s['f1_mean']:.3f} +/- {s['f1_std']:.3f}",
                "Sensitivity": f"{s['sensitivity_mean']:.3f} +/- {s['sensitivity_std']:.3f}",
                "Specificity": f"{s['specificity_mean']:.3f} +/- {s['specificity_std']:.3f}",
                "AUC": f"{s['auc_mean']:.3f} +/- {s['auc_std']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def quality_stratified_eval(cfg: dict, data: dict, variant: str, device: str, n_bins: int = 3) -> pd.DataFrame:
    """Bucket the validation windows of the last fold by signal quality and
    report metrics per bucket — shows whether/when fusion helps most, per
    proposal.md objective on quality-dependent performance."""
    ecg, ppg, label, record_id, quality = (
        data["ecg"], data["ppg"], data["label"], data["record_id"], data["quality"]
    )
    width_mult = get_width_mult(variant, ROOT)
    ckpt_dir = ROOT / cfg["paths"]["runs_dir"] / "checkpoints"
    ckpts = sorted(ckpt_dir.glob(f"{variant}_seed0_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found for {variant} in {ckpt_dir}")

    fold_i = len(ckpts) - 1
    train_idx, val_idx = list(record_wise_folds(record_id, cfg["split"]["n_folds"], cfg["seed"]))[fold_i]

    model = build_model(variant, cfg, width_mult).to(device)
    model.load_state_dict(torch.load(ckpt_dir / f"{variant}_seed0_fold{fold_i}.pt", map_location=device))
    model.eval()

    val_ds = AlarmWindowDataset(ecg[val_idx], ppg[val_idx], label[val_idx])
    val_quality = quality[val_idx]

    with torch.no_grad():
        e = val_ds.ecg.to(device)
        p = val_ds.ppg.to(device)
        logits, _ = model(e, p)
        prob = torch.sigmoid(logits).cpu().numpy()
    y = val_ds.label.numpy()

    bins = np.quantile(val_quality, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    bin_idx = np.digitize(val_quality, bins[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        m = binary_metrics(y[mask], prob[mask])
        rows.append({"quality_bin": b, "n": int(mask.sum()), "F1": m["f1"], "Sensitivity": m["sensitivity"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    results_path = ROOT / cfg["paths"]["runs_dir"] / "results.json"
    table = comparison_table(results_path)
    print("\n=== Baseline ladder comparison ===")
    print(table.to_string(index=False))
    table.to_csv(ROOT / cfg["paths"]["runs_dir"] / "comparison_table.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_path = Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz"
    data = load_processed(processed_path)

    print("\n=== Performance vs. signal quality (cross_attention) ===")
    q_table = quality_stratified_eval(cfg, data, "cross_attention", device)
    print(q_table.to_string(index=False))


if __name__ == "__main__":
    main()
