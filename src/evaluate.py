"""Turn runs/results.json into the comparison table and the analyses
proposal.md's methodology commits to: performance vs. signal quality,
sensitivity/specificity/F1 per alarm class, and model size/speed alongside
accuracy.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset, load_processed, record_wise_folds  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.models.classifier import build_model, count_params, VARIANTS  # noqa: E402
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


def collect_oof_predictions(cfg: dict, data: dict, variant: str, device: str, seed: int = 0) -> dict:
    """Every window gets exactly one prediction, from the fold checkpoint
    that held it out of training (out-of-fold). This covers the full
    dataset rather than a single fold's ~20% validation slice, so
    downstream breakdowns (per-alarm-type, per-quality-bin) have enough
    support to be meaningful."""
    ecg, ppg, label, record_id, quality, alarm_type = (
        data["ecg"], data["ppg"], data["label"], data["record_id"], data["quality"], data["alarm_type"]
    )
    width_mult = get_width_mult(variant, ROOT)
    ckpt_dir = ROOT / cfg["paths"]["runs_dir"] / "checkpoints"
    n_folds = cfg["split"]["n_folds"]

    n = len(label)
    prob = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)

    for fold_i, (_, val_idx) in enumerate(record_wise_folds(record_id, n_folds, cfg["seed"] + seed)):
        ckpt_path = ckpt_dir / f"{variant}_seed{seed}_fold{fold_i}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint {ckpt_path}; run src/train.py first")
        model = build_model(variant, cfg, width_mult).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        val_ds = AlarmWindowDataset(ecg[val_idx], ppg[val_idx], label[val_idx])
        with torch.no_grad():
            logits, _ = model(val_ds.ecg.to(device), val_ds.ppg.to(device))
            prob[val_idx] = torch.sigmoid(logits).cpu().numpy()
        covered[val_idx] = True

    assert covered.all(), "record_wise_folds did not partition every window across folds"
    return {"label": label, "prob": prob, "alarm_type": alarm_type, "quality": quality, "record_id": record_id}


def quality_stratified_eval(cfg: dict, data: dict, variant: str, device: str, n_bins: int = 3) -> pd.DataFrame:
    """Bucket every window (via out-of-fold predictions) by signal quality
    and report metrics per bucket — shows whether/when fusion helps most,
    per proposal.md's quality-dependent performance analysis."""
    oof = collect_oof_predictions(cfg, data, variant, device)
    y, prob, quality = oof["label"], oof["prob"], oof["quality"]

    bins = np.quantile(quality, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    bin_idx = np.digitize(quality, bins[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        m = binary_metrics(y[mask], prob[mask])
        rows.append({"quality_bin": b, "n": int(mask.sum()), "F1": m["f1"], "Sensitivity": m["sensitivity"]})
    return pd.DataFrame(rows)


def per_alarm_type_eval(cfg: dict, data: dict, variant: str, device: str) -> pd.DataFrame:
    """Sensitivity/specificity/F1 broken down by the five Challenge-2015
    alarm types, via out-of-fold predictions over the full dataset —
    proposal.md's methodology commitment this pipeline was missing."""
    oof = collect_oof_predictions(cfg, data, variant, device)
    y, prob, alarm_type = oof["label"], oof["prob"], oof["alarm_type"]

    rows = []
    for at in sorted(set(alarm_type)):
        mask = alarm_type == at
        m = binary_metrics(y[mask], prob[mask])
        rows.append({
            "alarm_type": at, "n": int(mask.sum()), "n_true_alarms": int(y[mask].sum()),
            "F1": m["f1"], "Sensitivity": m["sensitivity"], "Specificity": m["specificity"], "AUC": m["auc"],
        })
    return pd.DataFrame(rows)


def benchmark_speed(cfg: dict, variant: str, device: str, n_warmup: int = 10, n_iters: int = 100) -> dict:
    """Single-window inference latency and throughput, reported alongside
    accuracy per proposal.md ('model size/speed reported alongside
    accuracy') and needed to justify architecture.md's small-model choice."""
    width_mult = get_width_mult(variant, ROOT)
    model = build_model(variant, cfg, width_mult).to(device)
    model.eval()

    win_len = int(cfg["signal"]["window_seconds"] * cfg["signal"]["target_fs"])
    ecg = torch.randn(1, win_len, device=device)
    ppg = torch.randn(1, win_len, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(ecg, ppg)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_iters):
            model(ecg, ppg)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    ms_per_window = 1000 * elapsed / n_iters
    return {
        "variant": variant,
        "params": count_params(model),
        "latency_ms": ms_per_window,
        "throughput_per_sec": 1000 / ms_per_window,
    }


def significance_test(results_path: Path, variant_a: str, variant_b: str) -> dict:
    """Paired t-test on matched (seed, fold) F1 scores between two variants
    — every fold trains both models on the identical train/val split, so
    the pairing is exact."""
    results = json.loads(results_path.read_text())
    fm_a = {(m["seed"], m["fold"]): m["f1"] for m in results[variant_a]["fold_metrics"]}
    fm_b = {(m["seed"], m["fold"]): m["f1"] for m in results[variant_b]["fold_metrics"]}
    keys = sorted(set(fm_a) & set(fm_b))
    a = np.array([fm_a[k] for k in keys])
    b = np.array([fm_b[k] for k in keys])
    t_stat, p_value = stats.ttest_rel(b, a)
    return {
        "variant_a": variant_a, "variant_b": variant_b, "n_pairs": len(keys),
        "mean_diff": float(b.mean() - a.mean()), "t_stat": float(t_stat), "p_value": float(p_value),
    }


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

    print("\n=== Performance vs. signal quality (cross_attention, out-of-fold) ===")
    q_table = quality_stratified_eval(cfg, data, "cross_attention", device)
    print(q_table.to_string(index=False))
    q_table.to_csv(ROOT / cfg["paths"]["runs_dir"] / "quality_stratified.csv", index=False)

    print("\n=== Sensitivity/Specificity/F1 per alarm type (cross_attention, out-of-fold) ===")
    at_table = per_alarm_type_eval(cfg, data, "cross_attention", device)
    print(at_table.to_string(index=False))
    at_table.to_csv(ROOT / cfg["paths"]["runs_dir"] / "per_alarm_type.csv", index=False)

    print("\n=== Model size / inference speed ===")
    speed_rows = [benchmark_speed(cfg, v, device) for v in VARIANTS]
    speed_table = pd.DataFrame(speed_rows)
    print(speed_table.to_string(index=False))
    speed_table.to_csv(ROOT / cfg["paths"]["runs_dir"] / "speed_benchmark.csv", index=False)

    print("\n=== Statistical significance: cross_attention vs. concat (paired t-test on F1) ===")
    sig = significance_test(results_path, "concat", "cross_attention")
    print(f"mean F1 diff = {sig['mean_diff']:+.4f}, t = {sig['t_stat']:.3f}, "
          f"p = {sig['p_value']:.4f} (n={sig['n_pairs']} matched seed/fold pairs)")
    with open(ROOT / cfg["paths"]["runs_dir"] / "significance_test.json", "w") as f:
        json.dump(sig, f, indent=2)


if __name__ == "__main__":
    main()
