"""Train one baseline-ladder variant with record-wise cross-validation and
multiple seeds, per proposal.md ('error bars across folds/seeds, not a
single run').

Usage:
    python -m src.train --variant cross_attention
    python -m src.train --variant all --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset, load_processed, record_wise_folds  # noqa: E402
from src.metrics import binary_metrics, summarize_across_folds  # noqa: E402
from src.models.classifier import build_model, count_params  # noqa: E402
from src.models.losses import build_loss  # noqa: E402
from src.models.classifier import VARIANTS  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_width_mult(variant: str, root: Path) -> float:
    wm_path = root / "configs" / "width_mult.yaml"
    if wm_path.exists():
        wm = yaml.safe_load(wm_path.read_text())
        return float(wm.get(variant, 1.0))
    return 1.0


def train_one_fold(
    cfg: dict,
    variant: str,
    width_mult: float,
    train_ds: AlarmWindowDataset,
    val_ds: AlarmWindowDataset,
    device: str,
    seed: int,
) -> tuple[dict, torch.nn.Module]:
    t = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"] * 2, shuffle=False)

    model = build_model(variant, cfg, width_mult).to(device)

    n_pos = train_ds.label.sum().item()
    n_neg = len(train_ds) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    loss_fn = build_loss(t["loss"], pos_weight, t.get("focal_gamma", 2.0), device)

    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    best_f1 = -1.0
    best_state = None
    patience_left = t["early_stop_patience"]

    for epoch in range(t["epochs"]):
        model.train()
        for ecg, ppg, y in train_loader:
            ecg, ppg, y = ecg.to(device), ppg.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(ecg, ppg)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_prob, all_y = [], []
        with torch.no_grad():
            for ecg, ppg, y in val_loader:
                ecg, ppg = ecg.to(device), ppg.to(device)
                logits, _ = model(ecg, ppg)
                all_prob.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(y.numpy())
        val_prob = np.concatenate(all_prob)
        val_y = np.concatenate(all_y)
        m = binary_metrics(val_y, val_prob)

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = t["early_stop_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_prob, all_y = [], []
    with torch.no_grad():
        for ecg, ppg, y in val_loader:
            ecg, ppg = ecg.to(device), ppg.to(device)
            logits, _ = model(ecg, ppg)
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(y.numpy())
    val_prob = np.concatenate(all_prob)
    val_y = np.concatenate(all_y)
    final_metrics = binary_metrics(val_y, val_prob)
    final_metrics["params"] = count_params(model)
    return final_metrics, model


def run_variant(variant: str, cfg: dict, data: dict, device: str, n_seeds: int, out_dir: Path) -> dict:
    width_mult = get_width_mult(variant, ROOT)
    ecg, ppg, label, record_id = data["ecg"], data["ppg"], data["label"], data["record_id"]

    all_fold_metrics = []
    for seed in range(n_seeds):
        set_seed(cfg["seed"] + seed)
        for fold_i, (train_idx, val_idx) in enumerate(
            record_wise_folds(record_id, cfg["split"]["n_folds"], cfg["seed"] + seed)
        ):
            train_ds = AlarmWindowDataset(ecg[train_idx], ppg[train_idx], label[train_idx])
            val_ds = AlarmWindowDataset(ecg[val_idx], ppg[val_idx], label[val_idx])

            t0 = time.time()
            m, model = train_one_fold(cfg, variant, width_mult, train_ds, val_ds, device, seed)
            m["seed"] = seed
            m["fold"] = fold_i
            m["train_seconds"] = time.time() - t0
            all_fold_metrics.append(m)
            print(f"[{variant}] seed={seed} fold={fold_i} f1={m['f1']:.3f} "
                  f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                  f"params={m['params']:,} ({m['train_seconds']:.0f}s)")

            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"{variant}_seed{seed}_fold{fold_i}.pt")

    summary = summarize_across_folds(all_fold_metrics)
    summary["variant"] = variant
    summary["width_mult"] = width_mult
    summary["fold_metrics"] = all_fold_metrics
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=VARIANTS + ["all"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--processed", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_path = args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    data = load_processed(processed_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    out_dir = ROOT / cfg["paths"]["runs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = VARIANTS if args.variant == "all" else [args.variant]
    results = {}
    for variant in variants:
        results[variant] = run_variant(variant, cfg, data, device, args.seeds, out_dir)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults written to {results_path}")

    print("\n=== Summary (mean +/- std across folds/seeds) ===")
    for variant, s in results.items():
        print(
            f"{variant:16s} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f}  "
            f"Sens={s['sensitivity_mean']:.3f}+/-{s['sensitivity_std']:.3f}  "
            f"Spec={s['specificity_mean']:.3f}+/-{s['specificity_std']:.3f}  "
            f"AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f}"
        )


if __name__ == "__main__":
    main()
