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

from src.data.dataset import (  #
    AlarmWindowDataset,
    load_processed,
    record_wise_folds,
    record_wise_holdout,
)  # noqa: E402
from src.metrics import (  # noqa: E402
    binary_metrics,
    challenge_score,
    operating_points,
    summarize_across_folds,
)
from src.models.classifier import ALL_VARIANTS, VARIANTS, build_model, count_params  # noqa: E402
from src.models.losses import build_loss  # noqa: E402


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


def predict(model: torch.nn.Module, ds: AlarmWindowDataset, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for ecg, ppg, _ in loader:
            logits, _ = model(ecg.to(device), ppg.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def subsample_labels(fit_idx, data, frac: float, seed: int):
    """Keep `frac` of the training RECORDS, stratified by label, so the label
    budget shrinks without splitting any patient across kept/dropped."""
    if frac >= 1.0:
        return fit_idx
    rng = np.random.RandomState(seed)
    rec, y = data["record_id"][fit_idx], data["label"][fit_idx]
    keep = []
    for cls in (0.0, 1.0):
        recs = np.unique(rec[y == cls])
        rng.shuffle(recs)
        keep += list(recs[: max(2, int(round(frac * len(recs))))])
    return fit_idx[np.isin(rec, keep)]


def run_dir(cfg: dict, run_name: str) -> Path:
    return ROOT / cfg["paths"]["runs_dir"] / run_name


def train_one_fold(
    cfg: dict,
    variant: str,
    width_mult: float,
    train_ds: AlarmWindowDataset,
    val_ds: AlarmWindowDataset,
    test_ds: AlarmWindowDataset,
    device: str,
    seed: int,
) -> tuple[dict, torch.nn.Module, dict]:
    """val_ds drives early stopping / best-epoch selection; test_ds (the outer
    CV fold) is evaluated once at the end with the selected weights."""
    t = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"] * 2, shuffle=False)

    model = build_model(variant, cfg, width_mult).to(device)
    n_params = count_params(model)  # counted before any freezing
    if t.get("init_from"):
        pre = torch.load(t["init_from"], map_location=device)
        own = model.state_dict()
        # single-signal variants only own one of the two pretrained encoders
        usable = {k: v for k, v in pre.items() if k in own and own[k].shape == v.shape}
        if not usable:
            raise ValueError("no pretrained weights matched this model")
        model.load_state_dict(usable, strict=False)
    frozen = bool(t.get("freeze_encoders"))
    if frozen:
        for n, prm in model.named_parameters():
            if n.startswith(("ecg_encoder", "ppg_encoder")):
                prm.requires_grad = False

    def set_train():
        model.train()
        if frozen:  # keep frozen encoders' BatchNorm statistics fixed too
            for n, mod in model.named_children():
                if n in ("ecg_encoder", "ppg_encoder"):
                    mod.eval()

    n_pos = train_ds.label.sum().item()
    n_neg = len(train_ds) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    loss_fn = build_loss(t["loss"], pos_weight, t.get("focal_gamma", 2.0), device)

    optimizer = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=t["lr"], weight_decay=t["weight_decay"])

    best_f1 = -1.0
    best_state = None
    patience_left = t["early_stop_patience"]

    for epoch in range(t["epochs"]):
        set_train()
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
    bs = t["batch_size"] * 2
    val_prob = predict(model, val_ds, device, bs)
    test_prob = predict(model, test_ds, device, bs)
    val_y, test_y = val_ds.label.numpy(), test_ds.label.numpy()

    final_metrics = binary_metrics(test_y, test_prob)
    final_metrics["challenge_score"] = challenge_score(
        final_metrics["tp"], final_metrics["tn"], final_metrics["fp"], final_metrics["fn"]
    )
    final_metrics["ops"] = operating_points(val_y, val_prob, test_y, test_prob)
    final_metrics["params"] = n_params
    preds = {"val_prob": val_prob, "val_y": val_y, "test_prob": test_prob, "test_y": test_y}
    return final_metrics, model, preds


def iter_splits(data: dict, cfg: dict, seed: int, protocol: str):
    """Yields (fold_i, fit_idx, val_idx, test_idx).

    cv       : record-wise outer folds, inner record-wise validation holdout.
    official : the dataset's own published train/val/test split (VTaC),
               which is patient-disjoint by construction.
    """
    record_id = data["record_id"]
    if protocol == "official":
        sp = data["split"]
        yield 0, np.where(sp == "train")[0], np.where(sp == "val")[0], np.where(sp == "test")[0]
        return
    for fold_i, (train_idx, test_idx) in enumerate(
        record_wise_folds(record_id, cfg["split"]["n_folds"], cfg["seed"] + seed)
    ):
        fit_idx, val_idx = record_wise_holdout(
            record_id, train_idx, cfg["split"].get("val_fraction", 0.15), cfg["seed"] + seed + fold_i
        )
        yield fold_i, fit_idx, val_idx, test_idx


def run_variant(variant: str, cfg: dict, data: dict, device: str, n_seeds: int, out_dir: Path,
                resume: bool = True, protocol: str = "cv") -> dict:
    width_mult = cfg["train"].get("width_override") or get_width_mult(variant, ROOT)
    ecg, ppg, label = data["ecg"], data["ppg"], data["label"]
    if variant == "multimodal":  # (N,4,T) signals + (N,4) presence mask ride in the ecg/ppg slots
        from src.models.multimodal import to_multimodal_inputs

        ecg, ppg = to_multimodal_inputs(data)

    all_fold_metrics = []
    for seed in range(n_seeds):
        set_seed(cfg["seed"] + seed)
        for fold_i, fit_idx, val_idx, test_idx in iter_splits(data, cfg, seed, protocol):
            fit_idx = subsample_labels(fit_idx, data, cfg["train"].get("label_fraction", 1.0), cfg["seed"] + seed + fold_i)
            y_fit = label[fit_idx]
            if cfg["train"].get("soft_labels") and "annotator_frac_true" in data:
                # train on the fraction of annotators voting TRUE (validation/test stay hard
                # consensus labels); events without votes keep their hard label
                soft = data["annotator_frac_true"][fit_idx]
                y_fit = np.where(np.isnan(soft), y_fit, soft).astype(np.float32)
            train_ds = AlarmWindowDataset(ecg[fit_idx], ppg[fit_idx], y_fit)
            val_ds = AlarmWindowDataset(ecg[val_idx], ppg[val_idx], label[val_idx])
            test_ds = AlarmWindowDataset(ecg[test_idx], ppg[test_idx], label[test_idx])

            pred_path = out_dir / "preds" / f"{variant}_seed{seed}_fold{fold_i}.npz"
            m_path = pred_path.with_suffix(".json")
            if resume and pred_path.exists() and m_path.exists():
                all_fold_metrics.append(json.loads(m_path.read_text()))
                continue

            t0 = time.time()
            m, model, preds = train_one_fold(cfg, variant, width_mult, train_ds, val_ds, test_ds, device, seed)
            m["seed"] = seed
            m["fold"] = fold_i
            m["train_seconds"] = time.time() - t0
            all_fold_metrics.append(m)
            print(f"[{variant}] seed={seed} fold={fold_i} f1={m['f1']:.3f} "
                  f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                  f"params={m['params']:,} ({m['train_seconds']:.0f}s)")

            m["n_train"] = len(train_ds)
            m["n_test"] = len(test_ds)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pred_path, test_idx=test_idx, val_idx=val_idx, **preds)
            m_path.write_text(json.dumps(m, default=float))

            if seed < cfg["train"].get("save_ckpt_seeds", 1):
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
    parser.add_argument("--variant", default="ladder",
                        help="a variant name, 'ladder' (4 original), 'all', or comma-separated list")
    parser.add_argument("--seeds", type=int, default=None, help="default: train.n_seeds from config")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--processed", default=None)
    parser.add_argument("--run-name", default="challenge2015_ppg",
                        help="results go to runs/<run-name>/ (keeps cohorts/protocols separate)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-from", default=None, help="encoder weights from src/pretrain.py")
    parser.add_argument("--freeze-encoders", action="store_true", help="linear-probe style: train fusion+head only")
    parser.add_argument("--label-fraction", type=float, default=1.0, help="fraction of training records kept")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--soft-labels", action="store_true",
                        help="train on annotator vote fractions (VTaC v1.1) instead of consensus labels")
    parser.add_argument("--width-override", type=float, default=None,
                        help="force this width multiplier (needed to load width-1.0 pretrained encoders)")
    parser.add_argument("--protocol", default="cv", choices=["cv", "official"],
                        help="cv: record-wise CV; official: dataset's own train/val/test split (VTaC)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_seeds = args.seeds or cfg["train"].get("n_seeds", 3)
    if args.init_from:
        cfg["train"]["init_from"] = str(Path(args.init_from).resolve())
    cfg["train"]["freeze_encoders"] = args.freeze_encoders
    cfg["train"]["label_fraction"] = args.label_fraction
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    cfg["train"]["soft_labels"] = args.soft_labels
    if args.width_override:
        cfg["train"]["width_override"] = args.width_override
    processed_path = args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    data = load_processed(processed_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  seeds: {n_seeds}  data: {processed_path}  n={len(data['label'])}")

    out_dir = run_dir(cfg, args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "all":
        variants = ALL_VARIANTS + ["multimodal"]
    elif args.variant == "ladder":
        variants = VARIANTS
    else:
        variants = args.variant.split(",")
        bad = [v for v in variants if v not in ALL_VARIANTS + ["multimodal"]]
        if bad:
            raise SystemExit(f"unknown variants {bad}; choose from {ALL_VARIANTS}")

    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for variant in variants:
        results[variant] = run_variant(variant, cfg, data, device, n_seeds, out_dir, not args.no_resume, args.protocol)
        with open(results_path, "w") as f:  # written after every variant so a crash loses little
            json.dump(results, f, indent=2, default=str)
        s = results[variant]
        print(
            f"== {variant:20s} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f}  "
            f"Sens={s['sensitivity_mean']:.3f}  Spec={s['specificity_mean']:.3f}  AUC={s['auc_mean']:.3f}"
        )
    print(f"\nresults written to {results_path}")


if __name__ == "__main__":
    main()
