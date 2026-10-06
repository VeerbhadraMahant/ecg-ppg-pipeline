"""Foundation-model track (updates.md 5.3): PaPaGei PPG features on the alarm
verification task, evaluated with the repo's protocol (src.train.iter_splits,
same metrics, same preds/json/results.json layout) under runs/foundation/<name>/.

Variants
    pg_linear_s / pg_linear_p     frozen PaPaGei embedding -> logistic regression (PPG alone)
    pg_linear_s_pooled            same, 512-d mean-pooled trunk feature instead of outputs[0]
    pg_frozen_ppg_head            frozen PaPaGei-S tokens -> repo MLP head (PPG alone)
    pg_frozen_concat              scratch ECG CNN + frozen PaPaGei-S tokens -> ConcatFusion
    pg_frozen_cross_attention     scratch ECG CNN + frozen PaPaGei-S tokens -> CrossAttentionFusion
    pg_ft_ppg_only                last residual block + final BN of PaPaGei-S fine-tuned, PPG alone
    pg_ft_cross_attention         scratch ECG CNN + last-block-fine-tuned PaPaGei-S -> cross-attention

Usage
    python -m src.train_foundation --run-name challenge2015_ppg --seeds 5 --variants all
    python -m src.train_foundation --run-name vtac_official --protocol official \
        --processed data/processed/vtac_windows.npz --seeds 3 --variants all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset, load_processed  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.models.classifier import ClassificationHead, count_params  # noqa: E402
from src.models.encoders import CNNEncoder  # noqa: E402
from src.models.foundation import FEATURE_DIM, PaPaGeiEncoder  # noqa: E402
from src.models.fusion import ConcatFusion, CrossAttentionFusion  # noqa: E402
from src.models.losses import build_loss  # noqa: E402
from src.train import iter_splits, set_seed, subsample_labels  # noqa: E402

LINEAR = {"pg_linear_s": ("s", "upstream"), "pg_linear_p": ("p", "upstream"), "pg_linear_s_pooled": ("s", "pooled")}
FROZEN = ["pg_frozen_ppg_head", "pg_frozen_concat", "pg_frozen_cross_attention"]
FINETUNE = ["pg_ft_ppg_only", "pg_ft_cross_attention"]
ALL = list(LINEAR) + FROZEN + FINETUNE


# --------------------------------------------------------------------------- features
@torch.no_grad()
def extract(x: np.ndarray, variant: str, mode: str, kind: str, device: str, bs: int = 128) -> np.ndarray:
    enc = PaPaGeiEncoder(variant, mode="seq").to(device).eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs].astype(np.float32)).to(device)
        out.append((enc(xb) if kind == "seq" else enc.embedding(xb, kind)).cpu().numpy())
    return np.concatenate(out)


def cached(cache_dir: Path, tag: str, x, variant, kind, device):
    p = cache_dir / f"{tag}_{variant}_{kind}.npy"
    if p.exists():
        return np.load(p)
    f = extract(x, variant, "seq", kind, device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(p, f)
    return f


# --------------------------------------------------------------------------- model
class FoundationClassifier(nn.Module):
    """ECG (scratch CNN, optional) + PaPaGei PPG features -> repo head.

    precomputed=True : the `ppg` input is a cached (B, T', 512) feature tensor
                       (frozen encoder, run once).
    precomputed=False: the `ppg` input is the raw (B, 2500) window and a
                       (partially) trainable PaPaGei encoder sits inside.
    """

    def __init__(self, fusion: str, cfg: dict, precomputed: bool, trainable: str = "none", pg_variant: str = "s"):
        super().__init__()
        m = cfg["model"]
        self.fusion_kind, self.precomputed = fusion, precomputed
        self.ppg_encoder = None if precomputed else PaPaGeiEncoder(pg_variant, "seq", trainable)
        if fusion == "ppg_only":
            head_in = FEATURE_DIM
        else:
            self.ecg_encoder = CNNEncoder(m["cnn_channels"], m["cnn_kernel_sizes"], m["cnn_pool"], m["dropout"])
            cls = ConcatFusion if fusion == "concat" else CrossAttentionFusion
            args = (self.ecg_encoder.out_dim, FEATURE_DIM, m["attn_dim"])
            self.fusion = cls(*args) if fusion == "concat" else cls(*args, m["attn_heads"], m["dropout"])
            head_in = self.fusion.out_dim
        self.head = ClassificationHead(head_in, m["head_hidden"], m["dropout"])

    def forward(self, ecg, ppg):
        p = ppg if self.precomputed else self.ppg_encoder(ppg)
        if self.fusion_kind == "ppg_only":
            feat = p.mean(1)
        elif self.fusion_kind == "concat":
            feat = self.fusion(self.ecg_encoder(ecg), p)
        else:
            feat = self.fusion(self.ecg_encoder(ecg), p)[0].mean(1)
        return self.head(feat), None


def build(variant: str, cfg: dict) -> nn.Module:
    fusion = {"pg_frozen_ppg_head": "ppg_only", "pg_frozen_concat": "concat", "pg_frozen_cross_attention": "cross_attention",
              "pg_ft_ppg_only": "ppg_only", "pg_ft_cross_attention": "cross_attention"}[variant]
    return FoundationClassifier(fusion, cfg, precomputed=variant in FROZEN,
                                trainable="last_block" if variant in FINETUNE else "none")


# --------------------------------------------------------------------------- training
def _predict(model, ds, device, bs):
    model.eval()
    out = []
    with torch.no_grad():
        for ecg, ppg, _ in DataLoader(ds, batch_size=bs, shuffle=False):
            out.append(torch.sigmoid(model(ecg.to(device), ppg.to(device))[0]).cpu().numpy())
    return np.concatenate(out)


def fit_fold_nn(cfg, variant, train_ds, val_ds, test_ds, device, ft_lr):
    """Mirror of src.train.train_one_fold (same loss / early stopping / selection
    on inner-val F1) with one change: pretrained PaPaGei parameters get their own,
    smaller learning rate (ft_lr)."""
    t = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"] * 2, shuffle=False)
    model = build(variant, cfg).to(device)
    n_params = count_params(model)
    n_pos = train_ds.label.sum().item()
    loss_fn = build_loss(t["loss"], (len(train_ds) - n_pos) / n_pos if n_pos > 0 else 1.0, t.get("focal_gamma", 2.0), device)
    pre = [q for n, q in model.named_parameters() if q.requires_grad and n.startswith("ppg_encoder")]
    rest = [q for n, q in model.named_parameters() if q.requires_grad and not n.startswith("ppg_encoder")]
    groups = [{"params": rest, "lr": t["lr"]}] + ([{"params": pre, "lr": ft_lr}] if pre else [])
    opt = torch.optim.Adam(groups, weight_decay=t["weight_decay"])

    best_f1, best_state, patience = -1.0, None, t["early_stop_patience"]
    for _ in range(t["epochs"]):
        model.train()
        for ecg, ppg, y in train_loader:
            ecg, ppg, y = ecg.to(device), ppg.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(ecg, ppg)[0], y).backward()
            opt.step()
        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for ecg, ppg, y in val_loader:
                vp.append(torch.sigmoid(model(ecg.to(device), ppg.to(device))[0]).cpu().numpy())
                vy.append(y.numpy())
        f1 = binary_metrics(np.concatenate(vy), np.concatenate(vp))["f1"]
        if f1 > best_f1:
            best_f1, patience = f1, t["early_stop_patience"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    bs = t["batch_size"] * 2
    return n_params, _predict(model, val_ds, device, bs), _predict(model, test_ds, device, bs)


def fit_fold_linear(x_fit, y_fit, x_val, y_val, x_test, seed):
    sc = StandardScaler().fit(x_fit)
    xf, xv, xt = sc.transform(x_fit), sc.transform(x_val), sc.transform(x_test)
    best = None
    for C in (1e-3, 1e-2, 1e-1, 1.0):  # picked on the inner validation set by F1
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed).fit(xf, y_fit)
        f1 = binary_metrics(y_val, lr.predict_proba(xv)[:, 1])["f1"]
        if best is None or f1 > best[0]:
            best = (f1, C, lr)
    lr = best[2]
    return x_fit.shape[1] + 1, lr.predict_proba(xv)[:, 1], lr.predict_proba(xt)[:, 1], best[1]


def run(variant, cfg, data, feats, device, n_seeds, out_dir, protocol, ft_lr):
    ecg, ppg, label = data["ecg"], data["ppg"], data["label"]
    metrics = []
    for seed in range(n_seeds):
        set_seed(cfg["seed"] + seed)
        for fold_i, fit_idx, val_idx, test_idx in iter_splits(data, cfg, seed, protocol):
            fit_idx = subsample_labels(fit_idx, data, cfg["train"].get("label_fraction", 1.0), cfg["seed"] + seed + fold_i)
            pred_path = out_dir / "preds" / f"{variant}_seed{seed}_fold{fold_i}.npz"
            m_path = pred_path.with_suffix(".json")
            if pred_path.exists() and m_path.exists():
                metrics.append(json.loads(m_path.read_text()))
                continue
            t0 = time.time()
            extra = {}
            if variant in LINEAR:
                f = feats[variant]
                n_params, vp, tp, C = fit_fold_linear(f[fit_idx], label[fit_idx], f[val_idx], label[val_idx], f[test_idx], seed)
                extra["C"] = C
            else:
                src = feats["seq_s"] if variant in FROZEN else ppg
                mk = lambda idx: AlarmWindowDataset(ecg[idx], src[idx], label[idx])  # noqa: E731
                n_params, vp, tp = fit_fold_nn(cfg, variant, mk(fit_idx), mk(val_idx), mk(test_idx), device, ft_lr)
            val_y, test_y = label[val_idx], label[test_idx]
            m = binary_metrics(test_y, tp)
            m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
            m["ops"] = operating_points(val_y, vp, test_y, tp)
            m.update(params=n_params, seed=seed, fold=fold_i, train_seconds=time.time() - t0,
                     n_train=len(fit_idx), n_test=len(test_idx), **extra)
            metrics.append(m)
            print(f"[{variant}] seed={seed} fold={fold_i} f1={m['f1']:.3f} auc={m['auc']:.3f} "
                  f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} ({m['train_seconds']:.0f}s)", flush=True)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pred_path, test_idx=test_idx, val_idx=val_idx, val_prob=vp, val_y=val_y, test_prob=tp, test_y=test_y)
            m_path.write_text(json.dumps(m, default=float))
    s = summarize_across_folds(metrics)
    s.update(variant=variant, width_mult=1.0, fold_metrics=metrics)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed" / "challenge2015_windows.npz"))
    ap.add_argument("--protocol", default="cv", choices=["cv", "official"])
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="lr of the fine-tuned PaPaGei parameters")
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = load_processed(args.processed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = ALL if args.variants == "all" else args.variants.split(",")
    out_dir = ROOT / cfg["paths"]["runs_dir"] / "foundation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "_features"
    tag = Path(args.processed).stem

    feats = {}
    for v in variants:  # frozen features are computed once per dataset (they do not depend on the split)
        if v in LINEAR:
            pv, kind = LINEAR[v]
            feats[v] = cached(cache, tag, data["ppg"], pv, kind, device)
        elif v in FROZEN and "seq_s" not in feats:
            feats["seq_s"] = cached(cache, tag, data["ppg"], "s", "seq", device)
    print(f"device={device} n={len(data['label'])} variants={variants}", flush=True)

    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for v in variants:
        results[v] = run(v, cfg, data, feats, device, args.seeds, out_dir, args.protocol, args.ft_lr)
        res_path.write_text(json.dumps(results, indent=2, default=str))
        s = results[v]
        print(f"== {v:28s} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} Sens={s['sensitivity_mean']:.3f} "
              f"Spec={s['specificity_mean']:.3f} AUC={s['auc_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
