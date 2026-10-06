"""Explanation faithfulness (updates.md analysis 6.3).

Compares four rankings of which 0.5 s blocks of a modality matter:
  attention  : ECG->PPG cross-attention mass received per block (PPG only)
  occlusion  : drop in the model's confidence when the block is zeroed
  ig         : integrated gradients (zero baseline) summed |attr| per block
  random     : mean over random orderings (null reference)

Deletion curve: remove blocks most-important-first, track the model's
confidence in its ORIGINAL decision; faithful ranking => steep fall (low area).
Insertion curve: start from a blank modality and add blocks most-important
first; faithful ranking => fast rise (high area).

Usage:
    python -m src.analysis.faithfulness --run-name challenge2015_ppg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult, iter_splits, run_dir  # noqa: E402

FS = 250
BLOCK = FS // 2  # 0.5 s


def conf(model, ecg, ppg, pred_true: torch.Tensor):
    lo, w = model(ecg, ppg)
    p = torch.sigmoid(lo)
    return torch.where(pred_true, p, 1 - p), w


def block_scores_ig(model, ecg, ppg, target: str, steps: int = 24):
    x = (ppg if target == "ppg" else ecg).detach()
    grads = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps, device=x.device):
        xi = (a * x).clone().requires_grad_(True)
        lo, _ = model(*((ecg, xi) if target == "ppg" else (xi, ppg)))
        g, = torch.autograd.grad(lo.sum(), xi)
        grads += g
    attr = (x * grads / steps).abs()
    return attr.reshape(x.size(0), -1, BLOCK).sum(-1)  # (B, n_blocks)


def block_scores_occlusion(model, ecg, ppg, target, pred_true):
    base, _ = conf(model, ecg, ppg, pred_true)
    nb = ecg.size(1) // BLOCK
    out = []
    for b in range(nb):
        e2, p2 = ecg.clone(), ppg.clone()
        (p2 if target == "ppg" else e2)[:, b * BLOCK:(b + 1) * BLOCK] = 0
        c, _ = conf(model, e2, p2, pred_true)
        out.append(base - c)  # confidence lost when the block is removed
    return torch.stack(out, dim=1)


def block_scores_attention(w, T):
    """w: (B, Te, Tp) attention. PPG block importance = attention mass received,
    with each token assigned to the block containing its centre sample (the
    window length need not be a multiple of the token count)."""
    recv = w.mean(dim=1)  # (B, Tp)
    tp = recv.size(1)
    stride = T // tp
    nb = T // BLOCK
    idx = ((torch.arange(tp, device=w.device) * stride + stride // 2) // BLOCK).clamp(max=nb - 1)
    out = torch.zeros(recv.size(0), nb, device=w.device)
    return out.index_add_(1, idx, recv)


@torch.no_grad()
def curve(model, ecg, ppg, target, order, pred_true, mode):
    """order: (B, nb) block ids, most important first. Returns (B, nb+1) confidences.
    Vectorised: rank[b, i] = step at which block i is removed/added."""
    B, nb = order.shape
    sig = ppg if target == "ppg" else ecg
    rank = torch.argsort(order, dim=1)  # rank[b, i] = position of block i in the ordering
    res = []
    for k in range(nb + 1):
        hit = (rank < k).repeat_interleave(BLOCK, dim=1)  # blocks already deleted / inserted
        cur = torch.where(hit, torch.zeros_like(sig), sig) if mode == "deletion" else torch.where(hit, sig, torch.zeros_like(sig))
        c, _ = conf(model, *((ecg, cur) if target == "ppg" else (cur, ppg)), pred_true)
        res.append(c)
    return torch.stack(res, dim=1)


def auc01(c: torch.Tensor):  # mean over the curve == normalised area
    return c.mean(dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="challenge2015_ppg")
    ap.add_argument("--processed", default=None)
    ap.add_argument("--variant", default="cross_attention")
    ap.add_argument("--protocol", default="cv")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--max-windows", type=int, default=60, help="test windows per fold (subsampled, seeded)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    processed = args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    data = load_processed(processed)
    run_path = run_dir(cfg, args.run_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    ecg_all, ppg_all = data["ecg"], data["ppg"]

    rows = []
    for fold, fit, val, test in iter_splits(data, cfg, 0, args.protocol):
        model = build_model(args.variant, cfg, get_width_mult(args.variant, ROOT)).to(device)
        model.load_state_dict(torch.load(run_path / "checkpoints" / f"{args.variant}_seed0_fold{fold}.pt", map_location=device))
        model.eval()
        if len(test) > args.max_windows:
            test = np.sort(rng.choice(test, args.max_windows, replace=False))
        t_fold = time.time()
        ecg = torch.from_numpy(ecg_all[test]).to(device)
        ppg = torch.from_numpy(ppg_all[test]).to(device)
        with torch.no_grad():
            lo, w = model(ecg, ppg)
        pred_true = torch.sigmoid(lo) >= 0.5
        T = ecg.size(1)
        for target in ("ppg", "ecg"):
            methods = {"occlusion": block_scores_occlusion(model, ecg, ppg, target, pred_true),
                       "ig": block_scores_ig(model, ecg, ppg, target)}
            if target == "ppg" and w is not None:
                methods["attention"] = block_scores_attention(w, T)
            nb = T // BLOCK
            for name, sc in methods.items():
                order = torch.argsort(sc, dim=1, descending=True)
                d = auc01(curve(model, ecg, ppg, target, order, pred_true, "deletion")).cpu().numpy()
                ins = auc01(curve(model, ecg, ppg, target, order, pred_true, "insertion")).cpu().numpy()
                for i in range(len(d)):
                    rows.append({"fold": fold, "target": target, "method": name, "deletion": d[i], "insertion": ins[i]})
            rd, ri = [], []
            for _ in range(args.n_random):
                order = torch.from_numpy(np.stack([rng.permutation(nb) for _ in range(len(ecg))])).to(device)
                rd.append(auc01(curve(model, ecg, ppg, target, order, pred_true, "deletion")).cpu().numpy())
                ri.append(auc01(curve(model, ecg, ppg, target, order, pred_true, "insertion")).cpu().numpy())
            rd, ri = np.mean(rd, 0), np.mean(ri, 0)
            for i in range(len(rd)):
                rows.append({"fold": fold, "target": target, "method": "random", "deletion": rd[i], "insertion": ri[i]})
        print(f"fold {fold} done in {time.time() - t_fold:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = run_path / "faithfulness"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_window.csv", index=False)
    summ = df.groupby(["target", "method"])[["deletion", "insertion"]].agg(["mean", "sem"]).round(4)
    summ.to_csv(out / "summary.csv")
    print("\nDeletion AUC (lower = more faithful) / Insertion AUC (higher = more faithful):")
    print(summ.to_string())

    # paired difference vs random, per window
    print("\nFaithfulness vs random ordering (paired mean difference, window level):")
    for tgt in ("ppg", "ecg"):
        sub = df[df.target == tgt]
        rnd = sub[sub.method == "random"].reset_index(drop=True)
        for m in sub.method.unique():
            if m == "random":
                continue
            cur = sub[sub.method == m].reset_index(drop=True)
            dd = cur.deletion - rnd.deletion
            di = cur.insertion - rnd.insertion
            print(f"  {tgt}/{m}: deletion {dd.mean():+.4f} (se {dd.sem():.4f}), insertion {di.mean():+.4f} (se {di.sem():.4f})")


if __name__ == "__main__":
    main()
