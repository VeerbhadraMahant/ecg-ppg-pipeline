"""Self-supervised pretraining of the ECG and PPG encoders (updates.md 5.2).

Corpus (label-free, in-domain): the 60 s PRE-alarm segment of every VTaC event
whose patient is in the official train/val split (test patients are excluded,
so downstream test data never touch pretraining). Events need ECG + PLETH.
Random 10 s crops are drawn on the fly; each crop is z-normalised.

Objectives (--objective):
  contrastive : token-level InfoNCE. A token of the ECG encoder at time t must
                pick the PPG token at time t (+ --lag-tokens) among all PPG
                tokens of the batch, which includes other times of the same
                recording. Solving it requires learning *timing*, not just
                "this pulse looks healthy".
  masked      : zero 30 % of 0.5 s blocks of each modality, reconstruct the
                original waveform from that modality's own encoder.
  crossmodal  : predict the PPG waveform from ECG tokens and vice versa.

The encoders are exactly the CNNEncoder used by AlarmClassifier, so the saved
weights (`ecg_encoder.*`, `ppg_encoder.*`) load straight into it.

Usage:
    python -m src.pretrain --objective contrastive --epochs 30 --out runs/pretrain/contrastive.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.signal_ops import bandpass_filter  # noqa: E402
from src.data.vtac_loader import FS, ONSET, read_split, split_channels  # noqa: E402
from src.models.encoders import CNNEncoder  # noqa: E402

BLOCK = FS // 2


def build_corpus(vtac_dir: Path, cfg: dict, max_events: int | None = None):
    split = read_split(vtac_dir)
    s = cfg["signal"]
    ecgs, ppgs = [], []
    for f in sorted((vtac_dir / "events").glob("*.npz")):
        if split.get(f.stem) not in ("train", "val"):
            continue
        z = np.load(f)
        names = [str(n) for n in z["names"]]
        ch = split_channels(z["sig"].astype(np.float64), names)
        if ch["ecg"] is None or ch["ppg"] is None:
            continue
        e = ch["ecg"][:ONSET]  # strictly pre-alarm
        p = ch["ppg"][:ONSET]
        if e.std() < 1e-6 or p.std() < 1e-6:
            continue
        ecgs.append(bandpass_filter(e, FS, *s["ecg_band"], order=s["filter_order"]).astype(np.float32))
        ppgs.append(bandpass_filter(p, FS, *s["ppg_band"], order=s["filter_order"]).astype(np.float32))
        if max_events and len(ecgs) >= max_events:
            break
    return np.stack(ecgs), np.stack(ppgs)


def crops(ecg, ppg, idx, win, rng, device):
    n = len(idx)
    starts = rng.randint(0, ecg.shape[1] - win, size=n)
    e = np.stack([ecg[i, s:s + win] for i, s in zip(idx, starts)])
    p = np.stack([ppg[i, s:s + win] for i, s in zip(idx, starts)])
    z = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)  # noqa: E731
    return torch.from_numpy(z(e)).float().to(device), torch.from_numpy(z(p)).float().to(device)


class Pretrainer(nn.Module):
    def __init__(self, cfg: dict, objective: str, proj_dim: int = 64, tau: float = 0.1, lag_tokens: int = 0):
        super().__init__()
        m = cfg["model"]
        mk = lambda: CNNEncoder(m["cnn_channels"], m["cnn_kernel_sizes"], m["cnn_pool"], m["dropout"], 1.0)  # noqa: E731
        self.ecg_encoder, self.ppg_encoder = mk(), mk()
        d = self.ecg_encoder.out_dim
        self.stride = m["cnn_pool"] ** len(m["cnn_channels"])
        self.objective, self.tau, self.lag = objective, tau, lag_tokens
        self.proj_e, self.proj_p = nn.Linear(d, proj_dim), nn.Linear(d, proj_dim)
        self.dec_e, self.dec_p = nn.Linear(d, self.stride), nn.Linear(d, self.stride)

    def _decode(self, tokens, dec, length):
        x = dec(tokens).reshape(tokens.size(0), -1)  # (B, T'*stride)
        return F.pad(x, (0, max(0, length - x.size(1))))[:, :length]

    def forward(self, e, p):
        T = e.size(1)
        if self.objective == "contrastive":
            te, tp = self.ecg_encoder(e), self.ppg_encoder(p)  # (B,T',d)
            ze = F.normalize(self.proj_e(te), dim=-1)
            zp = F.normalize(self.proj_p(tp), dim=-1)
            B, Tn, D = ze.shape
            S = 48
            pos = torch.randint(0, Tn - max(self.lag, 0) - 1, (B, S), device=e.device)
            q = ze[torch.arange(B, device=e.device)[:, None], pos].reshape(B * S, D)
            keys = zp.reshape(B * Tn, D)
            target = (torch.arange(B, device=e.device)[:, None] * Tn + pos + self.lag).reshape(-1)
            logits_e = q @ keys.t() / self.tau
            loss_e = F.cross_entropy(logits_e, target)
            # symmetric direction
            q2 = zp[torch.arange(B, device=e.device)[:, None], pos + self.lag].reshape(B * S, D)
            keys2 = ze.reshape(B * Tn, D)
            target2 = (torch.arange(B, device=e.device)[:, None] * Tn + pos).reshape(-1)
            loss_p = F.cross_entropy(q2 @ keys2.t() / self.tau, target2)
            return 0.5 * (loss_e + loss_p)
        if self.objective == "masked":
            def mask(x):
                nb = x.size(1) // BLOCK
                m = (torch.rand(x.size(0), nb, device=x.device) < 0.3).repeat_interleave(BLOCK, dim=1)
                m = F.pad(m, (0, x.size(1) - m.size(1)))
                return torch.where(m, torch.zeros_like(x), x), m
            xe, me = mask(e)
            xp, mp = mask(p)
            re = self._decode(self.ecg_encoder(xe), self.dec_e, T)
            rp = self._decode(self.ppg_encoder(xp), self.dec_p, T)
            return 0.5 * (F.mse_loss(re[me], e[me]) + F.mse_loss(rp[mp], p[mp]))
        if self.objective == "crossmodal":
            pe = self._decode(self.ecg_encoder(e), self.dec_p, T)  # ECG -> PPG
            pp = self._decode(self.ppg_encoder(p), self.dec_e, T)  # PPG -> ECG
            return 0.5 * (F.mse_loss(pe, p) + F.mse_loss(pp, e))
        raise ValueError(self.objective)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--objective", default="contrastive", choices=["contrastive", "masked", "crossmodal"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lag-tokens", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    win = int(cfg["signal"]["window_seconds"] * FS)

    t0 = time.time()
    ecg, ppg = build_corpus(ROOT / "data" / "raw" / "vtac", cfg, args.max_events)
    print(f"corpus: {len(ecg)} pre-alarm segments of {ecg.shape[1] / FS:.0f}s ({time.time() - t0:.0f}s)")

    model = Pretrainer(cfg, args.objective, lag_tokens=args.lag_tokens).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * args.steps_per_epoch)
    out = Path(args.out or ROOT / "runs" / "pretrain" / f"{args.objective}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            idx = rng.randint(0, len(ecg), size=args.batch)
            e, p = crops(ecg, ppg, idx, win, rng, device)
            loss = model(e, p)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"epoch {ep + 1}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)
    state = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith(("ecg_encoder.", "ppg_encoder."))}
    torch.save(state, out)
    print(f"saved encoders to {out}")


if __name__ == "__main__":
    main()
