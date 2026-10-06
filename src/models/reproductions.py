"""Reproductions of published ICU false-alarm-reduction methods (updates.md 5.7).

These are RE-IMPLEMENTATIONS from the papers' published descriptions, matched
to this repo's input format (10 s, 250 Hz, z-scored ECG + PPG) and parameter
budget. They are not the authors' code and not their exact hyper-parameters.

(a) MousaviAttnCNNRNN  ('mousavi_attn_cnn_rnn')
    Mousavi, Afghah, Acharya et al., "Single-modal and multi-modal false
    arrhythmia alarm reduction using attention-based deep learning",
    arXiv:1909.11791 (2019).

    FAITHFUL (to the best of my recollection of the paper):
      - one branch per signal, each: 1D-CNN front end -> recurrent layer
        (bidirectional) -> temporal attention pooling;
      - attention is additive (Bahdanau-style): e_t = v^T tanh(W h_t + b),
        a = softmax_t(e), context = sum_t a_t h_t;
      - multi-modal fusion = concatenation of the per-signal attended context
        vectors, followed by a small fully connected classifier -> 1 logit.
    ASSUMED / NOT VERIFIED (I do not remember the paper's exact values):
      - number of conv layers, kernel sizes, channels, pooling factors
        (here 3 x [Conv-BN-ReLU-MaxPool(4)], 2500 -> ~39 steps);
      - GRU vs LSTM (paper used recurrent layers; GRU is used here);
      - hidden sizes, attention dimension, dropout, classifier width;
      - the paper's own pre-processing, augmentation, loss and class
        weighting (this repo's focal loss / trainer are used instead);
      - the paper also evaluated ABP and more signal combinations; only the
        ECG+PPG pair is available here;
      - BatchNorm in the CNN front end is a choice made here.

(b) AlarmTypeContrastive  ('alarm_type_contrastive')
    In the spirit of the Scientific Reports (2022) contrastive approach to
    false-alarm reduction: ONE model trained across all alarm types, with the
    alarm type fed as a learned embedding, and a supervised-contrastive
    auxiliary loss on the fused representation. FAITHFUL: only the general
    idea (shared model over alarm types, alarm-type embedding, contrastive
    auxiliary objective). ASSUMED: everything else (encoder = the (a) branch,
    embedding size, SupCon (Khosla et al. 2020) with label-positives, temperature,
    loss weight, projection head). I have not verified the original's
    architecture or its exact contrastive formulation.
    NOT registered in REPRODUCTIONS because its forward needs the alarm type,
    which the generic trainer does not pass; train it with
    src/train_reproductions.py.

Both expose forward(ecg (B,T), ppg (B,T)) -> (logits (B,), None); (b) also
accepts an optional third argument `alarm_type` (LongTensor (B,)).

Run `python -m src.models.reproductions` to compute/print the width_mult that
matches the repo's parameter budget and write
configs/width_mult_reproductions.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent

N_ALARM_TYPES = 5  # CinC2015: Asystole, Bradycardia, Tachycardia, VFlutter/Fib, VTach


def _w(base: int, mult: float, floor: int = 4) -> int:
    return max(floor, int(round(base * mult)))


class AdditiveAttention(nn.Module):
    """e_t = v^T tanh(W h_t + b); a = softmax(e); context = sum a_t h_t."""

    def __init__(self, in_dim: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor):  # h: (B, T', D)
        e = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T')
        a = torch.softmax(e, dim=-1)
        return (a.unsqueeze(-1) * h).sum(dim=1), a


class _Branch(nn.Module):
    """1D-CNN front end -> BiGRU -> additive attention pooling, for one signal."""

    def __init__(self, width_mult: float, dropout: float,
                 channels=(16, 32, 64), kernels=(15, 7, 5), pool=4, hidden=64, attn_dim=64):
        super().__init__()
        ch = [_w(c, width_mult) for c in channels]
        layers, cin = [], 1
        for c, k in zip(ch, kernels):
            layers += [nn.Conv1d(cin, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(pool)]
            cin = c
        self.cnn = nn.Sequential(*layers)
        h = _w(hidden, width_mult)
        self.rnn = nn.GRU(cin, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.attn = AdditiveAttention(2 * h, _w(attn_dim, width_mult))
        self.out_dim = 2 * h

    def forward(self, x: torch.Tensor):  # (B, T)
        z = self.cnn(x.unsqueeze(1)).transpose(1, 2)  # (B, T', C)
        h, _ = self.rnn(z)
        return self.attn(self.drop(h))  # (B, 2h), (B, T')


class MousaviAttnCNNRNN(nn.Module):
    def __init__(self, width_mult: float = 1.0, dropout: float = 0.3, head_hidden: int = 64):
        super().__init__()
        self.ecg_branch = _Branch(width_mult, dropout)
        self.ppg_branch = _Branch(width_mult, dropout)
        d = self.ecg_branch.out_dim + self.ppg_branch.out_dim
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        self.feat_dim = d

    def features(self, ecg: torch.Tensor, ppg: torch.Tensor) -> torch.Tensor:
        ce, _ = self.ecg_branch(ecg)
        cp, _ = self.ppg_branch(ppg)
        return torch.cat([ce, cp], dim=-1)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        return self.head(self.features(ecg, ppg)).squeeze(-1), None


class AlarmTypeContrastive(nn.Module):
    """Shared model across alarm types: learned alarm-type embedding concatenated
    to the fused features + projection head for a supervised-contrastive loss."""

    def __init__(self, width_mult: float = 1.0, dropout: float = 0.3, head_hidden: int = 64,
                 n_types: int = N_ALARM_TYPES, type_dim: int = 8, proj_dim: int = 64):
        super().__init__()
        self.n_types = n_types
        self.ecg_branch = _Branch(width_mult, dropout)
        self.ppg_branch = _Branch(width_mult, dropout)
        d = self.ecg_branch.out_dim + self.ppg_branch.out_dim
        self.type_emb = nn.Embedding(n_types + 1, type_dim)  # last index = unknown type
        self.head = nn.Sequential(nn.Linear(d + type_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, 1))
        self.proj = nn.Sequential(nn.Linear(d, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward_with_embedding(self, ecg, ppg, alarm_type=None):
        feat = torch.cat([self.ecg_branch(ecg)[0], self.ppg_branch(ppg)[0]], dim=-1)
        if alarm_type is None:
            alarm_type = torch.full((feat.size(0),), self.n_types, dtype=torch.long, device=feat.device)
        logits = self.head(torch.cat([feat, self.type_emb(alarm_type)], dim=-1)).squeeze(-1)
        z = F.normalize(self.proj(feat), dim=-1)
        return logits, z

    def forward(self, ecg, ppg, alarm_type=None):
        return self.forward_with_embedding(ecg, ppg, alarm_type)[0], None


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), positives = same label
    in the batch (pooled over all alarm types). z must be L2-normalised.
    Anchors without any positive in the batch are skipped; returns 0 if none."""
    n = z.size(0)
    y = y.long().view(-1)
    sim = z @ z.t() / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return z.sum() * 0.0
    mean_lp = (log_prob * pos).sum(1)[valid] / n_pos[valid]
    return -mean_lp.mean()


REPRODUCTIONS = {"mousavi_attn_cnn_rnn": MousaviAttnCNNRNN}


def build_reproduction(variant: str, cfg: dict, width_mult: float = 1.0) -> nn.Module:
    if variant not in REPRODUCTIONS:
        raise ValueError(f"unknown reproduction {variant!r}; choose from {list(REPRODUCTIONS)}")
    m = cfg["model"]
    return REPRODUCTIONS[variant](width_mult=width_mult, dropout=m["dropout"], head_hidden=m["head_hidden"])


def build_alarm_type_contrastive(cfg: dict, width_mult: float = 1.0) -> nn.Module:
    m = cfg["model"]
    return AlarmTypeContrastive(width_mult=width_mult, dropout=m["dropout"], head_hidden=m["head_hidden"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_width(builder, target: int, tol: float = 0.01) -> float:
    """Bisect width_mult so the parameter count is within tol of target. Widths are
    rounded to integers inside the model, so the count is a step function; the
    closest width found over the search is returned."""
    lo, hi, best = 0.25, 8.0, (1e18, 1.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        n = count_params(builder(mid))
        if abs(n - target) < best[0]:
            best = (abs(n - target), mid)
        if abs(n - target) / target < tol:
            break
        lo, hi = (mid, hi) if n < target else (lo, mid)
    return round(best[1], 4)


def main() -> None:
    import yaml

    sys.path.insert(0, str(ROOT))
    from src.models.classifier import build_model

    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    target = count_params(build_model("cross_attention", cfg, 1.0))
    print(f"target (cross_attention, width_mult=1.0): {target:,} params")
    out = {}
    builders = {v: (lambda w, v=v: build_reproduction(v, cfg, w)) for v in REPRODUCTIONS}
    builders["alarm_type_contrastive"] = lambda w: build_alarm_type_contrastive(cfg, w)
    for name, b in builders.items():
        wm = find_width(b, target)
        n = count_params(b(wm))
        out[name] = wm
        print(f"{name}: width_mult={wm} -> {n:,} params ({n / target:.1%} of target)")
    p = ROOT / "configs" / "width_mult_reproductions.yaml"
    p.write_text(yaml.dump(out, sort_keys=False))
    print(f"written to {p}")


if __name__ == "__main__":
    main()
