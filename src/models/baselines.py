"""Baseline zoo (updates.md section 3.4): strong generic sequence models that
see ECG and PPG as a 2-channel input (early fusion at the signal level).

Every model exposes forward(ecg, ppg) -> (logits (B,), None) like
AlarmClassifier, and takes `width_mult` so scripts/match_params.py can bring
each to the same parameter budget as the cross-attention reference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import PositionalEncoding

ZOO = ["resnet1d", "inceptiontime", "bigru", "tcn", "transformer", "ssm"]


def _w(base: int, mult: float) -> int:
    return max(4, int(round(base * mult)))


class _Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ZooBase(nn.Module):
    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        x = torch.stack([ecg, ppg], dim=1)  # (B, 2, T)
        return self.head(self.features(x)), None


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 9, stride, 4, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 9, 1, 4, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.drop(y)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class ResNet1D(_ZooBase):
    def __init__(self, width_mult=1.0, dropout=0.3, head_hidden=64):
        super().__init__()
        w = _w(24, width_mult)
        self.stem = nn.Sequential(nn.Conv1d(2, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.ReLU())
        chans = [w, 2 * w, 4 * w]
        blocks, cin = [], w
        for i, c in enumerate(chans):
            blocks += [_ResBlock(cin, c, 1 if i == 0 else 2, dropout), _ResBlock(c, c, 1, dropout)]
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = _Head(cin, head_hidden, dropout)

    def features(self, x):
        return self.blocks(self.stem(x)).mean(dim=-1)


class _Inception(nn.Module):
    def __init__(self, cin, nf, dropout):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        bc = nf if cin > 1 else cin
        self.convs = nn.ModuleList([nn.Conv1d(bc, nf, k, padding=k // 2, bias=False) for k in (39, 19, 9)])
        self.pool_conv = nn.Conv1d(cin, nf, 1, bias=False)
        self.bn = nn.BatchNorm1d(4 * nf)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.bottleneck(x)
        outs = [c(z) for c in self.convs] + [self.pool_conv(F.max_pool1d(x, 3, 1, 1))]
        return self.drop(F.relu(self.bn(torch.cat(outs, dim=1))))


class InceptionTime(_ZooBase):
    def __init__(self, width_mult=1.0, dropout=0.3, head_hidden=64):
        super().__init__()
        nf = _w(8, width_mult)
        self.down = nn.Conv1d(2, 2, 8, 4, 2)  # shorten T for the wide kernels
        self.m1, self.m2, self.m3 = _Inception(2, nf, dropout), _Inception(4 * nf, nf, dropout), _Inception(4 * nf, nf, dropout)
        self.res = nn.Sequential(nn.Conv1d(2, 4 * nf, 1, bias=False), nn.BatchNorm1d(4 * nf))
        self.head = _Head(4 * nf, head_hidden, dropout)

    def features(self, x):
        x = self.down(x)
        y = self.m3(self.m2(self.m1(x)))
        return F.relu(y + self.res(x)).mean(dim=-1)


class BiGRU(_ZooBase):
    def __init__(self, width_mult=1.0, dropout=0.3, head_hidden=64):
        super().__init__()
        w = _w(48, width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(2, w, 7, 4, 3), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, w, 7, 4, 3), nn.BatchNorm1d(w), nn.ReLU(),
        )
        self.gru = nn.GRU(w, w, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = _Head(2 * w, head_hidden, dropout)

    def features(self, x):
        h, _ = self.gru(self.stem(x).transpose(1, 2))
        return self.drop(h).mean(dim=1)


class _TCNBlock(nn.Module):
    def __init__(self, c, dilation, dropout):
        super().__init__()
        pad = 3 * dilation
        self.c1 = nn.Conv1d(c, c, 7, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(c, c, 7, padding=pad, dilation=dilation)
        self.n1, self.n2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.relu(self.n1(self.c1(x))))
        y = self.drop(F.relu(self.n2(self.c2(y))))
        return F.relu(x + y)


class TCN(_ZooBase):
    def __init__(self, width_mult=1.0, dropout=0.3, head_hidden=64):
        super().__init__()
        w = _w(40, width_mult)
        self.stem = nn.Sequential(nn.Conv1d(2, w, 8, 4, 2), nn.BatchNorm1d(w), nn.ReLU())
        self.blocks = nn.Sequential(*[_TCNBlock(w, d, dropout) for d in (1, 2, 4, 8, 16)])
        self.head = _Head(w, head_hidden, dropout)

    def features(self, x):
        return self.blocks(self.stem(x)).mean(dim=-1)


class PlainTransformer(_ZooBase):
    def __init__(self, width_mult=1.0, dropout=0.3, head_hidden=64):
        super().__init__()
        d = max(8, 8 * int(round(_w(64, width_mult) / 8)))
        self.embed = nn.Conv1d(2, d, 16, 16)  # patchify
        self.pos = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = _Head(d, head_hidden, dropout)

    def features(self, x):
        t = self.pos(self.embed(x).transpose(1, 2))
        return self.enc(t).mean(dim=1)


class S4DLayer(nn.Module):
    """Diagonal state-space layer (S4D-style): a global convolution whose
    kernel is generated from a complex diagonal SSM, applied with FFT."""

    def __init__(self, d_model: int, d_state: int = 16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.h, self.n = d_model, d_state
        self.log_dt = nn.Parameter(torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_a_re = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.a_im = nn.Parameter(math.pi * torch.arange(d_state).float().repeat(d_model, 1))
        self.c = nn.Parameter(torch.randn(d_model, d_state, 2) * (1.0 / math.sqrt(d_state)))
        self.d = nn.Parameter(torch.randn(d_model))

    def kernel(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (H,1)
        a = -torch.exp(self.log_a_re) + 1j * self.a_im  # (H,N)
        c = torch.view_as_complex(self.c)
        dta = a * dt
        k = torch.arange(length, device=dta.device)
        vander = torch.exp(dta.unsqueeze(-1) * k)  # (H,N,L)
        coef = c * (torch.exp(dta) - 1.0) / a
        return 2 * torch.einsum("hn,hnl->hl", coef, vander).real  # (H,L)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u: (B, H, L)
        length = u.size(-1)
        k = self.kernel(length)
        fk = torch.fft.rfft(k, n=2 * length)
        fu = torch.fft.rfft(u, n=2 * length)
        y = torch.fft.irfft(fu * fk, n=2 * length)[..., :length]
        return y + u * self.d.unsqueeze(-1)


class CompactSSM(_ZooBase):
    def __init__(self, width_mult=1.0, dropout=0.3, head_hidden=64, n_layers=3):
        super().__init__()
        h = _w(64, width_mult)
        self.stem = nn.Conv1d(2, h, 8, 8)  # T=2500 -> ~312
        self.layers = nn.ModuleList([S4DLayer(h) for _ in range(n_layers)])
        self.mix = nn.ModuleList([nn.Conv1d(h, 2 * h, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = _Head(h, head_hidden, dropout)

    def features(self, x):
        z = self.stem(x)
        for ssm, mix, norm in zip(self.layers, self.mix, self.norms):
            y = self.drop(F.gelu(ssm(norm(z))))
            z = z + F.glu(mix(y), dim=1)
        return z.mean(dim=-1)


_BUILDERS = {
    "resnet1d": ResNet1D,
    "inceptiontime": InceptionTime,
    "bigru": BiGRU,
    "tcn": TCN,
    "transformer": PlainTransformer,
    "ssm": CompactSSM,
}


def build_zoo_model(variant: str, cfg: dict, width_mult: float = 1.0) -> nn.Module:
    m = cfg["model"]
    return _BUILDERS[variant](width_mult=width_mult, dropout=m["dropout"], head_hidden=m["head_hidden"])
