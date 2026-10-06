"""Foundation-model PPG encoder wrapper (PaPaGei, Pillai et al., ICLR 2025).

Code  : https://github.com/Nokia-Bell-Labs/papagei-foundation-model (BSD-3-Clause-Clear)
Weights: https://zenodo.org/records/13983110 (papagei_s.pt, papagei_p.pt) - no licence
         field is set on the record, see docs/foundation_models.md.

The upstream architecture file (models/resnet.py) is NOT copied into src/; it is
loaded from data/external/papagei/code/ so the vendored upstream source stays
byte-identical to the original.

Input convention (taken from the upstream README / feature_extraction script):
    10 s segments, 125 Hz (1250 samples), per-segment z-score, shape (B, 1, 1250).
Ours: 10 s, 250 Hz (2500 samples), already band-passed 0.5-8 Hz and z-scored.
`PaPaGeiEncoder.prepare` therefore (1) low-pass + decimates 250 -> 125 Hz with
the same zero-phase-equivalent Kaiser FIR that scipy.signal.resample_poly(x,1,2)
uses (implemented in torch so it can sit inside a fine-tuned model), then
(2) re-z-scores every window. Window length is exactly 1250 samples, so no
cropping or padding is required.

Output: `mode="seq"` -> (B, T', 512) final feature map of the ResNet trunk
(T' = 3 for 1250-sample input, nine stride-2 stages) ; `mode="pooled"` ->
(B, 1, 512). `embedding(x)` returns the upstream "embedding" vector
(`outputs[0]`, the dense 512-d layer for PaPaGei-S) or the 512-d mean-pooled
trunk feature (`kind="pooled"`).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

ROOT = Path(__file__).resolve().parents[2]
PAPAGEI_DIR = ROOT / "data" / "external" / "papagei"
FEATURE_DIM = 512
PAPAGEI_FS = 125
PAPAGEI_LEN = 1250


def _load_upstream_resnet():
    path = PAPAGEI_DIR / "code" / "models" / "resnet.py"
    if not path.exists():
        raise FileNotFoundError(f"PaPaGei source not found at {path}; see docs/foundation_models.md")
    spec = importlib.util.spec_from_file_location("papagei_resnet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_papagei(variant: str = "s", weights_dir: Path | None = None) -> nn.Module:
    """variant 's' = PaPaGei-S (ResNet1DMoE), 'p' = PaPaGei-P (ResNet1D, dense 512)."""
    weights_dir = Path(weights_dir or PAPAGEI_DIR)
    rn = _load_upstream_resnet()
    if variant == "s":
        net = rn.ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1,
                             n_block=18, n_classes=512, n_experts=3)
    elif variant == "p":
        net = rn.ResNet1D(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1,
                          n_block=18, n_classes=512, use_mt_regression=False, use_projection=False)
    else:
        raise ValueError(variant)
    sd = torch.load(weights_dir / f"papagei_{variant}.pt", map_location="cpu")
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    net.load_state_dict(sd)  # strict: raises on any mismatch
    return net


class PaPaGeiEncoder(nn.Module):
    """(B, 2500) 250 Hz PPG window -> (B, T', 512) features.

    trainable: 'none' (frozen, BN in eval), 'last_block' (last residual block +
    final BN; everything else frozen) or 'all'.
    """

    def __init__(self, variant: str = "s", mode: str = "seq", trainable: str = "none",
                 weights_dir: Path | None = None, src_fs: int = 250):
        super().__init__()
        assert mode in ("seq", "pooled") and trainable in ("none", "last_block", "all")
        assert src_fs % PAPAGEI_FS == 0
        self.net = build_papagei(variant, weights_dir)
        self.variant, self.mode, self.trainable = variant, mode, trainable
        self.out_dim = FEATURE_DIM
        down = src_fs // PAPAGEI_FS
        self.down = down
        taps = firwin(20 * down + 1, 1.0 / down, window=("kaiser", 5.0))  # == resample_poly(x, 1, down)
        self.register_buffer("fir", torch.tensor(taps, dtype=torch.float32).view(1, 1, -1), persistent=False)
        self._set_trainable()

    # ------------------------------------------------------------------ freezing
    def _live_modules(self) -> list[nn.Module]:
        if self.trainable == "all":
            return [self.net]
        if self.trainable == "last_block":
            return [self.net.basicblock_list[-1], self.net.final_bn]
        return []

    def _set_trainable(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = False
        for m in self._live_modules():
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:  # frozen parts keep eval behaviour (BN statistics, dropout off)
            self.net.eval()
            for m in self._live_modules():
                m.train(True)
        return self

    # ------------------------------------------------------------------ signal path
    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) at src_fs -> (B, 1, 1250) at 125 Hz, per-window z-scored."""
        x = x.unsqueeze(1)
        if self.down > 1:
            pad = (self.fir.shape[-1] - 1) // 2
            x = F.conv1d(F.pad(x, (pad, pad)), self.fir, stride=self.down)
        x = x[..., :PAPAGEI_LEN]
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def trunk(self, x125: torch.Tensor) -> torch.Tensor:
        n = self.net
        out = n.first_block_relu(n.first_block_bn(n.first_block_conv(x125)))
        for blk in n.basicblock_list:
            out = blk(out)
        return n.final_relu(n.final_bn(out))  # (B, 512, T')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable != "none"):
            fmap = self.trunk(self.prepare(x))
        if self.mode == "pooled":
            return fmap.mean(-1, keepdim=True).transpose(1, 2)
        return fmap.transpose(1, 2)

    @torch.no_grad()
    def embedding(self, x: torch.Tensor, kind: str = "upstream") -> torch.Tensor:
        """kind='upstream': outputs[0] as in the PaPaGei README (dense 512-d for -S,
        dense 512-d for -P too); kind='pooled': mean-pooled 512-d trunk feature."""
        pooled = self.trunk(self.prepare(x)).mean(-1)
        if kind == "pooled":
            return pooled
        n = self.net
        return n.projector(pooled) if getattr(n, "use_projection", False) else n.dense(pooled)
