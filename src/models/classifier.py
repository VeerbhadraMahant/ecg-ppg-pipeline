"""The four baseline-ladder variants (architecture.md section 5), sharing a
single classification head implementation so the only thing that differs
between them is encoders + fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import CNNEncoder
from .fusion import ConcatFusion, CrossAttentionFusion

VARIANTS = ["ecg_only", "ppg_only", "concat", "cross_attention"]


class ClassificationHead(nn.Module):
    """Deliberately shallow — see architecture.md section 4 on overfitting risk."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AlarmClassifier(nn.Module):
    """Unified model for all four ladder variants.

    variant:
        ecg_only         - single CNN encoder on ECG, mean-pooled
        ppg_only         - single CNN encoder on PPG, mean-pooled
        concat           - two encoders, naive concatenation fusion
        cross_attention  - two encoders, transformer cross-attention fusion (the proposed model)
    """

    def __init__(
        self,
        variant: str,
        cnn_channels: list[int],
        cnn_kernel_sizes: list[int],
        cnn_pool: int,
        dropout: float,
        attn_dim: int,
        attn_heads: int,
        head_hidden: int,
        width_mult: float = 1.0,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
        self.variant = variant

        def make_encoder():
            return CNNEncoder(cnn_channels, cnn_kernel_sizes, cnn_pool, dropout, width_mult)

        if variant == "ecg_only":
            self.ecg_encoder = make_encoder()
            head_in = self.ecg_encoder.out_dim
        elif variant == "ppg_only":
            self.ppg_encoder = make_encoder()
            head_in = self.ppg_encoder.out_dim
        elif variant == "concat":
            self.ecg_encoder = make_encoder()
            self.ppg_encoder = make_encoder()
            self.fusion = ConcatFusion(self.ecg_encoder.out_dim, self.ppg_encoder.out_dim, attn_dim)
            head_in = self.fusion.out_dim
        else:  # cross_attention
            self.ecg_encoder = make_encoder()
            self.ppg_encoder = make_encoder()
            self.fusion = CrossAttentionFusion(
                self.ecg_encoder.out_dim, self.ppg_encoder.out_dim, attn_dim, attn_heads, dropout
            )
            head_in = self.fusion.out_dim

        self.head = ClassificationHead(head_in, head_hidden, dropout)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        """ecg, ppg: (B, T). Returns logits (B,), and attn_weights (or None)."""
        attn_weights = None
        if self.variant == "ecg_only":
            feat = self.ecg_encoder(ecg).mean(dim=1)
        elif self.variant == "ppg_only":
            feat = self.ppg_encoder(ppg).mean(dim=1)
        elif self.variant == "concat":
            ecg_seq = self.ecg_encoder(ecg)
            ppg_seq = self.ppg_encoder(ppg)
            feat = self.fusion(ecg_seq, ppg_seq)
        else:  # cross_attention
            ecg_seq = self.ecg_encoder(ecg)
            ppg_seq = self.ppg_encoder(ppg)
            fused_seq, attn_weights = self.fusion(ecg_seq, ppg_seq)
            feat = fused_seq.mean(dim=1)

        logits = self.head(feat)
        return logits, attn_weights


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant: str, cfg: dict, width_mult: float = 1.0) -> AlarmClassifier:
    m = cfg["model"]
    return AlarmClassifier(
        variant=variant,
        cnn_channels=m["cnn_channels"],
        cnn_kernel_sizes=m["cnn_kernel_sizes"],
        cnn_pool=m["cnn_pool"],
        dropout=m["dropout"],
        attn_dim=m["attn_dim"],
        attn_heads=m["attn_heads"],
        head_hidden=m["head_hidden"],
        width_mult=width_mult,
    )
