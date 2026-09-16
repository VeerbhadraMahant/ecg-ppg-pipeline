"""Modality-specific 1D-CNN encoders.

Per architecture.md section 2, each encoder outputs a SEQUENCE (T', d), not a
pooled vector — global average pooling here would remove the time axis that
cross-attention needs to align ECG against PPG.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return self.drop(x)


class CNNEncoder(nn.Module):
    """Stack of conv blocks turning (B, 1, T) into (B, T', d)."""

    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool: int = 2,
        dropout: float = 0.3,
        width_mult: float = 1.0,
    ):
        super().__init__()
        channels = channels or [32, 64, 128]
        kernel_sizes = kernel_sizes or [15, 9, 7]
        channels = [max(1, int(round(c * width_mult))) for c in channels]
        assert len(channels) == len(kernel_sizes)

        blocks = []
        in_ch = 1
        for out_ch, k in zip(channels, kernel_sizes):
            blocks.append(ConvBlock(in_ch, out_ch, k, pool, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, T', out_dim)"""
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.blocks(x)  # (B, out_dim, T')
        return x.transpose(1, 2)  # (B, T', out_dim)
