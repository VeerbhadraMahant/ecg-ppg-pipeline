"""Cross-attention fusion layer and a naive concatenation alternative.

Per architecture.md section 3: ECG supplies queries, PPG supplies keys/values.
Positional encodings are added before attention since attention is otherwise
permutation-invariant and would discard the timing information the whole
task depends on.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class CrossAttentionFusion(nn.Module):
    """ECG queries attend over PPG keys/values (one transformer MHA layer)."""

    def __init__(self, ecg_dim: int, ppg_dim: int, attn_dim: int, n_heads: int, dropout: float = 0.3):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim, attn_dim)
        self.ppg_proj = nn.Linear(ppg_dim, attn_dim)
        self.pos_enc = PositionalEncoding(attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.out_dim = attn_dim

    def forward(self, ecg_seq: torch.Tensor, ppg_seq: torch.Tensor):
        """ecg_seq: (B, T_e, d_e), ppg_seq: (B, T_p, d_p) -> (B, T_e, attn_dim), attn_weights (B, T_e, T_p)"""
        q = self.pos_enc(self.ecg_proj(ecg_seq))
        kv = self.pos_enc(self.ppg_proj(ppg_seq))
        attended, attn_weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        fused = self.norm(q + attended)
        return fused, attn_weights


class ConcatFusion(nn.Module):
    """Naive fusion baseline: project both sequences to the same dim, mean-pool
    each over time, and concatenate. No cross-modal comparison — isolates the
    value of *having* both signals vs. actually attending across them.
    """

    def __init__(self, ecg_dim: int, ppg_dim: int, out_dim: int):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim, out_dim // 2)
        self.ppg_proj = nn.Linear(ppg_dim, out_dim // 2)
        self.out_dim = out_dim

    def forward(self, ecg_seq: torch.Tensor, ppg_seq: torch.Tensor) -> torch.Tensor:
        ecg_vec = self.ecg_proj(ecg_seq).mean(dim=1)
        ppg_vec = self.ppg_proj(ppg_seq).mean(dim=1)
        return torch.cat([ecg_vec, ppg_vec], dim=-1)
