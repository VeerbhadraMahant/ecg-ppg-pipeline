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


# --------------------------------------------------------------------------
# Stronger fusion rivals (updates.md section 3.5). Every module here follows
# the CrossAttentionFusion contract: forward(ecg_seq, ppg_seq) ->
# (fused_seq (B, T', attn_dim), attn_weights or None), with `out_dim`.
# --------------------------------------------------------------------------


class BiCrossAttentionFusion(nn.Module):
    """ECG->PPG and PPG->ECG attention, outputs concatenated along features
    (after aligning lengths — both encoders share T' so they already match)."""

    def __init__(self, ecg_dim: int, ppg_dim: int, attn_dim: int, n_heads: int, dropout: float = 0.3):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim, attn_dim)
        self.ppg_proj = nn.Linear(ppg_dim, attn_dim)
        self.pos_enc = PositionalEncoding(attn_dim)
        self.attn_e2p = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_p2e = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_e = nn.LayerNorm(attn_dim)
        self.norm_p = nn.LayerNorm(attn_dim)
        self.merge = nn.Linear(2 * attn_dim, attn_dim)
        self.out_dim = attn_dim

    def forward(self, ecg_seq, ppg_seq):
        e = self.pos_enc(self.ecg_proj(ecg_seq))
        p = self.pos_enc(self.ppg_proj(ppg_seq))
        ae, w = self.attn_e2p(e, p, p, need_weights=True, average_attn_weights=True)
        ap, _ = self.attn_p2e(p, e, e, need_weights=False)
        e_out = self.norm_e(e + ae)
        p_out = self.norm_p(p + ap)
        n = min(e_out.size(1), p_out.size(1))
        return self.merge(torch.cat([e_out[:, :n], p_out[:, :n]], dim=-1)), w


class JointTransformerFusion(nn.Module):
    """Early-fusion joint transformer: both token sequences concatenated with
    learned modality embeddings and processed by one self-attention layer."""

    def __init__(self, ecg_dim: int, ppg_dim: int, attn_dim: int, n_heads: int, dropout: float = 0.3):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim, attn_dim)
        self.ppg_proj = nn.Linear(ppg_dim, attn_dim)
        self.pos_enc = PositionalEncoding(attn_dim)
        self.mod_emb = nn.Parameter(torch.zeros(2, attn_dim))
        nn.init.normal_(self.mod_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            attn_dim, n_heads, dim_feedforward=2 * attn_dim, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.out_dim = attn_dim

    def forward(self, ecg_seq, ppg_seq):
        e = self.pos_enc(self.ecg_proj(ecg_seq)) + self.mod_emb[0]
        p = self.pos_enc(self.ppg_proj(ppg_seq)) + self.mod_emb[1]
        return self.encoder(torch.cat([e, p], dim=1)), None


class BottleneckFusion(nn.Module):
    """Bottleneck-token fusion: K learned tokens read ECG, then PPG, so all
    cross-modal information must pass through a narrow shared channel."""

    def __init__(self, ecg_dim: int, ppg_dim: int, attn_dim: int, n_heads: int, dropout: float = 0.3, n_tokens: int = 8):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim, attn_dim)
        self.ppg_proj = nn.Linear(ppg_dim, attn_dim)
        self.pos_enc = PositionalEncoding(attn_dim)
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, attn_dim) * 0.02)
        self.read_ecg = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.read_ppg = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.out_dim = attn_dim

    def forward(self, ecg_seq, ppg_seq):
        e = self.pos_enc(self.ecg_proj(ecg_seq))
        p = self.pos_enc(self.ppg_proj(ppg_seq))
        b = self.tokens.expand(e.size(0), -1, -1)
        b = self.norm1(b + self.read_ecg(b, e, e, need_weights=False)[0])
        b = self.norm2(b + self.read_ppg(b, p, p, need_weights=False)[0])
        return b, None


class GatedCrossAttentionFusion(nn.Module):
    """Signal-quality-gated cross-attention: a learned scalar gate per
    modality (from pooled features) scales each sequence before attention, so
    the model can down-weight a corrupted channel."""

    def __init__(self, ecg_dim: int, ppg_dim: int, attn_dim: int, n_heads: int, dropout: float = 0.3):
        super().__init__()
        self.inner = CrossAttentionFusion(ecg_dim, ppg_dim, attn_dim, n_heads, dropout)
        self.gate_e = nn.Sequential(nn.Linear(ecg_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.gate_p = nn.Sequential(nn.Linear(ppg_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.out_dim = attn_dim
        self.last_gates = None

    def forward(self, ecg_seq, ppg_seq):
        ge = self.gate_e(ecg_seq.mean(dim=1)).unsqueeze(1)
        gp = self.gate_p(ppg_seq.mean(dim=1)).unsqueeze(1)
        self.last_gates = (ge.detach().squeeze(), gp.detach().squeeze())
        return self.inner(ecg_seq * ge, ppg_seq * gp)


FUSION_REGISTRY = {
    "cross_attention": CrossAttentionFusion,
    "bi_cross_attention": BiCrossAttentionFusion,
    "joint_transformer": JointTransformerFusion,
    "bottleneck": BottleneckFusion,
    "gated_cross_attention": GatedCrossAttentionFusion,
}
