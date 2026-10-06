"""Multi-modality alarm verifier (updates.md 5.1): one model for any available
combination of ECG lead 1, ECG lead 2, PPG and ABP.

Design
- One CNN encoder per signal *family*: ECG leads share an encoder (same
  physiology, different projection), PPG and ABP each get their own.
- Each modality yields a token sequence (T', d) -> linear projection to
  `attn_dim` + positional encoding + learned MODALITY embedding.
- All tokens go through one joint transformer layer; absent modalities are
  excluded with a key-padding mask, so the model never sees fabricated data.
- Modality dropout during training randomly hides present modalities (always
  keeping at least one) so the model degrades gracefully when sensors drop
  out at inference (updates.md analysis 8 and 9).

Interface quirk, so the generic trainer can be reused unchanged:
forward(x, mask) where x is (B, 4, T) and mask is (B, 4) in {0,1}; the
trainer passes them in the `ecg` and `ppg` slots.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import ClassificationHead
from .encoders import CNNEncoder
from .fusion import PositionalEncoding

MODALITIES = ["ecg", "ecg2", "ppg", "abp"]


def to_multimodal_inputs(data: dict):
    """Stack the preprocessed npz arrays into (N, 4, T) + presence mask (N, 4).
    `ppg` in the npz is the pulse channel (PPG, or ABP where PPG is absent), so a
    true PPG is present only when pulse_is_abp is False."""
    n, T = data["ecg"].shape
    pulse_is_abp = data.get("pulse_is_abp", np_zeros_bool(n))
    has_abp = data["abp"].std(axis=1) > 0 if "abp" in data else np_zeros_bool(n)
    has_ecg2 = data["has_ecg2"] if "has_ecg2" in data else np_zeros_bool(n)
    x = torch.zeros(n, 4, T)
    x[:, 0] = torch.from_numpy(data["ecg"])
    x[:, 1] = torch.from_numpy(data["ecg2"]) if "ecg2" in data else 0
    x[:, 2] = torch.from_numpy(data["ppg"])
    x[:, 3] = torch.from_numpy(data["abp"]) if "abp" in data else 0
    mask = torch.zeros(n, 4)
    mask[:, 0] = 1
    mask[:, 1] = torch.from_numpy(has_ecg2.astype("float32"))
    mask[:, 2] = torch.from_numpy((~pulse_is_abp).astype("float32"))
    mask[:, 3] = torch.from_numpy(has_abp.astype("float32"))
    # records whose only pulse is the substituted ABP: the ABP slot carries it
    sub = pulse_is_abp
    x[sub, 3] = x[sub, 2]
    mask[sub, 3] = 1.0
    x[sub, 2] = 0
    return x.numpy(), mask.numpy()


def np_zeros_bool(n):
    import numpy as np

    return np.zeros(n, dtype=bool)


class MultiModalAlarmClassifier(nn.Module):
    def __init__(self, cnn_channels, cnn_kernel_sizes, cnn_pool, dropout, attn_dim, attn_heads,
                 head_hidden, width_mult: float = 1.0, p_modality_drop: float = 0.3):
        super().__init__()
        mk = lambda: CNNEncoder(cnn_channels, cnn_kernel_sizes, cnn_pool, dropout, width_mult)  # noqa: E731
        self.enc_ecg, self.enc_ppg, self.enc_abp = mk(), mk(), mk()
        d_enc = self.enc_ecg.out_dim
        self.proj = nn.ModuleList([nn.Linear(d_enc, attn_dim) for _ in MODALITIES])
        self.mod_emb = nn.Parameter(torch.randn(len(MODALITIES), attn_dim) * 0.02)
        self.pos = PositionalEncoding(attn_dim)
        layer = nn.TransformerEncoderLayer(attn_dim, attn_heads, 2 * attn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = ClassificationHead(attn_dim, head_hidden, dropout)
        self.p_drop = p_modality_drop

    def _drop(self, mask: torch.Tensor) -> torch.Tensor:
        present = mask > 0
        drop = (torch.rand_like(mask) < self.p_drop) & present
        left = present & ~drop
        none_left = ~left.any(dim=1)
        if none_left.any():  # never drop everything: restore the original mask for those rows
            drop[none_left] = False
        return (present & ~drop).float()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.training and self.p_drop > 0:
            mask = self._drop(mask)
        encs = [self.enc_ecg, self.enc_ecg, self.enc_ppg, self.enc_abp]
        tokens, pad = [], []
        for m, enc in enumerate(encs):
            seq = enc(x[:, m])  # (B, T', d_enc)
            seq = self.pos(self.proj[m](seq)) + self.mod_emb[m]
            tokens.append(seq)
            pad.append((mask[:, m] <= 0).unsqueeze(1).expand(-1, seq.size(1)))
        tok = torch.cat(tokens, dim=1)
        pad = torch.cat(pad, dim=1)  # True = ignore
        out = self.encoder(tok, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (out * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(pooled), None
