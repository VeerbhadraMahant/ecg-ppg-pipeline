"""Multitask alarm model (updates.md 5.5/5.6/7.1/7.2): the cross_attention
variant's encoders + fusion, plus auxiliary heads on the shared backbone.

Heads
  main    : alarm TRUE/FALSE logit (identical ClassificationHead to AlarmClassifier)
  type    : 5-way alarm-type logits from the pooled fused feature
  quality : per-channel SQI regression (sigmoid, [ecg, ppg]) from each
            encoder's own time-pooled tokens (so ECG quality sees only ECG)
  beat    : per-token R-peak logit (from the ECG-aligned fused tokens) and
            per-token PPG-peak logit (from the PPG encoder tokens); shape (B, T', 2)

With all auxiliary loss weights = 0 the auxiliary heads receive no gradient and
the model is functionally the plain cross_attention classifier, which is the
"without auxiliary heads" arm of the task-interference guardrail.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import ClassificationHead
from .encoders import CNNEncoder
from .fusion import CrossAttentionFusion

N_ALARM_TYPES = 5


class MultiTaskAlarmModel(nn.Module):
    def __init__(self, cnn_channels, cnn_kernel_sizes, cnn_pool, dropout, attn_dim, attn_heads,
                 head_hidden, width_mult: float = 1.0, n_types: int = N_ALARM_TYPES):
        super().__init__()
        self.ecg_encoder = CNNEncoder(cnn_channels, cnn_kernel_sizes, cnn_pool, dropout, width_mult)
        self.ppg_encoder = CNNEncoder(cnn_channels, cnn_kernel_sizes, cnn_pool, dropout, width_mult)
        self.fusion = CrossAttentionFusion(self.ecg_encoder.out_dim, self.ppg_encoder.out_dim,
                                           attn_dim, attn_heads, dropout)
        self.head = ClassificationHead(self.fusion.out_dim, head_hidden, dropout)
        self.type_head = nn.Sequential(nn.Linear(self.fusion.out_dim, head_hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(head_hidden, n_types))
        self.q_ecg = nn.Linear(self.ecg_encoder.out_dim, 1)
        self.q_ppg = nn.Linear(self.ppg_encoder.out_dim, 1)
        self.beat_ecg = nn.Linear(self.fusion.out_dim, 1)
        self.beat_ppg = nn.Linear(self.ppg_encoder.out_dim, 1)

    def forward_all(self, ecg: torch.Tensor, ppg: torch.Tensor) -> dict:
        ecg_seq = self.ecg_encoder(ecg)
        ppg_seq = self.ppg_encoder(ppg)
        fused, attn = self.fusion(ecg_seq, ppg_seq)
        feat = fused.mean(dim=1)
        return {
            "logit": self.head(feat),
            "type_logits": self.type_head(feat),
            "quality": torch.sigmoid(torch.cat([self.q_ecg(ecg_seq.mean(1)), self.q_ppg(ppg_seq.mean(1))], dim=1)),
            "beat_logits": torch.cat([self.beat_ecg(fused), self.beat_ppg(ppg_seq)], dim=-1),  # (B, T', 2)
            "attn": attn,
        }

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        """Same contract as AlarmClassifier: (logits (B,), attn_weights)."""
        out = self.forward_all(ecg, ppg)
        return out["logit"], out["attn"]


def build_multitask(cfg: dict, width_mult: float = 1.0) -> MultiTaskAlarmModel:
    m = cfg["model"]
    return MultiTaskAlarmModel(m["cnn_channels"], m["cnn_kernel_sizes"], m["cnn_pool"], m["dropout"],
                               m["attn_dim"], m["attn_heads"], m["head_hidden"], width_mult)
