"""Loss functions for the (severely imbalanced) binary alarm classification
task. architecture.md section 4: class weighting or focal loss, not
resampling, since resampling risks duplicating patients across folds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        return (focal_term * bce).mean()


def build_loss(loss_name: str, pos_weight: float, gamma: float = 2.0, device: str = "cpu") -> nn.Module:
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma, pos_weight=pw)
    if loss_name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    raise ValueError(f"unknown loss {loss_name!r}")
