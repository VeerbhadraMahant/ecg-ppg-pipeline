"""Tests for the PaPaGei wrapper. Skipped if the weights are not downloaded."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "external" / "papagei" / "papagei_s.pt").exists(), reason="PaPaGei weights not present"
)

from src.models.foundation import PaPaGeiEncoder  # noqa: E402


def test_resample_matches_scipy():
    enc = PaPaGeiEncoder("s")
    x = np.random.RandomState(0).randn(3, 2500).astype(np.float32)
    ours = enc.prepare(torch.from_numpy(x)).squeeze(1).numpy()
    ref = resample_poly(x, 1, 2, axis=-1)
    ref = (ref - ref.mean(-1, keepdims=True)) / ref.std(-1, keepdims=True)
    assert ours.shape == (3, 1250)
    assert np.abs(ours - ref).max() < 5e-3  # filter-design rounding only


def test_shapes_and_freezing():
    x = torch.randn(4, 2500)
    enc = PaPaGeiEncoder("s", mode="seq", trainable="none").train()
    assert enc(x).shape == (4, 3, 512)
    assert not any(p.requires_grad for p in enc.parameters())
    assert not enc.net.training  # BN stays in eval when frozen
    assert enc.embedding(x).shape == (4, 512)
    ft = PaPaGeiEncoder("s", trainable="last_block")
    n = sum(p.numel() for p in ft.parameters() if p.requires_grad)
    assert 0 < n < sum(p.numel() for p in ft.parameters())
    assert PaPaGeiEncoder("s", mode="pooled")(x).shape == (4, 1, 512)


def test_deterministic_when_frozen():
    enc = PaPaGeiEncoder("s").eval()
    x = torch.randn(2, 2500)
    assert torch.allclose(enc(x), enc(x))
