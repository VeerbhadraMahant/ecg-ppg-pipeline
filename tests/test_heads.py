"""Fast unit tests for the supporting heads (no network, no GPU)."""
import numpy as np
import torch

from src.heads.beat_ptt import (
    analyze_window, beat_heatmap_targets, match_beats, per_beat_ptt, prf, synthetic_pair, validate_synthetic,
)
from src.heads.quality import QualityNet, quality_targets, reg_metrics
from src.models.multitask import MultiTaskAlarmModel

CFG = dict(cnn_channels=[8, 16, 16], cnn_kernel_sizes=[15, 9, 7], cnn_pool=2, dropout=0.1, attn_dim=16,
           attn_heads=2, head_hidden=8)


def test_match_beats_and_prf():
    tp, fp, fn = match_beats(np.array([100, 300, 500]), np.array([105, 310, 900]), tol=30)
    assert (tp, fp, fn) == (2, 1, 1)
    assert prf(2, 1, 1)["f1"] == 2 * 2 / (2 * 2 + 1 + 1)
    assert match_beats(np.array([100]), np.array([]), 10) == (0, 0, 1)


def test_analyze_window_synthetic_ptt_and_hr():
    e, p, ref = synthetic_pair(hr_period=0.8, ptt=0.25)
    r = analyze_window(e, p)
    assert abs(len(r["r_peaks"]) - len(ref)) <= 1
    assert abs(r["hr_ecg"] - 75) < 3 and abs(r["hr_ppg"] - 75) < 3
    assert len(r["ptt"]) == len(r["r_peaks"])
    assert 0.1 < r["ptt_mean"] < 0.5 and r["pulse_coverage"] > 0.8


def test_per_beat_ptt_nan_when_no_pulse():
    out = per_beat_ptt(np.array([100, 600]), np.array([150]), fs=250)  # only first beat has a foot in range
    assert not np.isnan(out[0]) and np.isnan(out[1])


def test_synthetic_validation_clean_is_accurate():
    res = validate_synthetic()["results"]["noise0.0"]
    assert res["f1"] > 0.95


def test_heatmap_targets_shape_and_positions():
    t = beat_heatmap_targets([np.array([0, 1250, 2499])], T=2500, n_tokens=312)
    assert t.shape == (1, 312) and t[0, 0] == 1 and t[0, 156] == 1 and t[0, 311] == 1 and t.sum() == 3
    sm = beat_heatmap_targets([np.array([1250])], 2500, 312, sigma_tokens=2.0)
    assert abs(sm.max() - 1.0) < 1e-5 and (sm > 0).sum() > 1


def test_quality_targets_rank_clean_above_noise():
    e, p, _ = synthetic_pair(noise=0.05)  # small noise: pure impulse trains are 'flat' between beats
    rng = np.random.RandomState(0)
    n = rng.randn(2500)
    q = quality_targets(np.stack([e, n]), np.stack([p, n]))
    assert q.shape == (2, 2) and np.all((q >= 0) & (q <= 1))
    assert q[0, 0] > q[1, 0] and q[0, 1] > q[1, 1]


def test_quality_net_and_metrics():
    net = QualityNet([8, 16], [15, 9], 2, 0.1)
    out = net(torch.randn(3, 2500), torch.randn(3, 2500))
    assert out.shape == (3, 2) and ((out >= 0) & (out <= 1)).all()
    y = np.random.RandomState(0).rand(20, 2)
    m = reg_metrics(y, y)
    assert m["ecg"]["mae"] == 0 and m["ppg"]["r2"] == 1.0


def test_multitask_shapes_and_zero_weight_equivalence_of_interface():
    m = MultiTaskAlarmModel(**CFG)
    ecg, ppg = torch.randn(2, 2500), torch.randn(2, 2500)
    o = m.forward_all(ecg, ppg)
    tokens = m.ecg_encoder(ecg).shape[1]
    assert o["logit"].shape == (2,) and o["type_logits"].shape == (2, 5) and o["quality"].shape == (2, 2)
    assert o["beat_logits"].shape == (2, tokens, 2)
    logits, attn = m(ecg, ppg)  # AlarmClassifier-compatible contract
    assert logits.shape == (2,) and attn.shape == (2, tokens, tokens)


def test_multitask_aux_loss_backprops_to_shared_encoder_only_when_weighted():
    m = MultiTaskAlarmModel(**CFG)
    o = m.forward_all(torch.randn(2, 2500), torch.randn(2, 2500))
    o["quality"].sum().backward()  # aux loss only
    assert m.ecg_encoder.blocks[0].conv.weight.grad is not None  # shares the backbone
    assert m.head.net[0].weight.grad is None  # but never touches the main head
