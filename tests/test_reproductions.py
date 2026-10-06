import torch

from src.models.reproductions import (
    REPRODUCTIONS,
    AlarmTypeContrastive,
    build_alarm_type_contrastive,
    build_reproduction,
    count_params,
    supcon_loss,
)

CFG = {"model": {"dropout": 0.1, "head_hidden": 8}}
B, T = 4, 2500


def test_reproduction_forward_shapes_and_params(capsys):
    for v in REPRODUCTIONS:
        m = build_reproduction(v, CFG, 0.5)
        logits, attn = m(torch.randn(B, T), torch.randn(B, T))
        assert logits.shape == (B,) and attn is None
        print(f"{v}: {count_params(m):,} params")
        assert count_params(m) > 0


def test_gradient_flows_to_every_parameter():
    m = build_reproduction("mousavi_attn_cnn_rnn", CFG, 0.5)
    m.train()
    logits, _ = m(torch.randn(B, T), torch.randn(B, T))
    logits.sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert any(p.grad.abs().sum() > 0 for p in m.parameters())


def test_alarm_type_contrastive_forward_grad_and_supcon():
    m = build_alarm_type_contrastive(CFG, 0.5)
    assert isinstance(m, AlarmTypeContrastive)
    ecg, ppg = torch.randn(B, T), torch.randn(B, T)
    at = torch.tensor([0, 1, 2, 4])
    logits, attn = m(ecg, ppg, at)
    assert logits.shape == (B,) and attn is None
    assert m(ecg, ppg)[0].shape == (B,)  # unknown-type fallback
    logits, z = m.forward_with_embedding(ecg, ppg, at)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = logits.sum() + supcon_loss(z, y)
    loss.backward()
    assert m.type_emb.weight.grad is not None and m.type_emb.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for n, p in m.named_parameters())


def test_supcon_degenerate_batch_is_zero():
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    assert supcon_loss(z, torch.tensor([0.0, 1.0, 2.0])).item() == 0.0


def test_param_count_close_to_budget_at_matched_width():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    wm = yaml.safe_load((root / "configs" / "width_mult_reproductions.yaml").read_text())
    n = count_params(build_reproduction("mousavi_attn_cnn_rnn", cfg, wm["mousavi_attn_cnn_rnn"]))
    assert abs(n - 261_505) / 261_505 < 0.03
