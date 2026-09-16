import torch

from src.models.classifier import VARIANTS, build_model, count_params

CFG = {
    "model": {
        "cnn_channels": [8, 16, 32],
        "cnn_kernel_sizes": [15, 9, 7],
        "cnn_pool": 2,
        "dropout": 0.1,
        "attn_dim": 16,
        "attn_heads": 2,
        "head_hidden": 8,
    }
}


def test_all_variants_forward_pass_shapes():
    batch, T = 4, 2500
    ecg = torch.randn(batch, T)
    ppg = torch.randn(batch, T)
    for variant in VARIANTS:
        model = build_model(variant, CFG)
        logits, attn = model(ecg, ppg)
        assert logits.shape == (batch,)
        if variant == "cross_attention":
            assert attn is not None
            assert attn.shape[0] == batch
        else:
            assert attn is None


def test_encoder_output_is_sequence_not_pooled_vector():
    from src.models.encoders import CNNEncoder

    enc = CNNEncoder([8, 16], [7, 5], pool=2, dropout=0.0)
    x = torch.randn(2, 1000)
    out = enc(x)
    assert out.dim() == 3  # (B, T', d) -- must retain the time axis
    assert out.shape[1] > 1


def test_param_counts_are_positive_and_differ_by_variant():
    counts = {v: count_params(build_model(v, CFG)) for v in VARIANTS}
    assert all(c > 0 for c in counts.values())
    assert counts["concat"] != counts["ecg_only"]
