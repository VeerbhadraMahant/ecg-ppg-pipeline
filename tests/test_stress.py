import numpy as np

from src.analysis import stress_benchmark as sb


def _clean(n=6, T=2500):
    t = np.arange(T) / 250
    x = np.stack([np.sin(2 * np.pi * 1.2 * t + i) for i in range(n)]).astype(np.float32)
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def _bank():
    r = np.random.RandomState(0)
    return {k: r.randn(2, 60000).astype(np.float32) for k in sb.ECG_NOISE_TYPES}


def _snr(x, n):
    return 10 * np.log10(np.mean(x ** 2, 1) / np.mean((n - n.mean(1, keepdims=True)) ** 2, 1))


def test_ecg_injection_hits_target_snr_and_is_znormalised():
    x = _clean()
    for snr in sb.SNR_LEVELS:
        out, n = sb.inject_ecg_noise(x, _bank(), "em", snr, np.random.RandomState(1), return_noise=True)
        assert np.allclose(_snr(x, n), snr, atol=1e-3)
        assert np.allclose(out.mean(1), 0, atol=1e-4) and np.allclose(out.std(1), 1, atol=1e-3)


def test_ppg_synthetic_noise_is_bandlimited_and_at_snr():
    x = _clean()
    out, n = sb.inject_ppg_noise(x, 6, np.random.RandomState(2), return_noise=True)
    assert np.allclose(_snr(x, n), 6, atol=1e-3)
    f = np.fft.rfftfreq(x.shape[1], 1 / 250)
    spec = np.abs(np.fft.rfft(n, axis=1)) ** 2
    assert spec[:, (f < 0.45) | (f > 5.05)].sum() < 1e-3 * spec.sum()


def test_ptt_shift_and_jitter():
    x = _clean()
    assert np.allclose(sb.shift_ppg(x, 0), x)
    s = sb.shift_ppg(x, 400)  # 100 samples
    assert np.corrcoef(s[0, 200:], x[0, 100:-100])[0, 1] > 0.99
    j = sb.jitter_ppg(x, 100, np.random.RandomState(0))
    assert j.shape == x.shape and np.isfinite(j).all()


def test_dropout_fraction_and_modes():
    x = _clean() + 3.0
    z = sb.dropout(x, 0.4, "zero", np.random.RandomState(0))
    assert np.allclose((z == 0).mean(1), 0.4, atol=1e-3)
    fl = sb.dropout(x, 0.4, "flatline", np.random.RandomState(0))
    assert fl.shape == x.shape and (np.abs(np.diff(fl, axis=1)) < 1e-9).mean() >= 0.39
    assert np.array_equal(sb.dropout(x, 0.0, "zero", np.random.RandomState(0)), x)


def test_conditions_do_not_touch_labels_or_shapes_and_are_reproducible():
    x, p = _clean(), _clean(6) * 0.5
    conds = sb.build_conditions(_bank())
    assert conds[0]["family"] == "clean"
    for c in conds:
        e1, p1 = c["fn"](x.copy(), p.copy(), np.random.RandomState(3))
        assert e1.shape == x.shape and p1.shape == p.shape and np.isfinite(e1).all() and np.isfinite(p1).all()
        e2, p2 = c["fn"](x.copy(), p.copy(), np.random.RandomState(3))
        assert np.array_equal(e1, e2) and np.array_equal(p1, p2)


def test_inputs_not_mutated():
    x = _clean(); x0 = x.copy()
    sb.inject_ecg_noise(x, _bank(), "ma", 0, np.random.RandomState(0))
    sb.dropout(x, 0.5, "zero", np.random.RandomState(0))
    assert np.array_equal(x, x0)


def test_nstdb_loads_at_250hz():
    import pytest

    if not (sb.NSTDB_DIR / "em.dat").exists():
        pytest.skip("nstdb not downloaded")
    bank = sb.load_nstdb()
    assert bank["em"].shape[0] == 2 and abs(bank["em"].shape[1] - 650000 * 250 / 360) < 5
