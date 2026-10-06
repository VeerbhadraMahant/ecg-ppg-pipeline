import numpy as np

from src.metrics import (
    challenge_score,
    operating_points,
    suppression_rate,
    threshold_for_sensitivity,
)
from src.stats import corrected_resampled_ttest, holm, record_bootstrap


def test_challenge_score_weights_false_negatives_five_times():
    # 10 TP, 10 TN, 0 FP, 2 FN -> 20 / (20 + 10) = 66.7
    assert abs(challenge_score(10, 10, 0, 2) - 100 * 20 / 30) < 1e-9
    # a false positive costs 1, a false negative costs 5
    assert challenge_score(10, 10, 1, 0) > challenge_score(10, 10, 0, 1)


def test_threshold_for_sensitivity_reaches_target_on_validation():
    rng = np.random.RandomState(0)
    y = np.array([1] * 40 + [0] * 60)
    p = np.concatenate([rng.uniform(0.3, 1, 40), rng.uniform(0, 0.7, 60)])
    for target in (0.95, 0.99, 1.0):
        t = threshold_for_sensitivity(y, p, target)
        sens = (p[y == 1] >= t).mean()
        assert sens >= target - 1e-9
    # 100% sensitivity threshold is the lowest positive score
    assert threshold_for_sensitivity(y, p, 1.0) == p[y == 1].min()


def test_operating_points_use_validation_threshold_on_test():
    yv = np.array([1, 1, 1, 0, 0, 0])
    pv = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
    yt = np.array([1, 1, 0, 0])
    pt = np.array([0.5, 0.35, 0.45, 0.05])
    ops = operating_points(yv, pv, yt, pt)
    assert ops["s100"]["threshold"] == 0.4  # lowest validation positive
    assert ops["s100"]["fn"] == 1  # the 0.35 test positive is missed at that threshold
    assert suppression_rate(ops["t50"]["tn"], ops["t50"]["fp"]) == 1.0


def test_nadeau_bengio_is_more_conservative_than_naive():
    from scipy import stats

    rng = np.random.RandomState(1)
    d = rng.normal(0.02, 0.03, 15)
    naive_p = stats.ttest_1samp(d, 0).pvalue
    nb = corrected_resampled_ttest(d, n_train=400, n_test=100)
    assert nb["p_value"] > naive_p


def test_record_bootstrap_detects_clear_and_null_differences():
    rng = np.random.RandomState(2)
    n = 300
    # model A: 70% correct on positives, model B: 95%; same negatives
    y = rng.rand(n) < 0.4

    def tallies(p_correct_pos):
        t = np.zeros((2, n, 4))
        for s in range(2):
            hit = rng.rand(n) < p_correct_pos
            t[s, :, 0] = y & hit
            t[s, :, 3] = y & ~hit
            t[s, :, 2] = ~y
        return t

    a, b = tallies(0.7), tallies(0.95)
    res = record_bootstrap(a, b, "sensitivity", n_boot=500)
    assert res["obs_diff"] > 0.1 and res["ci95_low"] > 0 and res["p_value"] < 0.05
    null = record_bootstrap(a, a, "sensitivity", n_boot=500)
    assert null["obs_diff"] == 0 and null["ci95_low"] <= 0 <= null["ci95_high"]


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-12
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] <= adj[2] <= adj[1] + 1e-12
