import numpy as np
import pytest

from vsearch.stats import draw_power, paired_ci


def test_paired_ci():
    m, ci = paired_ci(np.array([1.0, 2.0, 3.0]))
    assert m == 2.0 and ci == pytest.approx(1.96 * 1.0 / np.sqrt(3))


def test_draw_power_dominant_config():
    M = np.column_stack([np.full(200, 0.9), np.full(200, 0.1)])
    p = draw_power(M, n=50, draws=100)
    assert p["winner"] == 0 and p["winner_right"] == 1.0 and p["winner_share"] == [1.0, 0.0]
    assert p["order_right"] == 1.0


def test_draw_power_is_seeded():
    M = np.random.default_rng(1).random((300, 3))
    assert draw_power(M, draws=500) == draw_power(M, draws=500)


def test_draw_power_noise_is_not_certain():
    M = np.random.default_rng(2).random((300, 3))  # three configs from the same distribution
    p = draw_power(M, draws=2000)
    assert 0.2 < p["winner_right"] < 0.8 and sum(p["winner_share"]) == pytest.approx(1.0)
