"""B4b isolated: offline checks for the pieces that turn raw logs into the result."""
import numpy as np
import pytest

from bench import b4b_isolated as bi


def test_v2_is_a_new_name_with_sq8_and_never_prod_tuned():
    assert bi.V2 != bi.NAME
    spec = bi.v2_spec()
    assert spec.name == bi.V2 and spec.text_index.type == "HNSW_SQ"


def test_ps_cpu_time_parses_minutes_and_hours():
    assert bi.ps_seconds("0:01.50") == pytest.approx(1.5)
    assert bi.ps_seconds("12:03.25") == pytest.approx(723.25)
    assert bi.ps_seconds("1:02:03.00") == pytest.approx(3723.0)


def test_serving_is_v1_only_when_all_ten_distances_match():
    ref = {0: np.array([0.9, 0.8]), 1: np.array([0.7, 0.6])}
    got = [(0, np.array([0.9, 0.8])), (1, np.array([0.7001, 0.6])), (0, None)]
    assert bi.serving(got, ref) == ["v1", "v2", None]


def test_harness_lag_is_send_minus_due():
    # start is recovered as the earliest send minus its offset; query 2 left 50 ms late
    sends = {0: 10.000, 1: 10.050, 2: 10.150}
    lag = bi.harness_lag_ms(sends, qps=20)
    assert lag[0] == pytest.approx(0) and lag[1] == pytest.approx(0) and lag[2] == pytest.approx(50)
