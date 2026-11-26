"""B4d report: everything is re-cut from raw samples, per repeat, against that repeat's own control."""
import pytest

from bench import b4d_report as rep
from bench import b4d_single_ops as b4d
from vsearch.load import Sample


def S(t, ms, ok=True):
    return Sample(t=t, latency_ms=ms, ok=ok, error=None if ok else "x")


def test_best_attempt_prefers_level_but_keeps_the_unlevel_one_visible():
    passes = [{"slot": 1, "attempt": 0, "level_ok": False}, {"slot": 1, "attempt": 1, "level_ok": True},
              {"slot": 2, "attempt": 0, "level_ok": False}]
    assert rep.best_attempt(passes, 1)["attempt"] == 1
    assert rep.best_attempt(passes, 2)["level_ok"] is False  # unlevel, still returned, caller must say so
    assert rep.best_attempt(passes, 3) is None


def test_matching_control_is_the_same_repeats():
    passes = [{"slot": 0, "repeat": 0, "label": "control", "attempt": 0, "level_ok": True},
              {"slot": 13, "repeat": 1, "label": "control", "attempt": 0, "level_ok": True}]
    assert rep.matching_control(passes, 1)["slot"] == 13 and rep.matching_control(passes, 0)["slot"] == 0


def test_phase_stats_come_from_raw_samples_and_count_errors():
    s = [S(1, 10), S(2, 20), S(3, 0, ok=False), S(11, 100)]
    got = rep.cut(s, {"runin": [0, 10], "write": [10, 20], "post_0_5m": [20, 30]})
    assert got["runin"]["p50"] == pytest.approx(15) and got["runin"]["errors"] == 1 and got["runin"]["n"] == 3
    assert got["write"]["p99"] == pytest.approx(100)
    assert got["post_0_5m"]["p99"] is None


def test_control_p99_covers_everything_after_the_run_in():
    s = [S(1, 999), S(301, 10), S(302, 20)]
    assert rep.control_p99(s, {"runin": [0, 300]}) == pytest.approx(19.9)


def test_events_are_increments_relative_to_run_off_start():
    rows = [{"t": 0, "seal_idle": 1, "comp_MixCompaction_done": 5, "qn_L1_Sealed": 14},
            {"t": 5, "seal_idle": 1, "comp_MixCompaction_done": 5, "qn_L1_Sealed": 14},
            {"t": 10, "error": "x"},
            {"t": 15, "seal_idle": 2, "comp_MixCompaction_done": 6, "qn_L1_Sealed": 13}]
    ev = rep.events(rows, t0=10)
    assert ev == [{"t": 5, "key": "comp_MixCompaction_done", "delta": 1}, {"t": 5, "key": "qn_L1_Sealed", "delta": -1},
                  {"t": 5, "key": "seal_idle", "delta": 1}]


def test_spike_windows_list_slow_windows_after_the_write():
    s = [S(300 + i * 0.5, 10) for i in range(120)] + [S(360 + i * 0.5, 100) for i in range(60)]
    got = rep.spikes(s, t0=300, until=390, ref_p99=10)
    assert [w["t"] for w in got] == [60]
    assert got[0]["p99"] == pytest.approx(100)


def test_spike_windows_do_not_hide_total_stalls():
    """A window in which no search succeeded is the worst window there is, not a missing one."""
    s = [S(300 + i * 0.5, 10) for i in range(60)] + [S(330 + i * 0.5, 0, ok=False) for i in range(60)]
    got = rep.spikes(s, t0=300, until=360, ref_p99=10)
    assert [w["t"] for w in got] == [30] and got[0]["p99"] is None


def test_recovery_text_never_hides_not_reached():
    assert rep.recovery_text(None) == "not reached"
    assert rep.recovery_text(0) == "never left" and rep.recovery_text(90) == "90 s"


def test_row_sets_of_promo_pair_must_match():
    passes = [{"slot": 2, "repeat": 0, "label": "promo", "rows_sha": "a"},
              {"slot": 3, "repeat": 0, "label": "promo_flush", "rows_sha": "a"},
              {"slot": 10, "repeat": 1, "label": "promo_flush", "rows_sha": "b"},
              {"slot": 11, "repeat": 1, "label": "promo", "rows_sha": "c"}]
    assert rep.rowset_mismatches(passes) == [1]


def test_no_em_dash_in_module():
    import inspect
    assert "—" not in inspect.getsource(rep) and "—" not in inspect.getsource(b4d)
