import pytest

from bench.b4c_freshness import marker_stats, poll_update


def test_marker_stats_censors_timeouts():
    ev = [{"pid": "a", "t_write": 0.0, "t_visible": 1.0, "missing_s": 0.0},
          {"pid": "b", "t_write": 0.0, "t_visible": None, "missing_s": 900.0}]
    s = marker_stats(ev, timeout_s=900)
    assert s["n"] == 2 and s["timed_out"] == 1 and s["visible_p99"] > 800 and s["missing_share"] == 0.5
    assert s["visible_p50"] == pytest.approx(450.5)


def test_poll_update_visible_missing_and_stale():
    ev = {"a": {"pid": "a", "t_write": 0.0, "t_visible": None, "missing_s": 0.0, "price": 100000.01},
          "b": {"pid": "b", "t_write": 0.0, "t_visible": None, "missing_s": 0.0, "price": 100001.01},
          "c": {"pid": "c", "t_write": 0.0, "t_visible": None, "missing_s": 0.0, "price": 100002.01}}
    rows = [{"product_id": "a", "price": 100000.01}, {"product_id": "c", "price": 5.0}]  # b absent, c stale
    poll_update(ev, rows, now=2.0, dt=0.25)
    assert ev["a"]["t_visible"] == 2.0 and ev["b"]["missing_s"] == 0.25
    assert ev["c"]["t_visible"] is None and ev["c"]["missing_s"] == 0.0


def test_marker_stats_flags_censored_percentiles():
    ev = [{"pid": str(i), "t_write": 0.0, "t_visible": 800.0, "missing_s": 800.0} for i in range(125)]
    ev += [{"pid": f"t{i}", "t_write": 0.0, "t_visible": None, "missing_s": 900.0} for i in range(25)]
    s = marker_stats(ev, timeout_s=900)
    assert s["timed_out"] == 25 and s["visible_p50_censored"] is False and s["visible_p99_censored"] is True
    assert s["missing_p99_censored"] is True


def test_poll_once_stamps_send_time_and_measures_dt(monkeypatch):
    from types import SimpleNamespace
    from bench import b4c_freshness as b4c
    clock = iter([10.0, 10.4])  # send time; the query itself returns later
    monkeypatch.setattr(b4c.time, "perf_counter", lambda: next(clock))
    ev = {"a": {"pid": "a", "t_write": 0.0, "t_visible": None, "missing_s": 0.0, "price": 1.0},
          "b": {"pid": "b", "t_write": 0.0, "t_visible": None, "missing_s": 0.0, "price": 2.0}}
    c = SimpleNamespace(query=lambda name, filter, output_fields, **kw: [{"product_id": "a", "price": 1.0}])
    sent = b4c.poll_once(c, "t", ev, {}, prev_send=9.7)
    assert sent == 10.0 and ev["a"]["t_visible"] == 10.0 and abs(ev["b"]["missing_s"] - 0.3) < 1e-9


def test_storm_timeline_orders_hooks_around_writer_stop():
    from bench.b4c_freshness import storm_timeline
    tl = storm_timeline(300, {"flush_after": 300, "compact_mid": 150, "early": 0})
    assert tl == [(0, "early"), (150, "compact_mid"), (300, "stop"), (300, "flush_after")]
    assert storm_timeline(20, {}) == [(20, "stop")]
