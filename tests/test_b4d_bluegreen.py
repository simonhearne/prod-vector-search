"""B4d blue/green: offline checks. The safety rules matter most: never drop or rename prod_tuned."""
import pytest

from bench import b4d_bluegreen as bg
from bench import b4d_single_ops as b4d


def test_design_has_its_own_controls_first_and_last():
    assert [p.label for p in bg.design()] == ["control", "bluegreen", "bluegreen", "control"]
    assert [p.repeat for p in bg.design()] == [0, 0, 1, 1]
    assert [p.slot for p in bg.design()] == [0, 1, 2, 3]


def test_green_is_a_new_name_and_never_prod_tuned():
    assert bg.GREEN != bg.NAME and bg.GREEN != b4d.NAME
    assert bg.green_spec().name == bg.GREEN
    assert bg.green_spec().partition_key and bg.green_spec().scalar_indexes  # same shape as prod_tuned


class _C:
    def __init__(self, existing):
        self.calls, self.existing = [], set(existing)

    def has_collection(self, name):
        return name in self.existing

    def alter_alias(self, collection_name, alias, **kw):
        self.calls.append(("alter_alias", collection_name, alias))

    def drop_collection(self, name):
        self.calls.append(("drop", name))
        self.existing.discard(name)

    def rename_collection(self, *a, **k):
        raise AssertionError("rename_collection called")


def test_cleanup_points_the_alias_back_before_dropping_green_only():
    c = _C({bg.NAME, bg.GREEN})
    bg.cleanup(c)
    assert c.calls[0] == ("alter_alias", bg.NAME, bg.ALIAS)
    assert ("drop", bg.GREEN) in c.calls and ("drop", bg.NAME) not in c.calls
    assert bg.NAME in c.existing


def test_cleanup_is_safe_when_green_was_never_built():
    c = _C({bg.NAME})
    bg.cleanup(c)
    assert ("drop", bg.GREEN) not in c.calls and c.calls[0][0] == "alter_alias"


GIB = 2 ** 30


def test_cap_is_the_container_limit_in_gib():
    assert bg.CAP_BYTES == 12 * GIB


def test_fit_verdict_uses_milvus_own_90_percent_line():
    ok, msg = bg.fit_verdict(10.5 * GIB, cap_bytes=12 * GIB)
    assert ok is True
    ok, msg = bg.fit_verdict(11.0 * GIB, cap_bytes=12 * GIB)
    assert ok is False and "12.0 GiB" in msg and "10.8 GiB" in msg


def test_rss_reads_an_unlabelled_gauge():
    text = "# HELP x\nprocess_resident_memory_bytes 7.651201024e+09\nprocess_resident_memory_bytes_x{a=\"b\"} 1\n"
    assert bg.rss_bytes(text) == 7.651201024e+09


def test_rss_missing_is_an_error_not_zero():
    with pytest.raises(KeyError):
        bg.rss_bytes("other_metric 1\n")


class _Watch:
    def __init__(self, peak):
        self.peak, self.tripped, self.started, self.stopped = peak, False, False, False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def _fit(monkeypatch, build, rss_seq, peak=9 * GIB, tripped=False):
    c = _C({bg.NAME})
    w = _Watch(peak)
    w.tripped = tripped
    it = iter(rss_seq)
    monkeypatch.setattr(bg, "build_collection", build)
    monkeypatch.setattr(bg, "read_rss", lambda: next(it))
    monkeypatch.setattr(bg, "MemWatch", lambda *a, **k: w)
    monkeypatch.setattr(bg.time, "sleep", lambda s: None)
    return c, w, bg.fit_check(c, None, None, None, None, settle_s=0, samples=3)


def test_fit_check_measures_both_loaded_then_drops_green_only(monkeypatch):
    built = []
    c, w, r = _fit(monkeypatch, lambda cl, spec, *a: (built.append(spec.name), cl.existing.add(spec.name)),
                   [7 * GIB, 10 * GIB, 10.2 * GIB, 10.1 * GIB])
    assert built == [bg.GREEN]
    assert r["rss_one_gib"] == 7.0 and r["rss_both_gib"] == pytest.approx(10.1)
    assert r["fits"] is True and r["error"] is None
    assert w.started and w.stopped
    assert ("drop", bg.GREEN) in c.calls and bg.NAME in c.existing


def test_fit_check_drops_green_when_the_build_or_load_fails(monkeypatch):
    def boom(cl, spec, *a):
        cl.existing.add(spec.name)
        raise RuntimeError("memory not enough")
    c, w, r = _fit(monkeypatch, boom, [7 * GIB])
    assert r["fits"] is False and "memory not enough" in r["error"]
    assert ("drop", bg.GREEN) in c.calls and w.stopped


def test_fit_check_fails_when_the_peak_crossed_the_line(monkeypatch):
    c, w, r = _fit(monkeypatch, lambda cl, spec, *a: cl.existing.add(spec.name),
                   [7 * GIB, 10 * GIB, 10 * GIB, 10 * GIB], peak=11.5 * GIB)
    assert r["fits"] is False and r["peak_gib"] == 11.5


def test_memwatch_trips_and_records_the_peak():
    seq = iter([5 * GIB, 9 * GIB, 11.5 * GIB] + [11.5 * GIB] * 100)
    tripped = []
    w = bg.MemWatch(read=lambda: next(seq), limit_bytes=11 * GIB, on_trip=lambda: tripped.append(1), poll_s=0.001)
    w.start()
    import time as _t
    for _ in range(200):
        if w.tripped:
            break
        _t.sleep(0.005)
    w.stop()
    assert w.tripped and w.peak == 11.5 * GIB and tripped == [1]


def test_phases_include_build_and_switch_windows():
    tl = {"runin_s": 300.0, "build_start": 300.0, "build_end": 900.0, "switch_start": 900.0, "switch_return": 900.2}
    ph = bg.phases(tl, runoff_s=1800)
    assert ph["runin"] == [0.0, 300.0] and ph["build"] == [300.0, 900.0]
    assert ph["switch"] == [900.0, 900.2] and ph["post_0_5m"] == [900.2, 1200.2]
    assert ph["post_15_30m"] == [1800.2, 2700.2]


def test_no_em_dash():
    import inspect
    assert "—" not in inspect.getsource(bg)


def test_switch_window_counts_errors_either_side():
    from vsearch.load import Sample
    s = [Sample(898.0, 10, True, None), Sample(900.1, 500, False, "e"), Sample(905.0, 10, True, None)]
    w = bg.around_switch(s, 900.0, half_width=10)
    assert w["n"] == 3 and w["errors"] == 1 and w["max_ms"] == 500


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_build_and_switch_builds_green_then_points_the_alias_at_it(monkeypatch):
    c = _C({bg.NAME})
    built = []
    monkeypatch.setattr(bg, "build_collection", lambda cl, spec, *a: (built.append(spec.name), cl.existing.add(spec.name)))
    f = bg.build_and_switch(c, None, None, None, None, clock=_Clock())
    assert built == [bg.GREEN]
    assert c.calls == [("alter_alias", bg.GREEN, bg.ALIAS)]
    assert f["build_start"] < f["build_end"] <= f["switch_start"] < f["switch_return"] == f["runoff_start"]
    assert bg.NAME in c.existing  # blue stays: it is dropped by nothing, and released by nothing here


def test_control_waits_as_long_as_a_build_and_switches_nothing():
    c = _C({bg.NAME})
    waited = []
    f = bg.control_wait(c, wait=lambda s: waited.append(s), build_s=630.0, clock=_Clock())
    assert waited == [630.0] and c.calls == []
    assert f["build_start"] < f["build_end"] == f["switch_start"] == f["switch_return"] == f["runoff_start"]


def test_alias_must_start_on_blue():
    class A(_C):
        def describe_alias(self, alias):
            return {"collection_name": self.target}
    c = A({bg.NAME, bg.GREEN}); c.target = bg.GREEN
    with pytest.raises(RuntimeError, match="alias"):
        bg.check_alias(c)
    c.target = bg.NAME
    bg.check_alias(c)


def test_search_goes_through_the_alias():
    calls = []

    class S:
        def search(self, name, **kw):
            calls.append(name)
    bg.alias_search(S(), [[0.0]], 256)(0)
    assert calls == [bg.ALIAS]
