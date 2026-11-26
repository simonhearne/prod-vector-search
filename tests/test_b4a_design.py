from bench.b4a_write_storm import PassSpec, ReadOpts, design, level_flags, phase_windows


def test_design_stacking_symmetric():
    d = design("stacking")
    assert [p.label for p in d] == ["control", "single", "stacked", "stacked_strong",
                                    "stacked_ignore_growing", "stacked", "single", "control"]
    assert d[2].writes == ("partial", "feed", "partial") and d[0].writes == () and d[1].writes == ("partial",)
    assert d[3].read == ReadOpts("Strong") and d[4].read == ReadOpts("Bounded", ignore_growing=True)
    assert d[0].read == ReadOpts()  # default: Bounded, growing segments searched


def test_design_paths_abba():
    assert [p.writes for p in design("paths")] == [("partial",), ("full",), ("full",), ("partial",)]


def test_level_flags():
    assert level_flags([30.0, 44.0, 46.0], ref=30.0) == [True, True, False]


def test_phase_windows_back_to_back():
    ph = phase_windows(base=120, writes=[("partial", 300.0), ("feed", 95.5), ("partial", 300.0)], tail=1200)
    assert ph["baseline"] == [0, 120] and ph["w0_partial"] == [120, 420] and ph["w1_feed"] == [420, 515.5]
    assert ph["w2_partial"] == [515.5, 815.5] and ph["tail_0_5m"] == [815.5, 1115.5]
    assert ph["tail_10_20m"] == [1415.5, 2015.5]
