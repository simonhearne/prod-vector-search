"""Merchandising rules and page metrics (B5)."""
import pytest
from pymilvus import FunctionChainStage

from vsearch.rules import apply_app, boost, combined, leak, margin_chain, page_margin, reach, slot_share


def test_apply_app_is_stable_and_boosts():
    hits = [("a", 1.0), ("b", 0.9), ("c", 0.9)]
    assert apply_app(hits, {}) == ["a", "b", "c"]
    assert apply_app(hits, {"c": 1.2}) == ["c", "a", "b"]


def test_page_margin_averages_returned_items_only():
    assert page_margin(["a", "b"], {"a": 0.2, "b": 0.4}) == pytest.approx(0.3)
    assert page_margin([], {}) is None


def test_reach_ignores_ineligible_queries():
    assert reach([["x"], ["c"], ["x"]], {"c"}, [True, True, False]) == 0.5
    assert reach([["c"]], {"c"}, [False]) is None


def test_leak():
    assert leak([["h", "a"], ["a"]], {"h"}) == 0.5


def test_boost_carries_filter_and_weight():
    f = boost("hidden == true", 0.5, name="bury")
    assert f.name == "bury" and f.params == {"reranker": "boost", "filter": "hidden == true", "weight": 0.5}


def test_combined_sets_both_modes():
    fs = combined([boost("a == 1", 1.1, "a"), boost("b == 1", 1.2, "b")], "Sum")
    assert len(fs.functions) == 2 and fs.params == {"boost_mode": "Sum", "function_mode": "Sum"}


def test_margin_chain_is_l0_score_plus_weighted_margin():
    ch = margin_chain(0.2)
    (op,) = ch.ops
    assert ch.stage == FunctionChainStage.L0_RERANK and op.outputs == ("$score",)
    assert [a.name for a in op.expr.args] == ["$score", "margin"]
    assert op.expr.params == {"mode": "weighted", "weights": [1.0, 0.2]}


def test_slot_share_shows_over_boosting_that_reach_hides():
    assert slot_share([["c", "c2"], ["c", "x"]], {"c", "c2"}) == pytest.approx(0.75)
    assert slot_share([[]], {"c"}) is None
