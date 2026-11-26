"""B6 scoring is offline: cached top 10s and held-out ranks in, the payload out. No Milvus here."""
import json

import pandas as pd
import pytest

from bench import b6_perso as b6
from vsearch import judge

N_TUNE, N_TEST = 3, 8
TUNE = [f"t{i}" for i in range(N_TUNE)]
TEST = [f"q{i}" for i in range(N_TEST)]
BRAND_OF = {f"e{j}": f"b{j}" for j in range(6)} | {"h0": "b0", "x": "Unknown"}


@pytest.fixture(autouse=True)
def pool(monkeypatch):
    monkeypatch.setattr(judge, "cached_judgments", lambda: pd.DataFrame(columns=judge.COLUMNS))


@pytest.fixture
def users():
    def rows(split, qids):
        return pd.DataFrame({"user_id": [f"SECRET{split}{i}" for i in range(len(qids))], "item": ["e0"] * len(qids),
                             "cat": "A", "history": [["h0", "x", "x2"] + ["y"] * (i if i < 7 else 9) for i in range(len(qids))],
                             "query_id": qids, "stranger": 0, "peer": 0, "split": split})
    return pd.concat([rows("tune", TUNE), rows("test", TEST)], ignore_index=True)


@pytest.fixture
def qr():
    # even queries have 4 Exact brands (broad), odd ones 1 (specific)
    rows = [(q, f"e{j}", "E") for i, q in enumerate(TEST + TUNE) for j in range(4 if i % 2 == 0 else 1)]
    return pd.DataFrame(rows, columns=["query_id", "product_id", "label"])


def _run(qids, ranks, page=("e0", "n1")):
    ranks = [ranks] * len(qids) if not isinstance(ranks, list) else ranks
    return {"ids": {q: list(page) for q in qids}, "rank": dict(zip(qids, ranks))}


@pytest.fixture
def cache():
    runs = {"tune|none|0": _run(TUNE, 4)}
    for arm, grid in b6.GRID.items():
        for w in grid:
            runs[f"tune|{arm}|{w}"] = _run(TUNE, 5)
    runs["tune|brand_boost|1.1"] = _run(TUNE, 3)
    for arm in ("none", "popularity"):
        runs[f"test|{arm}|own"] = _run(TEST, 2)
    for arm in b6.PROFILE_ARMS:
        for prof in b6.PROFILES:
            runs[f"test|{arm}|{prof}"] = _run(TEST, 2)
    return {"query_ids": {"tune": TUNE, "test": TEST}, "runs": runs, "measured": {"test": {"segments": 2, "milvus_version": "v3.0.3"}},
            "eligible": {a: {p: 1.0 for p in b6.PROFILES} for a in b6.PROFILE_ARMS}}


def _arm(cache, users, qr, name="brand_boost"):
    return b6.score(cache, users, qr, TEST[:2], BRAND_OF)["arms"][name]


def test_counts_true_when_own_beats_all(cache, users, qr):
    cache["runs"]["test|brand_boost|own"] = _run(TEST, 1)
    a = _arm(cache, users, qr)
    assert a["counts"] is True
    assert a["own_minus_none"]["rr"] == [pytest.approx(0.5), pytest.approx(0.0)]


@pytest.mark.parametrize("rival", ["test|none|own", "test|popularity|own", "test|brand_boost|stranger"])
def test_counts_needs_all_three(cache, users, qr, rival):
    cache["runs"]["test|brand_boost|own"] = _run(TEST, 1)
    cache["runs"][rival] = _run(TEST, 1)  # ties one control
    a = _arm(cache, users, qr)
    assert a["counts"] is False
    lost = rival.split("|")[2] if "brand" in rival else rival.split("|")[1]
    assert a["beats"] == {k: k != lost for k in ("none", "popularity", "stranger")}


def test_a_peer_tie_does_not_change_counts_but_is_reported(cache, users, qr):
    cache["runs"]["test|brand_boost|own"] = cache["runs"]["test|brand_boost|peer"] = _run(TEST, 1)
    a = _arm(cache, users, qr)
    assert a["counts"] is True and a["own_minus_peer"]["rr"][0] == 0


def test_weight_chosen_on_tune_only_and_never_zero(cache):
    w = b6.pick_weights(cache)
    assert w["brand_boost"] == 1.1
    assert w["user_blend"] == b6.GRID["user_blend"][0]  # every weight worse than none: smallest, not off
    assert set(w) == set(b6.GRID) and all(v > 0 for v in w.values())
    cache["runs"]["test|brand_boost|own"] = _run(TEST, 1)
    assert b6.pick_weights(cache) == w


def test_best_weight_at_the_end_of_a_grid_is_flagged(cache):
    assert b6.at_grid_edge(cache)["brand_boost"] is False
    cache["runs"]["tune|popularity|2.0"] = _run(TUNE, 1)
    assert b6.at_grid_edge(cache)["popularity"] is True


def test_rank_null_scores_zero(cache, users, qr):
    cache["runs"]["test|brand_boost|own"] = _run(TEST, None)
    a = _arm(cache, users, qr)
    assert a["own"]["mrr"] == 0 and a["own"]["hit@10"] == 0


def test_hit_at_10_counts_rank_10_not_11(cache, users, qr):
    cache["runs"]["test|brand_boost|own"] = _run(TEST, [10] * 4 + [11] * 4)
    assert _arm(cache, users, qr)["own"]["hit@10"] == 0.5


def test_splits_partition_the_test_users(cache, users, qr):
    s = _arm(cache, users, qr)["splits"]
    assert set(s) == {"query", "history"}
    assert sum(c["n"] for c in s["query"].values()) == N_TEST == sum(c["n"] for c in s["history"].values())
    assert s["query"]["broad"]["n"] == 4 and s["query"]["specific"]["n"] == 4
    assert s["history"]["3-4"]["n"] == 2 and s["history"]["5-9"]["n"] == 5 and s["history"]["10+"]["n"] == 1


def test_score_refuses_another_runs_cache(cache, users, qr):
    cache["query_ids"] = {"tune": TUNE, "test": TEST[::-1]}
    with pytest.raises(ValueError, match="query_ids"):
        _arm(cache, users, qr)


def test_bought_items_are_dropped_before_rank_and_page():
    ids = ["h0"] + [f"n{i}" for i in range(9)] + ["e0", "z"]
    e = b6.entry(ids, {"h0"}, "e0")
    assert e["rank"] == 10 and e["ids"] == [f"n{i}" for i in range(9)] + ["e0"]
    assert b6.entry(ids, set(), "e0")["rank"] == 11
    assert b6.entry(ids, set(), "gone")["rank"] is None


def test_users_without_a_signal_get_the_base_ranking():
    base = [["a", "b"], ["c", "d"], ["e", "f"]]
    assert b6.merge_base(base, [1], [["D", "C"]]) == [["a", "b"], ["D", "C"], ["e", "f"]]


def test_ineligible_users_stay_in_the_mean(cache, users, qr):
    cache["runs"]["test|brand_boost|own"] = _run(TEST, [1] + [2] * 7)  # one user helped, seven unchanged
    assert _arm(cache, users, qr)["own_minus_none"]["rr"][0] == pytest.approx(0.5 / 8)


def test_profile_rows_follow_the_profile_and_missing_peers_have_none(users):
    u = users[users.split == "test"].reset_index(drop=True)
    u.loc[3, "peer"] = -1
    assert b6.profile_rows(u, "own")[:3] == [0, 1, 2]
    assert b6.profile_rows(u, "stranger") == [0] * N_TEST
    assert b6.profile_rows(u, "peer")[3] is None


def test_ceiling_counts_brand_in_history(cache, users, qr):
    c = b6.score(cache, users, qr, TEST[:2], BRAND_OF)["ceiling"]
    assert c["brand_in_history"] == 1.0  # h0 and e0 share b0
    assert c["base_top10"] == 1.0 and c["base_top200"] == 1.0


def test_offline_scoring_keeps_where_the_rankings_were_measured(cache, users, qr):
    p = b6.score(cache, users, qr, TEST[:2], BRAND_OF)
    assert p["measured"] == {"test": {"segments": 2, "milvus_version": "v3.0.3"}}


def test_guard_is_scored_on_guard_queries_only(cache, users, qr):
    g = _arm(cache, users, qr)["guard"]
    assert g["n"] == 2 and 0 < g["ndcg@10"] <= 1 and "judged@10" in g and len(g["own_minus_none"]) == 2


def test_payload_and_cache_hold_no_user_id(cache, users, qr):
    blob = json.dumps(b6.score(cache, users, qr, TEST[:2], BRAND_OF), default=float) + json.dumps(cache)
    assert "SECRET" not in blob


def test_sample_paths_differ_from_full_run():
    assert b6.rankings_path(True) != b6.rankings_path(False)
    assert b6.pool_path(True) != b6.pool_path(False) and b6.pool_path(False) != judge.CACHE


def test_fill_keys_are_the_own_runs_only(cache):
    assert set(b6.fill_keys()) == {"test|none|own", "test|popularity|own"} | {f"test|{a}|own" for a in b6.PROFILE_ARMS}


def test_check_cache_accepts_its_own_run(cache):
    cache["weights"] = b6.pick_weights(cache)
    b6.check_cache(cache, {"tune": TUNE, "test": TEST})


def test_check_cache_refuses_a_shorter_run(cache):
    cache["weights"] = b6.pick_weights(cache)
    with pytest.raises(ValueError, match="query_ids"):
        b6.check_cache(cache, {"tune": TUNE, "test": TEST + ["q99"]})


def test_check_cache_refuses_test_runs_made_at_other_weights(cache):
    cache["weights"] = {**b6.pick_weights(cache), "brand_boost": 1.5}
    with pytest.raises(ValueError, match="weights"):
        b6.check_cache(cache, {"tune": TUNE, "test": TEST})


@pytest.mark.parametrize("broken", [None, {"query_ids": {"tune": TUNE, "test": TEST}, "runs": {}}])
def test_check_cache_refuses_a_missing_or_untested_run(broken):
    with pytest.raises(ValueError):
        b6.check_cache(broken, {"tune": TUNE, "test": TEST})


def test_limits_say_the_stranger_barely_acts_on_the_brand_arms():
    assert any("stranger" in x and "moved@10" in x for x in b6.LIMITS)
