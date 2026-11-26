"""Held-out purchases from review histories (B6). Synthetic users only."""
import numpy as np
import pandas as pd
import pytest

from data.histories import assign_controls, assign_queries, build_interactions, held_out

QRELS = pd.DataFrame([("q1", "i1", "E"), ("q2", "i1", "E"), ("q2", "i2", "E"), ("q3", "i3", "E"), ("q4", "i3", "S"),
                      ("q5", "i4", "E"), ("q6", "i5", "E")], columns=["query_id", "product_id", "label"])


def _raw(rows):
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])


def _users(n=40, seed=0):
    """n users: three history items each, held-out items spread over i1..i5 and two categories."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"user_id": [f"u{i:02d}" for i in range(n)], "item": [f"i{i % 5 + 1}" for i in range(n)],
                         "cat": [f"c{i % 2}" for i in range(n)],
                         "history": [[f"h{j}" for j in rng.choice(50, 3, replace=False)] + [f"i{(i + 1) % 5 + 1}"]
                                     for i in range(n)]})


def test_interactions_keep_first_purchase_of_covered_products():
    x = build_interactions([("A", _raw([("u1", "p1", 9), ("u1", "p1", 1), ("u1", "zz", 2)])),
                            ("B", _raw([("u1", "p2", 5)]))], {"p1", "p2"})
    assert x[["user_id", "product_id", "timestamp", "cat"]].values.tolist() == [["u1", "p1", 1, "A"], ["u1", "p2", 5, "B"]]


def test_held_out_is_last_and_history_is_strictly_earlier():
    x = build_interactions([("A", _raw([("u1", "a", 1), ("u1", "b", 2), ("u1", "c", 3), ("u1", "d", 4), ("u1", "e", 4),
                                        ("u2", "a", 1), ("u2", "b", 2), ("u2", "c", 3)]))], set("abcde"))
    h = held_out(x)
    assert h.user_id.tolist() == ["u1"]
    assert h.item[0] == "e" and h.history[0] == ["a", "b", "c"] and h["cat"][0] == "A"


def test_queries_are_unique_unprotected_exact_and_have_vectors():
    vector_ids, protected = {"q1", "q2", "q3", "q4", "q5"}, {"q5"}
    out = assign_queries(_users(), QRELS, vector_ids, protected)
    assert len(out) == 3  # q1 and q2 and q3: q4 is not Exact, q5 is protected, q6 has no vector
    assert out.query_id.is_unique and set(out.query_id) == {"q1", "q2", "q3"}
    exact = set(zip(QRELS[QRELS.label == "E"].query_id, QRELS[QRELS.label == "E"].product_id))
    assert all((r.query_id, r.item) in exact for r in out.itertuples())


def test_one_query_wanted_by_two_users_goes_to_one_of_them():
    u = pd.DataFrame({"user_id": ["a", "b"], "item": ["i3", "i3"], "cat": ["c", "c"], "history": [["h"], ["h"]]})
    assert len(assign_queries(u, QRELS, {"q3"}, set())) == 1


def test_controls_never_self_and_never_know_the_answer():
    out = assign_controls(_users())
    assert (out.stranger >= 0).all() and (out.peer >= 0).all()
    for i, r in enumerate(out.itertuples()):
        for j in (r.stranger, r.peer):
            assert j != i and r.item not in out.history[j]
        assert out["cat"][r.peer] == r.cat


def test_no_peer_in_the_category_is_minus_one():
    u = _users(6)
    u.loc[0, "cat"] = "alone"
    assert assign_controls(u).peer[0] == -1


def test_same_seed_same_users_whatever_the_row_order():
    def run(u):
        return assign_controls(assign_queries(u, QRELS, {"q1", "q2", "q3", "q5", "q6"}, set()))
    u = _users()
    pd.testing.assert_frame_equal(run(u), run(u.sample(frac=1, random_state=1)))


def test_interactions_and_held_out_ignore_row_order():
    raw = _raw([(f"u{i % 5}", f"p{i}", i // 2) for i in range(40)])
    ids = set(raw.parent_asin)
    a = held_out(build_interactions([("A", raw)], ids))
    b = held_out(build_interactions([("A", raw.sample(frac=1, random_state=3))], ids))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("seed", [1, 2])
def test_seed_changes_the_controls(seed):
    u = _users()
    assert assign_controls(u, seed).stranger.tolist() != assign_controls(u, seed + 10).stranger.tolist()


def test_inputs_fall_back_to_the_published_dataset(tmp_path, monkeypatch):
    """A fresh clone has data/dataset (data.fetch) and no data/work."""
    import data.histories as dh
    from bench import _common
    work, ds = tmp_path / "work", tmp_path / "dataset"
    ds.mkdir()
    monkeypatch.setattr(_common, "DATA_WORK", work)
    monkeypatch.setattr(_common, "DATA_DATASET", ds)
    monkeypatch.setattr(dh, "DATA_WORK", work)
    monkeypatch.setattr(dh, "N_TUNE", 1)
    monkeypatch.setattr(dh, "N_TEST", 1)
    pd.DataFrame({"product_id": ["a", "b", "c", "i1", "i2"]}).to_parquet(ds / "products.parquet")
    pd.DataFrame({"query_id": ["q1", "q2"], "in_eval5k": [False, False]}).to_parquet(ds / "queries.parquet")
    pd.DataFrame([("q1", "i1", "E"), ("q2", "i2", "E")], columns=["query_id", "product_id", "label"]).to_parquet(ds / "qrels.parquet")
    np.save(ds / "q_ids.npy", np.array(["q1", "q2"]))
    rows = [(u, p, t) for u, last in (("u1", "i1"), ("u2", "i2")) for t, p in enumerate(["a", "b", "c", last])]
    monkeypatch.setattr(dh, "_frames", lambda: [("A", _raw(rows))])
    assert set(dh.build_users().query_id) == {"q1", "q2"}
