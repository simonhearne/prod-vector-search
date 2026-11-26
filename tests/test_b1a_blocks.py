import pandas as pd
from pytest import approx

from bench import b1a_eval_sets as b1a
from vsearch import judge


def test_wording_effect_sign(monkeypatch):
    # dense_image finds the Exact product only for the original wording; text finds it for both
    monkeypatch.setattr(judge, "cached_judgments", lambda: pd.DataFrame(columns=["query_id", "product_id", "label", "model"]))
    qr = pd.DataFrame({"query_id": ["q1", "q2"], "product_id": ["e1", "e2"], "label": ["E", "E"]})
    hit = {"q1": ["e1"], "q2": ["e2"]}
    miss = {"q1": ["x"], "q2": ["x"]}
    orig = {"dense_text": hit, "dense_image": hit, "hybrid_rrf": hit}
    para = {"dense_text": hit, "dense_image": miss, "hybrid_rrf": hit}
    w = b1a.wording_block(orig, para, qr)["filled"]
    assert w["wording_effect"]["dense_image"][0] == approx(-1.0)
    assert w["wording_effect"]["dense_text"][0] == approx(0.0)
    assert w["interaction"]["dense_image-dense_text"][0] == approx(-1.0)
