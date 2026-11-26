import pandas as pd


def test_b1c_queries_fall_back_to_published_dataset(tmp_path, monkeypatch):
    from bench import _common, b1c_judge
    monkeypatch.setattr(_common, "DATA_WORK", tmp_path / "work")
    monkeypatch.setattr(_common, "DATA_DATASET", tmp_path / "dataset")
    monkeypatch.setattr(b1c_judge, "DATA_WORK", tmp_path / "work")
    (tmp_path / "dataset").mkdir()
    pd.DataFrame({"query_id": ["1"], "query": ["red shoes"], "hard": [False]}).to_parquet(
        tmp_path / "dataset" / "queries.parquet", index=False)
    assert b1c_judge._queries_text().to_dict("records") == [{"query_id": "1", "query": "red shoes"}]
