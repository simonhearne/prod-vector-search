from vsearch import config


def test_constants():
    assert config.SEED == 27 and config.K == 10
    assert config.GAINS == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
    assert config.POC_CATEGORY == "Electronics" and config.POC_ROWS == 20_000


def test_export_dotenv_sets_missing_only(tmp_path, monkeypatch):
    from vsearch.config import export_dotenv
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-test\nHF_TOKEN=hf\n# x=y\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "already")
    export_dotenv(tmp_path / ".env", ["ANTHROPIC_API_KEY", "HF_TOKEN"])
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test" and os.environ["HF_TOKEN"] == "already"
