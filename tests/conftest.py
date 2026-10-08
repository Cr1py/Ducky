import pytest

from ducky import db


@pytest.fixture(autouse=True)
def ducky_home(tmp_path, monkeypatch):
    """Every test gets an isolated config + data dir, the offline stub LLM, and no real keys."""
    monkeypatch.setenv("DUCKY_HOME", str(tmp_path))
    monkeypatch.setenv("DUCKY_LLM", "stub")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
                "GEMINI_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def conn():
    connection = db.connect()
    yield connection
    connection.close()
