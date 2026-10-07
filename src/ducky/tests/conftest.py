import pytest

from ducky import db


@pytest.fixture(autouse=True)
def ducky_home(tmp_path, monkeypatch):
    """Every test gets an isolated config + data directory."""
    monkeypatch.setenv("DUCKY_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def conn():
    connection = db.connect()
    yield connection
    connection.close()
