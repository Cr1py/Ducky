import stat
import sys

import pytest

from ducky import keys
CLAUDE = "ANTHROPIC_API_KEY"
OPENAI = "OPENAI_API_KEY"


def test_missing():
    assert keys.get_key(CLAUDE) == (None, "missing")


def test_env_beats_file(monkeypatch):
    keys.save_key(CLAUDE, "from-file")
    assert keys.get_key(CLAUDE) == ("from-file", "file")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert keys.get_key(CLAUDE) == ("from-env", "env")


def test_file_parsing(ducky_home):
    path = keys.keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# comment\n\nexport OPENAI_API_KEY = "abc"\nXAI_API_KEY=xyz\nbad line\n')
    parsed = keys.read_keys_file()
    assert parsed == {"OPENAI_API_KEY": "abc", "XAI_API_KEY": "xyz"}


def test_save_replaces_existing_line():
    keys.save_key(CLAUDE, "one")
    keys.save_key(OPENAI, "other")
    keys.save_key(CLAUDE, "two")
    parsed = keys.read_keys_file()
    assert parsed == {"ANTHROPIC_API_KEY": "two", "OPENAI_API_KEY": "other"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_file_is_owner_only():
    path = keys.save_key(CLAUDE, "k")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_cwd_dotenv_is_ignored(tmp_path, monkeypatch):
    work = tmp_path / "project"
    work.mkdir()
    (work / ".env").write_text("ANTHROPIC_API_KEY=leaked\n")
    monkeypatch.chdir(work)
    assert keys.get_key(CLAUDE) == (None, "missing")
