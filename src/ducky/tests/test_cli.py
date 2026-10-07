from typer.testing import CliRunner

from ducky.cli import app

runner = CliRunner()


def invoke(*args, **kw):
    return runner.invoke(app, list(args), **kw)


def test_help_and_version():
    assert invoke("--help").exit_code == 0
    assert "0.1.0" in invoke("--version").stdout


def test_thoughts_creates_and_continues_session():
    r = invoke("thoughts", "--text", "It is a flask app with a redis cache for sessions")
    assert r.exit_code == 0, r.stdout
    assert "Ducky" in r.stdout and "What next?" in r.stdout
    assert "awaiting_system" in r.stdout

    r = invoke("thoughts", "--text", "I am stuck because the cache returns stale data")
    assert r.exit_code == 0
    assert "awaiting_problem" in r.stdout  # same session, advanced phase

    assert invoke("session", "list").stdout.count("*") == 1


def test_new_with_name_and_switch():
    invoke("thoughts", "--new", "--name", "two-sum", "--text", "hello there duck friend ok")
    invoke("thoughts", "--new", "--name", "other", "--text", "hello there duck friend ok")
    r = invoke("thoughts", "--session", "two-sum", "--text", "back again my friends ok")
    assert "two-sum" in r.stdout
    assert invoke("thoughts", "--new", "--session", "x", "--text", "a").exit_code == 1
    assert invoke("thoughts", "--session", "ghost", "--text", "a").exit_code == 1


def test_skip_intro():
    r = invoke("thoughts", "--skip-intro", "--text", "my recursion never ends")
    assert "iterating" in r.stdout


def test_voice_not_built_yet_message():
    # CliRunner stdin is not a tty, so force the voice path directly.
    from ducky.input import InputUnavailable, VoiceInput
    import pytest

    with pytest.raises(InputUnavailable):
        VoiceInput().get_transcript()


def test_history_and_end():
    assert invoke("history").exit_code == 1  # no active session
    invoke("thoughts", "--text", "a flask app with a redis cache here")
    assert "flask app" in invoke("history").stdout
    assert "Ended session" in invoke("end").stdout
    assert "No active session" in invoke("end").stdout


def test_session_rename_and_delete():
    invoke("thoughts", "--new", "--name", "a", "--text", "hello")
    assert invoke("session", "rename", "a", "b").exit_code == 0
    assert invoke("session", "delete", "b", "--yes").exit_code == 0
    assert invoke("session", "delete", "b", "--yes").exit_code == 1
    assert invoke("session", "delete", "b", input="n\n").exit_code == 1  # missing session


def test_delete_confirm_declined():
    invoke("thoughts", "--new", "--name", "keepme", "--text", "hello")
    r = invoke("session", "delete", "keepme", input="n\n")
    assert "Cancelled" in r.stdout
    assert "keepme" in invoke("session", "list").stdout


def test_config_commands():
    assert invoke("config", "set", "answers", "true").exit_code == 0
    assert "answers = True" in invoke("config", "show").stdout
    assert invoke("config", "set", "agent", "llama").exit_code == 1
    assert invoke("config", "set", "bogus", "1").exit_code == 1


def test_phrases_commands():
    assert "have any thoughts ducky" in invoke("phrases", "list").stdout
    assert invoke("phrases", "add", "over to you duck").exit_code == 0
    assert invoke("phrases", "add", "ducky").exit_code == 1
    assert invoke("phrases", "remove", "over to you duck").exit_code == 0
    assert invoke("phrases", "remove", "over to you duck").exit_code == 1
    assert invoke("phrases", "test").exit_code == 1


def test_quack():
    assert "Quack" in invoke("quack").stdout
