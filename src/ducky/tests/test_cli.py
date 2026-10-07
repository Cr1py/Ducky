from typer.testing import CliRunner

from ducky.cli import app

runner = CliRunner()


def invoke(*args, **kw):
    return runner.invoke(app, list(args), **kw)


def test_help_and_version():
    assert invoke("--help").exit_code == 0
    assert "0.1.0" in invoke("--version").stdout


def test_thoughts_creates_and_continues_session():
    r = invoke(
        "thoughts", "--text", "It is a flask app with a redis cache for sessions"
    )
    assert r.exit_code == 0, r.stdout
    assert "Ducky" in r.stdout and "What next?" in r.stdout
    assert "awaiting_system" in r.stdout

    r = invoke("thoughts", "--text", "I am stuck because the cache returns stale data")
    assert r.exit_code == 0
    assert "awaiting_problem" in r.stdout  # same session, advanced phase

    assert invoke("session", "list").stdout.count("*") == 1


def test_new_with_name_and_switch():
    invoke(
        "thoughts", "--new", "--name", "two-sum", "--text", "hello there duck friend ok"
    )
    invoke(
        "thoughts", "--new", "--name", "other", "--text", "hello there duck friend ok"
    )
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
    assert (
        invoke("session", "delete", "b", input="n\n").exit_code == 1
    )  # missing session


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


def test_summary_appears_in_history_after_many_long_turns():
    invoke("config", "set", "max_context_tokens", "500")
    for i in range(5):
        r = invoke("thoughts", "--text", f"turn {i} flask redis cache " + "word " * 120)
        assert r.exit_code == 0, r.stdout
    assert "older turns" in r.stdout or "summary" in invoke("history").stdout.lower()
    assert "[stub summary]" in invoke("history").stdout


def test_end_refreshes_summary_for_longer_sessions():
    for i in range(7):
        invoke("thoughts", "--text", f"turn {i} flask redis cache short")
    r = invoke("end")
    assert "Ended session" in r.stdout and "Folded" in r.stdout


def test_missing_key_is_friendly_and_input_is_saved(monkeypatch):
    monkeypatch.delenv("DUCKY_LLM")
    r = invoke("thoughts", "--text", "my loop never ends")
    assert r.exit_code == 1
    assert "ANTHROPIC_API_KEY" in r.output and "saved" in r.output
    assert "my loop never ends" in invoke("history").stdout


def test_config_keys_and_set_key():
    r = invoke("config", "keys")
    assert "ANTHROPIC_API_KEY" in r.stdout and "missing" in r.stdout
    assert invoke("config", "set-key", "claude", input="sk-abc\n").exit_code == 0
    r = invoke("config", "keys")
    assert "file" in r.stdout and "sk-abc" not in r.stdout
    assert invoke("config", "set-key", "llama", input="x\n").exit_code == 1
    assert "claude-haiku" in invoke("config", "show").stdout


def test_config_models_lists_registry_and_path():
    r = invoke("config", "models")
    assert r.exit_code == 0
    for name in ("claude", "chatgpt", "gemini", "ollama"):
        assert name in r.stdout
    assert "models.toml" in r.stdout


def test_config_agent_must_exist_in_registry():
    assert invoke("config", "set", "agent", "claude").exit_code == 0
    assert "ANTHROPIC_API_KEY" in invoke("config", "keys").stdout
    assert invoke("config", "set", "agent", "openai").exit_code == 1


def test_keyless_local_model_end_to_end(ducky_home):
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '[models.local]\nprovider = "openai_compat"\nbase_url = "http://localhost:11434/v1"\nmodel = "m"\n'
    )
    assert invoke("config", "set", "agent", "local").exit_code == 0
    assert "not needed" in invoke("config", "show").stdout
    assert "needs no API key" in invoke("config", "set-key", "local").stdout


def test_broken_models_file_reports_cleanly(ducky_home):
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[models.claude]\nbogus = 1\n")
    r = invoke("config", "models")
    assert r.exit_code == 1 and "bogus" in r.output


def test_ollama_is_selectable_and_keyless():
    assert invoke("config", "set", "agent", "ollama").exit_code == 0
    shown = invoke("config", "show").stdout
    assert "ollama: gemma4:e2b" in shown and "not needed" in shown
    assert "needs no API key" in invoke("config", "set-key", "ollama").stdout
