from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ducky.cli import app
from ducky.errors import EmptyTranscript, InputUnavailable

runner = CliRunner()


def flat(result):
    """Output with line-wrapping flattened, so assertions don't depend on the terminal width."""
    return " ".join(result.output.split())


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
    invoke("config", "set", "agent", "claude")  # a model that needs a key (the default is keyless Ollama)
    r = invoke("thoughts", "--text", "my loop never ends")
    assert r.exit_code == 1
    assert "ANTHROPIC_API_KEY" in r.output and "saved" in r.output
    assert "my loop never ends" in invoke("history").stdout


def test_config_keys_and_set_key():
    invoke("config", "set", "agent", "claude")
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
    for name in ("ollama", "claude", "chatgpt", "gemini"):
        assert name in r.stdout
    assert "models.toml" in r.stdout


def test_config_agent_must_exist_in_registry():
    assert invoke("config", "set", "agent", "gemini").exit_code == 0
    assert "GEMINI_API_KEY" in invoke("config", "keys").stdout
    for removed in ("grok", "qwen", "deepseek"):
        assert invoke("config", "set", "agent", removed).exit_code == 1
    assert invoke("config", "set", "agent", "openai").exit_code == 1


def test_keyless_local_model_end_to_end(ducky_home):
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[models.local]\nprovider = "openai_compat"\nbase_url = "http://localhost:11434/v1"\nmodel = "m"\n')
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


# --- voice -----------------------------------------------------------------------

class FakeInput:
    """Stands in for VoiceInput: returns text, or raises what the real one would."""

    def __init__(self, result):
        self.result = result

    def get_transcript(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeMic:
    fail = None

    def __init__(self, **kw):
        pass

    def check(self):
        if FakeMic.fail:
            raise FakeMic.fail


@pytest.fixture
def voice(monkeypatch):
    """Make `thoughts` take the voice path with a fake mic and a fake recognizer."""
    FakeMic.fail = None
    monkeypatch.setattr("ducky.audio.MicSource", FakeMic)
    monkeypatch.setattr("ducky.cli._use_text", lambda text: False)
    state = SimpleNamespace(result="my loop never ends", built=[])

    def fake_build(cfg, phrases, **kw):
        state.built.append((phrases, kw))
        return FakeInput(state.result)

    monkeypatch.setattr("ducky.cli.build_voice_input", fake_build)
    return state


def install_model(name="vosk-model-small-en-us-0.15"):
    from ducky.stt.models import models_dir

    (models_dir() / name / "am").mkdir(parents=True, exist_ok=True)


def test_thoughts_by_voice_records_the_transcript(voice):
    install_model()
    r = invoke("thoughts")
    assert r.exit_code == 0, r.output
    assert "Ducky" in r.stdout
    assert "my loop never ends" in invoke("history").stdout
    phrases, kw = voice.built[0]
    assert phrases == ["have any thoughts ducky"] and kw["silence_seconds"] == 6.0


def test_voice_uses_the_configured_silence_and_phrases(voice):
    install_model()
    invoke("config", "set", "silence", "9")
    invoke("phrases", "add", "over to you duck")
    invoke("thoughts")
    phrases, kw = voice.built[0]
    assert "over to you duck" in phrases and kw["silence_seconds"] == 9.0


@pytest.mark.parametrize("error,fragment", [
    (EmptyTranscript("I didn't catch anything."), "didn't catch anything"),
    (InputUnavailable("No microphone found (x)."), "No microphone found"),
])
def test_voice_failures_are_reported_cleanly(voice, error, fragment):
    install_model()
    voice.result = error
    r = invoke("thoughts")
    assert r.exit_code == 1 and fragment in r.output


def test_missing_microphone_is_reported_before_any_download(voice, monkeypatch):
    FakeMic.fail = InputUnavailable("No microphone found (x). Run `ducky mic list`")
    downloads = []
    monkeypatch.setattr("ducky.cli._download_with_progress", lambda name: downloads.append(name))
    r = invoke("thoughts", input="y\n")
    assert r.exit_code == 1 and "No microphone found" in r.output
    assert downloads == [] and "isn't installed" not in r.output   # never even asked


def test_missing_model_declined(voice):
    r = invoke("thoughts", input="n\n")
    assert r.exit_code == 1
    assert "isn't installed yet" in flat(r) and "ducky stt download" in flat(r)
    assert voice.built == []


def test_missing_model_accepted_downloads_then_listens(voice, monkeypatch):
    from ducky.stt.models import models_dir

    def fake_download(name):
        install_model(name)
        return models_dir() / name

    monkeypatch.setattr("ducky.cli._download_with_progress", fake_download)
    r = invoke("thoughts", input="y\n")
    assert r.exit_code == 0, r.output
    assert "my loop never ends" in invoke("history").stdout


def test_failed_download_is_reported(voice, monkeypatch):
    from ducky.stt.models import ModelError

    def boom(name):
        raise ModelError("Couldn't download the model (ConnectError). Check your internet connection.")

    monkeypatch.setattr("ducky.cli._download_with_progress", boom)
    r = invoke("thoughts", input="y\n")
    assert r.exit_code == 1 and "internet connection" in flat(r)


def test_configured_model_path_that_does_not_exist(voice):
    invoke("config", "set", "vosk_model", "C:\\no\\such\\folder")
    r = invoke("thoughts")
    assert r.exit_code == 1 and "No speech model found" in r.output


def test_a_model_folder_path_is_used_directly(voice, tmp_path):
    folder = tmp_path / "custom-model"
    (folder / "am").mkdir(parents=True)
    invoke("config", "set", "vosk_model", str(folder))
    assert invoke("thoughts").exit_code == 0 and "isn't installed" not in invoke("thoughts").stdout


# --- phrases test ----------------------------------------------------------------

def test_phrases_test_recognizes_a_working_phrase(voice):
    install_model()
    voice.result = "have any thoughts ducky"
    r = invoke("phrases", "test")
    assert r.exit_code == 0 and "That works" in r.stdout
    phrases, kw = voice.built[0]
    assert phrases == [] and kw["silence_seconds"] == 3.0    # the test must not end on a phrase


def test_phrases_test_offers_to_save_a_misheard_phrase_as_an_alias(voice):
    install_model()
    voice.result = "have any thoughts ducking"
    r = invoke("phrases", "test", input="y\n")
    assert "doesn't match" in r.stdout and "Saved: have any thoughts ducking" in r.stdout
    assert "have any thoughts ducking  (alias)" in invoke("phrases", "list").stdout


def test_phrases_test_can_be_declined(voice):
    install_model()
    voice.result = "have any thoughts ducking"
    invoke("phrases", "test", input="n\n")
    assert "ducking" not in invoke("phrases", "list").stdout


def test_phrases_test_rejects_a_single_word(voice):
    install_model()
    voice.result = "duck"
    r = invoke("phrases", "test")
    assert r.exit_code == 0 and "too short" in r.stdout


@pytest.mark.parametrize("error", [EmptyTranscript("I didn't hear anything."),
                                   InputUnavailable("No microphone found (x).")])
def test_phrases_test_failures(voice, error):
    install_model()
    voice.result = error
    assert invoke("phrases", "test").exit_code == 1


# --- mic / stt ---------------------------------------------------------------------

def test_mic_list(monkeypatch):
    rows = [{"index": 1, "name": "Built-in Mic", "channels": 2, "default": True},
            {"index": 4, "name": "USB Headset", "channels": 1, "default": False}]
    monkeypatch.setattr("ducky.audio.list_input_devices", lambda *a, **k: rows)
    r = invoke("mic", "list")
    assert r.exit_code == 0 and "Built-in Mic" in r.stdout and "USB Headset" in r.stdout
    assert "config set mic" in r.stdout


def test_mic_list_with_no_devices_or_a_broken_audio_stack(monkeypatch):
    monkeypatch.setattr("ducky.audio.list_input_devices", lambda *a, **k: [])
    assert invoke("mic", "list").exit_code == 1

    def boom(*a, **k):
        raise InputUnavailable("PortAudio couldn't be loaded (x).")

    monkeypatch.setattr("ducky.audio.list_input_devices", boom)
    r = invoke("mic", "list")
    assert r.exit_code == 1 and "PortAudio" in r.output


def test_stt_status_before_and_after_install():
    assert "not installed" in invoke("stt", "status").stdout
    install_model()
    r = invoke("stt", "status")
    assert "installed at" in r.stdout and "vosk-model-small-en-us-0.15" in r.stdout


def test_stt_download_flows(monkeypatch):
    from ducky.stt.models import ModelError, models_dir

    calls = []

    def fake_download(name):
        calls.append(name)
        install_model(name)
        return models_dir() / name

    monkeypatch.setattr("ducky.cli._download_with_progress", fake_download)
    assert invoke("stt", "download").exit_code == 0 and calls == ["vosk-model-small-en-us-0.15"]
    assert "already installed" in invoke("stt", "download").stdout and len(calls) == 1

    r = invoke("stt", "download", "vosk-model-en-us-0.22-lgraph")
    assert "config set vosk_model vosk-model-en-us-0.22-lgraph" in r.stdout   # tells you how to use it

    monkeypatch.setattr("ducky.cli._download_with_progress",
                        lambda name: (_ for _ in ()).throw(ModelError("No Vosk model named 'x'.")))
    assert invoke("stt", "download", "x").exit_code == 1
    assert invoke("stt", "download", "../etc").exit_code == 1    # not a name and not a folder
