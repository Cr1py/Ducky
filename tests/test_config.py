import pytest

from ducky import config
from ducky.config import ConfigError


def test_defaults():
    cfg = config.load_config()
    assert cfg["answers"] is False
    assert cfg["silence"] == 6.0
    assert cfg["agent"] == "ollama"  # local and keyless, so a fresh install needs no API key


def test_set_and_persist():
    config.set_value("answers", "true")
    config.set_value("silence", "9.5")
    config.set_value("agent", "gemini")
    cfg = config.load_config()
    assert cfg["answers"] is True and cfg["silence"] == 9.5 and cfg["agent"] == "gemini"


def test_agent_accepts_user_defined_model(ducky_home):
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[models.local]\nprovider = "openai_compat"\nbase_url = "http://localhost:11434/v1"\nmodel = "m"\n')
    assert config.coerce("agent", "LOCAL") == "local"


def test_agent_is_validated_against_registry():
    assert config.coerce("agent", "chatgpt") == "chatgpt"
    with pytest.raises(ConfigError, match="Options: .*claude"):
        config.coerce("agent", "openai")  # the old name; now `chatgpt`


@pytest.mark.parametrize(
    "key,value",
    [("answers", "maybe"), ("silence", "0"), ("silence", "abc"), ("agent", "llama"),
     ("tone", "pirate"), ("nope", "1"), ("max_context_tokens", "10")],
)
def test_invalid_values(key, value):
    with pytest.raises(ConfigError):
        config.coerce(key, value)
