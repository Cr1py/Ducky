import pytest

from ducky import config
from ducky.config import ConfigError


def test_defaults():
    cfg = config.load_config()
    assert cfg["answers"] is False
    assert cfg["silence"] == 6.0


def test_set_and_persist():
    config.set_value("answers", "true")
    config.set_value("silence", "9.5")
    config.set_value("agent", "gemini")
    cfg = config.load_config()
    assert cfg["answers"] is True and cfg["silence"] == 9.5 and cfg["agent"] == "gemini"


def test_agent_with_model():
    assert config.coerce("agent", "openai/gpt-4o") == "openai/gpt-4o"


@pytest.mark.parametrize(
    "key,value",
    [("answers", "maybe"), ("silence", "0"), ("silence", "abc"), ("agent", "llama"),
     ("tone", "pirate"), ("nope", "1"), ("max_context_tokens", "10")],
)
def test_invalid_values(key, value):
    with pytest.raises(ConfigError):
        config.coerce(key, value)
