"""Config: a TOML file in the user config dir, validated on write.

Set DUCKY_HOME to relocate everything (used by tests).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from platformdirs import user_config_dir, user_data_dir

APP_NAME = "ducky"

DEFAULTS: dict[str, Any] = {
    "answers": False,  # allow direct answers (hint level 4)
    "agent": "ollama",  # a name from models.toml
    "silence": 6.0,  # seconds of silence before Ducky responds
    "tone": "default",  # future implementation: fun response styles
    "stt": "vosk",  # speech-to-text engine
    "max_context_tokens": 6000,  # prune threshold
}
TONES = {"default"}
STT_ENGINES = {"vosk"}


class ConfigError(ValueError):
    pass


def config_dir() -> Path:
    home = os.environ.get("DUCKY_HOME")
    return Path(home) / "config" if home else Path(user_config_dir(APP_NAME))


def data_dir() -> Path:
    home = os.environ.get("DUCKY_HOME")
    return Path(home) / "data" if home else Path(user_data_dir(APP_NAME))


def config_path() -> Path:
    return config_dir() / "config.toml"


def db_path() -> Path:
    return data_dir() / "ducky.db"


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    path = config_path()
    if path.exists():
        with path.open("rb") as f:
            stored = tomllib.load(f)
        cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps({k: cfg[k] for k in DEFAULTS}), encoding="utf-8")


def _to_bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in {"true", "yes", "on", "1"}:
        return True
    if v in {"false", "no", "off", "0"}:
        return False
    raise ConfigError("Expected true or false.")


def _choice(raw: str, allowed: set[str], label: str) -> str:
    v = raw.strip().lower()
    if v not in allowed:
        raise ConfigError(
            f"Unknown {label} '{raw}'. Options: {', '.join(sorted(allowed))}"
        )
    return v


def coerce(key: str, raw: str) -> Any:
    if key not in DEFAULTS:
        raise ConfigError(f"Unknown setting '{key}'. Options: {', '.join(DEFAULTS)}")
    if key == "answers":
        return _to_bool(raw)
    if key == "agent":
        from ducky import registry  # lazy: registry imports this module

        return _choice(raw, set(registry.load_registry()), "model")
    if key == "silence":
        try:
            value = float(raw)
        except ValueError:
            raise ConfigError("silence must be a number of seconds.") from None
        if value <= 0:
            raise ConfigError("silence must be greater than 0.")
        return value
    if key == "tone":
        return _choice(raw, TONES, "tone")
    if key == "stt":
        return _choice(raw, STT_ENGINES, "stt engine")
    if key == "max_context_tokens":
        try:
            value = int(raw)
        except ValueError:
            raise ConfigError("max_context_tokens must be an integer.") from None
        if value < 500:
            raise ConfigError("max_context_tokens must be at least 500.")
        return value
    raise ConfigError(f"Unhandled setting '{key}'.")  # pragma: no cover


def set_value(key: str, raw: str) -> Any:
    value = coerce(key, raw)
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)
    return value
