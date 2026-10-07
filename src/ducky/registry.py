"""Model registry: built-in models.toml merged with the user's own.

Entries are merged field-by-field, so a user file can override just one field
of a built-in entry (e.g. the model ID) or define a new entry in full.
"""

from __future__ import annotations

import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ducky.config import ConfigError, config_dir

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class RegistryError(ConfigError):
    pass


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos like `maxtokens` fail loudly

    provider: Literal["openai_compat", "anthropic", "gemini", "ollama"]
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int = Field(default=4096, gt=0)
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    timeout: float = Field(default=60.0, gt=0)
    # Ollama only (its native API accepts these (the OpenAI-compatible endpoint does not))
    num_ctx: int | None = Field(default=None, gt=0)  # context window, in tokens
    keep_alive: str | int | None = None  # ex: "30m", or 0 to unload at once
    think: bool | str | None = None  # False/"low"/...; omit for the model default

    @model_validator(mode="after")
    def _check_provider_requirements(self) -> "ModelSpec":
        if self.provider == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat entries need a base_url")
        if self.provider in ("anthropic", "gemini") and not self.api_key_env:
            raise ValueError(f"{self.provider} entries need an api_key_env")
        if self.provider != "ollama":
            ollama_only = [
                name
                for name in ("num_ctx", "keep_alive", "think")
                if getattr(self, name) is not None
            ]
            if ollama_only:
                raise ValueError(
                    f"{', '.join(ollama_only)} only apply to provider = \"ollama\""
                )
        return self


def user_models_path() -> Path:
    return config_dir() / "models.toml"


def _parse(text: str, source: str) -> dict[str, dict[str, Any]]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RegistryError(f"{source}: invalid TOML ({e})") from e
    models = data.get("models", {})
    if not isinstance(models, dict) or not all(
        isinstance(v, dict) for v in models.values()
    ):
        raise RegistryError(f"{source}: expected [models.<name>] tables")
    return models


def load_registry() -> dict[str, ModelSpec]:
    builtin = (
        resources.files("ducky").joinpath("models.toml").read_text(encoding="utf-8")
    )
    merged = _parse(builtin, "built-in models.toml")

    user_path = user_models_path()
    user_names: set[str] = set()
    if user_path.exists():
        user = _parse(user_path.read_text(encoding="utf-8"), str(user_path))
        user_names = set(user)
        for name, fields in user.items():
            merged[name] = {**merged.get(name, {}), **fields}

    registry: dict[str, ModelSpec] = {}
    for name, fields in merged.items():
        where = f" (check {user_path})" if name in user_names else ""
        if not NAME_RE.match(name):
            raise RegistryError(
                f"Invalid model name '{name}': use lowercase letters, digits, '-' and '_'.{where}"
            )
        try:
            registry[name] = ModelSpec.model_validate(fields)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'entry'}: {err['msg']}"
                for err in e.errors()
            )
            raise RegistryError(f"Model '{name}' is invalid: {problems}.{where}") from e
    return registry


def get_spec(name: str, registry: dict[str, ModelSpec] | None = None) -> ModelSpec:
    registry = registry if registry is not None else load_registry()
    spec = registry.get(name.strip().lower())
    if spec is None:
        raise RegistryError(f"Unknown model '{name}'. Options: {', '.join(registry)}")
    return spec
