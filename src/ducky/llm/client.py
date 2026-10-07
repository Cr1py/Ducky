"""LLM access: a thin ModelClient over per-provider adapters.

DUCKY_LLM=stub forces the offline StubClient (used by the test-suite and handy
for trying the CLI without a key).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from pydantic import ValidationError

from ducky import keys, registry
from ducky.llm.adapters import ADAPTERS
from ducky.llm.errors import LLMError, friendly_error
from ducky.llm.schema import RESPONSE_JSON_SCHEMA, DuckyResponse
from ducky.registry import ModelSpec


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict[str, str]]) -> DuckyResponse:
        """One conversational turn -> structured response."""

    def complete_text(self, system: str, messages: list[dict[str, str]]) -> str:
        """Plain-text completion (used by the summarizer)."""


# --- response parsing -------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = _FENCE.sub("", text.strip())
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_response(text: str) -> DuckyResponse:
    """Parse the model's JSON, tolerating fences/prose; never lose the reply.

    If the model ignored the format, the raw text becomes the reply (state
    flags stay at their defaults), so the user still sees something useful.
    """
    data = _extract_json(text)
    if (
        data is None
        or not isinstance(data.get("reply"), str)
        or not data["reply"].strip()
    ):
        return DuckyResponse(reply=text.strip() or "(empty response)")
    try:
        data["hint_level"] = max(0, min(int(data.get("hint_level", 0)), 10))
    except (TypeError, ValueError):
        data["hint_level"] = 0
    if data.get("style") not in ("socratic", "hint"):
        data["style"] = "socratic"
    try:
        return DuckyResponse.model_validate(data)
    except ValidationError:
        return DuckyResponse(reply=data["reply"])


def merge_consecutive(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge back-to-back same-role messages.

    They occur when an LLM call failed (the user's turn was saved but got no reply).
    Providers differ on whether they tolerate this, so normalize once for all.
    """
    merged: list[dict[str, str]] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {
                "role": m["role"],
                "content": merged[-1]["content"] + "\n\n" + m["content"],
            }
        else:
            merged.append(dict(m))
    return merged


# --- clients ----------------------------------------------------------------


class ModelClient:
    """Resolves the key, builds the right adapter lazily, maps errors, parses replies."""

    def __init__(
        self,
        name: str,
        spec: ModelSpec,
        http_client: Any = None,
        **adapter_kwargs: Any,
    ) -> None:
        self.name = name
        self.spec = spec
        self.last_usage: tuple[int, int] | None = (
            None  # (input, output) tokens; hook for cost display
        )
        self._http_client = http_client
        self._adapter_kwargs = adapter_kwargs

    def _call(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        key: str | None = None
        if self.spec.api_key_env:
            key, _ = keys.get_key(self.spec.api_key_env)
            if not key:
                raise LLMError(
                    f"No API key for '{self.name}'. Set {self.spec.api_key_env} in your "
                    f"environment, or run `ducky config set-key {self.name}`."
                )
        try:
            adapter = ADAPTERS[self.spec.provider](
                self.spec, key, http_client=self._http_client, **self._adapter_kwargs
            )
            result = adapter.chat(
                system, merge_consecutive(messages), json_schema=json_schema
            )
        except LLMError:
            raise
        except Exception as e:  # SDKs raise many exception types
            raise LLMError(friendly_error(e, key, self.name)) from e

        self.last_usage = (result.input_tokens, result.output_tokens)
        if not result.text.strip():
            if result.truncated:
                raise LLMError(
                    f"'{self.name}' hit its token limit ({self.spec.max_tokens}) before answering, "
                    "which is common with reasoning models. Raise `max_tokens` for it in models.toml."
                )
            raise LLMError("The model returned an empty response. Try again.")
        return result.text

    def complete(self, system: str, messages: list[dict[str, str]]) -> DuckyResponse:
        return parse_response(
            self._call(system, messages, json_schema=RESPONSE_JSON_SCHEMA)
        )

    def complete_text(self, system: str, messages: list[dict[str, str]]) -> str:
        return self._call(system, messages).strip()


class StubClient:
    """Offline stand-in: crude heuristics so phase transitions can be exercised.

    5+ words counts as a system description; mentioning problem/bug/error/stuck
    counts as stating the problem.
    """

    PROBLEM_WORDS = ("problem", "bug", "error", "stuck", "broken", "fails", "wrong")

    def complete(self, system: str, messages: list[dict[str, str]]) -> DuckyResponse:
        last = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        lowered = last.lower()
        return DuckyResponse(
            reply=(
                f'[stub LLM] I heard: "{last}". '
                "What do you expect to happen next, and what actually happens?"
            ),
            style="socratic",
            hint_level=0,
            has_system_description=len(last.split()) >= 5,
            has_problem=any(w in lowered for w in self.PROBLEM_WORDS),
        )

    def complete_text(self, system: str, messages: list[dict[str, str]]) -> str:
        body = messages[-1]["content"] if messages else ""
        return "[stub summary] " + " ".join(body.split())[:300]


def get_client(cfg: dict[str, Any]) -> LLMClient:
    if os.environ.get("DUCKY_LLM", "").lower() == "stub":
        return StubClient()
    try:
        name = cfg["agent"]
        return ModelClient(name, registry.get_spec(name))
    except registry.RegistryError as e:  # bad agent name or a broken models.toml
        raise LLMError(str(e)) from e
