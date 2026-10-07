"""LLM access:

TODO(M1/M6): LiteLLMClient(provider/model from cfg["agent"]); keys come from
real env vars first, then a Ducky-specific env file in the config dir (never
the current directory's .env).
"""

from __future__ import annotations

from typing import Any, Protocol

from ducky.llm.schema import DuckyResponse


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(
        self, system: str, messages: list[dict[str, str]]
    ) -> DuckyResponse: ...


class StubClient:
    """Offline stand-in so the whole loop is runnable and testable.

    Uses crude heuristics for the has_* flags so phase transitions can be
    exercised: 5+ words counts as a system description, and mentioning
    'problem'/'bug'/'error'/'stuck' counts as stating the problem.
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


def get_client(cfg: dict[str, Any]) -> LLMClient:
    return StubClient()
