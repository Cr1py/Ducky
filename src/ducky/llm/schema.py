from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DuckyResponse(BaseModel):
    """What the LLM returns each turn: the reply plus state updates."""

    reply: str
    style: Literal["socratic", "hint"] = "socratic"
    hint_level: int = Field(default=0, ge=0, le=10)  # clamped by code, not trusted
    has_system_description: bool = False
    has_problem: bool = False
    system_description: str | None = None  # updated pinned text, if it changed
    problem_statement: str | None = None


# handed to providers that can enforce structured output (Ollama's format)
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "style": {"type": "string", "enum": ["socratic", "hint"]},
        "hint_level": {"type": "integer"},
        "has_system_description": {"type": "boolean"},
        "has_problem": {"type": "boolean"},
        "system_description": {"type": "string"},
        "problem_statement": {"type": "string"},
    },
    "required": [
        "reply",
        "style",
        "hint_level",
        "has_system_description",
        "has_problem",
    ],
}
