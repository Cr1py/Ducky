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
