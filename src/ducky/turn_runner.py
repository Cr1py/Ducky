"""One turn of conversation: capture -> save -> LLM -> apply -> save.

Pruning is triggered by the CLI *after* the reply is shown, so the summarizer's
extra LLM call never delays the answer.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from ducky import sessions
from ducky.input.base import EmptyTranscript, InputSource
from ducky.llm import prompts
from ducky.llm.client import LLMClient, get_client
from ducky.llm.schema import DuckyResponse
from ducky.sessions import Session

MAX_INTRO_NUDGES = 2

OPENING_PROMPTS = {
    "awaiting_system": "Walk me through your system first: what are you building, "
    "and how do the pieces fit together?",
    "awaiting_problem": "Thanks, that helps. Now, what's the problem you're running into?",
    "iterating": "I'm listening. Talk me through your thinking.",
}


@dataclass
class TurnResult:
    response: DuckyResponse
    hint_level: int
    session: Session


def opening_prompt(session: Session) -> str:
    return OPENING_PROMPTS[session.phase]


def hint_cap(cfg: dict[str, Any]) -> int:
    """Level 4 (full solution) only exists when direct answers are enabled."""
    return 4 if cfg.get("answers") else 3


def apply_response(
    conn: sqlite3.Connection,
    session: Session,
    response: DuckyResponse,
    transcript: str,
    cap: int,
) -> int:
    """Update phase, pinned fields and hint level. Returns the clamped hint level."""
    fields: dict[str, Any] = {}

    if response.has_system_description:
        fields["system_description"] = response.system_description or (
            session.system_description or transcript
        )
    if response.has_problem:
        fields["problem_statement"] = response.problem_statement or (
            session.problem_statement or transcript
        )

    phase, nudges = session.phase, session.intro_nudges
    if phase == "awaiting_system":
        if response.has_system_description:
            phase = "iterating" if response.has_problem else "awaiting_problem"
            nudges = 0
        else:
            nudges += 1
    elif phase == "awaiting_problem":
        if response.has_problem:
            phase, nudges = "iterating", 0
        else:
            nudges += 1

    # Don't nag: after MAX_INTRO_NUDGES misses, proceed with what we have.
    if phase != "iterating" and nudges >= MAX_INTRO_NUDGES:
        phase, nudges = "iterating", 0

    hint_level = max(0, min(response.hint_level, cap))
    fields.update(phase=phase, intro_nudges=nudges, hint_level=hint_level)
    sessions.update_session(conn, session.id, **fields)
    return hint_level


def run_turn(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    session: Session,
    source: InputSource,
    client: LLMClient | None = None,
) -> TurnResult:
    transcript = source.get_transcript().strip()
    if not transcript:
        raise EmptyTranscript("I didn't catch anything.")

    # Save before any network call so a failed LLM request never loses the user's words.
    sessions.add_turn(conn, session.id, "user", transcript)

    cap = hint_cap(cfg)
    client = client or get_client(cfg)
    system = prompts.build_system_prompt(cfg, session, cap)
    turns = sessions.get_turns(conn, session.id, include_summarized=False)
    response = client.complete(system, prompts.build_messages(turns))

    hint_level = apply_response(conn, session, response, transcript, cap)
    sessions.add_turn(
        conn,
        session.id,
        "ducky",
        response.reply,
        hint_level=hint_level,
        style=response.style,
    )

    updated = sessions.get_session(conn, session.id)
    assert updated is not None
    return TurnResult(response=response, hint_level=hint_level, session=updated)
