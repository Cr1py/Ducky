"""Pruning + rolling summary.

Rules
  * Pinned system_description / problem_statement live in their own columns and
    are never pruned.
  * The last KEEP_RECENT_TURNS turns always stay verbatim.
  * When the live context exceeds context_budget(cfg), older turns are
    folded into `summary` and flagged summarized=1 (rows are kept, so nothing is
    lost; they just stop being sent to the LLM).
  * A summary failure never loses data and never blocks the user: the turns stay
    unsummarized and we retry on the next opportunity.
  * Only text is stored; no audio ever.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from ducky import registry, sessions
from ducky.config import ConfigError
from ducky.llm import prompts
from ducky.llm.client import LLMClient, LLMError, get_client
from ducky.sessions import Session, Turn

KEEP_RECENT_TURNS = 6   # 3 exchanges stay verbatim
MIN_FOLD_TURNS = 4      # a forced refresh (end/switch) skips tiny sessions
CHARS_PER_TOKEN = 4     # rough heuristic; avoids a tokenizer dependency
STATIC_PROMPT_TOKENS = 800  # system prompt + format instructions, with headroom
MIN_BUDGET = 500


@dataclass
class PruneResult:
    folded: int = 0
    error: str | None = None


def estimate_tokens(text: str | None) -> int:
    return len(text) // CHARS_PER_TOKEN + 1 if text else 0


def context_tokens(session: Session, turns: list[Turn]) -> int:
    """Estimated size of what we send each turn (excluding the static prompt)."""
    pinned = (session.system_description, session.problem_statement, session.summary)
    return sum(estimate_tokens(t) for t in pinned) + sum(estimate_tokens(t.text) for t in turns)


def context_budget(cfg: dict[str, Any]) -> int:
    """Tokens of history we allow before pruning.

    Normally `max_context_tokens`. If the active model declares a context window
    (`num_ctx`, Ollama), it's capped so prompt + reply always fit: servers like Ollama
    silently drop whatever doesn't, which would eat the system prompt.
    """
    budget = cfg["max_context_tokens"]
    agent = cfg.get("agent")
    if agent:
        try:
            spec = registry.get_spec(agent)
        except ConfigError:
            return budget
        if spec.num_ctx:
            budget = min(budget, spec.num_ctx - STATIC_PROMPT_TOKENS - spec.max_tokens)
    return max(budget, MIN_BUDGET)


def select_foldable(turns: list[Turn], keep: int = KEEP_RECENT_TURNS) -> list[Turn]:
    """Oldest live turns eligible for folding, leaving `keep` verbatim.

    The kept window is nudged so it starts with a user turn (providers expect
    the conversation to open with one).
    """
    live = [t for t in turns if not t.summarized]
    if len(live) <= keep:
        return []
    boundary = len(live) - keep
    while boundary < len(live) and live[boundary].role != "user":
        boundary += 1
    return live[:boundary]


def needs_prune(conn: sqlite3.Connection, cfg: dict[str, Any], session: Session) -> bool:
    session = sessions.get_session(conn, session.id) or session
    turns = sessions.get_turns(conn, session.id, include_summarized=False)
    over = context_tokens(session, turns) > context_budget(cfg)
    return over and bool(select_foldable(turns))


def _fold(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    session: Session,
    foldable: list[Turn],
    client: LLMClient | None,
) -> PruneResult:
    system, messages = prompts.build_summary_request(session, foldable)
    try:
        # Building the client can fail too (unknown agent, broken models.toml), so it
        # lives inside the try: a summary problem must never crash the user's command.
        client = client or get_client(cfg)
        summary = client.complete_text(system, messages).strip()
    except LLMError as e:
        return PruneResult(error=str(e))
    if not summary:
        return PruneResult(error="The summarizer returned nothing.")
    sessions.apply_summary(conn, session.id, summary, [t.id for t in foldable])
    return PruneResult(folded=len(foldable))


def maybe_prune(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    session: Session,
    client: LLMClient | None = None,
) -> PruneResult:
    """Fold older turns into the summary if the context has grown too large."""
    if not needs_prune(conn, cfg, session):
        return PruneResult()
    session = sessions.get_session(conn, session.id) or session  # latest summary
    foldable = select_foldable(sessions.get_turns(conn, session.id, include_summarized=False))
    return _fold(conn, cfg, session, foldable, client)


def refresh_summary(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    session: Session,
    client: LLMClient | None = None,
) -> PruneResult:
    """Called before ending or switching away: fold everything older than the
    recent window, regardless of size, so the stored summary is current."""
    session = sessions.get_session(conn, session.id) or session
    foldable = select_foldable(sessions.get_turns(conn, session.id, include_summarized=False))
    if len(foldable) < MIN_FOLD_TURNS:
        return PruneResult()
    return _fold(conn, cfg, session, foldable, client)
