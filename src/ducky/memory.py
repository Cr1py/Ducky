"""Pruning + rolling summary (milestone M3). No-ops for now so the hooks exist.

Rules to implement:
  * pinned system_description / problem_statement are never pruned
  * keep last N turns verbatim; fold older turns into `summary`, set summarized=1
  * summarizer must preserve: current approach, hints given, hint level, decisions
  * refresh incrementally after turns, not only on switch/end
"""
from __future__ import annotations

import sqlite3
from typing import Any

from ducky.sessions import Session


def maybe_prune(conn: sqlite3.Connection, cfg: dict[str, Any], session: Session) -> None:
    """TODO(M3): prune when history exceeds cfg['max_context_tokens']."""


def refresh_summary(conn: sqlite3.Connection, cfg: dict[str, Any], session: Session) -> None:
    """TODO(M3): called before switching away from / ending a session."""
