"""Session + turn persistence. The CLI is stateless, so the active session
is a pointer stored in the `state` table."""

from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass

NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
UPDATABLE = {
    "phase",
    "system_description",
    "problem_statement",
    "summary",
    "hint_level",
    "intro_nudges",
}

ADJECTIVES = [
    "quiet",
    "curious",
    "sleepy",
    "plucky",
    "mellow",
    "brave",
    "dapper",
    "fuzzy",
]
NOUNS = ["duckling", "mallard", "pond", "puddle", "feather", "bill", "wader", "teal"]


class SessionError(Exception):
    pass


@dataclass
class Session:
    id: int
    name: str
    phase: str
    system_description: str | None
    problem_statement: str | None
    summary: str | None
    hint_level: int
    intro_nudges: int
    created_at: str
    last_active: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Session":
        return cls(**{k: row[k] for k in row.keys()})


@dataclass
class Turn:
    id: int
    session_id: int
    role: str
    text: str
    hint_level: int | None
    style: str | None
    summarized: bool
    created_at: str


def _validate_name(name: str) -> str:
    name = name.strip()
    if not NAME_RE.match(name):
        raise SessionError("Session names may only use letters, numbers, '-' and '_'.")
    return name


def generate_name(conn: sqlite3.Connection) -> str:
    for _ in range(20):
        candidate = f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
        if get_session_by_name(conn, candidate) is None:
            return candidate
    return f"duck-{random.randint(1000, 9999)}"


def create_session(conn: sqlite3.Connection, name: str | None = None) -> Session:
    name = _validate_name(name) if name else generate_name(conn)
    if get_session_by_name(conn, name):
        raise SessionError(f"A session named '{name}' already exists.")
    cur = conn.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
    conn.commit()
    return get_session(conn, cur.lastrowid)  # type: ignore[arg-type,return-value]


def get_session(conn: sqlite3.Connection, session_id: int) -> Session | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return Session.from_row(row) if row else None


def get_session_by_name(conn: sqlite3.Connection, name: str) -> Session | None:
    row = conn.execute("SELECT * FROM sessions WHERE name = ?", (name,)).fetchone()
    return Session.from_row(row) if row else None


def list_sessions(conn: sqlite3.Connection) -> list[Session]:
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY last_active DESC, id DESC"
    ).fetchall()
    return [Session.from_row(r) for r in rows]


def rename_session(conn: sqlite3.Connection, old: str, new: str) -> Session:
    session = get_session_by_name(conn, old)
    if session is None:
        raise SessionError(f"No session named '{old}'.")
    new = _validate_name(new)
    if get_session_by_name(conn, new):
        raise SessionError(f"A session named '{new}' already exists.")
    conn.execute("UPDATE sessions SET name = ? WHERE id = ?", (new, session.id))
    conn.commit()
    return get_session(conn, session.id)  # type: ignore[return-value]


def delete_session(conn: sqlite3.Connection, name: str) -> bool:
    """Delete a session. Returns True if it was the active one (pointer cleared)."""
    session = get_session_by_name(conn, name)
    if session is None:
        raise SessionError(f"No session named '{name}'.")
    was_active = get_active_id(conn) == session.id
    conn.execute("DELETE FROM sessions WHERE id = ?", (session.id,))
    if was_active:
        clear_active(conn)
    conn.commit()
    return was_active


def update_session(conn: sqlite3.Connection, session_id: int, **fields) -> None:
    bad = set(fields) - UPDATABLE
    if bad:
        raise ValueError(f"Cannot update fields: {', '.join(sorted(bad))}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE sessions SET {sets}, last_active = datetime('now') WHERE id = ?",
        (*fields.values(), session_id),
    )
    conn.commit()


def touch(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute(
        "UPDATE sessions SET last_active = datetime('now') WHERE id = ?", (session_id,)
    )
    conn.commit()


# --- active session pointer -------------------------------------------------


def get_active_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT value FROM state WHERE key = 'active_session_id'"
    ).fetchone()
    return int(row["value"]) if row and row["value"] else None


def get_active(conn: sqlite3.Connection) -> Session | None:
    active_id = get_active_id(conn)
    return get_session(conn, active_id) if active_id is not None else None


def set_active(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute(
        "INSERT INTO state (key, value) VALUES ('active_session_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(session_id),),
    )
    conn.commit()


def clear_active(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM state WHERE key = 'active_session_id'")
    conn.commit()


# --- turns ------------------------------------------------------------------


def add_turn(
    conn: sqlite3.Connection,
    session_id: int,
    role: str,
    text: str,
    hint_level: int | None = None,
    style: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO turns (session_id, role, text, hint_level, style) VALUES (?,?,?,?,?)",
        (session_id, role, text, hint_level, style),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_turns(
    conn: sqlite3.Connection, session_id: int, include_summarized: bool = True
) -> list[Turn]:
    sql = "SELECT * FROM turns WHERE session_id = ?"
    if not include_summarized:
        sql += " AND summarized = 0"
    rows = conn.execute(sql + " ORDER BY id", (session_id,)).fetchall()
    return [
        Turn(
            id=r["id"],
            session_id=r["session_id"],
            role=r["role"],
            text=r["text"],
            hint_level=r["hint_level"],
            style=r["style"],
            summarized=bool(r["summarized"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


# --- summary ------------------------------------------------------------------


def apply_summary(
    conn: sqlite3.Connection, session_id: int, summary: str, turn_ids: list[int]
) -> None:
    """Store the new summary and mark the folded turns, atomically."""
    with conn:  # one transaction: both writes happen or neither does
        conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id)
        )
        conn.executemany(
            "UPDATE turns SET summarized = 1 WHERE id = ? AND session_id = ?",
            [(tid, session_id) for tid in turn_ids],
        )
