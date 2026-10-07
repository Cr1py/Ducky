"""SQLite connection + migrations (tracked with PRAGMA user_version)"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ducky.config import db_path

MIGRATIONS: list[str] = [
    # v1: initial schema
    """
    CREATE TABLE sessions (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        name               TEXT    NOT NULL UNIQUE,
        phase              TEXT    NOT NULL DEFAULT 'awaiting_system'
                           CHECK (phase IN ('awaiting_system','awaiting_problem','iterating')),
        system_description TEXT,
        problem_statement  TEXT,
        summary            TEXT,
        hint_level         INTEGER NOT NULL DEFAULT 0,
        intro_nudges       INTEGER NOT NULL DEFAULT 0,
        created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
        last_active        TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE turns (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role       TEXT    NOT NULL CHECK (role IN ('user','ducky')),
        text       TEXT    NOT NULL,
        hint_level INTEGER,
        style      TEXT,
        summarized INTEGER NOT NULL DEFAULT 0,
        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX idx_turns_session ON turns(session_id, id);

    CREATE TABLE state (
        key   TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE end_phrases (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        phrase   TEXT    NOT NULL UNIQUE,
        is_alias INTEGER NOT NULL DEFAULT 0
    );
    INSERT INTO end_phrases (phrase) VALUES ('have any thoughts ducky');
    """,
]


def migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, sql in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.executescript(sql)
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn
