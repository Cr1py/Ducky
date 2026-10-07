"""End-phrase handling.

M0 ships exact tail matching + DB helpers. M5 upgrades matching to fuzzy
(rapidfuzz) and adds `phrases test` alias capture.
"""

from __future__ import annotations

import re
import sqlite3

MIN_WORDS = 2


class PhraseError(ValueError):
    pass


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def strip_end_phrase(transcript: str, phrases: list[str]) -> tuple[str, bool]:
    """If the transcript *ends* with an end phrase, return it stripped.

    Only the tail is checked so a phrase said mid-sentence doesn't end the turn.
    """
    words = transcript.split()
    for phrase in sorted(phrases, key=lambda p: -len(p.split())):
        target = _normalize(phrase)
        if not target or len(words) < len(target):
            continue
        tail = _normalize(" ".join(words[-len(target) :]))
        if tail == target:
            return " ".join(words[: -len(target)]).rstrip(" ,.;:-"), True
    return transcript, False


def list_phrases(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM end_phrases ORDER BY id").fetchall()


def add_phrase(conn: sqlite3.Connection, phrase: str, is_alias: bool = False) -> str:
    phrase = " ".join(_normalize(phrase))
    if len(phrase.split()) < MIN_WORDS:
        raise PhraseError(
            f"End phrases need at least {MIN_WORDS} words to avoid false triggers."
        )
    try:
        conn.execute(
            "INSERT INTO end_phrases (phrase, is_alias) VALUES (?, ?)",
            (phrase, int(is_alias)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise PhraseError(f"'{phrase}' is already an end phrase.") from None
    return phrase


def remove_phrase(conn: sqlite3.Connection, phrase: str) -> None:
    phrase = " ".join(_normalize(phrase))
    cur = conn.execute("DELETE FROM end_phrases WHERE phrase = ?", (phrase,))
    conn.commit()
    if cur.rowcount == 0:
        raise PhraseError(f"No end phrase '{phrase}'.")
