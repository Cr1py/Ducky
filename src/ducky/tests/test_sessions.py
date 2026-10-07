import pytest

from ducky import sessions
from ducky.sessions import SessionError


def test_create_generates_unique_names(conn):
    names = {sessions.create_session(conn).name for _ in range(5)}
    assert len(names) == 5


def test_create_with_name_and_duplicate(conn):
    sessions.create_session(conn, "two-sum")
    with pytest.raises(SessionError, match="already exists"):
        sessions.create_session(conn, "two-sum")


def test_invalid_name_rejected(conn):
    with pytest.raises(SessionError):
        sessions.create_session(conn, "has spaces")


def test_rename(conn):
    sessions.create_session(conn, "a")
    sessions.create_session(conn, "b")
    assert sessions.rename_session(conn, "a", "c").name == "c"
    with pytest.raises(SessionError, match="already exists"):
        sessions.rename_session(conn, "c", "b")
    with pytest.raises(SessionError, match="No session"):
        sessions.rename_session(conn, "ghost", "x")


def test_delete_active_clears_pointer_and_turns(conn):
    s = sessions.create_session(conn, "a")
    sessions.set_active(conn, s.id)
    sessions.add_turn(conn, s.id, "user", "hello")
    assert sessions.delete_session(conn, "a") is True
    assert sessions.get_active(conn) is None
    assert conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0


def test_delete_inactive_keeps_pointer(conn):
    a = sessions.create_session(conn, "a")
    sessions.create_session(conn, "b")
    sessions.set_active(conn, a.id)
    assert sessions.delete_session(conn, "b") is False
    assert sessions.get_active(conn).name == "a"


def test_update_whitelist(conn):
    s = sessions.create_session(conn)
    with pytest.raises(ValueError):
        sessions.update_session(conn, s.id, name="nope")
    sessions.update_session(conn, s.id, phase="iterating", hint_level=2)
    updated = sessions.get_session(conn, s.id)
    assert (updated.phase, updated.hint_level) == ("iterating", 2)


def test_migration_is_idempotent(conn):
    from ducky import db

    db.migrate(conn)
    assert conn.execute("SELECT COUNT(*) FROM end_phrases").fetchone()[0] == 1
