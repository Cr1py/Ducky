from ducky import sessions, turn_runner
from ducky.input import TextInput
from ducky.llm import DuckyResponse


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, messages))
        return DuckyResponse(reply="ok", **self.kwargs)


def run(conn, cfg, session, text, client):
    return turn_runner.run_turn(conn, cfg, session, TextInput(text), client)


def cfg(**over):
    return {"answers": False, "tone": "default", **over}


def test_phase_flow_system_then_problem(conn):
    s = sessions.create_session(conn)
    r = run(conn, cfg(), s, "it is a flask app with a redis cache",
            FakeClient(has_system_description=True))
    assert r.session.phase == "awaiting_problem"
    assert r.session.system_description == "it is a flask app with a redis cache"
    r = run(conn, cfg(), r.session, "the cache returns stale data",
            FakeClient(has_problem=True))
    assert r.session.phase == "iterating"
    assert r.session.problem_statement == "the cache returns stale data"


def test_nudges_then_proceeds(conn):
    s = sessions.create_session(conn)
    r = run(conn, cfg(), s, "umm", FakeClient())
    assert r.session.phase == "awaiting_system" and r.session.intro_nudges == 1
    r = run(conn, cfg(), r.session, "hmm", FakeClient())
    assert r.session.phase == "iterating"  # stopped nagging after 2 misses


def test_hint_clamped_without_answers(conn):
    s = sessions.create_session(conn)
    r = run(conn, cfg(answers=False), s, "x", FakeClient(hint_level=4))
    assert r.hint_level == 3


def test_hint_allowed_with_answers(conn):
    s = sessions.create_session(conn)
    r = run(conn, cfg(answers=True), s, "x", FakeClient(hint_level=4))
    assert r.hint_level == 4


def test_user_turn_saved_even_if_llm_fails(conn):
    class Boom:
        def complete(self, system, messages):
            raise RuntimeError("network down")

    s = sessions.create_session(conn)
    try:
        run(conn, cfg(), s, "important thought", Boom())
    except RuntimeError:
        pass
    turns = sessions.get_turns(conn, s.id)
    assert [t.text for t in turns] == ["important thought"]


def test_prompt_contains_pinned_and_cap(conn):
    s = sessions.create_session(conn)
    client = FakeClient(has_system_description=True)
    run(conn, cfg(), s, "a flask app with redis", client)
    system, messages = client.calls[0]
    assert "level 3" in system and "Never give the direct answer" in system
    assert messages == [{"role": "user", "content": "a flask app with redis"}]
