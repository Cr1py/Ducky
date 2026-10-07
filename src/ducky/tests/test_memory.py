from ducky import memory, sessions
from ducky.llm import LLMError


class FakeSummarizer:
    def __init__(self, text="SUMMARY", fail=False):
        self.text, self.fail, self.requests = text, fail, []

    def complete_text(self, system, messages):
        self.requests.append((system, messages))
        if self.fail:
            raise LLMError("network down")
        return self.text


CFG = {"max_context_tokens": 500}


def seed(conn, n_turns, chars=400, name="s"):
    s = sessions.create_session(conn, name)
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "ducky"
        sessions.add_turn(conn, s.id, role, f"{role}-{i} " + "x" * chars)
    return sessions.get_session(conn, s.id)


def test_estimate_tokens():
    assert memory.estimate_tokens("") == 0
    assert memory.estimate_tokens("a" * 400) > 90


def test_select_foldable_keeps_recent_window_starting_on_user(conn):
    s = seed(conn, 10)
    turns = sessions.get_turns(conn, s.id)
    foldable = memory.select_foldable(turns, keep=5)
    # 10 turns, keep 5 -> boundary 5 is a ducky turn -> nudged to 6 so the window starts with user
    assert len(foldable) == 6
    window = turns[len(foldable):]
    assert window[0].role == "user"


def test_nothing_to_fold_when_short(conn):
    s = seed(conn, 6)
    assert memory.select_foldable(sessions.get_turns(conn, s.id)) == []


def test_below_threshold_does_nothing(conn):
    s = seed(conn, 10, chars=5)
    fake = FakeSummarizer()
    assert memory.maybe_prune(conn, CFG, s, fake) == memory.PruneResult()
    assert fake.requests == []


def test_prune_folds_and_marks_turns(conn):
    s = seed(conn, 10)
    fake = FakeSummarizer("merged summary")
    result = memory.maybe_prune(conn, CFG, s, fake)
    assert result.folded == 4 and result.error is None
    updated = sessions.get_session(conn, s.id)
    assert updated.summary == "merged summary"
    live = sessions.get_turns(conn, s.id, include_summarized=False)
    assert len(live) == 6 and live[0].role == "user"
    assert len(sessions.get_turns(conn, s.id)) == 10  # rows are kept, only hidden from the LLM
    # the summarizer saw the folded turns and the existing summary slot
    body = fake.requests[0][1][0]["content"]
    assert "user-0" in body and "Existing summary" in body and "user-8" not in body


def test_second_prune_includes_previous_summary(conn):
    s = seed(conn, 10)
    memory.maybe_prune(conn, CFG, s, FakeSummarizer("first"))
    for i in range(4):
        sessions.add_turn(conn, s.id, "user" if i % 2 == 0 else "ducky", f"later-{i} " + "y" * 400)
    fake = FakeSummarizer("second")
    memory.maybe_prune(conn, CFG, sessions.get_session(conn, s.id), fake)
    assert "first" in fake.requests[0][1][0]["content"]
    assert sessions.get_session(conn, s.id).summary == "second"


def test_summarizer_failure_loses_nothing(conn):
    s = seed(conn, 10)
    result = memory.maybe_prune(conn, CFG, s, FakeSummarizer(fail=True))
    assert result.folded == 0 and "network down" in result.error
    assert sessions.get_session(conn, s.id).summary is None
    assert len(sessions.get_turns(conn, s.id, include_summarized=False)) == 10


def test_pinned_fields_are_untouched(conn):
    s = seed(conn, 10)
    sessions.update_session(conn, s.id, system_description="SYS", problem_statement="PROB")
    memory.maybe_prune(conn, CFG, sessions.get_session(conn, s.id), FakeSummarizer())
    updated = sessions.get_session(conn, s.id)
    assert (updated.system_description, updated.problem_statement) == ("SYS", "PROB")


def test_refresh_ignores_threshold_but_skips_tiny_sessions(conn):
    tiny = seed(conn, 8, chars=5, name="tiny")  # only 2 foldable (< MIN_FOLD_TURNS)
    fake = FakeSummarizer()
    assert memory.refresh_summary(conn, CFG, tiny, fake).folded == 0 and not fake.requests

    big = seed(conn, 12, chars=5, name="big")  # under token threshold, but 6 foldable
    assert memory.refresh_summary(conn, CFG, big, fake).folded == 6
    assert sessions.get_session(conn, big.id).summary == "SUMMARY"


def test_build_messages_wraps_history_as_json_and_starts_with_user(conn):
    import json

    from ducky.llm import prompts

    s = sessions.create_session(conn)
    sessions.add_turn(conn, s.id, "ducky", "orphan reply", hint_level=1, style="hint")
    sessions.add_turn(conn, s.id, "user", "question")
    sessions.add_turn(conn, s.id, "ducky", "answer", hint_level=2, style="hint")
    msgs = prompts.build_messages(sessions.get_turns(conn, s.id))
    assert msgs[0] == {"role": "user", "content": "question"}  # leading assistant dropped
    assert json.loads(msgs[1]["content"]) == {"reply": "answer", "style": "hint", "hint_level": 2}


def test_client_construction_failure_is_a_prune_error_not_a_crash(conn, monkeypatch):
    monkeypatch.delenv("DUCKY_LLM")
    s = seed(conn, 10)
    result = memory.maybe_prune(conn, {"max_context_tokens": 500, "agent": "nope"}, s)
    assert result.folded == 0 and "Unknown model 'nope'" in result.error
    assert len(sessions.get_turns(conn, s.id, include_summarized=False)) == 10


def _models_file(ducky_home, text):
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_context_budget_defaults_to_the_config_value():
    assert memory.context_budget({"max_context_tokens": 6000}) == 6000
    assert memory.context_budget({"max_context_tokens": 6000, "agent": "claude"}) == 6000
    assert memory.context_budget({"max_context_tokens": 6000, "agent": "no-such-model"}) == 6000


def test_context_budget_shrinks_to_fit_a_small_ollama_window(ducky_home):
    # Built-in `ollama`: num_ctx 8192, max_tokens 1024 -> 8192 - 800 - 1024 = 6368 > 6000, no change.
    assert memory.context_budget({"max_context_tokens": 6000, "agent": "ollama"}) == 6000
    _models_file(ducky_home, '[models.ollama]\nnum_ctx = 4096\n')
    # 4096 - 800 (prompt) - 1024 (reply) leaves 2272 tokens for history.
    assert memory.context_budget({"max_context_tokens": 6000, "agent": "ollama"}) == 2272


def test_context_budget_has_a_floor(ducky_home):
    _models_file(ducky_home, '[models.ollama]\nnum_ctx = 1000\n')
    assert memory.context_budget({"max_context_tokens": 6000, "agent": "ollama"}) == memory.MIN_BUDGET


def test_small_ollama_window_triggers_pruning_the_config_threshold_would_not(conn, ducky_home):
    _models_file(ducky_home, '[models.ollama]\nnum_ctx = 4096\n')
    s = seed(conn, 10, chars=1200)  # ~3000 tokens of history: under 6000, over 2272
    roomy = {"max_context_tokens": 6000, "agent": "claude"}
    tight = {"max_context_tokens": 6000, "agent": "ollama"}
    assert memory.needs_prune(conn, roomy, s) is False
    assert memory.needs_prune(conn, tight, s) is True
