"""Adapters run the REAL openai/anthropic SDK code and our Gemini REST code against
httpx2.MockTransport, so request shapes and response parsing are verified offline."""
import json

import httpx2
import pytest

from ducky.llm import LLMError, ModelClient
from ducky.llm.client import StubClient, get_client, merge_consecutive, parse_response
from ducky.llm.errors import APIStatusError, friendly_error
from ducky.registry import ModelSpec

USER = [{"role": "user", "content": "hi"}]


def mock(handler):
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def recorder(status=200, payload=None):
    """Handler that records the request and returns a canned response."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(status, json=payload)

    return handler, seen


def body(request):
    return json.loads(request.content)


def openai_payload(content="ok", finish="stop"):
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def anthropic_payload(text="ok", stop="end_turn"):
    return {
        "id": "m1", "type": "message", "role": "assistant", "model": "m",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def gemini_payload(text="ok", finish="STOP", extra_parts=()):
    parts = [{"text": text}, *extra_parts] if text is not None else []
    content = {"role": "model", "parts": parts} if parts else {"role": "model"}
    return {
        "candidates": [{"content": content, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
    }


OPENAI = ModelSpec(provider="openai_compat", base_url="https://api.test/v1", model="gpt-x",
                   api_key_env="OPENAI_API_KEY", max_tokens=123)
ANTHROPIC = ModelSpec(provider="anthropic", model="claude-x", api_key_env="ANTHROPIC_API_KEY", max_tokens=321)
GEMINI = ModelSpec(provider="gemini", model="gemini-x", api_key_env="GEMINI_API_KEY", max_tokens=222)


# --- parsing ---------------------------------------------------------------

def test_parse_plain_json():
    r = parse_response(json.dumps({"reply": "hi", "hint_level": 2, "has_problem": True}))
    assert (r.reply, r.hint_level, r.has_problem) == ("hi", 2, True)


def test_parse_fenced_and_chatty_json():
    r = parse_response('Sure!\n```json\n{"reply": "hello", "style": "hint"}\n```\nHope that helps')
    assert r.reply == "hello" and r.style == "hint"


def test_parse_non_json_falls_back_to_reply():
    r = parse_response("Just some prose from a stubborn model.")
    assert r.reply == "Just some prose from a stubborn model." and r.hint_level == 0


def test_parse_sanitizes_bad_fields():
    r = parse_response(json.dumps({"reply": "x", "hint_level": 99, "style": "weird"}))
    assert r.hint_level == 10 and r.style == "socratic"
    assert parse_response(json.dumps({"reply": "x", "hint_level": "high"})).hint_level == 0


def test_parse_missing_reply_uses_raw_text():
    raw = json.dumps({"style": "hint"})
    assert parse_response(raw).reply == raw


def test_merge_consecutive():
    merged = merge_consecutive([
        {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"},
    ])
    assert [m["role"] for m in merged] == ["user", "assistant", "user"]
    assert merged[0]["content"] == "a\n\nb"


# --- OpenAI-compatible -----------------------------------------------------

def test_openai_request_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    handler, seen = recorder(payload=openai_payload('{"reply": "yo"}'))
    client = ModelClient("chatgpt", OPENAI, http_client=mock(handler))
    out = client.complete("SYS", USER)

    assert out.reply == "yo"
    req = seen[0]
    assert req.url.path.endswith("/chat/completions") and req.url.host == "api.test"
    assert req.headers["authorization"] == "Bearer sk-test"
    sent = body(req)
    assert sent["model"] == "gpt-x"
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["max_tokens"] == 123 and "max_completion_tokens" not in sent
    assert "temperature" not in sent
    assert client.last_usage == (11, 7)


def test_openai_token_param_is_configurable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    spec = OPENAI.model_copy(update={"token_param": "max_completion_tokens"})
    handler, seen = recorder(payload=openai_payload())
    ModelClient("chatgpt", spec, http_client=mock(handler)).complete_text("s", USER)
    sent = body(seen[0])
    assert sent["max_completion_tokens"] == 123 and "max_tokens" not in sent


def test_keyless_local_server_needs_no_key():
    local = ModelSpec(provider="openai_compat", base_url="http://localhost:11434/v1", model="llama")
    handler, seen = recorder(payload=openai_payload("local ok"))
    assert ModelClient("local", local, http_client=mock(handler)).complete_text("s", USER) == "local ok"
    assert seen[0].headers["authorization"] == "Bearer not-needed"


def test_openai_auth_error_is_mapped_and_key_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
    handler, _ = recorder(401, {"error": {"message": "Incorrect API key: sk-secret-123", "type": "x"}})
    with pytest.raises(LLMError) as exc:
        ModelClient("chatgpt", OPENAI, http_client=mock(handler)).complete("s", USER)
    msg = str(exc.value)
    assert "rejected the API key" in msg and "set-key chatgpt" in msg
    assert "sk-secret-123" not in msg


def test_openai_404_mentions_models_toml(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    handler, _ = recorder(404, {"error": {"message": "model not found"}})
    with pytest.raises(LLMError, match="models.toml"):
        ModelClient("chatgpt", OPENAI, http_client=mock(handler)).complete("s", USER)


def test_reasoning_model_out_of_tokens_gets_actionable_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    handler, _ = recorder(payload=openai_payload("", finish="length"))
    with pytest.raises(LLMError, match="token limit.*max_tokens"):
        ModelClient("chatgpt", OPENAI, http_client=mock(handler)).complete("s", USER)


def test_empty_non_truncated_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    handler, _ = recorder(payload=openai_payload("   "))
    with pytest.raises(LLMError, match="empty"):
        ModelClient("chatgpt", OPENAI, http_client=mock(handler)).complete("s", USER)


# --- Anthropic -------------------------------------------------------------

def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    handler, seen = recorder(payload=anthropic_payload('{"reply": "quack"}'))
    client = ModelClient("claude", ANTHROPIC, http_client=mock(handler))
    out = client.complete("SYS", USER)

    assert out.reply == "quack"
    req = seen[0]
    assert req.url.path.endswith("/v1/messages")
    assert req.headers["x-api-key"] == "sk-ant"
    sent = body(req)
    assert sent["system"] == "SYS"                      # top-level, not a message
    assert all(m["role"] != "system" for m in sent["messages"])
    assert sent["max_tokens"] == 321 and sent["model"] == "claude-x"
    assert "temperature" not in sent
    assert client.last_usage == (5, 3)


def test_anthropic_merges_consecutive_user_turns(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    handler, seen = recorder(payload=anthropic_payload())
    ModelClient("claude", ANTHROPIC, http_client=mock(handler)).complete_text(
        "s", [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
    )
    assert body(seen[0])["messages"] == [{"role": "user", "content": "first\n\nsecond"}]


def test_anthropic_auth_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    handler, _ = recorder(401, {"type": "error", "error": {"type": "authentication_error",
                                                            "message": "invalid x-api-key sk-ant-secret"}})
    with pytest.raises(LLMError) as exc:
        ModelClient("claude", ANTHROPIC, http_client=mock(handler)).complete("s", USER)
    assert "rejected the API key" in str(exc.value) and "sk-ant-secret" not in str(exc.value)


def test_anthropic_max_tokens_stop(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    handler, _ = recorder(payload=anthropic_payload("", stop="max_tokens"))
    with pytest.raises(LLMError, match="token limit"):
        ModelClient("claude", ANTHROPIC, http_client=mock(handler)).complete("s", USER)


# --- Gemini (generateContent REST) -----------------------------------------

def test_gemini_request_shape(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    handler, seen = recorder(payload=gemini_payload('{"reply": "hello"}'))
    client = ModelClient("gemini", GEMINI, http_client=mock(handler))
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert client.complete("SYS", history).reply == "hello"

    req = seen[0]
    assert req.url.path.endswith("/models/gemini-x:generateContent")
    assert req.headers["x-goog-api-key"] == "g-key"
    assert "g-key" not in str(req.url)                  # the key never goes in the URL
    sent = body(req)
    assert sent["system_instruction"]["parts"][0]["text"] == "SYS"
    assert [c["role"] for c in sent["contents"]] == ["user", "model", "user"]
    assert sent["generationConfig"]["maxOutputTokens"] == 222
    assert client.last_usage == (9, 4)


def test_gemini_ignores_thought_parts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    payload = gemini_payload("answer", extra_parts=[{"text": "private reasoning", "thought": True}])
    handler, _ = recorder(payload=payload)
    out = ModelClient("gemini", GEMINI, http_client=mock(handler)).complete_text("s", USER)
    assert out == "answer"


def test_gemini_retries_once_on_429(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(429, json={"error": {"message": "slow down"}})
        return httpx2.Response(200, json=gemini_payload("recovered"))

    client = ModelClient("gemini", GEMINI, http_client=mock(handler), retry_delay=0)
    assert client.complete_text("s", USER) == "recovered" and len(calls) == 2


def test_gemini_gives_up_after_one_retry(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    handler, seen = recorder(503, {"error": {"message": "overloaded"}})
    with pytest.raises(LLMError, match="having trouble"):
        ModelClient("gemini", GEMINI, http_client=mock(handler), retry_delay=0).complete("s", USER)
    assert len(seen) == 2


def test_gemini_401_not_retried_and_key_scrubbed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-secret")
    handler, seen = recorder(401, {"error": {"message": "API key g-secret not valid"}})
    with pytest.raises(LLMError) as exc:
        ModelClient("gemini", GEMINI, http_client=mock(handler), retry_delay=0).complete("s", USER)
    assert len(seen) == 1
    assert "rejected the API key" in str(exc.value) and "g-secret" not in str(exc.value)


def test_gemini_blocked_prompt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    handler, _ = recorder(payload={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMError, match="blocked.*SAFETY"):
        ModelClient("gemini", GEMINI, http_client=mock(handler)).complete("s", USER)


def test_gemini_thinking_model_out_of_tokens(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    handler, _ = recorder(payload=gemini_payload(None, finish="MAX_TOKENS"))
    with pytest.raises(LLMError, match="token limit"):
        ModelClient("gemini", GEMINI, http_client=mock(handler)).complete("s", USER)


def test_gemini_custom_base_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    spec = GEMINI.model_copy(update={"base_url": "https://proxy.test/v1beta/"})
    handler, seen = recorder(payload=gemini_payload())
    ModelClient("gemini", spec, http_client=mock(handler)).complete_text("s", USER)
    assert seen[0].url.host == "proxy.test" and "//models" not in str(seen[0].url)


# --- keys, errors, factory -------------------------------------------------

def test_missing_key_message():
    with pytest.raises(LLMError) as exc:
        ModelClient("claude", ANTHROPIC).complete("s", USER)
    assert "ANTHROPIC_API_KEY" in str(exc.value) and "set-key claude" in str(exc.value)


class _Err(Exception):
    def __init__(self, status_code=None):
        super().__init__("boom")
        self.status_code = status_code


@pytest.mark.parametrize(
    "status,fragment",
    [(401, "rejected the API key"), (403, "rejected the API key"), (404, "not found"),
     (429, "Rate limited"), (400, "token_param"), (500, "having trouble"), (503, "having trouble")],
)
def test_friendly_error_by_status(status, fragment):
    assert fragment in friendly_error(_Err(status), entry="claude")


def test_friendly_error_by_name():
    assert "timed out" in friendly_error(type("APITimeoutError", (Exception,), {})("x"))
    assert "Couldn't reach" in friendly_error(type("ConnectError", (Exception,), {})("x"))
    assert "WeirdError" in friendly_error(type("WeirdError", (Exception,), {})("x"))
    assert friendly_error(APIStatusError(429, "slow")).startswith("Rate limited")


def test_get_client_stub_and_real(monkeypatch):
    assert isinstance(get_client({"agent": "claude"}), StubClient)
    monkeypatch.delenv("DUCKY_LLM")
    client = get_client({"agent": "claude"})
    assert isinstance(client, ModelClient) and client.spec.provider == "anthropic"


def test_get_client_unknown_agent(monkeypatch):
    monkeypatch.delenv("DUCKY_LLM")
    with pytest.raises(LLMError, match="Unknown model 'nope'.*Options"):
        get_client({"agent": "nope"})


def test_get_client_broken_models_file(monkeypatch, ducky_home):
    monkeypatch.delenv("DUCKY_LLM")
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True)
    path.write_text("[models.claude]\nbogus = 1\n")
    with pytest.raises(LLMError, match="bogus"):
        get_client({"agent": "claude"})
