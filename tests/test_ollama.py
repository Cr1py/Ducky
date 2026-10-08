"""Ollama's native /api/chat, run against httpx2.MockTransport (no server needed)."""
import json

import httpx2
import pytest

from ducky.llm import LLMError, ModelClient
from ducky.llm.schema import RESPONSE_JSON_SCHEMA, DuckyResponse
from ducky.registry import ModelSpec

USER = [{"role": "user", "content": "hi"}]

LOCAL = ModelSpec(provider="ollama", model="gemma4:e2b", num_ctx=8192, max_tokens=512,
                  keep_alive="30m", think=False)
MINIMAL = ModelSpec(provider="ollama", model="gemma4:e2b")
CLOUD = ModelSpec(provider="ollama", base_url="https://ollama.com", model="gemma4:31b",
                  api_key_env="OLLAMA_API_KEY")


def mock(handler):
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def recorder(status=200, payload=None):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(status, json=payload)

    return handler, seen


def payload(content="ok", done_reason="stop", thinking=None):
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {"model": "m", "created_at": "2026-01-01T00:00:00Z", "message": message, "done": True,
            "done_reason": done_reason, "prompt_eval_count": 21, "eval_count": 8}


def body(request):
    return json.loads(request.content)


def client(spec, handler, **kwargs):
    return ModelClient("ollama", spec, http_client=mock(handler), retry_delay=0, **kwargs)


# --- request shape ---------------------------------------------------------

def test_structured_turn_request_shape():
    handler, seen = recorder(payload=payload('{"reply": "quack", "hint_level": 1}'))
    out = client(LOCAL, handler).complete("SYS", USER)

    assert out.reply == "quack" and out.hint_level == 1
    req = seen[0]
    assert str(req.url) == "http://localhost:11434/api/chat"
    assert "authorization" not in req.headers          # local server: no key
    sent = body(req)
    assert sent["model"] == "gemma4:e2b" and sent["stream"] is False
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert sent["options"] == {"num_predict": 512, "num_ctx": 8192}
    assert sent["keep_alive"] == "30m" and sent["think"] is False
    assert sent["format"] == RESPONSE_JSON_SCHEMA       # schema-constrained JSON


def test_summary_call_does_not_request_json_format():
    handler, seen = recorder(payload=payload("a plain summary"))
    assert client(LOCAL, handler).complete_text("s", USER) == "a plain summary"
    assert "format" not in body(seen[0])


def test_optional_fields_are_omitted_when_unset():
    handler, seen = recorder(payload=payload())
    client(MINIMAL, handler).complete_text("s", USER)
    sent = body(seen[0])
    assert sent["options"] == {"num_predict": 4096}     # no num_ctx forced
    assert "keep_alive" not in sent and "think" not in sent


def test_think_accepts_model_specific_string():
    spec = ModelSpec(provider="ollama", model="gpt-oss", think="low")
    handler, seen = recorder(payload=payload())
    client(spec, handler).complete_text("s", USER)
    assert body(seen[0])["think"] == "low"


@pytest.mark.parametrize("base", ["http://localhost:11434/v1", "http://localhost:11434/v1/",
                                  "http://localhost:11434/api", "http://localhost:11434/"])
def test_openai_style_base_urls_are_normalized(base):
    handler, seen = recorder(payload=payload())
    spec = MINIMAL.model_copy(update={"base_url": base})
    client(spec, handler).complete_text("s", USER)
    assert str(seen[0].url) == "http://localhost:11434/api/chat"


def test_consecutive_user_turns_are_merged():
    handler, seen = recorder(payload=payload())
    client(MINIMAL, handler).complete_text(
        "s", [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    )
    assert [m["content"] for m in body(seen[0])["messages"][1:]] == ["one\n\ntwo"]


# --- response handling -----------------------------------------------------

def test_usage_and_thinking_trace_ignored():
    handler, _ = recorder(payload=payload("the answer", thinking="long private reasoning"))
    c = client(LOCAL, handler)
    assert c.complete_text("s", USER) == "the answer"
    assert c.last_usage == (21, 8)


def test_empty_reply_at_token_limit_is_actionable():
    handler, _ = recorder(payload=payload("", done_reason="length"))
    with pytest.raises(LLMError, match="token limit.*models.toml"):
        client(LOCAL, handler).complete("s", USER)


def test_http_200_with_error_body():
    handler, _ = recorder(payload={"error": "the model failed to generate a response"})
    with pytest.raises(LLMError, match="failed to generate"):
        client(LOCAL, handler).complete("s", USER)


# --- failures --------------------------------------------------------------

def test_missing_model_says_how_to_pull_it():
    handler, _ = recorder(404, {"error": "model 'gemma4:e2b' not found"})
    with pytest.raises(LLMError, match="ollama pull gemma4:e2b"):
        client(LOCAL, handler).complete("s", USER)


def test_server_not_running_explains_and_does_not_retry():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx2.ConnectError("connection refused")

    with pytest.raises(LLMError, match="Is it running.*ollama serve"):
        client(LOCAL, handler).complete("s", USER)
    assert len(calls) == 1


def test_cold_start_timeout_points_at_timeout_setting():
    def handler(request):
        raise httpx2.ReadTimeout("slow model load")

    spec = LOCAL.model_copy(update={"timeout": 45})
    with pytest.raises(LLMError, match="45s.*first request.*raise `timeout`"):
        client(spec, handler).complete("s", USER)


def test_transient_5xx_is_retried_once():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(503, json={"error": "busy"})
        return httpx2.Response(200, json=payload("recovered"))

    assert client(LOCAL, handler).complete_text("s", USER) == "recovered" and len(calls) == 2


def test_bad_request_surfaces_ollamas_message():
    handler, seen = recorder(400, {"error": "unsupported option foo"})
    with pytest.raises(LLMError, match="unsupported option foo"):
        client(LOCAL, handler).complete("s", USER)
    assert len(seen) == 1


# --- cloud -----------------------------------------------------------------

def test_cloud_sends_bearer_token_and_scrubs_it_from_errors(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ol-secret")
    handler, seen = recorder(payload=payload("hello cloud"))
    assert client(CLOUD, handler).complete_text("s", USER) == "hello cloud"
    assert seen[0].url.host == "ollama.com" and seen[0].url.path == "/api/chat"
    assert seen[0].headers["authorization"] == "Bearer ol-secret"

    handler, _ = recorder(401, {"error": "invalid key ol-secret"})
    with pytest.raises(LLMError) as exc:
        client(CLOUD, handler).complete("s", USER)
    assert "rejected the API key" in str(exc.value) and "ol-secret" not in str(exc.value)


def test_cloud_without_key_is_a_friendly_error():
    with pytest.raises(LLMError, match="OLLAMA_API_KEY.*set-key"):
        ModelClient("ollama-cloud", CLOUD).complete("s", USER)


# --- schema + other providers ----------------------------------------------

def test_response_schema_matches_the_model():
    props = RESPONSE_JSON_SCHEMA["properties"]
    assert set(props) == set(DuckyResponse.model_fields)
    assert set(RESPONSE_JSON_SCHEMA["required"]) <= set(props)
    assert props["style"]["enum"] == ["socratic", "hint"]
    sample = {"reply": "r", "style": "hint", "hint_level": 2,
              "has_system_description": True, "has_problem": False}
    assert DuckyResponse.model_validate(sample).hint_level == 2


def test_other_providers_do_not_receive_the_ollama_format_field(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    spec = ModelSpec(provider="openai_compat", base_url="https://api.test/v1", model="m",
                     api_key_env="OPENAI_API_KEY")
    handler, seen = recorder(payload={
        "id": "c", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": '{"reply": "ok"}'}}],
    })
    ModelClient("chatgpt", spec, http_client=mock(handler)).complete("s", USER)
    sent = body(seen[0])
    assert "format" not in sent and "response_format" not in sent
