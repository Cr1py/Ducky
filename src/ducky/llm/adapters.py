"""One adapter per wire format. Each knows how its API wants the system prompt,
the token limit, and the role names, and nothing else.

SDKs/HTTP libraries are imported lazily so `ducky --help` stays fast.

`json_schema` (optional) asks for schema-constrained output. Only Ollama can enforce it
today; the other adapters ignore it and rely on the prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from ducky.llm.errors import APIStatusError, LLMError
from ducky.registry import ModelSpec

SDK_RETRIES = 1  # explicit, rather than inheriting each SDK's default


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False  # hit the token limit (common with reasoning models)


class Adapter(Protocol):
    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult: ...


# --- shared plumbing for the adapters that speak raw HTTP (Gemini, Ollama) -----


def _error_message(response: Any) -> str:
    """Pull a message out of {"error": {"message": ...}} (Gemini) or {"error": "..."} (Ollama)."""
    try:
        err = response.json()["error"]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]
    if isinstance(err, dict):
        return str(err.get("message", err))
    return str(err)


def post_json(
    http: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    retry_delay: float,
    retryable: set[int],
    retry_on_connect_error: bool = True,
) -> dict[str, Any]:
    """POST JSON with one retry on transient failures; raise APIStatusError on HTTP errors.

    Connection-refused is not retried when `retry_on_connect_error` is False: for a local
    server it means "not running", and retrying only delays the explanation.
    """
    import httpx2

    for attempt in range(SDK_RETRIES + 1):
        last_try = attempt == SDK_RETRIES
        try:
            response = http.post(url, json=body, headers=headers)
        except httpx2.ConnectError:
            if last_try or not retry_on_connect_error:
                raise
            time.sleep(retry_delay)
            continue
        except httpx2.TransportError:
            if last_try:
                raise
            time.sleep(retry_delay)
            continue
        if response.status_code in retryable and not last_try:
            time.sleep(retry_delay)
            continue
        if response.status_code >= 400:
            raise APIStatusError(response.status_code, _error_message(response))
        return response.json()
    raise AssertionError("unreachable")  # pragma: no cover


# --- adapters ------------------------------------------------------------------


class OpenAICompatAdapter:
    """OpenAI and other OpenAI-style servers (LM Studio, llama.cpp, ...). System prompt = first message."""

    def __init__(
        self, spec: ModelSpec, api_key: str | None, http_client: Any = None
    ) -> None:
        import openai

        self.spec = spec
        self._client = openai.OpenAI(
            # The SDK insists on a non-empty key; local servers ignore it.
            api_key=api_key or "not-needed",
            base_url=spec.base_url,
            timeout=spec.timeout,
            max_retries=SDK_RETRIES,
            http_client=http_client,
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        response = self._client.chat.completions.create(
            model=self.spec.model,
            messages=[{"role": "system", "content": system}, *messages],
            **{self.spec.token_param: self.spec.max_tokens},
        )
        choice = response.choices[0]
        usage = response.usage
        return ChatResult(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            truncated=choice.finish_reason == "length",
        )


class AnthropicAdapter:
    """Claude. System prompt is a top-level field and max_tokens is mandatory."""

    def __init__(
        self, spec: ModelSpec, api_key: str | None, http_client: Any = None
    ) -> None:
        import anthropic

        self.spec = spec
        kwargs: dict[str, Any] = {}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=spec.timeout,
            max_retries=SDK_RETRIES,
            http_client=http_client,
            **kwargs,
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        response = self._client.messages.create(
            model=self.spec.model,
            max_tokens=self.spec.max_tokens,
            system=system,
            messages=messages,
        )
        text = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )
        usage = response.usage
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            truncated=response.stop_reason == "max_tokens",
        )


class GeminiAdapter:
    """Gemini via the stateless generateContent REST endpoint.

    Deliberately not the Interactions API: that stores interactions server-side by
    default, which clashes with Ducky keeping its history locally. Plain REST (no SDK)
    keeps dependencies down; the key travels in a header, never in the URL.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str | None,
        http_client: Any = None,
        retry_delay: float = 1.0,
    ) -> None:
        import httpx2

        self.spec = spec
        self._api_key = api_key or ""
        self._http = http_client or httpx2.Client(timeout=spec.timeout)
        self._retry_delay = retry_delay
        self._base = (spec.base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
            ],
            "generationConfig": {"maxOutputTokens": self.spec.max_tokens},
        }
        data = post_json(
            self._http,
            f"{self._base}/models/{self.spec.model}:generateContent",
            {"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            body,
            retry_delay=self._retry_delay,
            retryable=self.RETRYABLE,
        )

        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(
                f"Gemini blocked the prompt ({reason})."
                if reason
                else "Gemini returned no answer."
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata") or {}
        return ChatResult(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0) or 0,
            output_tokens=usage.get("candidatesTokenCount", 0) or 0,
            truncated=candidate.get("finishReason") == "MAX_TOKENS",
        )


class OllamaAdapter:
    """Ollama's native /api/chat (local server or Ollama Cloud).

    Native rather than the OpenAI-compatible /v1 endpoint because /v1 cannot set the context
    window (Ollama then silently truncates the prompt to its default, 4k on GPUs under 24 GiB)
    or keep_alive, and cannot enforce a JSON schema. The key is optional: local servers need
    none; Ollama Cloud (base_url = https://ollama.com) takes a bearer token.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    RETRYABLE = {429, 500, 502, 503, 504}  # 502 = a cloud model couldn't be reached

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str | None,
        http_client: Any = None,
        retry_delay: float = 1.0,
    ) -> None:
        import httpx2

        self.spec = spec
        self._api_key = api_key
        self._http = http_client or httpx2.Client(timeout=spec.timeout)
        self._retry_delay = retry_delay
        self._base = self._normalize_base(spec.base_url or self.DEFAULT_BASE_URL)

    @staticmethod
    def _normalize_base(url: str) -> str:
        """Tolerate an OpenAI-style URL pasted from elsewhere (…/v1 or …/api)."""
        url = url.rstrip("/")
        for suffix in ("/v1", "/api"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        return url

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        import httpx2

        options: dict[str, Any] = {"num_predict": self.spec.max_tokens}
        if self.spec.num_ctx:
            options["num_ctx"] = self.spec.num_ctx
        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            body["format"] = json_schema
        if self.spec.keep_alive is not None:
            body["keep_alive"] = self.spec.keep_alive
        if self.spec.think is not None:
            body["think"] = self.spec.think

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            data = post_json(
                self._http,
                f"{self._base}/api/chat",
                headers,
                body,
                retry_delay=self._retry_delay,
                retryable=self.RETRYABLE,
                retry_on_connect_error=False,
            )
        except httpx2.ConnectError as e:
            raise LLMError(
                f"Couldn't reach Ollama at {self._base}. Is it running? "
                "Open the Ollama app or run `ollama serve`."
            ) from e
        except httpx2.TimeoutException as e:
            raise LLMError(
                f"Ollama didn't answer within {self.spec.timeout:.0f}s. The first request after a "
                "pause loads the model into memory and can be slow; raise `timeout` for this "
                "model in models.toml."
            ) from e
        except APIStatusError as e:
            if e.status_code == 404:
                raise LLMError(
                    f"Ollama doesn't have the model '{self.spec.model}'. For a local server run "
                    f"`ollama pull {self.spec.model}`; otherwise check `model` in models.toml."
                ) from e
            raise

        message = data.get("message")
        if message is None and data.get("error"):
            raise LLMError(f"Ollama error: {data['error']}")
        message = message or {}
        return ChatResult(
            text=message.get("content")
            or "",  # `thinking` is a separate field and is ignored
            input_tokens=data.get("prompt_eval_count") or 0,
            output_tokens=data.get("eval_count") or 0,
            truncated=data.get("done_reason") == "length",
        )


ADAPTERS: dict[str, type] = {
    "openai_compat": OpenAICompatAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "ollama": OllamaAdapter,
}
