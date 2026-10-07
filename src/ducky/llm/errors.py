"""LLM errors and the mapping from SDK/HTTP failures to friendly messages."""
from __future__ import annotations


class LLMError(RuntimeError):
    pass


class APIStatusError(Exception):
    """Raised by our own HTTP adapter (Gemini) so it maps like the SDK errors."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def friendly_error(exc: Exception, secret: str | None = None, entry: str | None = None) -> str:
    """Turn an SDK/HTTP exception into an actionable one-liner.

    Maps by HTTP status (both SDKs and our Gemini adapter expose .status_code)
    and falls back to exception-name hints for timeouts/connection failures.
    """
    which = f"'{entry}'" if entry else "this model"
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)

    if status in (401, 403):
        cmd = f"`ducky config set-key {entry}`" if entry else "`ducky config set-key`"
        base = f"The provider rejected the API key. Update it with {cmd}."
    elif status == 404:
        base = f"Model or endpoint not found. Check `model` and `base_url` for {which} in models.toml."
    elif status == 429:
        base = "Rate limited or out of quota. Wait a moment, or check your billing."
    elif status == 400:
        base = (
            f"The provider rejected the request. A wrong `token_param`, `max_tokens` or "
            f"`model` for {which} in models.toml is a common cause."
        )
    elif status is not None and status >= 500:
        base = "The provider is having trouble. Try again shortly."
    elif "Timeout" in name:
        base = "The request timed out. Try again."
    elif "Connect" in name:
        base = "Couldn't reach the provider. Check your internet connection (and base_url for local servers)."
    else:
        base = f"LLM call failed ({name})."

    detail = (str(exc).strip().splitlines() or [""])[0]
    if secret:
        detail = detail.replace(secret, "***")
    return f"{base} [{detail[:200]}]" if detail else base
