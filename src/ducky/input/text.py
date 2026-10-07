from __future__ import annotations

import sys

from ducky.input.base import InputSource


class TextInput(InputSource):
    """Typed input: --text "...", piped stdin, or an interactive prompt"""

    def __init__(self, text: str | None = None) -> None:
        self._text = text

    def get_transcript(self) -> str:
        if self._text is not None:
            return self._text
        if not sys.stdin.isatty():
            return sys.stdin.read()
        return input("> ")
