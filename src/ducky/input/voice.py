from __future__ import annotations

from ducky.input.base import InputSource, InputUnavailable


class VoiceInput(InputSource):
    """sounddevice -> STTEngine -> silence/end-phrase/Enter

    Planned flow:
      1. open a mono 16 kHz stream, calibrate ambient noise
      2. feed chunks to the STTEngine, render partials live (Rich)
      3. stop on silence > cfg["silence"], end-phrase tail match, or Enter
      4. strip the end phrase, return the final transcript
    """

    def __init__(
        self, silence_seconds: float = 6.0, end_phrases: list[str] | None = None
    ):
        self.silence_seconds = silence_seconds
        self.end_phrases = end_phrases or []

    def get_transcript(self) -> str:
        raise InputUnavailable('Use --text "your thoughts" or pipe text in for now.')
