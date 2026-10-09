"""Interface every speech-to-text engine implements (Vosk, Whisper, etc.)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class STTResult:
    text: str
    is_final: bool  # False = user still yapping/ text may still change


class STTEngine(ABC):
    @abstractmethod
    def accept_audio(self, chunk: bytes) -> STTResult | None:
        """Feed a chunk of 16 kHz mono PCM; return a result if there is one."""

    @abstractmethod
    def flush(self) -> str:
        """Finish the utterance in progress and return its text ('' if none)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state between turns."""
