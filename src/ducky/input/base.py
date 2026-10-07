"""Every way of getting a user's thoughts implements InputSource.

The turn runner only ever sees a transcript string, so voice and text are
interchangeable (and tests never need a microphone).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class InputUnavailable(Exception):
    """The input source can't be used right now (no mic, not built yet, ...)."""


class EmptyTranscript(Exception):
    """The source produced no usable text."""


class InputSource(ABC):
    @abstractmethod
    def get_transcript(self) -> str:
        """Block until the user's turn is complete and return the transcript."""
