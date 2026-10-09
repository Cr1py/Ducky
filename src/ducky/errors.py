"""Exceptions shared by the input and audio layers.

A leaf module (it imports nothing from Ducky) so low-level code like audio capture can
raise these without importing the `ducky.input` package, which would be a circular import.
"""

from __future__ import annotations


class InputUnavailable(Exception):
    """The input source can't be used right now (no mic, no model, not built yet, etc)."""


class EmptyTranscript(Exception):
    """The source produced no usable text."""
