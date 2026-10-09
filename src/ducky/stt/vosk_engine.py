"""Vosk: offline, streaming speech recognition (partial text appears as you speak)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ducky.errors import InputUnavailable
from ducky.stt.base import STTEngine, STTResult


def _import_vosk() -> Any:
    try:
        import vosk
    except ImportError as e:  # pragma: no cover
        raise InputUnavailable("The `vosk` package isn't installed. Run `uv sync`.") from e
    return vosk


class VoskEngine(STTEngine):
    def __init__(self, model_path: Path | str, sample_rate: int = 16000, vosk_module: Any = None):
        vosk = vosk_module or _import_vosk()
        vosk.SetLogLevel(-1)  # keep Kaldi's native logging out of the terminal
        try:
            model = vosk.Model(str(model_path))
            self._recognizer = vosk.KaldiRecognizer(model, float(sample_rate))
        except Exception as e:  # Vosk raises a bare Exception("Failed to create a model")
            raise InputUnavailable(
                f"Couldn't load the speech model at {model_path} ({e}). "
                "Re-download it with `ducky stt download`."
            ) from e

    def accept_audio(self, chunk: bytes) -> STTResult | None:
        if self._recognizer.AcceptWaveform(chunk):
            return STTResult(json.loads(self._recognizer.Result()).get("text", ""), True)
        return STTResult(json.loads(self._recognizer.PartialResult()).get("partial", ""), False)

    def flush(self) -> str:
        return json.loads(self._recognizer.FinalResult()).get("text", "")

    def reset(self) -> None:
        self._recognizer.Reset()
