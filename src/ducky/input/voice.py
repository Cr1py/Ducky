"""Voice input: microphone -> streaming speech-to-text -> end of turn.

A turn ends on whichever comes first:
  * you pause for `silence_seconds` after your last recognized word
  * the tail of what you said matches an end phrase (which is then stripped)
  * you press Enter, or Ctrl+C (what was captured so far is kept)

Silence is measured in *audio* time (chunks received), not wall-clock time, so a slow
machine can't make it fire early. Everything the loop needs is injected, so it is tested
without a microphone or a speech model.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Protocol

from ducky.audio import AudioSource, EnterWatcher, rms_level
from ducky.endphrase import strip_end_phrase
from ducky.errors import EmptyTranscript, InputUnavailable
from ducky.input.base import InputSource
from ducky.stt.base import STTEngine

NO_SPEECH_TIMEOUT = 30.0  # give up if nothing is said at all
STALL_TIMEOUT = 5.0       # wall-clock seconds without any audio = the device died
READ_TIMEOUT = 0.1        # short, so Enter and Ctrl+C stay responsive


class Display(Protocol):
    def start(self, silence_seconds: float) -> None: ...
    def update(self, text: str, level: float, silence_left: float | None) -> None: ...
    def stop(self, final_text: str) -> None: ...


class NullDisplay:
    def start(self, silence_seconds: float) -> None: ...
    def update(self, text: str, level: float, silence_left: float | None) -> None: ...
    def stop(self, final_text: str) -> None: ...


def _join(finals: list[str], partial: str) -> str:
    return " ".join(part for part in (*finals, partial) if part)


class VoiceInput(InputSource):
    def __init__(
        self,
        engine: STTEngine,
        audio: AudioSource,
        *,
        silence_seconds: float = 6.0,
        end_phrases: list[str] | None = None,
        no_speech_timeout: float = NO_SPEECH_TIMEOUT,
        stall_timeout: float = STALL_TIMEOUT,
        display: Display | None = None,
        enter: EnterWatcher | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self.audio = audio
        self.silence_seconds = silence_seconds
        self.end_phrases = end_phrases or []
        self.no_speech_timeout = no_speech_timeout
        self.stall_timeout = stall_timeout
        self.display: Display = display or NullDisplay()
        self.enter = enter or EnterWatcher()
        self._clock = clock

    def get_transcript(self) -> str:
        finals: list[str] = []   # utterances the recognizer has finalized
        partial = ""             # the utterance in progress (may still change)
        audio_time = 0.0
        last_activity = 0.0
        spoke = False
        last_chunk_at = self._clock()

        self.engine.reset()
        self.display.start(self.silence_seconds)
        try:
            with self.audio as source:
                rate = source.sample_rate
                while True:
                    chunk = source.read(READ_TIMEOUT)
                    now = self._clock()
                    if chunk is None:
                        if now - last_chunk_at >= self.stall_timeout:
                            raise InputUnavailable(
                                "The microphone stopped delivering audio. Check that it's "
                                "connected, or pick another input with `ducky mic list`."
                            )
                        if self.enter.pressed():
                            break
                        continue
                    last_chunk_at = now
                    audio_time += len(chunk) / (2 * rate)  # 16-bit mono

                    changed = False
                    result = self.engine.accept_audio(chunk)
                    if result is not None:
                        if result.is_final:
                            if result.text:
                                finals.append(result.text)
                                changed = True
                            partial = ""
                        elif result.text != partial:
                            partial = result.text
                            changed = True
                    if changed:
                        last_activity, spoke = audio_time, True

                    full = _join(finals, partial)
                    if changed and strip_end_phrase(full, self.end_phrases)[1]:
                        break
                    silence_left = (
                        max(0.0, self.silence_seconds - (audio_time - last_activity)) if spoke else None
                    )
                    self.display.update(full, rms_level(chunk), silence_left)

                    if spoke and audio_time - last_activity >= self.silence_seconds:
                        break
                    if not spoke and audio_time >= self.no_speech_timeout:
                        raise EmptyTranscript(
                            "I didn't hear anything. Check your microphone with `ducky mic list`."
                        )
                    if self.enter.pressed():
                        break
        except KeyboardInterrupt:
            pass  # Ctrl+C ends the turn; keep what was captured
        except Exception:
            self.display.stop("")
            raise

        pending = self.engine.flush()  # the utterance still in progress, finalized
        text, _ = strip_end_phrase(_join(finals, pending or partial), self.end_phrases)
        text = text.strip()
        self.display.stop(text)
        if not text:
            raise EmptyTranscript("I didn't catch anything.")
        return text


def _require_model(name_or_path: str) -> Path:
    from ducky.stt.models import resolve_model

    path = resolve_model(name_or_path)
    if path is None:
        raise InputUnavailable(
            f"The speech model '{name_or_path}' isn't installed. "
            "Download it once with `ducky stt download`."
        )
    return path


def build_voice_input(
    cfg: dict[str, Any],
    end_phrases: list[str],
    *,
    display: Display | None = None,
    silence_seconds: float | None = None,
    ensure_model: Callable[[str], Path] | None = None,
) -> VoiceInput:
    """Wire up the real microphone and Vosk. `ensure_model` lets the CLI offer a download."""
    from ducky.audio import SAMPLE_RATE, MicSource, parse_device
    from ducky.stt.vosk_engine import VoskEngine

    audio = MicSource(device=parse_device(cfg.get("mic", "")), sample_rate=SAMPLE_RATE)
    audio.check()  # fail fast, before the slower model load
    model_path = (ensure_model or _require_model)(cfg["vosk_model"])
    engine = VoskEngine(model_path, SAMPLE_RATE)
    return VoiceInput(
        engine,
        audio,
        silence_seconds=silence_seconds if silence_seconds is not None else cfg["silence"],
        end_phrases=end_phrases,
        display=display,
    )
