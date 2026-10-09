"""Microphone capture and keyboard polling.

sounddevice (PortAudio) is imported lazily: a missing PortAudio becomes a friendly
error at use time instead of breaking `ducky --help`.
"""

from __future__ import annotations

import math
import queue
import sys
from array import array
from typing import Any, Protocol

from ducky.errors import InputUnavailable

SAMPLE_RATE = 16000  # what Vosk's models expect
BLOCK_SECONDS = 0.25  # audio handed to the recognizer in 250 ms chunks

_MIC_HELP = (
    "Run `ducky mic list` to see your inputs and `ducky config set mic <index>` to pick one. "
    "Check your device settings for microphone access."
)


class AudioSource(Protocol):
    sample_rate: int

    def __enter__(self) -> "AudioSource": ...
    def __exit__(self, *exc: object) -> None: ...
    def read(self, timeout: float) -> bytes | None:
        """Next chunk of 16-bit mono PCM, or None if nothing arrived within `timeout`."""


def parse_device(setting: str) -> int | str | None:
    """'' -> default device, '3' -> index 3, anything else -> a name to match."""
    setting = (setting or "").strip()
    if not setting:
        return None
    return int(setting) if setting.isdigit() else setting


def rms_level(chunk: bytes) -> float:
    """Loudness of a 16-bit PCM chunk, 0.0 (silence) to ~1.0 (full scale)."""
    samples = array("h")
    samples.frombytes(chunk[: len(chunk) - (len(chunk) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0


def _load_sounddevice() -> Any:
    try:
        import sounddevice
    except OSError as e:  # PortAudio shared library missing
        hint = (
            " On Linux: `sudo apt install libportaudio2`."
            if sys.platform.startswith("linux")
            else ""
        )
        raise InputUnavailable(f"PortAudio couldn't be loaded ({e}).{hint}") from e
    except ImportError as e:  # pragma: no cover
        raise InputUnavailable(
            "The `sounddevice` package isn't installed. Run `uv sync`."
        ) from e
    return sounddevice


class MicSource:
    """Streams microphone audio into a queue from PortAudio's callback thread."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        block_seconds: float = BLOCK_SECONDS,
        sd_module: Any = None,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._block = int(sample_rate * block_seconds)
        self._sd = sd_module
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stream: Any = None
        self.overflows = 0

    def _sounddevice(self) -> Any:
        if self._sd is None:
            self._sd = _load_sounddevice()
        return self._sd

    def check(self) -> None:
        """Fail fast (before loading the speech model) if there's no usable input device."""
        sd = self._sounddevice()
        try:
            sd.query_devices(self.device, "input")
        except (ValueError, sd.PortAudioError) as e:
            raise InputUnavailable(f"No microphone found ({e}). {_MIC_HELP}") from e

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if getattr(status, "input_overflow", False):
            self.overflows += 1
        self._queue.put(bytes(indata))

    def __enter__(self) -> "MicSource":
        self.check()
        sd = self._sounddevice()
        self._queue = queue.Queue()
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self._block,
                device=self.device,
                channels=1,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            self._stream = None
            raise InputUnavailable(
                f"Couldn't open the microphone: {e}. {_MIC_HELP}"
            ) from e
        return self

    def __exit__(self, *exc: object) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # closing must never mask the real error
                pass

    def read(self, timeout: float) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def list_input_devices(sd_module: Any = None) -> list[dict[str, Any]]:
    sd = sd_module or _load_sounddevice()
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
    except sd.PortAudioError as e:
        raise InputUnavailable(f"Couldn't list audio devices ({e}).") from e
    rows = []
    for index, info in enumerate(devices):
        if info["max_input_channels"] > 0:
            rows.append(
                {
                    "index": index,
                    "name": info["name"],
                    "channels": info["max_input_channels"],
                    "default": index == default_input,
                }
            )
    return rows


class EnterWatcher:
    """Polls for the Enter key without threads, so nothing is left reading stdin afterwards."""

    def pressed(self) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                hit = False
                while msvcrt.kbhit():
                    if msvcrt.getwch() in ("\r", "\n"):
                        hit = True
                return hit
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                return True
        except (OSError, ValueError):  # stdin isn't a real terminal
            return False
        return False
