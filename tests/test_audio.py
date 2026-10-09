import array
import sys
from types import SimpleNamespace

import pytest

from ducky.audio import EnterWatcher, MicSource, list_input_devices, parse_device, rms_level
from ducky.errors import InputUnavailable


class FakeStream:
    def __init__(self):
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSD:
    class PortAudioError(Exception):
        pass

    def __init__(self, no_input=False, fail_open=False, devices=None):
        self.no_input, self.fail_open = no_input, fail_open
        self.devices = devices or []
        self.stream = FakeStream()
        self.stream_kwargs = None
        self.default = SimpleNamespace(device=(1, 3))

    def query_devices(self, device=None, kind=None):
        if kind == "input":
            if self.no_input:
                raise ValueError("No input device matching")
            return {"name": "Fake Mic", "max_input_channels": 1}
        return self.devices

    def RawInputStream(self, **kwargs):
        self.stream_kwargs = kwargs
        if self.fail_open:
            raise self.PortAudioError("Invalid sample rate")
        return self.stream


OK = SimpleNamespace(input_overflow=False)


def test_parse_device():
    assert parse_device("") is None and parse_device("   ") is None
    assert parse_device("3") == 3
    assert parse_device("USB Microphone") == "USB Microphone"


def test_rms_level():
    assert rms_level(b"") == 0.0
    assert rms_level(b"\x00\x00" * 100) == 0.0
    loud = array.array("h", [32767, -32768] * 50).tobytes()
    assert rms_level(loud) == pytest.approx(1.0, abs=0.01)
    quiet = array.array("h", [327, -327] * 50).tobytes()
    assert 0.0 < rms_level(quiet) < 0.02
    assert rms_level(b"\x00\x00\x01") == 0.0  # a stray odd byte is ignored, not an error


def test_no_microphone_is_a_friendly_error():
    with pytest.raises(InputUnavailable) as exc:
        MicSource(sd_module=FakeSD(no_input=True)).check()
    assert "No microphone found" in str(exc.value) and "ducky mic list" in str(exc.value)


def test_open_failure_is_a_friendly_error():
    with pytest.raises(InputUnavailable) as exc:
        with MicSource(sd_module=FakeSD(fail_open=True)):
            pass
    assert "Couldn't open the microphone" in str(exc.value) and "Privacy" in str(exc.value)


def test_streams_chunks_and_cleans_up():
    sd = FakeSD()
    mic = MicSource(device=2, sd_module=sd)
    with mic:
        assert sd.stream.started
        assert sd.stream_kwargs == {
            "samplerate": 16000, "blocksize": 4000, "device": 2,
            "channels": 1, "dtype": "int16", "callback": mic._callback,
        }
        assert mic.read(timeout=0.01) is None            # nothing yet
        mic._callback(b"\x01\x00" * 4, 4, None, OK)      # PortAudio's thread delivers audio
        assert mic.read(timeout=0.01) == b"\x01\x00" * 4
        mic._callback(b"\x00\x00", 1, None, SimpleNamespace(input_overflow=True))
        assert mic.overflows == 1
    assert sd.stream.stopped and sd.stream.closed


def test_exiting_never_masks_an_error_from_a_broken_stream():
    sd = FakeSD()
    sd.stream.stop = lambda: (_ for _ in ()).throw(RuntimeError("device vanished"))
    with MicSource(sd_module=sd):
        pass  # no exception escapes


def test_list_input_devices_skips_outputs_and_marks_the_default():
    sd = FakeSD(devices=[
        {"name": "Speakers", "max_input_channels": 0},
        {"name": "Built-in Mic", "max_input_channels": 2},
        {"name": "Headset", "max_input_channels": 1},
    ])
    rows = list_input_devices(sd)
    assert [r["name"] for r in rows] == ["Built-in Mic", "Headset"]
    assert [r["default"] for r in rows] == [True, False]   # default input index is 1
    assert rows[0]["index"] == 1 and rows[0]["channels"] == 2


def test_enter_watcher_is_quiet_when_nothing_is_typed(monkeypatch):
    if sys.platform != "win32":
        import io

        monkeypatch.setattr(sys, "stdin", io.StringIO())  # no real file descriptor to poll
    assert EnterWatcher().pressed() is False
