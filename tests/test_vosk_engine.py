import json

import pytest

from ducky.errors import InputUnavailable
from ducky.stt.vosk_engine import VoskEngine


class FakeRecognizer:
    def __init__(self, model, rate):
        self.model, self.rate = model, rate
        self.accepts = []          # what AcceptWaveform returns, in order
        self.partial = {"partial": ""}
        self.result = {"text": ""}
        self.final = {"text": ""}
        self.was_reset = False

    def AcceptWaveform(self, data):
        return self.accepts.pop(0)

    def PartialResult(self):
        return json.dumps(self.partial)

    def Result(self):
        return json.dumps(self.result)

    def FinalResult(self):
        return json.dumps(self.final)

    def Reset(self):
        self.was_reset = True


class FakeVosk:
    def __init__(self, fail_model=False):
        self.log_level = None
        self.fail_model = fail_model
        self.recognizer = None

    def SetLogLevel(self, level):
        self.log_level = level

    def Model(self, path):
        if self.fail_model:
            raise Exception("Failed to create a model")
        return ("model", path)

    def KaldiRecognizer(self, model, rate):
        self.recognizer = FakeRecognizer(model, rate)
        return self.recognizer


def test_setup_is_quiet_and_uses_the_given_rate():
    vosk = FakeVosk()
    VoskEngine("/models/m", 16000, vosk_module=vosk)
    assert vosk.log_level == -1
    assert vosk.recognizer.model == ("model", "/models/m") and vosk.recognizer.rate == 16000.0


def test_partial_and_final_results():
    vosk = FakeVosk()
    engine = VoskEngine("/m", vosk_module=vosk)
    rec = vosk.recognizer
    rec.accepts = [False, True]
    rec.partial = {"partial": "hello wor"}
    rec.result = {"text": "hello world"}

    first = engine.accept_audio(b"\x00" * 8)
    assert (first.text, first.is_final) == ("hello wor", False)
    second = engine.accept_audio(b"\x00" * 8)
    assert (second.text, second.is_final) == ("hello world", True)


def test_missing_keys_are_tolerated():
    vosk = FakeVosk()
    engine = VoskEngine("/m", vosk_module=vosk)
    vosk.recognizer.accepts = [False, True]
    vosk.recognizer.partial = {}
    vosk.recognizer.result = {}
    assert engine.accept_audio(b"").text == "" and engine.accept_audio(b"").text == ""
    assert engine.flush() == ""


def test_flush_and_reset():
    vosk = FakeVosk()
    engine = VoskEngine("/m", vosk_module=vosk)
    vosk.recognizer.final = {"text": "the rest of it"}
    assert engine.flush() == "the rest of it"
    engine.reset()
    assert vosk.recognizer.was_reset


def test_a_broken_model_points_at_the_fix():
    with pytest.raises(InputUnavailable) as exc:
        VoskEngine("/bad", vosk_module=FakeVosk(fail_model=True))
    assert "/bad" in str(exc.value) and "ducky stt download" in str(exc.value)


def test_the_real_vosk_library_rejects_a_missing_model_with_our_message():
    """No fakes: proves we translate Vosk's bare Exception("Failed to create a model")."""
    with pytest.raises(InputUnavailable) as exc:
        VoskEngine("/definitely/not/a/model")
    assert "ducky stt download" in str(exc.value) and "/definitely/not/a/model" in str(exc.value)
