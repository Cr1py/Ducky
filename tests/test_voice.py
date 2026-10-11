"""The voice loop, tested with a fake microphone and a fake speech engine (no hardware)."""
import itertools

import pytest

from ducky.errors import EmptyTranscript, InputUnavailable
from ducky.input.voice import VoiceInput, build_voice_input
from ducky.stt.base import STTResult

CHUNK = b"\x00\x00" * 4000  # 0.25 s of silence at 16 kHz, 16-bit mono
PHRASES = ["have any thoughts ducky"]


def partial(text):
    return STTResult(text, False)


def final(text):
    return STTResult(text, True)


class FakeAudio:
    sample_rate = 16000

    def __init__(self, n_chunks=40):
        self.chunks = [CHUNK] * n_chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def read(self, timeout):
        return self.chunks.pop(0) if self.chunks else None


class InterruptingAudio(FakeAudio):
    """Delivers `after` chunks, then the user hits Ctrl+C."""

    def __init__(self, after):
        super().__init__(after)

    def read(self, timeout):
        if not self.chunks:
            raise KeyboardInterrupt
        return self.chunks.pop(0)


class FakeEngine:
    def __init__(self, script, flush_text=""):
        self.script, self.flush_text = script, flush_text
        self.calls = 0
        self.resets = 0

    def accept_audio(self, chunk):
        result = self.script.get(self.calls)
        self.calls += 1
        return result

    def flush(self):
        return self.flush_text

    def reset(self):
        self.resets += 1


class NoEnter:
    def pressed(self):
        return False


class EnterAfter:
    def __init__(self, n):
        self.n, self.calls = n, 0

    def pressed(self):
        self.calls += 1
        return self.calls >= self.n


class Recorder:
    def __init__(self):
        self.started = None
        self.updates = []
        self.stopped = None

    def start(self, silence_seconds):
        self.started = silence_seconds

    def update(self, text, level, silence_left):
        self.updates.append((text, level, silence_left))

    def stop(self, final_text):
        self.stopped = final_text


def make(engine, audio=None, **kw):
    kw.setdefault("enter", NoEnter())
    return VoiceInput(engine, audio or FakeAudio(), **kw)


# --- how a turn ends ---------------------------------------------------------

def test_ends_after_the_configured_silence():
    engine = FakeEngine({0: partial("my loop"), 1: partial("my loop never ends")}, "my loop never ends")
    audio = FakeAudio()
    out = make(engine, audio, silence_seconds=1.0).get_transcript()
    assert out == "my loop never ends"
    # last words at 0.5 s of audio; 1.0 s of silence later (1.5 s) = the 6th chunk
    assert engine.calls == 6
    assert audio.closed and engine.resets == 1


def test_finalized_utterances_and_the_one_in_progress_are_joined_without_duplicates():
    engine = FakeEngine(
        {0: partial("hello"), 1: final("hello there"), 2: partial("second"), 3: partial("second part")},
        flush_text="second part",
    )
    assert make(engine, silence_seconds=1.0).get_transcript() == "hello there second part"


def test_a_pause_shorter_than_the_limit_does_not_end_the_turn():
    script = {0: partial("one"), 4: partial("one two")}  # 0.75 s gap, limit is 1.0 s
    engine = FakeEngine(script, "one two")
    assert make(engine, silence_seconds=1.0).get_transcript() == "one two"
    assert engine.calls > 5  # it kept listening past the gap


def test_enter_ends_the_turn_and_keeps_the_text():
    engine = FakeEngine({0: partial("almost done")}, "almost done")
    out = make(engine, enter=EnterAfter(3), silence_seconds=60).get_transcript()
    assert out == "almost done" and engine.calls == 3


def test_end_phrase_in_a_partial_result_ends_the_turn_and_is_stripped():
    engine = FakeEngine(
        {0: partial("my loop never ends"), 1: partial("my loop never ends have any thoughts ducky")},
        "my loop never ends have any thoughts ducky",
    )
    out = make(engine, end_phrases=PHRASES, silence_seconds=60).get_transcript()
    assert out == "my loop never ends"
    assert engine.calls == 2  # stopped immediately, not after the silence timeout


def test_an_alias_heard_by_the_recognizer_works_like_the_phrase():
    engine = FakeEngine({0: partial("so what do you think have any thoughts ducking")},
                        "so what do you think have any thoughts ducking")
    out = make(engine, end_phrases=PHRASES + ["have any thoughts ducking"], silence_seconds=60).get_transcript()
    assert out == "so what do you think"


def test_an_end_phrase_in_the_middle_of_a_sentence_is_kept():
    text = "have any thoughts ducky is the phrase i will use later"
    engine = FakeEngine({0: partial(text)}, text)
    assert make(engine, end_phrases=PHRASES, silence_seconds=1.0).get_transcript() == text


def test_saying_only_the_end_phrase_is_an_empty_turn():
    engine = FakeEngine({0: partial("have any thoughts ducky")}, "have any thoughts ducky")
    with pytest.raises(EmptyTranscript):
        make(engine, end_phrases=PHRASES, silence_seconds=60).get_transcript()


# --- nothing said, interruptions, dead devices ----------------------------------

def test_no_speech_at_all_times_out():
    display = Recorder()
    with pytest.raises(EmptyTranscript, match="didn't hear anything"):
        make(FakeEngine({}), no_speech_timeout=2.0, display=display).get_transcript()
    assert display.stopped == ""  # the live view is always closed


def test_ctrl_c_keeps_what_was_captured():
    engine = FakeEngine({0: partial("half a thought")}, "half a thought")
    out = make(engine, InterruptingAudio(after=3), silence_seconds=60).get_transcript()
    assert out == "half a thought"


def test_ctrl_c_with_nothing_captured_is_an_empty_turn():
    with pytest.raises(EmptyTranscript):
        make(FakeEngine({}), InterruptingAudio(after=2), silence_seconds=60).get_transcript()


def test_a_stalled_microphone_is_reported():
    audio = FakeAudio(n_chunks=0)
    ticking = itertools.count()
    with pytest.raises(InputUnavailable, match="stopped delivering audio"):
        make(FakeEngine({}), audio, stall_timeout=3.0, clock=lambda: next(ticking)).get_transcript()
    assert audio.closed  # the stream is released even on failure


# --- what the user sees --------------------------------------------------------

def test_display_gets_progress_and_a_final_summary():
    display = Recorder()
    engine = FakeEngine({1: partial("hello there")}, "hello there")
    make(engine, display=display, silence_seconds=1.0).get_transcript()

    assert display.started == 1.0
    assert display.updates[0][2] is None          # before speech: no countdown
    spoken = [u for u in display.updates if u[0] == "hello there"]
    assert spoken and spoken[0][2] == pytest.approx(1.0)   # countdown starts at the full limit
    assert spoken[-1][2] < spoken[0][2]                    # and runs down during silence
    assert display.stopped == "hello there"


# --- wiring the real parts ---------------------------------------------------------

class _FakeMic:
    instances = []

    def __init__(self, **kw):
        self.kw = kw
        _FakeMic.instances.append(self)

    def check(self):
        pass


class _FakeVosk:
    def __init__(self, model_path, rate):
        self.model_path, self.rate = model_path, rate


def _cfg(**over):
    return {"mic": "", "vosk_model": "vosk-model-small-en-us-0.15", "silence": 6.0, **over}


def test_build_requires_an_installed_model(monkeypatch):
    monkeypatch.setattr("ducky.audio.MicSource", _FakeMic)
    with pytest.raises(InputUnavailable, match="ducky stt download"):
        build_voice_input(_cfg(), [])


def test_build_wires_config_into_the_loop(monkeypatch):
    from ducky.stt.models import models_dir

    (models_dir() / "vosk-model-small-en-us-0.15" / "am").mkdir(parents=True)
    _FakeMic.instances.clear()
    monkeypatch.setattr("ducky.audio.MicSource", _FakeMic)
    monkeypatch.setattr("ducky.stt.vosk_engine.VoskEngine", _FakeVosk)

    voice = build_voice_input(_cfg(mic="2", silence=4.5), PHRASES)
    assert isinstance(voice, VoiceInput)
    assert voice.silence_seconds == 4.5 and voice.end_phrases == PHRASES
    assert voice.engine.rate == 16000 and voice.engine.model_path.name == "vosk-model-small-en-us-0.15"
    assert _FakeMic.instances[0].kw["device"] == 2  # "2" in config means device index 2

    override = build_voice_input(_cfg(), [], silence_seconds=3.0)
    assert override.silence_seconds == 3.0
