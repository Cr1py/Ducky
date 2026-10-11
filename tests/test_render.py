import io

from rich.console import Console

from ducky import render


def capture(monkeypatch, width=80):
    buffer = io.StringIO()
    monkeypatch.setattr(render, "console", Console(file=buffer, width=width))
    return buffer


def test_level_bar_grows_with_loudness_and_is_clamped():
    silent, medium, loud, absurd = (render._level_bar(x) for x in (0.0, 0.05, 0.12, 5.0))
    assert silent.count("█") == 0 and len(silent) == 20
    assert 0 < medium.count("█") < loud.count("█") <= 20
    assert absurd == "█" * 20


def test_live_display_lifecycle_ends_with_what_was_heard(monkeypatch):
    out = capture(monkeypatch)
    display = render.LiveDisplay('Finish by pausing 6s, or saying "have any thoughts ducky".')
    display.start(6.0)
    display.update("", 0.0, None)
    display.update("my loop never ends", 0.08, 4.0)
    display.stop("my loop never ends")
    text = out.getvalue()
    assert "Heard:" in text and "my loop never ends" in text


def test_live_display_is_safe_to_misuse(monkeypatch):
    capture(monkeypatch)
    display = render.LiveDisplay("hint")
    display.update("before start", 0.0, None)   # no-op, not a crash
    display.stop("never started")               # no-op
    display.start(1.0)
    display.stop("")
    display.stop("twice")                        # second stop is harmless


def test_live_view_shows_countdown_only_after_speech(monkeypatch):
    capture(monkeypatch)
    display = render.LiveDisplay("hint")
    quiet = "".join(str(t) for t in display._view("", 0.0, None).renderables)
    talking = "".join(str(t) for t in display._view("hi", 0.1, 3.0).renderables)
    assert "ending in" not in quiet and "ending in 3s" in talking


def test_live_widgets_never_replace_stdout(monkeypatch):
    """Rich's stdout proxy can outlive the widget (a library imported meanwhile may keep it),
    which printed a traceback at shutdown on Windows. We must never install one."""
    import sys

    monkeypatch.setattr(render, "console", Console(file=io.StringIO(), force_terminal=True))
    real = sys.stdout

    with render.working("Loading speech model..."):
        assert sys.stdout is real
    display = render.LiveDisplay("hint")
    display.start(6.0)
    display.update("hello", 0.1, 3.0)
    assert sys.stdout is real
    display.stop("hello")
    with render.download_progress() as progress:
        progress.add_task("dl", total=10)
        assert sys.stdout is real
    assert sys.stdout is real


def test_spinner_prints_nothing_without_a_terminal(monkeypatch):
    buffer = capture(monkeypatch)
    with render.working("Ducky is thinking..."):
        pass
    assert buffer.getvalue() == ""
