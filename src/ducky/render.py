"""All terminal output lives here so the rest of the code stays testable."""
from __future__ import annotations

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

from ducky.sessions import Session, Turn

console = Console()
err_console = Console(stderr=True)


def error(msg: str) -> None:
    err_console.print(Text(f"Error: {msg}", style="bold red"))


def working(message: str):
    """Spinner context manager (prints nothing when not attached to a terminal).

    Built on Live directly (not console.status) so we can switch off stdout/stderr
    redirection. With redirection on, Rich replaces sys.stdout with a proxy while the spinner
    runs, and a library imported during that time (colorama, via tqdm, via vosk) can keep a
    reference to the proxy, which then gets finalized, noisily, at interpreter shutdown.
    """
    return Live(
        Spinner("dots", text=Text(message, style="dim")),
        console=console,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    )


def info(msg: str) -> None:
    console.print(Text(msg))


def dim(msg: str) -> None:
    console.print(Text(msg, style="dim"))


def session_header(session: Session) -> None:
    console.print(Text(f"Session: {session.name}  ({session.phase})", style="bold cyan"))


def duck_says(text: str, hint_level: int | None = None) -> None:
    subtitle = f"hint level {hint_level}" if hint_level is not None else None
    console.print(
        Panel(Text(text), title="Ducky", title_align="left", subtitle=subtitle,
              subtitle_align="right", border_style="yellow")
    )


def choices_footer() -> None:
    console.print(Text("What next?", style="bold"))
    console.print(Text("  1. Keep going        ducky thoughts"))
    console.print(Text("  2. New problem       ducky thoughts --new"))
    console.print(Text("  3. Switch session    ducky thoughts --session NAME"))
    console.print(Text("  4. Finish up         ducky end"))


def sessions_table(sessions: list[Session], active_id: int | None) -> None:
    table = Table(title="Sessions")
    table.add_column("")
    table.add_column("Name")
    table.add_column("Phase")
    table.add_column("Last active")
    for s in sessions:
        table.add_row("*" if s.id == active_id else "", s.name, s.phase, s.last_active)
    console.print(table)


def models_table(rows: list[tuple[str, str, str, str, str]], active: str) -> None:
    table = Table(title="Models")
    for column in ("", "Name", "Provider", "Model", "Key variable", "Key"):
        table.add_column(column, overflow="fold")  # wrap long IDs instead of truncating them
    for name, provider, model, env_var, status in rows:
        table.add_row("*" if name == active else "", name, provider, model, env_var, status)
    console.print(table)


def history_view(session: Session, turns: list[Turn]) -> None:
    console.print(Text(f"History: {session.name}", style="bold cyan"))
    if session.system_description:
        console.print(Panel(Text(session.system_description), title="System"))
    if session.problem_statement:
        console.print(Panel(Text(session.problem_statement), title="Problem"))
    if session.summary:
        console.print(Panel(Text(session.summary), title="Summary of earlier turns"))
    if not turns:
        dim("No turns yet.")
    for t in turns:
        who = "You" if t.role == "user" else "Ducky"
        style = "green" if t.role == "user" else "yellow"
        console.print(Text(f"{who}:", style=f"bold {style}"))
        console.print(Text(t.text))


# --- voice -------------------------------------------------------------------

def _level_bar(level: float, width: int = 20) -> str:
    filled = min(width, int(level * 8 * width))  # speech RMS is small; scale it up to be visible
    return "█" * filled + "░" * (width - filled)


class LiveDisplay:
    """The live 'Listening...' view: level meter, countdown, and the transcript so far."""

    def __init__(self, hint: str) -> None:
        self._hint = hint
        self._live: Live | None = None

    def _view(self, text: str, level: float, silence_left: float | None, done: bool = False) -> Group:
        if done:
            return Group(Text("Heard:", style="bold green"), Text(text or "(nothing)"))
        status = "Listening"
        if silence_left is not None:
            status += f"  (ending in {silence_left:.0f}s of silence)"
        return Group(
            Text(f"● {status}", style="bold red"),
            Text(_level_bar(level), style="dim"),
            Text(text or "…"),
            Text(self._hint, style="dim"),
        )

    def start(self, silence_seconds: float) -> None:
        self._live = Live(
            self._view("", 0.0, None),
            console=console,
            refresh_per_second=10,
            transient=False,
            redirect_stdout=False,  # see working(): never leave a stdout proxy behind
            redirect_stderr=False,
        )
        self._live.start()

    def update(self, text: str, level: float, silence_left: float | None) -> None:
        if self._live is not None:
            self._live.update(self._view(text, level, silence_left))

    def stop(self, final_text: str) -> None:
        if self._live is not None:
            self._live.update(self._view(final_text, 0.0, None, done=True))
            self._live.stop()
            self._live = None


def download_progress() -> Progress:
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        redirect_stdout=False,
        redirect_stderr=False,
    )


def devices_table(rows: list[dict]) -> None:
    table = Table(title="Input devices")
    for column in ("", "Index", "Name", "Channels"):
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row("*" if row["default"] else "", str(row["index"]), row["name"], str(row["channels"]))
    console.print(table)
