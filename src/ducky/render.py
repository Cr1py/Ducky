"""All terminal output lives here so the rest of the code stays testable."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ducky.sessions import Session, Turn

console = Console()
err_console = Console(stderr=True)


def error(msg: str) -> None:
    err_console.print(Text(f"Error: {msg}", style="bold red"))


def working(message: str):
    """Spinner context manager (no-op output when not attached to a terminal)."""
    return console.status(Text(message, style="dim"))


def info(msg: str) -> None:
    console.print(Text(msg))


def dim(msg: str) -> None:
    console.print(Text(msg, style="dim"))


def session_header(session: Session) -> None:
    console.print(
        Text(f"Session: {session.name}  ({session.phase})", style="bold cyan")
    )


def duck_says(text: str, hint_level: int | None = None) -> None:
    subtitle = f"hint level {hint_level}" if hint_level is not None else None
    console.print(
        Panel(
            Text(text),
            title="Ducky",
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style="yellow",
        )
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
        table.add_column(
            column, overflow="fold"
        )  # wrap long IDs instead of truncating them
    for name, provider, model, env_var, status in rows:
        table.add_row(
            "*" if name == active else "", name, provider, model, env_var, status
        )
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
