"""Typer app: subcommands for actions, flags for modifiers"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Iterator, Optional

import typer

from ducky import (
    __version__,
    db,
    endphrase,
    keys,
    memory,
    registry,
    render,
    sessions,
    turn_runner,
)
from ducky.config import ConfigError, load_config, set_value
from ducky.input import EmptyTranscript, InputUnavailable, TextInput, VoiceInput
from ducky.llm import LLMError
from ducky.sessions import SessionError

app = typer.Typer(
    name="ducky",
    help="A duck in your terminal you can walk your problems through.",
    no_args_is_help=True,
    add_completion=False,
)
session_app = typer.Typer(help="Manage sessions.", no_args_is_help=True)
config_app = typer.Typer(help="View and change settings.", no_args_is_help=True)
phrases_app = typer.Typer(
    help="Manage end phrases (say one to end your turn).", no_args_is_help=True
)
app.add_typer(session_app, name="session")
app.add_typer(config_app, name="config")
app.add_typer(phrases_app, name="phrases")


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = db.connect()
    try:
        yield conn
    except (SessionError, ConfigError, endphrase.PhraseError) as e:
        render.error(str(e))
        raise typer.Exit(1)
    finally:
        conn.close()


def _report_memory(result: memory.PruneResult) -> None:
    if result.folded:
        noun = "turn" if result.folded == 1 else "turns"
        render.dim(f"(Folded {result.folded} older {noun} into the session summary.)")
    if result.error:
        render.dim(
            f"(Couldn't update the session summary: {result.error} Nothing was lost.)"
        )


def _refresh_summary(
    conn: sqlite3.Connection, cfg: dict, session: sessions.Session
) -> None:
    """Save a current summary before leaving a session. Never blocks the user."""
    with render.working("Saving a summary of this session..."):
        result = memory.refresh_summary(conn, cfg, session)
    _report_memory(result)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"ducky {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show version."
        ),
    ] = False,
) -> None:
    pass


# --- thoughts ---------------------------------------------------------------


def _resolve_session(
    conn: sqlite3.Connection,
    cfg: dict,
    new: bool,
    name: Optional[str],
    session_name: Optional[str],
) -> sessions.Session:
    if new and session_name:
        raise SessionError("Use either --new or --session, not both.")
    if name and not new:
        raise SessionError("--name only applies together with --new.")

    previous = sessions.get_active(conn)
    if session_name:
        target = sessions.get_session_by_name(conn, session_name)
        if target is None:
            raise SessionError(f"No session named '{session_name}'.")
    elif new:
        target = sessions.create_session(conn, name)
    else:
        target = previous or sessions.create_session(conn)

    if previous and previous.id != target.id:
        _refresh_summary(conn, cfg, previous)  # save the old session's summary first
    sessions.set_active(conn, target.id)
    return target


@app.command()
def thoughts(
    text: Annotated[
        Optional[str],
        typer.Option("--text", "-t", help="Type your thoughts instead of speaking."),
    ] = None,
    new: Annotated[bool, typer.Option("--new", help="Start a fresh session.")] = False,
    name: Annotated[
        Optional[str],
        typer.Option("--name", help="Name for the new session (with --new)."),
    ] = None,
    session: Annotated[
        Optional[str],
        typer.Option("--session", "-s", help="Switch to a session by name."),
    ] = None,
    skip_intro: Annotated[
        bool, typer.Option("--skip-intro", help="Skip the system/problem walkthrough.")
    ] = False,
) -> None:
    """Talk through your approach. Continues the active session by default."""
    cfg = load_config()
    with _db() as conn:
        current = _resolve_session(conn, cfg, new, name, session)
        if skip_intro and current.phase != "iterating":
            sessions.update_session(conn, current.id, phase="iterating", intro_nudges=0)
            current = sessions.get_session(conn, current.id)  # type: ignore[assignment]

        render.session_header(current)
        render.dim(turn_runner.opening_prompt(current))

        use_text = text is not None or not sys.stdin.isatty()
        source = (
            TextInput(text)
            if use_text
            else VoiceInput(
                silence_seconds=cfg["silence"],
                end_phrases=[r["phrase"] for r in endphrase.list_phrases(conn)],
            )
        )
        try:
            with render.working("Ducky is thinking..."):
                result = turn_runner.run_turn(conn, cfg, current, source)
        except InputUnavailable as e:
            render.error(str(e))
            raise typer.Exit(1)
        except EmptyTranscript as e:
            render.error(f"{e} Try again, or use --text.")
            raise typer.Exit(1)
        except LLMError as e:
            detail = str(e).rstrip()
            if not detail.endswith((".", "]", "!")):
                detail += "."
            render.error(f"{detail} Your words were saved to this session.")
            raise typer.Exit(1)

        render.duck_says(result.response.reply, result.hint_level)

        # Prune after the reply is on screen so the summarizer never delays it.
        if memory.needs_prune(conn, cfg, result.session):
            with render.working("Tidying up older notes..."):
                pruned = memory.maybe_prune(conn, cfg, result.session)
            _report_memory(pruned)

        render.choices_footer()


@app.command()
def end() -> None:
    """Finish the active session (saves its summary, clears the active pointer)."""
    cfg = load_config()
    with _db() as conn:
        active = sessions.get_active(conn)
        if active is None:
            render.info("No active session.")
            return
        _refresh_summary(conn, cfg, active)
        sessions.clear_active(conn)
        render.info(
            f"Ended session '{active.name}'. Resume any time: ducky thoughts --session {active.name}"
        )


@app.command()
def history(
    session: Annotated[
        Optional[str],
        typer.Option("--session", "-s", help="Session name (default: active)."),
    ] = None,
) -> None:
    """Show a session's history (summary + remaining turns)."""
    with _db() as conn:
        target = (
            sessions.get_session_by_name(conn, session)
            if session
            else sessions.get_active(conn)
        )
        if target is None:
            raise SessionError(
                f"No session named '{session}'." if session else "No active session."
            )
        render.history_view(
            target, sessions.get_turns(conn, target.id, include_summarized=False)
        )


@app.command()
def quack() -> None:
    """Quack."""
    sound = Path(__file__).parent / "assets" / "quack.wav"
    render.info("Quack!")
    if not sound.exists():
        render.dim(
            "(Dev forgor to drop a quack.wav into src/ducky/assets/. Feel free to do it yourself tho.)"
        )
        return
    try:
        import simpleaudio  # optional dependency

        simpleaudio.WaveObject.from_wave_file(str(sound)).play().wait_done()
    except ImportError:
        render.dim("(Install simpleaudio to play the sound.)")


# --- session ----------------------------------------------------------------


@session_app.command("list")
def session_list() -> None:
    """List all sessions."""
    with _db() as conn:
        all_sessions = sessions.list_sessions(conn)
        if not all_sessions:
            render.info("No sessions yet. Start one with: ducky thoughts")
            return
        render.sessions_table(all_sessions, sessions.get_active_id(conn))


@session_app.command("rename")
def session_rename(old: str, new: str) -> None:
    """Rename a session."""
    with _db() as conn:
        renamed = sessions.rename_session(conn, old, new)
        render.info(f"Renamed '{old}' to '{renamed.name}'.")


@session_app.command("delete")
def session_delete(
    name: str,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
) -> None:
    """Delete a session and its history."""
    with _db() as conn:
        if sessions.get_session_by_name(conn, name) is None:
            raise SessionError(f"No session named '{name}'.")
        if not yes and not typer.confirm(
            f"Delete session '{name}' and all its history?"
        ):
            render.info("Cancelled.")
            return
        was_active = sessions.delete_session(conn, name)
        render.info(
            f"Deleted '{name}'."
            + (
                " It was the active session. No active session now."
                if was_active
                else ""
            )
        )


# --- config -----------------------------------------------------------------


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a value, e.g. `ducky config set answers true`."""
    try:
        new_value = set_value(key, value)
    except ConfigError as e:
        render.error(str(e))
        raise typer.Exit(1)
    render.info(f"{key} = {new_value}")


def _fail(e: Exception) -> typer.Exit:
    render.error(str(e))
    return typer.Exit(1)


def _key_status(spec: registry.ModelSpec) -> str:
    if not spec.api_key_env:
        return "not needed"
    return keys.get_key(spec.api_key_env)[1]


@config_app.command("show")
def config_show() -> None:
    """Show current settings and which model they resolve to."""
    cfg = load_config()
    for key, value in cfg.items():
        render.info(f"{key} = {value}")
    try:
        spec = registry.get_spec(cfg["agent"])
    except ConfigError as e:
        render.info(f"model = (unresolved: {e})")
        return
    render.info(
        f"model = {spec.provider}: {spec.model}  (API key: {_key_status(spec)})"
    )


@config_app.command("models")
def config_models() -> None:
    """List available models (built-in plus your own models.toml)."""
    try:
        models = registry.load_registry()
    except ConfigError as e:
        raise _fail(e)
    rows = [
        (name, spec.provider, spec.model, spec.api_key_env or "-", _key_status(spec))
        for name, spec in models.items()
    ]
    render.models_table(rows, load_config()["agent"])
    render.dim(f"Add or override models in: {registry.user_models_path()}")


@config_app.command("keys")
def config_keys() -> None:
    """Show which models have an API key (values are never printed)."""
    try:
        models = registry.load_registry()
    except ConfigError as e:
        raise _fail(e)
    for name, spec in models.items():
        render.info(f"{name:<10} {spec.api_key_env or '-':<20} {_key_status(spec)}")
    render.dim(f"Keys file: {keys.keys_path()}  (environment variables take priority)")


@config_app.command("set-key")
def config_set_key(name: str) -> None:
    """Store an API key for a model in Ducky's keys file (prompts without echoing)."""
    try:
        spec = registry.get_spec(name)
    except ConfigError as e:
        raise _fail(e)
    if not spec.api_key_env:
        render.info(f"'{name}' needs no API key.")
        return
    value = typer.prompt(spec.api_key_env, hide_input=True).strip()
    if not value:
        render.error("No key entered.")
        raise typer.Exit(1)
    path = keys.save_key(spec.api_key_env, value)
    render.info(f"Saved {spec.api_key_env} to {path}")


# --- phrases ----------------------------------------------------------------


@phrases_app.command("list")
def phrases_list() -> None:
    """List end phrases."""
    with _db() as conn:
        for row in endphrase.list_phrases(conn):
            render.info(f"{row['phrase']}" + ("  (alias)" if row["is_alias"] else ""))


@phrases_app.command("add")
def phrases_add(phrase: str) -> None:
    """Add an end phrase (min two words)."""
    with _db() as conn:
        added = endphrase.add_phrase(conn, phrase)
        render.info(f"Added end phrase: {added}")


@phrases_app.command("remove")
def phrases_remove(phrase: str) -> None:
    """Remove an end phrase."""
    with _db() as conn:
        endphrase.remove_phrase(conn, phrase)
        render.info("Removed.")


@phrases_app.command("test")
def phrases_test() -> None:
    """Say a phrase and see how it was transcribed (arrives with voice input)."""
    render.error("phrases test needs voice input (milestone M5).")
    raise typer.Exit(1)
