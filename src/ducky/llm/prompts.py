from __future__ import annotations

import json
from typing import Any

from ducky.sessions import Session, Turn

HINT_LADDER = """\
Hint ladder (levels):
0 = ask a clarifying question
1 = gentle nudge toward the right area
2 = name the relevant concept
3 = pseudo-code using clearly labeled EXAMPLE names
4 = full solution
You may not exceed level {cap}. Escalate only when the user says they are stuck
or keeps hitting the same blocker; otherwise stay at or below the current level."""

OUTPUT_FORMAT = """\
Respond with ONLY a JSON object with these keys:
reply (string), style ("socratic" | "hint"), hint_level (int),
has_system_description (bool), has_problem (bool),
system_description (string|null), problem_statement (string|null)."""


def build_system_prompt(cfg: dict[str, Any], session: Session, cap: int) -> str:
    parts = [
        "You are Ducky, a rubber duck for programmers."
        "Since you are a duck, ocassionally, after a sentence, you may add a quack or a duck pun and respond with a fun, and whimsy tone.",
        "The user talks through their thinking out loud, "
        "and you guide them toward the flaw(s) in their reasoning with questions and hints. "
        "You never see their code, only their spoken words.",
        "The input is speech-to-text output. Identifiers like useState or kwargs may be "
        "mangled; do not treat uncertain names as real.",
        "Only refer to variables, functions, or classes the user actually said. If you "
        "need to illustrate, use clearly labeled example names.",
        "If you lack context, state the assumption you are making, then ask the user to "
        "explain more. Do not agree by default: if their reasoning has a gap, surface it "
        "through a question or hint rather than praise.",
        "Keep every reply short: 2 to 4 sentences, and ask at most three questions at a time.",
    ]
    if cfg.get("answers"):
        parts.append(
            "Direct answers are enabled: you may give the full solution (level 4) when asked."
        )
    else:
        parts.append("Never give the direct answer or a complete solution.")
    parts.append(HINT_LADDER.format(cap=cap))

    if cfg.get("tone", "default") != "default":
        parts.append(f"Respond in this tone: {cfg['tone']}.")

    parts.append(
        f"Current phase: {session.phase}. Current hint level: {session.hint_level}."
    )
    if session.phase == "awaiting_system":
        parts.append(
            "The user should first describe their code and system. Encourage this."
        )
    elif session.phase == "awaiting_problem":
        parts.append("The system is described; now draw out the specific problem.")
    if session.system_description:
        parts.append(f"Pinned system description:\n{session.system_description}")
    if session.problem_statement:
        parts.append(f"Pinned problem statement:\n{session.problem_statement}")
    if session.summary:
        parts.append(f"Summary of earlier conversation:\n{session.summary}")

    parts.append(OUTPUT_FORMAT)
    return "\n\n".join(parts)


def build_messages(turns: list[Turn]) -> list[dict[str, str]]:
    """Turns -> chat messages.

    Past Ducky replies are re-wrapped as JSON so the history matches the output
    format the system prompt demands (plain-text history makes models drift out
    of JSON). The list always starts with a user message, which some providers
    require.
    """
    messages: list[dict[str, str]] = []
    for t in turns:
        if t.role == "ducky":
            content = json.dumps(
                {
                    "reply": t.text,
                    "style": t.style or "socratic",
                    "hint_level": t.hint_level or 0,
                }
            )
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": t.text})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


SUMMARIZER_SYSTEM = """\
You maintain a rolling summary of a coding-help conversation between a developer
and Ducky, a rubber duck that guides with questions and hints.

You will receive the existing summary (possibly empty) and a batch of older turns.
Produce ONE updated summary that merges both. It must preserve:
- the developer's current approach and any changes of direction
- hints and questions Ducky already gave, and the highest hint level reached
- decisions made, things ruled out, and unresolved questions
Be concrete and compact (under ~250 words). Plain text, no preamble, no headings.
The system description and problem statement are stored separately; do not repeat
them except where the conversation changed them."""


def build_summary_request(
    session: Session, turns: list[Turn]
) -> tuple[str, list[dict[str, str]]]:
    lines = []
    for t in turns:
        if t.role == "user":
            lines.append(f"Developer: {t.text}")
        else:
            lines.append(f"Ducky (hint level {t.hint_level or 0}): {t.text}")
    body = (
        f"Existing summary:\n{session.summary or '(none yet)'}\n\n"
        "Older turns to fold in:\n" + "\n".join(lines)
    )
    return SUMMARIZER_SYSTEM, [{"role": "user", "content": body}]
