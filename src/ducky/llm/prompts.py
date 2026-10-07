from __future__ import annotations

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
        "You are Ducky, a rubber duck for programmers. "
        "The user talks through their thinking out loud, "
        "and you guide them toward the flaw in their reasoning with questions and hints. "
        "You never see their code, only their spoken words.",
        "The input is speech-to-text output. Identifiers like useState or kwargs may be "
        "mangled; do not treat uncertain names as real.",
        "Only refer to variables, functions, or classes the user actually said. If you "
        "need to illustrate, use clearly labeled example names.",
        "If you lack context, state the assumption you are making, then ask the user to "
        "explain more. Do not agree by default: if their reasoning has a gap, surface it "
        "through a question or hint rather than praise.",
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
    return [
        {"role": "assistant" if t.role == "ducky" else "user", "content": t.text}
        for t in turns
    ]
