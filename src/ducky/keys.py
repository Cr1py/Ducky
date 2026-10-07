"""API key lookup.

Order: real environment variables first, then Ducky's own keys file in the
config dir. We deliberately never read a .env from the current directory, so
an unrelated project's keys can't leak into Ducky.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from ducky.config import config_dir

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def keys_path() -> Path:
    return config_dir() / "keys.env"


def read_keys_file(path: Path | None = None) -> dict[str, str]:
    path = path or keys_path()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            value = m.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[m.group(1)] = value
    return values


def get_key(env_var: str) -> tuple[str | None, str]:
    """Return (key, source) where source is 'env', 'file' or 'missing'."""
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value, "env"
    file_value = read_keys_file().get(env_var)
    if file_value:
        return file_value, "file"
    return None, "missing"


def save_key(env_var: str, value: str) -> Path:
    """Write/replace a key in the keys file (owner-only permissions on POSIX)."""
    path = keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    new_line = f'{env_var}="{value}"'
    for i, line in enumerate(lines):
        m = _LINE.match(line)
        if m and m.group(1) == env_var:
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(0o600)
    return path
