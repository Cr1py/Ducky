"""Speech-model management: where Vosk models live and how to download them.

Models are plain folders under the data dir, so everything stays local and works
offline once downloaded. Downloads are streamed to a temp file, extracted into a
staging folder (with a path-traversal check), validated, then moved into place, so a
failed or interrupted download never leaves a half-installed model behind.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from ducky.config import DEFAULTS, data_dir

DEFAULT_MODEL: str = DEFAULTS["vosk_model"]
BASE_URL = "https://alphacephei.com/vosk/models"
MIRROR_ENV = "DUCKY_VOSK_MODEL_URL"  # optional mirror, for restricted networks
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelError(Exception):
    pass


def is_valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


def models_dir() -> Path:
    return data_dir() / "models"


def is_installed(path: Path) -> bool:
    """A Vosk model folder contains an `am` (acoustic model) directory."""
    return path.is_dir() and (path / "am").is_dir()


def resolve_model(name_or_path: str) -> Path | None:
    """A path to a model folder, or a model name installed in Ducky's models dir."""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir() and is_installed(candidate):
        return candidate
    if _NAME_RE.match(name_or_path):
        installed = models_dir() / name_or_path
        if is_installed(installed):
            return installed
    return None


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in archive.namelist():
        target = (dest / member).resolve()
        if target != root and root not in target.parents:
            raise ModelError("The downloaded archive contains an unsafe path; refusing to unpack it.")
    archive.extractall(dest)


def download_model(
    name: str,
    *,
    root: Path | None = None,
    http_client: Any = None,
    base_url: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download and install a Vosk model by name. `on_progress(received, total)`; total may be 0."""
    if not _NAME_RE.match(name):
        raise ModelError(f"'{name}' isn't a valid model name. See {BASE_URL} for the list.")
    root = root or models_dir()
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    if is_installed(final):
        return final

    import httpx2

    client = http_client or httpx2.Client(
        timeout=httpx2.Timeout(30.0, read=120.0), follow_redirects=True
    )
    base_url = base_url or os.environ.get(MIRROR_ENV) or BASE_URL
    url = f"{base_url.rstrip('/')}/{name}.zip"
    zip_path: Path | None = None
    staging: Path | None = None
    try:
        with client.stream("GET", url) as response:
            if response.status_code == 404:
                raise ModelError(f"No Vosk model named '{name}'. Browse {BASE_URL} for valid names.")
            if response.status_code >= 400:
                raise ModelError(f"The download failed (HTTP {response.status_code}) from {url}.")
            total = int(response.headers.get("content-length") or 0)
            received = 0
            with tempfile.NamedTemporaryFile(dir=root, suffix=".zip.part", delete=False) as out:
                zip_path = Path(out.name)
                for chunk in response.iter_bytes(65536):
                    out.write(chunk)
                    received += len(chunk)
                    if on_progress:
                        on_progress(received, total)

        staging = Path(tempfile.mkdtemp(dir=root, prefix=".extract-"))
        try:
            with zipfile.ZipFile(zip_path) as archive:
                _safe_extract(archive, staging)
        except zipfile.BadZipFile as e:
            raise ModelError("The downloaded file is corrupt. Try the download again.") from e

        entries = list(staging.iterdir())
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        if not is_installed(source):
            raise ModelError("The download doesn't look like a Vosk model (no `am` folder inside).")
        if final.exists():  # a broken leftover from an earlier attempt
            shutil.rmtree(final)
        shutil.move(str(source), str(final))
        return final
    except httpx2.HTTPError as e:
        raise ModelError(
            f"Couldn't download the model ({type(e).__name__}). Check your internet connection."
        ) from e
    finally:
        if zip_path is not None and zip_path.exists():
            os.remove(zip_path)
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
