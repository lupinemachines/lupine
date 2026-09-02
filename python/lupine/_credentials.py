"""Shared credential storage for the Python and Go Lupine clients."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def state_dir() -> Path:
    """Return the directory shared with the Lupine CLI."""

    override = os.environ.get("LUPINE_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".lupine"


def credentials_path() -> Path:
    return state_dir() / "credentials.json"


def _normalized_api_url(api_url: str) -> str:
    return api_url.strip().rstrip("/")


def _load() -> dict[str, Any]:
    try:
        value = json.loads(credentials_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"tokens": {}}
    if not isinstance(value, dict) or not isinstance(value.get("tokens"), dict):
        return {"tokens": {}}
    return value


def token_for(api_url: str) -> str | None:
    """Return the explicit environment token or the stored API credential."""

    explicit = os.environ.get("LUPINE_API_TOKEN", "").strip()
    if explicit:
        return explicit
    token = _load()["tokens"].get(_normalized_api_url(api_url))
    return token if isinstance(token, str) and token else None


def _save(value: dict[str, Any]) -> None:
    path = credentials_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def save_token(api_url: str, token: str) -> None:
    """Atomically persist a token in the Go CLI-compatible credential store."""

    value = _load()
    value["tokens"][_normalized_api_url(api_url)] = token
    _save(value)


def delete_token(api_url: str) -> None:
    value = _load()
    value["tokens"].pop(_normalized_api_url(api_url), None)
    if credentials_path().exists():
        _save(value)
