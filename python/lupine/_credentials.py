"""Shared credential storage for the Python and Go Lupine clients."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from . import LupineError

DEFAULT_PROFILE = "default"
# Same rule as the Go CLI. fullmatch, because "$" would accept a trailing "\n".
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")


def validate_profile(name: str) -> str:
    """Return ``name`` if it is a valid profile name, else raise LupineError."""

    if not _PROFILE_NAME.fullmatch(name):
        raise LupineError(
            f"invalid profile name {name!r}: use 1-63 lowercase letters, "
            "digits, or hyphens, starting with a letter or digit"
        )
    return name


def profile_name(profile: str | None = None) -> str:
    """Resolve the explicit profile or use the default profile."""
    return DEFAULT_PROFILE if profile is None else validate_profile(profile)


def state_dir(profile: str | None = None) -> Path:
    """Return the selected profile's directory shared with the Lupine CLI."""
    name = profile_name(profile)
    root = Path.home() / ".lupine"
    if name == DEFAULT_PROFILE:
        return root
    return root / "profiles" / name


def credentials_path(profile: str | None = None) -> Path:
    return state_dir(profile) / "credentials.json"


def _normalized_api_url(api_url: str) -> str:
    return api_url.strip().rstrip("/")


def _load(profile: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(credentials_path(profile).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"tokens": {}}
    if not isinstance(value, dict) or not isinstance(value.get("tokens"), dict):
        return {"tokens": {}}
    return value


def token_for(api_url: str, profile: str | None = None) -> str | None:
    """Return the explicit environment token or the stored API credential.

    An invalid profile raises even when ``LUPINE_API_TOKEN`` is set (see
    ``profile_name``).
    """

    profile_name(profile)
    explicit = os.environ.get("LUPINE_API_TOKEN", "").strip()
    if explicit:
        return explicit
    token = _load(profile)["tokens"].get(_normalized_api_url(api_url))
    return token if isinstance(token, str) and token else None


def _save(value: dict[str, Any], profile: str | None = None) -> None:
    path = credentials_path(profile)
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


def save_token(api_url: str, token: str, profile: str | None = None) -> None:
    """Atomically persist a token in the Go CLI-compatible credential store."""

    value = _load(profile)
    value["tokens"][_normalized_api_url(api_url)] = token
    _save(value, profile)


def delete_token(api_url: str, profile: str | None = None) -> None:
    value = _load(profile)
    value["tokens"].pop(_normalized_api_url(api_url), None)
    if credentials_path(profile).exists():
        _save(value, profile)
