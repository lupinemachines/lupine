"""Resolve and cache the native client object selected by a LUPINE server."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

_ENDPOINT = "/.well-known/lupine/client/v1/"
_MAX_BUNDLE_BYTES = 128 * 1024 * 1024
_ETAG = re.compile(r'^"sha256:([0-9a-f]{64})"$')


def platform_name() -> str | None:
    """Return the public OS/architecture key for this Python process."""

    machine = platform.machine().lower()
    arch = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine)
    if arch is None:
        return None
    os_name = {"linux": "linux", "darwin": "macos", "win32": "windows"}.get(
        sys.platform
    )
    return None if os_name is None else f"{os_name}/{arch}"


def _cache_root() -> Path:
    override = os.environ.get("LUPINE_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "Lupine" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "lupine"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lupine"


def _bundle_url(server: str, platform_key: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    value = server.strip()
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid LUPINE_SERVER endpoint: {server!r}")
    return urlunsplit((parsed.scheme, parsed.netloc, _ENDPOINT + platform_key, "", ""))


def _response_endpoint(response: Any) -> str:
    """Return the final origin after coordinator and gateway redirects."""

    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(response.geturl())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid redirected LUPINE endpoint: {response.geturl()!r}")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _etag_digest(etag: str) -> str:
    match = _ETAG.fullmatch(etag)
    if match is None:
        raise ValueError(f"server returned an invalid strong ETag: {etag!r}")
    return match.group(1)


def _selector_path(cache: Path, server: str, platform_key: str) -> Path:
    digest = hashlib.sha256(f"{server}\0{platform_key}".encode()).hexdigest()
    return cache / "selectors" / digest


def _read_selector(path: Path) -> str | None:
    try:
        value = path.read_text().strip()
        _etag_digest(value)
        return value
    except (OSError, ValueError):
        return None


def _write_selector(path: Path, etag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".selector-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(etag + "\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _manifest(directory: Path) -> dict[str, Any]:
    with (directory / "manifest.json").open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError("client bundle manifest is not an object")
    return value


def _validate_directory(
    directory: Path, etag: str, platform_key: str, expected_names: tuple[str, ...]
) -> bool:
    try:
        if (directory / ".etag").read_text().strip() != etag:
            return False
        manifest = _manifest(directory)
        if manifest.get("schema") != 1 or platform_key not in manifest.get(
            "platforms", []
        ):
            return False
        files = manifest.get("files")
        if not isinstance(files, list):
            return False
        by_name = {item["path"]: item for item in files if isinstance(item, dict)}
        if set(by_name) != set(expected_names):
            return False
        for name in expected_names:
            contents = (directory / name).read_bytes()
            if hashlib.sha256(contents).hexdigest() != by_name[name].get("sha256"):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _verified_manifest(
    archive: zipfile.ZipFile, platform_key: str, expected_names: tuple[str, ...]
) -> dict[str, Any]:
    entries = archive.infolist()
    if sum(entry.file_size for entry in entries) > _MAX_BUNDLE_BYTES:
        raise ValueError("client bundle expands beyond the size limit")
    names: set[str] = set()
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError(f"unsafe client bundle path: {entry.filename!r}")
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError(f"client bundle contains a symlink: {entry.filename!r}")
        if entry.filename in names:
            raise ValueError(f"duplicate client bundle path: {entry.filename!r}")
        names.add(entry.filename)
    if names != {"manifest.json", *expected_names}:
        raise ValueError("client bundle contains an unexpected file set")

    manifest = json.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("unsupported client bundle manifest")
    if platform_key not in manifest.get("platforms", []):
        raise ValueError("client bundle does not support this platform")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("client bundle manifest has no file list")
    by_name = {item.get("path"): item for item in files if isinstance(item, dict)}
    if set(by_name) != set(expected_names):
        raise ValueError("client bundle manifest has an unexpected file set")
    for name in expected_names:
        contents = archive.read(name)
        if hashlib.sha256(contents).hexdigest() != by_name[name].get("sha256"):
            raise ValueError(f"client bundle file hash does not match: {name}")
    return manifest


def _install(
    cache: Path,
    body: bytes,
    etag: str,
    platform_key: str,
    expected_names: tuple[str, ...],
) -> Path:
    digest = _etag_digest(etag)
    if hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("client bundle bytes do not match the server ETag")
    destination = cache / "clients" / digest
    if _validate_directory(destination, etag, platform_key, expected_names):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".client-", dir=destination.parent))
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            manifest = _verified_manifest(archive, platform_key, expected_names)
            for item in manifest["files"]:
                name = item["path"]
                target = temporary / name
                target.write_bytes(archive.read(name))
                target.chmod(int(item.get("mode", "0755"), 8))
            (temporary / "manifest.json").write_bytes(archive.read("manifest.json"))
        (temporary / ".etag").write_text(etag + "\n")

        if destination.exists():
            shutil.rmtree(destination)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not _validate_directory(destination, etag, platform_key, expected_names):
                raise
        return destination
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _download(
    server: str,
    platform_key: str,
    expected_names: tuple[str, ...],
    *,
    unconditional: bool = False,
) -> tuple[Path, str, str]:
    cache = _cache_root()
    selector = _selector_path(cache, server, platform_key)
    previous = None if unconditional else _read_selector(selector)
    headers = {"Accept": "application/vnd.lupine.client-bundle.v1+zip"}
    session = os.environ.get("LUPINE_SESSION")
    if session:
        headers["x-lupine-session"] = session
    if previous:
        candidate = cache / "clients" / _etag_digest(previous)
        if _validate_directory(candidate, previous, platform_key, expected_names):
            headers["If-None-Match"] = previous

    request = urllib.request.Request(
        _bundle_url(server, platform_key), headers=headers, method="GET"
    )
    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as exc:
        if exc.code == 304 and previous:
            candidate = cache / "clients" / _etag_digest(previous)
            if _validate_directory(candidate, previous, platform_key, expected_names):
                return candidate, previous, _response_endpoint(exc)
            return _download(
                server,
                platform_key,
                expected_names,
                unconditional=True,
            )
        raise

    with response:
        endpoint = _response_endpoint(response)
        etag = response.headers.get("ETag", "")
        _etag_digest(etag)
        body = response.read(_MAX_BUNDLE_BYTES + 1)
        if len(body) > _MAX_BUNDLE_BYTES:
            raise ValueError("client bundle exceeds the download size limit")
        content_digest = response.headers.get("Content-Digest")
    digest = hashlib.sha256(body).digest()
    expected_digest = f"sha-256=:{base64.b64encode(digest).decode()}:"
    if content_digest and content_digest != expected_digest:
        raise ValueError("client bundle does not match Content-Digest")

    directory = _install(cache, body, etag, platform_key, expected_names)
    _write_selector(selector, etag)
    return directory, etag, endpoint


def resolve(
    servers: tuple[str, ...], expected_names: tuple[str, ...]
) -> tuple[Path, str, str, tuple[str, ...]]:
    """Resolve every endpoint and require one compatible native object."""

    platform_key = platform_name()
    if platform_key is None:
        raise ValueError(f"unsupported LUPINE client platform: {sys.platform}")
    selected: tuple[Path, str, str] | None = None
    endpoints = []
    for server in servers:
        current = _download(server, platform_key, expected_names)
        if selected is not None and current[1] != selected[1]:
            raise ValueError(
                "LUPINE servers selected different client bundles: "
                f"{selected[1]} != {current[1]}"
            )
        selected = current
        endpoints.append(current[2])
    if selected is None:
        raise ValueError("no LUPINE server was configured")
    return selected[0], selected[1], platform_key, tuple(endpoints)
