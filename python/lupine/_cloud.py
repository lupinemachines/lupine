"""Authenticated Lupine Cloud lease lifecycle for Python applications."""

from __future__ import annotations

import atexit
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import LupineAuthenticationError, LupineError
from ._credentials import token_for
from ._login import DEFAULT_API_URL, DEFAULT_CONSOLE_URL

LOGIN_HINT = (
    "Run `uvx lupine login` or `python -m lupine login`, "
    "or set LUPINE_API_TOKEN."
)
_HEARTBEAT_INTERVAL = 30.0


def _api_url(value: str | None = None) -> str:
    return (value or os.environ.get("LUPINE_API_URL") or DEFAULT_API_URL).rstrip("/")


def _request(
    method: str,
    url: str,
    token: str,
    *,
    body: dict[str, Any] | None = None,
    expected: tuple[int, ...],
    timeout: float = 30,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", "Bearer " + token)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise LupineAuthenticationError(
                "Lupine Cloud authentication is required. " + LOGIN_HINT
            ) from exc
        if exc.code == 402:
            raise LupineError(
                "Lupine Cloud requires a payment method; visit "
                f"{DEFAULT_CONSOLE_URL}/billing."
            ) from exc
        if exc.code == 503:
            raise LupineError("No Lupine Cloud GPU capacity is currently available.") from exc
        raise LupineError(f"Lupine Cloud request failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LupineError(f"Could not reach Lupine Cloud: {exc.reason}") from exc

    with response:
        if response.status not in expected:
            raise LupineError(f"Lupine Cloud request failed: HTTP {response.status}")
        contents = response.read()
    if not contents:
        return {}
    try:
        value = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise LupineError("Lupine Cloud returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LupineError("Lupine Cloud returned an invalid response")
    return value


@dataclass
class CloudSession:
    """One authenticated cloud lease and its process-local CUDA connection."""

    api_url: str = DEFAULT_API_URL
    token: str | None = field(default=None, repr=False)
    gpu_type: str | None = None
    gpu_count: int | None = None
    region: str | None = None
    lease_id: str | None = field(default=None, init=False)
    gpu: dict[str, Any] = field(default_factory=dict, init=False)
    data_plane: dict[str, Any] = field(default_factory=dict, init=False)
    _loaded: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _heartbeat_stop: threading.Event | None = field(default=None, init=False, repr=False)
    _heartbeat_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _previous_env: dict[str, str | None] = field(
        default_factory=dict, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def start(self) -> CloudSession:
        """Acquire, bind, preload, and begin heartbeating this lease."""

        if self.lease_id is not None:
            return self
        self.api_url = _api_url(self.api_url)
        credential = self.token or token_for(self.api_url)
        if not credential:
            raise LupineAuthenticationError(
                "Lupine Cloud is not authenticated. " + LOGIN_HINT
            )
        self.token = credential

        options: dict[str, Any] = {}
        if self.gpu_type:
            options["gpu_type"] = self.gpu_type
        if self.gpu_count is not None:
            options["gpu_count"] = self.gpu_count
        if self.region:
            options["region"] = self.region

        created = _request(
            "POST",
            self.api_url + "/v1/sessions",
            credential,
            body=options,
            expected=(201,),
        )
        lease_id = created.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            raise LupineError("Lupine Cloud did not return a lease ID")
        self.lease_id = lease_id

        try:
            bound = _request(
                "POST",
                self._session_url("/bind"),
                credential,
                body=options,
                expected=(200,),
            )
            self.gpu = bound.get("gpu") if isinstance(bound.get("gpu"), dict) else {}
            self.data_plane = (
                bound.get("data_plane")
                if isinstance(bound.get("data_plane"), dict)
                else {}
            )
            self._previous_env = {
                name: os.environ.get(name)
                for name in ("LUPINE_SESSION", "LUPINE_SERVER")
            }
            os.environ["LUPINE_SESSION"] = lease_id
            os.environ["LUPINE_SERVER"] = self.api_url
            self._start_heartbeat()

            from . import load_native

            self._loaded = dict(load_native(missing_ok=False))
        except Exception:
            self.close()
            raise

        atexit.register(self.close)
        return self

    def _session_url(self, suffix: str = "") -> str:
        return (
            self.api_url
            + "/v1/sessions/"
            + urllib.parse.quote(self.lease_id or "", safe="")
            + suffix
        )

    def _start_heartbeat(self) -> None:
        stop = threading.Event()
        self._heartbeat_stop = stop

        def heartbeat() -> None:
            while not stop.is_set():
                try:
                    _request(
                        "POST",
                        self._session_url("/heartbeat"),
                        self.token or "",
                        expected=(200,),
                    )
                except LupineError:
                    pass
                stop.wait(_HEARTBEAT_INTERVAL)

        thread = threading.Thread(
            target=heartbeat, name="lupine-cloud-heartbeat", daemon=True
        )
        self._heartbeat_thread = thread
        thread.start()

    def close(self) -> None:
        """Stop heartbeating and release this process-owned lease."""

        if self._closed:
            return
        self._closed = True
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        if self.lease_id and self.token:
            try:
                _request(
                    "DELETE",
                    self._session_url(),
                    self.token,
                    expected=(200, 204),
                    timeout=5,
                )
            except LupineError:
                pass
        for name, previous in self._previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        atexit.unregister(self.close)

    def __enter__(self) -> CloudSession:  # noqa: PYI034
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def devices(self) -> list[Any]:
        from . import devices

        return devices()

    def device(self, index: int = 0) -> Any:
        available = self.devices()
        if index >= len(available) or index < -len(available):
            raise LupineError(
                f"device index {index} out of range ({len(available)} devices)"
            )
        return available[index]

    @property
    def loaded(self) -> dict[str, str]:
        return dict(self._loaded)


def connect(
    *,
    api_url: str | None = None,
    token: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    region: str | None = None,
) -> CloudSession:
    """Acquire and activate an authenticated Lupine Cloud session."""

    return CloudSession(
        api_url=_api_url(api_url),
        token=token,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        region=region,
    ).start()
