"""Browser-based OAuth login shared by the Python entry points."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import LupineError
from ._credentials import save_token

DEFAULT_API_URL = "https://api.lupine.sh"
DEFAULT_CONSOLE_URL = "https://console.lupine.sh"
_USER_AGENT = "lupine-python/2.0.3"
_SUCCESS_HTML = b"""<!doctype html><html><body style="font-family:monospace;background:#0a0a0a;color:#a3a3a3;padding:2rem"><h2 style="color:#a78bfa">Logged in to Lupine Cloud</h2><p>You can close this tab and return to your terminal.</p></body></html>"""


@dataclass(frozen=True)
class LoginResult:
    token: str = field(repr=False)
    email: str = ""
    name: str = ""


def _request_json(
    method: str, url: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("User-Agent", _USER_AGENT)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            contents = response.read()
            try:
                value = json.loads(contents) if contents else {}
            except json.JSONDecodeError as exc:
                raise LupineError("Lupine login returned invalid JSON") from exc
            if not isinstance(value, dict):
                raise LupineError("Lupine login returned an invalid response")
            return response.status, value
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except urllib.error.URLError as exc:
        raise LupineError(f"Could not reach Lupine Cloud: {exc.reason}") from exc


def _callback_server() -> tuple[ThreadingHTTPServer, threading.Event]:
    callback = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urllib.parse.urlsplit(self.path).path != "/callback":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_SUCCESS_HTML)))
            self.end_headers()
            self.wfile.write(_SUCCESS_HTML)
            callback.set()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler), callback


def login(
    *,
    api_url: str | None = None,
    console_url: str | None = None,
    timeout: float = 300,
    open_browser: Callable[[str], bool] = webbrowser.open,
    output: Callable[[str], None] = print,
) -> LoginResult:
    """Authenticate in a browser and store the resulting bearer token."""

    api_url = (api_url or os.environ.get("LUPINE_API_URL") or DEFAULT_API_URL).rstrip(
        "/"
    )
    console_url = (
        console_url or os.environ.get("LUPINE_CONSOLE_URL") or DEFAULT_CONSOLE_URL
    ).rstrip("/")
    login_id = secrets.token_hex(24)
    callback_server, callback = _callback_server()
    callback_thread = threading.Thread(
        target=callback_server.serve_forever,
        name="lupine-login-callback",
        daemon=True,
    )
    callback_thread.start()
    try:
        status, _ = _request_json(
            "POST", api_url + "/v1/cli-login", {"login_id": login_id}
        )
        if status != 201:
            raise LupineError(f"Could not start Lupine login: HTTP {status}")

        port = callback_server.server_address[1]
        query = urllib.parse.urlencode({"callback_port": port})
        login_url = f"{console_url}/login/{urllib.parse.quote(login_id)}?{query}"
        output(f"Open this URL to log in:\n  {login_url}")
        try:
            open_browser(login_url)
        except (OSError, webbrowser.Error):
            output("Could not open a browser automatically; use the URL above.")
        output("Waiting for login to complete (up to 5 minutes)...")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, value = _request_json(
                "GET", api_url + "/v1/cli-login/" + urllib.parse.quote(login_id)
            )
            token = value.get("token")
            if status == 200 and value.get("status") == "complete" and token:
                user = value.get("user")
                user = user if isinstance(user, dict) else {}
                result = LoginResult(
                    token=str(token),
                    email=str(user.get("email", "")),
                    name=str(user.get("name", "")),
                )
                save_token(api_url, result.token)
                return result
            callback.wait(min(2, max(0, deadline - time.monotonic())))
            callback.clear()
        raise LupineError("Lupine login timed out after 5 minutes")
    finally:
        callback_server.shutdown()
        callback_server.server_close()
        callback_thread.join()
