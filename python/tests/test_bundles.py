import base64
import hashlib
import io
import json
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lupine import _bundles

NAMES = ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1")


def bundle_bytes(*, platforms=("linux/amd64",), extra=None):
    files = []
    contents = {}
    for name in NAMES:
        contents[name] = name.encode()
        files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(contents[name]).hexdigest(),
                "mode": "0755",
            }
        )
    manifest = json.dumps(
        {"schema": 1, "platforms": list(platforms), "files": files}
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", manifest)
        for name, value in contents.items():
            archive.writestr(name, value)
        if extra is not None:
            archive.writestr(extra, b"unsafe")
    return output.getvalue()


@contextmanager
def bundle_server(body):
    state = {"requests": []}
    digest = hashlib.sha256(body).digest()
    etag = f'"sha256:{digest.hex()}"'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(
                {name.lower(): value for name, value in self.headers.items()}
            )
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("ETag", etag)
            self.send_header(
                "Content-Digest", f"sha-256=:{base64.b64encode(digest).decode()}:"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}", etag, state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@contextmanager
def redirect_server(target):
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(
                {name.lower(): value for name, value in self.headers.items()}
            )
            safe_path = self.path.replace("\r", "").replace("\n", "")
            self.send_response(307)
            self.send_header("Location", target + safe_path)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_resolve_downloads_verifies_and_revalidates(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LUPINE_SESSION", "lease-test")
    monkeypatch.setattr(_bundles, "platform_name", lambda: "linux/amd64")
    with bundle_server(bundle_bytes()) as (server, etag, state):
        directory, selected, platform_name = _bundles.resolve(
            (server,), NAMES
        )
        assert selected == etag
        assert platform_name == "linux/amd64"
        assert {path.name for path in directory.iterdir()} == {
            *NAMES,
            "manifest.json",
            ".etag",
        }
        assert state["requests"][0]["x-lupine-session"] == "lease-test"

        again, again_etag, _ = _bundles.resolve((server,), NAMES)
        assert again == directory
        assert again_etag == etag
        assert state["requests"][1]["if-none-match"] == etag


def test_resolve_repairs_a_corrupt_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_bundles, "platform_name", lambda: "linux/amd64")
    with bundle_server(bundle_bytes()) as (server, _, state):
        directory, _, _ = _bundles.resolve((server,), NAMES)
        (directory / NAMES[0]).write_bytes(b"corrupt")
        repaired, _, _ = _bundles.resolve((server,), NAMES)
        assert repaired == directory
        assert (repaired / NAMES[0]).read_bytes() == NAMES[0].encode()
        assert "if-none-match" not in state["requests"][1]


def test_resolve_follows_redirect_and_preserves_session(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LUPINE_SESSION", "lease-redirect")
    monkeypatch.setattr(_bundles, "platform_name", lambda: "linux/amd64")
    with (
        bundle_server(bundle_bytes()) as (gateway, etag, gateway_state),
        redirect_server(f"http://{gateway}") as (coordinator, coordinator_state),
    ):
        _, selected, _ = _bundles.resolve((coordinator,), NAMES)

    assert selected == etag
    assert coordinator_state["requests"][0]["x-lupine-session"] == "lease-redirect"
    assert gateway_state["requests"][0]["x-lupine-session"] == "lease-redirect"


def test_resolve_rejects_mixed_server_bundles(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_bundles, "platform_name", lambda: "linux/amd64")
    with (
        bundle_server(bundle_bytes()) as (first, _, _),
        bundle_server(bundle_bytes(platforms=("linux/amd64", "linux/arm64"))) as (
            second,
            _,
            _,
        ),
        pytest.raises(ValueError, match="different client bundles"),
    ):
        _bundles.resolve((first, second), NAMES)


def test_resolve_rejects_unsafe_archive_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_bundles, "platform_name", lambda: "linux/amd64")
    with (
        bundle_server(bundle_bytes(extra="../escape")) as (server, _, _),
        pytest.raises(ValueError, match="unsafe client bundle path"),
    ):
        _bundles.resolve((server,), NAMES)
