import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lupine
import pytest


@contextmanager
def cloud_server(gateway_endpoint="gw-east.lupine.sh:9443"):
    state = {"requests": [], "heartbeat": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def _record(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            state["requests"].append(
                (self.command, self.path, self.headers.get("Authorization"), body)
            )

        def do_POST(self):
            self._record()
            if self.headers.get("Authorization") != "Bearer lup_test":
                self.send_response(401)
                self.end_headers()
                return
            if self.path == "/v1/sessions":
                self.send_response(201)
                body = json.dumps({"lease_id": "lease-python"}).encode()
            elif self.path.endswith("/bind"):
                self.send_response(200)
                data_plane = {}
                if gateway_endpoint is not None:
                    data_plane["gateway_endpoint"] = gateway_endpoint
                body = json.dumps(
                    {
                        "gpu": {"gpu_type": "RTX 4090", "gpu_count": 1},
                        "data_plane": data_plane,
                    }
                ).encode()
            elif self.path.endswith("/heartbeat"):
                state["heartbeat"].set()
                self.send_response(200)
                body = b'{"status":"ok"}'
            else:
                self.send_response(404)
                body = b""
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            self._record()
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_cloud_session_authenticates_binds_heartbeats_and_releases(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("LUPINE_SESSION", raising=False)
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.setattr(
        lupine,
        "load_native",
        lambda *, missing_ok: {"libcuda.so.1": str(tmp_path / "libcuda.so.1")},
    )

    with (
        cloud_server() as (api_url, state),
        lupine.cloud(
            api_url=api_url,
            token="lup_test",
            gpu_type="RTX_4090",
            gpu_count=1,
        ) as session,
    ):
        assert session.lease_id == "lease-python"
        assert session.gpu["gpu_type"] == "RTX 4090"
        assert session.loaded == {"libcuda.so.1": str(tmp_path / "libcuda.so.1")}
        assert os.environ["LUPINE_SESSION"] == "lease-python"
        assert os.environ["LUPINE_SERVER"] == "https://gw-east.lupine.sh:9443"
        assert state["heartbeat"].wait(1)

    assert "LUPINE_SESSION" not in os.environ
    assert "LUPINE_SERVER" not in os.environ
    assert [request[0] for request in state["requests"]].count("DELETE") == 1
    assert all(request[2] == "Bearer lup_test" for request in state["requests"])
    create_body = json.loads(state["requests"][0][3])
    assert create_body == {"gpu_type": "RTX_4090", "gpu_count": 1}


def test_cloud_session_requires_login(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("LUPINE_API_TOKEN", raising=False)

    with pytest.raises(
        lupine.LupineAuthenticationError, match="python -m lupine login"
    ):
        lupine.cloud()


def test_cloud_session_rejects_invalid_login():
    with (
        cloud_server() as (api_url, _),
        pytest.raises(lupine.LupineAuthenticationError, match="uvx lupine login"),
    ):
        lupine.cloud(api_url=api_url, token="lup_invalid")


def test_cloud_session_releases_lease_when_bind_omits_gateway(monkeypatch):
    monkeypatch.delenv("LUPINE_SESSION", raising=False)
    monkeypatch.delenv("LUPINE_SERVER", raising=False)

    with (
        cloud_server(gateway_endpoint=None) as (api_url, state),
        pytest.raises(lupine.LupineError, match="did not include a gateway"),
    ):
        lupine.cloud(api_url=api_url, token="lup_test")

    assert [request[0] for request in state["requests"]].count("DELETE") == 1
    assert "LUPINE_SESSION" not in os.environ
    assert "LUPINE_SERVER" not in os.environ
