"""The torch backend's torch-free parts: the socket framing and the version
rule, with a fake worker on a local socket."""

from __future__ import annotations

import socket
import threading

import pytest

import lupine
import lupine._backend.transport


def fake_worker(hello: dict, handler=None):
    """A worker that answers with ``handler`` or, without one, hangs up
    after the first request."""

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        server = lupine._backend.transport.Server.accept(listener)
        server.hello(hello)
        for ticket, kind, meta, body, extra in server.requests():
            if handler is None:
                break
            status, payload = handler(kind, meta, body, extra)
            server.respond(ticket, status, payload)
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    return f"127.0.0.1:{listener.getsockname()[1]}"


def test_frames_round_trip_with_payload_and_error():
    seen = []

    def handler(kind, meta, body, extra):
        seen.append((kind, meta, body, bytes(extra) if extra is not None else None))
        if kind == 9:
            return 1, b"boom"
        return 0, body[::-1]

    client = lupine._backend.transport.Client.connect(fake_worker({"torch": "x"}, handler))
    assert client.hello == {"torch": "x"}
    assert client.send(1, b"m", b"fire", bytearray(b"z" * 100_000)) == 1
    assert client.call(2, b"", b"abc") == (0, b"cba")
    into = bytearray(3)
    assert client.call(2, b"", b"xyz", into=into) == (0, b"")
    assert bytes(into) == b"zyx"
    assert client.call(9, b"", b"") == (1, b"boom")
    assert seen[0] == (1, b"m", b"fire", b"z" * 100_000)
    assert client.messages == 4
    client.close()


def test_closed_connection_wakes_callers():
    client = lupine._backend.transport.Client.connect(fake_worker({"torch": "x"}))
    with pytest.raises(lupine._backend.transport.ConnectionClosed):
        client.call(2, b"", b"never answered")


@pytest.mark.parametrize("failure", ["partial_header", "partial_body", "partial_copy", "copy_size"])
def test_broken_response_wakes_active_and_pending_callers(failure):
    transport = lupine._backend.transport
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    address = f"127.0.0.1:{listener.getsockname()[1]}"

    def serve():
        with listener:
            server = transport.Server.accept(listener)
            try:
                server.hello({"torch": "x"})
                requests = server.requests()
                ticket, *_ = next(requests)
                next(requests)  # Both callers must be waiting when the response fails.
                header = transport.RESPONSE.pack(ticket, 0, 10)
                if failure == "partial_header":
                    server.sock.sendall(header[:5])
                else:
                    server.sock.sendall(header + b"xx")
            finally:
                server.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = transport.Client.connect(address)
    outcomes = []

    def call():
        into = None
        if failure == "partial_copy":
            into = bytearray(10)
        elif failure == "copy_size":
            into = bytearray(3)
        try:
            outcomes.append(client.call(2, b"", b"probe", into=into))
        except Exception as exc:
            outcomes.append(exc)

    callers = [threading.Thread(target=call, daemon=True) for _ in range(2)]
    try:
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=3)
        assert not any(caller.is_alive() for caller in callers), "a caller never woke up"
        assert len(outcomes) == 2
        assert all(isinstance(outcome, transport.ConnectionClosed) for outcome in outcomes)
        assert not client.waiters
    finally:
        client.close()
        worker.join(timeout=3)


def test_release_rule_ignores_patch_and_local_version():
    assert lupine._backend.release_of("2.12.1+cpu") == lupine._backend.release_of("2.12.0+cu130") == (2, 12)
    assert lupine._backend.release_of("2.11.2") != lupine._backend.release_of("2.12.1")


def test_cross_minor_worker_is_refused_before_the_extension_loads(monkeypatch):
    monkeypatch.setattr(lupine._backend, "_started", {})
    address = fake_worker({"torch": "1.0.0+cu130", "device_count": 1})
    with pytest.raises(lupine.LupineError, match=r"torch version mismatch.*1\.0\.0\+cu130.*same major\.minor"):
        lupine._backend.start(address)
