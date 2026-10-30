import os
import sys

import pytest

import lupine._backend


def test_macos_session_always_starts_native_backend(monkeypatch):
    calls = []
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(lupine._backend, "start", lambda: calls.append("native"))
    with lupine.connect(host="gpu:14833") as session:
        assert session.backend == "torch"
        assert session._device_type() == "lupine"
    assert calls == ["native"]


def test_failed_start_restores_server_for_retry(monkeypatch):
    calls = []

    def start():
        calls.append(os.environ["LUPINE_SERVER"])
        if len(calls) == 1:
            raise lupine.LupineError("unreachable")

    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(lupine._backend, "start", start)
    with pytest.raises(lupine.LupineError, match="unreachable"):
        lupine.connect(host="a:14833")
    assert "LUPINE_SERVER" not in os.environ
    with lupine.connect(host="b:14833"):
        pass
    assert calls == ["a:14833", "b:14833"]
