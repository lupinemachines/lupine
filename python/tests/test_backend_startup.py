import sys

import lupine
from lupine import _backend


def test_macos_session_always_starts_native_backend(monkeypatch):
    calls = []
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_backend, "start", lambda: calls.append("native"))
    with lupine.connect(host="gpu:14833") as session:
        assert session.backend == "torch"
        assert session._device_type() == "lupine"
    assert calls == ["native"]
