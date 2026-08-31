import os

import lupine
import lupine_autocloud


def test_activate_defaults_to_cloud_without_changing_local_policy(monkeypatch):
    monkeypatch.delenv("LUPINE_AUTOCLOUD", raising=False)
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.delenv("LUPINE_DISABLE_LOCAL", raising=False)
    calls = []
    monkeypatch.setattr(
        lupine,
        "load_native",
        lambda *, missing_ok: calls.append(missing_ok) or {"libcuda": "/shim"},
    )

    assert lupine_autocloud.activate() == {"libcuda": "/shim"}
    assert os.environ["LUPINE_SERVER"] == lupine_autocloud.DEFAULT_SERVER
    assert "LUPINE_DISABLE_LOCAL" not in os.environ
    assert calls == [False]


def test_activate_respects_explicit_server_and_local_policy(monkeypatch):
    monkeypatch.delenv("LUPINE_AUTOCLOUD", raising=False)
    monkeypatch.setenv("LUPINE_SERVER", "gpu.example:7443")
    monkeypatch.setenv("LUPINE_DISABLE_LOCAL", "caller-value")

    def load_native(*, missing_ok):
        assert missing_ok is False
        assert os.environ["LUPINE_DISABLE_LOCAL"] == "caller-value"
        return {}

    monkeypatch.setattr(lupine, "load_native", load_native)
    lupine_autocloud.activate()

    assert os.environ["LUPINE_SERVER"] == "gpu.example:7443"
    assert os.environ["LUPINE_DISABLE_LOCAL"] == "caller-value"


def test_activate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LUPINE_AUTOCLOUD", "0")
    monkeypatch.delenv("LUPINE_SERVER", raising=False)

    def unexpected_load(**kwargs):
        raise AssertionError("unexpected load")

    monkeypatch.setattr(lupine, "load_native", unexpected_load)

    assert lupine_autocloud.activate() == {}
    assert "LUPINE_SERVER" not in os.environ
