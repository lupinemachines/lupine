import os
from types import SimpleNamespace

import lupine_auto
import pytest

import lupine


def test_activate_defaults_to_cloud_without_changing_local_policy(monkeypatch):
    monkeypatch.delenv("LUPINE_AUTO", raising=False)
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.delenv("LUPINE_DISABLE_LOCAL", raising=False)
    monkeypatch.setenv("LUPINE_SESSION", "lease-test")
    calls = []
    monkeypatch.setattr(
        lupine,
        "load_native",
        lambda *, missing_ok: calls.append(missing_ok) or {"libcuda": "/shim"},
    )

    assert lupine_auto.activate() == {"libcuda": "/shim"}
    assert lupine_auto.DEFAULT_SERVER == "https://api.lupine.sh"
    assert os.environ["LUPINE_SERVER"] == lupine_auto.DEFAULT_SERVER
    assert "LUPINE_DISABLE_LOCAL" not in os.environ
    assert calls == [False]


def test_activate_acquires_cloud_session_when_unbound(monkeypatch):
    monkeypatch.delenv("LUPINE_AUTO", raising=False)
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.delenv("LUPINE_SESSION", raising=False)
    monkeypatch.setattr(lupine_auto, "_cloud_session", None)
    calls = []
    monkeypatch.setattr(
        lupine,
        "cloud",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(loaded={"libcuda": "/cloud/shim"}),
    )

    assert lupine_auto.activate() == {"libcuda": "/cloud/shim"}
    assert calls == [
        {
            "api_url": "https://api.lupine.sh",
            "gpu_type": None,
            "gpu_count": None,
            "region": None,
        }
    ]


def test_startup_shows_login_hint_without_aborting(monkeypatch):
    monkeypatch.setattr(
        lupine_auto,
        "activate",
        lambda: (_ for _ in ()).throw(
            lupine.LupineAuthenticationError(
                "Run `uvx lupine login` or `python -m lupine login`."
            )
        ),
    )

    with pytest.warns(RuntimeWarning, match="python -m lupine login"):
        assert lupine_auto.startup() == {}


def test_install_defers_activation_until_cuda_import(monkeypatch):
    monkeypatch.delenv("LUPINE_AUTO", raising=False)
    monkeypatch.setattr(lupine_auto, "_finder", None)
    calls = []
    monkeypatch.setattr(lupine_auto, "startup", lambda: calls.append(True) or {})

    finder = lupine_auto.install()
    assert finder is not None
    assert calls == []
    assert finder.find_spec("json", None) is None
    assert calls == []
    assert finder.find_spec("torch", None) is None
    assert calls == [True]


def test_activate_respects_explicit_server_and_local_policy(monkeypatch):
    monkeypatch.delenv("LUPINE_AUTO", raising=False)
    monkeypatch.setenv("LUPINE_SERVER", "gpu.example:7443")
    monkeypatch.setenv("LUPINE_DISABLE_LOCAL", "caller-value")

    def load_native(*, missing_ok):
        assert missing_ok is False
        assert os.environ["LUPINE_DISABLE_LOCAL"] == "caller-value"
        return {}

    monkeypatch.setattr(lupine, "load_native", load_native)
    lupine_auto.activate()

    assert os.environ["LUPINE_SERVER"] == "gpu.example:7443"
    assert os.environ["LUPINE_DISABLE_LOCAL"] == "caller-value"


def test_activate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LUPINE_AUTO", "0")
    monkeypatch.delenv("LUPINE_SERVER", raising=False)

    def unexpected_load(**kwargs):
        raise AssertionError("unexpected load")

    monkeypatch.setattr(lupine, "load_native", unexpected_load)

    assert lupine_auto.activate() == {}
    assert "LUPINE_SERVER" not in os.environ
