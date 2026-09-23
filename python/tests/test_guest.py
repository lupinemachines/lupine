import os
import sys

import pytest

import lupine
import lupine._guest


def test_torch_backend_selected_follows_env_then_platform(monkeypatch):
    monkeypatch.setenv("LUPINE_TORCH_BACKEND", "1")
    assert lupine.torch_backend_selected()
    monkeypatch.setenv("LUPINE_TORCH_BACKEND", "0")
    assert not lupine.torch_backend_selected()
    monkeypatch.delenv("LUPINE_TORCH_BACKEND")
    assert lupine.torch_backend_selected() == (sys.platform == "darwin")


def test_subprocess_worker_needs_an_interpreter(monkeypatch):
    monkeypatch.delenv("LUPINE_WORKER_PYTHON", raising=False)
    with pytest.raises(lupine.LupineError):
        lupine._guest._subprocess_command(("h:1",))


def test_worker_environment_inherits_session_and_devices(monkeypatch):
    monkeypatch.setenv("LUPINE_SESSION", "lease")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    monkeypatch.setenv("LUPINE_WORKER_LIBDIR", "/tmp/libs")
    env = lupine._guest._worker_environment(("a:1", "b:2"))
    assert env["LUPINE_SERVER"] == "a:1,b:2"
    assert env["LUPINE_SESSION"] == "lease"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert env["LUPINE_LIBDIR"] == "/tmp/libs"
    assert os.environ.get("LUPINE_LIBDIR") is None


def test_macos_without_a_worker_to_attach_names_the_follow_up(monkeypatch):
    monkeypatch.delenv("LUPINE_WORKER", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(lupine.LupineError, match="python/torch-worker-container"):
        lupine._guest.start(("h:1",))
