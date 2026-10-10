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


def test_native_backend_does_not_provision_a_worker(monkeypatch):
    from lupine._backend import native

    calls = []
    monkeypatch.setenv("LUPINE_TORCH_BACKEND", "native")
    monkeypatch.setattr(native, "start", lambda: calls.append("native"))

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("the native backend must not spawn a torch worker")

    monkeypatch.setattr(lupine._guest, "_spawn", unexpected_spawn)
    worker = lupine._guest.start(("gpu:14833",))
    assert calls == ["native"]
    assert worker.address == "native" and worker.process is None


def test_subprocess_worker_needs_an_interpreter(monkeypatch):
    monkeypatch.delenv("LUPINE_WORKER_PYTHON", raising=False)
    with pytest.raises(lupine.LupineError):
        lupine._guest._subprocess_command(("h:1",))


def test_worker_environment_sets_server_and_libdir(monkeypatch):
    monkeypatch.setenv("LUPINE_WORKER_LIBDIR", "/tmp/libs")
    env = lupine._guest._worker_environment(("a:1", "b:2"))
    assert env["LUPINE_SERVER"] == "a:1,b:2"
    assert env["LUPINE_LIBDIR"] == "/tmp/libs"
    assert os.environ.get("LUPINE_LIBDIR") is None


def test_subprocess_command_inherits_session_and_devices(monkeypatch):
    monkeypatch.setenv("LUPINE_WORKER_PYTHON", "python3")
    monkeypatch.setenv("LUPINE_SESSION", "lease")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    _, env = lupine._guest._subprocess_command(("a:1", "b:2"))
    assert env["LUPINE_SERVER"] == "a:1,b:2"
    assert env["LUPINE_SESSION"] == "lease"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"


def test_macos_without_a_worker_to_attach_names_the_follow_up(monkeypatch):
    monkeypatch.delenv("LUPINE_WORKER", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(lupine.LupineError, match="python/torch-worker-container"):
        lupine._guest.start(("h:1",))
