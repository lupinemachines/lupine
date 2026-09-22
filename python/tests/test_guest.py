import os
import sys

import pytest

import lupine
from lupine import LupineError, _guest, container


def test_select_worker_image_picks_newest_supported_cuda():
    tags = [
        "torch2.12.1-cu126",
        "torch2.12.1-cu129",
        "torch2.12.1-cu130",
        "torch2.11.0-cu130",
        "latest",
    ]
    assert _guest.select_worker_image("2.12.1+cpu", "13.3", tags) == (
        f"{_guest.WORKER_IMAGE_REPOSITORY}:torch2.12.1-cu130"
    )
    assert _guest.select_worker_image("2.12.1", "12.9", tags).endswith("cu129")
    assert _guest.select_worker_image("2.12.1", "12.8", tags).endswith("cu126")


def test_select_worker_image_requires_matching_torch_release():
    with pytest.raises(LupineError):
        _guest.select_worker_image("2.10.0", "13.3", ["torch2.12.1-cu130"])
    with pytest.raises(LupineError):
        _guest.select_worker_image("2.12.1", "12.4", ["torch2.12.1-cu130"])


def test_docker_command_publishes_the_worker_port():
    runtime = container.DockerCompatibleRuntime(
        name="docker", image="img:tag", platform="linux/arm64", env={"LUPINE_SERVER": "h:1"}
    )
    runtime._runtime = lambda: "/usr/bin/docker"  # type: ignore[method-assign]
    command = runtime.command(["lupine-torch-worker", "--listen", "0.0.0.0:14834"], (43210, 14834))
    assert command[:5] == ["/usr/bin/docker", "run", "--rm", "--interactive", "--platform"]
    assert "127.0.0.1:43210:14834" in command
    assert "LUPINE_SERVER=h:1" in command
    assert command[-3:] == ["lupine-torch-worker", "--listen", "0.0.0.0:14834"]


def test_apple_container_command_publishes_the_worker_port():
    runtime = container.AppleContainerRuntime(image="img:tag", env={})
    runtime._runtime = lambda: "/usr/local/bin/container"  # type: ignore[method-assign]
    command = runtime.command(["lupine-torch-worker"], (5000, 14834))
    assert command[0] == "/usr/local/bin/container"
    assert "127.0.0.1:5000:14834/tcp" in command
    assert "linux/arm64" in command


def test_runtime_names_reject_unknown_runtime():
    with pytest.raises(container.ContainerError):
        container.prepare_runtime("rkt", image="img", platform=None, env={})


def test_torch_backend_selected_follows_env_then_platform(monkeypatch):
    monkeypatch.setenv("LUPINE_TORCH_BACKEND", "1")
    assert lupine.torch_backend_selected()
    monkeypatch.setenv("LUPINE_TORCH_BACKEND", "0")
    assert not lupine.torch_backend_selected()
    monkeypatch.delenv("LUPINE_TORCH_BACKEND")
    assert lupine.torch_backend_selected() == (sys.platform == "darwin")


def test_subprocess_worker_needs_an_interpreter(monkeypatch):
    monkeypatch.delenv("LUPINE_WORKER_PYTHON", raising=False)
    with pytest.raises(LupineError):
        _guest._subprocess_command(("h:1",))


def test_worker_environment_inherits_session_and_devices(monkeypatch):
    monkeypatch.setenv("LUPINE_SESSION", "lease")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    monkeypatch.setenv("LUPINE_WORKER_LIBDIR", "/tmp/libs")
    env = _guest._worker_environment(("a:1", "b:2"))
    assert env["LUPINE_SERVER"] == "a:1,b:2"
    assert env["LUPINE_SESSION"] == "lease"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert env["LUPINE_LIBDIR"] == "/tmp/libs"
    assert os.environ.get("LUPINE_LIBDIR") is None
