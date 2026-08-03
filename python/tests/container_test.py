import subprocess

import pytest

pytest.importorskip("torch")
container_support = pytest.importorskip("lupine.container")

_IMAGE = "registry.example/lupine-worker:cuda"


def test_apple_container_defaults_to_arm64(monkeypatch):
    monkeypatch.setattr(
        container_support.shutil, "which", lambda name: "/usr/bin/container"
    )
    monkeypatch.setattr(container_support.sys, "platform", "darwin")

    command = container_support.AppleContainerRuntime(
        image=_IMAGE,
        server="host-a:14833",
    ).command("print(1)")

    assert command[:8] == [
        "/usr/bin/container",
        "run",
        "--rm",
        "--interactive",
        "--progress",
        "none",
        "--platform",
        "linux/arm64",
    ]
    assert "--rosetta" not in command
    assert "LUPINE_SERVER=host-a:14833" in command


def test_apple_container_is_macos_only(monkeypatch):
    monkeypatch.setattr(container_support.sys, "platform", "linux")

    with pytest.raises(container_support.SidecarError, match="only supported on macOS"):
        container_support.AppleContainerRuntime(image=_IMAGE).command("print(1)")


def test_apple_container_requires_cli(monkeypatch):
    monkeypatch.setattr(container_support.shutil, "which", lambda name: None)
    monkeypatch.setattr(container_support.sys, "platform", "darwin")

    with pytest.raises(
        container_support.SidecarError,
        match="brew install --cask container",
    ):
        container_support.AppleContainerRuntime(image=_IMAGE).command("print(1)")


def test_apple_container_starts_services(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["system", "status", "--format"]:
            return subprocess.CompletedProcess(args, 0, '{"status":"stopped"}', "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(
        container_support.shutil, "which", lambda name: "/usr/bin/container"
    )
    monkeypatch.setattr(container_support.sys, "platform", "darwin")
    monkeypatch.setattr(container_support.subprocess, "run", fake_run)

    container_support.AppleContainerRuntime(image=_IMAGE).prepare()

    assert calls == [
        ["/usr/bin/container", "system", "status", "--format", "json"],
        ["/usr/bin/container", "system", "start"],
        ["/usr/bin/container", "image", "inspect", _IMAGE],
    ]


def test_apple_container_pulls_missing_image(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(args, 1, "", "missing")
        return subprocess.CompletedProcess(args, 0, '{"status":"running"}', "")

    monkeypatch.setattr(
        container_support.shutil, "which", lambda name: "/usr/bin/container"
    )
    monkeypatch.setattr(container_support.sys, "platform", "darwin")
    monkeypatch.setattr(container_support.subprocess, "run", fake_run)

    container_support.AppleContainerRuntime(image=_IMAGE).prepare()

    assert calls[-1] == [
        "/usr/bin/container",
        "image",
        "pull",
        "--progress",
        "none",
        "--platform",
        "linux/arm64",
        _IMAGE,
    ]


@pytest.mark.parametrize("runtime", ["docker", "podman", "nerdctl"])
def test_docker_compatible_runtime_command(runtime, monkeypatch):
    monkeypatch.setattr(
        container_support.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    command = container_support.DockerCompatibleRuntime(
        name=runtime,
        image=_IMAGE,
        server="host-a:14833",
        env={"EXAMPLE": "value"},
    ).command("print(1)")

    assert command[:4] == [
        f"/usr/bin/{runtime}",
        "run",
        "--rm",
        "--interactive",
    ]
    assert "--platform" not in command
    assert "EXAMPLE=value" in command
    assert "LUPINE_SERVER=host-a:14833" in command


def test_docker_compatible_runtime_pulls_requested_platform(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            1 if args[1:3] == ["image", "inspect"] else 0,
            "",
            "",
        )

    monkeypatch.setattr(
        container_support.shutil, "which", lambda name: "/usr/bin/podman"
    )
    monkeypatch.setattr(container_support.subprocess, "run", fake_run)

    container_support.DockerCompatibleRuntime(
        name="podman",
        image=_IMAGE,
        platform="linux/amd64",
    ).prepare()

    assert calls[-1] == [
        "/usr/bin/podman",
        "pull",
        "--platform",
        "linux/amd64",
        _IMAGE,
    ]


def test_auto_runtime_falls_back_to_next_installed_runtime(monkeypatch):
    def fake_run(args, **kwargs):
        if args[0] == "/usr/bin/docker":
            return subprocess.CompletedProcess(args, 1, "", "daemon unavailable")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(container_support.sys, "platform", "linux")
    monkeypatch.setattr(
        container_support.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"docker", "podman"} else None,
    )
    monkeypatch.setattr(container_support.subprocess, "run", fake_run)

    runtime = container_support.prepare_runtime(
        "auto",
        image=_IMAGE,
        server="host-a:14833",
        platform=None,
        rosetta=False,
        env={},
    )

    assert runtime.name == "podman"


def test_auto_runtime_prefers_apple_container_on_macos(monkeypatch):
    monkeypatch.setattr(container_support.sys, "platform", "darwin")
    monkeypatch.setattr(
        container_support.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        container_support.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            '{"status":"running"}',
            "",
        ),
    )

    runtime = container_support.prepare_runtime(
        "auto",
        image=_IMAGE,
        server="host-a:14833",
        platform=None,
        rosetta=False,
        env={},
    )

    assert runtime.name == "container"


def test_prepare_runtime_rejects_unknown_runtime():
    with pytest.raises(
        container_support.SidecarError,
        match="unsupported container runtime",
    ):
        container_support.prepare_runtime(
            "runc",
            image=_IMAGE,
            server="host-a:14833",
            platform=None,
            rosetta=False,
            env={},
        )


def test_environment_inherits_lease_from_process(monkeypatch):
    monkeypatch.setenv("LUPINE_SESSION", "lease-123")

    environment = container_support._environment({}, "host-a:14833")

    assert environment == {
        "LUPINE_SESSION": "lease-123",
        "LUPINE_SERVER": "host-a:14833",
    }


def test_environment_prefers_explicit_lease(monkeypatch):
    monkeypatch.setenv("LUPINE_SESSION", "inherited")

    environment = container_support._environment({"LUPINE_SESSION": "explicit"}, None)

    assert environment == {"LUPINE_SESSION": "explicit"}


def test_environment_omits_lease_when_unset(monkeypatch):
    monkeypatch.delenv("LUPINE_SESSION", raising=False)

    assert container_support._environment({}, None) == {}
