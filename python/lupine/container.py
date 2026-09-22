"""Container runtimes that can host the torch worker on a machine with no
NVIDIA runtime: Apple Container first, then Docker, Podman and nerdctl."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Protocol

_DOCKER_COMPATIBLE_RUNTIMES = ("docker", "podman", "nerdctl")
SUPPORTED_RUNTIMES = ("auto", "container", *_DOCKER_COMPATIBLE_RUNTIMES)


class ContainerError(RuntimeError):
    """Raised when no container runtime can host the worker."""


class ContainerRuntime(Protocol):
    name: str

    def prepare(self) -> None:
        raise NotImplementedError

    def command(self, args: list[str], publish: tuple[int, int]) -> list[str]:
        raise NotImplementedError


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ContainerError(f"container runtime {name!r} is not installed")
    return executable


def _env_arguments(env: dict[str, str]) -> list[str]:
    arguments: list[str] = []
    for key, value in env.items():
        arguments.extend(["--env", f"{key}={value}"])
    return arguments


@dataclass
class DockerCompatibleRuntime:
    """Launch the worker with Docker, Podman, or nerdctl."""

    name: str
    image: str
    platform: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def _runtime(self) -> str:
        if self.name not in _DOCKER_COMPATIBLE_RUNTIMES:
            raise ContainerError(f"unsupported Docker-compatible runtime {self.name!r}")
        return _executable(self.name)

    def prepare(self) -> None:
        runtime = self._runtime()
        if _run([runtime, "image", "inspect", self.image]).returncode == 0:
            return
        command = [runtime, "pull"]
        if self.platform:
            command.extend(["--platform", self.platform])
        command.append(self.image)
        pulled = _run(command)
        if pulled.returncode != 0:
            detail = (pulled.stderr or pulled.stdout).strip()
            raise ContainerError(
                f"{self.name} could not pull the LUPINE worker image {self.image!r}:\n{detail}"
            )

    def command(self, args: list[str], publish: tuple[int, int]) -> list[str]:
        host_port, container_port = publish
        command = [self._runtime(), "run", "--rm", "--interactive"]
        if self.platform:
            command.extend(["--platform", self.platform])
        command.extend(["--publish", f"127.0.0.1:{host_port}:{container_port}"])
        command.extend(_env_arguments(self.env))
        command.append(self.image)
        command.extend(args)
        return command


def _system_running(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return "running" in output.lower()
    return payload.get("status") == "running"


@dataclass
class AppleContainerRuntime:
    """Launch a Linux worker with Apple's Container CLI."""

    image: str
    platform: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    name: str = field(default="container", init=False)

    def _runtime(self) -> str:
        if sys.platform != "darwin":
            raise ContainerError("Apple Container is only available on macOS")
        try:
            return _executable(self.name)
        except ContainerError as exc:
            raise ContainerError(
                "Apple Container CLI is not installed. Install it with "
                "`brew install --cask container`, or download the signed installer "
                "from https://github.com/apple/container/releases."
            ) from exc

    @property
    def _platform(self) -> str:
        return self.platform or "linux/arm64"

    def prepare(self) -> None:
        runtime = self._runtime()
        status = _run([runtime, "system", "status", "--format", "json"])
        if status.returncode != 0 or not _system_running(status.stdout):
            started = _run([runtime, "system", "start"])
            if started.returncode != 0:
                detail = (started.stderr or started.stdout).strip()
                raise ContainerError(
                    "Apple Container services are not running and automatic "
                    f"startup failed:\n{detail}"
                )
        if _run([runtime, "image", "inspect", self.image]).returncode == 0:
            return
        pulled = _run(
            [runtime, "image", "pull", "--progress", "none", "--platform", self._platform, self.image]
        )
        if pulled.returncode != 0:
            detail = (pulled.stderr or pulled.stdout).strip()
            raise ContainerError(
                f"Apple Container could not pull the LUPINE worker image {self.image!r}:\n{detail}"
            )

    def command(self, args: list[str], publish: tuple[int, int]) -> list[str]:
        host_port, container_port = publish
        command = [
            self._runtime(),
            "run",
            "--rm",
            "--interactive",
            "--platform",
            self._platform,
            "--publish",
            f"127.0.0.1:{host_port}:{container_port}/tcp",
        ]
        command.extend(_env_arguments(self.env))
        command.append(self.image)
        command.extend(args)
        return command


def _runtime_names(runtime: str) -> tuple[str, ...]:
    if runtime not in SUPPORTED_RUNTIMES:
        choices = ", ".join(repr(name) for name in SUPPORTED_RUNTIMES)
        raise ContainerError(f"unsupported container runtime {runtime!r}; use {choices}")
    if runtime != "auto":
        return (runtime,)
    if sys.platform == "darwin":
        return ("container", *_DOCKER_COMPATIBLE_RUNTIMES)
    return _DOCKER_COMPATIBLE_RUNTIMES


def prepare_runtime(
    runtime: str,
    *,
    image: str,
    platform: str | None,
    env: dict[str, str],
) -> ContainerRuntime:
    """Pick the first installed runtime that can provide ``image``."""

    names = _runtime_names(runtime)
    if runtime == "auto":
        names = tuple(name for name in names if shutil.which(name) is not None)
        if not names:
            expected = ", ".join(_runtime_names(runtime))
            raise ContainerError(
                f"no supported container runtime is installed; tried {expected}"
            )

    failures: list[tuple[str, str]] = []
    for name in names:
        launcher: ContainerRuntime
        if name == "container":
            launcher = AppleContainerRuntime(image=image, platform=platform, env=env)
        else:
            launcher = DockerCompatibleRuntime(
                name=name, image=image, platform=platform, env=env
            )
        try:
            launcher.prepare()
        except ContainerError as exc:
            failures.append((name, str(exc)))
        else:
            return launcher

    if len(failures) == 1:
        raise ContainerError(failures[0][1])
    detail = "\n".join(f"- {name}: {failure}" for name, failure in failures)
    raise ContainerError(f"could not prepare a container runtime:\n{detail}")


def inherited_environment(names: tuple[str, ...]) -> dict[str, str]:
    return {name: os.environ[name] for name in names if os.environ.get(name)}
