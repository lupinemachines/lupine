from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Protocol

from .tensor import SidecarError

_DOCKER_COMPATIBLE_RUNTIMES = ("docker", "podman", "nerdctl")
SUPPORTED_RUNTIMES = ("auto", "container", *_DOCKER_COMPATIBLE_RUNTIMES)

# Inherited by the worker so its LUPINE client authenticates the same way the
# host process would. LUPINE_SESSION carries the lease a hosted gateway routes
# on; without it the gateway drops the worker's connection.
_INHERITED_ENV = ("LUPINE_SESSION",)


class ContainerRuntime(Protocol):
    name: str

    def prepare(self) -> None:
        raise NotImplementedError

    def command(self, script: str) -> list[str]:
        raise NotImplementedError


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise SidecarError(f"container runtime {name!r} is not installed")
    return executable


def _environment(
    env: dict[str, str],
    server: str | None,
) -> dict[str, str]:
    environment = dict(env)
    for name in _INHERITED_ENV:
        value = os.environ.get(name)
        # An explicit env= entry wins over the inherited one.
        if value and name not in environment:
            environment[name] = value
    if server:
        environment["LUPINE_SERVER"] = server
    return environment


@dataclass
class DockerCompatibleRuntime:
    """Launch a worker with Docker, Podman, or nerdctl."""

    name: str
    image: str
    server: str | None = None
    platform: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def _runtime(self) -> str:
        if self.name not in _DOCKER_COMPATIBLE_RUNTIMES:
            raise SidecarError(f"unsupported Docker-compatible runtime {self.name!r}")
        return _executable(self.name)

    def prepare(self) -> None:
        runtime = self._runtime()
        inspected = _run([runtime, "image", "inspect", self.image])
        if inspected.returncode == 0:
            return

        command = [runtime, "pull"]
        if self.platform:
            command.extend(["--platform", self.platform])
        command.append(self.image)
        pulled = _run(command)
        if pulled.returncode != 0:
            detail = (pulled.stderr or pulled.stdout).strip()
            raise SidecarError(
                f"{self.name} could not pull LUPINE sidecar image "
                f"{self.image!r}:\n{detail}"
            )

    def command(self, script: str) -> list[str]:
        command = [
            self._runtime(),
            "run",
            "--rm",
            "--interactive",
        ]
        if self.platform:
            command.extend(["--platform", self.platform])
        for key, value in _environment(self.env, self.server).items():
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.image, "python3", "-u", "-c", script])
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
    server: str | None = None
    platform: str | None = None
    rosetta: bool = False
    env: dict[str, str] = field(default_factory=dict)
    name: str = field(default="container", init=False)

    def _runtime(self) -> str:
        if sys.platform != "darwin":
            raise SidecarError("Apple Container sidecars are only supported on macOS")
        try:
            return _executable(self.name)
        except SidecarError as exc:
            raise SidecarError(
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
                raise SidecarError(
                    "Apple Container services are not running and automatic "
                    f"startup failed:\n{detail}"
                )

        inspected = _run([runtime, "image", "inspect", self.image])
        if inspected.returncode == 0:
            return

        pulled = _run(
            [
                runtime,
                "image",
                "pull",
                "--progress",
                "none",
                "--platform",
                self._platform,
                self.image,
            ]
        )
        if pulled.returncode != 0:
            detail = (pulled.stderr or pulled.stdout).strip()
            raise SidecarError(
                f"Apple Container could not pull LUPINE sidecar image "
                f"{self.image!r}:\n{detail}"
            )

    def command(self, script: str) -> list[str]:
        command = [
            self._runtime(),
            "run",
            "--rm",
            "--interactive",
            "--progress",
            "none",
            "--platform",
            self._platform,
        ]
        if self.rosetta:
            command.append("--rosetta")
        for key, value in _environment(self.env, self.server).items():
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.image, "python3", "-u", "-c", script])
        return command


def _runtime_names(runtime: str) -> tuple[str, ...]:
    if runtime not in SUPPORTED_RUNTIMES:
        choices = ", ".join(repr(name) for name in SUPPORTED_RUNTIMES)
        raise SidecarError(f"unsupported container runtime {runtime!r}; use {choices}")
    if runtime != "auto":
        return (runtime,)
    if sys.platform == "darwin":
        return ("container", *_DOCKER_COMPATIBLE_RUNTIMES)
    return _DOCKER_COMPATIBLE_RUNTIMES


def prepare_runtime(
    runtime: str,
    *,
    image: str,
    server: str | None,
    platform: str | None,
    rosetta: bool,
    env: dict[str, str],
) -> ContainerRuntime:
    names = _runtime_names(runtime)
    if runtime == "auto":
        names = tuple(name for name in names if shutil.which(name) is not None)
        if not names:
            expected = ", ".join(_runtime_names(runtime))
            raise SidecarError(
                f"no supported container runtime is installed; tried {expected}"
            )

    failures: list[tuple[str, str]] = []
    for name in names:
        if name == "container":
            launcher: ContainerRuntime = AppleContainerRuntime(
                image=image,
                server=server,
                platform=platform,
                rosetta=rosetta,
                env=env,
            )
        else:
            if rosetta:
                failures.append((name, "rosetta=True requires Apple Container"))
                continue
            launcher = DockerCompatibleRuntime(
                name=name,
                image=image,
                server=server,
                platform=platform,
                env=env,
            )
        try:
            launcher.prepare()
        except SidecarError as exc:
            failures.append((name, str(exc)))
        else:
            return launcher

    if len(failures) == 1:
        raise SidecarError(failures[0][1])
    detail = "\n".join(f"- {name}: {failure}" for name, failure in failures)
    raise SidecarError(f"could not prepare a container runtime:\n{detail}")
