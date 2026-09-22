"""Provisioning of the torch worker for the torch backend.

The worker is a CUDA torch of the host's release. On Linux and Windows it is
a subprocess of another interpreter (``LUPINE_WORKER_PYTHON``); on macOS,
where no CUDA torch exists, it is an arm64 Linux container running the
published worker image, whose port is published into the host. Either way
the worker runs the client shims against the GPU server, so its CUDA traffic
is the ordinary driver path.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import IO, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from . import LupineError
from ._worker import READY_PREFIX

WORKER_REGISTRY_HOST = "ghcr.io"
WORKER_REPOSITORY = "lupinemachines/lupine-pytorch-worker"
WORKER_IMAGE_REPOSITORY = f"{WORKER_REGISTRY_HOST}/{WORKER_REPOSITORY}"
# The port the worker listens on inside a container; the runtime publishes it
# to an ephemeral port on the host loopback.
CONTAINER_WORKER_PORT = 14834
_CUDA_VERSION_HEADER = "x-lupine-cuda-version"
_TAG = re.compile(r"^torch(?P<torch>\d+\.\d+\.\d+)-cu(?P<cuda>\d{3})$")

# Passed through to the worker so it authenticates and selects devices the
# way the host process would.
_INHERITED_ENV = ("LUPINE_SESSION", "CUDA_VISIBLE_DEVICES", "LUPINE_LOG_LEVEL")


@dataclass
class Worker:
    address: str
    process: subprocess.Popen[str] | None = None

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        # Dropping the connection lets the worker leave through its own exit
        # path (LUPINE_RPC_STATS dumps at exit); a stuck one is terminated.
        try:
            from ._backend import _extension

            _extension().disconnect()
        except Exception:
            pass
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _server_url(server: str) -> str:
    value = server if "://" in server else f"http://{server}"
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))


def server_cuda_version(server: str) -> str:
    """The CUDA version a server advertises, e.g. ``13.3``."""

    headers = {}
    session = os.environ.get("LUPINE_SESSION")
    if session:
        headers["x-lupine-session"] = session
    request = urllib.request.Request(_server_url(server), headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            version = response.headers.get(_CUDA_VERSION_HEADER)
    except (urllib.error.URLError, OSError) as exc:
        raise LupineError(f"could not query LUPINE server {server!r}: {exc}") from exc
    if not version:
        raise LupineError(
            f"LUPINE server {server!r} did not advertise {_CUDA_VERSION_HEADER}; "
            "set LUPINE_WORKER_IMAGE"
        )
    return version.strip()


def _registry_json(path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://{WORKER_REGISTRY_HOST}{path}", headers=headers or {}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise LupineError(
            f"could not query the worker image registry: {exc}; set LUPINE_WORKER_IMAGE"
        ) from exc
    if not isinstance(payload, dict):
        raise LupineError("the worker image registry returned an invalid response")
    return payload


def worker_image_tags() -> list[str]:
    query = urlencode(
        {"service": WORKER_REGISTRY_HOST, "scope": f"repository:{WORKER_REPOSITORY}:pull"}
    )
    token = _registry_json(f"/token?{query}").get("token")
    if not isinstance(token, str) or not token:
        raise LupineError("the worker image registry did not return an access token")
    payload = _registry_json(
        f"/v2/{WORKER_REPOSITORY}/tags/list?n=1000", {"Authorization": f"Bearer {token}"}
    )
    tags = payload.get("tags")
    return [tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else []


def select_worker_image(torch_version: str, cuda_version: str, tags: list[str]) -> str:
    """Pick ``torch<host release>-cu<xyz>`` with the newest CUDA the server runs.

    A torch built for CUDA ``x.y`` needs a driver of at least ``x.y``, and
    the server's CUDA version states what its driver supports.
    """

    from ._backend import base_version

    wanted = base_version(torch_version)
    server = tuple(int(part) for part in cuda_version.split(".")[:2])
    candidates: list[tuple[tuple[int, int], str]] = []
    for tag in tags:
        match = _TAG.match(tag)
        if match is None or match.group("torch") != wanted:
            continue
        cuda = match.group("cuda")
        needed = (int(cuda[:2]), int(cuda[2:]))
        if needed <= server:
            candidates.append((needed, tag))
    if not candidates:
        raise LupineError(
            f"no published worker image runs torch {wanted} on a CUDA {cuda_version} "
            f"server; set LUPINE_WORKER_IMAGE"
        )
    return f"{WORKER_IMAGE_REPOSITORY}:{max(candidates)[1]}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _worker_environment(servers: tuple[str, ...]) -> dict[str, str]:
    env = {name: os.environ[name] for name in _INHERITED_ENV if os.environ.get(name)}
    env["LUPINE_SERVER"] = ",".join(servers)
    env["PYTHONUNBUFFERED"] = "1"
    libdir = os.environ.get("LUPINE_WORKER_LIBDIR")
    if libdir:
        env["LUPINE_LIBDIR"] = libdir
    stats = os.environ.get("LUPINE_WORKER_RPC_STATS")
    if stats:
        env["LUPINE_RPC_STATS"] = stats
    profile = os.environ.get("LUPINE_WORKER_TORCH_PROFILE")
    if profile:
        env["LUPINE_TORCH_PROFILE"] = profile
    return env


def _subprocess_command(servers: tuple[str, ...]) -> tuple[list[str], dict[str, str], str | None]:
    python = os.environ.get("LUPINE_WORKER_PYTHON")
    if not python:
        raise LupineError(
            "LUPINE_WORKER_PYTHON must name an interpreter with a CUDA torch of the "
            "same release to run the torch worker as a subprocess"
        )
    env = dict(os.environ)
    env.pop("LUPINE_TORCH_BACKEND", None)
    env.pop("LUPINE_LIBDIR", None)
    env.pop("LUPINE_RPC_STATS", None)
    env.pop("LUPINE_TORCH_PROFILE", None)
    env.pop("LUPINE_TORCH_TRACE", None)
    env.update(_worker_environment(servers))
    return [python, "-m", "lupine._worker", "--listen", "127.0.0.1:0"], env, None


def _container_command(
    servers: tuple[str, ...], runtime: str
) -> tuple[list[str], dict[str, str], str | None]:
    from . import container

    image = os.environ.get("LUPINE_WORKER_IMAGE")
    if not image:
        import torch

        image = select_worker_image(
            torch.__version__, server_cuda_version(servers[0]), worker_image_tags()
        )
    try:
        launcher = container.prepare_runtime(
            runtime,
            image=image,
            platform=os.environ.get("LUPINE_WORKER_PLATFORM"),
            env=_worker_environment(servers),
        )
    except container.ContainerError as exc:
        raise LupineError(str(exc)) from exc
    host_port = _free_port()
    command = launcher.command(
        ["lupine-torch-worker", "--listen", f"0.0.0.0:{CONTAINER_WORKER_PORT}"],
        (host_port, CONTAINER_WORKER_PORT),
    )
    return command, dict(os.environ), f"127.0.0.1:{host_port}"


def _forward(stream: IO[str]) -> None:
    for line in stream:
        sys.stderr.write(line)
        sys.stderr.flush()


def _spawn(command: list[str], env: dict[str, str], address: str | None) -> Worker:
    process = subprocess.Popen(
        command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    output: list[str] = []
    for line in process.stdout:
        if line.startswith(READY_PREFIX):
            _, port, _ = line.split(maxsplit=2)
            if address is None:
                address = f"127.0.0.1:{port}"
            threading.Thread(target=_forward, args=(process.stdout,), daemon=True).start()
            worker = Worker(address=address, process=process)
            atexit.register(worker.close)
            return worker
        output.append(line)
    process.wait()
    raise LupineError(
        f"the torch worker exited with status {process.returncode} before it was ready:\n"
        + "".join(output)
    )


def start(servers: tuple[str, ...]) -> Worker:
    """Start (or attach to) the worker and load the torch backend on it."""

    from . import _backend

    if _backend.is_started():
        return Worker(address=_backend._started["address"])
    attach = os.environ.get("LUPINE_WORKER")
    if attach:
        worker = Worker(address=attach)
    else:
        runtime = os.environ.get("LUPINE_WORKER_RUNTIME")
        if runtime is None:
            runtime = "auto" if sys.platform == "darwin" else "subprocess"
        if runtime == "subprocess":
            command, env, address = _subprocess_command(servers)
        else:
            command, env, address = _container_command(servers, runtime)
        worker = _spawn(command, env, address)
    try:
        _backend.start(worker.address)
    except BaseException:
        worker.close()
        raise
    return worker
