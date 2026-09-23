"""Provisioning of the torch worker for the torch backend.

The worker is a CUDA torch of the host's major.minor release: a subprocess of
another interpreter that has one (``LUPINE_WORKER_PYTHON``), or one started by hand
and attached by address (``LUPINE_WORKER=host:port``). It runs the client
shims against the GPU server, so its CUDA traffic is the ordinary driver
path. macOS has no CUDA torch to run as a subprocess; the container worker
for it follows in its own PR (branch ``python/torch-worker-container``), so
there only attaching works for now.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import IO

from . import LupineError
from ._worker import READY_PREFIX

# Passed through to the worker so it authenticates and selects devices the
# way the host process would.
_INHERITED_ENV = ("LUPINE_SESSION", "CUDA_VISIBLE_DEVICES", "LUPINE_LOG_LEVEL", "LUPINE_WORKER_DEVICE")


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
            from . import _backend

            _backend.close()
        except Exception as exc:
            print(f"lupine: worker disconnect failed: {exc}", file=sys.stderr)
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


def _subprocess_command(servers: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    python = os.environ.get("LUPINE_WORKER_PYTHON")
    if not python:
        raise LupineError(
            "LUPINE_WORKER_PYTHON must name an interpreter with a CUDA torch of the "
            "same major.minor release to run the torch worker as a subprocess"
        )
    env = dict(os.environ)
    for name in (
        "LUPINE_TORCH_BACKEND",
        "LUPINE_LIBDIR",
        "LUPINE_RPC_STATS",
        "LUPINE_TORCH_PROFILE",
        "LUPINE_TORCH_TRACE",
    ):
        env.pop(name, None)
    env.update(_worker_environment(servers))
    return [python, "-m", "lupine._worker", "--listen", "127.0.0.1:0"], env


def _forward(stream: IO[str]) -> None:
    for line in stream:
        sys.stderr.write(line)
        sys.stderr.flush()


def _spawn(command: list[str], env: dict[str, str]) -> Worker:
    process = subprocess.Popen(
        command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    output: list[str] = []
    for line in process.stdout:
        if line.startswith(READY_PREFIX):
            _, port, _ = line.split(maxsplit=2)
            threading.Thread(target=_forward, args=(process.stdout,), daemon=True).start()
            worker = Worker(address=f"127.0.0.1:{port}", process=process)
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
    elif sys.platform == "darwin":
        raise LupineError(
            "no torch worker for macOS: it has no CUDA torch to run one as a "
            "subprocess, and the container worker follows in its own PR (branch "
            "python/torch-worker-container); set LUPINE_WORKER=host:port to attach "
            "to a worker started on a Linux machine with `lupine-torch-worker "
            "--listen host:port`"
        )
    else:
        command, env = _subprocess_command(servers)
        worker = _spawn(command, env)
    try:
        _backend.start(worker.address)
    except BaseException:
        worker.close()
        raise
    return worker
