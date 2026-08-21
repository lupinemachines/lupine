"""Native LUPINE client library loader.

The ``lupine`` wheel ships the LUPINE client shims for every supported
platform under ``lupine/_libs/<tag>/``:

============ ============================== =========================== =======================
Platform    driver shim (CUDA)             runtime shim                 NVML shim
============ ============================== =========================== =======================
linux-x86_64  libcuda.so.1                   libcudart.so.13              libnvidia-ml.so.1
linux-aarch64 libcuda.so.1                   libcudart.so.13              libnvidia-ml.so.1
macosx-*      libcuda.dylib                  libcudart.dylib              libnvidia-ml.dylib
win-amd64     nvcuda.dll                     cudart64_13.dll              nvml.dll
============ ============================== =========================== =======================

``load()`` preloads them into the process with global visibility so that
CUDA consumers resolve the LUPINE shims instead of (or, where a real NVIDIA
stack exists, in front of) the real libraries:

* PyTorch builds with CUDA link ``libcudart`` and load the driver lazily by
  soname, so preloading the LUPINE ``libcuda`` routes every later driver
  call to the LUPINE server.
* Natively compiled CUDA code (nvcc/clang) resolves both shims directly.
* On platforms without any NVIDIA runtime (macOS, GPU-less Windows hosts)
  the bundled ``libcudart`` is the only runtime available, so CUDA binaries
  keep working unmodified.

The loader is dependency-free (stdlib only) and never imports torch.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Library file names per platform tag shipped in the wheel.
_LIBS = {
    "linux": ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1"),
    "darwin": ("libcuda.dylib", "libcudart.dylib", "libnvidia-ml.dylib"),
    "win32": ("nvcuda.dll", "cudart64_13.dll", "nvml.dll"),
}

# Wheel platform tags for the non-standard entries (matches the artifact
# directory names produced by CI).
_DIR_BY_TAG = {
    "linux-x86_64": "linux-x86_64",
    "linux-aarch64": "linux-aarch64",
    "macosx-universal2": "macosx-universal2",
    "win-amd64": "win-amd64",
    "win-arm64": "win-arm64",
}

_loaded: dict[str, str] = {}


def _platform_dir() -> Path | None:
    # Under WOW64, PROCESSOR_ARCHITECTURE reports the emulated process arch
    # and PROCESSOR_ARCHITEW6432 the real machine.
    raw = os.environ.get(
        "PROCESSOR_ARCHITEW6432", os.environ.get("PROCESSOR_ARCHITECTURE", "")
    ).lower()
    machine = {
        "amd64": "amd64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "aarch64",
    }.get(raw)
    if sys.platform == "win32":
        if machine in ("amd64", "x86_64"):
            tag = "win-amd64"
        elif machine in ("arm64", "aarch64"):
            tag = "win-arm64"
        else:
            tag = None
    elif sys.platform == "darwin":
        # The wheel ships one universal2 slice set for macOS.
        tag = "macosx-universal2"
    else:
        import platform

        arch = platform.machine().lower()
        tag = "linux-x86_64" if arch in ("x86_64", "amd64") else (
            "linux-aarch64" if arch in ("aarch64", "arm64") else None
        )
    if tag is None:
        return None
    return Path(__file__).resolve().parent / "_libs" / _DIR_BY_TAG[tag]


def libdir() -> Path | None:
    """Directory holding the native shims for this platform, if bundled."""

    override = os.environ.get("LUPINE_LIBDIR")
    if override:
        return Path(override)
    return _platform_dir()


def load(*, missing_ok: bool = True) -> dict[str, str]:
    """Preload the LUPINE driver, runtime, and NVML shims.

    Returns a map of library name to loaded path. Idempotent: later calls
    only load libraries not already loaded into this process. Raises
    ``LupineError`` (from :mod:`lupine`) when a bundled library is missing
    and ``missing_ok`` is false.
    """

    from . import LupineError

    names = _LIBS.get(sys.platform)
    directory = libdir()
    if names is None or directory is None or not directory.is_dir():
        if missing_ok:
            return dict(_loaded)
        raise LupineError(
            "No bundled LUPINE native libraries for this platform "
            f"({sys.platform}); set LUPINE_LIBDIR or install a wheel with "
            "lupine._libs."
        )

    if sys.platform == "win32":
        # LoadLibrary search order includes directories added here; CUDA
        # consumers load "nvcuda.dll" by name.
        os.add_dll_directory(str(directory))

    previous = os.environ.get("LUPINE_DISABLE_LOCAL")
    try:
        # Route every device through the LUPINE server by default; the
        # bundled shims exist precisely to avoid depending on a local GPU.
        os.environ.setdefault("LUPINE_DISABLE_LOCAL", "1")
        for name in names:
            if name in _loaded:
                continue
            path = directory / name
            if not path.exists():
                if missing_ok:
                    continue
                raise LupineError(f"Bundled LUPINE library missing: {path}")
            # RTLD_GLOBAL so dlopen("libcuda.so.1") from other libraries
            # (torch's libcudart, cublas, ...) resolves to the shim.
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            _loaded[name] = str(path)
    finally:
        if previous is None:
            os.environ.pop("LUPINE_DISABLE_LOCAL", None)
        else:
            os.environ["LUPINE_DISABLE_LOCAL"] = previous
    return dict(_loaded)


def loaded() -> dict[str, str]:
    """Libraries this process loaded through :func:`load`, by name."""

    return dict(_loaded)
