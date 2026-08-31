"""Native LUPINE client library loader.

LUPINE servers publish the compatible client shims for every supported
platform. The loader downloads the exact object selected by ``LUPINE_SERVER``
before CUDA consumers are imported. Wheels may retain ``lupine/_libs`` as a
release-transition fallback when no server is configured.

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
  the selected ``libcudart`` is the only runtime available, so CUDA binaries
  keep working unmodified.

The loader is dependency-free (stdlib only) and never imports torch.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from . import _bundles

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

_TAG_BY_PLATFORM = {
    "linux/amd64": "linux-x86_64",
    "linux/arm64": "linux-aarch64",
    "macos/amd64": "macosx-universal2",
    "macos/arm64": "macosx-universal2",
    "windows/amd64": "win-amd64",
    "windows/arm64": "win-arm64",
}

_loaded: dict[str, str] = {}


def _platform_dir() -> Path | None:
    platform_key = _bundles.platform_name()
    tag = None if platform_key is None else _TAG_BY_PLATFORM.get(platform_key)
    if tag is None:
        return None
    return Path(__file__).resolve().parent / "_libs" / _DIR_BY_TAG[tag]


def libdir() -> Path | None:
    """Directory holding the selected native shims for this platform."""

    override = os.environ.get("LUPINE_LIBDIR")
    if override:
        return Path(override)
    servers = tuple(
        value.strip()
        for value in os.environ.get("LUPINE_SERVER", "").split(",")
        if value.strip()
    )
    if servers:
        from . import LupineError

        names = _LIBS.get(sys.platform)
        if names is None:
            raise LupineError(f"Unsupported LUPINE client platform: {sys.platform}")
        try:
            directory, etag, platform_key = _bundles.resolve(servers, names)
        except Exception as exc:
            raise LupineError(
                f"Could not resolve the server's LUPINE client: {exc}"
            ) from exc
        os.environ["LUPINE_LIBDIR"] = str(directory)
        os.environ["LUPINE_CLIENT_ETAG"] = etag
        os.environ["LUPINE_CLIENT_PLATFORM"] = platform_key
        return directory
    return _platform_dir()


def load(*, missing_ok: bool = True) -> dict[str, str]:
    """Preload the LUPINE driver, runtime, and NVML shims.

    Returns a map of library name to loaded path. Idempotent: later calls
    only load libraries not already loaded into this process. Raises
    ``LupineError`` (from :mod:`lupine`) when a selected library is missing
    and ``missing_ok`` is false.
    """

    from . import LupineError

    names = _LIBS.get(sys.platform)
    try:
        directory = libdir()
    except LupineError:
        if missing_ok:
            return dict(_loaded)
        raise
    if names is None or directory is None or not directory.is_dir():
        if missing_ok:
            return dict(_loaded)
        raise LupineError(
            "No LUPINE native libraries for this platform "
            f"({sys.platform}); set LUPINE_LIBDIR or install a wheel with "
            "lupine._libs."
        )

    if sys.platform == "win32":
        # LoadLibrary search order includes directories added here; CUDA
        # consumers load "nvcuda.dll" by name.
        os.add_dll_directory(str(directory))
    elif sys.platform in ("linux", "darwin") and (directory / names[0]).is_file():
        # Triton compiles a small launcher against libcuda and therefore needs
        # a filesystem directory even when the shim is already RTLD_GLOBAL.
        os.environ.setdefault("TRITON_LIBCUDA_PATH", str(directory))

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
    return dict(_loaded)


def loaded() -> dict[str, str]:
    """Libraries this process loaded through :func:`load`, by name."""

    return dict(_loaded)
