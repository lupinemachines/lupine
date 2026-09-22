"""Native LUPINE client library loader.

LUPINE servers publish the compatible client shims for every supported
platform. The loader downloads the exact bundle selected by ``LUPINE_SERVER``
before CUDA consumers are imported. The wheel itself is pure Python: every
native object comes from a bound server or an explicit ``LUPINE_LIBDIR``.

A bundle always carries the driver and NVML shims, and on Linux the NCCL
and nvSHMEM shims, which cannot work through the driver by itself. The CUDA
runtime and its libraries are the program's own:

============= ================== ====================
Platform      driver shim        NVML shim
============= ================== ====================
linux-x86_64  libcuda.so.1       libnvidia-ml.so.1
linux-aarch64 libcuda.so.1       libnvidia-ml.so.1
macosx-*      libcuda.dylib      libnvidia-ml.dylib
win-*         nvcuda.dll         nvml.dll
============= ================== ====================

The NCCL and nvSHMEM shims keep the soname their real counterparts use, and
those majors move independently of the toolkit's, so the manifest — not
this module — is authoritative for the names in a bundle.

``load()`` preloads them into the process with global visibility so that
CUDA consumers resolve the LUPINE shims instead of (or, where a real NVIDIA
stack exists, in front of) the real libraries:

* PyTorch builds with CUDA bring ``libcudart``, ``libcublas`` and friends
  and load the driver lazily by soname, so preloading the LUPINE shims
  routes every driver call to the LUPINE server.
* Natively compiled CUDA code (nvcc/clang) resolves the shims directly.
* On platforms without any NVIDIA runtime (macOS, GPU-less Windows hosts),
  the server-selected shims are the only CUDA stack in the process.

The loader is dependency-free (stdlib only) and never imports torch.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from . import _bundles

# Shims every bundle must carry, in load order: the driver first, because the
# runtime and library shims link against it.
_REQUIRED = {
    "linux": ("libcuda.so.1", "libnvidia-ml.so.1"),
    "darwin": ("libcuda.dylib", "libnvidia-ml.dylib"),
    "win32": ("nvcuda.dll", "nvml.dll"),
}

_DRIVER = {"linux": "libcuda.so.1", "darwin": "libcuda.dylib", "win32": "nvcuda.dll"}

_loaded: dict[str, str] = {}
_names: tuple[str, ...] = ()


def libdir() -> Path | None:
    """Directory holding the selected native client for this platform."""

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

        required = _REQUIRED.get(sys.platform)
        if required is None:
            raise LupineError(f"Unsupported LUPINE client platform: {sys.platform}")
        try:
            directory, etag, platform_key, names = _bundles.resolve(servers, required)
        except Exception as exc:
            raise LupineError(
                f"Could not resolve the server's LUPINE client: {exc}"
            ) from exc
        global _names
        _names = names
        os.environ["LUPINE_LIBDIR"] = str(directory)
        os.environ["LUPINE_CLIENT_ETAG"] = etag
        os.environ["LUPINE_CLIENT_PLATFORM"] = platform_key
        return directory
    return None


def _shim_names(directory: Path) -> tuple[str, ...]:
    """Every shim in the selected directory, driver first.

    A resolved bundle names them in its manifest. ``LUPINE_LIBDIR`` points at
    a plain directory instead, so fall back to its contents.
    """

    driver = _DRIVER[sys.platform]
    if _names:
        names = _names
    else:
        suffixes = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}
        suffix = suffixes[sys.platform]
        names = tuple(
            sorted(
                entry.name
                for entry in directory.iterdir()
                if entry.is_file() and suffix in entry.name
            )
        )
    rest = [name for name in names if name != driver]
    return (driver, *rest) if driver in names else tuple(rest)


def load(*, missing_ok: bool = True) -> dict[str, str]:
    """Preload every LUPINE shim the selected client carries.

    Returns a map of library name to loaded path. Idempotent: later calls
    only load libraries not already loaded into this process. Raises
    ``LupineError`` (from :mod:`lupine`) when the client cannot be selected
    and ``missing_ok`` is false.
    """

    from . import LupineError

    try:
        directory = libdir()
    except LupineError:
        if missing_ok:
            return dict(_loaded)
        raise
    if sys.platform not in _REQUIRED or directory is None or not directory.is_dir():
        if missing_ok:
            return dict(_loaded)
        raise LupineError(
            "No LUPINE native client for this platform "
            f"({sys.platform}); set LUPINE_SERVER or LUPINE_LIBDIR."
        )

    names = _shim_names(directory)
    missing = [name for name in _REQUIRED[sys.platform] if name not in names]
    if missing:
        if missing_ok:
            return dict(_loaded)
        raise LupineError(f"Selected LUPINE client is missing: {sorted(missing)}")

    if sys.platform == "win32":
        # LoadLibrary search order includes directories added here; CUDA
        # consumers load "nvcuda.dll" by name.
        os.add_dll_directory(str(directory))
    elif (directory / _DRIVER[sys.platform]).is_file():
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
            raise LupineError(f"Selected LUPINE library missing: {path}")
        # RTLD_GLOBAL so dlopen("libcuda.so.1") from other libraries
        # (torch's libcudart, libcublas, ...) resolves to the shim.
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        _loaded[name] = str(path)
    return dict(_loaded)


def loaded() -> dict[str, str]:
    """Libraries this process loaded through :func:`load`, by name."""

    return dict(_loaded)
