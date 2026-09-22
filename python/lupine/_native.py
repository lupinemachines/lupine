"""Native LUPINE client library loader.

LUPINE servers publish the compatible client for every supported platform.
The loader downloads the exact bundle selected by ``LUPINE_SERVER`` before
CUDA consumers are imported. The wheel itself is pure Python: every native
object comes from a bound server or an explicit ``LUPINE_LIBDIR``.

``load()`` preloads the driver and NVML shims:

============= ================== ====================
Platform      driver shim        NVML shim
============= ================== ====================
linux-x86_64  libcuda.so.1       libnvidia-ml.so.1
linux-aarch64 libcuda.so.1       libnvidia-ml.so.1
macosx-*      libcuda.dylib      libnvidia-ml.dylib
win-*         nvcuda.dll         nvml.dll
============= ================== ====================

NCCL and nvSHMEM cannot work through the driver alone, so their shims are
loaded when the bundle names them and the platform has no native copy (an
``nvidia-nccl-*`` wheel or a system library). A bundle names its shims in its
manifest; a ``LUPINE_LIBDIR`` directory is filtered by the same names.

The shims are preloaded with global visibility so that CUDA consumers
resolve them instead of (or, where a real NVIDIA stack exists, in front of)
the real driver; the program's own CUDA runtime and libraries run against
it:

* PyTorch's CUDA wheels load the driver lazily by soname, so preloading the
  shim routes every driver call to the LUPINE server.
* Natively compiled CUDA code (nvcc/clang) resolves the driver shim directly.

The loader is dependency-free (stdlib only) and never imports torch.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import os
import sys
from pathlib import Path

from . import _bundles

# Shims every bundle must carry, in load order: the driver first, because
# every other shim links against it.
_REQUIRED = {
    "linux": ("libcuda.so.1", "libnvidia-ml.so.1"),
    "darwin": ("libcuda.dylib", "libnvidia-ml.dylib"),
    "win32": ("nvcuda.dll", "nvml.dll"),
}

_DRIVER = {"linux": "libcuda.so.1", "darwin": "libcuda.dylib", "win32": "nvcuda.dll"}

# Shims loaded only when the bundle carries them and no native library
# answers to the same soname: soname stem -> (``nvidia.*`` wheel package,
# ``ctypes.util.find_library`` name).
_CONDITIONAL = {
    "libnccl": ("nvidia.nccl", "nccl"),
    "libnvshmem_host": ("nvidia.nvshmem", "nvshmem_host"),
}

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


def _soname_stem(name: str) -> str:
    return name.split(".", 1)[0]


def _native_available(package: str, library: str) -> bool:
    """Whether the platform already supplies ``library`` outside LUPINE."""

    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if (Path(location) / "lib").is_dir():
                return True
    return ctypes.util.find_library(library) is not None


def _shim_names(directory: Path) -> tuple[str, ...]:
    """The shims to load from the selected directory, driver first.

    A resolved bundle names its files in its manifest; ``LUPINE_LIBDIR``
    points at a plain directory instead, so its contents stand in. Either
    way only the driver, NVML, and a conditional shim with no native
    counterpart are selected.
    """

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
    selected = [name for name in _REQUIRED[sys.platform] if name in names]
    for name in names:
        native = _CONDITIONAL.get(_soname_stem(name))
        if native is not None and not _native_available(*native):
            selected.append(name)
    return tuple(selected)


def load(*, missing_ok: bool = True) -> dict[str, str]:
    """Preload the LUPINE client.

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
        # RTLD_GLOBAL so dlopen("libcuda.so.1") from the program's CUDA
        # runtime resolves to the shim.
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        _loaded[name] = str(path)
    return dict(_loaded)


def loaded() -> dict[str, str]:
    """Libraries this process loaded through :func:`load`, by name."""

    return dict(_loaded)
