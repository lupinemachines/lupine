"""Where the ``_C`` extension is and how it gets built.

The extension is the device registration only, a few hundred lines against
libtorch (``csrc``); every operator and the wire are Python. It still has to
be compiled per torch release (major.minor) on each platform because libtorch
has no stable ABI for what it does: torch's stable ABI (``torch/csrc/stable``,
the ``aoti_torch_*`` shim) cannot register an allocator, a device guard,
device hooks or a generator. It is built with the CPython limited API, so the
artifact is one per torch release and platform, not per Python.

Two ways to get it, tried in this order:

1. A prebuilt ``_C.abi3.so`` next to this file (``setup.py build_ext
   --inplace`` from a checkout, or the wheel ``setup.py bdist_wheel`` makes).
2. A JIT build, cached under ``~/.cache/lupine/torch-backend/<torch
   version>-<platform>/``, made on first use with
   ``torch.utils.cpp_extension.load`` in a subprocess (the host process must
   not import torch before the extension is loaded, see ``__init__``). It
   needs a C++ compiler: the Xcode Command Line Tools on macOS
   (``xcode-select --install``), gcc or clang on Linux, plus ``ninja``
   (``pip install ninja``).

``python -m lupine._backend.ext`` runs the JIT build by hand.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
SOURCES = [CSRC / "host.cpp", CSRC / "module.cpp"]
MODULE = "lupine._backend._C"
# csrc/backend.h: a storage's data pointer is its handle << HANDLE_SHIFT.
HANDLE_SHIFT = 20


def torch_version() -> str:
    """The installed torch's version without importing it."""

    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as exc:
        from .. import LupineError

        raise LupineError("the lupine torch backend needs torch installed") from exc


def extra_compile_args() -> list[str]:
    return ["-O2", "-fvisibility=hidden"]


def cache_dir() -> Path:
    base = Path(os.environ.get("LUPINE_CACHE_DIR") or Path.home() / ".cache" / "lupine")
    return base / "torch-backend" / f"{torch_version()}-{sysconfig.get_platform()}"


def prebuilt() -> Path | None:
    for suffix in ("abi3" + sysconfig.get_config_var("SHLIB_SUFFIX"), sysconfig.get_config_var("EXT_SUFFIX")):
        path = HERE / f"_C.{suffix.lstrip('.')}"
        if path.is_file():
            return path
    return None


def sources_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(CSRC.iterdir()):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached() -> Path | None:
    """The JIT build in the cache, unless the sources changed since."""

    path = cache_dir() / ("_C" + sysconfig.get_config_var("SHLIB_SUFFIX"))
    stamp = cache_dir() / "sources.sha256"
    if path.is_file() and stamp.is_file() and stamp.read_text().strip() == sources_digest():
        return path
    return None


def _compiler_requirement() -> str:
    if sys.platform == "darwin":
        return "the Xcode Command Line Tools (`xcode-select --install`)"
    return "gcc or clang (`apt install g++` / `dnf install gcc-c++`)"


def _check_toolchain() -> None:
    from .. import LupineError

    if shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None:
        raise LupineError(
            f"building the lupine torch backend for torch {torch_version()} needs a "
            f"C++ compiler: install {_compiler_requirement()}, or drop a prebuilt "
            f"_C.abi3.so into {HERE}"
        )
    if shutil.which("ninja", path=_build_path()) is None:
        raise LupineError(
            "building the lupine torch backend needs ninja: `pip install ninja`"
        )


def _build_path() -> str:
    scripts = Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin")
    return os.pathsep.join([str(scripts), os.environ.get("PATH", "")])


def build(verbose: bool = False) -> Path:
    """JIT-build the extension into the cache directory and return its path."""

    from .. import LupineError

    _check_toolchain()
    target = cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=_build_path())
    command = [sys.executable, "-m", "lupine._backend.ext", str(target)]
    if verbose:
        command.append("--verbose")
    result = subprocess.run(command, env=env, capture_output=not verbose, text=True)
    path = cached()
    if result.returncode != 0 or path is None:
        output = "" if verbose else (result.stdout + result.stderr)
        raise LupineError(
            f"building the lupine torch backend for torch {torch_version()} failed "
            f"(needs {_compiler_requirement()} and ninja):\n{output}"
        )
    return path


def _build_here(target: Path, verbose: bool) -> None:
    import torch.utils.cpp_extension

    target.mkdir(parents=True, exist_ok=True)
    torch.utils.cpp_extension.load(
        name="_C",
        sources=[str(path) for path in SOURCES],
        extra_cflags=extra_compile_args() + ["-DPy_LIMITED_API=0x030A0000"],
        extra_include_paths=[str(CSRC)],
        build_directory=str(target),
        is_python_module=False,
        verbose=verbose,
    )
    (target / "sources.sha256").write_text(sources_digest() + "\n")


def _preload_torch_libraries() -> None:
    """Load libtorch for the extension without importing torch.

    On a torch with no CUDA build the extension has to supply the CUDA hooks
    before torch's own import first asks for them (they are created once, on
    first use), so the extension is loaded before ``import torch``.
    """

    import ctypes

    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
    suffix = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    for name in ("libc10", "libtorch_cpu", "libtorch", "libtorch_python"):
        path = lib / f"{name}{suffix}"
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def load() -> ModuleType:
    """Import the extension, building it first when nothing prebuilt exists."""

    if MODULE in sys.modules:
        return sys.modules[MODULE]
    path = prebuilt() or cached() or build()
    _preload_torch_libraries()
    loader = importlib.machinery.ExtensionFileLoader(MODULE, str(path))
    spec = importlib.util.spec_from_file_location(MODULE, str(path), loader=loader)
    if spec is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[MODULE] = module
    return module


if __name__ == "__main__":
    targets = [a for a in sys.argv[1:] if not a.startswith("--")]
    _build_here(Path(targets[0]) if targets else cache_dir(), "--verbose" in sys.argv)
