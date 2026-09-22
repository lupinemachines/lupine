"""Builds ``lupine._backend._C`` against the installed torch.

The extension carries the Lupine RPC core, so it is built from a source
checkout of the repository (this directory sits at
``python/lupine/_backend`` inside it). Rebuild it whenever the torch minor
version changes: the boxed operator schema is the wire contract between the
host and the worker.

    pip install torch==<version> ninja
    python setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CSRC = HERE / "csrc"

if not (ROOT / "rpc.cpp").is_file():
    sys.exit(
        "lupine._backend must be built from a repository checkout; "
        f"{ROOT / 'rpc.cpp'} is missing"
    )

CORE = [
    "rpc.cpp",
    "h2.cpp",
    "dispatch.cpp",
    "client_bundle.cpp",
    "transport.cpp",
    "monitor_client.cpp",
]
LZ4 = ["lz4.c", "lz4frame.c", "lz4hc.c", "xxhash.c"]

include_dirs = [str(ROOT), str(ROOT / "codegen"), str(ROOT / "third_party/lz4/lib")]
library_dirs: list[str] = []
prefix = os.environ.get("LUPINE_NGHTTP2_PREFIX")
if prefix is None and sys.platform == "darwin":
    for candidate in ("/opt/homebrew/opt/nghttp2", "/usr/local/opt/nghttp2"):
        if Path(candidate, "include/nghttp2/nghttp2.h").is_file():
            prefix = candidate
            break
if prefix:
    include_dirs.append(str(Path(prefix, "include")))
    library_dirs.append(str(Path(prefix, "lib")))

_BuildExtension = BuildExtension.with_options(use_ninja=True)


class build_ext(_BuildExtension):  # noqa: N801
    """``build_ext --inplace`` links the vendored LZ4, so build it first."""

    def run(self) -> None:
        self.run_command("build_clib")
        super().run()


setup(
    name="lupine-backend",
    version="0",
    libraries=[
        (
            "lupine_lz4",
            {
                "sources": [str(ROOT / "third_party/lz4/lib" / f) for f in LZ4],
                "include_dirs": [str(ROOT / "third_party/lz4/lib")],
                "cflags": ["-O2", "-fPIC"],
            },
        )
    ],
    ext_modules=[
        CppExtension(
            name="lupine._backend._C",
            sources=[str(CSRC / f) for f in ("host.cpp", "worker.cpp", "module.cpp")]
            + [str(ROOT / f) for f in CORE],
            include_dirs=include_dirs + [str(CSRC)],
            library_dirs=library_dirs,
            libraries=["nghttp2", "lupine_lz4"],
            extra_compile_args=["-O2", "-fvisibility=hidden", "-Wno-unused-parameter"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    # --inplace then drops _C next to this file, inside the lupine package.
    packages=["lupine._backend"],
    package_dir={"lupine._backend": "."},
)
