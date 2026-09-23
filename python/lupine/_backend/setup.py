"""Prebuilds ``lupine._backend._C`` against the installed torch.

The extension is the device registration only (see ``csrc``), built with the
CPython limited API, so one artifact serves one torch release (major.minor)
on one platform for every Python. It links libtorch, never libtorch_python.

    pip install torch==<version> setuptools ninja
    python python/lupine/_backend/setup.py build_ext --inplace

drops ``_C.abi3.so`` next to this file, where ``lupine._backend`` finds it
ahead of its own JIT build (``lupine._backend.ext``);
``python setup.py bdist_wheel`` gives the same artifact as an abi3 wheel
named after the torch release it was built for.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

from ext import SOURCES, extra_compile_args, torch_version

HERE = Path(__file__).resolve().parent

setup(
    name="lupine-torch-backend",
    version=torch_version().split("+", 1)[0],
    ext_modules=[
        CppExtension(
            name="lupine._backend._C",
            sources=[str(path) for path in SOURCES],
            include_dirs=[str(HERE / "csrc")],
            extra_compile_args=extra_compile_args(),
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
    packages=["lupine._backend"],
    package_dir={"lupine._backend": "."},
)
