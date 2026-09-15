#!/usr/bin/env python3
"""Stage the CUDA headers the client shims build against.

Each shim is header-gated, so every library it can forward needs that
library's header present at configure time -- a missing one drops its shim from
the build silently, so a new shim belongs in PACKAGES below. NVIDIA publishes no header-only
package on PyPI, so the headers come out of the runtime wheels — which carry
the libraries too, several hundred megabytes each. Take one wheel at a time
and keep only its include tree, so a builder never holds more than one.

The wheels are pulled for a fixed set of platform tags rather than the host's:
headers are architecture-independent, and NVIDIA builds these against several
glibc baselines (cuda-runtime 2.17, NCCL 2.18, CUPTI 2.25, cuBLAS and cuDNN 2.27), none of
which a manylinux2014 builder could install.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PACKAGES = (
    "nvidia-cuda-runtime",
    # CUDA 13 moved the compiler's own headers out of the runtime package, and
    # cuda_runtime.h still includes them.
    "nvidia-cuda-crt",
    "nvidia-cuda-cccl",
    "nvidia-nvml-dev",
    "nvidia-cublas",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-cusparse",
    "nvidia-cusolver",
    "nvidia-cuda-nvrtc",
    "nvidia-nvjitlink",
    "nvidia-nvjpeg",
    "nvidia-npp",
    "nvidia-cufile",
    "nvidia-cuda-cupti",
    "nvidia-cusparselt-cu13",
    "nvidia-cudnn-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvshmem-cu13",
)

PLATFORMS = (
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_18_x86_64",
    "manylinux_2_25_x86_64",
    "manylinux_2_27_x86_64",
    "manylinux_2_28_x86_64",
)


def stage(package: str, destination: Path) -> int:
    with tempfile.TemporaryDirectory() as scratch:
        command = [
            sys.executable, "-m", "pip", "download",
            "--dest", scratch, "--only-binary=:all:", "--no-deps",
            "--implementation", "py", "--python-version", "3", "--abi", "none",
        ]
        for platform in PLATFORMS:
            command += ["--platform", platform]
        subprocess.run([*command, package], check=True, stdout=subprocess.DEVNULL)

        staged = 0
        for wheel in Path(scratch).glob("*.whl"):
            with zipfile.ZipFile(wheel) as archive:
                headers = [
                    name
                    for name in archive.namelist()
                    if "/include/" in name and not name.endswith("/")
                ]
                archive.extractall(destination, headers)
                staged += len(headers)
        return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    if arguments.destination.exists():
        shutil.rmtree(arguments.destination)
    arguments.destination.mkdir(parents=True)

    for package in PACKAGES:
        count = stage(package, arguments.destination)
        print(f"{package}: {count} headers", flush=True)
        if not count:
            print(f"no headers in {package}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
