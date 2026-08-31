#!/usr/bin/env python3
"""Assemble the platform native libraries into the wheel source tree.

Usage:
    python build.py stage <platform-tag> <dir-with-libs> [<dir>...]
    python build.py build [--uv]

``stage`` copies the shims for one platform into ``lupine/_libs/<tag>/``;
``build`` verifies every expected platform is present and builds the wheel
(with ``uv build`` by default, falling back to ``python -m build``).
Expected tags: linux-x86_64, linux-aarch64, macosx-universal2,
win-amd64, win-arm64.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _libs_dir() -> Path:
    return ROOT / "lupine" / "_libs"

EXPECTED = {
    "linux-x86_64": ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1"),
    "linux-aarch64": ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1"),
    "macosx-universal2": ("libcuda.dylib", "libcudart.dylib", "libnvidia-ml.dylib"),
    "win-amd64": ("nvcuda.dll", "cudart64_13.dll", "nvml.dll"),
    "win-arm64": ("nvcuda.dll", "cudart64_13.dll", "nvml.dll"),
}


def stage(tag: str, sources: list[Path]) -> int:
    if tag not in EXPECTED:
        print(f"unknown platform tag {tag!r}; expected one of {sorted(EXPECTED)}")
        return 2
    dest = _libs_dir() / tag
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in sources:
        for name in EXPECTED[tag]:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, dest / name)
                staged.append(name)
    missing = [name for name in EXPECTED[tag] if name not in staged]
    if missing:
        print(f"warning: {tag}: missing {missing} (searched: {[str(s) for s in sources]})")
    else:
        print(f"staged {tag}: {sorted(staged)}")
    return 0


def build(use_uv: bool) -> int:
    libs = _libs_dir()
    libs.mkdir(parents=True, exist_ok=True)
    present = sorted(p.name for p in libs.iterdir() if p.is_dir())
    missing = [tag for tag in EXPECTED if tag not in present]
    if missing:
        print(f"warning: no libraries staged for {missing}; wheel will not bundle them")
    for tag in present:
        if tag not in EXPECTED:
            print(f"warning: unexpected _libs entry {tag}")
            continue
        for name in EXPECTED[tag]:
            if not (libs / tag / name).is_file():
                print(f"warning: {tag}/{name} missing")
    if use_uv:
        return subprocess.call(["uv", "build"], cwd=ROOT)
    return subprocess.call([sys.executable, "-m", "build"], cwd=ROOT)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "stage":
        if len(argv) < 4:
            print("stage needs: <tag> <dir> [<dir>...]")
            return 2
        return stage(argv[2], [Path(p) for p in argv[3:]])
    if argv[1] == "build":
        return build(use_uv=argv[2:3] != ["--no-uv"])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
