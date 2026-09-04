#!/usr/bin/env python3
"""Reject Linux clients that exceed the manylinux2014 ABI contract."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


MAX_GLIBC = (2, 17)
PRIVATE_NEEDED = {
    "libcrypto.so",
    "libgcc_s.so",
    "libnghttp2.so",
    "libssl.so",
    "libstdc++.so",
}
PUBLIC_SYMBOLS = {
    "libcuda.so.1": re.compile(r"(?:cu[A-Z].*|dlsym|lupine_checkpoint_.*)"),
    "libcudart.so.13": re.compile(r"(?:__cuda.*|cuda.*|libcudart\.so\.13)"),
    "libnvidia-ml.so.1": re.compile(r"nvml.*"),
}


def readelf(*args: str) -> str:
    return subprocess.run(
        ["readelf", *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def dynamic_symbols(path: Path) -> set[str]:
    output = subprocess.run(
        ["nm", "--dynamic", "--defined-only", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return {line.split()[-1].split("@", 1)[0] for line in output.splitlines()}


def verify(path: Path) -> list[str]:
    errors: list[str] = []
    dynamic = readelf("--dynamic", str(path))
    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)]", dynamic)
    private = sorted(
        library
        for library in needed
        if any(
            library == name or library.startswith(name + ".")
            for name in PRIVATE_NEEDED
        )
    )
    if private:
        errors.append(f"private dependencies remain dynamic: {', '.join(private)}")

    versions = readelf("--version-info", str(path))
    for namespace in ("GLIBCXX", "CXXABI", "GCC"):
        matches = sorted(set(re.findall(rf"\b{namespace}_[0-9.]+", versions)))
        if matches:
            errors.append(
                f"dynamic {namespace} requirements remain: {', '.join(matches)}"
            )

    glibc_versions = {
        tuple(int(part) for part in value.split("."))
        for value in re.findall(r"\bGLIBC_([0-9]+(?:\.[0-9]+)+)", versions)
    }
    newer = sorted(version for version in glibc_versions if version > MAX_GLIBC)
    if newer:
        rendered = ", ".join(
            "GLIBC_" + ".".join(map(str, version)) for version in newer
        )
        errors.append(f"requirements exceed GLIBC_2.17: {rendered}")

    allowed = PUBLIC_SYMBOLS.get(path.name)
    if allowed is not None:
        unexpected = sorted(
            symbol
            for symbol in dynamic_symbols(path)
            if allowed.fullmatch(symbol) is None
        )
        if unexpected:
            errors.append(
                "private symbols are exported: " + ", ".join(unexpected[:20])
            )
    return errors


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {Path(sys.argv[0]).name} LIBRARY...", file=sys.stderr)
        return 2

    failed = False
    for value in sys.argv[1:]:
        path = Path(value)
        errors = verify(path)
        if errors:
            failed = True
            for error in errors:
                print(f"{path}: {error}", file=sys.stderr)
        else:
            print(f"{path}: manylinux2014 compatible")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
