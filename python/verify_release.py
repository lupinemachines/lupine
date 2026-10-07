#!/usr/bin/env python3
"""Verify that the tandem Python distributions form one release unit."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent


def _project(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)["project"]


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# The client a release loads: the driver and NVML per platform, nothing else.
REQUIRED = {
    "linux": ("libcuda.so.1", "libnvidia-ml.so.1"),
    "darwin": ("libcuda.dylib", "libnvidia-ml.dylib"),
    "win32": ("nvcuda.dll", "nvml.dll"),
}
CONDITIONAL = {"libnccl", "libnvshmem_host"}


def verify_loader(source: bytes) -> None:
    """Check that ``lupine/_native.py`` preloads only the driver and NVML."""

    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "_REQUIRED",
                    "_CONDITIONAL",
                ):
                    found[target.id] = ast.literal_eval(node.value)
    if found.get("_REQUIRED") != REQUIRED:
        raise ValueError(
            f"loader must require the driver and NVML, found {found.get('_REQUIRED')}"
        )
    if set(found.get("_CONDITIONAL", {})) != CONDITIONAL:
        raise ValueError(
            f"loader may condition only NCCL and nvSHMEM, found {found.get('_CONDITIONAL')}"
        )


def verify_metadata(tag: str | None = None) -> str:
    lupine = _project(ROOT / "pyproject.toml")
    auto = _project(ROOT / "auto" / "pyproject.toml")
    version = lupine["version"]

    if auto["version"] != version:
        raise ValueError(
            f"package versions differ: lupine={version}, "
            f"lupine-auto={auto['version']}"
        )

    auto_requirement = f"lupine-auto=={version}"
    if auto_requirement not in lupine["optional-dependencies"]["auto"]:
        raise ValueError(f"lupine[auto] must require {auto_requirement}")

    lupine_requirement = f"lupine=={version}"
    if lupine_requirement not in auto["dependencies"]:
        raise ValueError(f"lupine-auto must require {lupine_requirement}")

    if tag is not None and tag != f"v{version}":
        raise ValueError(f"release tag {tag!r} does not match version {version}")

    verify_loader((ROOT / "lupine" / "_native.py").read_bytes())
    return version


def verify_wheels(directory: Path, version: str) -> None:
    expected = {
        "lupine": f"lupine-auto=={version}",
        "lupine-auto": f"lupine=={version}",
    }
    found: dict[str, tuple[str, list[str], set[str]]] = {}

    for wheel in directory.rglob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            metadata_paths = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"{wheel} has {len(metadata_paths)} METADATA files")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            if "lupine/_native.py" in names:
                verify_loader(archive.read("lupine/_native.py"))

        name = _normalized(metadata["Name"])
        if name in found:
            raise ValueError(f"multiple wheels found for {name}")
        native_files = {path for path in names if path.startswith("lupine/_libs/")}
        found[name] = (
            metadata["Version"],
            metadata.get_all("Requires-Dist", []),
            native_files,
        )

    if set(found) != set(expected):
        raise ValueError(
            f"expected wheels for {sorted(expected)}, found {sorted(found)}"
        )

    for name, requirement in expected.items():
        wheel_version, requirements, native_files = found[name]
        if wheel_version != version:
            raise ValueError(
                f"{name} wheel has version {wheel_version}, expected {version}"
            )
        if not any(req.split(";", 1)[0].strip() == requirement for req in requirements):
            raise ValueError(f"{name} wheel must require {requirement}")
        if native_files:
            raise ValueError(
                f"{name} wheel must be pure Python, found {sorted(native_files)}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="GitHub release tag to validate")
    parser.add_argument("--dist", type=Path, help="directory containing built wheels")
    args = parser.parse_args(argv)

    try:
        version = verify_metadata(args.tag)
        if args.dist is not None:
            verify_wheels(args.dist, version)
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"release metadata is consistent for v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
