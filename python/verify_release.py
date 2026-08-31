#!/usr/bin/env python3
"""Verify that the tandem Python distributions form one release unit."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _project(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)["project"]


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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

    return version


def verify_wheels(directory: Path, version: str) -> None:
    expected = {
        "lupine": f"lupine-auto=={version}",
        "lupine-auto": f"lupine=={version}",
    }
    found: dict[str, tuple[str, list[str]]] = {}

    for wheel in directory.rglob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            metadata_paths = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"{wheel} has {len(metadata_paths)} METADATA files")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))

        name = _normalized(metadata["Name"])
        if name in found:
            raise ValueError(f"multiple wheels found for {name}")
        found[name] = (metadata["Version"], metadata.get_all("Requires-Dist", []))

    if set(found) != set(expected):
        raise ValueError(
            f"expected wheels for {sorted(expected)}, found {sorted(found)}"
        )

    for name, requirement in expected.items():
        wheel_version, requirements = found[name]
        if wheel_version != version:
            raise ValueError(
                f"{name} wheel has version {wheel_version}, expected {version}"
            )
        if not any(req.split(";", 1)[0].strip() == requirement for req in requirements):
            raise ValueError(f"{name} wheel must require {requirement}")


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
