#!/usr/bin/env python3
"""Prepare an ARM64 CUDA library closure using vendored machso."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/machso"))
from compile_bundle import prepare_bundle

CUDA_RUNTIME_VERSION = "13.3.29"
PLATFORMS = (
    "manylinux2014_aarch64",
    "manylinux_2_17_aarch64",
    "manylinux_2_27_aarch64",
    "manylinux_2_28_aarch64",
)


def download_libraries(directory, package, destination):
    if "==" not in package:
        raise ValueError("CUDA input packages must use an exact == version pin")
    directory.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(directory),
        "--only-binary=:all:",
        "--no-deps",
        "--implementation",
        "py",
        "--python-version",
        "3",
        "--abi",
        "none",
    ]
    for platform in PLATFORMS:
        command += ["--platform", platform]
    subprocess.run([*command, package], check=True)
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one wheel per package")
    libraries = []
    with zipfile.ZipFile(wheels[0]) as archive:
        for name in archive.namelist():
            if "/lib/" not in name or name.endswith("/"):
                continue
            with archive.open(name) as member:
                if member.read(4) != b"\x7fELF":
                    continue
            # Dynamic helpers belong in the same closure as their consumer,
            # even when they have no DT_NEEDED edge (e.g. compiler builtins).
            data = archive.read(name)
            output = destination / Path(name).name
            if output.exists() and output.read_bytes() != data:
                raise ValueError(f"conflicting library inputs: {output.name}")
            output.write_bytes(data)
            libraries.append(output)
    if not libraries:
        raise ValueError(f"{package} contains no ELF libraries")
    return libraries, {
        "package": package,
        "wheel": wheels[0].name,
        "sha256": hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "-L", "--library-path", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--runtime", type=Path, help="use an existing ARM64 Linux runtime binary"
    )
    parser.add_argument("--version", default=CUDA_RUNTIME_VERSION)
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="also convert every ELF library in this pinned NVIDIA wheel; repeatable",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        inputs = directory / "libraries"
        inputs.mkdir()
        libraries, packages = [], []
        if args.runtime:
            source = args.runtime
        else:
            found, provenance = download_libraries(
                directory / "runtime", "nvidia-cuda-runtime==" + args.version, inputs
            )
            source = next((p for p in found if p.name == "libcudart.so.13"), None)
            if source is None:
                raise ValueError("runtime wheel must contain libcudart.so.13")
            libraries.extend(found)
            packages.append(provenance)
        for index, package in enumerate(args.package):
            found, provenance = download_libraries(
                directory / f"package-{index}", package, inputs
            )
            libraries.extend(found)
            packages.append(provenance)
        extra = sorted({p for p in libraries if p.resolve() != source.resolve()})
        report = prepare_bundle(
            source,
            args.output,
            [inputs, *args.library_path],
            experimental=True,
            additional_roots=extra,
        )
        report["runtime_source"] = (
            {"path": str(args.runtime)}
            if args.runtime
            else {"package": "nvidia-cuda-runtime", "version": args.version}
        )
        report["input_packages"] = packages
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
