#!/usr/bin/env python3
"""Prepare an ARM64 CUDA library closure using vendored machso."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/machso"))
from compile_bundle import prepare_bundle
from cuda_wheels import download_libraries

CUDA_RUNTIME_VERSION = "13.3.29"


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
        help="convert this pinned NVIDIA wheel and its dependencies; repeatable",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        help="replay a generated packages.lock with hash verification",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        inputs = directory / "libraries"
        inputs.mkdir()
        requested = list(args.package)
        if not args.runtime:
            requested.append("nvidia-cuda-runtime==" + args.version)
        libraries, packages = (
            download_libraries(
                directory / "wheels", requested, inputs, args.requirements
            )
            if requested or args.requirements
            else ([], [])
        )
        source = args.runtime or next(
            (p for p in libraries if p.name == "libcudart.so.13"), None
        )
        if source is None:
            raise ValueError("runtime wheel must contain libcudart.so.13")
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
        (args.output / "packages.lock").write_text(
            "".join(
                f"{package['package']} --hash=sha256:{package['sha256']}\n"
                for package in packages
            )
        )
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
