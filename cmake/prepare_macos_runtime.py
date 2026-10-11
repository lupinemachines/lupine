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
from aot.graph import Graph
from machso import ELF
from cuda_wheels import download_libraries

CUDA_RUNTIME_VERSION = "13.3.29"
API_HEADERS = (
    "cuda_runtime_api.h", "cublas_v2.h", "cublasLt.h", "cudnn.h", "nvrtc.h",
    "nvJitLink.h", "cufft.h", "curand.h", "cusolverDn.h", "cusparse.h",
)


def runtime_library(libraries):
    candidates = []
    for path in dict.fromkeys(libraries):
        soname = ELF.from_file(path, allow_tls=True, machines=(183,)).soname
        if soname and soname.startswith("libcudart.so."):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"expected one CUDA runtime SONAME, found {len(candidates)}")
    return candidates[0]


def export_interface(source, paths, roots, providers, dependency_provider, directory, includes):
    from aot.headers import read_signatures
    from aot.native_abi import validate_signatures

    graph = Graph(source, paths, roots, providers, dependency_provider)
    definitions = graph.public_definitions()
    functions = [name for name, (node, index) in definitions.items()
                 if node.elf.symbols[index]['info'] & 15 == 2]
    headers = [next((p / name for p in includes if (p / name).is_file()), None)
               for name in API_HEADERS]
    headers = [p for p in headers if p is not None]
    if not headers:
        raise ValueError("native export ABI inspection requires CUDA API headers")
    report = read_signatures(headers, functions, includes)
    signatures = report.pop("signatures")
    for name in ("cudaGetDriverEntryPoint", "cudaGetDriverEntryPointByVersion"):
        if name in signatures:
            signatures[name]["function_pointer_output"] = {
                "argument": 1, "unavailable_result": 801,
            }
    interface = {"schema": 1, "architecture": "arm64", "exports": sorted(definitions),
                 "signatures": signatures, "signature_report": report}
    validate_signatures(interface)
    path = directory / "export-interface.json"
    path.write_text(json.dumps(interface, indent=2) + "\n")
    return path


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
    parser.add_argument("--native-provider", type=Path, action="append", default=[])
    parser.add_argument("--torch-operators", action="store_true", help="prepare packaged PyTorch CUDA operators against the SDK")
    parser.add_argument("--native-exports", action="store_true", help="derive native export ABIs from SDK headers")
    parser.add_argument("--api-include", type=Path, action="append", default=[])
    parser.add_argument("--ubuntu-cache", type=Path)
    parser.add_argument(
        "--ubuntu-keyring", type=Path,
        default=Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg"),
    )
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
    provider = None
    if args.ubuntu_cache:
        from aot.ubuntu import UbuntuDependencies
        provider = UbuntuDependencies(args.ubuntu_cache, args.ubuntu_keyring)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        inputs = directory / "libraries"
        inputs.mkdir()
        headers = directory / "headers"
        requested = list(args.package)
        if not args.runtime and not args.requirements:
            requested.append("nvidia-cuda-runtime==" + args.version)
        libraries, packages = (
            download_libraries(
                directory / "wheels", requested, inputs, args.requirements,
                headers if args.native_exports else None,
            )
            if requested or args.requirements
            else ([], [])
        )
        source = args.runtime or runtime_library(libraries)
        extra = sorted({p for p in libraries if p.resolve() != source.resolve()})
        paths = [inputs, *args.library_path]
        interface = (export_interface(source, paths, extra, args.native_provider, provider,
                                      directory, [headers, *args.api_include])
                     if args.native_exports else None)
        report = prepare_bundle(
            source,
            args.output,
            paths,
            experimental=True,
            additional_roots=extra,
            native_providers=args.native_provider,
            dependency_provider=provider,
            export_interface=interface,
        )
        if args.torch_operators:
            if interface is None:
                raise ValueError("PyTorch operators require --native-exports")
            from prepare_macos_operators import prepare_operators
            operators = prepare_operators(
                args.output / "torch-operators", report, interface, directory,
                [headers, *args.api_include], paths, args.native_provider, provider,
            )
            report["torch_operators"] = operators["torch_input"]
        if args.runtime:
            report["runtime_source"] = {"path": str(args.runtime)}
        else:
            package = next(p["package"] for p in packages if source.name in p["libraries"])
            name, version = package.split("==", 1)
            report["runtime_source"] = {"package": name, "version": version}
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
