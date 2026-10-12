"""Prepare packaged PyTorch CUDA operators against the converted SDK."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

from cuda_wheels import download_libraries
from compile_bundle import prepare_bundle
from machso import ELF
from aot.graph import exported

TORCH_VERSION = "2.14.1+cu130"
TORCH_URL = "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl"
TORCH_SHA256 = "6afc5e5944fb0bd927e08d32a11f71d19154001f1c81fa00a6ed81a217451795"
PACKAGES = (
    "nvidia-cusparselt-cu13==0.8.1",
    "nvidia-nccl-cu13==2.30.7",
    "nvidia-nvshmem-cu13==3.4.5",
    "nvidia-cufile==1.15.1.6",
    "nvidia-cuda-cupti==13.0.85",
)


def torch_inputs(directory):
    wheel = directory / "torch.whl"
    digest = hashlib.sha256()
    with urllib.request.urlopen(TORCH_URL, timeout=60) as response, wheel.open(
        "wb"
    ) as target:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != TORCH_SHA256:
        raise ValueError("PyTorch CUDA wheel SHA256 mismatch")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith(("torch/include/", "torch/lib/")):
                continue
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"invalid PyTorch wheel member: {name}")
            archive.extract(name, directory)
    wheel.unlink()
    return directory / "torch"


def sdk_providers(report, signatures, directory):
    adapted = set(report["export_abi_bridges"])
    providers = []
    for node in report["nodes"]:
        if node["intrinsic"] or not node["name"].startswith(("libcu", "libnv")):
            continue
        elf = ELF.from_file(node["input"], allow_tls=True)
        symbols = [s for s in elf.symbols if exported(s) and not s["version_hidden"]]
        names = sorted({s["name"] for s in symbols})
        interface = {
            "schema": 1,
            "architecture": "arm64",
            "soname": node["name"],
            "native_install_name": "@rpath/libcudart.dylib",
            "exports": names,
            "symbol_versions": {
                s["name"]: s["version"] for s in symbols if s["version"]
            },
            # Adapted SDK entries are Darwin calls; unadapted entries retain
            # the original Linux ABI and can be called directly by Linux ATen.
            "signatures": {name: signatures[name] for name in names if name in adapted},
        }
        output = directory / (node["name"] + ".json")
        output.write_text(json.dumps(interface, indent=2) + "\n")
        providers.append(output)
    return providers


def prepare_operators(
    output,
    sdk,
    export_interface,
    inputs,
    includes,
    paths,
    providers,
    dependency_provider,
):
    directory = inputs / "operators"
    directory.mkdir()
    torch = torch_inputs(directory)
    libraries = directory / "libraries"
    libraries.mkdir()
    _, packages = download_libraries(directory / "wheels", list(PACKAGES), libraries)
    source = directory / "liblupine_aten.so"
    command = [
        "aarch64-linux-gnu-g++",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++20",
        "-I",
        str(torch / "include"),
        "-I",
        str(torch / "include/torch/csrc/api/include"),
    ]
    for include in includes:
        command += ["-I", str(include)]
    command += [
        str(Path(__file__).with_name("aten_bridge.cpp")),
        "-L",
        str(torch / "lib"),
        "-Wl,--no-as-needed",
        "-ltorch_cuda",
        "-ltorch_cpu",
        "-lc10",
        "-Wl,-soname,liblupine_aten.so",
        "-o",
        str(source),
    ]
    subprocess.run(command, check=True)
    interfaces = directory / "providers"
    interfaces.mkdir()
    signatures = json.loads(export_interface.read_text())["signatures"]
    sdk_interfaces = sdk_providers(sdk, signatures, interfaces)
    interface = directory / "exports.json"
    interface.write_text(
        json.dumps(
            {
                "schema": 1,
                "architecture": "arm64",
                "exports": [
                    "lupine_aten_run",
                    "lupine_aten_error",
                    "lupine_aten_close",
                    "lupine_aten_alloc",
                    "lupine_aten_free",
                ],
                "signatures": {
                    "lupine_aten_run": {
                        "result": "i32",
                        "arguments": ["i32", "ptr", "ptr", "ptr", "ptr"],
                    },
                    "lupine_aten_error": {"result": "ptr", "arguments": []},
                    "lupine_aten_close": {"result": "i32", "arguments": []},
                    "lupine_aten_alloc": {"result": "i32", "arguments": ["i64", "ptr"]},
                    "lupine_aten_free": {"result": "i32", "arguments": ["ptr"]},
                },
            },
            indent=2,
        )
        + "\n"
    )
    report = prepare_bundle(
        source,
        output,
        [torch / "lib", libraries, *paths],
        experimental=True,
        native_providers=[*providers, *sdk_interfaces],
        dependency_provider=dependency_provider,
        export_interface=interface,
    )
    report["torch_input"] = {
        "version": TORCH_VERSION,
        "url": TORCH_URL,
        "sha256": TORCH_SHA256,
    }
    report["input_packages"] = packages
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "packages.lock").write_text(
        "".join(
            f"{package['package']} --hash=sha256:{package['sha256']}\n"
            for package in packages
        )
    )
    shutil.copyfile(interface, output / "export-interface.json")
    return report
