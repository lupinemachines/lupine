"""Verify trusted inputs and ABI declarations for packaged CUDA operators."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytest.importorskip("capstone")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cmake"), str(ROOT / "third_party/machso")]
import prepare_macos_operators as operators


def test_torch_wheel_hash_mismatch_prevents_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        operators.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"untrusted")
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        operators.torch_inputs(tmp_path)
    assert not (tmp_path / "torch").exists()


def test_only_operator_libraries_and_headers_are_extracted(tmp_path, monkeypatch):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("torch/lib/libtorch_cuda.so", b"binary")
        archive.writestr("torch/include/ATen/ATen.h", b"header")
        archive.writestr("torch/__init__.py", b"raise RuntimeError('must not import')")
    raw = data.getvalue()
    monkeypatch.setattr(operators, "TORCH_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        operators.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(raw)
    )
    torch = operators.torch_inputs(tmp_path)
    assert (torch / "lib/libtorch_cuda.so").read_bytes() == b"binary"
    assert (torch / "include/ATen/ATen.h").read_bytes() == b"header"
    assert not (torch / "__init__.py").exists()


@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="requires ARM64 cross compiler"
)
def test_sdk_provider_keeps_versions_and_only_bridges_adapted_exports(tmp_path):
    source = tmp_path / "sdk.c"
    source.write_text(
        "int native_entry(int x){return x;} int linux_entry(int x){return x;}\n"
    )
    version = tmp_path / "versions.map"
    version.write_text("SDK_1 { global: native_entry; linux_entry; };\n")
    library = tmp_path / "sdk.so"
    subprocess.run(
        [
            "aarch64-linux-gnu-gcc",
            "-shared",
            "-fPIC",
            "-nostdlib",
            str(source),
            "-Wl,-soname,libcudart.so.99",
            "-Wl,--version-script=" + str(version),
            "-o",
            str(library),
        ],
        check=True,
    )
    report = {
        "export_abi_bridges": ["native_entry"],
        "nodes": [
            {"name": "libcudart.so.99", "input": str(library), "intrinsic": False}
        ],
    }
    signature = {"result": "i32", "arguments": ["i32"]}
    paths = operators.sdk_providers(report, {"native_entry": signature}, tmp_path)
    declaration = json.loads(paths[0].read_text())
    assert set(declaration["exports"]) >= {"native_entry", "linux_entry"}
    assert declaration["symbol_versions"]["linux_entry"] == "SDK_1"
    assert declaration["signatures"] == {"native_entry": signature}
    assert declaration["native_install_name"] == "@rpath/libcudart.dylib"
