"""CUDA runtime roots and exported ABIs are discovered from input metadata."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("capstone")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cmake"))
import prepare_macos_runtime as preparation
from prepare_macos_runtime import runtime_library, export_interface

pytestmark = pytest.mark.skipif(not shutil.which("aarch64-linux-gnu-gcc"), reason="requires ARM64 cross compiler")


def library(directory, soname):
    source = directory / "api.c"
    source.write_text("int cudaToy(int value){return value;}\n")
    output = directory / "renamed.so"
    subprocess.run(["aarch64-linux-gnu-gcc", "-shared", "-fPIC", "-nostdlib", str(source),
                    "-Wl,-soname," + soname, "-o", str(output)], check=True)
    return output


def test_runtime_root_uses_soname_and_accepts_duplicate_identical_inputs(tmp_path):
    path = library(tmp_path, "libcudart.so.99")
    assert runtime_library([path, path]) == path


def test_missing_runtime_root_is_reported(tmp_path):
    path = library(tmp_path, "other.so")
    with pytest.raises(ValueError, match="expected one CUDA runtime SONAME"):
        runtime_library([path])


@pytest.mark.skipif(not importlib.util.find_spec("clang") or not shutil.which("clang"), reason="requires header inspector")
def test_export_types_are_read_from_api_header(tmp_path):
    path = library(tmp_path, "libcudart.so.99")
    (tmp_path / "cuda_runtime_api.h").write_text("int cudaToy(int);\n")
    interface = export_interface(path, [], [], [], None, tmp_path, [tmp_path])
    report = json.loads(interface.read_text())
    assert report["signatures"]["cudaToy"] == {"arguments": ["i32"], "result": "i32"}
    assert report["signature_report"]["headers"][0]["sha256"]


def test_lock_replays_runtime_version_without_default_override(tmp_path, monkeypatch):
    binary = library(tmp_path, "libcudart.so.99")
    lock = tmp_path / "packages.lock"
    lock.write_text("nvidia-cuda-runtime==99.1 --hash=sha256:example\n")
    output = tmp_path / "prepared"
    output.mkdir()

    def download(directory, requested, destination, requirements, headers):
        assert requested == []
        assert requirements == lock
        return [binary], [{"package": "nvidia-cuda-runtime==99.1",
                           "libraries": [binary.name], "sha256": "example"}]

    monkeypatch.setattr(preparation, "download_libraries", download)
    monkeypatch.setattr(preparation, "prepare_bundle", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["prepare_macos_runtime.py", "--output", str(output),
                                    "-L", str(tmp_path), "--requirements", str(lock)])
    preparation.main()
    report = json.loads((output / "manifest.json").read_text())
    assert report["runtime_source"] == {"package": "nvidia-cuda-runtime", "version": "99.1"}
    assert (output / "packages.lock").read_text() == lock.read_text()
