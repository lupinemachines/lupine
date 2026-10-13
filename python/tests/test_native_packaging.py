"""Check the built Mac wheel ships application code and excludes wire shims."""

import os
import subprocess
import zipfile
from pathlib import Path


def test_macos_wheel_contains_runtime_but_no_driver(tmp_path):
    source = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    names = ("libcudart.dylib", "liblupine_aten.dylib")
    for name in (*names, "libcuda.dylib", "libnvidia-ml.dylib"):
        (runtime / name).write_bytes(name.encode())
    output = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", str(source), "--out-dir", str(output)],
        env={**os.environ, "LUPINE_MACOS_RUNTIME_DIR": str(runtime)},
        check=True,
    )
    wheel, = output.glob("*.whl")
    assert wheel.name.endswith("-py3-none-macosx_26_0_arm64.whl")
    with zipfile.ZipFile(wheel) as archive:
        dylibs = {name for name in archive.namelist() if name.endswith(".dylib")}
        assert dylibs == {f"lupine/_libs/{name}" for name in names}
        for name in names:
            assert archive.read(f"lupine/_libs/{name}") == (runtime / name).read_bytes()
