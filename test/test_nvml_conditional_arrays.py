#!/usr/bin/env python3
"""Compile and exercise the checked-in generated NVML RPCs without a GPU.

Usage: python3 test/test_nvml_conditional_arrays.py /path/to/cuda/include
"""
import pathlib
import os
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = [f"nvmlDeviceGet{kind}RunningProcesses{version}"
         for kind in ("Compute", "Graphics", "MPSCompute")
         for version in ("", "_v2")]
NAMES.append("nvmlDeviceGetProcessUtilization")


def extract_function(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def main():
    with tempfile.TemporaryDirectory(prefix="lupine-nvml-arrays-") as directory:
        output = pathlib.Path(directory)
        for side in ("client", "server"):
            source = (ROOT / f"codegen/gen_nvml_{side}.inc").read_text()
            snippets = []
            for name in NAMES:
                if side == "client":
                    snippets.append(extract_function(source, f"static nvmlReturn_t lupine_rpc_{name}("))
                    snippets.append(extract_function(source, f'extern "C" nvmlReturn_t {name}('))
                else:
                    snippets.append(extract_function(source, f"int handle_{name}("))
            (output / f"{side}.inc").write_text("\n".join(snippets))
        executable = output / "test"
        subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror", "-g",
            "-I", sys.argv[1], "-I", str(ROOT), "-I", str(output),
            str(ROOT / "test/test_nvml_conditional_arrays.cpp"), "-o", str(executable),
        ], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
