#!/usr/bin/env python3
"""Build the pinned multi-GPU samples inside the native CUDA build container."""

from pathlib import Path
import platform
import shutil
import subprocess

from validate_specs import SAMPLE_SPEC, SAMPLES


def main():
    if platform.machine() not in SAMPLE_SPEC["architectures"]:
        print("No upstream multi-GPU samples supported on this architecture")
        return
    source = Path("/cuda-samples")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1", "--branch=v12.4.1",
                    "https://github.com/NVIDIA/cuda-samples.git", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "sparse-checkout", "set", "Common",
                    *(f"Samples/{sample['source']}" for sample in SAMPLES.values())], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", SAMPLE_SPEC["revision"]], check=True)
    output = Path("/out/samples")
    output.mkdir()
    for name, spec in SAMPLES.items():
        directory = source / "Samples" / spec["source"]
        subprocess.run(["make", "-j4", "SMS=89"], cwd=directory, check=True)
        shutil.copy2(directory / name, output / name)
    # Keep the CUDA libraries beside the samples so native controls use the
    # same versions. Preserve symlinks rather than copying large files twice.
    for pattern in ("libcufft.so*", "libcurand.so*"):
        for library in Path("/usr/local/cuda/lib64").glob(pattern):
            shutil.copy2(library, output / library.name, follow_symlinks=False)
    (output / "revision.txt").write_text(SAMPLE_SPEC["revision"] + "\n")


if __name__ == "__main__":
    main()
