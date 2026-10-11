"""Resolve CUDA wheels and collect their ELF libraries without importing SDK code."""

from email.parser import BytesParser
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

PLATFORMS = (
    "manylinux2014_aarch64",
    "manylinux_2_17_aarch64",
    "manylinux_2_18_aarch64",
    "manylinux_2_25_aarch64",
    "manylinux_2_27_aarch64",
    "manylinux_2_28_aarch64",
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_libraries(directory, requested, destination, requirements=None, headers=None):
    if any("==" not in package or "*" in package for package in requested):
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
        "--implementation",
        "py",
        "--python-version",
        "3",
        "--abi",
        "none",
    ]
    for platform in PLATFORMS:
        command += ["--platform", platform]
    if requirements is not None:
        command += ["--require-hashes", "--requirement", str(requirements)]
    # Resolve all requested versions together, including Requires-Dist edges.
    # A late-loaded helper may be declared only by its wheel's metadata.
    subprocess.run([*command, *requested], check=True)
    libraries, provenance = [], []
    for wheel in sorted(directory.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            metadata = [
                n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
            ]
            if len(metadata) != 1:
                raise ValueError(
                    f"{wheel.name} must have exactly one package metadata file"
                )
            package = BytesParser().parsebytes(archive.read(metadata[0]))
            if not package["Name"] or not package["Version"]:
                raise ValueError(f"{wheel.name} has incomplete package metadata")
            members, header_members = [], []
            for name in archive.namelist():
                if headers is not None and "/include/" in name and not name.endswith("/"):
                    relative = Path(name.split("/include/", 1)[1])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"invalid header input path: {name}")
                    output = headers / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.read(name)
                    digest = hashlib.sha256(data).hexdigest()
                    if output.exists() and file_hash(output) != digest:
                        raise ValueError(f"conflicting header inputs: {relative}")
                    output.write_bytes(data)
                    header_members.append({"path": str(relative), "sha256": digest})
                if "/lib/" not in name or name.endswith("/"):
                    continue
                with archive.open(name) as member:
                    if member.read(4) != b"\x7fELF":
                        continue
                output = destination / Path(name).name
                partial = output.with_name(output.name + ".partial")
                with archive.open(name) as member, partial.open("wb") as target:
                    shutil.copyfileobj(member, target, 1024 * 1024)
                if output.exists():
                    if file_hash(output) != file_hash(partial):
                        raise ValueError(f"conflicting library inputs: {output.name}")
                    partial.unlink()
                else:
                    partial.replace(output)
                libraries.append(output)
                members.append(output.name)
            provenance.append(
                {
                    "package": f"{package['Name']}=={package['Version']}",
                    "wheel": wheel.name,
                    "sha256": file_hash(wheel),
                    "requires_dist": package.get_all("Requires-Dist", []),
                    "libraries": members,
                    "headers": header_members,
                }
            )
    if not libraries:
        raise ValueError("selected packages contain no ELF libraries")
    return libraries, provenance
