#!/usr/bin/env python3
"""Assemble Python runtime stubs and server-served client bundles.

Usage:
    python build.py stage-runtime <platform-tag> <dir-with-runtime> [<dir>...]
    python build.py bundles <complete-client-dir> <output-dir> <source-revision>
    python build.py build [--uv]

``stage-runtime`` copies only the CUDA runtime translation stub used by the
Python wheel. ``bundles`` consumes complete clients from a separate artifact
tree; driver and NVML libraries never enter the wheel staging tree.
``build`` verifies every expected platform is present and builds that wheel
directly from the staged checkout (with ``uv build --wheel`` by default,
falling back to ``python -m build --wheel``).
Expected tags: linux-x86_64, linux-aarch64, macosx-universal2,
win-amd64, win-arm64.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _libs_dir() -> Path:
    return ROOT / "lupine" / "_libs"


EXPECTED = {
    "linux-x86_64": ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1"),
    "linux-aarch64": ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1"),
    "macosx-universal2": ("libcuda.dylib", "libcudart.dylib", "libnvidia-ml.dylib"),
    "win-amd64": ("nvcuda.dll", "cudart64_13.dll", "nvml.dll"),
    "win-arm64": ("nvcuda.dll", "cudart64_13.dll", "nvml.dll"),
}

CLIENT_BUNDLES = {
    ("linux/amd64",): "linux-x86_64",
    ("linux/arm64",): "linux-aarch64",
    ("macos/amd64", "macos/arm64"): "macosx-universal2",
    ("windows/amd64",): "win-amd64",
    ("windows/arm64",): "win-arm64",
}

RUNTIMES = {tag: names[1] for tag, names in EXPECTED.items()}


def stage_runtime(tag: str, sources: list[Path]) -> int:
    if tag not in RUNTIMES:
        print(f"unknown platform tag {tag!r}; expected one of {sorted(RUNTIMES)}")
        return 2
    name = RUNTIMES[tag]
    destination = _libs_dir() / tag
    for source in sources:
        candidate = source / name
        if candidate.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination / name)
            print(f"staged {tag} runtime: {name}")
            return 0
    print(f"{tag}: runtime {name} not found in {[str(source) for source in sources]}")
    return 1


def build(use_uv: bool) -> int:
    libs = _libs_dir()
    libs.mkdir(parents=True, exist_ok=True)
    present = sorted(p.name for p in libs.iterdir() if p.is_dir())
    missing = [tag for tag in EXPECTED if tag not in present]
    if missing:
        print(f"warning: no libraries staged for {missing}; wheel will not bundle them")
    for tag in present:
        if tag not in EXPECTED:
            print(f"warning: unexpected _libs entry {tag}")
            continue
        name = RUNTIMES[tag]
        if not (libs / tag / name).is_file():
            print(f"warning: {tag}/{name} missing")
    if use_uv:
        return subprocess.call(["uv", "build", "--wheel"], cwd=ROOT)
    return subprocess.call([sys.executable, "-m", "build", "--wheel"], cwd=ROOT)


def _zip_entry(
    name: str, contents: bytes, mode: int = 0o644
) -> tuple[zipfile.ZipInfo, bytes]:
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.create_system = 3
    entry.external_attr = mode << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    return entry, contents


def _bundle_bytes(
    clients: Path, tag: str, platforms: tuple[str, ...], revision: str
) -> bytes:
    source = clients / f"lupine-client-{tag}"
    missing = [name for name in EXPECTED[tag] if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{tag}: missing native libraries: {missing}")

    files = []
    payloads: list[tuple[zipfile.ZipInfo, bytes]] = []
    for name in EXPECTED[tag]:
        contents = (source / name).read_bytes()
        files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "mode": "0755",
            }
        )
        payloads.append(_zip_entry(name, contents, 0o755))
    manifest = json.dumps(
        {
            "schema": 1,
            "platforms": list(platforms),
            "source_revision": revision,
            "files": files,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        entry, contents = _zip_entry("manifest.json", manifest)
        archive.writestr(entry, contents)
        for entry, contents in payloads:
            archive.writestr(entry, contents)
    return output.getvalue()


def bundles(clients: Path, output: Path, revision: str) -> int:
    """Build the server-served, content-addressed client objects."""

    if not revision:
        print("source revision must not be empty")
        return 2
    output.mkdir(parents=True, exist_ok=True)
    entries: dict[str, object] = {}
    index: dict[str, object] = {
        "schema": 1,
        "source_revision": revision,
        "bundles": entries,
    }

    try:
        for platforms, tag in CLIENT_BUNDLES.items():
            contents = _bundle_bytes(clients, tag, platforms, revision)
            digest = hashlib.sha256(contents).digest()
            digest_hex = digest.hex()
            etag = f'"sha256:{digest_hex}"'
            content_digest = f"sha-256=:{base64.b64encode(digest).decode()}:"
            for platform_name in platforms:
                destination = output / platform_name
                destination.mkdir(parents=True, exist_ok=True)
                bundle_path = destination / "client.zip"
                bundle_path.write_bytes(contents)
                bundle_path.with_suffix(".zip.etag").write_text(etag + "\n")
                bundle_path.with_suffix(".zip.digest").write_text(content_digest + "\n")
                entries[platform_name] = {
                    "etag": etag,
                    "content_digest": content_digest,
                    "size": len(contents),
                }
    except FileNotFoundError as exc:
        print(exc)
        return 1

    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"built {len(entries)} client bundle routes in {output}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "stage-runtime":
        if len(argv) < 4:
            print("stage-runtime needs: <tag> <dir> [<dir>...]")
            return 2
        return stage_runtime(argv[2], [Path(p) for p in argv[3:]])
    if argv[1] == "build":
        return build(use_uv=argv[2:3] != ["--no-uv"])
    if argv[1] == "bundles":
        if len(argv) != 5:
            print("bundles needs: <complete-client-dir> <output-dir> <source-revision>")
            return 2
        return bundles(Path(argv[2]), Path(argv[3]), argv[4])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
