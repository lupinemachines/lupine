"""Wheel dependencies and late-loaded ELF helpers join one portable closure."""

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "cuda_wheels", ROOT / "cmake/cuda_wheels.py"
)
wheels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheels)


def write_wheel(directory, name, files, requires=""):
    output = directory / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(output, "w") as archive:
        metadata = f"Name: {name}\nVersion: 1.0\n"
        if requires:
            metadata += f"Requires-Dist: {requires}\n"
        archive.writestr(f"{name}-1.0.dist-info/METADATA", metadata)
        for filename, data in files.items():
            archive.writestr(f"nvidia/{name}/lib/{filename}", data)
    return output


def test_collects_transitive_helpers_and_records_provenance(tmp_path, monkeypatch):
    destination = tmp_path / "libraries"
    destination.mkdir()
    calls = []

    def download(command, check):
        calls.append(command)
        directory = Path(command[command.index("--dest") + 1])
        write_wheel(
            directory,
            "consumer",
            {"libconsumer.so": b"\x7fELFconsumer", "notes.txt": b"notes"},
            "helper>=1",
        )
        write_wheel(directory, "helper", {"libhelper.so": b"\x7fELFhelper"})

    monkeypatch.setattr(wheels.subprocess, "run", download)
    libraries, packages = wheels.download_libraries(
        tmp_path / "wheels", ["consumer==1.0"], destination
    )
    assert {p.name for p in libraries} == {"libconsumer.so", "libhelper.so"}
    assert "--no-deps" not in calls[0]
    assert {p["package"] for p in packages} == {"consumer==1.0", "helper==1.0"}
    assert packages[0]["requires_dist"] == ["helper>=1"]
    for package in packages:
        assert (
            package["sha256"]
            == hashlib.sha256(
                (tmp_path / "wheels" / package["wheel"]).read_bytes()
            ).hexdigest()
        )


def test_conflicting_library_basenames_are_rejected(tmp_path, monkeypatch):
    destination = tmp_path / "libraries"
    destination.mkdir()

    def download(command, check):
        directory = Path(command[command.index("--dest") + 1])
        write_wheel(directory, "first", {"libshared.so": b"\x7fELFfirst"})
        write_wheel(directory, "second", {"libshared.so": b"\x7fELFsecond"})

    monkeypatch.setattr(wheels.subprocess, "run", download)
    with pytest.raises(ValueError, match="conflicting library"):
        wheels.download_libraries(tmp_path / "wheels", ["first==1.0"], destination)


def test_lock_replay_requires_hash_verification(tmp_path, monkeypatch):
    destination = tmp_path / "libraries"
    destination.mkdir()
    lock = tmp_path / "packages.lock"
    lock.write_text("helper==1.0 --hash=sha256:example\n")
    commands = []

    def download(command, check):
        commands.append(command)
        write_wheel(
            Path(command[command.index("--dest") + 1]),
            "helper",
            {"libhelper.so": b"\x7fELFhelper"},
        )

    monkeypatch.setattr(wheels.subprocess, "run", download)
    wheels.download_libraries(tmp_path / "wheels", [], destination, lock)
    assert "--require-hashes" in commands[0]
    assert commands[0][commands[0].index("--requirement") + 1] == str(lock)
