import hashlib
import json
import zipfile

import build


def test_stage_runtime_copies_only_the_wheel_stub(monkeypatch, tmp_path):
    source = tmp_path / "complete-client"
    source.mkdir()
    for name in build.EXPECTED["linux-x86_64"]:
        (source / name).write_bytes(name.encode())
    destination = tmp_path / "wheel-libs"
    monkeypatch.setattr(build, "_libs_dir", lambda: destination)

    assert build.stage_runtime("linux-x86_64", [source]) == 0
    assert {path.name for path in (destination / "linux-x86_64").iterdir()} == {
        "libcudart.so.13"
    }


def test_builds_every_client_route_with_strong_etags(tmp_path):
    clients = tmp_path / "clients"
    for tag, names in build.EXPECTED.items():
        directory = clients / f"lupine-client-{tag}"
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(f"{tag}/{name}".encode())

    output = tmp_path / "bundles"
    assert build.bundles(clients, output, "deadbeef") == 0

    index = json.loads((output / "index.json").read_text())
    assert set(index["bundles"]) == {
        "linux/amd64",
        "linux/arm64",
        "macos/amd64",
        "macos/arm64",
        "windows/amd64",
        "windows/arm64",
    }
    assert index["source_revision"] == "deadbeef"

    for platform_name, metadata in index["bundles"].items():
        path = output / platform_name / "client.zip"
        contents = path.read_bytes()
        assert metadata["etag"] == f'"sha256:{hashlib.sha256(contents).hexdigest()}"'
        assert path.with_suffix(".zip.etag").read_text().strip() == metadata["etag"]
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        assert manifest["source_revision"] == "deadbeef"
        assert platform_name in manifest["platforms"]

    assert (output / "macos/amd64/client.zip").read_bytes() == (
        output / "macos/arm64/client.zip"
    ).read_bytes()

    repeated = tmp_path / "repeated"
    assert build.bundles(clients, repeated, "deadbeef") == 0
    for platform_name in index["bundles"]:
        assert (output / platform_name / "client.zip").read_bytes() == (
            repeated / platform_name / "client.zip"
        ).read_bytes()
