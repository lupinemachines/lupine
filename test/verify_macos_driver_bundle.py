"""Verify the running server serves its staged macOS drivers, without a GPU."""

import base64
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def verify(server: str, inputs: Path) -> None:
    for architecture, tag in (("amd64", "macosx-universal2"), ("arm64", "macosx-arm64")):
        directory = inputs / f"lupine-client-{tag}"
        if not directory.exists():
            directory = inputs / "lupine-client-macosx-universal2"
        url = f"{server}/.well-known/lupine/client/v1/macos/{architecture}"
        for attempt in range(30):
            try:
                response = urllib.request.urlopen(url, timeout=5)
                break
            except urllib.error.URLError:
                if attempt == 29:
                    raise
                time.sleep(1)
        with response:
            body = response.read()
            digest = hashlib.sha256(body).digest()
            etag = f'"sha256:{digest.hex()}"'
            assert response.headers["ETag"] == etag
            assert response.headers["Content-Digest"] == (
                f"sha-256=:{base64.b64encode(digest).decode()}:"
            )
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            expected = {"libcuda.dylib", "libnvidia-ml.dylib"}
            assert set(archive.namelist()) == expected | {"manifest.json"}
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["schema"] == 1
            assert f"macos/{architecture}" in manifest["platforms"]
            assert {item["path"] for item in manifest["files"]} == expected
            for item in manifest["files"]:
                data = archive.read(item["path"])
                assert data == (directory / item["path"]).read_bytes()
                assert item["sha256"] == hashlib.sha256(data).hexdigest()
        request = urllib.request.Request(url, headers={"If-None-Match": etag})
        try:
            with urllib.request.urlopen(request, timeout=5):
                raise AssertionError("unchanged driver bundle must return 304")
        except urllib.error.HTTPError as error:
            assert error.code == 304
            assert error.headers["ETag"] == etag
        print(f"Verified server-selected macos/{architecture} driver and NVML")


if __name__ == "__main__":
    verify(sys.argv[1].rstrip("/"), Path(sys.argv[2]))
