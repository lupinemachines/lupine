import os
import sys
from pathlib import Path

import pytest

from lupine import (
    LupineError,
    _native,
    connect,
    is_configured,
    servers,
)
from lupine import (
    _normalize_hosts as normalize_hosts,
)
from lupine import (
    _normalize_server as normalize_server,
)


def test_normalize_server_appends_default_port():
    assert normalize_server("host") == "host:14833"
    assert normalize_server("host:9999") == "host:9999"
    assert normalize_server("host", 1234) == "host:1234"
    assert normalize_server("host:9999", 1234) == "host:1234"


def test_normalize_server_ipv6():
    assert normalize_server("fe80::1") == "[fe80::1]:14833"
    assert normalize_server("[fe80::1]:2000") == "[fe80::1]:2000"
    assert normalize_server("http://[fe80::1]") == "http://[fe80::1]:14833"


def test_normalize_server_urls():
    assert normalize_server("http://h") == "http://h:14833"
    assert normalize_server("http://h:2000") == "http://h:2000"
    assert normalize_server("https://h") == "https://h"
    assert normalize_server("https://h", 1500) == "https://h:1500"


def test_normalize_server_rejects_empty():
    for bad in ("", "  ", "http://"):
        with pytest.raises(LupineError):
            normalize_server(bad)


def test_normalize_hosts_sequence():
    assert normalize_hosts(["a", "b:2"]) == ("a:14833", "b:2")
    assert normalize_hosts("a") == ("a:14833",)


def test_servers_from_env_splits():
    os.environ["LUPINE_SERVER"] = "a:1, b:2 ,"
    try:
        assert servers() == ("a:1", "b:2")
        assert is_configured()
    finally:
        del os.environ["LUPINE_SERVER"]
    assert not is_configured()


def test_session_requires_host_or_env(monkeypatch):
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    with pytest.raises(LupineError):
        connect()


def test_session_configures_env_and_restores(monkeypatch):
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    calls = []
    monkeypatch.setattr(
        _native, "load", lambda missing_ok=True: calls.append(missing_ok) or {}
    )
    with connect(host="gpu.example:14833") as session:
        assert os.environ["LUPINE_SERVER"] == "gpu.example:14833"
        assert calls == [False]
        assert session._loaded
    assert "LUPINE_SERVER" not in os.environ


def test_session_rejects_conflicting_env(monkeypatch):
    monkeypatch.setenv("LUPINE_SERVER", "other:14833")
    with pytest.raises(LupineError, match="configured differently"):
        connect(host="gpu.example:14833")


def test_session_accepts_matching_env(monkeypatch):
    monkeypatch.setenv("LUPINE_SERVER", "gpu.example:14833")
    monkeypatch.setattr(_native, "load", lambda missing_ok=True: {})
    session = connect(host="gpu.example:14833")
    assert session.servers == ("gpu.example:14833",)
    assert servers() == ("gpu.example:14833",)


def test_libdir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_LIBDIR", str(tmp_path))
    assert _native.libdir() == tmp_path


def test_libdir_keeps_configured_server(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_SERVER", "https://gw-east.lupine.sh")
    monkeypatch.delenv("LUPINE_LIBDIR", raising=False)
    monkeypatch.setattr(
        _native._bundles,
        "resolve",
        lambda servers, names: (
            tmp_path,
            '"sha256:' + "a" * 64 + '"',
            "linux/amd64",
            names,
        ),
    )

    assert _native.libdir() == tmp_path
    assert os.environ["LUPINE_SERVER"] == "https://gw-east.lupine.sh"


def test_load_missing_ok_without_libs(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_LIBDIR", str(tmp_path / "does-not-exist"))
    assert _native.load(missing_ok=True) == {}
    with pytest.raises(LupineError):
        _native.load(missing_ok=False)


def _stage(libdir, extra=()):
    """Write a client directory holding the required shims plus ``extra``."""

    libdir.mkdir(exist_ok=True)
    names = (*_native._REQUIRED[sys.platform], *extra)
    for name in names:
        (libdir / name).write_bytes(b"")
    return names


def test_loads_native_libraries(monkeypatch, tmp_path):
    libdir = tmp_path / "libs"
    names = _stage(libdir)

    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))
    monkeypatch.delenv("TRITON_LIBCUDA_PATH", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path, mode=None: None)
    result = _native.load(missing_ok=False)
    assert len(result) == len(names)
    if sys.platform in ("linux", "darwin"):
        assert os.environ["TRITON_LIBCUDA_PATH"] == str(libdir)
    # Idempotent: second call loads nothing new.
    monkeypatch.setattr(
        _native.ctypes,
        "CDLL",
        lambda *a, **k: pytest.fail("should not load again"),
    )
    assert _native.load() == result

def test_load_respects_existing_triton_libcuda_path(monkeypatch, tmp_path):
    libdir = tmp_path / "libs"
    _stage(libdir)

    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))
    monkeypatch.setenv("TRITON_LIBCUDA_PATH", "/caller/libcuda")
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path, mode=None: None)
    _native.load()

    assert os.environ["TRITON_LIBCUDA_PATH"] == "/caller/libcuda"


# A file no shim answers to, to show that only shim names are loaded.
_PROBE = {
    "linux": "liblupine_probe.so",
    "darwin": "liblupine_probe.dylib",
    "win32": "lupine_probe.dll",
}[sys.platform]


def _record_loads(monkeypatch):
    loaded = []
    monkeypatch.setattr(_native, "_loaded", {})
    monkeypatch.setattr(_native, "_names", ())
    monkeypatch.setattr(
        _native.ctypes,
        "CDLL",
        lambda path, mode=None: loaded.append(Path(path).name),
    )
    return loaded


def test_load_takes_the_driver_and_nvml_from_the_bundle(monkeypatch, tmp_path):
    libdir = tmp_path / "client"
    names = _stage(libdir, (_PROBE,))
    loaded = _record_loads(monkeypatch)
    monkeypatch.setattr(_native, "_names", names)
    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))

    result = _native.load(missing_ok=False)

    assert tuple(loaded) == _native._REQUIRED[sys.platform]
    assert loaded[0] == _native._DRIVER[sys.platform]
    assert set(result) == set(_native._REQUIRED[sys.platform])


def test_libdir_directory_is_filtered_like_a_bundle(monkeypatch, tmp_path):
    """LUPINE_LIBDIR has no manifest; its contents are filtered the same way."""

    libdir = tmp_path / "build"
    _stage(libdir, (_PROBE,))
    (libdir / "notes.txt").write_bytes(b"")
    loaded = _record_loads(monkeypatch)
    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))

    _native.load(missing_ok=False)

    assert tuple(loaded) == _native._REQUIRED[sys.platform]


@pytest.mark.skipif(sys.platform != "linux", reason="NCCL and nvSHMEM are Linux only")
def test_nccl_shim_loads_only_when_named_and_absent_natively(monkeypatch, tmp_path):
    libdir = tmp_path / "client"
    names = _stage(libdir, ("libnccl.so.2", "libnvshmem_host.so.3"))
    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))

    # Nothing native: both conditional shims join the driver and NVML.
    loaded = _record_loads(monkeypatch)
    monkeypatch.setattr(_native, "_names", names)
    monkeypatch.setattr(_native, "_native_available", lambda package, library: False)
    _native.load(missing_ok=False)
    assert tuple(loaded) == (
        *_native._REQUIRED["linux"],
        "libnccl.so.2",
        "libnvshmem_host.so.3",
    )

    # A native NCCL (an nvidia-nccl wheel or a system library) wins.
    loaded = _record_loads(monkeypatch)
    monkeypatch.setattr(_native, "_names", names)
    monkeypatch.setattr(
        _native, "_native_available", lambda package, library: library == "nccl"
    )
    _native.load(missing_ok=False)
    assert tuple(loaded) == (*_native._REQUIRED["linux"], "libnvshmem_host.so.3")

    # A bundle that does not name NCCL never loads a stray copy of it.
    loaded = _record_loads(monkeypatch)
    monkeypatch.setattr(_native, "_names", tuple(n for n in names if "nccl" not in n))
    monkeypatch.setattr(_native, "_native_available", lambda package, library: False)
    _native.load(missing_ok=False)
    assert "libnccl.so.2" not in loaded


@pytest.mark.skipif(sys.platform != "linux", reason="NCCL is Linux only")
def test_native_available_finds_the_nvidia_wheel(monkeypatch, tmp_path):
    # A stand-in for nvidia.nccl: the test environment may hold the real one.
    (tmp_path / "nvidia" / "lupine_test" / "lib").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_native.ctypes.util, "find_library", lambda name: None)
    assert _native._native_available("nvidia.lupine_test", "lupine_test")
    assert not _native._native_available("nvidia.lupine_absent", "lupine_absent")
    monkeypatch.setattr(_native.ctypes.util, "find_library", lambda name: "lib.so")
    assert _native._native_available("nvidia.lupine_absent", "lupine_absent")


def test_load_uses_the_bundle_manifest_names(monkeypatch, tmp_path):
    """A resolved bundle names its shims; a stray file must not be loaded."""

    libdir = tmp_path / "client"
    names = _stage(libdir)
    driver = _native._DRIVER[sys.platform]
    (libdir / driver.replace("cuda", "stray")).write_bytes(b"")
    loaded = _record_loads(monkeypatch)
    monkeypatch.setattr(_native, "_names", names)
    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))

    _native.load(missing_ok=False)

    assert tuple(loaded) == names
