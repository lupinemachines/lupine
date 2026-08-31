import os
import sys

import pytest

from lupine import (
    LupineError,
    _native,
    _normalize_hosts as normalize_hosts,
    _normalize_server as normalize_server,
    connect,
    is_configured,
    servers,
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


def test_load_missing_ok_without_libs(monkeypatch, tmp_path):
    monkeypatch.setenv("LUPINE_LIBDIR", str(tmp_path / "does-not-exist"))
    assert _native.load(missing_ok=True) == {}
    with pytest.raises(LupineError):
        _native.load(missing_ok=False)


def test_load_does_not_set_disable_local(monkeypatch, tmp_path):
    libdir = tmp_path / "libs"
    libdir.mkdir()
    for name in _native._LIBS[sys.platform]:
        (libdir / name).write_bytes(b"")

    seen_env = []

    class FakeCDLL:
        def __init__(self, path, mode=None):
            seen_env.append(os.environ.get("LUPINE_DISABLE_LOCAL"))

    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))
    monkeypatch.delenv("LUPINE_DISABLE_LOCAL", raising=False)
    monkeypatch.delenv("TRITON_LIBCUDA_PATH", raising=False)
    monkeypatch.setattr(_native.ctypes, "CDLL", FakeCDLL)
    result = _native.load(missing_ok=False)
    assert len(result) == len(_native._LIBS[sys.platform])
    assert seen_env == [None] * len(_native._LIBS[sys.platform])
    assert "LUPINE_DISABLE_LOCAL" not in os.environ
    if sys.platform in ("linux", "darwin"):
        assert os.environ["TRITON_LIBCUDA_PATH"] == str(libdir)
    # Idempotent: second call loads nothing new.
    monkeypatch.setattr(
        _native.ctypes,
        "CDLL",
        lambda *a, **k: pytest.fail("should not load again"),
    )
    assert _native.load() == result


def test_load_respects_existing_disable_local(monkeypatch, tmp_path):
    libdir = tmp_path / "libs"
    libdir.mkdir()
    for name in _native._LIBS[sys.platform]:
        (libdir / name).write_bytes(b"")

    class FakeCDLL:
        def __init__(self, path, mode=None):
            assert os.environ.get("LUPINE_DISABLE_LOCAL") == "0"

    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))
    monkeypatch.setenv("LUPINE_DISABLE_LOCAL", "0")
    monkeypatch.setattr(_native.ctypes, "CDLL", FakeCDLL)
    _native.load()
    assert os.environ.get("LUPINE_DISABLE_LOCAL") == "0"


def test_load_respects_existing_triton_libcuda_path(monkeypatch, tmp_path):
    libdir = tmp_path / "libs"
    libdir.mkdir()
    for name in _native._LIBS[sys.platform]:
        (libdir / name).write_bytes(b"")

    monkeypatch.setenv("LUPINE_LIBDIR", str(libdir))
    monkeypatch.setenv("TRITON_LIBCUDA_PATH", "/caller/libcuda")
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path, mode=None: None)
    _native.load()

    assert os.environ["TRITON_LIBCUDA_PATH"] == "/caller/libcuda"
