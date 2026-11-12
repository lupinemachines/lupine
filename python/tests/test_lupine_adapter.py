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


def test_normalize_server_preserves_url_userinfo():
    assert normalize_server("http://token@gpu-host") == "http://token@gpu-host:14833"
    assert (
        normalize_server("http://user:token@gpu-host")
        == "http://user:token@gpu-host:14833"
    )
    assert (
        normalize_server("https://user:pass@gpu-host") == "https://user:pass@gpu-host"
    )
    assert (
        normalize_server("https://token@gpu-host", 8443)
        == "https://token@gpu-host:8443"
    )
    assert (
        normalize_server("http://user:pass@[fe80::1]")
        == "http://user:pass@[fe80::1]:14833"
    )


def test_normalize_server_rejects_empty():
    for bad in (
        "",
        "  ",
        ":",
        ":14833",
        "[]",
        "[]:14833",
        "host:",
        "http://",
        "https://",
        "http://:14833",
        "https://:443",
        "http://[]:14833",
    ):
        with pytest.raises(LupineError):
            normalize_server(bad)
    for bad in (":14833", "[]:14833", "http://:14833", "https://:443"):
        with pytest.raises(LupineError):
            normalize_server(bad, 8443)


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


def test_connect_keeps_launcher_configured_env_on_exit(monkeypatch):
    monkeypatch.setenv("LUPINE_SERVER", "gpu.example:14833")
    monkeypatch.setattr(_native, "load", lambda missing_ok=True: {})
    with connect(host="gpu.example:14833") as session:
        assert session._loaded
    assert os.environ["LUPINE_SERVER"] == "gpu.example:14833"


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


@pytest.mark.parametrize('manifest', [False, True])
def test_macos_runtime_loads_locally_after_native_driver(monkeypatch, tmp_path, manifest):
    libdir = tmp_path/'client'
    monkeypatch.setattr(_native.sys, 'platform', 'darwin')
    names = _stage(libdir, ('libcudart.dylib', 'liblupine_aten.dylib', 'unrelated.dylib'))
    loaded = []
    monkeypatch.setattr(_native, '_loaded', {})
    monkeypatch.setattr(_native, '_names', names if manifest else ())
    monkeypatch.setenv('LUPINE_LIBDIR', str(libdir))
    monkeypatch.setattr(_native.ctypes, 'CDLL',
                        lambda path, mode: loaded.append((Path(path).name, mode)))
    runtime_dir = tmp_path / 'runtime'
    runtime_dir.mkdir()
    for name in ('libcudart.dylib', 'liblupine_aten.dylib'):
        (runtime_dir / name).write_bytes(b'packaged-runtime')
    monkeypatch.setenv('LUPINE_RUNTIME_LIBDIR', str(runtime_dir))
    result = _native.load(missing_ok=False)
    assert Path(result['libcudart.dylib']).parent == runtime_dir
    assert loaded == [
        ('libcuda.dylib', _native.ctypes.RTLD_GLOBAL),
        ('libnvidia-ml.dylib', _native.ctypes.RTLD_GLOBAL),
        ('libcudart.dylib', _native.ctypes.RTLD_LOCAL),
        ('liblupine_aten.dylib', _native.ctypes.RTLD_LOCAL),
    ]
    assert set(result) == {name for name, mode in loaded}


def test_macos_downloads_driver_but_uses_packaged_runtime(monkeypatch, tmp_path):
    from test_bundles import bundle_bytes, bundle_server

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_native._bundles, "platform_name", lambda: "macos/arm64")
    monkeypatch.setenv("LUPINE_CACHE_DIR", str(tmp_path / "cache"))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_names = _native._RUNTIMES["darwin"]
    for name in runtime_names:
        (runtime / name).write_bytes(b"application-runtime")
    monkeypatch.setenv("LUPINE_RUNTIME_LIBDIR", str(runtime))
    calls = []
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path, mode: calls.append(Path(path)))
    monkeypatch.setattr(_native, "_loaded", {})
    monkeypatch.setattr(_native, "_names", ())
    # Even an older server carrying runtimes must not supply application code.
    body = bundle_bytes(platforms=("macos/arm64",), names=(
        *_native._REQUIRED["darwin"], *runtime_names,
    ))
    with bundle_server(body) as (server, etag, state):
        monkeypatch.delenv("LUPINE_LIBDIR", raising=False)
        monkeypatch.setenv("LUPINE_SERVER", server)
        loaded = _native.load(missing_ok=False)
        assert len(state["requests"]) == 1
        assert os.environ["LUPINE_CLIENT_ETAG"] == etag
        driver_directory = tmp_path / "cache" / "clients" / _native._bundles._etag_digest(etag)
        assert calls == [
            *(driver_directory / name for name in _native._REQUIRED["darwin"]),
            *(runtime / name for name in runtime_names),
        ]
        assert Path(loaded["libcuda.dylib"]).read_bytes() == b"libcuda.dylib"
        assert Path(loaded["libcudart.dylib"]).read_bytes() == b"application-runtime"


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


def _install_fake_torch(monkeypatch):
    from types import ModuleType, SimpleNamespace

    class FakeDevice:
        def __init__(self, kind, index=None):
            if isinstance(kind, FakeDevice):
                self.type, self.index = kind.type, kind.index
                return
            if isinstance(kind, str) and ":" in kind and index is None:
                kind, raw = kind.split(":", 1)
                index = int(raw)
            self.type = kind
            self.index = index

        def __repr__(self):
            return f"{self.type}:{self.index}" if self.index is not None else self.type

    fake_torch = ModuleType("torch")
    fake_torch.device = FakeDevice
    fake_torch.cuda = SimpleNamespace(
        get_device_name=None,
        get_device_capability=None,
        get_device_properties=None,
    )
    fake_torch._C = SimpleNamespace(_get_privateuse1_backend_name=lambda: "lupine")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "lupine._backend.device", raising=False)
    import lupine._backend.device as device_mod

    monkeypatch.setattr(device_mod, "current_device", lambda: 0)
    return device_mod


def test_native_backend_caches_device_properties(monkeypatch):
    import ctypes as ct
    from contextlib import contextmanager

    import lupine._backend as backend_pkg
    from lupine._backend.native import Backend

    device_mod = _install_fake_torch(monkeypatch)
    calls = {"attribute": 0, "mem_info": 0, "bind": 0, "get_name": 0}

    class FakeRuntime:
        closed = False
        driver = object()
        _cudnn = None
        _blas = None
        _operators = None

        @contextmanager
        def active(self):
            if self.closed:
                raise RuntimeError("Lupine native runtime is closed")
            yield

        @staticmethod
        def check(status, name):
            if status:
                raise RuntimeError(f"{name}: {status}")

        def attribute(self, out_ptr, attr, dev):
            calls["attribute"] += 1
            assert dev == 0
            ct.cast(out_ptr, ct.POINTER(ct.c_int))[0] = {75: 8, 76: 9, 16: 58}[attr]
            return 0

        def mem_info(self, free_ptr, total_ptr):
            calls["mem_info"] += 1
            ct.cast(free_ptr, ct.POINTER(ct.c_size_t))[0] = 20 * 1024**3
            ct.cast(total_ptr, ct.POINTER(ct.c_size_t))[0] = 24 * 1024**3
            return 0

        def bind(self, library, symbol, *types):
            calls["bind"] += 1
            assert library is self.driver and symbol == "cuDeviceGetName"

            def cu_device_get_name(buf, length, dev):
                calls["get_name"] += 1
                assert dev == 0 and length >= 10
                buf.value = b"NVIDIA L4"
                return 0

            return cu_device_get_name

        def sync(self):
            return 0

    backend = Backend.__new__(Backend)
    backend.runtime = FakeRuntime()
    backend.allocations = {}
    monkeypatch.setitem(backend_pkg._started, "backend", backend)

    assert backend.cuda_call("get_device_capability") == (8, 9)
    assert calls == {"attribute": 3, "mem_info": 1, "bind": 1, "get_name": 1}

    # Repeated queries (both direct and through device.__getattr__) hit the cache.
    for _ in range(5):
        assert backend.cuda_call("get_device_capability") == (8, 9)
        assert backend.cuda_call("get_device_capability", 0) == (8, 9)
        assert backend.cuda_call("get_device_name") == "NVIDIA L4"
        props = backend.cuda_call("get_device_properties")
        assert (props.name, props.major, props.minor) == ("NVIDIA L4", 8, 9)
        assert props.multi_processor_count == 58
        assert props.total_memory == 24 * 1024**3
        assert device_mod.get_device_capability() == (8, 9)
        assert device_mod.get_device_name() == "NVIDIA L4"
    assert calls == {"attribute": 3, "mem_info": 1, "bind": 1, "get_name": 1}

    with pytest.raises(ValueError, match="device 0"):
        backend.cuda_call("get_device_capability", 1)

    backend.close()
    assert backend._device_properties is None
    with pytest.raises(RuntimeError, match="closed"):
        backend.cuda_call("get_device_capability")
    monkeypatch.delitem(sys.modules, "lupine._backend.device", raising=False)


def test_stream_query_synchronizes(monkeypatch):
    from types import SimpleNamespace

    import lupine._backend as backend_pkg

    device_mod = _install_fake_torch(monkeypatch)
    syncs = []
    fake_backend = SimpleNamespace(
        runtime=SimpleNamespace(closed=False),
        synchronize=lambda: syncs.append(1),
    )
    monkeypatch.setitem(backend_pkg._started, "backend", fake_backend)

    stream = device_mod.Stream()
    assert syncs == []
    assert stream.query() is True
    assert syncs == [1]
    monkeypatch.delitem(sys.modules, "lupine._backend.device", raising=False)
