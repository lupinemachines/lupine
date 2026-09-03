"""Automatic startup configuration for LUPINE."""

from __future__ import annotations

import importlib.abc
import os
import sys
import threading
import warnings
from importlib.machinery import ModuleSpec
from types import ModuleType

DEFAULT_SERVER = "https://api.lupine.sh"
_DISABLED_VALUES = {"0", "false", "no", "off"}
_CUDA_IMPORTS = {"cupy", "jax", "jaxlib", "numba", "tensorflow", "torch", "triton"}
_cloud_session = None
_finder = None


def enabled() -> bool:
    """Return whether the automatic startup hook is enabled."""

    value = os.environ.get("LUPINE_AUTO", "1")
    return value.strip().lower() not in _DISABLED_VALUES


def activate() -> dict[str, str]:
    """Authenticate if needed, bind a cloud GPU, and preload native shims."""

    if not enabled():
        return {}

    import lupine

    server = os.environ.get("LUPINE_SERVER", "").strip()
    session = os.environ.get("LUPINE_SESSION", "").strip()
    api_url = (os.environ.get("LUPINE_API_URL") or DEFAULT_SERVER).rstrip("/")
    if not session and (not server or server.rstrip("/") == api_url):
        global _cloud_session
        if _cloud_session is None:
            gpu_count = os.environ.get("LUPINE_GPU_COUNT")
            try:
                parsed_gpu_count = int(gpu_count) if gpu_count else None
            except ValueError as exc:
                raise lupine.LupineError("LUPINE_GPU_COUNT must be an integer") from exc
            _cloud_session = lupine.cloud(
                api_url=api_url,
                gpu_type=os.environ.get("LUPINE_GPU_TYPE"),
                gpu_count=parsed_gpu_count,
                region=os.environ.get("LUPINE_REGION"),
            )
        return dict(_cloud_session.loaded)

    if session and not server:
        raise lupine.LupineError(
            "LUPINE_SERVER must name the bound gateway when LUPINE_SESSION is set"
        )
    return dict(lupine.load_native(missing_ok=False))


def startup() -> dict[str, str]:
    """Site hook that reports actionable activation failures without aborting Python."""

    import lupine

    try:
        return activate()
    except lupine.LupineError as exc:
        warnings.warn(f"lupine[auto] was not activated: {exc}", RuntimeWarning)
        return {}


class _AutoFinder(importlib.abc.MetaPathFinder):
    """Activate LUPINE immediately before the first CUDA consumer imports."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._attempted = False

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.partition(".")[0] not in _CUDA_IMPORTS:
            return None
        with self._lock:
            if self._attempted:
                return None
            self._attempted = True
            startup()
        return None


def install() -> _AutoFinder | None:
    """Install the one-shot lazy activation hook."""

    if not enabled():
        return None
    global _finder
    if _finder is None:
        _finder = _AutoFinder()
        sys.meta_path.insert(0, _finder)
    return _finder


__all__ = ["DEFAULT_SERVER", "activate", "enabled", "install", "startup"]
