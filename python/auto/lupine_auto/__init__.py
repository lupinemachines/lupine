"""Automatic startup configuration for LUPINE."""

from __future__ import annotations

import os

DEFAULT_SERVER = "https://api.lupine.sh"
_DISABLED_VALUES = {"0", "false", "no", "off"}


def enabled() -> bool:
    """Return whether the automatic startup hook is enabled."""

    value = os.environ.get("LUPINE_AUTO", "1")
    return value.strip().lower() not in _DISABLED_VALUES


def activate() -> dict[str, str]:
    """Select the stable cloud endpoint and preload LUPINE's native shims."""

    if not enabled():
        return {}

    os.environ.setdefault("LUPINE_SERVER", DEFAULT_SERVER)

    import lupine

    return dict(lupine.load_native(missing_ok=False))


__all__ = ["DEFAULT_SERVER", "activate", "enabled"]
