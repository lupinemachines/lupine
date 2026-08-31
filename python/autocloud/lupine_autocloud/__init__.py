"""Opt-in startup configuration for LUPINE's hosted cloud GPU."""

from __future__ import annotations

import os

DEFAULT_SERVER = "demo.lupinemachines.com:14833"
_DISABLED_VALUES = {"0", "false", "no", "off"}


def enabled() -> bool:
    """Return whether the autocloud startup hook is enabled."""

    value = os.environ.get("LUPINE_AUTOCLOUD", "1")
    return value.strip().lower() not in _DISABLED_VALUES


def activate() -> dict[str, str]:
    """Select the default cloud endpoint and preload LUPINE's native shims."""

    if not enabled():
        return {}

    os.environ.setdefault("LUPINE_SERVER", DEFAULT_SERVER)

    import lupine

    return dict(lupine.load_native(missing_ok=False))


__all__ = ["DEFAULT_SERVER", "activate", "enabled"]
