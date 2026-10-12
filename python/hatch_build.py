"""Package the separately built Mac runtime; never package driver shims."""

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class MacRuntimeHook(BuildHookInterface):
    def initialize(self, version, build_data):
        directory = os.environ.get("LUPINE_MACOS_RUNTIME_DIR")
        if self.target_name != "wheel" or not directory:
            return
        for name in ("libcudart.dylib", "liblupine_aten.dylib"):
            path = Path(directory).resolve() / name
            if not path.is_file():
                raise ValueError(f"Mac runtime package is missing {path}")
            build_data["force_include"][str(path)] = f"lupine/_libs/{name}"
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-macosx_26_0_arm64"
