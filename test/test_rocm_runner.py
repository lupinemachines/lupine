#!/usr/bin/env python3
"""CPU-only checks for the ROCm harness: python3 -m unittest discover -s test -p test_rocm_runner.py."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rocm_runner import Server, Suite, Unit, execute, native_environment, shim_environment, verify_loader


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = native_environment(self.root)

    def run_unit(self, command, timeout=5):
        return execute(Unit("test", self.root, command), self.env, timeout, self.root / "run.log")

    def test_native_environment_drops_client_settings(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "shim.so", "LUPINE_SERVER": "remote:1234",
                                     "LD_LIBRARY_PATH": "/shim", "LD_BIND_NOW": "1"}):
            env = native_environment(self.root)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("LUPINE_SERVER", env)
        self.assertNotIn("/shim", env["LD_LIBRARY_PATH"])

    def test_empty_test_selection_is_not_a_pass(self):
        self.assertEqual("FAIL:output", self.run_unit(
            [sys.executable, "-c", "print('[==========] Running 0 tests from 0 test suites.')"]))

    def test_timeout_reaps_child_processes(self):
        marker = self.root / "leaked-child"
        status = self.run_unit(["sh", "-c", '(sleep 0.3; touch "$1") & wait', "sh", str(marker)], 0.05)
        self.assertEqual("FAIL:124", status)
        time.sleep(0.4)
        self.assertFalse(marker.exists())

    def test_existing_listener_is_not_accepted_as_our_server(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with patch.dict(os.environ, {"SERVER_MODE": "local", "SERVER_HOST": "127.0.0.1"}):
                server = Server(self.env)
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                with server.running(port, self.root / "server.log"):
                    self.fail("used a server owned by another invocation")

    @unittest.skipUnless(shutil.which("setsid"), "needs setsid")
    def test_remote_shell_stops_its_server_process_group(self):
        executable = self.root / "server"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import os, socket\n"
            "server = socket.socket()\n"
            "server.bind(('127.0.0.1', int(os.environ['LUPINE_PORT'])))\n"
            "server.listen()\n"
            "print('ready', flush=True)\n"
            "while True:\n"
            "    connection, _ = server.accept()\n"
            "    connection.close()\n")
        executable.chmod(0o755)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with patch.dict(os.environ, {"SERVER_MODE": "ssh", "SERVER_HOST": "127.0.0.1",
                                     "SERVER_REMOTE_BIN": str(executable)}):
            server = Server(self.env)
        # Exercise the same POSIX shell commands without needing SSH credentials.
        server.ssh = ["sh", "-c"]
        log = self.root / "server.log"
        with server.running(port, log):
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1)
        self.assertIn("ready", log.read_text())
        self.assertFalse(Path(f"{executable}-{port}.log").exists())

    @unittest.skipUnless(shutil.which("cc"), "needs a C compiler")
    def test_missing_hip_symbols_cannot_fall_back_to_native_runtime(self):
        native = self.root / "native"
        aliases = self.root / "aliases"
        native.mkdir()
        aliases.mkdir()
        (native / "runtime.c").write_text("int hipMissing(void) { return 0; }\n")
        (native / "version.map").write_text("hip_4.2 { global: hipMissing; };\n")
        (self.root / "shim.c").write_text("int hipPresent(void) { return 0; }\n")
        (self.root / "main.c").write_text("int hipMissing(void); int main(void) { return hipMissing(); }\n")
        shim = self.root / "libamdhip64.so.1"
        for command in (
            ["cc", "-shared", "-fPIC", str(native / "runtime.c"), "-Wl,-soname,libamdhip64.so.7",
             f"-Wl,--version-script,{native / 'version.map'}",
             "-o", str(native / "libamdhip64.so.7")],
            ["cc", "-shared", "-fPIC", str(self.root / "shim.c"), "-Wl,-soname,libamdhip64.so.1", "-o", str(shim)],
            ["cc", str(self.root / "main.c"), f"-L{native}", "-l:libamdhip64.so.7", f"-Wl,-rpath,{native}",
             "-o", str(self.root / "client")],
        ):
            subprocess.run(command, check=True, env=self.env)
        unit = Unit("client", self.root, [str(self.root / "client")], executables=[self.root / "client"])
        self.assertEqual("PASS", execute(unit, self.env, 5, self.root / "native.log"))
        with patch.dict(os.environ, {"LUPINE_LIB": str(shim)}):
            env = shim_environment(self.env, aliases)
        verify_loader(unit, env)
        self.assertTrue(execute(unit, env, 5, self.root / "shim.log").startswith("FAIL:"))
        self.assertIn("undefined symbol: hipMissing", (self.root / "shim.log").read_text())
        # Old DT_RPATH overrides the alias directory. Refuse such a binary
        # rather than reporting a native execution as a Lupine pass.
        subprocess.run(["cc", str(self.root / "main.c"), f"-L{native}", "-l:libamdhip64.so.7",
                        f"-Wl,--disable-new-dtags,-rpath,{native}", "-o", str(self.root / "client")],
                       check=True, env=self.env)
        with self.assertRaisesRegex(RuntimeError, "not the Lupine shim"):
            verify_loader(unit, env)

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja") and shutil.which("c++"),
                         "needs CMake, Ninja and a C++ compiler")
    def test_failed_rebuild_does_not_run_a_stale_executable(self):
        sample = self.root / "source/HIP-Basic/sample"
        sample.mkdir(parents=True)
        (sample / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.21)\nproject(sample LANGUAGES CXX)\nadd_executable(sample main.cpp)\n")
        (sample / "main.cpp").write_text("int main() { return 0; }\n")
        compiler = self.root / "rocm/bin/hipcc"
        compiler.parent.mkdir(parents=True)
        compiler.symlink_to(shutil.which("c++"))
        with patch.dict(os.environ, {"ROCM_EXAMPLES_DIR": str(self.root / "source"),
                                     "ROCM_PATH": str(self.root / "rocm"), "ROCM_ARCH": "gfx1100",
                                     "BUILD_SAMPLES": "1"}):
            suite = Suite("rocm-examples")
        units = suite.units("HIP-Basic/sample")
        self.assertEqual("", units[0].status)
        old_executable = units[0].command[0]
        (sample / "main.cpp").write_text("#error deliberate rebuild failure\n")
        units = suite.units("HIP-Basic/sample")
        self.assertTrue(Path(old_executable).exists())
        self.assertEqual("SKIP:build-failed", units[0].status)
        self.assertEqual([], units[0].command)


if __name__ == "__main__":
    unittest.main()
