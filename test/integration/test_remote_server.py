"""Exercise the remote lifecycle on localhost without SSH or a GPU."""

import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


class RemoteServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = self.root / "server"
        self.server.write_text(
            "#!/usr/bin/env python3\n"
            "import os, socket, time\n"
            "time.sleep(0.1)\n"
            "s = socket.socket()\n"
            "s.bind(('127.0.0.1', int(os.environ['LUPINE_PORT'])))\n"
            "s.listen()\n"
            "s.accept()\n"
            "raise RuntimeError('readiness must not open a connection')\n"
        )
        self.server.chmod(0o755)

    def run_lifecycle(self, port):
        env = dict(os.environ, SERVER_REMOTE_BIN=str(self.server),
                   SERVER_LD_LIBRARY_PATH="", TEST_ROOT=str(self.root),
                   TEST_PORT=str(port))
        return subprocess.run(
            ["bash", "-c", '''
set -eu
source "$1"
ssh_with_timeout() { bash -c "$1"; }
pidfile="$TEST_ROOT/server.pid"
log="$TEST_ROOT/server.log"
trap 'stop_remote_server "$pidfile" "$log"' EXIT
start_remote_server "$pidfile" "$log" "$TEST_PORT"
kill -0 "$(cat "$pidfile")"
test ! -s "$log"
''', "bash", str(Path(__file__).with_name("remote_server.sh"))],
            env=env, capture_output=True, text=True, timeout=45,
        )

    def test_delayed_listener_is_ready_without_accepting_a_connection(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        result = self.run_lifecycle(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "server.pid").exists())
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)

    def test_failed_process_is_not_ready_and_is_cleaned_up(self):
        self.server.write_text("#!/bin/sh\necho startup-failed >&2\nexit 1\n")
        result = self.run_lifecycle(1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("startup-failed", result.stderr)
        self.assertFalse((self.root / "server.pid").exists())

    def test_another_process_listener_does_not_count_as_ready(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            result = self.run_lifecycle(listener.getsockname()[1])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Address already in use", result.stderr)
            self.assertFalse((self.root / "server.pid").exists())


if __name__ == "__main__":
    unittest.main()
