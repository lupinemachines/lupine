"""Exercise concurrency, driver enforcement and result classification without GPUs."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

from run_topologies import Runner
from validate_specs import ROOT, SAMPLES


class RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runner = Runner({"hosts": {}, "run_id": "test"}, Path("key"), Path("artifacts"),
                             Path(directory.name), [])
        self.runner.gpu = lambda reference: reference
        self.runner.native_peer = lambda source, destination: 0
        self.runner.native = {"a": {"driver_library": "/native/libcuda.so.1"}}

    def test_eight_clients_overlap_and_all_results_are_collected_before_return(self):
        runner = self.runner
        barrier = threading.Barrier(8, timeout=5)
        lock = threading.Lock()
        active = started = peak = 0
        record_threads = []
        original_record = runner.record

        def execute(*args):
            nonlocal active, started, peak
            with lock:
                index = started
                started += 1
                active += 1
                peak = max(peak, active)
            try:
                if index < 8:
                    barrier.wait()
                if index == 0:
                    raise RuntimeError("client launch failed")
                return subprocess.CompletedProcess([], 0, "completed\n"), 0.1, None
            finally:
                with lock:
                    active -= 1

        def record(*args):
            record_threads.append(threading.get_ident())
            original_record(*args)

        runner.test = execute
        runner.record = record
        with redirect_stdout(io.StringIO()):
            runner.workloads("copy", "client", ["a/0", "a/1", "b/0", "b/1"],
                             [0, 0, 1, 1], {}, ["peer-copy"])
        self.assertEqual((started, peak, active), (12, 8, 0))
        self.assertEqual(set(record_threads), {threading.get_ident()})
        cases = json.loads((runner.results / "results.json").read_text())
        self.assertEqual(len({case["name"] for case in cases}), 12)
        self.assertEqual(sum(case["returncode"] != 0 for case in cases), 1)
        suite = ET.parse(runner.results / "junit.xml").getroot()
        self.assertEqual((suite.get("tests"), suite.get("failures")), ("12", "1"))

    def test_success_without_guard_is_failure(self):
        self.runner.command = Mock(return_value=subprocess.CompletedProcess([], 0, "PASS\n"))
        result, _, skip = self.runner.test("test", "a", "kernel", ["a/0"], {})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing in-process", result.stdout)
        self.assertIsNone(skip)

    def test_waiver_requires_no_peer_access_and_driver_attestation(self):
        for can_peer, attest, reason, waived in (
                (False, True, True, True), (True, True, True, False),
                (False, False, True, False), (False, True, False, False)):
            with self.subTest(can_peer=can_peer, attest=attest, reason=reason):
                def command(role, args, env, **kwargs):
                    if "enumeration" in args:
                        return subprocess.CompletedProcess(args, 0, "device map verified\n")
                    marker = f"LUPINE_TEST_DRIVER_OK {env['LUPINE_TEST_EXECUTABLE']} {env['LUPINE_TEST_DRIVER']}\n"
                    output = marker if attest else ""
                    output += SAMPLES["simpleP2P"]["peer_waiver"] if reason else "unrelated waiver"
                    return subprocess.CompletedProcess(args, 2, output)
                self.runner.command = command
                result, elapsed, skip = self.runner.test("test", "a", "sample:simpleP2P",
                    ["a/0", "a/1"], {}, can_peer=can_peer)
                self.assertEqual(bool(skip), waived)
                with redirect_stdout(io.StringIO()):
                    self.runner.cases.clear()
                    self.runner.record("test", result, elapsed, skip)
                suite = ET.parse(self.runner.results / "junit.xml").getroot()
                self.assertEqual(suite.get("skipped"), "1" if waived else "0")
                self.assertEqual(suite.get("failures"), "0" if waived else "1")


class DriverGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.guard = cls.root / "guard.so"
        subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                        str(ROOT / "driver_guard.cpp"), "-ldl", "-o", str(cls.guard)], check=True)
        # Tiny ELF fixtures test loader behavior only; GPU correctness is tested
        # on real L4s in CI.
        source = cls.root / "driver.c"
        source.write_text("void cuInit(void) {}\nvoid cuGetProcAddress(void) {}\n")
        cls.driver = cls.root / "libcuda.so.1"
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(cls.driver)], check=True)

    def run_guard(self, expected, executable="/usr/bin/true", command=None):
        env = dict(os.environ, LD_PRELOAD=str(self.guard), LD_LIBRARY_PATH=str(self.root),
                   LUPINE_TEST_EXECUTABLE=executable, LUPINE_TEST_DRIVER=str(expected))
        return subprocess.run(command or [executable], env=env, capture_output=True, text=True)

    def test_expected_driver_is_attested_by_workload_only(self):
        result = self.run_guard(self.driver, command=["timeout", "5s", "/usr/bin/true"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("LUPINE_TEST_DRIVER_OK"), 1)

    def test_wrong_driver_fails_before_main(self):
        result = self.run_guard(self.guard)
        self.assertEqual(result.returncode, 126)
        self.assertIn("DRIVER GUARD FAILED", result.stderr)
        self.assertNotIn("LUPINE_TEST_DRIVER_OK", result.stderr)

    def test_helper_cannot_attest_for_a_different_executable(self):
        result = self.run_guard(self.driver, executable="/usr/bin/false", command=["/usr/bin/true"])
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("LUPINE_TEST_DRIVER_OK", result.stderr)


if __name__ == "__main__":
    unittest.main()
