"""Protect CI coverage and ensure sparse selections do not expand silently."""

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from run_topologies import Runner
from validate_specs import MODES, WORKLOADS, ROOT, load_specs


class RecordingRunner(Runner):
    def __init__(self, tests):
        super().__init__({"hosts": {}, "run_id": "test"}, Path("key"), Path("artifacts"), Path("results"), tests)
        self.native = {role: {"slots": [0, 1]} for role in ("a", "b")}
        self.selected = []

    def gpu(self, reference):
        return reference

    def native_peer(self, source, destination):
        return 1

    def test(self, name, role, mode, expected, env, pair=(), peer=None, can_peer=False):
        return subprocess.CompletedProcess([], 0, (mode, pair)), 0, None

    def record(self, name, result, elapsed=0, skip_reason=None):
        mode, pair = result.stdout
        self.selected.append((name, mode, pair))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.matrix, self.selections = load_specs()

    def collect(self, name):
        run = self.selections["runs"][name]
        modes = [mode for mode in WORKLOADS if any(mode in selected for selected in run["cases"].values())]
        runner = RecordingRunner(modes)
        runner.baseline()
        for name, modes in run["cases"].items():
            topology = self.matrix["topologies"][name]
            devices = topology["expect_devices"]
            runner.workloads(name, topology["client"]["host"], devices, list(range(len(devices))), {}, modes)
        return runner.selected

    def test_regression_retains_all_existing_coverage(self):
        selected = self.collect("regression")
        original = [(name, mode, pair) for name, mode, pair in selected if mode in MODES]
        self.assertEqual(len(original), 172)
        self.assertEqual(sum(name.startswith("native-") for name, _, _ in original), 20)
        self.assertEqual(len(selected), 228)  # Eight samples on five layouts + both native hosts.
        self.assertEqual({name.split(".")[0] for name, _, _ in selected if not name.startswith("native-")},
                         set(self.matrix["topologies"]))

    def test_new_platforms_only_run_selected_smoke_cases(self):
        for name in ("linux-versions", "arm64-client"):
            with self.subTest(run=name):
                selected = self.collect(name)
                self.assertEqual(len(selected), 5)  # Two native controls + three cases.
                self.assertEqual({mode for _, mode, _ in selected}, {"kernel"})
                self.assertEqual({case.split(".")[0] for case, _, _ in selected if not case.startswith("native-")},
                                 set(self.selections["runs"][name]["cases"]))

    def test_single_peer_mode_does_not_enable_other_peer_tests(self):
        runner = RecordingRunner([])
        devices = ["a/0", "a/1", "b/0", "b/1"]
        runner.workloads("copy-only", "client", devices, [0, 0, 1, 1], {}, ["peer-copy"])
        self.assertEqual(len(runner.selected), 12)
        self.assertEqual({mode for _, mode, _ in runner.selected}, {"peer-copy"})
        self.assertEqual({pair for _, _, pair in runner.selected},
                         {(src, dst) for src in range(4) for dst in range(4) if src != dst})

    def test_invalid_selections_fail_before_provisioning(self):
        mutations = [
            lambda spec: spec["runs"]["regression"]["cases"].update({"typo": ["kernel"]}),
            lambda spec: spec["runs"]["regression"]["cases"].update({"remote-single": ["typo"]}),
            lambda spec: spec["runs"]["regression"]["cases"].update({"remote-single": []}),
            lambda spec: spec["runs"]["regression"]["hosts"].update({"a": "ubuntu-2404-arm64"}),
            lambda spec: spec["platforms"]["ubuntu-2404-x86_64"].update({"os": "windows"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.yaml"
            for mutate in mutations:
                spec = deepcopy(self.selections)
                mutate(spec)
                path.write_text(yaml.safe_dump(spec))
                with self.assertRaises(ValueError):
                    load_specs(ROOT / "topologies.yaml", path)


if __name__ == "__main__":
    unittest.main()
