"""Check offline discovery and shallow, version-matched test preparation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TEST_DIR = Path(__file__).resolve().parents[1]


class CudaPythonSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        self.results = self.root / "results"
        self.env = dict(os.environ, CUDA_PYTHON_DIR=str(self.checkout),
                        RESULTS_DIR=str(self.results), LIST_TESTS="1",
                        PYTHON_BIN="/missing/python", CUDA_HOME="/missing/cuda",
                        LUPINE_LIB="/missing/build/libcuda.so.1",
                        SERVER_LOCAL_BIN="/missing/server")

    def discover(self):
        return subprocess.run(["bash", str(TEST_DIR / "run_cuda_python_tests.sh")],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_list_modern_checkout_without_python_cuda_or_build_artifacts(self):
        tests = self.checkout / "cuda_bindings/tests"
        examples = self.checkout / "cuda_bindings/examples/intro"
        tests.mkdir(parents=True)
        examples.mkdir(parents=True)
        for name in ["test_cuda.py", "test_cudart.py", "test_nvvm.py", "test_examples.py"]:
            (tests / name).touch()
        (examples / "vector_add.py").touch()
        (examples / "__init__.py").touch()
        result = self.discover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["test_cuda.py", "test_cudart.py", "intro/vector_add.py"])
        self.assertFalse(self.results.exists())

    def test_list_legacy_checkout(self):
        tests = self.checkout / "cuda/tests"
        examples = self.checkout / "examples/intro"
        tests.mkdir(parents=True)
        examples.mkdir(parents=True)
        (tests / "test_cuda.py").touch()
        (examples / "vector_add_test.py").touch()
        (examples / "numba_vector_add_test.py").touch()
        result = self.discover()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["test_cuda.py", "intro/vector_add_test.py"])

    def test_missing_checkout_reports_explicit_setup_step(self):
        result = self.discover()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prepare_cuda_python.sh", result.stderr)
        self.assertFalse(self.checkout.exists())
        self.assertFalse(self.results.exists())

    def test_prepare_installed_version_and_update_with_shallow_fetch(self):
        upstream = self.root / "upstream"
        subprocess.run(["git", "init", "-q", str(upstream)], check=True)
        def git(*args):
            return subprocess.run(["git", "-C", str(upstream), *args], check=True,
                                  capture_output=True, text=True)
        git("config", "user.email", "test@example.com")
        git("config", "user.name", "Test")
        for version in ["13.3.0", "13.3.1"]:
            (upstream / "version").write_text(version)
            git("add", ".")
            git("commit", "-qm", version)
            git("tag", "v" + version)
        python = self.root / "python"
        calls = self.root / "calls"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            f"with open({str(calls)!r}, 'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if sys.argv[1] == '-c': print(os.environ['TEST_VERSION'])\n"
        )
        python.chmod(0o755)
        cuda = self.root / "cuda"
        cuda.mkdir()
        (cuda / "version.json").write_text('{"cuda":{"version":"13.3.1"}}')
        env = dict(self.env, PYTHON_BIN=str(python), CUDA_HOME=str(cuda),
                   CUDA_PYTHON_URL=upstream.as_uri(), CUDA_PYTHON_REF="")
        for version in ["13.3.0", "13.3.1"]:
            env["TEST_VERSION"] = version
            result = subprocess.run(["bash", str(TEST_DIR / "prepare_cuda_python.sh")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((self.checkout / "version").read_text(), version)
            count = subprocess.check_output(["git", "-C", str(self.checkout), "rev-list", "--count", "HEAD"], text=True)
            self.assertEqual(count.strip(), "1")
        installs = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertIn("cuda-bindings==13.*", installs[0])


if __name__ == "__main__":
    unittest.main()
