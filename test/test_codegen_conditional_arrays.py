# /// script
# requires-python = ">=3.10"
# dependencies = ["clang-format==18.1.3", "cxxheaderparser==1.9.2"]
# ///
"""Generator policy tests; run with uv run test/test_codegen_conditional_arrays.py."""
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from cxxheaderparser.simple import parse_string

ROOT = Path(__file__).resolve().parents[1]
SDK_INCLUDE = os.environ.get("CUDA_INCLUDE_DIR", "/usr/local/cuda/include")
sys.path.insert(0, str(ROOT / "codegen"))
from codegen import NVML, parse_annotation
from emit import Backend, write_client_rpc, write_client_wrapper, write_server_handler
from ops import ArrayReceivePolicy, NullableArrayOperation


def parse(modifiers, *, count="unsigned int *"):
    source = f"""
/**
 * @param device SEND_ONLY
 * @param count {'SEND_RECV' if '*' in count else 'SEND_ONLY'}
 * @param samples {modifiers}
 */
nvmlReturn_t nvmlConditionalError(nvmlDevice_t device, {count} count, nvmlProcessInfo_v1_t *samples);
"""
    function = parse_string(source).namespace.functions[0]
    return function, parse_annotation(function.doxygen, function.parameters)


class ConditionalArrayTests(unittest.TestCase):
    def test_policy_selection(self):
        for modifier, policy in (("", ArrayReceivePolicy.ALWAYS),
                                 ("ON_SUCCESS", ArrayReceivePolicy.ON_SUCCESS),
                                 ("ON_ERROR", ArrayReceivePolicy.ON_ERROR)):
            with self.subTest(modifier=modifier):
                _, metadata = parse("RECV_ONLY NULLABLE LENGTH:count " + modifier)
                array = next(op for op in metadata.operations if isinstance(op, NullableArrayOperation))
                self.assertEqual(array.receive_policy, policy)

    def test_invalid_combinations(self):
        for modifier in ("ON_SUCCESS", "ON_ERROR"):
            for transfer in ("SEND_ONLY NULLABLE LENGTH:count", "RECV_ONLY LENGTH:count",
                             "RECV_ONLY NULLABLE", "RECV_ONLY NULLABLE LENGTH:count DEREF",
                             "RECV_ONLY NULLABLE LENGTH:count SCALAR:device",
                             "RECV_ONLY NULLABLE LENGTH:count VERSIONED",
                             "RECV_ONLY NULLABLE LENGTH:count SIZE:2",
                             "RECV_ONLY NULLABLE LENGTH:count NULL_TERMINATED"):
                with self.subTest(modifier=modifier, transfer=transfer):
                    with self.assertRaises(NotImplementedError):
                        parse(transfer + " " + modifier)
        with self.assertRaises(NotImplementedError):
            parse("RECV_ONLY NULLABLE LENGTH:count ON_SUCCESS ON_ERROR")

    def test_conditional_outputs_require_synchronous_status_backend(self):
        for policy in ("ON_SUCCESS", "ON_ERROR"):
            function, metadata = parse("RECV_ONLY NULLABLE LENGTH:count " + policy)
            backend = Backend(result="nvmlReturn_t", invalid_argument="INVALID",
                              device_routing_kind="", symbol_lookup="")
            with self.assertRaises(RuntimeError):
                write_client_rpc(io.StringIO(), backend, function, metadata.operations, metadata)
            metadata.async_fire_forget = True
            with self.assertRaises(RuntimeError):
                write_server_handler(io.StringIO(), NVML, function, metadata.operations, metadata)

    def test_only_success_policy_needs_invalid_result_code(self):
        backend = replace(NVML, invalid_result="")
        for policy in ("ON_SUCCESS", "ON_ERROR"):
            function, metadata = parse("RECV_ONLY NULLABLE LENGTH:count " + policy)
            if policy == "ON_SUCCESS":
                with self.assertRaises(RuntimeError):
                    write_client_rpc(io.StringIO(), backend, function, metadata.operations, metadata)
            else:
                write_client_rpc(io.StringIO(), backend, function, metadata.operations, metadata)

    def test_checked_in_nvml_rpcs(self):
        subprocess.run([sys.executable, str(ROOT / "test/test_nvml_conditional_arrays.py"),
                        SDK_INCLUDE], check=True)

    def test_generated_on_error_client_and_server(self):
        for count_type in ("unsigned int", "unsigned int *"):
            with self.subTest(count_type=count_type):
                self.exercise_generated_on_error(count_type)

    def exercise_generated_on_error(self, count_type):
        # ON_ERROR previously only had a generated client and manual CUDA server.
        # Exercise fixed capacities and pointer counts using both generated halves.
        function, metadata = parse("RECV_ONLY NULLABLE LENGTH:count ON_ERROR", count=count_type)
        generated = io.StringIO()
        write_server_handler(generated, NVML, function, metadata.operations, metadata)
        write_client_rpc(generated, NVML, function, metadata.operations, metadata)
        write_client_wrapper(generated, NVML, function, metadata.operations, metadata)
        mock = (ROOT / "test/test_nvml_conditional_arrays.cpp").read_text().split('#include "server.inc"')[0]
        mock = mock[:mock.index('template <typename Fn> Fn nvml_symbol')]
        count_pointer = "count" if "*" in count_type else "&count"
        fixture = f"""
nvmlReturn_t mock_error(nvmlDevice_t device, {count_type} count, nvmlProcessInfo_v1_t *samples) {{
  return mock_processes(device, {count_pointer}, samples);
}}
template <typename Fn> Fn nvml_symbol(const char *) {{ return mock_error; }}
#define RPC_nvmlConditionalError 1
"""
        exercise = r'''
int rpc_wait_for_response(conn_t *conn) {
  conn->server = true; conn->cursor = 0;
  int result = handle_nvmlConditionalError(conn);
  conn->server = false; conn->cursor = 0;
  return result;
}
int main() {
  nvmlProcessInfo_v1_t samples[3];
  auto call = [](unsigned int count, nvmlProcessInfo_v1_t *samples) {
    return nvmlConditionalError(nullptr, COUNT_ARGUMENT, samples);
  };
  auto unchanged = [&] {
    for (auto byte : std::vector<unsigned char>(reinterpret_cast<unsigned char *>(samples),
         reinterpret_cast<unsigned char *>(samples) + sizeof(samples))) assert(byte == 0xa5);
  };
  std::memset(samples, 0xa5, sizeof(samples));
  assert(call(3, samples) == NVML_SUCCESS); unchanged();
  native_status = NVML_ERROR_UNKNOWN;
  assert(call(3, samples) == NVML_ERROR_UNKNOWN);
  assert(samples[0].pid == 123 && samples[1].pid == 124);
  std::memset(samples, 0xa5, sizeof(samples));
  native_count = 5;
  assert(call(1, samples) == NVML_ERROR_UNKNOWN);
  assert(samples[0].pid == 123 && samples[1].pid == 0xa5a5a5a5U);
  std::memset(samples, 0xa5, sizeof(samples));
  assert(call(0, samples) == NVML_ERROR_UNKNOWN); unchanged();
  assert(call(3, nullptr) == NVML_ERROR_UNKNOWN);
}
'''
        exercise = exercise.replace("COUNT_ARGUMENT", "&count" if "*" in count_type else "count")
        # Only mocks used in this fixture are required; unused globals are fine.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.cpp"
            executable = Path(directory) / "test"
            source.write_text(mock + fixture + generated.getvalue() + exercise)
            subprocess.run(["c++", "-std=c++17", "-I", SDK_INCLUDE, "-I", str(ROOT),
                            str(source), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
