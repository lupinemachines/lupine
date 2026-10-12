"""Codegen contracts for explicitly success-only nullable output arrays."""
import dataclasses
import io
from pathlib import Path
import sys
import unittest

from cxxheaderparser.simple import parse_string

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
import codegen as gen
from emit import write_client_rpc, write_server_handler
from ops import InOutCountOperation, NullableArrayOperation


class SuccessArrayTests(unittest.TestCase):
    def fixture(self, modifiers="ON_SUCCESS", count_type="unsigned int *",
                result="nvmlReturn_t", second_array=False):
        function = parse_string(
            f"{result} query({count_type} count, unsigned int *values"
            + (", unsigned int *other" if second_array else "") + ");"
        ).namespace.functions[0]
        metadata = gen.parse_annotation(
            "@param count SEND_RECV\n"
            f"@param values RECV_ONLY NULLABLE LENGTH:count {modifiers}\n"
            + ("@param other RECV_ONLY NULLABLE LENGTH:count ON_SUCCESS\n"
               if second_array else ""), function.parameters)
        return function, metadata

    def render(self, writer, modifiers="ON_SUCCESS", backend=None, **kwargs):
        function, metadata = self.fixture(modifiers, **kwargs)
        stream = io.StringIO()
        writer(stream, backend or gen.NVML, function, metadata.operations, metadata)
        return stream.getvalue()

    def test_parser_preserves_count_and_success_only_array(self):
        _, metadata = self.fixture()
        self.assertIsInstance(metadata.operations[0], InOutCountOperation)
        self.assertTrue(metadata.operations[1].recv_on_success)

    def test_error_and_success_modifiers_are_exclusive(self):
        with self.assertRaises(NotImplementedError):
            self.fixture("ON_SUCCESS ON_ERROR")

    def test_conflicting_modifiers_are_rejected(self):
        for modifier in ("DEREF", "SCALAR", "VERSIONED", "SIZE:4", "NULL_TERMINATED"):
            with self.subTest(modifier=modifier), self.assertRaises(NotImplementedError):
                self.fixture("ON_SUCCESS " + modifier)

    def test_client_reads_status_before_count(self):
        output = self.render(write_client_rpc)
        self.assertLess(output.index("rpc_read(conn, &return_value"),
                        output.index("rpc_read(conn, count,"))
        self.assertEqual(output.count("rpc_read(conn, &return_value"), 1)
        self.assertIn("return_value != NVML_SUCCESS || values == nullptr", output)

    def test_server_writes_status_before_count(self):
        output = self.render(write_server_handler)
        self.assertLess(output.index("rpc_write(conn, &return_value"),
                        output.index("rpc_write(conn, &count,"))
        self.assertEqual(output.count("rpc_write(conn, &return_value"), 1)
        self.assertIn("return_value == NVML_SUCCESS && !values_null", output)

    def test_server_rejects_oversized_success_before_response(self):
        output = self.render(write_server_handler)
        self.assertIn("!values_null && count > count_requested", output)
        self.assertLess(output.index("return_value = NVML_ERROR_UNKNOWN"),
                        output.index("rpc_write_start_response"))

    def test_client_bounds_check_precedes_copy_and_closes_bad_response(self):
        output = self.render(write_client_rpc)
        self.assertIn("*count > count_requested", output)
        self.assertIn("*count > size_t(-1) / sizeof(unsigned int)", output)
        self.assertLess(output.index("rpc_close_transport_socket(conn)"),
                        output.index("rpc_read(conn, values,"))

    def test_fixed_capacity_arrays_are_supported(self):
        client = self.render(write_client_rpc, count_type="unsigned int")
        server = self.render(write_server_handler, count_type="unsigned int")
        self.assertIn("rpc_read(conn, values, count * sizeof(unsigned int))", client)
        self.assertIn("return_value == NVML_SUCCESS && !values_null", server)
        self.assertNotIn("count_requested", server)

    def test_shared_count_arrays_validate_all_outputs(self):
        output = self.render(write_server_handler, second_array=True)
        self.assertIn("!values_null && count > count_requested", output)
        self.assertIn("!other_null && count > count_requested", output)
        self.assertEqual(output.count("return_value = NVML_ERROR_UNKNOWN"), 1)

    def test_default_wire_order_is_unchanged(self):
        client = self.render(write_client_rpc, modifiers="")
        server = self.render(write_server_handler, modifiers="")
        self.assertLess(client.index("rpc_read(conn, values,"),
                        client.index("rpc_read(conn, &return_value"))
        self.assertLess(server.index("rpc_write(conn, values,"),
                        server.index("rpc_write(conn, &return_value"))
        self.assertNotIn("return_value = NVML_ERROR_UNKNOWN", server)

    def test_backend_must_define_result_codes(self):
        backend = dataclasses.replace(gen.NVML, success="", invalid_result="")
        for writer in (write_client_rpc, write_server_handler):
            with self.subTest(writer=writer.__name__), self.assertRaises(RuntimeError):
                self.render(writer, backend=backend)

    def test_wrong_result_type_is_rejected(self):
        for writer in (write_client_rpc, write_server_handler):
            with self.subTest(writer=writer.__name__), self.assertRaises(RuntimeError):
                self.render(writer, result="int")

    def test_async_calls_are_rejected(self):
        function, metadata = self.fixture()
        metadata.async_fire_forget = True
        for writer in (write_client_rpc, write_server_handler):
            with self.subTest(writer=writer.__name__), self.assertRaises(RuntimeError):
                writer(io.StringIO(), gen.NVML, function, metadata.operations, metadata)

    def test_non_status_aware_emitters_cannot_silently_ignore_modifier(self):
        _, metadata = self.fixture()
        operation = next(op for op in metadata.operations
                         if isinstance(op, NullableArrayOperation))
        for method in (operation.client_rpc_read, operation.server_rpc_write):
            with self.subTest(method=method.__name__), self.assertRaises(RuntimeError):
                method(io.StringIO())


if __name__ == "__main__":
    unittest.main()
