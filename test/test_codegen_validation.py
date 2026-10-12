"""Regression tests for required counts paired with nullable output arrays."""
import io
from pathlib import Path
import sys
import unittest

from cxxheaderparser.simple import parse_string

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
from emit import Backend, write_client_validation
from ops import DereferenceOperation, InOutCountOperation, NullableArrayOperation


class CountValidationTests(unittest.TestCase):
    def setUp(self):
        self.function = parse_string(
            "nvmlReturn_t query(unsigned int *count, unsigned int *values);"
        ).namespace.functions[0]
        self.count, self.values = self.function.parameters
        self.backend = Backend(
            result="nvmlReturn_t", invalid_argument="NVML_ERROR_INVALID_ARGUMENT",
            device_routing_kind="DEVICE", symbol_lookup="unused",
        )

    def render(self, operations):
        stream = io.StringIO()
        write_client_validation(stream, self.backend, self.function, operations)
        return stream.getvalue()

    def test_inout_count_is_required_but_output_array_is_nullable(self):
        operations = [
            InOutCountOperation(send=True, recv=True, parameter=self.count,
                                anchor="values"),
            NullableArrayOperation(parameter=self.values, ptr=self.values.type,
                                   count=self.count),
        ]
        self.assertEqual(self.render(operations),
                         "  if (count == nullptr) {\n"
                         "    return NVML_ERROR_INVALID_ARGUMENT;\n"
                         "  }\n")

    def test_nullable_output_array_is_not_rejected(self):
        operation = NullableArrayOperation(parameter=self.values,
                                           ptr=self.values.type, count=self.count)
        self.assertEqual(self.render([operation]), "")

    def test_existing_dereference_validation_is_preserved(self):
        operation = DereferenceOperation(send=True, recv=True,
                                         parameter=self.count, type_=self.count.type)
        self.assertEqual(self.render([operation]),
                         "  if (count == nullptr) {\n"
                         "    return NVML_ERROR_INVALID_ARGUMENT;\n"
                         "  }\n")


if __name__ == "__main__":
    unittest.main()
