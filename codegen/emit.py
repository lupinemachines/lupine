"""Writers shared by the forwarding backends.

A forwarding backend sends every annotated call to the server and resolves the
vendor entry point there by name. A backend states its own types and codes;
nothing here asks which one it is writing.
"""

from dataclasses import dataclass

from cxxheaderparser.types import Function, Parameter, Pointer

from ops import (
    ArrayOperation,
    DereferenceOperation,
    NullableOperation,
    NullTerminatedOperation,
)


@dataclass(frozen=True)
class Backend:
    result: str
    invalid_argument: str
    device_routing_kind: str
    symbol_lookup: str
    # A route may have no server behind it.
    guard_null_conn: bool = False
    # A device named by handle rather than by ordinal has to be looked for on
    # every connection.
    lookup_on_all_connections: bool = False


def format_function_params(function: Function) -> list[str]:
    params = []
    for param in function.parameters:
        if param.name and "[]" in param.type.format():
            params.append(
                "{type} {name}".format(
                    type=param.type.format().replace("[]", ""),
                    name=param.name + "[]",
                )
            )
        elif param.name:
            params.append(
                f"{param.type.format()} {param.name}"
            )
        else:
            params.append(param.type.format())
    return params


def format_call_args(function: Function) -> list[str]:
    return [param.name for param in function.parameters if param.name]


def signed_length_arrays(operations) -> list[ArrayOperation]:
    # A signed element count is read before the buffer it sizes is allocated.
    # Rejecting negatives keeps signed multiplication from becoming a huge
    # allocation or wire size.
    return [
        operation
        for operation in operations
        if isinstance(operation, ArrayOperation)
        and isinstance(operation.length, Parameter)
        and operation.length.type.format() == "int"
    ]


def write_client_validation(f, backend: Backend, operations):
    checks = [
        f"{operation.length.name} < 0"
        for operation in signed_length_arrays(operations)
    ]
    for operation in operations:
        name = operation.parameter.name
        if isinstance(operation, NullTerminatedOperation) and operation.send:
            checks.append(f"{name} == nullptr")
        elif isinstance(operation, DereferenceOperation):
            checks.append(f"{name} == nullptr")
        elif isinstance(operation, ArrayOperation):
            checks.append(
                f"({operation.transfer_size_expr()} != 0 && {name} == nullptr)"
            )
    if checks:
        f.write("  if (" + " ||\n      ".join(checks) + ") {\n")
        f.write(f"    return {backend.invalid_argument};\n")
        f.write("  }\n")


def write_cleared_fields(f, metadata, indent, reference):
    for parameter, fields in metadata.clear_fields:
        for name in fields:
            f.write(f"{indent}{parameter}{reference}{name} = nullptr;\n")


def write_client_rpc(f, backend: Backend, function, operations, metadata):
    name = function.name.format()
    params = ", ".join(format_function_params(function))
    f.write(f"static {backend.result} lupine_rpc_{name}(conn_t *conn")
    if params:
        f.write(f", {params}")
    f.write(") {\n")
    f.write(f"  {backend.result} return_value = rpc_error();\n")
    for operation in operations:
        if isinstance(operation, NullTerminatedOperation):
            f.write(
                f"  {operation.length_type} {operation.parameter.name}_len = static_cast<{operation.length_type}>(std::strlen({operation.parameter.name}) + 1);\n"
            )
        elif isinstance(operation, NullableOperation) and operation.recv:
            f.write(
                f"  {operation.ptr.format()} {operation.parameter.name}_null_check = nullptr;\n"
            )

    opening = "  if (conn == nullptr ||\n      " if backend.guard_null_conn else "  if ("
    f.write(f"{opening}rpc_write_start_request(conn, RPC_{name}) < 0 ||\n")
    for operation in operations:
        operation.client_rpc_write(f)
    f.write("      rpc_wait_for_response(conn) < 0 ||\n")
    for operation in operations:
        operation.client_rpc_read(f)
    f.write("      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||\n")
    f.write("      rpc_read_end(conn) < 0) {\n")
    write_cleared_fields(f, metadata, "    ", "->")
    f.write("    return rpc_error();\n")
    f.write("  }\n")
    write_cleared_fields(f, metadata, "  ", "->")
    f.write("  return return_value;\n")
    f.write("}\n\n")


def write_client_wrapper(f, backend: Backend, function, operations, metadata):
    if metadata.disabled_client:
        return

    name = function.name.format()
    params = ", ".join(format_function_params(function))
    f.write(f'extern "C" {backend.result} {name}({params}) {{\n')
    write_client_validation(f, backend, operations)

    call_args = format_call_args(function)
    if metadata.routing_kind == "ALL" and backend.lookup_on_all_connections:
        owners = [
            owner
            for owner in metadata.record_owners
            if owner.kind == backend.device_routing_kind
        ]
        if len(owners) != 1 or not isinstance(owners[0].parameter.type, Pointer):
            raise RuntimeError(
                f"{name}: ALL routing requires one "
                f"{backend.device_routing_kind} output"
            )
        output_name = owners[0].parameter.name
        lambda_args = [
            "remote_device" if arg == output_name else arg for arg in call_args
        ]
        f.write(
            f"  return lookup_device_on_all_connections({output_name},\n"
            "      [&](conn_t *conn, nvmlDevice_t *remote_device) {\n"
            f"        return lupine_rpc_{name}(conn, {', '.join(lambda_args)});\n"
            "      });\n"
        )
        f.write("}\n\n")
        return

    if metadata.routing_kind == backend.device_routing_kind:
        if metadata.routing_parameter is None:
            raise RuntimeError(
                f"{name}: {metadata.routing_kind} routing requires a parameter"
            )
        f.write(
            "  conn_t *conn = connection_for_device("
            f"&{metadata.routing_parameter.name});\n"
        )
    elif metadata.routing_kind is None:
        f.write("  conn_t *conn = connection();\n")
    else:
        raise RuntimeError(f"{name}: unsupported routing key {metadata.routing_kind}")
    suffix = f", {', '.join(call_args)}" if call_args else ""
    f.write(f"  return lupine_rpc_{name}(conn{suffix});\n")
    f.write("}\n\n")


def write_server_buffer_cleanup(f, owned_buffers, indent):
    for buffer_name in reversed(owned_buffers):
        f.write(f"{indent}free((void *){buffer_name});\n")


def write_server_handler(f, backend: Backend, function, operations, metadata):
    name = function.name.format()
    fn_params = ", ".join(
        parameter.type.format() for parameter in function.parameters
    )
    cleared = {parameter for parameter, _ in metadata.clear_fields}
    f.write(f"int handle_{name}(conn_t *conn) {{\n")
    owned_buffers = []
    for operation in operations:
        f.write(operation.server_declaration)
        if (
            isinstance(operation, DereferenceOperation)
            and operation.recv
            and not operation.send
        ):
            argument = operation.parameter.name
            # Zeroed whole so the cleared fields read as unset, not padding.
            if argument in cleared:
                f.write(f"  std::memset(&{argument}, 0, sizeof({argument}));\n")
            else:
                f.write(f"  {argument} = {{}};\n")
    f.write("  int request_id;\n")
    f.write(f"  {backend.result} return_value;\n")
    f.write(f"  using fn_t = {backend.result} (*)({fn_params});\n")
    f.write("  fn_t fn = nullptr;\n")
    f.write("  if (\n")
    signed_lengths = signed_length_arrays(operations)
    for operation in operations:
        if operation in signed_lengths:
            f.write(f"      {operation.length.name} < 0 ||\n")
        if owned_buffer := operation.server_rpc_read(f):
            owned_buffers.append(owned_buffer)
    f.write("      false)\n")
    f.write("    goto ERROR_0;\n\n")
    f.write("  request_id = rpc_read_end(conn);\n")
    f.write("  if (request_id < 0)\n")
    f.write("    goto ERROR_0;\n\n")

    call_args = []
    for parameter in function.parameters:
        operation = next(
            op for op in operations if op.parameter.name == parameter.name
        )
        call_args.append(operation.server_reference)
    f.write(f'  fn = {backend.symbol_lookup}<fn_t>("{name}");\n')
    f.write(
        "  return_value = fn == nullptr ? function_not_found()\n"
        f"                               : fn({', '.join(call_args)});\n\n"
    )
    if metadata.clear_fields:
        write_cleared_fields(f, metadata, "  ", ".")
        f.write("\n")
    f.write("  if (rpc_write_start_response(conn, request_id) < 0 ||\n")
    for operation in operations:
        operation.server_rpc_write(f)
    f.write("      rpc_write(conn, &return_value, sizeof(return_value)) < 0 ||\n")
    f.write("      rpc_write_end(conn) < 0)\n")
    f.write("    goto ERROR_0;\n")
    write_server_buffer_cleanup(f, owned_buffers, "  ")
    f.write("  return 0;\n")
    f.write("ERROR_0:\n")
    write_server_buffer_cleanup(f, owned_buffers, "  ")
    f.write("  return -1;\n")
    f.write("}\n\n")
