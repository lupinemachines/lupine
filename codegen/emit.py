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
    InOutCountOperation,
    NullableArrayOperation,
    NullableOperation,
    NullTerminatedOperation,
)


@dataclass(frozen=True)
class Backend:
    name: str
    result: str
    success: str
    invalid_argument: str
    not_supported: str
    symbol_lookup: str
    # Manual client function every generated result passes through, for a
    # backend that keeps a sticky error.
    sticky_error: str = ""
    # Public names the SDK header maps onto versioned ABI symbols.
    remappings: tuple[tuple[str, str], ...] = ()
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
        f.write(f"    return {result_expr(backend, backend.invalid_argument)};\n")
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
    for operation in operations:
        if isinstance(
            operation, (InOutCountOperation, NullableArrayOperation)
        ) or (isinstance(operation, NullTerminatedOperation) and operation.recv):
            f.write(operation.client_declaration())
    # Reject invalid send buffers before the request is framed, so a bad
    # argument returns cleanly instead of desyncing the RPC stream.
    for operation in operations:
        if isinstance(operation, ArrayOperation):
            operation.client_preflight(f, result_expr(backend, backend.invalid_argument))
    f.write(f"  {backend.result} return_value = rpc_error();\n")
    for operation in operations:
        if isinstance(operation, NullTerminatedOperation) and operation.recv:
            continue
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
            owner for owner in metadata.record_owners if owner.kind == "DEVICE"
        ]
        if len(owners) != 1 or not isinstance(owners[0].parameter.type, Pointer):
            raise RuntimeError(f"{name}: ALL routing requires one DEVICE output")
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

    if metadata.routing_kind is None:
        f.write("  conn_t *conn = connection();\n")
    else:
        if metadata.routing_parameter is None:
            raise RuntimeError(
                f"{name}: {metadata.routing_kind} routing requires a parameter"
            )
        # A device ordinal goes by address because the selector rewrites it to
        # the server's; every other kind is a handle passed by value.
        argument = metadata.routing_parameter.name
        if metadata.routing_kind == "DEVICE":
            argument = f"&{argument}"
        f.write(
            f"  conn_t *conn = connection_for_{metadata.routing_kind.lower()}"
            f"({argument});\n"
        )
    suffix = f", {', '.join(call_args)}" if call_args else ""
    call = result_expr(backend, f"lupine_rpc_{name}(conn{suffix})")
    if not metadata.record_owners:
        f.write(f"  return {call};\n")
        f.write("}\n\n")
        return
    f.write(f"  {backend.result} return_value = {call};\n")
    for owner in metadata.record_owners:
        handle = owner.parameter.name
        guard = f" && {handle} != nullptr" if isinstance(owner.parameter.type, Pointer) else ""
        value = f"*{handle}" if isinstance(owner.parameter.type, Pointer) else handle
        f.write(
            f"  if (return_value == {backend.success}{guard}) "
            f"note_{owner.kind.lower()}_owner(conn, {value});\n"
        )
    f.write("  return return_value;\n")
    f.write("}\n\n")


def result_expr(backend: Backend, expression: str) -> str:
    return f"{backend.sticky_error}({expression})" if backend.sticky_error else expression


def unsupported(function, metadata) -> bool:
    """A call the backend does not describe cannot be forwarded.

    The annotation file is the list of what a backend supports. A declaration
    it leaves out, and one it carries without saying what to do with the
    parameters, mean the same thing, and both get a stub.
    """
    if metadata.disabled_client:
        return False
    return bool(function.parameters) and not metadata.operations


def write_stub(f, backend: Backend, function):
    name = function.name.format()
    params = ", ".join(format_function_params(function))
    f.write("// Not annotated for this backend, so not supported.\n")
    f.write(f'extern "C" {function.return_type.format()} {name}({params}) {{\n')
    for parameter in function.parameters:
        if parameter.name:
            f.write(f"  (void){parameter.name};\n")
    f.write(f"  return {result_expr(backend, backend.not_supported)};\n")
    f.write("}\n\n")


def write_guarded(f, metadata, write):
    if metadata.guard is not None:
        f.write(f"#if {metadata.guard}\n")
    write()
    if metadata.guard is not None:
        f.write("#endif\n\n")


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
    f.write(
        f"  fn = {backend.symbol_lookup}<fn_t>(LUPINE_SYMBOL_NAME({name}));\n"
    )
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


def write_backend(backend: Backend, functions_with_annotations, sdk_functions=()):
    """The three files a forwarding backend generates."""
    annotated = {
        function.name.format() for function, _, _, _ in functions_with_annotations
    }
    with open(f"gen_{backend.name}_client.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        client_functions = {}
        for function, _, operations, metadata in functions_with_annotations:
            if unsupported(function, metadata):
                write_guarded(f, metadata, lambda: write_stub(f, backend, function))
                continue
            if metadata.disabled_server:
                continue
            if metadata.disabled_client and not metadata.client_forwards:
                continue
            if not metadata.disabled_client:
                client_functions[function.name.format()] = function

            def write_client():
                write_client_rpc(f, backend, function, operations, metadata)
                write_client_wrapper(f, backend, function, operations, metadata)

            write_guarded(f, metadata, write_client)

        for alias, remapped in backend.remappings:
            target = client_functions.get(remapped)
            if alias in client_functions or target is None:
                continue
            params = ", ".join(format_function_params(target))
            f.write(f"#ifdef {alias}\n#undef {alias}\n#endif\n")
            f.write(f'extern "C" {target.return_type.format()} {alias}({params}) {{\n')
            f.write(f"  return {remapped}({', '.join(format_call_args(target))});\n")
            f.write("}\n\n")

        for function in sdk_functions:
            # A declaration the header defines for itself, as a template or an
            # inline, is not an entry point the library exports.
            if function.template or function.inline or function.has_body:
                continue
            # A stub can only report that it did nothing in the backend's own
            # result type, so a declaration returning anything else is left out.
            if (
                function.name.format() not in annotated
                and function.return_type.format() == backend.result
            ):
                write_stub(f, backend, function)

    with open(f"gen_{backend.name}_server.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        f.write("#define LUPINE_SYMBOL_NAME_(name) #name\n")
        f.write("#define LUPINE_SYMBOL_NAME(name) LUPINE_SYMBOL_NAME_(name)\n\n")
        for function, _, operations, metadata in functions_with_annotations:
            if metadata.disabled_server or unsupported(function, metadata):
                continue
            write_guarded(
                f,
                metadata,
                lambda: write_server_handler(
                    f, backend, function, operations, metadata
                ),
            )

    with open(f"gen_{backend.name}_server.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, _, metadata in functions_with_annotations:
            if metadata.disabled_server or unsupported(function, metadata):
                continue
            write_guarded(
                f,
                metadata,
                lambda: f.write(
                    f"int handle_{function.name.format()}(conn_t *conn);\n"
                ),
            )
