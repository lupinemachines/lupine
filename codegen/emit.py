"""Writers shared by the forwarding backends.

A forwarding backend sends every annotated call to the server and resolves the
vendor entry point there by name. A backend states its own types and codes;
nothing here asks which one it is writing.
"""

from dataclasses import dataclass
import textwrap

from cxxheaderparser.types import Function, FunctionType, Parameter, Pointer

from ops import (
    status_only,
    unanswered,
    ArrayOperation,
    InOutCountOperation,
    NullableArrayOperation,
    NullableOperation,
    NullTerminatedOperation,
    VersionedStructOperation,
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
    # The status a declaration the annotation file says nothing about returns.
    not_supported: str = ""
    # When set, an @async call is submitted fire-and-forget whenever the
    # client's submit_async(conn) says so and returns this status; otherwise
    # it is an ordinary request. The server answers only the requests.
    async_success: str = ""
    # The library also exports every entry point under this prefix (NCCL's
    # profiling interface), as an alias of the same definition.
    alias_prefix: str = ""
    # Status codes for conditional output arrays.
    success: str = ""
    invalid_result: str = ""
    # Calls on no stream wait only for earlier calls on no stream, except
    # REQUEST_ORDER's "all" ones. Unset, every call on no stream waits for
    # everything.
    stream_less_scoped: bool = False
    # Keeps an unanswered call's failure for the next synchronize to report.
    keep_async_error: str = ""


# How a call's request is ordered where its arguments do not say. "all": it
# waits for the device natively, so it waits for every earlier unanswered
# request, not only those on its stream or events. "soft": a query that answers
# CUDA_ERROR_NOT_READY while work it is ordered behind has not been submitted,
# instead of waiting for it. "publish": every later request waits for it, as it
# records events and signals semaphores it does not name.
REQUEST_ORDER = {
    "cuArrayDestroy": "all",
    "cuCtxDestroy_v2": "all",
    "cuCtxDetach": "all",
    "cuCtxDisablePeerAccess": "all",
    "cuCtxEnablePeerAccess": "all",
    "cuCtxResetPersistingL2Cache": "all",
    "cuCtxSetLimit": "all",
    "cuCtxSynchronize": "all",
    "cuCtxSynchronize_v2": "all",
    "cuCtxRecordEvent": "all",
    "cuCtxWaitEvent": "all",
    "cuDestroyExternalMemory": "all",
    "cuDestroyExternalSemaphore": "all",
    "cuDevicePrimaryCtxRelease_v2": "all",
    "cuDevicePrimaryCtxReset_v2": "all",
    "cuGraphicsUnregisterResource": "all",
    "cuGreenCtxDestroy": "all",
    "cuGreenCtxRecordEvent": "all",
    "cuGreenCtxWaitEvent": "all",
    "cuIpcCloseMemHandle": "all",
    "cuMemFreeHost": "all",
    "cuMemHostUnregister": "all",
    "cuMipmappedArrayDestroy": "all",
    "cuModuleUnload": "all",
    "cuStreamBeginCapture_v2": "all",
    # Releasing what queued collectives may still use waits for them.
    "ncclCommAbort": "all",
    "ncclCommDeregister": "all",
    "ncclCommDestroy": "all",
    "ncclCommFinalize": "all",
    "ncclCommWindowDeregister": "all",
    "ncclMemFree": "all",
    "ncclRedOpDestroy": "all",
    "cuGraphLaunch": "publish",
    "cuStreamQuery": "soft",
}

# Hints whose failure only means the hint did not take: an unanswered one drops
# it rather than failing the client's next synchronize.
HINT_FUNCTIONS = {
    "cuCtxSetCacheConfig",
    "cuCtxSetSharedMemConfig",
    "cuFuncSetCacheConfig",
    "cuKernelSetCacheConfig",
    "cuMemAdvise",
    "cuMemAdvise_v2",
    "cuMemPrefetchAsync",
    "cuMemPrefetchAsync_v2",
    "cuStreamAttachMemAsync",
}


def optional_async(backend: "Backend", function, metadata) -> bool:
    """Whether the call skips its answer only when submit_async says so."""
    return bool(backend.async_success) and unanswered(
        function, metadata, backend.result
    )


def may_go_unanswered(backend: "Backend", function, metadata) -> bool:
    """Whether the server must check that its client waits before answering.

    Keyed on the status alone, not @blocking: a client that waits always gets
    its answer, whatever the annotation says.
    """
    return bool(backend.keep_async_error or backend.async_success) and status_only(
        function, metadata, backend.result
    )


def unsupported(function, metadata) -> bool:
    """A declaration with parameters but no @param lines gets a stub."""
    return bool(function.parameters) and not metadata.operations and not (
        metadata.disabled_client or metadata.disabled_server
    )


def write_stub(f, backend: Backend, function):
    name = function.name.format()
    params = ", ".join(format_function_params(function))
    f.write(f'extern "C" {function.return_type.format()} {name}({params}) {{\n')
    for parameter in function.parameters:
        if parameter.name:
            f.write(f"  (void){parameter.name};\n")
    f.write(f"  return {backend.not_supported};\n}}\n\n")


def format_function_params(function: Function) -> list[str]:
    params = []
    for param in function.parameters:
        if param.name and isinstance(param.type, Pointer) and isinstance(
            param.type.ptr_to, FunctionType
        ):
            params.append(param.type.format_decl(param.name))
        elif param.name:
            params.append(
                f"{param.type.format()} {param.name}"
            )
        else:
            params.append(param.type.format())
    return params


def format_call_args(function: Function) -> list[str]:
    return [param.name for param in function.parameters if param.name]


def write_cleared_fields(f, metadata, indent, reference):
    for parameter, fields in metadata.clear_fields:
        for name in fields:
            if reference == "->":
                f.write(
                    f"{indent}if ({parameter} != nullptr)\n"
                    f"{indent}  {parameter}->{name} = nullptr;\n"
                )
            else:
                f.write(f"{indent}{parameter}{reference}{name} = nullptr;\n")


def conditional_output_arrays(backend, function, operations, metadata):
    arrays = [
        op for op in operations
        if isinstance(op, NullableArrayOperation)
        and op.recv_on
    ]
    if arrays and (
        function.return_type.format() != backend.result
        or not backend.success or unanswered(function, metadata, backend.result)
    ):
        raise RuntimeError(
            "Conditional output requires a synchronous status-returning backend "
            "with a success code"
        )
    success_arrays = [op for op in arrays
                      if op.recv_on == "ON_SUCCESS"]
    if success_arrays and not backend.invalid_result:
        raise RuntimeError("ON_SUCCESS requires a backend invalid-result code")
    return success_arrays


def write_client_rpc(f, backend: Backend, function, operations, metadata):
    name = function.name.format()
    result = function.return_type.format()
    params = ", ".join(format_function_params(function))
    conditional_output_arrays(backend, function, operations, metadata)
    submit = optional_async(backend, function, metadata)
    maybe_unused = "[[maybe_unused]] " if metadata.disabled_client else ""
    f.write(f"{maybe_unused}static {result} lupine_rpc_{name}(conn_t *conn")
    if submit:
        f.write(", bool submit_async")
    if params:
        f.write(f", {params}")
    f.write(") {\n")
    if result != "void":
        initial_value = "rpc_error()" if result == backend.result else "{}"
        f.write(f"  {result} return_value = {initial_value};\n")
    for operation in operations:
        f.write(operation.client_declaration())

    opening = "  if (conn == nullptr ||\n      " if backend.guard_null_conn else "  if ("
    start = f"rpc_write_start_request(conn, RPC_{name})"
    if backend.stream_less_scoped and REQUEST_ORDER.get(name) != "all":
        start = f"rpc_write_start_request_on(conn, RPC_{name}, RPC_SCOPE_NONE)"
    # A call on a stream is ordered as CUDA orders that stream's work.
    stream = next(
        (p.name for p in function.parameters if p.type.format() == "cudaStream_t"),
        None,
    )
    if stream is not None:
        start = f"rpc_write_start_request_on_stream(conn, RPC_{name}, {stream})"
    wait, returned = "rpc_wait_for_response(conn) < 0 ||", "return_value"
    if submit:
        # A request sent without waiting gets no answer.
        wait = "(submit_async ? rpc_write_end(conn) : rpc_wait_for_response(conn)) < 0 ||\n      (!submit_async && ("
        returned = f"submit_async ? {backend.async_success} : return_value"
    f.write(f"{opening}{start} < 0 ||\n")
    for operation in operations:
        operation.client_rpc_write(f)
    f.write(f"      {wait}\n")
    for operation in operations:
        operation.client_rpc_read(f)
    if result != "void":
        f.write("      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||\n")
    f.write(f"      rpc_read_end(conn) < 0{')))' if submit else ')'} {{\n")
    write_cleared_fields(f, metadata, "    ", "->")
    if result == backend.result:
        f.write("    return rpc_error();\n")
    elif result == "void":
        f.write("    return;\n")
    else:
        f.write("    return {};\n")
    f.write("  }\n")
    write_cleared_fields(f, metadata, "  ", "->")
    if result != "void":
        f.write(f"  return {returned};\n")
    f.write("}\n\n")


def write_client_wrapper(f, backend: Backend, function, operations, metadata):
    if metadata.disabled_client:
        return

    name = function.name.format()
    result = function.return_type.format()
    params = ", ".join(format_function_params(function))
    f.write(f'extern "C" {result} {name}({params}) {{\n')

    call_args = format_call_args(function)
    if optional_async(backend, function, metadata):
        call_args = ["submit_async(conn)"] + call_args
    suffix = f", {', '.join(call_args)}" if call_args else ""
    call = f"lupine_rpc_{name}(conn{suffix})"
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
        call = (
            f"lookup_device_on_all_connections({output_name},\n"
            "      [&](conn_t *conn, nvmlDevice_t *remote_device) {\n"
            f"        return lupine_rpc_{name}(conn, {', '.join(lambda_args)});\n"
            "      })"
        )
    elif metadata.routing_kind == backend.device_routing_kind:
        if metadata.routing_parameter is None:
            raise RuntimeError(
                f"{name}: {metadata.routing_kind} routing requires a parameter"
            )
        f.write(
            "  conn_t *conn = connection_for_device("
            f"&{metadata.routing_parameter.name});\n"
        )
    elif metadata.routing_kind == "EVENT":
        if metadata.routing_parameter is None:
            raise RuntimeError(f"{name}: EVENT routing requires a parameter")
        f.write(
            "  conn_t *conn = connection_for_event("
            f"{metadata.routing_parameter.name});\n"
        )
    elif metadata.routing_kind == "HANDLE":
        if metadata.routing_parameter is None:
            raise RuntimeError(f"{name}: HANDLE routing requires a parameter")
        f.write(
            "  conn_t *conn = connection_for_handle("
            f"{metadata.routing_parameter.name});\n"
        )
    elif metadata.routing_kind == "STREAM":
        if metadata.routing_parameter is None:
            raise RuntimeError(f"{name}: STREAM routing requires a parameter")
        f.write(
            "  conn_t *conn = connection_for_stream("
            f"{metadata.routing_parameter.name});\n"
        )
    elif metadata.routing_kind is None:
        f.write("  conn_t *conn = connection();\n")
    else:
        raise RuntimeError(f"{name}: unsupported routing key {metadata.routing_kind}")
    template = metadata.client_call_template
    if template is not None and result == "void":
        f.write(textwrap.indent(template.before_call, "  "))
        f.write(f"  {call};\n")
        f.write(textwrap.indent(template.after_call, "  "))
    elif template is not None:
        f.write(textwrap.indent(template.before_call, "  "))
        f.write(f"  {result} return_value = {call};\n")
        f.write(textwrap.indent(template.after_call, "  "))
        f.write("  return return_value;\n")
    else:
        f.write(f"  return {call};\n")
    f.write("}\n\n")


def write_server_handler(f, backend: Backend, function, operations, metadata):
    name = function.name.format()
    result = function.return_type.format()
    success_arrays = conditional_output_arrays(backend, function, operations, metadata)
    # The driver exports this one unversioned.
    entry = "cuEventElapsedTime" if name == "cuEventElapsedTime_v2" else name
    cleared = {parameter for parameter, _ in metadata.clear_fields}
    if metadata.guard is not None:
        f.write(f"#if {metadata.guard}\n")
    f.write(f"int handle_{name}(conn_t *conn) {{\n")
    for operation in operations:
        f.write(operation.server_declaration)
        if (
            isinstance(operation, NullableOperation)
            and operation.recv
            and not operation.send
            and operation.parameter.name in cleared
        ):
            argument = operation.parameter.name
            # Zeroed whole so the cleared fields read as unset, not padding.
            f.write(f"  std::memset(&{argument}, 0, sizeof({argument}));\n")
    unawaited = may_go_unanswered(backend, function, metadata)
    sync = metadata.synchronize
    stdout = sync is not None and sync.stdout
    if sync is not None:
        f.write("  uint64_t delivered = 0;\n")
    if stdout:
        f.write("  lupine_captured_stdout capture;\n")
    if metadata.graph_exec_update is not None:
        f.write("  lupine_htod_graph_binding update_binding;\n")
    f.write("  int request_id;\n")
    if result != "void":
        f.write(f"  {result} return_value;\n")
    if backend.symbol_lookup:
        fn_params = ", ".join(
            parameter.type.format() for parameter in function.parameters
        )
        f.write(f"  using fn_t = {result} (*)({fn_params});\n")
        f.write("  fn_t fn = nullptr;\n")
    f.write("  if (\n")
    for operation in operations:
        operation.server_rpc_read(f)
    f.write("      false)\n")
    f.write("    goto ERROR_0;\n\n")
    f.write("  request_id = rpc_read_end(conn);\n")
    f.write("  if (request_id < 0)\n")
    f.write("    goto ERROR_0;\n\n")

    for operation in operations:
        parameter = operation.parameter
        if (
            parameter.name in ("dependencies", "from")
            and "CUgraphNode" in parameter.type.format()
            and getattr(operation, "send", False)
            and not getattr(operation, "recv", False)
        ):
            # A dependency on a graph device-to-host copy waits for the node
            # that pushes its bytes.
            f.write(
                f"  lupine_graph_push_dependencies({parameter.name}, "
                f"{operation.server_element_count_expr()});\n"
            )
    if metadata.graph_exec_node is not None:
        graph_exec = metadata.graph_exec_node.graph_exec.name
        node = metadata.graph_exec_node.node.name
        f.write(f"  {node} = lupine_htod_graph_exec_node({graph_exec}, {node});\n")

    if stdout:
        f.write("  lupine_start_stdout_capture(&capture);\n")
    update = metadata.graph_exec_update
    call_args = []
    for parameter in function.parameters:
        operation = next(
            (op for op in operations if op.parameter.name == parameter.name), None
        )
        if operation is None:
            raise RuntimeError(
                f"{name}: parameter {parameter.name} has no @param annotation"
            )
        if update is not None and parameter.name == update.graph.name:
            call_args.append("update_binding.prepared")
        else:
            call_args.append(operation.server_reference)
        if update is not None and parameter.name == update.error.name:
            update_error = operation.server_reference
    args = ", ".join(call_args)
    if update is not None:
        f.write(
            f"  return_value = lupine_prepare_graph_exec_update("
            f"{update.graph_exec.name}, {update.graph.name}, &update_binding);\n"
        )
        f.write("  if (return_value == CUDA_SUCCESS)\n  ")
    if backend.symbol_lookup:
        f.write(
            f'  {{ static const fn_t cached = {backend.symbol_lookup}<fn_t>("{name}"); fn = cached; }}\n'
        )
        if result == backend.result:
            f.write(
                "  return_value = fn == nullptr ? function_not_found()\n"
                f"                               : fn({args});\n\n"
            )
        else:
            f.write("  if (fn == nullptr)\n")
            f.write("    goto ERROR_0;\n")
            if result == "void":
                f.write(f"  fn({args});\n\n")
            else:
                f.write(f"  return_value = fn({args});\n\n")
    elif result == "void":
        f.write(f"  {entry}({args});\n\n")
    elif REQUEST_ORDER.get(name) == "soft":
        f.write(
            "  return_value = lupine_stream_query_unsettled(conn, hStream)\n"
            "                     ? CUDA_ERROR_NOT_READY\n"
            f"                     : {entry}({args});\n\n"
        )
    else:
        f.write(f"  return_value = {entry}({args});\n\n")
    if unawaited:
        # A client that does not wait gets no answer; a failure reaches it from
        # its next synchronize instead, except for a hint's.
        keeps_error = backend.keep_async_error and name not in HINT_FUNCTIONS
        if keeps_error:
            f.write("  if (lupine_unanswered(conn, return_value))\n    return 0;\n\n")
        else:
            f.write("  if (!rpc_request_awaited(conn))\n    return 0;\n\n")

    if stdout:
        f.write("  lupine_finish_stdout_capture(&capture);\n")
    if update is not None:
        f.write(
            f"  lupine_finish_graph_exec_update({update.graph_exec.name}, "
            f"{update.graph.name}, &update_binding, return_value, "
            f"{update_error});\n\n"
        )
    if sync is not None:
        # Device-to-host copies the call waited for have numbered their pushed
        # delivery by now; the client applies them before it returns.
        f.write("  delivered = lupine_dtoh_watermark(conn);\n")
        f.write("  return_value = lupine_take_async_error(conn, return_value);\n\n")
    if metadata.clear_fields:
        write_cleared_fields(f, metadata, "  ", ".")
        f.write("\n")
    # On success a populated output must fit the caller's capacity. Keep a
    # count-only query valid even when its reported count exceeds capacity 0.
    invalid_counts = [
        f"(!{op.parameter.name}_null && {op.count.name} > {op.requested_count_expr()})"
        for op in success_arrays if isinstance(op.count.type, Pointer)
    ]
    if invalid_counts:
        f.write(f"  if (return_value == {backend.success} && (\n      "
                + " ||\n      ".join(invalid_counts) + ")) {\n")
        f.write(f"    return_value = {backend.invalid_result};\n  }}\n")
    f.write("  if (rpc_write_start_response(conn, request_id) < 0 ||\n")
    if sync is not None:
        f.write("      rpc_write(conn, &delivered, sizeof(delivered)) < 0 ||\n")
    if stdout:
        f.write("      lupine_write_captured_stdout(conn, capture) < 0 ||\n")
    for operation in operations:
        if isinstance(operation, NullableArrayOperation):
            operation.server_rpc_write(f, success=backend.success)
        else:
            operation.server_rpc_write(f)
    if result != "void":
        f.write(f"      rpc_write(conn, &return_value, sizeof({result})) < 0 ||\n")
    f.write("      rpc_write_end(conn) < 0)\n")
    f.write("    goto ERROR_0;\n")
    f.write("  return 0;\n")
    f.write("ERROR_0:\n")
    f.write("  return -1;\n")
    f.write("}\n\n")
    if metadata.guard is not None:
        f.write("#endif\n\n")


def write_versioned_struct_helper(f, functions_with_annotations):
    """Widens a caller's configuration struct and points its strings at copies."""
    if not any(
        isinstance(operation, VersionedStructOperation)
        for _, _, operations, _ in functions_with_annotations
        for operation in operations
    ):
        return
    f.write(
        "template <typename T>\n"
        "static T *lupine_versioned_struct(\n"
        "    unsigned char *bytes, uint64_t size,\n"
        "    std::initializer_list<std::pair<size_t, const char *>> strings,\n"
        "    std::initializer_list<size_t> cleared) {\n"
        "  if (size == 0 || bytes == nullptr) {\n"
        "    return nullptr;\n"
        "  }\n"
        "  for (const auto &string : strings) {\n"
        "    if (string.first + sizeof(char *) <= size) {\n"
        "      std::memcpy(bytes + string.first, &string.second,\n"
        "                  sizeof(char *));\n"
        "    }\n"
        "  }\n"
        "  const void *null = nullptr;\n"
        "  for (size_t offset : cleared) {\n"
        "    if (offset + sizeof(void *) <= size) {\n"
        "      std::memcpy(bytes + offset, &null, sizeof(void *));\n"
        "    }\n"
        "  }\n"
        "  return reinterpret_cast<T *>(bytes);\n"
        "}\n\n"
    )


def write_profiling_aliases(f, backend: Backend, functions_with_annotations):
    """Exports each public entry point again under the backend's alias prefix."""
    if not backend.alias_prefix:
        return
    for function, _, _, metadata in functions_with_annotations:
        name = function.name.format()
        if name.startswith("lupine"):
            continue
        if metadata.guard is not None:
            f.write(f"#if {metadata.guard}\n")
        params = ", ".join(format_function_params(function))
        f.write(
            f'extern "C" {function.return_type.format()} {backend.alias_prefix}{name}({params})\n'
            f'    __attribute__((alias("{name}")));\n'
        )
        if metadata.guard is not None:
            f.write("#endif\n")
