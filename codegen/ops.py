"""Wire-marshalling operations and annotation metadata for codegen.py.

Each Operation models how one annotated parameter is serialized across the RPC
boundary (what the client writes/reads and how the server reads/calls/writes).
codegen.py parses the @param/@deeparray annotations into these objects and asks
each one to emit its fragment of the generated client and server code.
"""

from cxxheaderparser.types import Type, Pointer, Parameter, Array
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class NullableOperation:
    """
    Nullable operations are operations that are passed as a pointer that can be null.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer

    def client_rpc_write(self, f):
        if not (self.send or self.recv):
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        if not self.send:
            return

        f.write(
            "        ({param_name} != nullptr && rpc_write(conn, {param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                # void is treated differently from non void pointer types
                base_type=(
                    self.ptr.format()
                    if self.ptr.ptr_to.format() == "const void"
                    else self.ptr.ptr_to.format()
                ),
            )
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, cudaMemcpyDeviceToHost) < 0)\n".format(
                name=self.parameter.name
            )
        )
        f.write("      return {error};\n".format(error=error))

    @property
    def server_declaration(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        # void is treated differently from non void pointer types
        s = (
            f"    {self.ptr.format()} {self.parameter.name}_null_check;\n"
            + f"""    {self.ptr.format() if self.ptr.ptr_to.format() == "void" else self.ptr.ptr_to.format()} {self.parameter.name};\n"""
        )
        self.ptr.ptr_to.const = c
        return s

    def server_rpc_read(self, f):
        if not (self.send or self.recv):
            return
        f.write(
            "        rpc_read(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        if not self.send:
            return
        f.write(
            "        ({param_name}_null_check && rpc_read(conn, &{param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                # void is treated differently from non void pointer types
                base_type=(
                    self.ptr.format()
                    if self.ptr.ptr_to.format() == "const void"
                    else self.ptr.ptr_to.format()
                ),
            )
        )

    @property
    def server_reference(self) -> str:
        return f"{self.parameter.name}_null_check ? &{self.parameter.name} : nullptr"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "        ({param_name}_null_check && rpc_write(conn, &{param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                base_type=self.ptr.ptr_to.format(),
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "        ({param_name}_null_check && rpc_read(conn, {param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                base_type=self.ptr.ptr_to.format(),
            )
        )


@dataclass
class ArrayOperation:
    """
    Array operations are operations that are passed as a pointer to an array.
    """

    send: bool
    recv: bool
    iter: bool
    parameter: Parameter
    ptr: Pointer
    # if int, it's a constant length, if Parameter, it's a variable length.
    length: Union[int, Parameter]
    # compressible payloads use the rpc_*_payload helpers which optionally
    # apply LZ4 framing for large transfers (see compress.cpp).
    compressible: bool = False

    @property
    def is_void_bytes(self) -> bool:
        return self.ptr.ptr_to.format() in ("void", "const void")

    def byte_count_expr(self) -> str:
        if isinstance(self.length, int):
            return str(self.length)
        if isinstance(self.length.type, Pointer):
            return f"*{self.length.name}"
        return self.length.name

    def element_count_expr(self) -> str:
        if isinstance(self.length, int):
            return str(self.length)
        if isinstance(self.length.type, Pointer):
            return f"*{self.length.name}"
        return self.length.name

    def transfer_size_expr(self) -> str:
        if self.is_void_bytes:
            return self.byte_count_expr()
        return f"{self.element_count_expr()} * sizeof({self.ptr.ptr_to.format()})"

    def mutable_ptr_format(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.format()
        self.ptr.ptr_to.const = c
        return result

    def client_rpc_write(self, f):
        if self.iter:
            loop_template = """
                [=]() -> bool {{
                    for (size_t i = 0; i < {length}; ++i) {{
                        if (rpc_write(0, &{param_name}[i], sizeof({param_type})) < 0) {{
                            printf("Failed to write Dependency[%zu]\\n", i);
                            return false;
                        }}
                    }}
                    return true;
                }}() == false ||
                """.strip()

            f.write(loop_template.format(
                length=self.length.name,
                param_type=self.ptr.ptr_to.format(),
                param_name=self.parameter.name,
            ))
            return

        if not self.send:
            return
        if isinstance(self.length, int):
            f.write(
                "        ({size} != 0 && rpc_write(conn, {param_name}, {size}) < 0) ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        # array length operations are handled differently than char
        elif isinstance(self.ptr, Array):
            f.write(
                "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.array_of.format(),
                )
            )
        else:
            f.write(
                "        ({size} != 0 && {param_name} == nullptr) ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )
            f.write(
                "        ({size} != 0 && {write_fn}(conn, {param_name}, {size}) < 0) ||\n".format(
                    write_fn="rpc_write_payload" if self.compressible else "rpc_write",
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )

    def client_prepare_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        (lupine_prepare_host_range_write({param_name}, {size}), false) ||\n".format(
                param_name=self.parameter.name,
                size=self.transfer_size_expr(),
            )
        )

    def client_post_rpc_read_success(self, f):
        if not self.recv:
            return
        f.write(
            "    if (return_value == CUDA_SUCCESS) lupine_mark_host_range_clean({param_name}, {size});\n".format(
                param_name=self.parameter.name,
                size=self.transfer_size_expr(),
            )
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                name=self.parameter.name, direction=direction
            )
        )
        f.write("      return {error};\n".format(error=error))

        if isinstance(self.length, int):
            f.write(
                "    for (int i = 0; i < {name} && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                    param=self.parameter.name, name=self.length
                )
            )
            f.write(
                "      if (maybe_copy_unified_arg(conn, (void*)&{name}[i], {direction}) < 0 )\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("        return {error};\n".format(error=error))

            return

        if hasattr(self.length.type, "ptr_to"):
            # need to cast the int a bit differently here
            f.write(
                "    for (int i = 0; i < static_cast<int>(*{name}) && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                    param=self.parameter.name, name=self.length.name
                )
            )
            f.write(
                "      if (maybe_copy_unified_arg(conn, (void*)&{name}[i], {direction}) < 0)\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("        return {error};\n".format(error=error))
        else:
            if hasattr(self.parameter.type, "ptr_to"):
                f.write(
                    "    for (int i = 0; i < static_cast<int>({name}) && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                        param=self.parameter.name, name=self.length.name
                    )
                )
                f.write(
                    "      if (maybe_copy_unified_arg(conn, (void*)&{name}[i], {direction}) < 0)\n".format(
                        name=self.parameter.name, direction=direction
                    )
                )
                f.write("        return {error};\n".format(error=error))
            else:
                f.write(
                    "    for (int i = 0; i < static_cast<int>({name}) && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                        param=self.parameter.name, name=self.length.name
                    )
                )
                f.write(
                    "      if (maybe_copy_unified_arg(conn, (void*){name}[i], {direction}) < 0)\n".format(
                        name=self.parameter.name, direction=direction
                    )
                )
                f.write("        return {error};\n".format(error=error))

    @property
    def server_declaration(self) -> str:
        if isinstance(self.ptr, Array):
            c = self.ptr.array_of.const
            self.ptr.array_of.const = False
            s = f"    {self.ptr.array_of.format()}* {self.parameter.name} = nullptr;\n"
            self.ptr.array_of.const = c
        elif self.iter:
            c = self.ptr.ptr_to.const
            self.ptr.ptr_to.const = False
            s = f"    std::vector<{self.ptr.ptr_to.format()}> {self.parameter.name};\n"
            self.ptr.ptr_to.const = c
        else:
            c = self.ptr.ptr_to.const
            self.ptr.ptr_to.const = False
            s = f"    {self.ptr.format()} {self.parameter.name};\n"
            if self.send:
                s += f"    size_t {self.parameter.name}_size;\n"
            self.ptr.ptr_to.const = c
        return s

    def server_rpc_read(self, f, index) -> Optional[str]:
        if self.iter:
            lambda_template = """
            [=, &{param_name}]() -> bool {{
                {param_name}.resize({length});  // Resize the dependencies vector
                for (size_t i = 0; i < {length}; ++i) {{
                    if (rpc_read(conn, &{param_name}[i], sizeof({param_type})) < 0) {{
                        return false;
                    }}
                }}
                return true;
            }}() == false ||
            """.strip()

            f.write(lambda_template.format(
                param_name=self.parameter.name,
                length=self.length.name,
                param_type=self.ptr.ptr_to.format(),
            ))
            return

        if not self.send:
            # if this parameter is recv only and it's a type pointer, it needs to be malloc'd.
            if isinstance(self.ptr, Pointer):
                f.write("        false)\n")
                f.write("        goto ERROR_{index};\n".format(index=index))
                f.write(
                    "    {param_name} = ({server_type})malloc({size});\n".format(
                        param_name=self.parameter.name,
                        server_type=self.ptr.format(),
                        size=self.transfer_size_expr(),
                    )
                )
                f.write("    if(")
                return self.parameter.name
            return
        elif isinstance(self.ptr, Pointer):
            f.write("        false)\n")
            f.write("        goto ERROR_{index};\n".format(index=index))
            f.write(
                "    {param_name}_size = {size};\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )
            f.write(
                "    {param_name} = ({server_type})malloc({size});\n".format(
                    param_name=self.parameter.name,
                    server_type=self.mutable_ptr_format(),
                    size=f"{self.parameter.name}_size",
                )
            )
            f.write(
                "    if ({param_name}_size != 0 && {param_name} == nullptr)\n".format(
                    param_name=self.parameter.name
                )
            )
            f.write("        goto ERROR_{index};\n".format(index=index))
            f.write("    if(\n")
            f.write(
                "        ({size} != 0 && {read_fn}(conn, {param_name}, {size}) < 0) ||\n".format(
                    read_fn="rpc_read_payload" if self.compressible else "rpc_read",
                    param_name=self.parameter.name,
                    size=f"{self.parameter.name}_size",
                )
            )
            defer = self.parameter.name
        elif isinstance(self.length, int):
            f.write(
                "        rpc_read(conn, &{param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        elif isinstance(self.ptr, Array):
            f.write(
                "        rpc_read(conn, &{param_name}, sizeof({param_type}*)) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.array_of.format(),
                )
            )
        else:
            f.write(
                "        rpc_read(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )
        if 'defer' in locals():
            return defer

    @property
    def server_reference(self) -> str:
        if self.iter:
            return f"{self.parameter.name}.data()"
        if isinstance(self.length, int):
            return f"{self.parameter.name}"
        return f"({self.transfer_size_expr()} == 0 ? nullptr : {self.parameter.name})"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        if isinstance(self.length, int):
            f.write(
                "        ({size} != 0 && rpc_write(conn, {param_name}, {size}) < 0) ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        else:
            f.write(
                "        ({size} != 0 && {write_fn}(conn, {param_name}, {size}) < 0) ||\n".format(
                    write_fn="rpc_write_payload" if self.compressible else "rpc_write",
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        if isinstance(self.length, int):
            f.write(
                "        ({size} != 0 && rpc_read(conn, {param_name}, {size}) < 0) ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        else:
            f.write(
                "        ({size} != 0 && {read_fn}(conn, {param_name}, {size}) < 0) ||\n".format(
                    read_fn="rpc_read_payload" if self.compressible else "rpc_read",
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )


@dataclass
class InOutCountOperation:
    """
    A ``size_t *`` count that is simultaneously an input capacity and an output
    count for one or more :class:`OptionalArrayOperation` out-arrays -- the
    cuGraphGetNodes pattern. The client sends the requested capacity (0 when the
    anchor array is null, which is a count-only query); the server runs the API
    once with that capacity and returns the actual count.
    """

    send: bool
    recv: bool
    parameter: Parameter
    # array param whose presence decides query-vs-fill on the client side.
    anchor: str

    @property
    def server_declaration(self) -> str:
        return (
            f"    size_t {self.parameter.name} = 0;\n"
            f"    size_t {self.parameter.name}_requested = 0;\n"
        )

    def client_declaration(self) -> str:
        return (
            f"    size_t {self.parameter.name}_requested =\n"
            f"        ({self.anchor} != nullptr) ? *{self.parameter.name} : 0;\n"
        )

    def client_rpc_write(self, f):
        f.write(
            f"        rpc_write(conn, &{self.parameter.name}_requested, sizeof(size_t)) < 0 ||\n"
        )

    def client_rpc_read(self, f):
        f.write(
            f"        rpc_read(conn, {self.parameter.name}, sizeof(size_t)) < 0 ||\n"
        )

    def server_rpc_read(self, f):
        f.write(
            f"        rpc_read(conn, &{self.parameter.name}_requested, sizeof(size_t)) < 0 ||\n"
        )
        # Seed the count with the requested capacity so a non-null buffer is
        # filled (a null buffer makes the API ignore it and report the total).
        f.write(
            f"        (({self.parameter.name} = {self.parameter.name}_requested), false) ||\n"
        )

    @property
    def server_reference(self) -> str:
        return f"&{self.parameter.name}"

    def server_rpc_write(self, f):
        f.write(
            f"        rpc_write(conn, &{self.parameter.name}, sizeof(size_t)) < 0 ||\n"
        )


@dataclass
class OptionalArrayOperation:
    """
    An optional out-array sized by an in/out :class:`InOutCountOperation`. The
    array may be null (the caller is querying the count, or does not want this
    particular array). Several optional arrays can share one count, e.g.
    cuGraphGetEdges' from/to/edgeData.
    """

    parameter: Parameter
    ptr: Pointer
    count: Parameter

    def element_type(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.ptr_to.format()
        self.ptr.ptr_to.const = c
        return result

    @property
    def server_declaration(self) -> str:
        return (
            f"    {self.element_type()} *{self.parameter.name} = nullptr;\n"
            f"    uint8_t {self.parameter.name}_present = 0;\n"
        )

    def client_declaration(self) -> str:
        return (
            f"    uint8_t {self.parameter.name}_present = "
            f"{self.parameter.name} != nullptr ? 1 : 0;\n"
        )

    def client_rpc_write(self, f):
        f.write(
            f"        rpc_write(conn, &{self.parameter.name}_present, sizeof(uint8_t)) < 0 ||\n"
        )

    def server_rpc_read(self, f, index) -> Optional[str]:
        elem = self.element_type()
        name = self.parameter.name
        count = self.count.name
        f.write(
            f"        rpc_read(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
        )
        f.write("        false)\n")
        f.write(f"        goto ERROR_{index};\n")
        f.write(f"    if ({name}_present && {count}_requested != 0) {{\n")
        f.write(
            f"        {name} = ({elem} *)malloc({count}_requested * sizeof({elem}));\n"
        )
        f.write(f"        if ({name} == nullptr)\n")
        f.write(f"            goto ERROR_{index};\n")
        f.write("    }\n")
        f.write("    if (\n")
        return name

    @property
    def server_reference(self) -> str:
        return self.parameter.name

    def server_rpc_write(self, f):
        elem = self.element_type()
        name = self.parameter.name
        count = self.count.name
        f.write(
            f"        ({name}_present && {count} != 0 && "
            f"rpc_write(conn, {name}, {count} * sizeof({elem})) < 0) ||\n"
        )

    def client_rpc_read(self, f):
        elem = self.element_type()
        name = self.parameter.name
        count = self.count.name
        f.write(
            f"        ({name} != nullptr && *{count} != 0 && "
            f"rpc_read(conn, {name}, *{count} * sizeof({elem})) < 0) ||\n"
        )


@dataclass
class DeepStructOperation:
    """
    A pointer to a struct that embeds array pointers sized by sibling count
    members, declared in the function doxygen with:

        @deeparray <param> <array_member> <count_member>

    The struct header is copied by value, then each embedded array is walked
    using sizeof(*member) / decltype(member). No element types or struct names
    are needed here -- the generated text is type-checked by the C++ compiler
    (a shallow copy would instead send dangling client pointers).

    SEND is an input deep-copy (cuGraphAdd* / *SetParams). RECV fills the
    caller's struct from node-owned memory (*GetParams); the array storage lives
    in a per-output-pointer client cache (lupine_deep_cache_*) so the returned
    pointers stay valid until the next deep query into the same struct.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer
    members: list  # list of (array_member, count_member)

    def struct_type(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.ptr_to.format()
        self.ptr.ptr_to.const = c
        return result

    def client_declaration(self) -> str:
        # Reject a null struct pointer before the request is framed so we return
        # cleanly instead of desyncing the RPC stream.
        return (
            f"    if ({self.parameter.name} == nullptr) "
            "return CUDA_ERROR_INVALID_VALUE;\n"
        )

    @property
    def server_declaration(self) -> str:
        s = f"    {self.struct_type()} {self.parameter.name} = {{}};\n"
        if self.send:
            for member, _count in self.members:
                s += (
                    "    std::vector<unsigned char> "
                    f"{self.parameter.name}_{member}_buf;\n"
                )
        return s

    def server_rpc_read(self, f):
        if not self.send:
            return
        name = self.parameter.name
        f.write(f"        rpc_read(conn, &{name}, sizeof({name})) < 0 ||\n")
        for member, count in self.members:
            buf = f"{name}_{member}_buf"
            f.write(
                f"        (({buf}.resize({name}.{count} * sizeof(*{name}.{member})),"
                " false)) ||\n"
            )
            f.write(
                f"        ({name}.{count} != 0 && "
                f"rpc_read(conn, {buf}.data(), {buf}.size()) < 0) ||\n"
            )
            f.write(
                f"        (({name}.{member} = (decltype({name}.{member})){buf}.data()),"
                " false) ||\n"
            )

    @property
    def server_reference(self) -> str:
        return f"&{self.parameter.name}"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        f.write(f"        rpc_write(conn, &{name}, sizeof({name})) < 0 ||\n")
        for member, count in self.members:
            f.write(
                f"        ({name}.{count} != 0 && "
                f"rpc_write(conn, {name}.{member}, "
                f"{name}.{count} * sizeof(*{name}.{member})) < 0) ||\n"
            )

    def client_rpc_write(self, f):
        if not self.send:
            return
        name = self.parameter.name
        f.write(f"        rpc_write(conn, {name}, sizeof(*{name})) < 0 ||\n")
        for member, count in self.members:
            f.write(
                f"        ({name}->{count} != 0 && "
                f"rpc_write(conn, {name}->{member}, "
                f"{name}->{count} * sizeof(*{name}->{member})) < 0) ||\n"
            )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        f.write(
            f"        (lupine_deep_cache_reset((const void *){name}), false) ||\n"
        )
        f.write(f"        rpc_read(conn, {name}, sizeof(*{name})) < 0 ||\n")
        for member, count in self.members:
            esz = f"{name}->{count} * sizeof(*{name}->{member})"
            f.write(
                f"        (({name}->{member} = ({name}->{count} != 0 ? "
                f"(decltype({name}->{member}))"
                f"lupine_deep_cache_add((const void *){name}, {esz}) : nullptr)),"
                " false) ||\n"
            )
            f.write(
                f"        ({name}->{count} != 0 && {name}->{member} == nullptr) ||\n"
            )
            f.write(
                f"        ({name}->{count} != 0 && "
                f"rpc_read(conn, (void *){name}->{member}, {esz}) < 0) ||\n"
            )


@dataclass
class NullTerminatedOperation:
    """
    Null terminated operations are operations that are passed as a null terminated string.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer

    def client_rpc_write(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_write(conn, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )
        f.write(
            "        rpc_write(conn, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )

    @property
    def server_declaration(self) -> str:
        return (
            f"    {self.ptr.format()} {self.parameter.name};\n"
            + f"    std::size_t {self.parameter.name}_len;\n"
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                name=self.parameter.name, direction=direction
            )
        )
        f.write("      return {error};\n".format(error=error))

    def server_rpc_read(self, f, index) -> Optional[str]:
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_len, sizeof(std::size_t)) < 0)\n".format(
                param_name=self.parameter.name
            )
        )
        f.write("        goto ERROR_{index};\n".format(index=index))
        f.write(
            "    {param_name} = ({server_type})malloc({param_name}_len);\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "    if (rpc_read(conn, (void *){param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )
        return self.parameter.name

    @property
    def server_reference(self) -> str:
        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )
        f.write(
            "        rpc_write(conn, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )
        f.write(
            "        rpc_read(conn, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )


@dataclass
class OpaqueTypeOperation:
    """
    Opaque type operations are operations that are passed as an opaque type. That is, the
    data is written directly without any additional dereferencing.
    """

    send: bool
    recv: bool
    parameter: Parameter
    type_: Union[Type, Pointer]

    def is_sent_cufunction(self) -> bool:
        type_name = self.type_.format().replace("const ", "").strip()
        return self.send and type_name == "CUfunction"

    def client_declaration(self) -> str:
        if self.is_sent_cufunction():
            return (
                f"    CUfunction {self.parameter.name}_rpc = "
                f"lupine_translate_private_function_for_rpc({self.parameter.name});\n"
            )
        return ""

    def client_rpc_write(self, f):
        if not self.send:
            return
        else:
            param_name = (
                f"{self.parameter.name}_rpc"
                if self.is_sent_cufunction()
                else self.parameter.name
            )
            f.write(
                "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                    param_name=param_name,
                    param_type=self.type_.format(),
                )
            )

    @property
    def server_declaration(self) -> str:
        if isinstance(self.type_, Pointer) and self.recv:
            return f"    {self.type_.ptr_to.format()} {self.parameter.name};\n"
        # ensure we don't have a const struct, otherwise we can't initialise it properly; ex: "const cudnnTensorDescriptor_t xDesc;" is invalid...
        # but "const cudnnTensorDescriptor_t *xDesc" IS valid. This subtle change carries reprecussions.
        elif (
            "const " in self.type_.format()
            and "void" not in self.type_.format()
            and "*" not in self.type_.format()
        ):
            return f"   {self.type_.format().replace('const', '')} {self.parameter.name};\n"
        else:
            return f"    {self.type_.format()} {self.parameter.name};\n"

    def client_unified_copy(self, f, direction, error):
        if isinstance(self.type_, Pointer):
            f.write(
                "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("      return {error};\n".format(error=error))
        else:
            f.write(
                "    if (maybe_copy_unified_arg(conn, (void*)&{name}, {direction}) < 0)\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("      return {error};\n".format(error=error))

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.format(),
            )
        )

    @property
    def server_reference(self) -> str:
        if self.recv:
            return f"&{self.parameter.name}"
        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.format(),
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.format(),
            )
        )


@dataclass
class DereferenceOperation:
    """
    Opaque type operations are operations that are passed as an opaque type. That is, the
    data is written directly without any additional dereferencing.
    """

    send: bool
    recv: bool
    parameter: Parameter
    type_: Pointer

    def client_rpc_write(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_write(conn, {param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    @property
    def server_declaration(self) -> str:
        c = self.type_.ptr_to.const
        self.type_.ptr_to.const = False
        result = f"    {self.type_.ptr_to.format()} {self.parameter.name};\n"
        self.type_.ptr_to.const = c
        return result

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                name=self.parameter.name, direction=direction
            )
        )
        f.write("      return {error};\n".format(error=error))

    @property
    def server_reference(self) -> str:
        return f"&{self.parameter.name}"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        # if this parameter is recv only then dereference it.
        f.write(
            "        rpc_read(conn, {param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )


Operation = Union[
    NullableOperation,
    ArrayOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    DereferenceOperation,
]


@dataclass
class OwnerAnnotation:
    kind: str
    parameter: Parameter


@dataclass
class CrossServerCopyAnnotation:
    dst: Parameter
    src: Parameter
    bytes: Parameter
    stream: Optional[Parameter] = None
    async_: bool = False


@dataclass
class FunctionAnnotationMetadata:
    operations: list[Operation]
    disabled_client: bool = False
    disabled_server: bool = False
    routing_kind: Optional[str] = None
    routing_parameter: Optional[Parameter] = None
    record_owners: list[OwnerAnnotation] = None
    cross_server_copy: Optional[CrossServerCopyAnnotation] = None

    def __post_init__(self):
        if self.record_owners is None:
            self.record_owners = []
