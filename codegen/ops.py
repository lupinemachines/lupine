"""Wire-marshalling operations and annotation metadata for codegen.py.

Each Operation models how one annotated parameter is serialized across the RPC
boundary (what the client writes/reads and how the server reads/calls/writes).
codegen.py parses the @param/@deeparray annotations into these objects and asks
each one to emit its fragment of the generated client and server code.
"""

from cxxheaderparser.types import Type, Pointer, Parameter
from typing import Optional, Union
from dataclasses import dataclass, field

@dataclass
class NullableOperation:
    """
    A pointer to one value that may be null; a presence byte leads it on the
    wire. An out-value travels in with the caller's contents too, so a call
    that fails without writing it leaves the caller's variable as it was.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer
    nullable: bool = True

    @property
    def base_type(self) -> str:
        # void is treated differently from non void pointer types
        return (
            self.ptr.format()
            if self.ptr.ptr_to.format() == "const void"
            else self.ptr.ptr_to.format()
        )

    def client_declaration(self) -> str:
        name = self.parameter.name
        if not self.nullable:
            return f"    if ({name} == nullptr) return CUDA_ERROR_INVALID_VALUE;\n"
        return f"    uint8_t {name}_present = {name} != nullptr;\n"

    def client_rpc_write(self, f):
        name = self.parameter.name
        if not self.nullable:
            if self.send:
                f.write(
                    f"        rpc_write(conn, {name}, sizeof({self.ptr.ptr_to.format()})) < 0 ||\n"
                )
            return
        f.write(
            f"        rpc_write(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
            f"        ({name}_present && rpc_write(conn, {name}, sizeof({self.base_type})) < 0) ||\n"
        )

    @property
    def server_declaration(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        # void is treated differently from non void pointer types
        decl_type = (
            self.ptr.format()
            if self.ptr.ptr_to.format() == "void"
            else self.ptr.ptr_to.format()
        )
        if not self.nullable:
            s = f"    {decl_type} {self.parameter.name}{{}};\n"
        else:
            s = (
                f"    uint8_t {self.parameter.name}_present = 0;\n"
                f"    {decl_type} {self.parameter.name}{{}};\n"
            )
        self.ptr.ptr_to.const = c
        return s

    def server_rpc_read(self, f):
        name = self.parameter.name
        if not self.nullable:
            if self.send:
                f.write(
                    f"        rpc_read(conn, &{name}, sizeof({self.ptr.ptr_to.format()})) < 0 ||\n"
                )
            return
        f.write(
            f"        rpc_read(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
            f"        ({name}_present && rpc_read(conn, &{name}, sizeof({self.base_type})) < 0) ||\n"
        )

    @property
    def server_reference(self) -> str:
        if not self.nullable:
            return f"&{self.parameter.name}"
        return f"{self.parameter.name}_present ? &{self.parameter.name} : nullptr"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        if not self.nullable:
            f.write(
                f"        rpc_write(conn, &{self.parameter.name}, sizeof({self.ptr.ptr_to.format()})) < 0 ||\n"
            )
            return
        f.write(
            f"        ({self.parameter.name}_present && rpc_write(conn, &{self.parameter.name}, sizeof({self.ptr.ptr_to.format()})) < 0) ||\n"
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        if not self.nullable:
            f.write(
                f"        rpc_read(conn, {self.parameter.name}, sizeof({self.ptr.ptr_to.format()})) < 0 ||\n"
            )
            return
        f.write(
            f"        ({self.parameter.name}_present && rpc_read(conn, {self.parameter.name}, sizeof({self.ptr.ptr_to.format()})) < 0) ||\n"
        )


@dataclass
class ArrayOperation:
    """
    Array operations are operations that are passed as a pointer to an array;
    a null-flag byte leads the array on the wire.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer
    # if int, it's a constant length, if Parameter, it's a variable length
    length: Union[int, Parameter]
    nullable: bool = True
    length_offset: int = 0

    @property
    def is_void_bytes(self) -> bool:
        return self.ptr.ptr_to.format() in ("void", "const void")

    @property
    def has_signed_length(self) -> bool:
        if isinstance(self.length, int):
            return False
        t = (
            self.length.type.ptr_to
            if isinstance(self.length.type, Pointer)
            else self.length.type
        )
        fmt = t.format()
        return (
            "unsigned" not in fmt
            and fmt != "size_t"
            and not fmt.startswith(("uint", "cuuint"))
        )

    def _base_count_expr(self) -> str:
        if isinstance(self.length, int):
            return str(self.length)
        if isinstance(self.length.type, Pointer):
            return f"({self.length.name} != nullptr ? *{self.length.name} : 0)"
        return self.length.name

    def count_expr(self) -> str:
        base = self._base_count_expr()
        if self.length_offset < 0:
            return f"({base} - {-self.length_offset})"
        if self.length_offset > 0:
            return f"({base} + {self.length_offset})"
        return base

    def transfer_size_expr(self) -> str:
        # SIZE:<bytes> is a byte count whatever the pointee type.
        if self.is_void_bytes or isinstance(self.length, int):
            raw = self.count_expr()
        else:
            raw = f"{self.count_expr()} * sizeof({self.ptr.ptr_to.format()})"
        if self.length_offset < 0:
            return f"({self._base_count_expr()} <= {-self.length_offset} ? 0 : {raw})"
        if self.has_signed_length:
            return f"({self.count_expr()} <= 0 ? 0 : {raw})"
        return raw

    def server_element_count_expr(self) -> str:
        if isinstance(self.length, int):
            return str(self.length)
        # Pointer length parameters are unmarshalled into scalar server locals.
        if self.length_offset < 0:
            return f"({self.length.name} - {-self.length_offset})"
        if self.length_offset > 0:
            return f"({self.length.name} + {self.length_offset})"
        return self.length.name

    def server_transfer_size_expr(self) -> str:
        if self.is_void_bytes or isinstance(self.length, int):
            raw = self.server_element_count_expr()
        else:
            raw = (
                f"{self.server_element_count_expr()} * "
                f"sizeof({self.ptr.ptr_to.format()})"
            )
        if self.length_offset < 0:
            return f"({self.length.name} <= {-self.length_offset} ? 0 : {raw})"
        if self.has_signed_length:
            return f"({self.server_element_count_expr()} <= 0 ? 0 : {raw})"
        return raw

    def mutable_ptr_format(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.format()
        self.ptr.ptr_to.const = c
        return result

    def client_declaration(self) -> str:
        name = self.parameter.name
        if not self.nullable:
            size = self.transfer_size_expr()
            return f"    if ({size} != 0 && {name} == nullptr) return CUDA_ERROR_INVALID_VALUE;\n"
        return f"    uint8_t {name}_null = {name} == nullptr ? 1 : 0;\n"

    def client_rpc_write(self, f):
        name = self.parameter.name
        size = self.transfer_size_expr()
        if not self.nullable:
            if self.send:
                f.write(
                    f"        ({size} != 0 && rpc_write(conn, {name}, {size}) < 0) ||\n"
                )
            return
        f.write(
            f"        rpc_write(conn, &{name}_null, sizeof(uint8_t)) < 0 ||\n"
        )
        if not self.send:
            return
        f.write(
            f"        (!{name}_null && {size} != 0 && "
            f"rpc_write(conn, {name}, {size}) < 0) ||\n"
        )

    @property
    def server_declaration(self) -> str:
        name = self.parameter.name
        s = f"    {self.mutable_ptr_format()} {name} = nullptr;\n"
        if self.nullable:
            s += f"    uint8_t {name}_null = 0;\n"
        return s

    def server_rpc_read(self, f) -> Optional[str]:
        name = self.parameter.name
        size = self.server_transfer_size_expr()
        null_read = (
            f"        rpc_read(conn, &{name}_null, sizeof(uint8_t)) < 0 ||\n"
            if self.nullable
            else ""
        )
        cond = f"!{name}_null" if self.nullable else "true"
        f.write(
            f"{null_read}"
            "        false)\n"
            "        goto ERROR_0;\n"
            f"    if ({cond}) {{\n"
            f"        {name} = ({self.mutable_ptr_format()})calloc({size} == 0 ? 1 : {size}, 1);\n"
            f"        if ({name} == nullptr)\n"
            "            goto ERROR_0;\n"
            "    }\n"
            "    if(\n"
        )
        if self.send:
            guard = f"!{name}_null && " if self.nullable else ""
            f.write(
                f"        ({guard}{size} != 0 && "
                f"rpc_read(conn, {name}, {size}) < 0) ||\n"
            )
        return name

    @property
    def server_reference(self) -> str:
        if (
            not self.nullable
            and not isinstance(self.length, int)
            and not (not self.send and self.ptr.ptr_to.format() == "char")
        ):
            return (
                f"({self.server_transfer_size_expr()} == 0 ? "
                f"nullptr : {self.parameter.name})"
            )
        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        size = self.server_transfer_size_expr()
        guard = f"!{name}_null && " if self.nullable else ""
        f.write(
            f"        ({guard}{size} != 0 && "
            f"rpc_write(conn, {name}, {size}) < 0) ||\n"
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        size = self.transfer_size_expr()
        guard = f"!{name}_null && " if self.nullable else ""
        f.write(
            f"        ({guard}{size} != 0 && "
            f"rpc_read(conn, {name}, {size}) < 0) ||\n"
        )


@dataclass
class InOutCountOperation:
    """
    An integer pointer that is simultaneously an input capacity and an output
    count for one or more :class:`NullableArrayOperation` out-arrays -- the
    cuGraphGetNodes and cuDevSmResourceSplitByCount patterns. The client sends
    the requested capacity (0 when the anchor array is null, which is a
    count-only query); the server runs the API once with that capacity and
    returns the actual count.
    """

    send: bool
    recv: bool
    parameter: Parameter
    # array param whose presence decides query-vs-fill on the client side.
    anchor: str

    def count_type(self) -> str:
        return self.parameter.type.ptr_to.format()

    @property
    def server_declaration(self) -> str:
        name = self.parameter.name
        return (
            f"    uint8_t {name}_present = 0;\n"
            f"    {self.count_type()} {name} = 0;\n"
            f"    {self.count_type()} {name}_requested = 0;\n"
        )

    def client_declaration(self) -> str:
        name = self.parameter.name
        return (
            f"    uint8_t {name}_present = {name} != nullptr;\n"
            f"    {self.count_type()} {name}_requested =\n"
            f"        ({name}_present && {self.anchor} != nullptr) ? *{name} : 0;\n"
        )

    def client_rpc_write(self, f):
        name = self.parameter.name
        f.write(
            f"        rpc_write(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
            f"        rpc_write(conn, &{name}_requested, "
            f"sizeof({self.count_type()})) < 0 ||\n"
        )

    def client_rpc_read(self, f):
        name = self.parameter.name
        f.write(
            f"        ({name}_present && rpc_read(conn, {name}, "
            f"sizeof({self.count_type()})) < 0) ||\n"
        )

    def server_rpc_read(self, f):
        name = self.parameter.name
        f.write(
            f"        rpc_read(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
            f"        rpc_read(conn, &{name}_requested, "
            f"sizeof({self.count_type()})) < 0 ||\n"
        )
        # Seed the count with the requested capacity so a non-null buffer is
        # filled (a null buffer makes the API ignore it and report the total).
        f.write(
            f"        (({name} = {name}_requested), false) ||\n"
        )

    @property
    def server_reference(self) -> str:
        name = self.parameter.name
        return f"{name}_present ? &{name} : nullptr"

    def server_rpc_write(self, f):
        name = self.parameter.name
        f.write(
            f"        ({name}_present && rpc_write(conn, &{name}, "
            f"sizeof({self.count_type()})) < 0) ||\n"
        )


@dataclass
class NullableArrayOperation:
    """
    A ``NULLABLE LENGTH:<count>`` out-array. Pointer counts are promoted to an
    :class:`InOutCountOperation`; value counts are fixed capacities. The array
    may be null (the caller is querying the count, or does not want this
    particular array). Several nullable arrays can share one count, e.g.
    cuGraphGetEdges' from/to/edgeData.
    """

    parameter: Parameter
    ptr: Pointer
    count: Parameter
    recv_on: Optional[str] = None

    def element_type(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.ptr_to.format()
        self.ptr.ptr_to.const = c
        return result

    def requested_count_expr(self) -> str:
        if isinstance(self.count.type, Pointer):
            return f"{self.count.name}_requested"
        return self.count.name

    @property
    def server_declaration(self) -> str:
        name = self.parameter.name
        declaration = (
            f"    {self.element_type()} *{name} = nullptr;\n"
            f"    uint8_t {name}_null = 0;\n"
        )
        if self.recv_on:
            declaration += f"    uint8_t {name}_has_data = 0;\n"
        return declaration

    def client_declaration(self) -> str:
        return (
            f"    uint8_t {self.parameter.name}_null = "
            f"{self.parameter.name} == nullptr ? 1 : 0;\n"
        )

    def client_rpc_write(self, f):
        f.write(
            f"        rpc_write(conn, &{self.parameter.name}_null, sizeof(uint8_t)) < 0 ||\n"
        )

    def server_rpc_read(self, f) -> Optional[str]:
        elem = self.element_type()
        name = self.parameter.name
        requested = self.requested_count_expr()
        f.write(
            f"        rpc_read(conn, &{name}_null, sizeof(uint8_t)) < 0 ||\n"
        )
        f.write("        false)\n")
        f.write("        goto ERROR_0;\n")
        # A present pointer must reach CUDA non-null even at zero capacity:
        # null selects count-query semantics, which can report a count larger
        # than the zero-length storage the response would then be read from.
        f.write(f"    if (!{name}_null) {{\n")
        # The whole capacity goes back on the wire whether or not the library
        # filled it, so the allocation supplies the untouched remainder.
        allocation = (
            f"calloc(({requested} != 0 ? {requested} : 1), sizeof({elem}))"
        )
        f.write(
            f"        {name} = ({elem} *){allocation};\n"
        )
        f.write(f"        if ({name} == nullptr)\n")
        f.write("            goto ERROR_0;\n")
        f.write("    }\n")
        f.write("    if (\n")
        return name

    @property
    def server_reference(self) -> str:
        return self.parameter.name

    def transfer_count_expr(self, *, client=False):
        requested = self.requested_count_expr()
        if not isinstance(self.count.type, Pointer):
            return requested
        returned = (
            f"({self.count.name}_present ? *{self.count.name} : 0)"
            if client
            else self.count.name
        )
        if self.recv_on == "ON_SUCCESS":
            return returned
        return f"({returned} < {requested} ? {returned} : {requested})"

    def server_rpc_write(self, f, *, success: str = ""):
        name = self.parameter.name
        count = self.transfer_count_expr()
        present = f"!{name}_null"
        if self.recv_on:
            if not success:
                raise RuntimeError("Conditional output requires a backend success code")
            comparison = "==" if self.recv_on == "ON_SUCCESS" else "!="
            f.write(
                f"        (({name}_has_data = !{name}_null && "
                f"return_value {comparison} {success}), false) ||\n"
                f"        rpc_write(conn, &{name}_has_data, sizeof(uint8_t)) < 0 ||\n"
            )
            present = f"{name}_has_data && {count} != 0"
        f.write(
            f"        ({present} && rpc_write(conn, {name}, "
            f"{count} * sizeof({self.element_type()})) < 0) ||\n"
        )

    def client_rpc_read(self, f):
        elem = self.element_type()
        name = self.parameter.name
        count = self.transfer_count_expr(client=True)
        requested = self.requested_count_expr()
        if self.recv_on:
            invalid = [f"{name}_has_data != 1", f"{name} == nullptr"]
            if self.recv_on == "ON_SUCCESS":
                invalid.append(f"{count} > {requested}")
            invalid.append(f"{count} > size_t(-1) / sizeof({elem})")
            f.write(f"""        ([&]() {{
          uint8_t {name}_has_data = 0;
          if (rpc_read(conn, &{name}_has_data, sizeof(uint8_t)) < 0)
            return true;
          if ({name}_has_data == 0)
            return false;
          if ({' || '.join(invalid)}) {{
            rpc_close_transport_socket(conn);
            return true;
          }}
          return {count} != 0 &&
                 rpc_read(conn, {name}, {count} * sizeof({elem})) < 0;
        }}()) ||
""")
            return
        present = f"{name} != nullptr && {requested} != 0"
        if isinstance(self.count.type, Pointer):
            present += f" && {self.count.name}_present && *{self.count.name} != 0"
        f.write(
            f"        ({present} && rpc_read(conn, {name}, "
            f"{count} * sizeof({elem})) < 0) ||\n"
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
    in a per-node client cache until the node's parameters change or the node
    is destroyed. Routing metadata identifies the owner; no extra annotation
    is needed. SEND_RECV returns struct fields while preserving caller-owned
    input arrays; these are not written back.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer
    members: list  # list of (array_member, count_member)
    node_owner: Optional[str] = None

    def struct_type(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.ptr_to.format()
        self.ptr.ptr_to.const = c
        return result

    def client_declaration(self) -> str:
        name = self.parameter.name
        err = (
            "hipErrorInvalidValue"
            if self.struct_type().startswith("hip")
            else "CUDA_ERROR_INVALID_VALUE"
        )
        code = f"    if ({name} == nullptr) return {err};\n"
        if self.send:
            for member, count in self.members:
                code += (
                    f"    if ({name}->{count} > SIZE_MAX / sizeof(*{name}->{member}) ||\n"
                    f"        ({name}->{count} != 0 && {name}->{member} == nullptr))\n"
                    f"        return {err};\n"
                )
                if self.recv:
                    code += (
                        f"    auto {name}_{member}_input = {name}->{member};\n"
                    )
        return code

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
        if self.send:
            return
        for member, count in self.members:
            f.write(
                f"        rpc_write(conn, {name}.{member}, "
                f"{name}.{count} * sizeof(*{name}.{member})) < 0 ||\n"
            )

    def client_rpc_write(self, f):
        if not self.send:
            return
        name = self.parameter.name
        f.write(f"        rpc_write(conn, {name}, sizeof(*{name})) < 0 ||\n")
        for member, count in self.members:
            f.write(
                f"        rpc_write(conn, {name}->{member}, "
                f"{name}->{count} * sizeof(*{name}->{member})) < 0 ||\n"
            )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        f.write(f"        rpc_read(conn, {name}, sizeof(*{name})) < 0 ||\n")
        for index, (member, count) in enumerate(self.members):
            esz = f"{name}->{count} * sizeof(*{name}->{member})"
            if self.send:
                # SEND_RECV arrays belong to the caller; the returned header
                # contains server scratch pointers that must never escape.
                f.write(
                    f"        (({name}->{member} = {name}_{member}_input), false) ||\n"
                )
                continue
            storage = (
                f"(decltype({name}->{member}))lupine_deep_node_cache_get("
                f"{self.node_owner}, {index}, {esz})"
            )
            f.write(
                f"        (({name}->{member} = ({name}->{count} != 0 ? "
                f"{storage} : nullptr)), false) ||\n"
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
    A null-terminated input string or a driver-owned string returned through
    ``const char **``.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer
    length_type: str = "std::size_t"
    nullable: bool = True

    def client_declaration(self) -> str:
        name = self.parameter.name
        if self.send:
            return (
                f"    {self.length_type} {name}_len = {name} == nullptr ? 0 : "
                f"static_cast<{self.length_type}>(std::strlen({name}) + 1);\n"
            )
        present = (
            f"    uint8_t {name}_present = {name} != nullptr;\n"
            if self.nullable
            else f"    if ({name} == nullptr) return CUDA_ERROR_INVALID_VALUE;\n"
        )
        return (
            present
            + f"    {self.length_type} {name}_len = 0;\n"
            + f"    std::string {name}_result;\n"
        )

    def client_rpc_write(self, f):
        name = self.parameter.name
        if self.recv:
            if self.nullable:
                f.write(
                    f"        rpc_write(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
                )
            return
        if not self.send:
            return
        f.write(
            f"        rpc_write(conn, &{name}_len, sizeof({self.length_type})) < 0 ||\n"
            f"        ({name}_len != 0 && rpc_write(conn, {name}, {name}_len) < 0) ||\n"
        )

    @property
    def server_declaration(self) -> str:
        type_ = self.ptr.ptr_to.format() if self.recv else self.ptr.format()
        present = (
            f"    uint8_t {self.parameter.name}_present = 0;\n"
            if self.recv and self.nullable
            else ""
        )
        initializer = " = 0" if self.recv else ""
        return (
            present
            + f"    {type_} {self.parameter.name} = nullptr;\n"
            + f"    {self.length_type} {self.parameter.name}_len{initializer};\n"
        )

    def server_rpc_read(self, f) -> Optional[str]:
        name = self.parameter.name
        if self.recv:
            if self.nullable:
                f.write(
                    f"        rpc_read(conn, &{name}_present, sizeof(uint8_t)) < 0 ||\n"
                )
            return None
        if not self.send:
            return None
        f.write(
            f"        rpc_read(conn, &{name}_len, sizeof({self.length_type})) < 0)\n"
            "        goto ERROR_0;\n"
            f"    if ({name}_len != 0) {{\n"
            f"        {name} = ({self.ptr.format()})malloc({name}_len);\n"
            f"        if ({name} == nullptr)\n"
            "            goto ERROR_0;\n"
            "    }\n"
            f"    if (({name}_len != 0 && rpc_read(conn, (void *){name}, {name}_len) < 0) ||\n"
        )
        return name

    @property
    def server_reference(self) -> str:
        if self.recv:
            if not self.nullable:
                return f"&{self.parameter.name}"
            return f"{self.parameter.name}_present ? &{self.parameter.name} : nullptr"
        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        guard = f"{name}_present && " if self.nullable else ""
        f.write(
            f"        (({name}_len = {name} != nullptr\n"
            f"              ? static_cast<{self.length_type}>(std::strlen({name}))\n"
            f"              : 0), false) ||\n"
            f"        ({guard}rpc_write(conn, &{name}_len, sizeof({self.length_type})) < 0) ||\n"
            f"        ({guard}{name}_len != 0 && rpc_write(conn, {name}, {name}_len) < 0) ||\n"
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        if not self.nullable:
            f.write(
                f"        rpc_read(conn, &{name}_len, sizeof({self.length_type})) < 0 ||\n"
                f"        {name}_len > (1U << 20) ||\n"
                f"        ({name}_result.resize({name}_len), false) ||\n"
                f"        ({name}_len != 0 &&\n"
                f"         rpc_read(conn, {name}_result.data(), {name}_len) < 0) ||\n"
            )
            return
        f.write(
            f"        ({name}_present && (\n"
            f"            rpc_read(conn, &{name}_len, sizeof({self.length_type})) < 0 ||\n"
            f"            {name}_len > (1U << 20) ||\n"
            f"            ({name}_result.resize({name}_len), false) ||\n"
            f"            ({name}_len != 0 &&\n"
            f"             rpc_read(conn, {name}_result.data(), {name}_len) < 0))) ||\n"
        )

    def client_post_rpc(
        self,
        f,
        success_value: str,
        allocation_error: str,
        owner_name: str,
    ):
        if not self.recv:
            return
        name = self.parameter.name
        guard = f" && {name}_present" if self.nullable else ""
        f.write(
            f"    if (return_value == {success_value}{guard}) {{\n"
            f"        const char *{name}_stored = lupine_retain_returned_string(\n"
            f"            reinterpret_cast<const void *>({owner_name}), "
            f"{name}_result.data(), {name}_result.size());\n"
            f"        if ({name}_stored == nullptr)\n"
            f"            return {allocation_error};\n"
            f"        *{name} = {name}_stored;\n"
            "    }\n"
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

    def wire_size(self) -> str:
        return f"sizeof({self.type_.format()})"

    def client_slot(self, name: str) -> str:
        return f"&{name}"

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
                f"        rpc_write(conn, {self.client_slot(param_name)}, {self.wire_size()}) < 0 ||\n"
            )

    @property
    def server_declaration(self) -> str:
        init = "{}" if not self.send else ""
        # ensure we don't have a const struct, otherwise we can't initialise it properly; ex: "const cudnnTensorDescriptor_t xDesc;" is invalid...
        # but "const cudnnTensorDescriptor_t *xDesc" IS valid. This subtle change carries reprecussions.
        if (
            "const " in self.type_.format()
            and "void" not in self.type_.format()
            and "*" not in self.type_.format()
        ):
            return f"   {self.type_.format().replace('const', '')} {self.parameter.name}{init};\n"
        return f"    {self.type_.format()} {self.parameter.name}{init};\n"

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            f"        rpc_read(conn, &{self.parameter.name}, {self.wire_size()}) < 0 ||\n"
        )

    @property
    def server_reference(self) -> str:
        return self.parameter.name

    def server_rpc_write(self, f):
        pass

    def client_rpc_read(self, f):
        pass


@dataclass
class ScalarOperation:
    """
    A pointer to a scalar that the library's pointer mode places on the host or
    on the device (a cuBLAS alpha, beta, or result). The client decides per
    call: in host mode the value travels by copy, in device mode the address
    travels unchanged. The width leads on the wire, and zero means an address
    follows. `mode` is the handle or descriptor whose pointer mode decides,
    asked as `scalar_on_host(mode, "<name>")` since a mode may place alpha and
    beta differently; `width` is a C++ expression for the value's size, or None
    for the pointee's sizeof.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer
    mode: Parameter
    width: Optional[str]

    def width_expr(self) -> str:
        return self.width or f"sizeof({self.ptr.ptr_to.format()})"

    def client_declaration(self) -> str:
        name = self.parameter.name
        return (
            f"    uint32_t {name}_width = {name} != nullptr && "
            f'scalar_on_host({self.mode.name}, "{name}") ? '
            f"static_cast<uint32_t>({self.width_expr()}) : 0;\n"
        )

    def client_rpc_write(self, f):
        name = self.parameter.name
        f.write(
            f"        rpc_write(conn, &{name}_width, sizeof({name}_width)) < 0 ||\n"
        )
        f.write(
            f"        ({name}_width == 0 && rpc_write(conn, &{name}, sizeof({name})) < 0) ||\n"
        )
        if self.send:
            f.write(
                f"        ({name}_width != 0 && rpc_write(conn, {name}, {name}_width) < 0) ||\n"
            )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        f.write(
            f"        ({name}_width != 0 && rpc_read(conn, {name}, {name}_width) < 0) ||\n"
        )

    # The value lands in an inline slot, or on the heap when wider (a grouped
    # GEMM's one-scalar-per-group alpha).
    def server_slot(self) -> str:
        name = self.parameter.name
        return (
            f"lupine_scalar_slot({name}_heap, {name}_value, "
            f"sizeof({name}_value), {name}_width)"
        )

    @property
    def server_declaration(self) -> str:
        name = self.parameter.name
        return (
            f"    uint32_t {name}_width;\n"
            f"    alignas(16) unsigned char {name}_value[64];\n"
            f"    std::vector<unsigned char> {name}_heap;\n"
            f"    {self.ptr.format()} {name} = nullptr;\n"
        )

    def server_rpc_read(self, f):
        name = self.parameter.name
        f.write(
            f"        rpc_read(conn, &{name}_width, sizeof({name}_width)) < 0 ||\n"
        )
        f.write(
            f"        ({name}_width == 0 && rpc_read(conn, &{name}, sizeof({name})) < 0) ||\n"
        )
        if self.send:
            f.write(
                f"        ({name}_width != 0 && rpc_read(conn, {self.server_slot()}, {name}_width) < 0) ||\n"
            )

    @property
    def server_reference(self) -> str:
        name = self.parameter.name
        return (
            f"{name}_width != 0 ? reinterpret_cast<{self.ptr.format()}>({self.server_slot()}) : {name}"
        )

    def server_rpc_write(self, f):
        if not self.recv:
            return
        name = self.parameter.name
        f.write(
            f"        ({name}_width != 0 && rpc_write(conn, {self.server_slot()}, {name}_width) < 0) ||\n"
        )


@dataclass
class VersionedStructOperation:
    """
    An optional pointer to a versioned, append-only configuration struct whose
    leading `size` member says how many bytes the caller's header defines (an
    NCCL `ncclConfig_t`). The caller's bytes travel, then each string member's
    length and text; the server widens the struct to its own size and points
    the string members at its copies. Members named in `cleared` are client
    addresses the library ignores, so the server nulls them. A member is
    patched only when it lies within the caller's size. A member some headers
    lack names the preprocessor condition that declares it in
    `member_guards`; its length still travels, so builds against different
    headers keep one wire format.
    """

    parameter: Parameter
    ptr: Pointer
    strings: tuple[str, ...]
    cleared: tuple[str, ...]
    member_guards: dict
    send: bool = True
    recv: bool = False

    def struct_type(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        name = self.ptr.ptr_to.format()
        self.ptr.ptr_to.const = c
        return name

    def guarded(self, member: str, text: str) -> str:
        guard = self.member_guards.get(member)
        return f"#if {guard}\n{text}#endif\n" if guard else text

    def client_declaration(self) -> str:
        name = self.parameter.name
        type_ = self.struct_type()
        s = f"    uint64_t {name}_size = {name} != nullptr ? {name}->size : 0;\n"
        for member in self.strings:
            s += f"    uint32_t {name}_{member}_len = 0;\n"
            s += self.guarded(
                member,
                f"    if (offsetof({type_}, {member}) + sizeof(char *) <= {name}_size &&\n"
                f"        {name}->{member} != nullptr)\n"
                f"        {name}_{member}_len = static_cast<uint32_t>(std::strlen({name}->{member}) + 1);\n",
            )
        return s

    def client_rpc_write(self, f):
        name = self.parameter.name
        f.write(f"        rpc_write(conn, &{name}_size, sizeof({name}_size)) < 0 ||\n")
        f.write(
            f"        ({name}_size != 0 && rpc_write(conn, {name}, {name}_size) < 0) ||\n"
        )
        for member in self.strings:
            f.write(
                f"        rpc_write(conn, &{name}_{member}_len, sizeof(uint32_t)) < 0 ||\n"
            )
            f.write(
                self.guarded(
                    member,
                    f"        ({name}_{member}_len != 0 &&\n"
                    f"         rpc_write(conn, {name}->{member}, {name}_{member}_len) < 0) ||\n",
                )
            )

    def client_rpc_read(self, f):
        return

    @property
    def server_declaration(self) -> str:
        name = self.parameter.name
        s = (
            f"    uint64_t {name}_size = 0;\n"
            f"    std::vector<unsigned char> {name}_bytes;\n"
        )
        for member in self.strings:
            s += (
                f"    uint32_t {name}_{member}_len = 0;\n"
                f"    std::string {name}_{member};\n"
            )
        return s

    def server_rpc_read(self, f):
        name = self.parameter.name
        type_ = self.struct_type()
        f.write(
            f"        rpc_read(conn, &{name}_size, sizeof({name}_size)) < 0 ||\n"
            f"        ({name}_bytes.assign(std::max<uint64_t>({name}_size, sizeof({type_})), 0),\n"
            f"         {name}_size != 0 && rpc_read(conn, {name}_bytes.data(), {name}_size) < 0) ||\n"
        )
        for member in self.strings:
            f.write(
                f"        rpc_read(conn, &{name}_{member}_len, sizeof(uint32_t)) < 0 ||\n"
                f"        ({name}_{member}.resize({name}_{member}_len),\n"
                f"         {name}_{member}_len != 0 &&\n"
                f"             rpc_read(conn, &{name}_{member}[0], {name}_{member}_len) < 0) ||\n"
            )
        return None

    @property
    def server_reference(self) -> str:
        name = self.parameter.name
        type_ = self.struct_type()
        strings = "".join(
            self.guarded(
                member,
                f"{{offsetof({type_}, {member}), {name}_{member}_len != 0 ? {name}_{member}.c_str() : nullptr}},\n",
            )
            for member in self.strings
        )
        cleared = "".join(
            self.guarded(member, f"offsetof({type_}, {member}),\n")
            for member in self.cleared
        )
        return (
            f"lupine_versioned_struct<{type_}>({name}_bytes, {name}_size,\n"
            f"{{\n{strings}}},\n{{\n{cleared}}})"
        )

    def server_rpc_write(self, f):
        return


Operation = Union[
    NullableOperation,
    ArrayOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    ScalarOperation,
    VersionedStructOperation,
]


@dataclass
class OwnerAnnotation:
    kind: str
    parameter: Parameter


@dataclass
class RetainAnnotation:
    parameter: Parameter
    handle: Parameter


@dataclass
class ReleaseAnnotation:
    kind: str
    parameter: Parameter


@dataclass
class ParentAnnotation:
    kind: str
    child: Parameter
    parent: Parameter


@dataclass
class CrossServerCopyAnnotation:
    dst: Parameter
    src: Parameter
    bytes: Parameter
    stream: Optional[Parameter] = None
    async_: bool = False


@dataclass
class RoutingFallbackAnnotation:
    kind: str
    parameter: Parameter


@dataclass
class SynchronizeAnnotation:
    deferred_dtoh: bool = False
    stdout: bool = False


@dataclass
class GraphExecNodeAnnotation:
    graph_exec: Parameter
    node: Parameter


@dataclass
class GraphExecUpdateAnnotation:
    graph_exec: Parameter
    graph: Parameter
    error: Parameter


@dataclass(frozen=True)
class ClientCallTemplate:
    return_type: str
    before_call: str
    after_call: str


@dataclass
class FunctionAnnotationMetadata:
    operations: list[Operation]
    guard: Optional[str] = None
    disabled_client: bool = False
    disabled_server: bool = False
    # @async: fire-and-forget op; client does not wait, server sends no response.
    async_fire_forget: bool = False
    # Optional response fields are emitted by the corresponding manual server
    # handler before the generated CUDA result.
    synchronize: Optional[SynchronizeAnnotation] = None
    # @synchronize DEVICEPTR <param>: addresses the call writes from the
    # device. Mapped host memory there is refreshed at the next synchronize.
    device_writes: list[Parameter] = field(default_factory=list)
    routing_kind: Optional[str] = None
    routing_parameter: Optional[Parameter] = None
    routing_fallback: Optional[RoutingFallbackAnnotation] = None
    record_owners: list[OwnerAnnotation] = field(default_factory=list)
    retains: list[RetainAnnotation] = field(default_factory=list)
    releases: list[ReleaseAnnotation] = field(default_factory=list)
    parents: list[ParentAnnotation] = field(default_factory=list)
    cross_server_copy: Optional[CrossServerCopyAnnotation] = None
    graph_exec_node: Optional[GraphExecNodeAnnotation] = None
    graph_exec_update: Optional[GraphExecUpdateAnnotation] = None
    client_call_template: Optional[ClientCallTemplate] = None
    # @clearfields: members holding host addresses, cleared on both sides.
    clear_fields: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
