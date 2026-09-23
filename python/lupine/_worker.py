"""The torch worker: a CUDA torch that executes the backend's operators.

Run as ``lupine-torch-worker`` (or ``python -m lupine._worker``) in an
interpreter whose torch is the same major.minor release as the host's. It
loads the client shims against ``LUPINE_SERVER`` (or ``LUPINE_LIBDIR``)
before torch initialises CUDA, then serves one backend connection and exits
when the host disconnects. Pure Python: requests arrive over
``lupine._backend.transport`` and are decoded with ``lupine._backend.wire``;
every operator resolves to ``torch.ops.<ns>.<name>.<overload>`` from its
schema name.

One thread reads requests off the socket in the order the host issued them
and runs each before reading the next, so it is the only thread that touches
CUDA and the driver shim sees one client thread and one lane, ordered on the
wire, as the driver path itself does. The socket buffers are the read-ahead:
a host that issues faster than the worker runs blocks in its send.

A plan the host replays is a template here (``_Template``): the synchronous
call that taught the host the plan names the plan number, and its operator,
call and the table entries' specs are kept, so an ``OP_PLAN`` frame carries
only handles and offsets, plus the call when it differs from the template's.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Any

READY_PREFIX = "LUPINE_WORKER_READY"
# Above every handle the host will assign (see csrc/backend.h).
_FIRST_WORKER_HANDLE = 1 << 40

# The namespace exec/eval messages from the host run in.
_namespace: dict[str, Any] = {}
_graphs: dict[int, tuple[Any, Any]] = {}


# LUPINE_WORKER_DEVICE=cpu runs the operators on the worker's CPU torch: no
# GPU, no shims, for testing the backend itself.
def device_kind() -> str:
    return os.environ.get("LUPINE_WORKER_DEVICE", "cuda")


def info() -> dict[str, Any]:
    import torch

    kind = device_kind()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": kind,
        "device_count": torch.cuda.device_count() if kind == "cuda" else 1,
    }


def _exec(code: str) -> None:
    if "torch" not in _namespace:
        _prepare_namespace()
    exec(code, _namespace)


def _eval(code: str) -> str:
    if "torch" not in _namespace:
        _prepare_namespace()
    return json.dumps(eval(code, _namespace))


# The host's torch.cuda module forwards its queries and setters here (see
# _backend.device): pickled arguments, the host's current device selected,
# a pickled result. Device properties are rebuilt as the host's own class.
def _cuda(name: str, payload: str, device: int) -> str:
    import base64
    import pickle

    import torch

    args, kwargs = pickle.loads(base64.b64decode(payload))
    if device_kind() != "cuda":
        # The CPU worker of the tests: seeds go to the CPU generator, the
        # rest is whatever a CUDA-less torch.cuda answers.
        if name in ("manual_seed", "manual_seed_all"):
            torch.manual_seed(*args)
            result = None
        else:
            result = getattr(torch.cuda, name)(*args, **kwargs)
    else:
        with torch.cuda.device(device):
            result = getattr(torch.cuda, name)(*args, **kwargs)
    if isinstance(result, torch.cuda._CudaDeviceProperties):
        from ._backend.device import _Properties

        result = _Properties(
            {
                key: value if isinstance(value, (bool, int, float, str)) else str(value)
                for key in dir(result)
                if not key.startswith("_")
                for value in [getattr(result, key)]
            }
        )
    return base64.b64encode(pickle.dumps(result)).decode()


# CUDA graph capture happens here, on the worker's torch: every op runs on
# the executor thread, so making the capture stream that thread's current
# stream records the ops the host keeps issuing during capture.
def _capture_begin(graph_id: int, capture_error_mode: str, pool: Any = None) -> None:
    import torch

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    graph.capture_begin(pool=pool, capture_error_mode=capture_error_mode)
    _graphs[graph_id] = (graph, stream)


def _capture_end(graph_id: int) -> None:
    import torch

    graph, stream = _graphs[graph_id]
    graph.capture_end()
    torch.cuda.set_stream(torch.cuda.default_stream())
    torch.cuda.current_stream().wait_stream(stream)


def _replay(graph_id: int) -> None:
    _graphs[graph_id][0].replay()


def _reset(graph_id: int) -> None:
    entry = _graphs.pop(graph_id, None)
    if entry is not None:
        entry[0].reset()


def _prepare_namespace() -> None:
    import torch

    _namespace.update(
        {
            "torch": torch,
            "_capture_begin": _capture_begin,
            "_capture_end": _capture_end,
            "_replay": _replay,
            "_reset": _reset,
            "_cuda": _cuda,
        }
    )


class _Entry:
    """A storage the host knows by handle, with a tensor per dtype to take
    views from and the views taken so far by (spec, offset); a view stays
    valid when the storage grows in place."""

    __slots__ = ("storage", "cdata", "nbytes", "bases", "views")

    def __init__(self, storage: Any):
        self.storage = storage
        self.cdata = storage._cdata
        self.nbytes = storage.nbytes()
        self.bases: dict[Any, Any] = {}
        self.views: dict[Any, Any] = {}


class _Spec:
    """A view's dtype, sizes and strides, interned per executor so views key
    on the spec's identity."""

    __slots__ = ("dtype", "sizes", "strides", "nbytes")

    def __init__(self, dtype: Any, sizes: tuple[int, ...], strides: tuple[int, ...]):
        self.dtype = dtype
        self.sizes = sizes
        self.strides = strides
        self.nbytes = 0 if 0 in sizes else (1 + sum((s - 1) * st for s, st in zip(sizes, strides))) * dtype.itemsize


class _Resolved:
    """An operator by schema name: the callable and the positions of its
    Tensor arguments, where torch hands a Python fallback a wrapped number
    (a 0-dim CPU tensor standing in for a Python scalar) as the scalar and
    the op wants the tensor back."""

    __slots__ = ("op", "tensor_args")

    def __init__(self, op: Any):
        self.op = op
        self.tensor_args = tuple(
            i for i, a in enumerate(op._schema.arguments) if not a.kwarg_only and str(a.type) in ("Tensor", "Tensor?")
        )


class _Slot:
    """Stands for table entry ``index`` when a call is unpickled to build its
    template: two entries can materialise to one view (the same slice twice),
    so slots are told apart by position, never by the tensor."""

    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index


class _Template:
    """A plan as the worker replays it (see ``wire.OP_PLAN``): the operator,
    each table entry's spec and device, a generated builder that puts the
    materialised entries back into the call, and the results' sizes and
    specs."""

    __slots__ = ("name", "resolved", "slots", "build", "consts", "sizes", "results")

    def __init__(self, name: str, resolved: _Resolved, slots: list[tuple[_Spec, int]], args: Any, kwargs: Any, sizes: list[Any], results: list[Any]):
        self.name = name
        self.resolved = resolved
        self.slots = slots
        self.build, self.consts = _compile(args, kwargs)
        self.sizes = sizes
        self.results = results


def _compile(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, list[Any]]:
    consts: list[Any] = []

    def gen(value: Any) -> str:
        if type(value) is _Slot:
            return f"i[{value.index}]"
        if type(value) is tuple:
            return "(" + "".join(gen(item) + "," for item in value) + ")"
        if type(value) is list:
            return "[" + ",".join(gen(item) for item in value) + "]"
        consts.append(value)
        return f"c[{len(consts) - 1}]"

    source = "lambda i, c: (" + gen(args) + ", {" + ", ".join(f"{k!r}: {gen(v)}" for k, v in kwargs.items()) + "})"
    return eval(source), consts


_NUMBERS = (bool, int, float, complex)


class Executor:
    def __init__(self, server: Any):
        import torch

        from ._backend import wire

        self.torch = torch
        self.wire = wire
        self.server = server
        self.table: dict[int, _Entry] = {}
        self.by_storage: dict[int, int] = {}
        self.next_handle = _FIRST_WORKER_HANDLE
        self.ops: dict[str, _Resolved] = {}
        self.specs: dict[Any, _Spec] = {}
        self.templates: dict[int, _Template] = {}
        self.wrapped: dict[Any, Any] = {}
        self.wrapped_dtype = {bool: torch.bool, int: torch.int64, float: torch.float64, complex: torch.complex128}
        self.kind = device_kind()
        self.devices = (
            [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
            if self.kind == "cuda"
            else [torch.device(self.kind)]
        )
        self.current_device = 0
        self.error: str | None = None
        self.trace = os.environ.get("LUPINE_TORCH_TRACE") is not None
        self.profile: dict[str, list[int]] | None = None

    # --- storages ---------------------------------------------------------

    def spec(self, dtype: str, sizes: tuple[int, ...], strides: tuple[int, ...]) -> _Spec:
        key = (dtype, sizes, strides)
        spec = self.specs.get(key)
        if spec is None:
            spec = self.specs[key] = _Spec(self.wire.dtype_of(dtype), sizes, strides)
        return spec

    def view(self, handle: int, nbytes: int, device: int, spec: _Spec, offset: int) -> Any:
        entry = self.table.get(handle)
        if entry is None:
            # A storage the host allocated and never sent an op for is created
            # on first use; the current stream is the capture stream during
            # capture, so even that allocation lands in the graph.
            if offset == 0 and spec.nbytes == nbytes:
                t = self.torch.empty_strided(spec.sizes, spec.strides, dtype=spec.dtype, device=self.devices[device])
                self.file(handle, t.untyped_storage(), t, spec, 0)
                return t
            bytes_ = self.torch.empty(nbytes, dtype=self.torch.uint8, device=self.devices[device])
            entry = self.file(handle, bytes_.untyped_storage())
        elif nbytes > entry.nbytes:
            # The host grew it (resize_, set_); grow here in place, keeping
            # the contents and the storage identity every base refers to.
            self.torch.empty(0, dtype=self.torch.uint8, device=entry.storage.device).set_(entry.storage).resize_(nbytes)
            entry.nbytes = nbytes
        key = (spec, offset)
        view = entry.views.get(key)
        if view is None:
            view = entry.views[key] = self.base(entry, spec.dtype).as_strided(spec.sizes, spec.strides, offset)
        return view

    def materialize(self, desc: Any) -> Any:
        handle, nbytes, device, dtype, sizes, strides, offset = desc
        return self.view(handle, nbytes, device, self.spec(dtype, sizes, strides), offset)

    def file(self, handle: int, storage: Any, tensor: Any = None, spec: _Spec | None = None, offset: int = 0) -> _Entry:
        """Files a storage under a handle; a result tensor over it is the
        first base and view, so the op that consumes it next takes no
        as_strided."""

        entry = _Entry(storage)
        self.table[handle] = entry
        self.by_storage[entry.cdata] = handle
        if tensor is not None:
            if spec is None:
                spec = self.spec_of(tensor)
                offset = tensor.storage_offset()
            entry.bases[spec.dtype] = tensor
            entry.views[(spec, offset)] = tensor
        return entry

    def spec_of(self, t: Any) -> _Spec:
        return self.spec(self.wire.dtype_name(t.dtype), tuple(t.shape), t.stride())

    def base(self, entry: _Entry, dtype: Any) -> Any:
        base = entry.bases.get(dtype)
        if base is None:
            base = entry.bases[dtype] = self.torch.empty(0, dtype=dtype, device=entry.storage.device).set_(entry.storage)
        return base

    def wrap_numbers(self, resolved: _Resolved, args: tuple[Any, ...]) -> tuple[Any, ...]:
        for i in resolved.tensor_args:
            if i < len(args) and type(args[i]) in _NUMBERS:
                args = args[:i] + (self.wrap(args[i]),) + args[i + 1 :]
        return args

    def wrap(self, value: Any) -> Any:
        """The tensor a wrapped number stands for, with the dtype wrapped
        numbers carry so type promotion sees the same operand."""

        key = (type(value), value)
        t = self.wrapped.get(key)
        if t is None:
            if len(self.wrapped) >= 4096:
                self.wrapped.clear()
            t = self.wrapped[key] = self.torch.tensor(value, dtype=self.wrapped_dtype[type(value)])
        return t

    def handle_of(self, t: Any) -> int:
        storage = t.untyped_storage()
        handle = self.by_storage.get(storage._cdata)
        if handle is None:
            handle = self.next_handle
            self.next_handle += 1
            self.file(handle, storage, t)
        return handle

    def describe(self, t: Any) -> Any:
        return (
            self.handle_of(t),
            t.untyped_storage().nbytes(),
            t.device.index or 0,
            self.wire.dtype_name(t.dtype),
            tuple(t.shape),
            t.stride(),
            t.storage_offset(),
        )

    def apply_frees(self, frees: Any) -> None:
        for handle in frees:
            entry = self.table.pop(handle, None)
            if entry is not None:
                self.by_storage.pop(entry.cdata, None)

    def select_device(self, device: int) -> None:
        if device != self.current_device and self.kind == "cuda":
            self.current_device = device
            self.torch.cuda.set_device(device)

    # A handle the host already holds (an argument the plan says the op
    # returns or views) must come back over the same storage; anything else
    # is a plan that does not describe this op.
    def bind(self, name: str, results: list[Any], handles: Any, sizes: list[Any], seeds: list[Any]) -> None:
        if len(results) != len(handles):
            raise RuntimeError(f"{name} returned {len(results)} tensors where the host's plan has {len(handles)}")
        for i, t in enumerate(results):
            if t.shape != sizes[i]:
                raise RuntimeError(
                    f"{name} produced shape {tuple(t.shape)} where the host's plan has {sizes[i]}: "
                    "its output shape depends on data, so it must run synchronously"
                )
            storage = t.untyped_storage()
            entry = self.table.get(handles[i])
            if entry is None:
                self.file(handles[i], storage, t, *seeds[i])
            elif entry.cdata != storage._cdata:
                raise RuntimeError(
                    f"{name} returned a new storage where the host's plan has an argument's; "
                    "it must run synchronously"
                )

    # --- operators --------------------------------------------------------

    def resolve(self, name: str) -> _Resolved:
        resolved = self.ops.get(name)
        if resolved is None:
            namespace, _, rest = name.partition("::")
            base, _, overload = rest.partition(".")
            try:
                packet = getattr(getattr(self.torch.ops, namespace), base)
                op = getattr(packet, overload or "default")
            except (AttributeError, RuntimeError) as exc:
                raise _MissingOp(f"the worker's torch {self.torch.__version__} has no operator {name} ({exc})") from exc
            resolved = self.ops[name] = _Resolved(op)
        return resolved

    def run(self, meta: bytes, body: bytes) -> tuple[Any, list[Any], str]:
        started = time.perf_counter_ns() if self.profile is not None else 0
        frees, descs, _, plan = self.wire.unpack_meta(meta)
        self.apply_frees(frees)
        inputs = [self.materialize(d) for d in descs]
        name, args, kwargs = self.wire.loads(body, inputs)
        resolved = self.resolve(name)
        args = self.wrap_numbers(resolved, args)
        if self.trace:
            print(f"lupine-torch worker {name}", *(tuple(t.shape) for t in inputs), file=sys.stderr)
        if descs:
            self.select_device(descs[0][2])
        decoded = time.perf_counter_ns() if started else 0
        if name == "aten::_fused_sdp_choice" and self.kind != "cuda":
            # The host's composite attention maps the answer to CUDA kernels;
            # the CPU worker of the tests has only the math path.
            value = 0
        else:
            value = resolved.op(*args, **kwargs)
        if plan:
            results = self.wire.result_tensors(value)
            slots = [(self.spec(dtype, sz, st), device) for _, _, device, dtype, sz, st, _ in descs]
            _, slotted, skwargs = self.wire.loads(body, [_Slot(i) for i in range(len(descs))])
            slotted = self.wrap_numbers(resolved, slotted)
            sizes = [tuple(t.shape) for t in results]
            seeds = [(self.spec_of(t), t.storage_offset()) for t in results]
            self.templates[plan] = _Template(name, resolved, slots, slotted, skwargs, sizes, seeds)
        if started:
            self.note(name, started, decoded, time.perf_counter_ns())
        return value, inputs, name

    def replay(self, meta: bytes, body: bytes) -> None:
        started = time.perf_counter_ns() if self.profile is not None else 0
        ints = self.wire.unpack_plan(meta)
        template = self.templates.get(ints[0])
        if template is None:
            raise RuntimeError(f"replay of plan {ints[0]} before its template")
        p = 2 + ints[1]
        if ints[1]:
            self.apply_frees(ints[2:p])
        table = self.table
        inputs = []
        for spec, device in template.slots:
            handle = ints[p]
            offset = ints[p + 2]
            entry = table.get(handle)
            view = entry.views.get((spec, offset)) if entry is not None and ints[p + 1] <= entry.nbytes else None
            if view is None:
                view = self.view(handle, ints[p + 1], device, spec, offset)
            inputs.append(view)
            p += 3
        if inputs:
            self.select_device(template.slots[0][1])
        if self.trace:
            print(f"lupine-torch worker {template.name}", *(tuple(t.shape) for t in inputs), file=sys.stderr)
        if body:
            _, args, kwargs = self.wire.loads(body, inputs)
            args = self.wrap_numbers(template.resolved, args)
        else:
            args, kwargs = template.build(inputs, template.consts)
        decoded = time.perf_counter_ns() if started else 0
        value = template.resolved.op(*args, **kwargs)
        self.bind(template.name, self.wire.result_tensors(value), ints[p:], template.sizes, template.results)
        if started:
            self.note(template.name, started, decoded, time.perf_counter_ns())

    def note(self, name: str, started: int, decoded: int, finished: int) -> None:
        entry = self.profile.get(name)
        if entry is None:
            entry = self.profile[name] = [0, 0, 0]
        entry[0] += 1
        entry[1] += decoded - started
        entry[2] += finished - decoded

    def report(self, value: Any, inputs: list[Any]) -> bytes:
        rtable: list[Any] = []
        rbody = self.wire.dumps(value, self.wire.table_reference(rtable), lambda d: d.__reduce__())
        aliases = []
        for t in rtable:
            alias = (self.wire.FRESH, -1)
            for i, inp in enumerate(inputs):
                if t is inp:
                    alias = (self.wire.INPUT, i)
                    break
            else:
                cdata = t.untyped_storage()._cdata
                for i, inp in enumerate(inputs):
                    if inp.untyped_storage()._cdata == cdata:
                        alias = (self.wire.INPUT_STORAGE, i)
                        break
            aliases.append(alias)
        rdescs = [self.describe(t) for t in rtable]
        import pickle

        return pickle.dumps((aliases, rdescs, rbody), protocol=pickle.HIGHEST_PROTOCOL)

    # --- requests ---------------------------------------------------------

    def record(self, message: str) -> None:
        print(f"lupine torch worker: {message}", file=sys.stderr)
        if self.error is None:
            self.error = message

    def take_error(self) -> str | None:
        error, self.error = self.error, None
        return error

    def handle(self, frame: Any) -> None:
        wire = self.wire
        ticket, kind, meta, body, extra = frame
        if kind == wire.OP_PLAN:
            try:
                self.replay(meta, body)
            except Exception as exc:
                self.record(_message(exc))
            return
        if kind == wire.COPY_FROM_HOST:
            try:
                frees, descs, _, _ = wire.unpack_meta(meta)
                self.apply_frees(frees)
                target = self.materialize(descs[0])
                source = self.torch.frombuffer(extra, dtype=target.dtype).reshape(target.shape) if extra else self.torch.empty(target.shape, dtype=target.dtype)
                target.copy_(source)
            except Exception as exc:
                self.record(_message(exc))
            return
        if kind == wire.EXEC:
            try:
                frees, _, _, _ = wire.unpack_meta(meta)
                self.apply_frees(frees)
                _exec(body.decode())
            except Exception as exc:
                self.record(_message(exc))
            return
        payload: Any = b""
        try:
            if kind == wire.OP_SYNC:
                value, inputs, _ = self.run(meta, body)
                payload = self.report(value, inputs)
            elif kind == wire.COPY_TO_HOST:
                frees, descs, dtype, _ = wire.unpack_meta(meta)
                self.apply_frees(frees)
                t = self.materialize(descs[0])
                if t.dtype != wire.dtype_of(dtype):
                    t = t.to(wire.dtype_of(dtype))
                cpu = t.contiguous().cpu()
                payload = wire.cpu_buffer(cpu)
            elif kind == wire.EVAL:
                frees, _, _, _ = wire.unpack_meta(meta)
                self.apply_frees(frees)
                payload = _eval(body.decode()).encode()
            elif kind == wire.SYNC:
                frees, _, _, _ = wire.unpack_meta(meta)
                self.apply_frees(frees)
                if self.kind == "cuda":
                    for i in range(len(self.devices)):
                        self.torch.cuda.synchronize(i)
            else:
                raise RuntimeError(f"unknown request kind {kind}")
            error = self.take_error()
        except _MissingOp as exc:
            self.server.respond(ticket, wire.MISSING_OP, str(exc).encode())
            return
        except Exception as exc:
            error = _message(exc)
        if error is not None:
            self.server.respond(ticket, wire.ERROR, error.encode())
        else:
            self.server.respond(ticket, wire.OK, payload)


class _MissingOp(RuntimeError):
    pass


def _message(exc: BaseException) -> str:
    return str(exc).strip() or type(exc).__name__


def serve(server: Any) -> None:
    """Runs every request in arrival order on this thread, the only one that
    touches CUDA. LUPINE_TORCH_PROFILE=<path> (LUPINE_WORKER_TORCH_PROFILE on
    the host) writes, per op, its count and the nanoseconds spent decoding
    and running it."""

    import torch

    executor = Executor(server)
    executor.profile = {} if os.environ.get("LUPINE_TORCH_PROFILE") else None
    with torch.no_grad():
        for frame in server.requests():
            executor.handle(frame)
    if executor.profile is not None:
        with open(os.environ["LUPINE_TORCH_PROFILE"], "w") as f:
            f.write(f"__messages__\t{server.messages}\t0\t0\n")
            for name, (count, decode, run) in executor.profile.items():
                f.write(f"{name}\t{count}\t{decode}\t{run}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lupine-torch-worker")
    parser.add_argument(
        "--listen",
        default="127.0.0.1:0",
        help="host:port to accept the backend connection on (port 0 picks one)",
    )
    args = parser.parse_args(argv)
    host, _, port = args.listen.rpartition(":")

    if device_kind() == "cuda":
        from . import load_native

        load_native(missing_ok=False)
    import torch

    if device_kind() == "cuda":
        torch.cuda.init()
        if torch.cuda.device_count() == 0:
            print("lupine-torch-worker: no CUDA device is visible", file=sys.stderr)
            return 1
    from ._backend import transport

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host or "127.0.0.1", int(port)))
    listener.listen(1)
    print(f"{READY_PREFIX} {listener.getsockname()[1]} {torch.__version__}", flush=True)
    server = transport.Server.accept(listener)
    listener.close()
    server.hello(info())
    serve(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
