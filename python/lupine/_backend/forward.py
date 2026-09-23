"""Operator forwarding: the boxed Python fallback and its plan cache.

Every operator torch dispatches to the backend device lands in ``fallback``
through ``torch.library.Library("_", "IMPL").fallback``. The first call of
an operator with a given argument metadata (the pickled call with its device
tensors described by device, dtype, sizes and strides, storages numbered by
first appearance; in-place ops key on their tensors alone) is a synchronous
request: the worker runs it and reports, per result tensor, a descriptor and
what it is relative to the arguments (an argument itself, a view over an
argument's storage, or a new storage). The host builds the results from the
report and keeps it as the operator's *plan* for that key; the worker keeps
the same call as the plan's template. Every later call with the same key
builds its results from the plan over host-assigned handles and goes
fire-and-forget as a plan replay (``wire.OP_PLAN``): the plan number, the
storage handles and offsets, the result handles, and the call only when it
differs from the template's (an in-place op's scalar). The worker checks
the replay's results against the plan. Ops whose output shape depends on
data are synchronous every time. Storage releases ride at the front of the
next request.

Copies and ``_local_scalar_dense`` are Python kernels registered the same
way; only the metadata kernels are C++ (``csrc``).
"""

from __future__ import annotations

import atexit
import json
import os
import pickle
import sys
import threading
import time
from typing import Any

import torch
from torch.library import Library

from . import ext, transport, wire

_TENSOR = 0
_LIST = 1
_VALUE = 2

_DATA_DEPENDENT = {
    "aten::nonzero",
    "aten::masked_select",
    "aten::_unique",
    "aten::_unique2",
    "aten::unique_dim",
    "aten::unique_consecutive",
    "aten::unique_dim_consecutive",
    "aten::bincount",
    "aten::repeat_interleave",
    "aten::equal",
}
# Ops with a CompositeExplicitAutograd decomposition never reach the
# fallback; these have a fused CUDA kernel the worker should run instead.
_WHOLE = ("native_layer_norm", "native_group_norm", "_fused_rms_norm", "convolution", "convolution_backward")
# CPU tensor arguments travel by value; a large one makes the call
# uncacheable rather than the key huge.
_KEY_CPU_NUMEL = 64


class _Op:
    __slots__ = ("name", "mutates", "sync", "bool_index", "memo", "inplace", "returns", "plans")

    def __init__(self, op: Any):
        schema = op._schema
        self.name = schema.name + ("." + schema.overload_name if schema.overload_name else "")
        self.memo = schema.name == "aten::_fused_sdp_choice"
        self.sync = schema.name in _DATA_DEPENDENT
        self.bool_index = schema.name == "aten::index"
        self.mutates = any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments)
        has_out = any(a.name == "out" for a in schema.arguments)
        # In-place ops (foo_, no out= argument) cannot change their arguments'
        # metadata through a Scalar or float value, so the plan key leaves
        # those out and an optimizer's per-step coefficients hit the same plan.
        self.inplace = not has_out and schema.name.endswith("_")
        self.returns = len(schema.returns)
        self.plans: dict[Any, _Plan] = {}


class _Plan:
    __slots__ = ("rets", "groups", "needs_rpc", "number", "body")

    def __init__(self, rets: list[Any], groups: list[tuple[int, bool]], needs_rpc: bool, number: int, body: bytes):
        self.rets = rets
        # (nbytes, direct): direct when one output covers the storage from
        # offset 0, so torch.empty_strided makes it in one call.
        self.groups = groups
        self.needs_rpc = needs_rpc
        # The plan's number on the wire and the call the worker's template
        # holds under it.
        self.number = number
        self.body = body


# An output in a plan: (kind, index, same, dtype, sizes, strides, offset).
# kind INPUT: argument `index` returned as is (same: with its metadata
# untouched); INPUT_STORAGE: a view over argument `index`'s storage; FRESH:
# group `index`. The offset is relative to the argument's for the first two.


class _Encoder(wire.Encoder):
    """The host's pickler: device tensors file into ``table`` and pickle as
    their index, CPU tensors go by value (a large one marks the call
    uncacheable)."""

    def __init__(self, types: tuple[str, ...], device: Any):
        super().__init__(self.tensor, device)
        self.types = types
        self.table: list[torch.Tensor] = []
        self.big_cpu = False

    def tensor(self, t: torch.Tensor) -> Any:
        if t.device.type in self.types:
            self.table.append(t)
            return (wire._at, (len(self.table) - 1,))
        if t.device.type != "cpu":
            raise RuntimeError(f"lupine: cannot send a tensor on {t.device} to the worker")
        if t.numel() > _KEY_CPU_NUMEL:
            self.big_cpu = True
        return wire.cpu_value(t)

    def encode(self, value: Any) -> tuple[bytes, list[torch.Tensor], bool]:
        self.table = table = []
        self.big_cpu = False
        try:
            body = self.dumps(value)
        finally:
            # The table must not outlive the call for the same reason the
            # memo must not: a gradient with an extra reference gets cloned.
            self.table = []
        return body, table, self.big_cpu


class Backend:
    def __init__(self, client: transport.Client, dual: bool, C: Any):
        self.client = client
        self.C = C
        self.dual = dual
        self.device_type = "cuda" if dual else "lupine"
        self.types = ("cuda", "lupine") if dual else ("lupine",)
        self.frees: list[int] = []
        self.ops: dict[Any, _Op] = {}
        self.numbered = 0
        self.local = threading.local()
        self.devices: dict[int, torch.device] = {}
        self.device_rule = wire.device_as(client.hello.get("device", "cuda"), self.types)
        self.trace = os.environ.get("LUPINE_TORCH_TRACE") is not None
        self.profile_path = os.environ.get("LUPINE_TORCH_PROFILE")
        self.profile: dict[str, list[int]] = {}
        self.host_version = torch.__version__
        self.worker_version = client.hello.get("torch", "?")

    # --- helpers ---------------------------------------------------------

    def is_ours(self, t: torch.Tensor) -> bool:
        return t.device.type in self.types

    def device(self, index: int) -> torch.device:
        d = self.devices.get(index)
        if d is None:
            d = self.devices[index] = torch.device(self.device_type, index)
        return d

    def describe(self, t: torch.Tensor) -> wire.Desc:
        s = t.untyped_storage()
        return (
            s.data_ptr() >> ext.HANDLE_SHIFT,
            s.nbytes(),
            t.device.index,
            wire.dtype_name(t.dtype),
            tuple(t.shape),
            t.stride(),
            t.storage_offset(),
        )

    def encoder(self) -> _Encoder:
        enc = getattr(self.local, "encoder", None)
        if enc is None:
            enc = self.local.encoder = _Encoder(self.types, self.device_rule)
        return enc

    def meta(self, descs: list[wire.Desc], extra: Any = None, plan: int = 0) -> bytes:
        frees = self.frees
        if frees:
            self.frees = []
        return wire.pack_meta(frees, descs, extra, plan)

    def plan_meta(self, number: int, descs: list[wire.Desc], handles: list[int]) -> bytes:
        frees = self.frees
        if frees:
            self.frees = []
        ints = [number, len(frees), *frees]
        for handle, nbytes, _, _, _, _, offset in descs:
            ints += (handle, nbytes, offset)
        ints += handles
        return wire.pack_plan(ints)

    def raise_status(self, status: int, payload: bytes) -> None:
        message = payload.decode(errors="replace")
        if status == wire.MISSING_OP:
            raise RuntimeError(
                f"lupine: {message}: this process runs torch {self.host_version} and the "
                f"worker torch {self.worker_version}; the worker needs a torch of the same "
                "major.minor release"
            )
        raise RuntimeError(f"lupine worker: {message}")

    def call(self, kind: int, meta: bytes, body: bytes, extra: Any = b"", into: Any = None) -> bytes:
        status, payload = self.client.call(kind, meta, body, extra, into)
        if status != wire.OK:
            self.raise_status(status, payload)
        return payload

    def op_state(self, op: Any) -> _Op:
        st = self.ops.get(op)
        if st is None:
            st = self.ops[op] = _Op(op)
        return st

    # --- the fallback ----------------------------------------------------

    def fallback(self, op: Any, *args: Any, **kwargs: Any) -> Any:
        st = self.op_state(op)
        started = time.perf_counter_ns() if self.profile_path else 0
        body, table, big_cpu = self.encoder().encode((st.name, args, kwargs))
        descs = [self.describe(t) for t in table]
        key = None
        if not big_cpu and not st.sync and not (st.bool_index and _has_bool_index(args[1])):
            key = self.plan_key(st, body, descs)
            plan = st.plans.get(key)
            if plan is not None:
                result = self.replay(st, plan, table, descs, body, args)
                if started:
                    self.record(st.name, 0, time.perf_counter_ns() - started)
                if self.trace:
                    print(f"lupine-torch replay {st.name}", file=sys.stderr)
                return result
        if self.trace:
            print(f"lupine-torch sync {st.name}", file=sys.stderr)
        result, plan = self.run_sync(st, table, descs, body, args, key is not None)
        if key is not None and plan is not None:
            st.plans[key] = plan
        if started:
            self.record(st.name, 1, time.perf_counter_ns() - started)
        return result

    def plan_key(self, st: _Op, body: bytes, descs: list[wire.Desc]) -> Any:
        if st.memo:
            # The CUDA _fused_sdp_choice depends on dtype, head count, head
            # size and the last stride, not on batch or sequence length, so
            # one answer serves a whole generation.
            shapes = []
            for _, _, device, dtype, sizes, strides, _ in descs:
                ndim = len(sizes)
                kept = tuple(s if i + 1 == ndim or (ndim == 4 and i == 1) else 0 for i, s in enumerate(sizes))
                shapes.append((device, dtype, kept, 1 if strides and strides[-1] == 1 else 0))
            return (body, tuple(shapes))
        ids: dict[int, int] = {}
        shapes = tuple(
            (ids.setdefault(handle, len(ids)), device, dtype, sizes, strides)
            for handle, _, device, dtype, sizes, strides, _ in descs
        )
        return (None if st.inplace else body, shapes)

    def result_device(self, table: list[torch.Tensor], args: tuple[Any, ...]) -> torch.device:
        if table:
            return table[0].device
        for a in args:
            if isinstance(a, torch.device) and a.type in self.types:
                return a if a.index is not None else self.device(self.C.current_device())
        return self.device(self.C.current_device())

    def build(self, out: Any, table: list[torch.Tensor], groups: list[Any], plan: _Plan, device: torch.device) -> torch.Tensor:
        kind, index, same, dtype, sizes, strides, offset = out
        if kind == wire.INPUT:
            t = table[index]
            if not same:
                torch.ops.lupine.set_metadata_(t, sizes, strides, t.storage_offset() + offset)
            return t
        if kind == wire.INPUT_STORAGE:
            base = table[index]
            return torch.ops.lupine.view_over(base, dtype, sizes, strides, base.storage_offset() + offset)
        nbytes, direct = plan.groups[index]
        if direct:
            return torch.empty_strided(sizes, strides, dtype=dtype, device=device)
        base = groups[index]
        if base is None:
            base = groups[index] = torch.empty(nbytes, dtype=torch.uint8, device=device)
        return torch.ops.lupine.view_over(base, dtype, sizes, strides, offset)

    def replay(self, st: _Op, plan: _Plan, table: list[torch.Tensor], descs: list[wire.Desc], body: bytes, args: tuple[Any, ...]) -> Any:
        device = self.result_device(table, args) if plan.groups else None
        groups: list[Any] = [None] * len(plan.groups)
        results = []
        for kind, item in plan.rets:
            if kind == _TENSOR:
                results.append(self.build(item, table, groups, plan, device))
            elif kind == _LIST:
                results.append([self.build(o, table, groups, plan, device) for o in item])
            else:
                results.append(item)
        if st.returns == 0:
            value = None
        elif st.returns == 1:
            value = results[0]
        else:
            value = tuple(results)
        if plan.needs_rpc:
            handles = [t.untyped_storage().data_ptr() >> ext.HANDLE_SHIFT for t in wire.result_tensors(value)]
            self.client.send(wire.OP_PLAN, self.plan_meta(plan.number, descs, handles), b"" if body == plan.body else body)
        return value

    def run_sync(self, st: _Op, table: list[torch.Tensor], descs: list[wire.Desc], body: bytes, args: tuple[Any, ...], cacheable: bool) -> tuple[Any, _Plan | None]:
        # The worker files this call as the plan's template under its number.
        number = 0
        if cacheable:
            self.numbered += 1
            number = self.numbered
        payload = self.call(wire.OP_SYNC, self.meta(descs, plan=number), body)
        aliases, rdescs, rbody = pickle.loads(payload)
        device = self.result_device(table, args)
        outputs: dict[int, Any] = {}
        rtable: list[torch.Tensor] = []
        group_of: dict[int, int] = {}
        groups: list[tuple[int, bool]] = []
        bases: list[torch.Tensor] = []
        for (handle, nbytes, _, dtype_name, sizes, strides, offset), (akind, aindex) in zip(rdescs, aliases):
            dtype = wire.dtype_of(dtype_name)
            if akind == wire.INPUT:
                t = table[aindex]
                rel = offset - t.storage_offset()
                same = rel == 0 and sizes == tuple(t.shape) and strides == t.stride()
                if not same:
                    torch.ops.lupine.set_metadata_(t, sizes, strides, offset)
                out = (wire.INPUT, aindex, same, dtype, sizes, strides, rel)
            elif akind == wire.INPUT_STORAGE:
                base = table[aindex]
                t = torch.ops.lupine.view_over(base, dtype, sizes, strides, offset)
                out = (wire.INPUT_STORAGE, aindex, False, dtype, sizes, strides, offset - base.storage_offset())
            else:
                g = group_of.get(handle)
                if g is None:
                    g = group_of[handle] = len(groups)
                    groups.append((nbytes, False))
                    bases.append(torch.ops.lupine.storage(handle, nbytes, device))
                t = torch.ops.lupine.view_over(bases[g], dtype, sizes, strides, offset)
                out = (wire.FRESH, g, False, dtype, sizes, strides, offset)
            outputs[id(t)] = out
            rtable.append(t)
        value = wire.loads(rbody, rtable)
        # A group is direct when exactly one output covers it from offset 0.
        members: dict[int, list[Any]] = {}
        for out in outputs.values():
            if out[0] == wire.FRESH:
                members.setdefault(out[1], []).append(out)
        for g, outs in members.items():
            if len(outs) == 1:
                _, _, _, dtype, sizes, strides, offset = outs[0]
                if offset == 0 and _storage_nbytes(sizes, strides, dtype) == groups[g][0]:
                    groups[g] = (groups[g][0], True)
        rets: list[Any] = []
        for v in (value,) if st.returns == 1 else (value if st.returns else ()):
            if isinstance(v, torch.Tensor):
                rets.append((_TENSOR, outputs[id(v)]))
            elif isinstance(v, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in v):
                rets.append((_LIST, [outputs[id(t)] for t in v]))
            elif v is None or type(v) in (bool, int, float):
                rets.append((_VALUE, v))
            else:
                return value, None
        return value, _Plan(rets, groups, st.mutates or bool(groups), number, body)

    # --- copies, scalars, control ---------------------------------------

    def copy_to_device(self, self_: torch.Tensor, src: torch.Tensor) -> None:
        payload = src
        if src.dtype != self_.dtype or src.shape != self_.shape or not src.is_contiguous():
            payload = torch.empty(self_.shape, dtype=self_.dtype)
            payload.copy_(src)
        self.client.send(wire.COPY_FROM_HOST, self.meta([self.describe(self_)]), b"", wire.cpu_buffer(payload))

    def copy_to_host(self, self_: torch.Tensor, src: torch.Tensor) -> None:
        direct = self_.is_contiguous() and self_.shape == src.shape
        target = self_ if direct else torch.empty(src.shape, dtype=self_.dtype)
        into = wire.cpu_buffer(target) if target.numel() else None
        self.call(wire.COPY_TO_HOST, self.meta([self.describe(src)], wire.dtype_name(self_.dtype)), b"", into=into)
        if not direct:
            self_.copy_(target)

    def copy_(self, self_: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        if self_.numel() == 0:
            return self_
        self_ours = self.is_ours(self_)
        src_ours = self.is_ours(src)
        if self_ours and src_ours:
            self.fallback(torch.ops.aten.copy_.default, self_, src, False)
        elif self_ours and src.device.type == "cpu":
            self.copy_to_device(self_, src)
        elif self_.device.type == "cpu" and src_ours:
            self.copy_to_host(self_, src)
        else:
            raise RuntimeError(f"lupine: unsupported copy from {src.device} to {self_.device}")
        return self_

    def _copy_from(self, self_: torch.Tensor, dst: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        self.copy_(dst, self_, non_blocking)
        return dst

    def _copy_from_and_resize(self, self_: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        dst.resize_(self_.shape)
        self.copy_(dst, self_, False)
        return dst

    def _local_scalar_dense(self, self_: torch.Tensor) -> Any:
        cpu = torch.empty(self_.shape, dtype=self_.dtype)
        self.copy_to_host(cpu, self_)
        return cpu.item()

    def synchronize(self) -> None:
        self.call(wire.SYNC, self.meta([]), b"")

    def exec_(self, code: str) -> None:
        self.client.send(wire.EXEC, self.meta([]), code.encode())

    def eval_(self, code: str) -> Any:
        return json.loads(self.call(wire.EVAL, self.meta([]), code.encode()))

    def release(self, handle: int) -> None:
        self.frees.append(handle)

    def record(self, name: str, phase: int, nanoseconds: int) -> None:
        entry = self.profile.get(name)
        if entry is None:
            entry = self.profile[name] = [0, 0, 0]
        entry[0] += 1
        entry[1 + phase] += nanoseconds

    def dump_profile(self) -> None:
        if not self.profile_path:
            return
        with open(self.profile_path, "w") as f:
            f.write(f"__messages__\t{self.client.messages}\t{self.client.bytes}\t{self.client.writes}\n")
            for name, (count, replay, sync) in self.profile.items():
                f.write(f"{name}\t{count}\t{replay}\t{sync}\n")

    def close(self) -> None:
        if not self.client.closed:
            try:
                self.synchronize()
            except RuntimeError:
                # A worker that already failed cannot drain; closing the
                # connection below is what releases it.
                pass
        self.client.close()


def _has_bool_index(indices: Any) -> bool:
    return any(t is not None and t.dtype in (torch.bool, torch.uint8) for t in indices)


def _storage_nbytes(sizes: tuple[int, ...], strides: tuple[int, ...], dtype: torch.dtype) -> int:
    if any(s == 0 for s in sizes):
        return 0
    return (1 + sum((s - 1) * st for s, st in zip(sizes, strides))) * dtype.itemsize


_backend: Backend | None = None
_libraries: list[Library] = []


def current() -> Backend:
    if _backend is None:
        raise RuntimeError("lupine: the torch backend is not connected")
    return _backend


def install(client: transport.Client, dual: bool, C: Any) -> Backend:
    """Registers the fallback and the Python kernels for the backend's key(s)."""

    global _backend
    backend = Backend(client, dual, C)
    _backend = backend
    fallback = backend.fallback

    def bound(op: Any) -> Any:
        return lambda *args, **kwargs: fallback(op, *args, **kwargs)

    for key in ("PrivateUse1", "CUDA") if dual else ("PrivateUse1",):
        universal = Library("_", "IMPL")
        universal.fallback(fallback, key)
        aten = Library("aten", "IMPL")
        aten.impl("copy_", backend.copy_, key)
        aten.impl("_copy_from", backend._copy_from, key)
        aten.impl("_copy_from_and_resize", backend._copy_from_and_resize, key)
        aten.impl("_local_scalar_dense", backend._local_scalar_dense, key)
        for name in _WHOLE:
            packet = getattr(torch.ops.aten, name, None)
            if packet is not None:
                aten.impl(name, bound(packet.default), key)
        for qualified in torch._C._dispatch_get_all_op_names():
            if qualified.startswith("aten::_foreach_"):
                name = qualified[6:]
                base, _, overload = name.partition(".")
                aten.impl(name, bound(getattr(getattr(torch.ops.aten, base), overload or "default")), key)
        _libraries.extend((universal, aten))
    atexit.register(backend.dump_profile)
    return backend
