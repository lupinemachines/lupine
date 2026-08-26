# /// script
# requires-python = ">=3.10"
# dependencies = ["clang-format==18.1.3", "cxxheaderparser==1.9.2"]
# ///
from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function, Array
from typing import Optional, Union
from dataclasses import dataclass
from string import Template
import argparse
import io
import os
import glob
import re
import subprocess
import zlib
from ops import (
    NullableOperation,
    ArrayOperation,
    InOutCountOperation,
    NullableArrayOperation,
    DeepStructOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    DereferenceOperation,
    Operation,
    OwnerAnnotation,
    RetainAnnotation,
    ReleaseAnnotation,
    ParentAnnotation,
    CrossServerCopyAnnotation,
    FunctionAnnotationMetadata,
    RoutingFallbackAnnotation,
    SynchronizeAnnotation,
)

# this table is manually generated from the cuda.h headers
MANUAL_REMAPPINGS = [
    ("cuDeviceTotalMem", "cuDeviceTotalMem_v2"),
    ("cuDeviceGetUuid", "cuDeviceGetUuid_v2"),
    ("cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"),
    ("cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"),
    ("cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2"),
    ("cuCtxDestroy", "cuCtxDestroy_v2"),
    ("cuCtxPopCurrent", "cuCtxPopCurrent_v2"),
    ("cuCtxPushCurrent", "cuCtxPushCurrent_v2"),
    ("cuModuleGetGlobal", "cuModuleGetGlobal_v2"),
    ("cuMemAlloc", "cuMemAlloc_v2"),
    ("cuMemAllocPitch", "cuMemAllocPitch_v2"),
    ("cuMemcpyHtoD", "cuMemcpyHtoD_v2"),
    ("cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"),
    ("cuMemcpyDtoH", "cuMemcpyDtoH_v2"),
    ("cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"),
    ("cuMemcpyDtoD", "cuMemcpyDtoD_v2"),
    ("cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2"),
    ("cuMemsetD8", "cuMemsetD8_v2"),
    ("cuMemsetD2D8", "cuMemsetD2D8_v2"),
    ("cuMemsetD2D16", "cuMemsetD2D16_v2"),
    ("cuMemsetD2D32", "cuMemsetD2D32_v2"),
    ("cuIpcOpenMemHandle", "cuIpcOpenMemHandle_v2"),
    ("cuStreamBeginCapture", "cuStreamBeginCapture_v2"),
    ("cuGraphExecUpdate", "cuGraphExecUpdate_v2"),
    ("cuMemcpy_ptds", "cuMemcpy"),
    ("cuMemcpyAsync_ptsz", "cuMemcpyAsync"),
    ("cuMemcpyPeer_ptds", "cuMemcpyPeer"),
    ("cuMemcpyPeerAsync_ptsz", "cuMemcpyPeerAsync"),
    ("cuMemcpy3DPeer_ptds", "cuMemcpy3DPeer"),
    ("cuMemcpy3DPeerAsync_ptsz", "cuMemcpy3DPeerAsync"),
    ("cuMemPrefetchAsync_ptsz", "cuMemPrefetchAsync"),
    ("cuMemsetD8Async_ptsz", "cuMemsetD8Async"),
    ("cuMemsetD16Async_ptsz", "cuMemsetD16Async"),
    ("cuMemsetD32Async_ptsz", "cuMemsetD32Async"),
    ("cuMemsetD2D8Async_ptsz", "cuMemsetD2D8Async"),
    ("cuMemsetD2D16Async_ptsz", "cuMemsetD2D16Async"),
    ("cuMemsetD2D32Async_ptsz", "cuMemsetD2D32Async"),
    ("cuStreamGetPriority_ptsz", "cuStreamGetPriority"),
    ("cuStreamGetId_ptsz", "cuStreamGetId"),
    ("cuStreamGetFlags_ptsz", "cuStreamGetFlags"),
    ("cuStreamGetCtx_ptsz", "cuStreamGetCtx"),
    ("cuStreamWaitEvent_ptsz", "cuStreamWaitEvent"),
    ("cuStreamEndCapture_ptsz", "cuStreamEndCapture"),
    ("cuStreamIsCapturing_ptsz", "cuStreamIsCapturing"),
    ("cuStreamUpdateCaptureDependencies_ptsz", "cuStreamUpdateCaptureDependencies"),
    ("cuStreamAddCallback_ptsz", "cuStreamAddCallback"),
    ("cuStreamAttachMemAsync_ptsz", "cuStreamAttachMemAsync"),
    ("cuStreamQuery_ptsz", "cuStreamQuery"),
    ("cuStreamSynchronize_ptsz", "cuStreamSynchronize"),
    ("cuEventRecord_ptsz", "cuEventRecord"),
    ("cuEventRecordWithFlags_ptsz", "cuEventRecordWithFlags"),
    ("cuLaunchKernel_ptsz", "cuLaunchKernel"),
    ("cuLaunchKernelEx_ptsz", "cuLaunchKernelEx"),
    ("cuLaunchHostFunc_ptsz", "cuLaunchHostFunc"),
    ("cuGraphicsMapResources_ptsz", "cuGraphicsMapResources"),
    ("cuGraphicsUnmapResources_ptsz", "cuGraphicsUnmapResources"),
    ("cuSignalExternalSemaphoresAsync_ptsz", "cuSignalExternalSemaphoresAsync"),
    ("cuWaitExternalSemaphoresAsync_ptsz", "cuWaitExternalSemaphoresAsync"),
    ("cuGraphInstantiateWithParams_ptsz", "cuGraphInstantiateWithParams"),
    ("cuGraphUpload_ptsz", "cuGraphUpload"),
    ("cuGraphLaunch_ptsz", "cuGraphLaunch"),
    ("cuStreamCopyAttributes_ptsz", "cuStreamCopyAttributes"),
    ("cuStreamGetAttribute_ptsz", "cuStreamGetAttribute"),
    ("cuStreamSetAttribute_ptsz", "cuStreamSetAttribute"),
    ("cuMemMapArrayAsync_ptsz", "cuMemMapArrayAsync"),
    ("cuMemFreeAsync_ptsz", "cuMemFreeAsync"),
    ("cuMemAllocAsync_ptsz", "cuMemAllocAsync"),
    ("cuMemAllocFromPoolAsync_ptsz", "cuMemAllocFromPoolAsync"),
]

KERNEL_PARAM_LAYOUT_INVALIDATORS = {
    "cuLibraryUnload",
    "cuModuleUnload",
}

MANUAL_REMAPPING_GUARDS = {
    "cuGraphExecUpdate": "CUDA_VERSION >= 12000",
}

NVML_RPC_FUNCTIONS = [
    "nvmlInit_v2",
    "nvmlInitWithFlags",
    "nvmlShutdown",
    "nvmlSystemGetDriverVersion",
    "nvmlSystemGetNVMLVersion",
    "nvmlSystemGetCudaDriverVersion",
    "nvmlSystemGetCudaDriverVersion_v2",
    "nvmlDeviceGetCount_v2",
    "nvmlDeviceGetHandleByIndex_v2",
    "nvmlDeviceGetHandleByUUID",
    "nvmlDeviceGetHandleByPciBusId_v2",
    "nvmlDeviceGetName",
    "nvmlDeviceGetUUID",
    "nvmlDeviceGetIndex",
    "nvmlDeviceGetMinorNumber",
    "nvmlDeviceGetPciInfo_v3",
    "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetUtilizationRates",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetPowerManagementLimit",
    "nvmlDeviceGetClockInfo",
    "nvmlDeviceGetMaxClockInfo",
    "nvmlDeviceGetPerformanceState",
    "nvmlDeviceGetComputeMode",
    "nvmlDeviceGetPersistenceMode",
    "nvmlDeviceGetFanSpeed",
    "nvmlDeviceGetBrand",
    "nvmlDeviceGetVbiosVersion",
    "nvmlDeviceGetSerial",
    "nvmlDeviceGetBoardPartNumber",
    "nvmlDeviceGetDisplayMode",
    "nvmlDeviceGetDisplayActive",
    "nvmlDeviceGetCurrPcieLinkGeneration",
    "nvmlDeviceGetCurrPcieLinkWidth",
    "nvmlDeviceGetMaxPcieLinkGeneration",
    "nvmlDeviceGetMaxPcieLinkWidth",
    "nvmlDeviceGetPcieThroughput",
    "nvmlDeviceGetPcieReplayCounter",
    "nvmlDeviceGetComputeRunningProcesses",
    "nvmlDeviceGetComputeRunningProcesses_v2",
    "nvmlDeviceGetGraphicsRunningProcesses",
    "nvmlDeviceGetGraphicsRunningProcesses_v2",
    "nvmlDeviceGetMPSComputeRunningProcesses",
    "nvmlDeviceGetMPSComputeRunningProcesses_v2",
    "nvmlEventSetCreate",
    "nvmlEventSetFree",
    "nvmlEventSetWait_v2",
    "nvmlDeviceRegisterEvents",
    "nvmlDeviceGetMaxMigDeviceCount",
    "nvmlDeviceGetTotalEccErrors",
    "nvmlDeviceGetDetailedEccErrors",
    "nvmlDeviceGetMemoryErrorCounter",
    "nvmlDeviceGetEccMode",
    "nvmlDeviceGetTemperatureV",
    "nvmlDeviceGetEnforcedPowerLimit",
    "nvmlDeviceGetMemoryInfo_v2",
    "nvmlDeviceGetMigMode",
    "nvmlDeviceGetVirtualizationMode",
    "nvmlDeviceIsMigDeviceHandle",
    "nvmlDeviceGetNvLinkRemoteDeviceType",
    "nvmlDeviceGetNvLinkRemotePciInfo_v2",
    "nvmlDeviceGetCudaComputeCapability",
]

PRIVATE_RPC_FUNCTIONS = [
    "cuGetExportTableMetadata",
    "cuGraphAddNode_v2",
    "cuGraphConditionalHandleCreate",
    "cuPrivateGetModuleNode",
    "cuStreamBeginCaptureToGraph",
    "cuStreamGetCaptureInfo_v3",
    "lupineDeviceSnapshot",
    "lupineEventQueryBatch",
    "lupineFunctionAttributeSnapshot",
    "lupineFunctionParamLayoutSnapshot",
    "lupineLibraryAttributeSnapshot",
    "lupineLibrarySnapshot",
    "lupineManagedHostFlush",
]

REGISTRY_CPP_TEMPLATE = Template(
    r'''#include "rpc_server.h"

#ifdef LUPINE_BUILD_CUDA_BACKEND
#include <cuda.h>
#endif
#include "gen_rpc_ids.h"

// clang-format off
#define LUPINE_CUDA_RPC_HANDLERS(HANDLER) \
$cuda_registry_entries
#define LUPINE_NVML_RPC_HANDLERS(HANDLER) \
$nvml_registry_entries
// clang-format on

#define LUPINE_DECLARE_HANDLER(operation, handler, backend)                    \
  int handler(conn_t *conn);
#ifdef LUPINE_BUILD_CUDA_BACKEND
LUPINE_CUDA_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$cuda_guarded_declarations
#endif
#ifdef LUPINE_BUILD_NVML_BACKEND
LUPINE_NVML_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$nvml_guarded_declarations
#endif
#undef LUPINE_DECLARE_HANDLER

const rpc_handler_registry &lupine_rpc_handlers() {
#define LUPINE_REGISTER_HANDLER(operation, handler, backend)                    \
  {operation, {handler, backend}},
  // clang-format treats registrations separated by preprocessor guards as a
  // single continued expression, adding another indentation level per guard.
  // clang-format off
  static const rpc_handler_registry handlers = {
#ifdef LUPINE_BUILD_CUDA_BACKEND
      LUPINE_CUDA_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$cuda_guarded_handlers
#endif
#ifdef LUPINE_BUILD_NVML_BACKEND
      LUPINE_NVML_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$nvml_guarded_handlers
#endif
  };
  // clang-format on
#undef LUPINE_REGISTER_HANDLER
  return handlers;
}

#undef LUPINE_CUDA_RPC_HANDLERS
#undef LUPINE_NVML_RPC_HANDLERS
'''
)


@dataclass(frozen=True)
class ServerBinding:
    name: str
    backend: str
    handler: str
    guard: Optional[str] = None

    @property
    def backend_symbol(self) -> str:
        return SERVER_BACKENDS[self.backend]


SERVER_BACKENDS = {
    "CUDA": "rpc_backend::cuda",
    "NVML": "rpc_backend::nvml",
}


def annotation_directives(annotation: str) -> list[str]:
    directives = []
    for line in (annotation or "").splitlines():
        line = line.strip().lstrip("*").strip()
        if line.startswith("@"):
            directives.append(line)
    return directives


def parse_server_binding(name: str, annotation: str) -> Optional[ServerBinding]:
    backend = None
    handler = None
    guard = None
    for directive in annotation_directives(annotation):
        parts = directive.split(maxsplit=2)
        if parts[0] == "@server":
            if backend is not None or len(parts) < 2:
                raise RuntimeError(
                    f"Invalid @server annotation for {name}"
                )
            backend = parts[1].upper()
            if backend not in SERVER_BACKENDS:
                raise RuntimeError(f"Unknown RPC server backend {backend}")
            if len(parts) == 3:
                handler = parts[2]
        elif parts[0] == "@guard":
            if guard is not None or len(parts) < 2:
                raise RuntimeError(f"Invalid @guard annotation for {name}")
            guard = directive.removeprefix("@guard").strip()

    if backend is None:
        return None

    if handler is None:
        handler = "handle_" + name
    return ServerBinding(name, backend, handler, guard)


def collect_server_bindings(path: str) -> dict[str, ServerBinding]:
    bindings = {}
    with open(path) as annotations_file:
        source = annotations_file.read()
    for match in re.finditer(r"/\*\*(.*?)\*/\s*([^;{]*?\()", source, re.DOTALL):
        annotation, declaration = match.groups()
        name_match = re.search(r"([A-Za-z_]\w*)\s*\($", declaration)
        if name_match is None:
            continue
        binding = parse_server_binding(name_match.group(1), annotation)
        if binding is None:
            continue
        previous = bindings.get(binding.name)
        if previous is not None and previous != binding:
            raise RuntimeError(f"Conflicting @server annotations for {binding.name}")
        bindings[binding.name] = binding
    return bindings


def rpc_id(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def annotated_rpc_names(annotations: ParsedData) -> list[str]:
    names: set[str] = set()
    for function in annotations.namespace.functions:
        name = function.name.format()
        if len(name) > 2 and name.startswith("cu") and name[2].isupper():
            names.add(name)
    return sorted(names)


SKIP_FUNCTIONS = {
    "cuStreamUpdateCaptureDependencies_v2",
    "cuGraphGetEdges_v2",
    "cuGraphNodeGetDependencies_v2",
    "cuGraphNodeGetDependentNodes_v2",
    "cuGraphAddDependencies_v2",
    "cuGraphRemoveDependencies_v2",
}



def annotation_param(params: list[Parameter], name: str) -> Parameter:
    try:
        return next(p for p in params if p.name == name)
    except StopIteration:
        raise NotImplementedError(f"Parameter {name} not found")


def infer_routing_key(
    params: list[Parameter],
) -> tuple[Optional[str], Optional[Parameter]]:
    for param in params:
        if isinstance(param.type, (Pointer, Array)):
            continue
        type_name = param.type.format().replace("const ", "").strip()
        if type_name == "nvmlDevice_t":
            return "NVML_DEVICE", param
        if type_name == "CUdevice":
            return "DEVICE", param
        if type_name == "CUcontext":
            return "CONTEXT", param
        if type_name == "CUmodule":
            return "MODULE", param
        if type_name == "CUlibrary":
            return "LIBRARY", param
        if type_name == "CUfunction":
            return "FUNCTION", param
        if type_name == "CUstream":
            return "STREAM", param
        if type_name == "CUevent":
            return "EVENT", param
        if type_name == "CUmemoryPool":
            return "MEMORY_POOL", param
        if type_name == "CUgraph":
            return "GRAPH", param
        if type_name == "CUgraphNode":
            return "GRAPH_NODE", param
        if type_name == "CUgraphExec":
            return "GRAPH_EXEC", param
        if type_name == "CUdeviceptr":
            return "DEVICEPTR", param
    return None, None


# Parses a function annotation into marshalling operations and metadata.
def parse_annotation(
    annotation: str, params: list[Parameter]
) -> FunctionAnnotationMetadata:
    operations: list[Operation] = []
    metadata = FunctionAnnotationMetadata(operations=operations)
    # @deeparray <param> <array_member> <count_member> entries, grouped by the
    # struct-pointer param they describe (see DeepStructOperation).
    deep_arrays: dict[str, list[tuple[str, str]]] = {}

    if not annotation:
        metadata.routing_kind, metadata.routing_parameter = infer_routing_key(params)
        return metadata
    for line in annotation.split("\n"):
        # Disabled annotations can apply to client generation, server
        # generation, or both. Bare @disabled keeps the historical behavior
        # by setting both scoped flags.
        if "@disabled" in line or "@DISABLED" in line:
            disabled_parts = line.lower().lstrip(" *").split()
            scope = disabled_parts[1] if len(disabled_parts) > 1 else "both"
            if scope == "client":
                metadata.disabled_client = True
                continue
            elif scope == "server":
                metadata.disabled_server = True
                continue
            else:
                metadata.disabled_client = True
                metadata.disabled_server = True
                return metadata
        if line.startswith("/**"):
            continue
        if line.startswith("*/"):
            continue
        if line.startswith("*"):
            line = line[2:]
        if line.strip().startswith("@async"):
            metadata.async_fire_forget = True
            continue
        if line.strip().startswith("@synchronize"):
            parts = line.split()
            options = set(parts[1:])
            unknown = options - {"DEFERRED_DTOH", "STDOUT"}
            if unknown:
                raise NotImplementedError(
                    "Unknown @synchronize option(s): " + ", ".join(sorted(unknown))
                )
            metadata.synchronize = SynchronizeAnnotation(
                deferred_dtoh="DEFERRED_DTOH" in options,
                stdout="STDOUT" in options,
            )
            continue
        if line.strip().startswith("@guard"):
            guard = line.strip().removeprefix("@guard").strip()
            if not guard or metadata.guard is not None:
                raise RuntimeError("Invalid @guard annotation")
            metadata.guard = guard
            continue
        if line.strip() == "@hidden":
            continue
        if line.startswith("@server"):
            continue
        if line.startswith("@routingkey"):
            parts = line.split()
            if len(parts) < 2:
                continue
            metadata.routing_kind = parts[1].upper()
            if len(parts) >= 3:
                metadata.routing_parameter = annotation_param(params, parts[2])
            continue
        if line.startswith("@routingfallback"):
            parts = line.split()
            if len(parts) < 3:
                continue
            metadata.routing_fallback = RoutingFallbackAnnotation(
                kind=parts[1].upper(),
                parameter=annotation_param(params, parts[2]),
            )
            continue
        if line.startswith("@recordowner"):
            parts = line.split()
            if len(parts) < 3:
                continue
            param = annotation_param(params, parts[2])
            metadata.record_owners.append(OwnerAnnotation(parts[1].upper(), param))
            continue
        if line.startswith("@retain"):
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError("@retain requires an output parameter and a handle")
            metadata.retains.append(
                RetainAnnotation(
                    parameter=annotation_param(params, parts[1]),
                    handle=annotation_param(params, parts[2]),
                )
            )
            continue
        if line.startswith("@release"):
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError("@release requires a handle kind and parameter")
            metadata.releases.append(
                ReleaseAnnotation(
                    kind=parts[1].upper(),
                    parameter=annotation_param(params, parts[2]),
                )
            )
            continue
        if line.startswith("@recordparent"):
            parts = line.split()
            if len(parts) != 4:
                raise RuntimeError(
                    "@recordparent requires a parent kind, child, and parent"
                )
            metadata.parents.append(
                ParentAnnotation(
                    kind=parts[1].upper(),
                    child=annotation_param(params, parts[2]),
                    parent=annotation_param(params, parts[3]),
                )
            )
            continue
        if line.startswith("@crossservercopy"):
            parts = line.split()
            if len(parts) < 4:
                continue
            stream_arg = next(
                (arg for arg in parts[4:] if arg.startswith("STREAM:")), None
            )
            metadata.cross_server_copy = CrossServerCopyAnnotation(
                dst=annotation_param(params, parts[1]),
                src=annotation_param(params, parts[2]),
                bytes=annotation_param(params, parts[3]),
                stream=(
                    annotation_param(params, stream_arg.split(":", 1)[1])
                    if stream_arg is not None
                    else None
                ),
                async_="ASYNC" in parts[4:],
            )
            continue
        if line.startswith("@deeparray"):
            # @deeparray <param> <array_member> <count_member>
            parts = line.split()
            if len(parts) < 4:
                continue
            deep_arrays.setdefault(parts[1], []).append((parts[2], parts[3]))
            continue
        if line.startswith("@param"):
            parts = line.split()

            if len(parts) < 3:
                continue
            param = annotation_param(params, parts[1])
            args = parts[3:]
            send = parts[2] == "SEND_ONLY" or parts[2] == "SEND_RECV"
            recv = parts[2] == "RECV_ONLY" or parts[2] == "SEND_RECV"

            # if there's a length or size arg, use the type, otherwise use the ptr_to type
            length_arg = next((arg for arg in args if arg.startswith("LENGTH:")), None)

            if isinstance(param.type, Pointer):
                if param.type.ptr_to.const:
                    recv = False

                size_arg = next((arg for arg in args if arg.startswith("SIZE:")), None)
                iter_arg = next((arg for arg in args if arg.startswith("ITER:")), None)
                null_terminated = "NULL_TERMINATED" in args
                nullable = "NULLABLE" in args
                deref = "DEREF" in args
                recv_on_error = "ON_ERROR" in args

                # NULLABLE composes with LENGTH (an optional out-array
                # sized by an in/out count); every other combination is
                # mutually exclusive.
                if (
                    sum([bool(length_arg), bool(size_arg), null_terminated, nullable])
                    > 1
                ) and not (nullable and length_arg and not size_arg and not null_terminated):
                    raise NotImplementedError(
                        "Only one of LENGTH, SIZE, NULL_TERMINATED, or NULLABLE can be specified (except NULLABLE LENGTH)"
                    )
                if recv_on_error and not (
                    nullable and length_arg and recv and not send
                ):
                    raise NotImplementedError(
                        "ON_ERROR requires a RECV_ONLY NULLABLE LENGTH out-array"
                    )

                if deref:
                    operations.append(
                        DereferenceOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        )
                    )
                elif length_arg:
                    # if it has a length, it's an array operation with variable length
                    length_param = next(
                        p for p in params if p.name == length_arg.split(":")[1]
                    )
                    if nullable:
                        # NULLABLE LENGTH: an optional out-array sized by an
                        # in/out count param (the cuGraphGetNodes query
                        # pattern); linked to its count in the post-pass below.
                        if send or not recv:
                            raise NotImplementedError(
                                "NULLABLE LENGTH requires a RECV_ONLY out-array"
                            )
                        operations.append(
                            NullableArrayOperation(
                                parameter=param,
                                ptr=param.type,
                                count=length_param,
                                recv_on_error=recv_on_error,
                            )
                        )
                    else:
                        operations.append(
                            ArrayOperation(
                                send=send,
                                recv=recv,
                                parameter=param,
                                ptr=param.type,
                                length=length_param,
                                iter=False,
                                compressible="COMPRESSIBLE" in args,
                            )
                        )
                elif size_arg:
                    # if it has a size, it's an array operation with constant length
                    operations.append(
                        ArrayOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            length=int(size_arg.split(":")[1]),
                            iter=False
                        )
                    )
                elif iter_arg:
                    print(f"ITER FOUND!! {param}")
                    length_param = next(
                        p for p in params if p.name == iter_arg.split(":")[1]
                    )
                    operations.append(
                        ArrayOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            length=length_param,
                            iter=True
                        )
                    )
                elif null_terminated:
                    if recv:
                        if (
                            send
                            or not isinstance(param.type.ptr_to, Pointer)
                            or param.type.ptr_to.ptr_to.format() != "const char"
                        ):
                            raise NotImplementedError(
                                "received NULL_TERMINATED parameters must be "
                                "RECV_ONLY const char **"
                            )
                    operations.append(
                        NullTerminatedOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                        )
                    )
                elif nullable:
                    # if it's nullable, it's a nullable operation
                    operations.append(
                        NullableOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                        )
                    )
                else:
                    # otherwise, it's a pointer to a single value or another pointer
                    if recv:
                        if param.type.ptr_to.format() == "void":
                            raise NotImplementedError(
                                "Cannot dereference a void pointer"
                            )
                        # this is an out parameter so use the base type as the server declaration
                        operations.append(
                            DereferenceOperation(
                                send=send,
                                recv=recv,
                                parameter=param,
                                type_=param.type,
                            )
                        )
                    else:
                        # otherwise, treat it as an opaque type
                        operations.append(
                            OpaqueTypeOperation(
                                send=send,
                                recv=recv,
                                parameter=param,
                                type_=param.type,
                            )
                        )
            elif isinstance(param.type, Type):
                if param.type.const:
                    recv = False
                operations.append(
                    OpaqueTypeOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        type_=param.type,
                    )
                )
            elif isinstance(param.type, Array):
                length_param = next(
                    p for p in params if p.name == length_arg.split(":")[1]
                )
                if param.type.const:
                    recv = False
                operations.append(
                    ArrayOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                        length=length_param,
                    )
                )
            elif size_arg:
                # if it has a size, it's an array operation with constant length
                operations.append(
                    ArrayOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                        length=int(size_arg.split(":")[1]),
                    )
                )
            elif null_terminated:
                # if it's null terminated, it's a null terminated operation
                operations.append(
                    NullTerminatedOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                    )
                )
            elif nullable:
                # if it's nullable, it's a nullable operation
                operations.append(
                    NullableOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                    )
                )
            else:
                # otherwise, it's a pointer to a single value or another pointer
                if recv:
                    if param.type.ptr_to.format() == "void":
                        raise NotImplementedError("Cannot dereference a void pointer")
                    # this is an out parameter so use the base type as the server declaration
                    operations.append(
                        DereferenceOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        )
                    )
                else:
                    # otherwise, treat it as an opaque type
                    operations.append(
                        OpaqueTypeOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        )
                    )
        elif isinstance(param.type, Type):
            if param.type.const:
                recv = False
            operations.append(
                OpaqueTypeOperation(
                    send=send,
                    recv=recv,
                    parameter=param,
                    type_=param.type,
                )
            )
        elif isinstance(param.type, Array):
            length_param = next(p for p in params if p.name == length_arg.split(":")[1])
            if param.type.array_of.const:
                recv = False
            operations.append(
                ArrayOperation(
                    send=send,
                    recv=recv,
                    parameter=param,
                    ptr=param.type,
                    length=length_param,
                )
            )
        else:
            raise NotImplementedError("Unknown type")
    # Promote the count param of any optional out-array to an
    # InOutCountOperation. Several arrays may share one count (cuGraphGetEdges);
    # the first one is the anchor whose presence the client uses to decide
    # between a count-only query and a fill.
    optional_ops = [
        op for op in operations if isinstance(op, NullableArrayOperation)
    ]
    if optional_ops:
        anchors: dict[str, str] = {}
        for op in optional_ops:
            anchors.setdefault(op.count.name, op.parameter.name)
        for count_name, anchor in anchors.items():
            for i, op in enumerate(operations):
                if op.parameter.name == count_name and isinstance(
                    op, DereferenceOperation
                ):
                    operations[i] = InOutCountOperation(
                        send=True,
                        recv=True,
                        parameter=op.parameter,
                        anchor=anchor,
                    )
                    break
    # Promote any param with @deeparray entries to a DeepStructOperation,
    # inheriting the send/recv direction from its @param line.
    for pname, members in deep_arrays.items():
        for i, op in enumerate(operations):
            if op.parameter.name == pname:
                operations[i] = DeepStructOperation(
                    send=getattr(op, "send", True),
                    recv=getattr(op, "recv", False),
                    parameter=op.parameter,
                    ptr=op.parameter.type,
                    members=members,
                )
                break
    # An array is sized from another parameter, so that parameter has to be on
    # the wire before the array. Parameter order does not guarantee it, and the
    # server would otherwise size its buffer from an unread variable. Move each
    # length source ahead of the first array that depends on it; both sides walk
    # this same list, so they stay symmetric.
    for i, op in enumerate(operations):
        length = getattr(op, "length", None)
        if not isinstance(length, Parameter):
            continue
        source = next(
            (j for j, other in enumerate(operations)
             if other.parameter.name == length.name),
            None,
        )
        if source is not None and source > i:
            operations.insert(i, operations.pop(source))

    retained_names = set()
    for retain in metadata.retains:
        name = retain.parameter.name
        if name in retained_names:
            raise RuntimeError(f"Duplicate @retain for parameter {name}")
        retained_names.add(name)
        operation = next(
            (op for op in operations if op.parameter.name == name), None
        )
        if not isinstance(operation, NullTerminatedOperation) or not operation.recv:
            raise NotImplementedError(
                "@retain currently requires a RECV_ONLY NULL_TERMINATED parameter"
            )

    if metadata.routing_kind is None:
        metadata.routing_kind, metadata.routing_parameter = infer_routing_key(params)
    return metadata


def client_routing_key_expr(
    kind: Optional[str], param: Optional[Parameter], metadata: FunctionAnnotationMetadata
) -> str:
    if kind is None:
        return "lupine_route_for_default()"
    if kind == "CURRENT_CONTEXT":
        return "lupine_route_for_current_context()"
    if param is None:
        raise NotImplementedError(f"Routing key {kind} requires a parameter")
    name = param.name
    if kind == "DEVICE":
        return f"lupine_route_for_device(&{name})"
    if kind == "CONTEXT":
        return f"lupine_route_for_context({name})"
    if kind == "MODULE":
        return f"lupine_route_for_module({name})"
    if kind == "LIBRARY":
        return f"lupine_route_for_library({name})"
    if kind == "FUNCTION":
        if param.type.format() == "CUkernel":
            name = f"reinterpret_cast<CUfunction>({name})"
        return f"lupine_route_for_function({name})"
    if kind == "STREAM":
        if metadata.routing_fallback is not None:
            fallback = client_routing_key_expr(
                metadata.routing_fallback.kind,
                metadata.routing_fallback.parameter,
                metadata,
            )
            return f"({name} != nullptr ? lupine_route_for_stream({name}) : {fallback})"
        return f"({name} != nullptr ? lupine_route_for_stream({name}) : lupine_route_for_default())"
    if kind == "EVENT":
        return f"lupine_route_for_event({name})"
    if kind == "MEMORY_POOL":
        return f"lupine_route_for_memory_pool({name})"
    if kind == "GRAPH":
        return f"lupine_route_for_graph({name})"
    if kind == "GRAPH_NODE":
        return f"lupine_route_for_graph_node({name})"
    if kind == "GRAPH_EXEC":
        return f"lupine_route_for_graph_exec({name})"
    if kind == "DEVICEPTR":
        return f"lupine_route_for_deviceptr({name})"
    raise NotImplementedError(f"Unknown routing key kind: {kind}")


def client_routing_route_expr(metadata: FunctionAnnotationMetadata) -> str:
    return client_routing_key_expr(
        metadata.routing_kind, metadata.routing_parameter, metadata
    )


def client_call_args(function: Function, metadata: FunctionAnnotationMetadata) -> list[str]:
    return [param.name for param in function.parameters if param.name]


def write_client_rpc_write(f, operation: Operation, metadata: FunctionAnnotationMetadata):
    operation.client_rpc_write(f)


def client_record_owner_stmt(owner: OwnerAnnotation) -> str:
    kind = owner.kind
    name = owner.parameter.name
    value = f"*{name}" if isinstance(owner.parameter.type, Pointer) else name
    null_guard = f" && {name} != nullptr" if isinstance(owner.parameter.type, Pointer) else ""
    if kind == "CONTEXT":
        fn = "lupine_note_context_owner"
    elif kind == "MODULE":
        fn = "lupine_note_module_owner"
    elif kind == "LIBRARY":
        fn = "lupine_note_library_owner"
    elif kind == "FUNCTION":
        fn = "lupine_note_function_owner"
    elif kind == "STREAM":
        fn = "lupine_note_stream_owner"
    elif kind == "EVENT":
        fn = "lupine_note_event_owner"
    elif kind == "MEMORY_POOL":
        fn = "lupine_note_memory_pool_owner"
    elif kind == "GRAPH":
        fn = "lupine_note_graph_owner"
    elif kind == "GRAPH_NODE":
        fn = "lupine_note_graph_node_owner"
    elif kind == "GRAPH_EXEC":
        fn = "lupine_note_graph_exec_owner"
    elif kind == "DEVICEPTR":
        fn = "lupine_note_deviceptr_owner"
    else:
        raise NotImplementedError(f"Unknown owner kind: {kind}")
    return (
        f"    if (return_value == CUDA_SUCCESS{null_guard}) {{\n"
        f"        {fn}_route({value}, route);\n"
        "    }\n"
    )


def write_client_post_call(f, function: Function, metadata: FunctionAnnotationMetadata):
    if function.name.format() == "cuDriverGetVersion":
        f.write("    if (driverVersion != nullptr) {\n")
        f.write("        const char *override_version = getenv(\"LUPINE_DRIVER_VERSION_OVERRIDE\");\n")
        f.write("        if (override_version != nullptr) *driverVersion = atoi(override_version);\n")
        f.write("    }\n")

    for owner in metadata.record_owners:
        f.write(client_record_owner_stmt(owner))
    for parent in metadata.parents:
        if parent.kind != "LIBRARY":
            raise NotImplementedError(
                f"Unsupported recorded parent kind: {parent.kind}"
            )
        f.write(
            f"    if (return_value == CUDA_SUCCESS && "
            f"{parent.child.name} != nullptr) "
            f"lupine_record_library_module(*{parent.child.name}, "
            f"{parent.parent.name});\n"
        )
    for release in metadata.releases:
        if release.kind == "MODULE":
            release_fn = "lupine_release_module_retained_strings"
        elif release.kind == "LIBRARY":
            release_fn = "lupine_release_library_retained_strings"
        else:
            raise NotImplementedError(
                f"Unsupported retained-string release kind: {release.kind}"
            )
        f.write(
            f"    if (return_value == CUDA_SUCCESS) "
            f"{release_fn}({release.parameter.name});\n"
        )

    if function.name.format() == "cuMemAlloc_v2":
        f.write("    if (return_value == CUDA_SUCCESS && dptr != nullptr) lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);\n")
    if function.name.format() == "cuMemAllocPitch_v2":
        f.write("    if (return_value == CUDA_SUCCESS && dptr != nullptr) {\n")
        f.write("        size_t allocation_size = 0;\n")
        f.write("        if (pPitch != nullptr) allocation_size = (*pPitch) * Height;\n")
        f.write("        else allocation_size = WidthInBytes * Height;\n")
        f.write("        lupine_note_deviceptr_allocation_route(*dptr, allocation_size, route);\n")
        f.write("    }\n")
    if function.name.format() in {"cuMemAllocAsync", "cuMemAllocFromPoolAsync"}:
        f.write("    if (return_value == CUDA_SUCCESS && dptr != nullptr) lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);\n")
    if function.name.format() == "cuMemFreeAsync":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_forget_deviceptr_owner(dptr);\n")
    if function.name.format() == "cuStreamDestroy_v2":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_forget_stream_owner(hStream);\n")
    # Record the global's size so offset pointers into it route by range.
    if function.name.format() in {"cuModuleGetGlobal_v2", "cuLibraryGetGlobal", "cuLibraryGetManaged"}:
        f.write("    if (return_value == CUDA_SUCCESS && dptr != nullptr && bytes != nullptr) lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);\n")
    if metadata.synchronize:
        f.write("    if (return_value == CUDA_SUCCESS) return_value = lupine_sync_mapped_device_to_host();\n")

    if function.name.format() == "cuCtxDestroy_v2":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_forget_destroyed_context(ctx);\n")
    if function.name.format() == "cuCtxFromGreenCtx":
        f.write("    if (return_value == CUDA_SUCCESS && pContext != nullptr) lupine_mark_context_green(*pContext);\n")
    if function.name.format() == "cuGreenCtxDestroy":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_forget_destroyed_context(reinterpret_cast<CUcontext>(hCtx));\n")
    if function.name.format() in {
        "cuCtxDestroy_v2",
        "cuCtxDetach",
        "cuDevicePrimaryCtxRelease_v2",
        "cuDevicePrimaryCtxReset_v2",
    }:
        f.write("    if (return_value == CUDA_SUCCESS) lupine_invalidate_current_context_cache();\n")
    if function.name.format() in KERNEL_PARAM_LAYOUT_INVALIDATORS:
        f.write("    if (return_value == CUDA_SUCCESS) lupine_invalidate_function_caches();\n")
    if function.name.format() == "cuKernelSetAttribute":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_kernel_attribute_cache_erase(lupine_route_identity(route), kernel, (int)attrib, (int)dev);\n")
    if function.name.format() == "cuFuncSetAttribute":
        f.write("    if (return_value == CUDA_SUCCESS) {\n")
        f.write("        lupine_invalidate_kernel_attribute_cache();\n")
        f.write("        lupine_invalidate_function_attribute_cache();\n")
        f.write("    }\n")
    if function.name.format() == "cuModuleGetFunction":
        f.write("    if (return_value == CUDA_SUCCESS && hfunc != nullptr) return_value = lupine_record_module_function(*hfunc, hmod, name, route);\n")
    if function.name.format() == "cuLibraryGetKernel":
        f.write("    if (return_value == CUDA_SUCCESS && pKernel != nullptr) return_value = lupine_record_library_kernel(*pKernel, library, name, route);\n")


def error_const(return_type: str) -> str:
    if return_type == "nvmlReturn_t":
        return "NVML_ERROR_GPU_IS_LOST"
    if return_type == "CUresult":
        return "CUDA_ERROR_DEVICE_UNAVAILABLE"
    if return_type == "cudaError_t":
        return "cudaErrorDevicesUnavailable"
    if return_type == "cublasStatus_t":
        return "CUBLAS_STATUS_NOT_INITIALIZED"
    if return_type == "cudnnStatus_t":
        return "CUDNN_STATUS_NOT_INITIALIZED"
    if return_type == "size_t":
        return "size_t"
    if return_type == "const char*":
        return "const char*"
    if return_type == "void":
        return "void"
    if return_type == "struct cudaChannelFormatDesc":
        return "struct cudaChannelFormatDesc"
    raise NotImplementedError("Unknown return type: %s" % return_type)


def invalid_device_const(return_type: str) -> str:
    if return_type == "CUresult":
        return "CUDA_ERROR_INVALID_DEVICE"
    if return_type == "cudaError_t":
        return "cudaErrorInvalidDevice"
    raise NotImplementedError(
        "No invalid-device error for return type: %s" % return_type
    )


def invalid_argument_const(return_type: str) -> str:
    if return_type == "nvmlReturn_t":
        return "NVML_ERROR_INVALID_ARGUMENT"
    if return_type == "CUresult":
        return "CUDA_ERROR_INVALID_VALUE"
    if return_type == "cudaError_t":
        return "cudaErrorInvalidValue"
    if return_type == "cublasStatus_t":
        return "CUBLAS_STATUS_INVALID_VALUE"
    if return_type == "cudnnStatus_t":
        return "CUDNN_STATUS_BAD_PARAM"
    return error_const(return_type)


def prefix_std(type: str) -> str:
    # if type in ["size_t", "std::size_t"]:
    #     return "std::size_t"
    return type


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
                "{type} {name}".format(
                    type=param.type.format(),
                    name=param.name,
                )
            )
        else:
            params.append(param.type.format())
    return params


def format_call_args(function: Function) -> list[str]:
    return [param.name for param in function.parameters if param.name]


def server_call_name(function_name: str) -> str:
    if function_name == "cuEventElapsedTime_v2":
        return "cuEventElapsedTime"
    return function_name


def collect_nvml_functions(
    annotations: ParsedData, server_bindings: dict[str, ServerBinding]
):
    by_name = {
        function.name.format(): function
        for function in annotations.namespace.functions
    }
    result = []
    for name in NVML_RPC_FUNCTIONS:
        if name in server_bindings:
            continue
        function = by_name.get(name)
        if function is None:
            raise RuntimeError(f"NVML annotation for {name} not found")
        metadata = parse_annotation(function.doxygen, function.parameters)
        for operation in metadata.operations:
            if isinstance(operation, NullTerminatedOperation):
                # Preserve the existing NVML wire format. CUDA RPC strings use
                # size_t lengths, while the NVML protocol historically used
                # unsigned int lengths.
                operation.length_type = "unsigned int"
        result.append((function, function, metadata.operations, metadata))
    return result


def write_nvml_client_validation(f, operations):
    checks = []
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
        f.write("    return NVML_ERROR_INVALID_ARGUMENT;\n")
        f.write("  }\n")


def write_nvml_client_rpc(f, function, operations):
    name = function.name.format()
    params = ", ".join(format_function_params(function))
    f.write(f"static nvmlReturn_t lupine_rpc_{name}(conn_t *conn")
    if params:
        f.write(f", {params}")
    f.write(") {\n")
    f.write("  nvmlReturn_t return_value = rpc_error();\n")
    for operation in operations:
        if isinstance(operation, NullTerminatedOperation):
            f.write(
                "  {length_type} {name}_len = static_cast<{length_type}>("
                "std::strlen({name}) + 1);\n".format(
                    length_type=operation.length_type,
                    name=operation.parameter.name,
                )
            )
        elif isinstance(operation, NullableOperation) and operation.recv:
            f.write(
                "  {type_} {name}_null_check = nullptr;\n".format(
                    type_=operation.ptr.format(), name=operation.parameter.name
                )
            )

    f.write(f"  if (rpc_write_start_request(conn, RPC_{name}) < 0 ||\n")
    for operation in operations:
        operation.client_rpc_write(f)
    f.write("      rpc_wait_for_response(conn) < 0 ||\n")
    for operation in operations:
        operation.client_rpc_read(f)
    f.write("      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||\n")
    f.write("      rpc_read_end(conn) < 0) {\n")
    f.write("    return rpc_error();\n")
    f.write("  }\n")
    f.write("  return return_value;\n")
    f.write("}\n\n")


def write_nvml_client_wrapper(f, function, operations, metadata):
    if metadata.disabled_client:
        return

    name = function.name.format()
    params = ", ".join(format_function_params(function))
    f.write(f'extern "C" nvmlReturn_t {name}({params}) {{\n')
    write_nvml_client_validation(f, operations)

    call_args = format_call_args(function)
    if metadata.routing_kind == "ALL":
        owners = [
            owner
            for owner in metadata.record_owners
            if owner.kind == "NVML_DEVICE"
        ]
        if len(owners) != 1 or not isinstance(owners[0].parameter.type, Pointer):
            raise RuntimeError(
                f"{name}: ALL-routed NVML lookup requires one NVML_DEVICE output"
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
    else:
        if metadata.routing_kind == "NVML_DEVICE":
            if metadata.routing_parameter is None:
                raise RuntimeError(f"{name}: NVML_DEVICE routing requires a parameter")
            route_name = metadata.routing_parameter.name
            f.write(f"  conn_t *conn = connection_for_device(&{route_name});\n")
        elif metadata.routing_kind is None:
            f.write("  conn_t *conn = connection();\n")
        else:
            raise RuntimeError(
                f"{name}: unsupported NVML routing key {metadata.routing_kind}"
            )
        suffix = f", {', '.join(call_args)}" if call_args else ""
        f.write(f"  return lupine_rpc_{name}(conn{suffix});\n")
    f.write("}\n\n")


def write_server_buffer_cleanup(f, owned_buffers, indent):
    for buffer_name in reversed(owned_buffers):
        f.write(f"{indent}free((void *){buffer_name});\n")


def write_nvml_server_handler(f, function, operations):
    name = function.name.format()
    fn_params = ", ".join(
        parameter.type.format() for parameter in function.parameters
    )
    f.write(f"int handle_{name}(conn_t *conn) {{\n")
    owned_buffers = []
    for operation in operations:
        f.write(operation.server_declaration)
        if (
            isinstance(operation, DereferenceOperation)
            and operation.recv
            and not operation.send
        ):
            f.write(f"  {operation.parameter.name} = {{}};\n")
    f.write("  int request_id;\n")
    f.write("  nvmlReturn_t return_value;\n")
    f.write(f"  using fn_t = nvmlReturn_t (*)({fn_params});\n")
    f.write("  fn_t fn = nullptr;\n")
    f.write("  if (\n")
    for operation in operations:
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
    f.write(f'  fn = nvml_symbol<fn_t>("{name}");\n')
    f.write(
        "  return_value = fn == nullptr ? function_not_found()\n"
        f"                               : fn({', '.join(call_args)});\n\n"
    )
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


# List of possible directories to search for header files
COMMON_INCLUDE_DIRS = [
    "./",
    "/usr/local/cuda/include/",
    "/opt/cuda/include/",
    "/usr/local/include/",
    "/usr/include/",
    "/usr/include/nvidia/",
]


# Function to locate a file in common include directories
def find_header_file(filename):
    for include_dir in COMMON_INCLUDE_DIRS:
        matches = glob.glob(os.path.join(include_dir, "**", filename), recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Header file '{filename}' not found in common include directories."
    )


def validate_async_annotation(
    function: Function, metadata: FunctionAnnotationMetadata
) -> None:
    if not metadata.async_fire_forget:
        return
    name = function.name.format()
    return_type = function.return_type.format()
    if return_type != "CUresult":
        raise RuntimeError(
            f"{name}: @async requires a CUresult return type, got {return_type}"
        )
    for operation in metadata.operations:
        # NullableArrayOperation is an out-parameter with no send/recv flags.
        if getattr(operation, "recv", True):
            raise RuntimeError(
                f"{name}: @async requires every parameter to be SEND_ONLY, "
                f"but {operation.parameter.name} is received back"
            )


def main():
    options = ParserOptions(
        preprocessor=make_gcc_preprocessor(
            include_paths=["/usr/local/cuda/include"],
        ),
    )

    try:
        cuda_header = find_header_file("cuda.h")
        annotations_header = find_header_file("annotations.h")
    except FileNotFoundError as e:
        print(e)
        return

    # Parse the files
    cuda_ast: ParsedData = parse_file(cuda_header, options=options)
    annotations: ParsedData = parse_file(annotations_header, options=options)
    server_bindings = collect_server_bindings(annotations_header)
    functions = [
        function
        for function in cuda_ast.namespace.functions
        if function.name.format().startswith("cu")
        and function.name.format() not in SKIP_FUNCTIONS
    ]

    functions_with_annotations: list[
        tuple[Function, Function, list[Operation], FunctionAnnotationMetadata]
    ] = []

    dupes = {}

    for function in functions:
        # ensure duplicate functions can't be written
        if dupes.get(function.name.format()):
            continue

        dupes[function.name.format()] = True

        try:
            annotation = next(
                f for f in annotations.namespace.functions if f.name == function.name
            )
        except StopIteration:
            print(f"Annotation for {function.name} not found")
            continue
        try:
            metadata = parse_annotation(annotation.doxygen, function.parameters)
        except Exception as e:
            print(f"Error parsing annotation for {function.name}: {e}")
            continue
        validate_async_annotation(function, metadata)
        functions_with_annotations.append(
            (function, annotation, metadata.operations, metadata)
        )

    # cuda.h hides some legacy ABI entry points behind macros. @hidden
    # generates both sides from annotations.h; @disabled client keeps a manual
    # client while generating the ordinary server marshalling.
    server_functions_with_annotations = list(functions_with_annotations)
    server_function_names = {
        function.name.format()
        for function, _, _, _ in server_functions_with_annotations
    }
    hidden_functions = []
    annotation_only_server_functions = []
    for annotation in annotations.namespace.functions:
        name = annotation.name.format()
        if (
            len(name) <= 2
            or not name.startswith("cu")
            or not name[2].isupper()
            or name in server_function_names
        ):
            continue
        directives = annotation_directives(annotation.doxygen)
        hidden = "@hidden" in directives
        client_disabled = any(
            directive.startswith("@disabled client")
            for directive in directives
        )
        if not hidden and not client_disabled:
            continue
        if any(
            directive == "@disabled"
            or directive.startswith("@disabled server")
            for directive in directives
        ):
            continue
        metadata = parse_annotation(annotation.doxygen, annotation.parameters)
        validate_async_annotation(annotation, metadata)
        annotated_function = (
            annotation,
            annotation,
            metadata.operations,
            metadata,
        )
        if hidden:
            if metadata.disabled_client:
                raise RuntimeError(f"{name}: @hidden cannot disable the client")
            functions_with_annotations.append(annotated_function)
            hidden_functions.append(annotated_function)
        server_functions_with_annotations.append(annotated_function)
        if not hidden:
            annotation_only_server_functions.append(annotated_function)
        server_function_names.add(name)

    nvml_functions_with_annotations = collect_nvml_functions(
        annotations, server_bindings
    )

    annotated_names = sorted(
        set(annotated_rpc_names(annotations))
        | {
            name
            for name, binding in server_bindings.items()
            if binding.backend == "CUDA"
        }
    )

    with open("gen_rpc_ids.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n")
        f.write("// RPC ids are stable 31-bit CRC32 hashes of their operation names.\n\n")

        seen_rpc_ids: dict[int, str] = {}
        emitted_macros: set[str] = set()

        def write_rpc_define(macro_name: str, operation_name: str) -> None:
            if macro_name in emitted_macros:
                return
            value = rpc_id(operation_name)
            if value in seen_rpc_ids:
                raise RuntimeError(
                    f"RPC id collision: {operation_name} and {seen_rpc_ids[value]} "
                    f"both hash to {value}"
                )
            seen_rpc_ids[value] = operation_name
            emitted_macros.add(macro_name)
            f.write(f"#define {macro_name} {value}\n")

        for function, _, _, _ in functions_with_annotations:
            name = function.name.format()
            if name in PRIVATE_RPC_FUNCTIONS:
                continue
            write_rpc_define(f"RPC_{name}", name)
        for name in annotated_names:
            if name in PRIVATE_RPC_FUNCTIONS:
                continue
            write_rpc_define(f"RPC_{name}", name)
        for name in NVML_RPC_FUNCTIONS:
            write_rpc_define(f"RPC_{name}", name)
        f.write("\n")
        for name in PRIVATE_RPC_FUNCTIONS:
            write_rpc_define(f"LUPINE_RPC_{name}", name)

    with open("gen_nvml_client.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, operations, metadata in nvml_functions_with_annotations:
            if metadata.disabled_client:
                continue
            write_nvml_client_rpc(f, function, operations)
            write_nvml_client_wrapper(f, function, operations, metadata)

    with open("gen_nvml_server.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, operations, metadata in nvml_functions_with_annotations:
            if metadata.disabled_server:
                continue
            write_nvml_server_handler(f, function, operations)

    with open("gen_nvml_server.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, _, metadata in nvml_functions_with_annotations:
            if metadata.disabled_server:
                continue
            f.write(f"int handle_{function.name.format()}(conn_t *conn);\n")

    with open("gen_cuda_client.cpp", "w") as f:
        f.write(
            "#include <cuda.h>\n"
            "\n"
            "#define LUPINE_CUDA_COMPAT_TYPES_ONLY\n"
            '#include "cuda_compat.h"\n'
            "#undef LUPINE_CUDA_COMPAT_TYPES_ONLY\n"
            "\n"
            "#include <algorithm>\n"
            "#include <cstdint>\n"
            "#include <cstdio>\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n"
            "#include <vector>\n\n"
            '#include "gen_rpc_ids.h"\n\n'
            '#include "client_routing.h"\n'
            '#include "rpc.h"\n\n'
            "extern int rpc_size();\n"
            "extern conn_t *rpc_client_get_connection(unsigned int index);\n"
            "extern void rpc_close(conn_t *conn);\n"
            'extern "C" void lupine_deep_cache_reset(const void *key);\n'
            'extern "C" void *lupine_deep_cache_add(const void *key, '
            "size_t bytes);\n\n"
            'extern "C" conn_t *lupine_rpc_conn_for_device(CUdevice *device);\n'
            'extern "C" conn_t *lupine_rpc_conn_for_current_context();\n'
            'extern "C" conn_t *lupine_rpc_conn_for_context(CUcontext ctx);\n'
            'extern "C" conn_t *lupine_rpc_conn_for_module(CUmodule module);\n'
            'extern "C" conn_t *lupine_rpc_conn_for_function(CUfunction function);\n'
            'extern "C" conn_t *lupine_rpc_conn_for_stream(CUstream stream);\n'
            'extern "C" conn_t *lupine_rpc_conn_for_event(CUevent event);\n'
            'extern "C" conn_t *lupine_rpc_conn_for_deviceptr(CUdeviceptr ptr);\n'
            'extern "C" CUfunction lupine_translate_private_function_for_rpc(CUfunction function);\n'
            'extern "C" void lupine_note_context_owner(CUcontext ctx, conn_t *conn);\n'
            'extern "C" void lupine_note_module_owner(CUmodule module, conn_t *conn);\n'
            'extern "C" void lupine_note_library_owner(CUlibrary library, conn_t *conn);\n'
            'extern "C" void lupine_note_function_owner(CUfunction function, conn_t *conn);\n'
            'extern "C" void lupine_note_stream_owner(CUstream stream, conn_t *conn);\n'
            'extern "C" void lupine_note_event_owner(CUevent event, conn_t *conn);\n'
            'extern "C" void lupine_note_memory_pool_owner(CUmemoryPool pool, conn_t *conn);\n'
            'extern "C" void lupine_note_graph_owner(CUgraph graph, conn_t *conn);\n'
            'extern "C" void lupine_note_graph_node_owner(CUgraphNode node, conn_t *conn);\n'
            'extern "C" void lupine_note_graph_exec_owner(CUgraphExec exec, conn_t *conn);\n'
            'extern "C" void lupine_note_deviceptr_owner(CUdeviceptr ptr, conn_t *conn);\n\n'
            'extern "C" void lupine_note_deviceptr_allocation(CUdeviceptr ptr, size_t size, conn_t *conn);\n\n'
            'extern "C" void lupine_forget_deviceptr_owner(CUdeviceptr ptr);\n\n'
            'extern "C" void lupine_forget_stream_owner(CUstream stream);\n\n'
            'extern "C" const char *lupine_retain_returned_string(const void *handle, const char *data, size_t size);\n\n'
            'extern "C" void lupine_release_module_retained_strings(CUmodule module);\n'
            'extern "C" void lupine_release_library_retained_strings(CUlibrary library);\n\n'
            'extern "C" void lupine_record_library_module(CUmodule module, CUlibrary library);\n\n'
            'extern "C" CUresult lupine_record_library_kernel(CUkernel kernel, CUlibrary library, const char *name, lupine_route route);\n\n'
            'extern "C" CUresult lupine_record_module_function(CUfunction function, CUmodule module, const char *name, lupine_route route);\n\n'
            'extern "C" bool lupine_deviceptrs_share_route(CUdeviceptr first, CUdeviceptr second);\n'
            'extern "C" CUresult lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice,\n'
            '                                                   CUdeviceptr srcDevice,\n'
            '                                                   size_t ByteCount,\n'
            '                                                   CUstream hStream,\n'
            '                                                   bool async);\n\n'
            'extern "C" void lupine_invalidate_current_context_cache();\n'
            'extern "C" void lupine_forget_destroyed_context(CUcontext ctx);\n'
            'extern "C" void lupine_mark_context_green(CUcontext ctx);\n'
            'extern "C" void lupine_invalidate_function_caches();\n'
            'extern "C" void lupine_invalidate_kernel_attribute_cache();\n'
            'extern "C" void lupine_kernel_attribute_cache_erase(int route_id, CUkernel kernel, int attrib, int dev);\n'
            'extern "C" void lupine_invalidate_function_attribute_cache();\n'
            'extern "C" int lupine_read_deferred_dtoh_copies(conn_t *conn);\n'
            'extern "C" int lupine_forward_remote_stdout(conn_t *conn);\n'
            'extern "C" CUresult lupine_sync_mapped_device_to_host();\n'
            'extern "C" const void *lupine_mapped_host_read_source(const void *host, size_t size);\n\n'
        )
        for function, _, _, _ in hidden_functions:
            name = function.name.format()
            f.write(f"#ifdef {name}\n#undef {name}\n#endif\n")
            f.write(
                'extern "C" {return_type} CUDAAPI {name}({params});\n\n'.format(
                    return_type=function.return_type.format(),
                    name=name,
                    params=", ".join(format_function_params(function)),
                )
            )
        for function, annotation, operations, metadata in functions_with_annotations:
            # We don't generate client function definitions for client-disabled
            # functions; their RPC/server definitions may still be generated.
            if metadata.disabled_client:
                continue

            if metadata.guard is not None:
                f.write(f"#if {metadata.guard}\n")

            joined_params = ", ".join(format_function_params(function))

            f.write(
                "{return_type} {name}({params})\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=joined_params,
                )
            )
            f.write("{\n")

            all_output = metadata.routing_parameter
            if metadata.routing_kind == "ALL":
                if (
                    function.return_type.format() != "CUresult"
                    or all_output is None
                    or not isinstance(all_output.type, Pointer)
                    or all_output.type.ptr_to.format() != "CUdevice"
                ):
                    raise RuntimeError(
                        f"{function.name.format()}: ALL routing requires a CUdevice * output"
                    )
                if metadata.async_fire_forget:
                    raise RuntimeError(
                        f"{function.name.format()}: ALL routing cannot be fire-and-forget"
                    )

                output_name = all_output.name
                checks = [f"{output_name} == nullptr"]
                for operation in operations:
                    if isinstance(operation, NullTerminatedOperation) and operation.send:
                        checks.append(f"{operation.parameter.name} == nullptr")
                    elif (
                        isinstance(operation, DereferenceOperation)
                        and operation.parameter.name != output_name
                    ):
                        checks.append(f"{operation.parameter.name} == nullptr")
                    elif isinstance(operation, ArrayOperation):
                        checks.append(
                            f"({operation.transfer_size_expr()} != 0 && "
                            f"{operation.parameter.name} == nullptr)"
                        )
                f.write("    if (" + " || ".join(checks) + ") {\n")
                f.write("        return CUDA_ERROR_INVALID_VALUE;\n")
                f.write("    }\n")
                f.write(
                    f"    return lupine_lookup_device_on_all_routes({output_name},\n"
                    "        [&](lupine_route route, CUdevice *route_output) {\n"
                    f"            {all_output.type.format()} {output_name} = route_output;\n"
                )
            else:
                f.write(
                    "    lupine_route route = {route_expr};\n".format(
                        route_expr=client_routing_route_expr(metadata)
                    )
                )
                if metadata.routing_kind == "DEVICE":
                    f.write("    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)\n")
                    f.write(
                        "        return {error_return};\n".format(
                            error_return=invalid_device_const(
                                function.return_type.format()
                            )
                        )
                    )
            if metadata.cross_server_copy is not None:
                copy = metadata.cross_server_copy
                stream_arg = (
                    copy.stream.name
                    if copy.stream is not None
                    else "nullptr"
                )
                async_arg = "true" if copy.async_ else "false"
                f.write(
                    "    if (!lupine_deviceptrs_share_route({dst}, {src})) {{\n".format(
                        dst=copy.dst.name,
                        src=copy.src.name,
                    )
                )
                f.write(
                    "        return lupine_cuMemcpyDtoD_via_client({dst}, {src}, {bytes}, {stream}, {async_});\n".format(
                        dst=copy.dst.name,
                        src=copy.src.name,
                        bytes=copy.bytes.name,
                        stream=stream_arg,
                        async_=async_arg,
                    )
                )
                f.write("    }\n")
            f.write(
                "    {return_type} return_value;\n".format(
                    return_type=function.return_type.format()
                )
            )
            if function.name.format() == "cuCtxDestroy_v2":
                # Destroying the current context implicitly pops it; mirror
                # that in the client's virtual context state before the call.
                f.write("    CUcontext lupine_current_before_destroy = nullptr;\n")
                f.write("    if (cuCtxGetCurrent(&lupine_current_before_destroy) ==\n")
                f.write("            CUDA_SUCCESS &&\n")
                f.write("        lupine_current_before_destroy == ctx) {\n")
                f.write("        cuCtxSetCurrent(nullptr);\n")
                f.write("    }\n")
            call_args = ", ".join(client_call_args(function, metadata))
            helper_args = f", {call_args}" if call_args else ""
            local_call = 'lupine_call_real_cuda_fn("{name}"{args})'.format(
                name=function.name.format(), args=helper_args
            )
            local_post_call = io.StringIO()
            write_client_post_call(local_post_call, function, metadata)
            if local_post_call.getvalue():
                f.write("    if (lupine_route_is_local(route)) {\n")
                f.write(f"        return_value = {local_call};\n")
                f.write(local_post_call.getvalue())
                f.write("        return return_value;\n")
                f.write("    }\n")
            else:
                f.write("    if (lupine_route_is_local(route))\n")
                f.write(f"        return {local_call};\n")
            f.write("    conn_t *conn = lupine_route_remote_conn(route);\n")

            for operation in operations:
                if isinstance(operation, OpaqueTypeOperation):
                    f.write(operation.client_declaration())
                if (
                    isinstance(operation, InOutCountOperation)
                    or isinstance(operation, NullableArrayOperation)
                    or isinstance(operation, DeepStructOperation)
                    or (
                        isinstance(operation, NullTerminatedOperation)
                        and operation.recv
                    )
                ):
                    f.write(operation.client_declaration())

            for operation in operations:
                if isinstance(operation, NullTerminatedOperation) and operation.send:
                    f.write(
                        "    std::size_t {param_name}_len = std::strlen({param_name}) + 1;\n".format(
                            param_name=operation.parameter.name
                        )
                    )
                if isinstance(operation, NullableOperation) and operation.recv:
                    f.write(
                        "    {server_type} {param_name}_null_check;\n".format(
                            server_type=operation.ptr.format(),
                            param_name=operation.parameter.name,
                        )
                    )

            # Reject invalid send buffers before lupine_prepare_rpc() flushes
            # pending writes and rpc_write_start_request() acquires the
            # connection's call/write locks. Conditions in the builder below
            # may skip optional writes, but only rpc_write* calls themselves
            # are allowed to fail the builder.
            for operation in operations:
                if isinstance(operation, ArrayOperation):
                    operation.client_preflight(
                        f, invalid_argument_const(function.return_type.format())
                    )
                elif (
                    isinstance(operation, NullTerminatedOperation)
                    and operation.recv
                ):
                    operation.client_preflight(
                        f, invalid_argument_const(function.return_type.format())
                    )

            if metadata.async_fire_forget:
                error_return = error_const(function.return_type.format())
                f.write(
                    "    if (lupine_prepare_rpc(conn) < 0 ||\n"
                    "        rpc_write_start_request(conn, RPC_{name}) < 0 ||\n".format(
                        name=function.name.format()
                    )
                )
                for operation in operations:
                    write_client_rpc_write(f, operation, metadata)
                f.write("        rpc_write_end(conn) < 0) {\n")
                f.write("        return {r};\n".format(r=error_return))
                f.write("    }\n")
                post_call = io.StringIO()
                write_client_post_call(post_call, function, metadata)
                if post_call.getvalue():
                    f.write("    return_value = CUDA_SUCCESS;\n")
                    f.write(post_call.getvalue())
                    f.write("    return return_value;\n")
                else:
                    f.write("    return CUDA_SUCCESS;\n")
                f.write("}\n\n")
                if metadata.guard is not None:
                    f.write("#endif\n\n")
                continue

            f.write(
                "    if (lupine_prepare_rpc(conn) < 0 ||\n"
                "        rpc_write_start_request(conn, RPC_{name}) < 0 ||\n".format(
                    name=function.name.format()
                )
            )

            for operation in operations:
                write_client_rpc_write(f, operation, metadata)

            f.write("        rpc_wait_for_response(conn) < 0 ||\n")

            if metadata.synchronize and metadata.synchronize.deferred_dtoh:
                f.write("        lupine_read_deferred_dtoh_copies(conn) < 0 ||\n")
            if metadata.synchronize and metadata.synchronize.stdout:
                f.write("        lupine_forward_remote_stdout(conn) < 0 ||\n")

            for operation in operations:
                operation.client_rpc_read(f)

            f.write(
                "        rpc_read(conn, &return_value, sizeof({return_type})) < 0 ||\n".format(
                    return_type=function.return_type.format()
                )
            )
            f.write("        rpc_read_end(conn) < 0)\n")
            f.write(
                "        return {error_return};\n".format(
                    error_return=error_const(function.return_type.format())
                )
            )

            for operation in operations:
                if isinstance(operation, NullTerminatedOperation) and operation.recv:
                    retain = next(
                        (
                            item
                            for item in metadata.retains
                            if item.parameter.name == operation.parameter.name
                        ),
                        None,
                    )
                    if retain is None:
                        raise RuntimeError(
                            f"{function.name.format()}: returned string requires @retain"
                        )
                    operation.client_post_rpc(
                        f,
                        "CUDA_SUCCESS",
                        "CUDA_ERROR_OUT_OF_MEMORY",
                        retain.handle.name,
                    )

            write_client_post_call(f, function, metadata)
            f.write("    return return_value;\n")
            if metadata.routing_kind == "ALL":
                f.write("        });\n")
            f.write("}\n\n")
            if metadata.guard is not None:
                f.write("#endif\n\n")

        function_by_name = {
            function.name.format(): function
            for function, _, _, metadata in functions_with_annotations
            if not metadata.disabled_client
        }
        for alias, target in MANUAL_REMAPPINGS:
            if alias in function_by_name or target not in function_by_name:
                continue
            target_function = function_by_name[target]
            guard = MANUAL_REMAPPING_GUARDS.get(alias)
            if guard is not None:
                f.write("#if {guard}\n".format(guard=guard))
            f.write("#ifdef {name}\n#undef {name}\n#endif\n".format(name=alias))
            f.write(
                'extern "C" {return_type} {name}({params})\n'.format(
                    return_type=target_function.return_type.format(),
                    name=alias,
                    params=", ".join(format_function_params(target_function)),
                )
            )
            f.write("{\n")
            call = "{target}({args})".format(
                target=target,
                args=", ".join(format_call_args(target_function)),
            )
            if target_function.return_type.format() == "void":
                f.write("    {call};\n".format(call=call))
                f.write("}\n\n")
            else:
                f.write("    return {call};\n".format(call=call))
                f.write("}\n\n")
            if guard is not None:
                f.write("#endif\n\n")

        f.write("std::unordered_map<std::string, void *> functionMap = {\n")
        for function, _, _, metadata in functions_with_annotations:
            if metadata.disabled_client and metadata.disabled_server:
                continue

            if metadata.guard is not None:
                f.write(f"#if {metadata.guard}\n")
            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.name.format()
                )
            )
            if metadata.guard is not None:
                f.write("#endif\n")
        # write manual overrides
        function_names = set(
            f.name.format()
            for f, _, _, metadata in functions_with_annotations
            if not metadata.disabled_client
        )
        for x, y in MANUAL_REMAPPINGS:
            # ensure y exists in the function list
            if y not in function_names:
                print(f"Skipping manual remapping {x} -> {y}")
                continue
            f.write(
                '    {{"{x}", (void *){y}}},\n'.format(
                    x=x,
                    y=y,
                )
            )
        f.write("};\n\n")

        f.write("void *get_function_pointer(const char *name)\n")
        f.write("{\n")
        f.write("    auto it = functionMap.find(name);\n")
        f.write("    if (it == functionMap.end())\n")
        f.write("        return nullptr;\n")
        f.write("    return it->second;\n")
        f.write("}\n")

    with open("gen_cuda_server.cpp", "w") as f:
        f.write(
            "#include <iostream>\n"
            "#include <cuda.h>\n"
            '#include "cuda_compat.h"\n'
            "\n"
            "#include <cstring>\n"
            "#include <string>\n"
            '#include "gen_rpc_ids.h"\n\n'
            '#include <vector>\n\n'
            '#include <cstdio>\n\n'
            '#include "rpc.h"\n\n'
        )
        annotation_only_functions = (
            hidden_functions + annotation_only_server_functions
        )
        for function, _, _, _ in annotation_only_functions:
            name = function.name.format()
            f.write(f"#ifdef {name}\n#undef {name}\n#endif\n")
            f.write(
                'extern "C" {return_type} CUDAAPI {name}({params});\n\n'.format(
                    return_type=function.return_type.format(),
                    name=name,
                    params=", ".join(format_function_params(function)),
                )
            )
        for (
            function,
            annotation,
            operations,
            metadata,
        ) in server_functions_with_annotations:
            if (
                metadata.disabled_server
                or function.name.format() in server_bindings
            ):
                continue

            if metadata.guard is not None:
                f.write(f"#if {metadata.guard}\n")

            # parse the annotation doxygen
            f.write(
                "int handle_{name}(conn_t *conn)\n".format(
                    name=function.name.format(),
                )
            )
            f.write("{\n")

            owned_buffers = []

            for operation in operations:
                f.write(operation.server_declaration)

            f.write("    int request_id;\n")

            # we only generate return from non-void types
            if metadata.async_fire_forget:
                pass
            elif function.return_type.format() != "void":
                f.write(
                    "    {return_type} lupine_intercept_result;\n".format(
                        return_type=function.return_type.format()
                    )
                )
            else:
                f.write("    void* lupine_intercept_result;\n")

            f.write("    if (\n")
            for operation in operations:
                if owned_buffer := operation.server_rpc_read(f):
                    owned_buffers.append(owned_buffer)
            f.write("        false)\n")
            f.write("        goto ERROR_0;\n")

            f.write("\n")

            f.write("    request_id = rpc_read_end(conn);\n")
            f.write("    if (request_id < 0)\n")
            f.write("        goto ERROR_0;\n")

            params: list[str] = []
            # these need to be in function param order, not operation order.
            for param in function.parameters:
                for op in operations:
                    if op.parameter.name == param.name:
                        params.append(op.server_reference)

            if metadata.async_fire_forget or function.return_type.format() == "void":
                f.write(
                    "    {name}({params});\n\n".format(
                        name=server_call_name(function.name.format()),
                        params=", ".join(params),
                    )
                )
            else:
                f.write(
                    "    lupine_intercept_result = {name}({params});\n\n".format(
                        name=server_call_name(function.name.format()),
                        params=", ".join(params),
                    )
                )

            if metadata.async_fire_forget:
                write_server_buffer_cleanup(f, owned_buffers, "    ")
                f.write("    return 0;\n")
                f.write("ERROR_0:\n")
                write_server_buffer_cleanup(f, owned_buffers, "    ")
                f.write("    return -1;\n")
                f.write("}\n\n")
                if metadata.guard is not None:
                    f.write("#endif\n\n")
                continue

            f.write("    if (rpc_write_start_response(conn, request_id) < 0 ||\n")

            for operation in operations:
                operation.server_rpc_write(f)

            f.write(
                "        rpc_write(conn, &lupine_intercept_result, sizeof({return_type})) < 0 ||\n".format(
                    return_type=function.return_type.format()
                )
            )
            f.write("        rpc_write_end(conn) < 0)\n")
            f.write("        goto ERROR_0;\n")
            f.write("\n")
            write_server_buffer_cleanup(f, owned_buffers, "    ")
            f.write("    return 0;\n")

            f.write("ERROR_0:\n")
            write_server_buffer_cleanup(f, owned_buffers, "    ")
            f.write("    return -1;\n")
            f.write("}\n\n")
            if metadata.guard is not None:
                f.write("#endif\n\n")

    generated_bindings = [
        ServerBinding(
            function.name.format(),
            "CUDA",
            f"handle_{function.name.format()}",
            metadata.guard,
        )
        for function, _, _, metadata in server_functions_with_annotations
        if not metadata.disabled_server
        and function.name.format() not in server_bindings
    ]
    generated_bindings.extend(
        ServerBinding(name, "NVML", f"handle_{name}")
        for name in NVML_RPC_FUNCTIONS
        if name not in server_bindings
    )
    bindings = list(server_bindings.values()) + generated_bindings

    operations_by_id = {}
    for binding in bindings:
        operation = rpc_id(binding.name)
        if operation in operations_by_id:
            raise RuntimeError(
                f"Duplicate RPC operation for {operations_by_id[operation]} "
                f"and {binding.name}"
            )
        operations_by_id[operation] = binding.name

    def registry_entry(binding: ServerBinding, macro: str) -> str:
        operation_prefix = (
            "LUPINE_RPC_" if binding.name in PRIVATE_RPC_FUNCTIONS else "RPC_"
        )
        return (
            f"{macro}({operation_prefix}{binding.name}, {binding.handler}, "
            f"{binding.backend_symbol})"
        )

    registry_entries = {backend: [] for backend in SERVER_BACKENDS}
    guarded_handlers = {backend: [] for backend in SERVER_BACKENDS}
    guarded_declarations = {backend: [] for backend in SERVER_BACKENDS}
    for binding in bindings:
        if binding.guard is None:
            registry_entries[binding.backend].append(
                "  " + registry_entry(binding, "HANDLER")
            )
        else:
            guarded_declarations[binding.backend].append(
                f"#if {binding.guard}\n"
                f"{registry_entry(binding, 'LUPINE_DECLARE_HANDLER')}\n"
                f'#endif'
            )
            guarded_handlers[binding.backend].append(
                f"#if {binding.guard}\n"
                f"      {registry_entry(binding, 'LUPINE_REGISTER_HANDLER')}\n"
                f'#endif'
            )

    with open("registry.cpp", "w") as f:
        f.write(
            REGISTRY_CPP_TEMPLATE.substitute(
                cuda_registry_entries=" \\\n".join(registry_entries["CUDA"]),
                nvml_registry_entries=" \\\n".join(registry_entries["NVML"]),
                cuda_guarded_declarations="\n".join(
                    guarded_declarations["CUDA"]
                ),
                nvml_guarded_declarations="\n".join(
                    guarded_declarations["NVML"]
                ),
                cuda_guarded_handlers="\n".join(guarded_handlers["CUDA"]),
                nvml_guarded_handlers="\n".join(guarded_handlers["NVML"]),
            )
        )

    subprocess.run(
        [
            "clang-format",
            "-i",
            "gen_cuda_client.cpp",
            "gen_nvml_client.inc",
            "gen_nvml_server.h",
            "gen_nvml_server.inc",
            "gen_cuda_server.cpp",
            "registry.cpp",
        ],
        check=True,
    )


def verify_backend_boundaries(backend: str) -> None:
    backend_files = {
        "cuda": ["gen_cuda_client.cpp", "gen_cuda_server.cpp"],
        "nvml": [
            "gen_nvml_client.inc",
            "gen_nvml_server.inc",
            "gen_nvml_server.h",
        ],
    }
    backends = backend_files if backend == "all" else [backend]
    for name in backends:
        forbidden = (
            ["nvml"]
            if name == "cuda"
            else ["cuda_compat", "<cuda.h>", "get_cuda_handler", "handle_cu"]
        )
        paths = backend_files[name]
        for path in paths:
            contents = open(path, encoding="utf-8").read().lower()
            for token in forbidden:
                if token.lower() in contents:
                    raise RuntimeError(
                        f"{path} contains forbidden {name} boundary token: {token}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-backend",
        choices=("all", "cuda", "nvml"),
        help="verify existing generated files without loading backend SDK headers",
    )
    args = parser.parse_args()
    # Inputs (annotations.h) and gen_* outputs are CWD-relative.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.verify_backend:
        verify_backend_boundaries(args.verify_backend)
    else:
        main()
        verify_backend_boundaries("all")
