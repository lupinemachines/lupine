# /// script
# requires-python = ">=3.10"
# dependencies = ["clang-format==18.1.3", "cxxheaderparser==1.9.2"]
# ///
from cxxheaderparser.simple import parse_file, parse_string, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function, Array
from typing import Optional, Union
from dataclasses import dataclass
from string import Template
from types import SimpleNamespace
import argparse
from collections import namedtuple
import io
import itertools
import os
import glob
import re
import subprocess
import textwrap
import zlib
from client_templates import collect_client_call_templates
from emit import (
    Backend,
    unsupported,
    write_backend,
    format_call_args,
    format_function_params,
    write_client_rpc,
    write_client_wrapper,
    write_server_buffer_cleanup,
    write_server_handler,
)
from ops import (
    NullableOperation,
    ArrayOperation,
    InOutCountOperation,
    NullableArrayOperation,
    DeepStructOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    DereferenceOperation,
    ScalarOperation,
    Operation,
    OwnerAnnotation,
    RetainAnnotation,
    ReleaseAnnotation,
    ParentAnnotation,
    CrossServerCopyAnnotation,
    ClientCallTemplate,
    FunctionAnnotationMetadata,
    GraphExecNodeAnnotation,
    RoutingFallbackAnnotation,
    SynchronizeAnnotation,
)

# CUDA headers omit some legacy ABI entry points from their public declarations.
# Keep the small set that still needs RPC wrappers explicit here.
LEGACY_ABI_FUNCTIONS = {
    "cuGraphExecUpdate",
    "cuGraphInstantiate_v2",
}

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

# Versioned graph-query ABIs are also exported under their public, unversioned
# names. Unlike MANUAL_REMAPPINGS, these aliases must not emit C wrappers: the
# literal unversioned symbols retain their older signatures for binary
# compatibility, and cuGetProcAddress selects by the requested API version.
FUNCTION_MAP_ALIASES = [
    ("cuGraphGetEdges", "cuGraphGetEdges_v2", "CUDA_VERSION >= 12030"),
    (
        "cuGraphNodeGetDependencies",
        "cuGraphNodeGetDependencies_v2",
        "CUDA_VERSION >= 12030",
    ),
    (
        "cuGraphNodeGetDependentNodes",
        "cuGraphNodeGetDependentNodes_v2",
        "CUDA_VERSION >= 12030",
    ),
]

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
#include <cuda_runtime_api.h>
#endif
#ifdef LUPINE_BUILD_CUBLAS_BACKEND
#include <cublas_v2.h>
#endif
#include "gen_rpc_ids.h"

// clang-format off
#define LUPINE_CUDA_RPC_HANDLERS(HANDLER) \
$cuda_registry_entries
#define LUPINE_CUDART_RPC_HANDLERS(HANDLER) \
$cudart_registry_entries
#define LUPINE_CUBLAS_RPC_HANDLERS(HANDLER) \
$cublas_registry_entries
#define LUPINE_NVML_RPC_HANDLERS(HANDLER) \
$nvml_registry_entries
#define LUPINE_HIP_RPC_HANDLERS(HANDLER) \
$hip_registry_entries
// clang-format on

#define LUPINE_DECLARE_HANDLER(operation, handler, backend)                    \
  int handler(conn_t *conn);
#ifdef LUPINE_BUILD_CUDA_BACKEND
LUPINE_CUDA_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$cuda_guarded_declarations
LUPINE_CUDART_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$cudart_guarded_declarations
#endif
#ifdef LUPINE_BUILD_CUBLAS_BACKEND
LUPINE_CUBLAS_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$cublas_guarded_declarations
#endif
#ifdef LUPINE_BUILD_NVML_BACKEND
LUPINE_NVML_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$nvml_guarded_declarations
#endif
#ifdef LUPINE_BUILD_HIP_BACKEND
LUPINE_HIP_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$hip_guarded_declarations
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
      LUPINE_CUDART_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$cudart_guarded_handlers
#endif
#ifdef LUPINE_BUILD_CUBLAS_BACKEND
      LUPINE_CUBLAS_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$cublas_guarded_handlers
#endif
#ifdef LUPINE_BUILD_NVML_BACKEND
      LUPINE_NVML_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$nvml_guarded_handlers
#endif
#ifdef LUPINE_BUILD_HIP_BACKEND
      LUPINE_HIP_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$hip_guarded_handlers
#endif
  };
  // clang-format on
#undef LUPINE_REGISTER_HANDLER
  return handlers;
}

#undef LUPINE_CUDA_RPC_HANDLERS
#undef LUPINE_CUDART_RPC_HANDLERS
#undef LUPINE_CUBLAS_RPC_HANDLERS
#undef LUPINE_NVML_RPC_HANDLERS
#undef LUPINE_HIP_RPC_HANDLERS
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
    "HIP": "rpc_backend::hip",
    "CUDART": "rpc_backend::cudart",
    "CUBLAS": "rpc_backend::cublas",
}


def annotation_directives(annotation: str) -> list[str]:
    directives = []
    for line in (annotation or "").splitlines():
        line = line.strip().lstrip("*").strip()
        if line.startswith("@"):
            directives.append(line)
    return directives


def parse_server_binding(
    name: str, annotation: str, backend: str
) -> Optional[ServerBinding]:
    """A hand-written server handler: `handle_<call>`, or the name given."""
    handler = None
    guard = None
    for directive in annotation_directives(annotation):
        parts = directive.split()
        if parts[0] == "@disabled" and parts[1:2] != ["client"]:
            if handler is not None:
                raise RuntimeError(f"Duplicate @disabled for {name}")
            rest = parts[2:] if parts[1:2] == ["server"] else parts[1:]
            handler = rest[0] if rest and rest[0] != "-" else "handle_" + name
        elif parts[0] == "@guard":
            if guard is not None or len(parts) < 2:
                raise RuntimeError(f"Invalid @guard annotation for {name}")
            guard = directive.removeprefix("@guard").strip()

    if handler is None:
        return None
    return ServerBinding(name, backend, handler, guard)


def collect_server_bindings(path: str, backend: str) -> dict[str, ServerBinding]:
    bindings = {}
    with open(path) as annotations_file:
        source = annotations_file.read()
    for match in re.finditer(r"/\*\*(.*?)\*/\s*([^;{]*?\()", source, re.DOTALL):
        annotation, declaration = match.groups()
        name_match = re.search(r"([A-Za-z_]\w*)\s*\($", declaration)
        if name_match is None:
            continue
        binding = parse_server_binding(name_match.group(1), annotation, backend)
        if binding is None:
            continue
        previous = bindings.get(binding.name)
        if previous is not None and previous != binding:
            raise RuntimeError(f"Conflicting @disabled for {binding.name}")
        bindings[binding.name] = binding
    return bindings


# One annotation file per shim target. A declaration belongs to the target
# whose file it lives in, so no API-name prefix is needed to tell them apart.
CUDA = Backend(
    name="cuda",
    result="CUresult",
    success="CUDA_SUCCESS",
    invalid_argument="CUDA_ERROR_INVALID_VALUE",
    not_supported="CUDA_ERROR_NOT_SUPPORTED",
    # The driver shim links against libcuda, so its handlers call the entry
    # point directly instead of resolving it by name.
    symbol_lookup="",
)

NVML = Backend(
    name="nvml",
    result="nvmlReturn_t",
    success="NVML_SUCCESS",
    invalid_argument="NVML_ERROR_INVALID_ARGUMENT",
    not_supported="NVML_ERROR_NOT_SUPPORTED",
    symbol_lookup="nvml_symbol",
    lookup_on_all_connections=True,
)

HIP = Backend(
    name="hip",
    result="hipError_t",
    success="hipSuccess",
    invalid_argument="hipErrorInvalidValue",
    not_supported="hipErrorNotSupported",
    symbol_lookup="hip_symbol",
    guard_null_conn=True,
    remappings=(("hipGetDeviceProperties", "hipGetDevicePropertiesR0600"),),
)

CUDART = Backend(
    name="cudart",
    result="cudaError_t",
    success="cudaSuccess",
    invalid_argument="cudaErrorInvalidValue",
    not_supported="cudaErrorNotSupported",
    symbol_lookup="cudart_symbol",
    sticky_error="record",
    guard_null_conn=True,
)

# cuBLAS calls run on the driver shim's connections like the runtime's, and
# the client keeps no sticky error: every status is the call's own.
CUBLAS = Backend(
    name="cublas",
    result="cublasStatus_t",
    success="CUBLAS_STATUS_SUCCESS",
    invalid_argument="CUBLAS_STATUS_INVALID_VALUE",
    not_supported="CUBLAS_STATUS_NOT_SUPPORTED",
    symbol_lookup="cublas_symbol",
    guard_null_conn=True,
)

ANNOTATION_FILES = {
    "cuda": "annotations_cuda.h",
    "cudart": "annotations_cudart.h",
    "cublas": "annotations_cublas.h",
    "nvml": "annotations_nvml.h",
    "hip": "annotations_hip.h",
}


SKIP_FUNCTIONS = {
    "cuStreamUpdateCaptureDependencies_v2",
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
        if type_name in ("nvmlDevice_t", "CUdevice"):
            return "DEVICE", param
        if type_name == "cudaStream_t":
            return "STREAM", param
        if type_name == "cudaEvent_t":
            return "EVENT", param
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
        # A library handle is created on one server and routes every later
        # call there.
        if type_name == "cublasHandle_t":
            return "HANDLE", param
    return None, None


# Parses a function annotation into marshalling operations and metadata.
# Directives that name their annotation's fields in order. A field called
# "kind" is taken as written; the rest name parameters.
UNIFORM_ANNOTATIONS = {
    "routingfallback": (RoutingFallbackAnnotation, None),
    "recordowner": (OwnerAnnotation, "record_owners"),
    "recordparent": (ParentAnnotation, "parents"),
    "retain": (RetainAnnotation, "retains"),
    "release": (ReleaseAnnotation, "releases"),
}


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
        # @disabled client / @disabled server skip one generated side; bare
        # @disabled (optionally naming the server handler) skips both.
        if "@disabled" in line:
            if metadata.disabled_client or metadata.disabled_server:
                raise RuntimeError("Duplicate @disabled")
            parts = line.lstrip(" *").split()
            scope = parts[1:2]
            if scope == ["client"]:
                metadata.disabled_client = True
                # "forwards": the manual client provides the entry point but
                # still sends the call, so it needs the generated builder.
                metadata.client_forwards = parts[2:3] == ["forwards"]
                continue
            elif scope == ["server"]:
                metadata.disabled_server = True
                continue
            else:
                metadata.disabled_client = True
                metadata.disabled_server = True
                continue
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
        if line.startswith("@routingkey"):
            parts = line.split()
            if len(parts) < 2:
                continue
            metadata.routing_kind = parts[1].upper()
            if len(parts) >= 3:
                metadata.routing_parameter = annotation_param(params, parts[2])
            continue
        directive = next(
            (name for name in UNIFORM_ANNOTATIONS if line.startswith("@" + name)),
            None,
        )
        if directive is not None:
            annotation, target = UNIFORM_ANNOTATIONS[directive]
            fields = list(annotation.__annotations__)
            parts = line.split()
            if len(parts) != len(fields) + 1:
                raise RuntimeError(f"@{directive} requires {' '.join(fields)}")
            values = {
                field: part.upper() if field == "kind" else annotation_param(params, part)
                for field, part in zip(fields, parts[1:])
            }
            if target is None:
                metadata.routing_fallback = annotation(**values)
            else:
                getattr(metadata, target).append(annotation(**values))
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
        if line.startswith("@clearfields"):
            # @clearfields <param> <member>...
            parts = line.split()
            if len(parts) < 3:
                raise RuntimeError("@clearfields requires a parameter and members")
            metadata.clear_fields.append((parts[1], tuple(parts[2:])))
            continue
        if line.startswith("@graphexecnode"):
            parts = line.split()
            if len(parts) != 3 or metadata.graph_exec_node is not None:
                raise RuntimeError(
                    "@graphexecnode requires graph exec and graph node parameters"
                )
            metadata.graph_exec_node = GraphExecNodeAnnotation(
                graph_exec=annotation_param(params, parts[1]),
                node=annotation_param(params, parts[2]),
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
                null_terminated = "NULL_TERMINATED" in args
                nullable = "NULLABLE" in args
                deref = "DEREF" in args
                recv_on_error = "ON_ERROR" in args
                scalar = "SCALAR" in args

                if scalar:
                    # SCALAR: a pointer-mode scalar, sized by SIZE:<expr> when
                    # the pointee is void. The call's first parameter is the
                    # handle whose pointer mode decides where it lives.
                    if length_arg or null_terminated or nullable or deref:
                        raise NotImplementedError(
                            "SCALAR composes only with SIZE"
                        )
                    width = size_arg.split(":", 1)[1] if size_arg else None
                    if width is None and param.type.ptr_to.format() in (
                        "void",
                        "const void",
                    ):
                        raise NotImplementedError("SCALAR on void needs SIZE")
                    operations.append(
                        ScalarOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            mode=params[0],
                            width=width,
                        )
                    )
                    continue

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
    # The kind names the selector, so a new kind needs one but no change here.
    if kind is None:
        return "lupine_route_for_default()"
    route = f"lupine_route_for_{kind.lower()}"
    if kind == "CURRENT_CONTEXT":
        return f"{route}()"
    if param is None:
        raise NotImplementedError(f"Routing key {kind} requires a parameter")
    name = param.name
    if kind == "DEVICE":
        # By address: the selector rewrites the ordinal to the server's.
        name = f"&{name}"
    elif kind == "FUNCTION" and param.type.format() == "CUkernel":
        name = f"reinterpret_cast<CUfunction>({name})"
    elif kind == "STREAM":
        # The default stream belongs to no route, so a null handle falls back.
        fallback = (
            client_routing_key_expr(
                metadata.routing_fallback.kind,
                metadata.routing_fallback.parameter,
                metadata,
            )
            if metadata.routing_fallback is not None
            else "lupine_route_for_default()"
        )
        return f"({name} != nullptr ? {route}({name}) : {fallback})"
    return f"{route}({name})"


def client_record_owner_stmt(owner: OwnerAnnotation) -> str:
    name = owner.parameter.name
    output = isinstance(owner.parameter.type, Pointer)
    value = f"*{name}" if output else name
    null_guard = f" && {name} != nullptr" if output else ""
    return (
        f"    if (return_value == CUDA_SUCCESS{null_guard}) {{\n"
        f"        lupine_note_{owner.kind.lower()}_owner_route({value}, route);\n"
        "    }\n"
    )


def write_client_post_call(f, metadata: FunctionAnnotationMetadata):
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

    if metadata.client_call_template is not None:
        f.write(textwrap.indent(metadata.client_call_template.after_call, "    "))

    if metadata.synchronize:
        f.write("    if (return_value == CUDA_SUCCESS) return_value = lupine_sync_mapped_device_to_host();\n")


# The results generated code produces on its own. An entry naming a type rather
# than a code marks a result that carries no error value.
ResultCodes = namedtuple(
    "ResultCodes", "unavailable invalid_device invalid_argument"
)

RESULT_CODES = {
    "CUresult": ResultCodes(
        "CUDA_ERROR_DEVICE_UNAVAILABLE",
        "CUDA_ERROR_INVALID_DEVICE",
        "CUDA_ERROR_INVALID_VALUE",
    ),
    "cudaError_t": ResultCodes(
        "cudaErrorDevicesUnavailable",
        "cudaErrorInvalidDevice",
        "cudaErrorInvalidValue",
    ),
    "hipError_t": ResultCodes(
        "hipErrorUnknown", "hipErrorInvalidDevice", "hipErrorInvalidValue"
    ),
    "nvmlReturn_t": ResultCodes("NVML_ERROR_GPU_IS_LOST", None, "NVML_ERROR_INVALID_ARGUMENT"),
    "cublasStatus_t": ResultCodes(
        "CUBLAS_STATUS_NOT_INITIALIZED",
        None,
        "CUBLAS_STATUS_INVALID_VALUE",
    ),
    "cudnnStatus_t": ResultCodes("CUDNN_STATUS_NOT_INITIALIZED", None, "CUDNN_STATUS_BAD_PARAM"),
    "size_t": ResultCodes("size_t", None, None),
    "const char*": ResultCodes("const char*", None, None),
    "void": ResultCodes("void", None, None),
    "struct cudaChannelFormatDesc": ResultCodes("struct cudaChannelFormatDesc", None, None),
}


def result_code(return_type: str, result: str = "unavailable") -> str:
    codes = RESULT_CODES.get(return_type)
    if codes is None:
        raise NotImplementedError(f"Unknown return type: {return_type}")
    return getattr(codes, result) or codes.unavailable


def collect_backend_functions(
    annotations: ParsedData,
    names=None,
    server_bindings: dict[str, ServerBinding] = {},
    string_length_type: str = None,
):
    """Calls in the order named, or every call the annotation file declares."""
    by_name = {
        function.name.format(): function
        for function in annotations.namespace.functions
    }
    result = []
    for name in names if names is not None else by_name:
        if name in server_bindings:
            continue
        function = by_name.get(name)
        if function is None:
            raise RuntimeError(f"Annotation for {name} not found")
        metadata = parse_annotation(function.doxygen, function.parameters)
        if string_length_type is not None:
            for operation in metadata.operations:
                if isinstance(operation, NullTerminatedOperation):
                    operation.length_type = string_length_type
        result.append((function, function, metadata.operations, metadata))
    return result


HIP_ROOT = (
    os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH") or "/opt/rocm"
)

# List of possible directories to search for header files
COMMON_INCLUDE_DIRS = [
    os.path.join(HIP_ROOT, "include"),
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


def sdk_header(path: str) -> str:
    """The SDK header a target's annotations describe: the one they include."""
    # Only the leading includes are parsed. The rest of an annotation file needs
    # a preprocessor, which would expand the include away before it is read.
    with open(path, encoding="utf-8") as f:
        head = itertools.takewhile(
            lambda line: not line.strip() or line.startswith("#include"),
            f,
        )
        includes = parse_string("".join(head)).includes
    if len(includes) != 1:
        raise RuntimeError(f"{path}: expected one #include of the SDK header")
    return includes[0].filename.strip('<>"')


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


def attach_client_call_template(
    function: Function,
    metadata: FunctionAnnotationMetadata,
    client_call_templates: dict[str, ClientCallTemplate],
) -> None:
    name = function.name.format()
    template = client_call_templates.get(name)
    if template is None:
        return
    if metadata.disabled_client:
        raise RuntimeError(
            f"{name}: client call template cannot disable "
            "client generation"
        )
    return_type = function.return_type.format()
    if template.return_type != return_type:
        raise RuntimeError(
            f"{name}: client call template returns "
            f"{template.return_type}, but the API returns {return_type}"
        )
    metadata.client_call_template = template


def write_rpc_ids(
    functions_with_annotations, annotated_names, forwarded_functions, forwarded_bindings
):
    with open("gen_rpc_ids.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n")
        f.write("// RPC ids are stable 31-bit CRC32 hashes of their operation names.\n\n")

        seen_rpc_ids: dict[int, str] = {}
        emitted_macros: set[str] = set()

        def write_rpc_define(macro_name: str, operation_name: str) -> None:
            if macro_name in emitted_macros:
                return
            value = zlib.crc32(operation_name.encode("utf-8")) & 0x7FFFFFFF
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
        for function, _, _, metadata in forwarded_functions:
            if unsupported(function, metadata) or (
                metadata.disabled_client and metadata.disabled_server
            ):
                continue
            name = function.name.format()
            write_rpc_define(f"RPC_{name}", name)
        for name, binding in forwarded_bindings:
            write_rpc_define(f"RPC_{name}", name)
        f.write("\n")
        for name in PRIVATE_RPC_FUNCTIONS:
            write_rpc_define(f"LUPINE_RPC_{name}", name)


def write_cuda_client(functions_with_annotations, legacy_abi_functions):
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
            'extern "C" CUresult lupine_sync_mapped_device_to_host();\n\n'
        )
        for function, _, _, _ in legacy_abi_functions:
            name = function.name.format()
            f.write(f"#ifdef {name}\n#undef {name}\n#endif\n")
            params = ", ".join(format_function_params(function))
            f.write(
                f'extern "C" {function.return_type.format()} CUDAAPI '
                f"{name}({params});\n\n"
            )
        for function, annotation, operations, metadata in functions_with_annotations:
            # We don't generate client function definitions for client-disabled
            # functions; their RPC/server definitions may still be generated.
            if metadata.disabled_client:
                continue

            if metadata.guard is not None:
                f.write(f"#if {metadata.guard}\n")

            joined_params = ", ".join(format_function_params(function))

            f.write(f"{function.return_type.format()} {function.name.format()}({joined_params})\n")
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
                route_expr = client_routing_key_expr(
                    metadata.routing_kind, metadata.routing_parameter, metadata
                )
                f.write(f"    lupine_route route = {route_expr};\n")
                if metadata.routing_kind == "DEVICE":
                    f.write("    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)\n")
                    invalid = result_code(
                        function.return_type.format(), "invalid_device"
                    )
                    f.write(f"        return {invalid};\n")
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
                    "        return lupine_cuMemcpyDtoD_via_client("
                    f"{copy.dst.name}, {copy.src.name}, {copy.bytes.name}, "
                    f"{stream_arg}, {async_arg});\n"
                )
                f.write("    }\n")
            if metadata.client_call_template is not None:
                f.write(
                    textwrap.indent(metadata.client_call_template.before_call, "    ")
                )
            f.write(f"    {function.return_type.format()} return_value;\n")
            call_args = ", ".join(
                param.name for param in function.parameters if param.name
            )
            helper_args = f", {call_args}" if call_args else ""
            local_call = f'lupine_call_real_cuda_fn("{function.name.format()}"{helper_args})'
            local_post_call = io.StringIO()
            write_client_post_call(local_post_call, metadata)
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
                    f.write(f"    std::size_t {operation.parameter.name}_len = std::strlen({operation.parameter.name}) + 1;\n")
                if isinstance(operation, NullableOperation) and operation.recv:
                    f.write(f"    {operation.ptr.format()} {operation.parameter.name}_null_check;\n")

            # Reject invalid send buffers before lupine_prepare_rpc() flushes
            # pending writes and rpc_write_start_request() acquires the
            # connection's call/write locks. Conditions in the builder below
            # may skip optional writes, but only rpc_write* calls themselves
            # are allowed to fail the builder.
            for operation in operations:
                if isinstance(operation, ArrayOperation):
                    operation.client_preflight(
                        f, result_code(function.return_type.format(), "invalid_argument")
                    )
                elif (
                    isinstance(operation, NullTerminatedOperation)
                    and operation.recv
                ):
                    operation.client_preflight(
                        f, result_code(function.return_type.format(), "invalid_argument")
                    )

            if metadata.async_fire_forget:
                error_return = result_code(function.return_type.format())
                f.write(
                    f"    uint64_t async_sequence = 0;\n"
                    f"    if (lupine_prepare_rpc(conn) < 0 ||\n"
                    f"        rpc_write_start_async_request(\n"
                    f"            conn, RPC_{function.name.format()}, &async_sequence) < 0 ||\n"
                    f"        rpc_write(conn, &async_sequence, sizeof(async_sequence)) < 0 ||\n"
                )
                for operation in operations:
                    operation.client_rpc_write(f)
                f.write("        rpc_write_end(conn) < 0) {\n")
                f.write(f"        return {error_return};\n")
                f.write("    }\n")
                post_call = io.StringIO()
                write_client_post_call(post_call, metadata)
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
                f"    if (lupine_prepare_rpc(conn) < 0 ||\n"
                f"        rpc_write_start_request(conn, RPC_{function.name.format()}) < 0 ||\n"
            )

            for operation in operations:
                operation.client_rpc_write(f)

            f.write("        rpc_wait_for_response(conn) < 0 ||\n")

            if metadata.synchronize and metadata.synchronize.deferred_dtoh:
                f.write("        lupine_read_deferred_dtoh_copies(conn) < 0 ||\n")
            if metadata.synchronize and metadata.synchronize.stdout:
                f.write("        lupine_forward_remote_stdout(conn) < 0 ||\n")

            for operation in operations:
                operation.client_rpc_read(f)

            f.write(f"        rpc_read(conn, &return_value, sizeof({function.return_type.format()})) < 0 ||\n")
            f.write("        rpc_read_end(conn) < 0)\n")
            f.write(f"        return {result_code(function.return_type.format())};\n")

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

            write_client_post_call(f, metadata)
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
            f.write(f"#ifdef {alias}\n#undef {alias}\n#endif\n")
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
                f.write(f"    {call};\n")
                f.write("}\n\n")
            else:
                f.write(f"    return {call};\n")
                f.write("}\n\n")
        f.write("std::unordered_map<std::string, void *> functionMap = {\n")
        for function, _, _, metadata in functions_with_annotations:
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
        for alias, target, guard in FUNCTION_MAP_ALIASES:
            if target not in function_names:
                continue
            f.write(f"#if {guard}\n")
            f.write(
                '    {{"{alias}", (void *){target}}},\n'.format(
                    alias=alias,
                    target=target,
                )
            )
            f.write("#endif\n")
        f.write("};\n\n")

        f.write("void *get_function_pointer(const char *name)\n")
        f.write("{\n")
        f.write("    auto it = functionMap.find(name);\n")
        f.write("    if (it == functionMap.end())\n")
        f.write("        return nullptr;\n")
        f.write("    return it->second;\n")
        f.write("}\n")


def write_cuda_server(
    server_functions_with_annotations,
    legacy_abi_functions,
    server_bindings,
    annotation_only_server_functions,
):
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
            '#include "cuda_server_memcpy.h"\n'
            '#include "rpc.h"\n\n'
        )
        annotation_only_functions = (
            legacy_abi_functions + annotation_only_server_functions
        )
        for function, _, _, _ in annotation_only_functions:
            name = function.name.format()
            f.write(f"#ifdef {name}\n#undef {name}\n#endif\n")
            params = ", ".join(format_function_params(function))
            f.write(
                f'extern "C" {function.return_type.format()} CUDAAPI '
                f"{name}({params});\n\n"
            )
        for function, _, operations, metadata in server_functions_with_annotations:
            if metadata.disabled_server or function.name.format() in server_bindings:
                continue
            write_server_handler(f, CUDA, function, operations, metadata)


def write_registry(registry_entries, guarded_declarations, guarded_handlers):
    with open("registry.cpp", "w") as f:
        f.write(
            REGISTRY_CPP_TEMPLATE.substitute(
                cuda_registry_entries=" \\\n".join(registry_entries["CUDA"]),
                cudart_registry_entries=" \\\n".join(
                    registry_entries["CUDART"]
                ),
                cublas_registry_entries=" \\\n".join(
                    registry_entries["CUBLAS"]
                ),
                nvml_registry_entries=" \\\n".join(registry_entries["NVML"]),
                hip_registry_entries=" \\\n".join(registry_entries["HIP"]),
                cuda_guarded_declarations="\n".join(
                    guarded_declarations["CUDA"]
                ),
                cudart_guarded_declarations="\n".join(
                    guarded_declarations["CUDART"]
                ),
                cublas_guarded_declarations="\n".join(
                    guarded_declarations["CUBLAS"]
                ),
                nvml_guarded_declarations="\n".join(
                    guarded_declarations["NVML"]
                ),
                hip_guarded_declarations="\n".join(
                    guarded_declarations["HIP"]
                ),
                cuda_guarded_handlers="\n".join(guarded_handlers["CUDA"]),
                cudart_guarded_handlers="\n".join(guarded_handlers["CUDART"]),
                cublas_guarded_handlers="\n".join(guarded_handlers["CUBLAS"]),
                nvml_guarded_handlers="\n".join(guarded_handlers["NVML"]),
                hip_guarded_handlers="\n".join(guarded_handlers["HIP"]),
            )
        )


def main():
    cuda_header = find_header_file("cuda.h")
    hip_header = find_header_file("hip_runtime_api.h")

    cuda_include_dir = os.path.dirname(cuda_header)
    hip_include_dir = os.path.dirname(os.path.dirname(hip_header))
    options = ParserOptions(
        preprocessor=make_gcc_preprocessor(
            # cublas_api.h refuses direct inclusion until its umbrella
            # header has defined this marker.
            defines=["__HIP_PLATFORM_AMD__", "CUBLASAPI="],
            include_paths=[cuda_include_dir, hip_include_dir],
        ),
    )

    # Parse the files
    cuda_ast: ParsedData = parse_file(cuda_header, options=options)
    annotations_by_target = {
        target: parse_file(path, options=options)
        for target, path in ANNOTATION_FILES.items()
    }
    cuda_annotations = annotations_by_target["cuda"]
    annotations = SimpleNamespace(
        namespace=SimpleNamespace(
            functions=[
                function
                for parsed in annotations_by_target.values()
                for function in parsed.namespace.functions
            ]
        )
    )
    definition_return_types = {
        function.name.format(): function.return_type.format()
        for function in cuda_annotations.namespace.functions
        if function.has_body
    }
    client_call_templates = collect_client_call_templates(
        ANNOTATION_FILES["cuda"], definition_return_types
    )
    server_bindings = {}
    # A handler belongs to the backend whose annotation file declares it.
    for target, path in ANNOTATION_FILES.items():
        for name, binding in collect_server_bindings(path, target.upper()).items():
            if name in server_bindings and server_bindings[name] != binding:
                raise RuntimeError(f"Conflicting @disabled for {name}")
            server_bindings[name] = binding
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
        attach_client_call_template(function, metadata, client_call_templates)
        validate_async_annotation(function, metadata)
        functions_with_annotations.append(
            (function, annotation, metadata.operations, metadata)
        )

    # Generate explicitly listed legacy ABI entry points that cuda.h hides
    # behind macros. A legacy entry point may still use @disabled server when
    # its server handler needs lifecycle management beyond ordinary marshalling.
    server_functions_with_annotations = list(functions_with_annotations)
    server_function_names = {
        function.name.format()
        for function, _, _, _ in server_functions_with_annotations
    }
    legacy_abi_functions = []
    annotation_only_server_functions = []
    for annotation in cuda_annotations.namespace.functions:
        name = annotation.name.format()
        if name in server_function_names:
            continue
        directives = annotation_directives(annotation.doxygen)
        legacy_abi = name in LEGACY_ABI_FUNCTIONS
        client_disabled = any(
            directive.startswith("@disabled client")
            for directive in directives
        )
        if not legacy_abi and not client_disabled:
            continue
        if not legacy_abi and any(
            directive.startswith("@disabled")
            and not directive.startswith("@disabled client")
            for directive in directives
        ):
            continue
        metadata = parse_annotation(annotation.doxygen, annotation.parameters)
        attach_client_call_template(annotation, metadata, client_call_templates)
        validate_async_annotation(annotation, metadata)
        annotated_function = (
            annotation,
            annotation,
            metadata.operations,
            metadata,
        )
        if legacy_abi:
            if metadata.disabled_client:
                raise RuntimeError(
                    f"{name}: legacy ABI function cannot disable the client"
                )
            functions_with_annotations.append(annotated_function)
            legacy_abi_functions.append(annotated_function)
        server_functions_with_annotations.append(annotated_function)
        if not legacy_abi:
            annotation_only_server_functions.append(annotated_function)
        server_function_names.add(name)

    attached_client_call_templates = {
        function.name.format()
        for function, _, _, metadata in functions_with_annotations
        if metadata.client_call_template is not None
    }
    unused_client_call_templates = (
        set(client_call_templates) - attached_client_call_templates
    )
    if unused_client_call_templates:
        raise RuntimeError(
            "client call templates do not match generated CUDA functions: "
            + ", ".join(sorted(unused_client_call_templates))
        )

    found_legacy_abi_functions = {
        function.name.format() for function, _, _, _ in legacy_abi_functions
    }
    missing_legacy_abi_functions = (
        LEGACY_ABI_FUNCTIONS - found_legacy_abi_functions
    )
    if missing_legacy_abi_functions:
        raise RuntimeError(
            "Legacy ABI annotations not found: "
            + ", ".join(sorted(missing_legacy_abi_functions))
        )

    nvml_functions_with_annotations = collect_backend_functions(
        annotations_by_target["nvml"],
        NVML_RPC_FUNCTIONS,
        server_bindings,
        # NVML's protocol predates the size_t string lengths CUDA RPC uses.
        string_length_type="unsigned int",
    )
    hip_functions_with_annotations = collect_backend_functions(
        annotations_by_target["hip"]
    )
    cudart_functions_with_annotations = collect_backend_functions(
        annotations_by_target["cudart"]
    )
    cublas_functions_with_annotations = collect_backend_functions(
        annotations_by_target["cublas"]
    )
    # The SDK's own declarations, so a call an annotation file leaves out still
    # gets a symbol.
    forwarding_backends = [
        (HIP, hip_functions_with_annotations),
        (CUDART, cudart_functions_with_annotations),
        (CUBLAS, cublas_functions_with_annotations),
    ]
    sdk_functions_for = {
        backend.name: parse_file(
            find_header_file(sdk_header(ANNOTATION_FILES[backend.name])),
            options=options,
        ).namespace.functions
        for backend, _ in forwarding_backends
    }

    annotated_names = sorted(
        {function.name.format() for function in cuda_annotations.namespace.functions}
        | {
            name
            for name, binding in server_bindings.items()
            if binding.backend == "CUDA"
        }
    )

    write_rpc_ids(
        functions_with_annotations,
        annotated_names,
        [entry for _, functions in forwarding_backends for entry in functions],
        [
            (name, binding)
            for name, binding in server_bindings.items()
            if binding.backend in {backend.name.upper() for backend, _ in forwarding_backends}
        ],
    )

    write_backend(NVML, nvml_functions_with_annotations)
    for backend, functions in forwarding_backends:
        write_backend(backend, functions, sdk_functions_for[backend.name])

    write_cuda_client(functions_with_annotations, legacy_abi_functions)

    write_cuda_server(
        server_functions_with_annotations,
        legacy_abi_functions,
        server_bindings,
        annotation_only_server_functions,
    )

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
    for backend, functions in forwarding_backends:
        generated_bindings.extend(
            ServerBinding(
                function.name.format(),
                backend.name.upper(),
                f"handle_{function.name.format()}",
                metadata.guard,
            )
            for function, _, _, metadata in functions
            if not metadata.disabled_server
            and not unsupported(function, metadata)
            and function.name.format() not in server_bindings
        )
    bindings = list(server_bindings.values()) + generated_bindings

    operations_by_id = {}
    for binding in bindings:
        operation = zlib.crc32(binding.name.encode("utf-8")) & 0x7FFFFFFF
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

    write_registry(registry_entries, guarded_declarations, guarded_handlers)

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
            "gen_hip_client.inc",
            "gen_hip_server.inc",
            "gen_hip_server.h",
            "gen_cudart_client.inc",
            "gen_cudart_server.inc",
            "gen_cudart_server.h",
            "gen_cublas_client.inc",
            "gen_cublas_server.inc",
            "gen_cublas_server.h",
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
        "hip": [
            "gen_hip_client.inc",
            "gen_hip_server.inc",
            "gen_hip_server.h",
        ],
        "cudart": [
            "gen_cudart_client.inc",
            "gen_cudart_server.inc",
            "gen_cudart_server.h",
        ],
        "cublas": [
            "gen_cublas_client.inc",
            "gen_cublas_server.inc",
            "gen_cublas_server.h",
        ],
    }
    forbidden = {
        "cuda": ["nvml", "hip"],
        "cudart": ["nvml", "hip"],
        "cublas": ["nvml", "hip"],
        "nvml": ["cuda_compat", "<cuda.h>", "handle_cu", "hip"],
        "hip": ["cuda", "nvml"],
    }
    selected = (
        backend_files if backend == "all" else {backend: backend_files[backend]}
    )
    for name, paths in selected.items():
        for path in paths:
            contents = open(path, encoding="utf-8").read().lower()
            for token in forbidden[name]:
                if token.lower() in contents:
                    raise RuntimeError(
                        f"{path} contains forbidden {name} boundary token: {token}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-backend",
        choices=("all", "cuda", "nvml", "hip", "cudart", "cublas"),
        help="verify existing generated files without loading backend SDK headers",
    )
    args = parser.parse_args()
    # Annotation inputs and gen_* outputs are CWD-relative.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.verify_backend:
        verify_backend_boundaries(args.verify_backend)
    else:
        main()
        verify_backend_boundaries("all")
