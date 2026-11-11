# /// script
# requires-python = ">=3.10"
# dependencies = ["clang-format==18.1.3", "cxxheaderparser==1.9.2"]
# ///
from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function
from typing import Optional
from dataclasses import dataclass
from string import Template
from types import SimpleNamespace
import argparse
from collections import namedtuple
import io
import os
import glob
import re
import subprocess
import textwrap
import zlib
from client_templates import collect_client_call_templates
from emit import (
    REQUEST_ORDER,
    Backend,
    format_call_args,
    format_function_params,
    write_client_rpc,
    write_client_wrapper,
    write_server_handler,
    unsupported,
    write_scalar_slot,
    write_stub,
    write_profiling_aliases,
    write_versioned_struct_helper,
)
from ops import (
    unanswered,
    NullableOperation,
    ArrayOperation,
    InOutCountOperation,
    NullableArrayOperation,
    DeepStructOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    ScalarOperation,
    VersionedStructOperation,
    Operation,
    OwnerAnnotation,
    RetainAnnotation,
    ReleaseAnnotation,
    ParentAnnotation,
    CrossServerCopyAnnotation,
    ClientCallTemplate,
    FunctionAnnotationMetadata,
    GraphExecNodeAnnotation,
    GraphExecUpdateAnnotation,
    RoutingFallbackAnnotation,
    SynchronizeAnnotation,
)

# CUDA headers omit some legacy ABI entry points from their public declarations.
# Keep the small set that still needs RPC wrappers explicit here.
LEGACY_ABI_FUNCTIONS = {
    "cuGraphExecUpdate",
    "cuGraphInstantiate_v2",
    "cuMemAdvise",
    "cuMemPrefetchAsync",
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
    ("cuStreamDestroy", "cuStreamDestroy_v2"),
    ("cuEventDestroy", "cuEventDestroy_v2"),
    ("cuEventElapsedTime", "cuEventElapsedTime_v2"),
    ("cuStreamGetDevice_ptsz", "cuStreamGetDevice"),
    ("cuMemGetAddressRange", "cuMemGetAddressRange_v2"),
    ("cuMemGetInfo", "cuMemGetInfo_v2"),
    ("cuLinkCreate", "cuLinkCreate_v2"),
    ("cuLinkAddData", "cuLinkAddData_v2"),
    ("cuLinkAddFile", "cuLinkAddFile_v2"),
    ("cuLaunchCooperativeKernel_ptsz", "cuLaunchCooperativeKernel"),
    ("cuGraphKernelNodeGetParams", "cuGraphKernelNodeGetParams_v2"),
    ("cuGraphKernelNodeSetParams", "cuGraphKernelNodeSetParams_v2"),
    ("cuGraphAddKernelNode", "cuGraphAddKernelNode_v2"),
    ("cuGraphExecKernelNodeSetParams", "cuGraphExecKernelNodeSetParams_v2"),
    ("cuStreamWaitValue32", "cuStreamWaitValue32_v2"),
    ("cuStreamWaitValue64", "cuStreamWaitValue64_v2"),
    ("cuStreamWriteValue32", "cuStreamWriteValue32_v2"),
    ("cuStreamWriteValue64", "cuStreamWriteValue64_v2"),
    ("cuStreamBatchMemOp", "cuStreamBatchMemOp_v2"),
    ("cuArrayCreate", "cuArrayCreate_v2"),
    ("cuArray3DCreate", "cuArray3DCreate_v2"),
    ("cuArray3DGetDescriptor", "cuArray3DGetDescriptor_v2"),
    ("cuMemAllocHost", "cuMemAllocHost_v2"),
    ("cuMemHostGetDevicePointer", "cuMemHostGetDevicePointer_v2"),
    ("cuMemFree", "cuMemFree_v2"),
    ("cuMemcpyDtoA", "cuMemcpyDtoA_v2"),
    ("cuMemcpyAtoD", "cuMemcpyAtoD_v2"),
    ("cuMemcpyAtoA", "cuMemcpyAtoA_v2"),
    ("cuMemcpyHtoA", "cuMemcpyHtoA_v2"),
    ("cuMemcpyHtoA_v2_ptds", "cuMemcpyHtoA_v2"),
    ("cuMemcpyHtoAAsync", "cuMemcpyHtoAAsync_v2"),
    ("cuMemcpyHtoAAsync_v2_ptsz", "cuMemcpyHtoAAsync_v2"),
    ("cuMemcpyAtoH", "cuMemcpyAtoH_v2"),
    ("cuMemcpyAtoH_v2_ptds", "cuMemcpyAtoH_v2"),
    ("cuMemcpyAtoHAsync", "cuMemcpyAtoHAsync_v2"),
    ("cuMemcpyAtoHAsync_v2_ptsz", "cuMemcpyAtoHAsync_v2"),
    ("cuMemcpy2D", "cuMemcpy2D_v2"),
    ("cuMemcpy2DUnaligned", "cuMemcpy2DUnaligned_v2"),
    ("cuMemcpy2DAsync", "cuMemcpy2DAsync_v2"),
    ("cuMemcpy2DAsync_ptsz", "cuMemcpy2DAsync_v2"),
    ("cuMemcpy3D", "cuMemcpy3D_v2"),
    ("cuMemcpy3DAsync", "cuMemcpy3DAsync_v2"),
    ("cuMemcpy3DAsync_ptsz", "cuMemcpy3DAsync_v2"),
    ("cuMemcpyPeer_ptds", "cuMemcpyPeer"),
    ("cuMemcpyPeerAsync_ptsz", "cuMemcpyPeerAsync"),
    ("cuMemcpy3DPeer_ptds", "cuMemcpy3DPeer"),
    ("cuMemcpy3DPeerAsync_ptsz", "cuMemcpy3DPeerAsync"),
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
    ("cuStreamUpdateCaptureDependencies_v2_ptsz", "cuStreamUpdateCaptureDependencies_v2"),
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
    (
        "cuGraphAddDependencies",
        "cuGraphAddDependencies_v2",
        "CUDA_VERSION >= 12030",
    ),
    (
        "cuGraphRemoveDependencies",
        "cuGraphRemoveDependencies_v2",
        "CUDA_VERSION >= 12030",
    ),
    (
        "cuStreamUpdateCaptureDependencies",
        "cuStreamUpdateCaptureDependencies_v2",
        "CUDA_VERSION >= 12030",
    ),
    (
        "cuStreamUpdateCaptureDependencies_ptsz",
        "cuStreamUpdateCaptureDependencies_v2",
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
    "nvmlDeviceGetComputeRunningProcesses_v3",
    "nvmlDeviceGetProcessUtilization",
    "nvmlDeviceGetGraphicsRunningProcesses",
    "nvmlDeviceGetGraphicsRunningProcesses_v2",
    "nvmlDeviceGetGraphicsRunningProcesses_v3",
    "nvmlDeviceGetMPSComputeRunningProcesses",
    "nvmlDeviceGetMPSComputeRunningProcesses_v2",
    "nvmlDeviceGetMPSComputeRunningProcesses_v3",
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

HIP_MANUAL_REMAPPINGS = [
    ("hipGetDeviceProperties", "hipGetDevicePropertiesR0600"),
    ("hipChooseDevice", "hipChooseDeviceR0600"),
]

PRIVATE_RPC_FUNCTIONS = [
    "cuGetExportTableMetadata",
    "cuPrivateGetModuleNode",
    "cuStreamBeginCaptureToGraph",
    "cuStreamGetCaptureInfo_v3",
    "lupineDeviceSnapshot",
    "lupineFunctionAttributeSnapshot",
    "lupineFunctionParamLayoutSnapshot",
    "lupineManagedHostFlush",
    "lupineBulkChunk",
    "lupineMemcpyHtoDBulk",
    "lupineBulkRead",
    "lupineMemcpyDtoHBulk",
    "lupineDtohPush",
    "lupineStreamPoolInit",
    "lupineEventCreateBatch",
    "lupineLibraryLoadBatch",
    "lupineDeviceOpen",
    "lupineDeviceClose",
    "lupineDeviceIoctl",
]

REGISTRY_CPP_TEMPLATE = Template(
    r'''#include "rpc_server.h"

#ifdef LUPINE_BUILD_CUDA_BACKEND
#include <cuda.h>
#endif
#ifdef LUPINE_BUILD_NCCL_BACKEND
#include <nccl.h>
#endif
#include "gen_rpc_ids.h"

// clang-format off
#define LUPINE_CUDA_RPC_HANDLERS(HANDLER) \
$cuda_registry_entries
#define LUPINE_NCCL_RPC_HANDLERS(HANDLER) \
$nccl_registry_entries
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
#endif
#ifdef LUPINE_BUILD_NCCL_BACKEND
LUPINE_NCCL_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
$nccl_guarded_declarations
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
#endif
#ifdef LUPINE_BUILD_NCCL_BACKEND
      LUPINE_NCCL_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
$nccl_guarded_handlers
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
#undef LUPINE_NCCL_RPC_HANDLERS
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
    "NCCL": "rpc_backend::nccl",
    "NVML": "rpc_backend::nvml",
    "HIP": "rpc_backend::hip",
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
            if rest and rest[0] == "none":
                return None
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
    result="CUresult",
    invalid_argument="CUDA_ERROR_INVALID_VALUE",
    device_routing_kind="DEVICE",
    # The driver shim links against libcuda, so its handlers call the entry
    # point directly instead of resolving it by name.
    symbol_lookup="",
    not_supported="CUDA_ERROR_NOT_SUPPORTED",
    success="CUDA_SUCCESS",
    invalid_result="CUDA_ERROR_UNKNOWN",
    keep_async_error="lupine_keep_async_error",
)

NVML = Backend(
    result="nvmlReturn_t",
    invalid_argument="NVML_ERROR_INVALID_ARGUMENT",
    device_routing_kind="NVML_DEVICE",
    symbol_lookup="nvml_symbol",
    lookup_on_all_connections=True,
    success="NVML_SUCCESS",
    invalid_result="NVML_ERROR_UNKNOWN",
)

HIP = Backend(
    result="hipError_t",
    invalid_argument="hipErrorInvalidValue",
    device_routing_kind="HIP_DEVICE",
    symbol_lookup="hip_symbol",
    guard_null_conn=True,
    not_supported="hipErrorNotSupported",
)

# A communicator is a pointer the server's library hands out and routes to
# the connection that created it. Collectives submitted inside a group are
# fire-and-forget: NCCL only queues them until ncclGroupEnd.
NCCL = Backend(
    result="ncclResult_t",
    invalid_argument="ncclInvalidArgument",
    device_routing_kind="DEVICE",
    symbol_lookup="nccl_symbol",
    guard_null_conn=True,
    not_supported="ncclInvalidUsage",
    async_success="ncclSuccess",
    alias_prefix="p",
    stream_less_scoped=True,
)

ANNOTATION_FILES = {
    "cuda": "annotations_cuda.h",
    "nccl": "annotations_nccl.h",
    "nvml": "annotations_nvml.h",
    "hip": "annotations_hip.h",
}


def annotation_param(params: list[Parameter], name: str) -> Parameter:
    try:
        return next(p for p in params if p.name == name)
    except StopIteration:
        raise NotImplementedError(f"Parameter {name} not found")


# Types whose value is an address in one server's library.
LIBRARY_HANDLES = {
    "ncclComm_t",
    "ncclParamHandle_t",
}


def infer_routing_key(
    params: list[Parameter],
) -> tuple[Optional[str], Optional[Parameter]]:
    for param in params:
        if isinstance(param.type, Pointer):
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
        # A library handle is created on one server and routes every later
        # call there.
        if type_name in LIBRARY_HANDLES:
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
    annotation: str,
    params: list[Parameter],
    disabled_client_non_nullable: bool = False,
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
            scope = line.lstrip(" *").split()[1:2]
            if scope == ["client"]:
                metadata.disabled_client = True
            elif scope == ["server"]:
                metadata.disabled_server = True
            else:
                metadata.disabled_client = True
                metadata.disabled_server = True
    for line in annotation.split("\n"):
        if "@disabled" in line:
            continue
        if line.startswith("/**"):
            continue
        if line.startswith("*/"):
            continue
        if line.startswith("*"):
            line = line[2:]
        if line.strip().startswith("@async"):
            raise NotImplementedError(
                "@async is inferred; mark a status-only call that must wait @blocking"
            )
        if line.strip().startswith("@blocking"):
            metadata.blocking = True
            continue
        if line.strip().startswith("@synchronize"):
            parts = line.split()
            if parts[1:2] == ["DEVICEPTR"]:
                if len(parts) != 3:
                    raise RuntimeError("@synchronize DEVICEPTR requires a parameter")
                metadata.device_writes.append(annotation_param(params, parts[2]))
                continue
            options = set(parts[1:])
            unknown = options - {"STDOUT"}
            if unknown:
                raise NotImplementedError(
                    "Unknown @synchronize option(s): " + ", ".join(sorted(unknown))
                )
            metadata.synchronize = SynchronizeAnnotation(stdout="STDOUT" in options)
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
        if line.startswith("@graphexecupdate"):
            parts = line.split()
            if len(parts) != 4 or metadata.graph_exec_update is not None:
                raise RuntimeError(
                    "@graphexecupdate requires graph exec, graph and error parameters"
                )
            metadata.graph_exec_update = GraphExecUpdateAnnotation(
                graph_exec=annotation_param(params, parts[1]),
                graph=annotation_param(params, parts[2]),
                error=annotation_param(params, parts[3]),
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
                if "ON_SUCCESS" in args and "ON_ERROR" in args:
                    raise NotImplementedError("ON_SUCCESS and ON_ERROR are mutually exclusive")
                recv_on = next((arg for arg in args if arg in ("ON_SUCCESS", "ON_ERROR")), None)
                scalar_arg = next(
                    (arg for arg in args if arg.split(":")[0] == "SCALAR"), None
                )
                if recv_on and (
                    deref or scalar_arg or "VERSIONED" in args or size_arg
                    or null_terminated
                    or not (nullable and length_arg and recv and not send)
                ):
                    raise NotImplementedError(
                        "ON_SUCCESS/ON_ERROR require a RECV_ONLY NULLABLE LENGTH "
                        "out-array without conflicting transfer modifiers"
                    )

                if "VERSIONED" in args:
                    # VERSIONED [STRINGS:<member>,...] [CLEARED:<member>,...]
                    # [MEMBERGUARD:<member>=<condition>...]:
                    # an optional size-led configuration struct whose string
                    # members travel as text.
                    if not send or recv:
                        raise NotImplementedError("VERSIONED is SEND_ONLY")

                    def members(prefix):
                        arg = next((a for a in args if a.startswith(prefix)), None)
                        return tuple(arg.split(":", 1)[1].split(",")) if arg else ()

                    # MEMBERGUARD:<member>=<condition> names the preprocessor
                    # condition under which the struct declares the member.
                    member_guards = dict(
                        a.split(":", 1)[1].split("=", 1)
                        for a in args
                        if a.startswith("MEMBERGUARD:")
                    )
                    operations.append(
                        VersionedStructOperation(
                            parameter=param,
                            ptr=param.type,
                            strings=members("STRINGS:"),
                            cleared=members("CLEARED:"),
                            member_guards=member_guards,
                        )
                    )
                    continue

                if scalar_arg is not None:
                    # SCALAR[:<owner>]: a pointer-mode scalar, sized by
                    # SIZE:<expr> when the pointee is void. The owner is the
                    # handle or descriptor whose pointer mode decides where it
                    # lives; the call's first parameter when unnamed.
                    if length_arg or null_terminated or nullable or deref:
                        raise NotImplementedError(
                            "SCALAR composes only with SIZE"
                        )
                    owner = (
                        annotation_param(params, scalar_arg.split(":", 1)[1])
                        if ":" in scalar_arg
                        else params[0]
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
                            mode=owner,
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

                deref_nullable = not (
                    (metadata.disabled_client and disabled_client_non_nullable)
                    or metadata.disabled_server
                )
                if deref:
                    operations.append(
                        NullableOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            nullable=deref_nullable,
                        )
                    )
                elif length_arg:
                    # if it has a length, it's an array operation with variable length
                    length_spec = length_arg.split(":", 1)[1]
                    if "-" in length_spec:
                        length_name, offset_str = length_spec.split("-", 1)
                        length_offset = -int(offset_str)
                    else:
                        length_name = length_spec
                        length_offset = 0
                    length_param = next(
                        (p for p in params if p.name == length_name), None
                    )
                    if length_param is None:
                        raise NotImplementedError(
                            f"LENGTH parameter {length_name} not found"
                        )
                    if nullable and send:
                        # SEND_ONLY NULLABLE LENGTH: an optional in-array the
                        # caller may leave null (ncclCommInitAll's devlist).
                        if recv:
                            raise NotImplementedError(
                                "NULLABLE LENGTH is SEND_ONLY or RECV_ONLY"
                            )
                        operations.append(
                            ArrayOperation(
                                send=True,
                                recv=False,
                                parameter=param,
                                ptr=param.type,
                                length=length_param,
                                nullable=True,
                                length_offset=length_offset,
                            )
                        )
                    elif nullable:
                        # NULLABLE LENGTH: an optional out-array sized by an
                        # in/out count param (the cuGraphGetNodes query
                        # pattern); linked to its count in the post-pass below.
                        if not recv:
                            raise NotImplementedError(
                                "NULLABLE LENGTH requires a RECV_ONLY out-array"
                            )
                        operations.append(
                            NullableArrayOperation(
                                parameter=param,
                                ptr=param.type,
                                count=length_param,
                                recv_on=recv_on,
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
                                nullable=deref_nullable,
                                length_offset=length_offset,
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
                            nullable=deref_nullable,
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
                            nullable=deref_nullable,
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
                            NullableOperation(
                                send=send,
                                recv=recv,
                                parameter=param,
                                ptr=param.type,
                                nullable=deref_nullable,
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
                    op, NullableOperation
                ):
                    operations[i] = InOutCountOperation(
                        send=True,
                        recv=True,
                        parameter=op.parameter,
                        anchor=anchor,
                    )
                    break
    if metadata.routing_kind is None:
        metadata.routing_kind, metadata.routing_parameter = infer_routing_key(params)
    # Promote any param with @deeparray entries to a DeepStructOperation,
    # inheriting the send/recv direction from its @param line.
    for pname, members in deep_arrays.items():
        for i, op in enumerate(operations):
            if op.parameter.name == pname:
                node_owner = None
                if getattr(op, "recv", False) and not getattr(op, "send", True):
                    if metadata.routing_kind != "GRAPH_NODE":
                        raise NotImplementedError(
                            "RECV_ONLY @deeparray requires a graph-node owner"
                        )
                    node_owner = metadata.routing_parameter.name
                operations[i] = DeepStructOperation(
                    send=getattr(op, "send", True),
                    recv=getattr(op, "recv", False),
                    parameter=op.parameter,
                    ptr=op.parameter.type,
                    members=members,
                    node_owner=node_owner,
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

    return metadata


def client_routing_key_expr(
    kind: Optional[str],
    param: Optional[Parameter],
    metadata: FunctionAnnotationMetadata,
    allow_fallback: bool = True,
) -> str:
    # The kind names the selector, so a new kind needs one but no change here.
    if kind is None:
        return "lupine_route_for_default()"
    route = f"lupine_route_for_{kind.lower()}"
    if kind in ("CURRENT_CONTEXT", "DEFAULT"):
        return f"{route}()"
    if param is None:
        raise NotImplementedError(f"Routing key {kind} requires a parameter")
    name = param.name
    if kind == "DEVICE":
        # By address: the selector rewrites the ordinal to the server's.
        name = f"&{name}"
    elif kind == "FUNCTION" and param.type.format() == "CUkernel":
        name = f"reinterpret_cast<CUfunction>({name})"
    elif kind == "DEVICEPTR" and isinstance(param.type, Pointer):
        name = f"reinterpret_cast<CUdeviceptr>({name})"
    elif kind == "STREAM" or (
        allow_fallback and metadata.routing_fallback is not None
    ):
        # The default stream belongs to no route, so a null handle falls back.
        fallback = (
            client_routing_key_expr(
                metadata.routing_fallback.kind,
                metadata.routing_fallback.parameter,
                metadata,
                allow_fallback=False,
            )
            if allow_fallback and metadata.routing_fallback is not None
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
    "hipError_t": ResultCodes(
        "hipErrorUnknown", "hipErrorInvalidDevice", "hipErrorInvalidValue"
    ),
    "nvmlReturn_t": ResultCodes("NVML_ERROR_GPU_IS_LOST", None, "NVML_ERROR_INVALID_ARGUMENT"),
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
    client_call_templates: dict[str, ClientCallTemplate] = None,
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
        if client_call_templates is not None:
            attach_client_call_template(function, metadata, client_call_templates)
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


def client_request_start(function: Function, metadata) -> str:
    """The request start for a call, ordered as CUDA orders what it touches.

    A call on a stream is ordered as that stream's work is; a synchronous copy
    or set runs on the legacy stream; an event call by its events; a call that
    waits for the device by every earlier unanswered request; and any other
    call only behind earlier calls that touched no stream or event.
    """
    name = function.name.format()
    op = f"RPC_{name}"

    def first(type_name):
        return [p.name for p in function.parameters if p.type.format() == type_name]

    streams, events = first("CUstream"), first("CUevent")
    execs = first("CUgraphExec")

    def also_on(keys):
        return "".join(
            " < 0 ||\n"
            f"        rpc_request_on(conn, reinterpret_cast<uintptr_t>({key}))"
            for key in keys
        )

    order = REQUEST_ORDER.get(name)
    if order == "all":
        # Waits for everything already; an event it records is published too.
        return f"rpc_write_start_request(conn, {op})" + also_on(events)
    flags = {
        "publish": ", RPC_REQUEST_PUBLISH_CONNECTION",
        "soft": ", RPC_REQUEST_SOFT",
    }.get(order, "")
    if streams:
        start = f"lupine_rpc_start_on_stream(conn, {op}, {streams[0]}{flags})"
        rest = streams[1:] + events
    elif (re.match(r"cuMem(cpy|set)", name) and "Async" not in name) or name in (
        "cuLaunch",
        "cuLaunchGrid",
    ):
        start = f"lupine_rpc_start_on_stream(conn, {op}, nullptr)"
        rest = events
    elif events:
        start = (
            f"rpc_write_start_request_on(conn, {op},\n"
            f"            reinterpret_cast<uintptr_t>({events[0]}))"
        )
        rest = events[1:]
    elif execs:
        start = (
            f"rpc_write_start_request_on(conn, {op},\n"
            f"            reinterpret_cast<uintptr_t>({execs[0]}))"
        )
        rest = execs[1:]
        execs = []
    else:
        start = f"rpc_write_start_request_on(conn, {op}, RPC_SCOPE_NONE)"
        rest = []
    # A graph executable runs one launch at a time, and is updated or
    # destroyed only after the launches its caller made before.
    start += also_on(rest + execs)
    # A semaphore's wait is submitted after the signals sent before it.
    for operation in metadata.operations:
        if (
            isinstance(operation, ArrayOperation)
            and operation.ptr.ptr_to.format() == "const CUexternalSemaphore"
        ):
            start += (
                " < 0 ||\n"
                "        lupine_rpc_request_on_each(conn,\n"
                f"            reinterpret_cast<const void *const *>({operation.parameter.name}),\n"
                f"            {operation.count_expr()})"
            )
    return start


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
    functions_with_annotations,
    annotated_names,
    hip_functions_with_annotations,
    nccl_functions_with_annotations,
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
            line = f"#define {macro_name} {value}"
            if len(line) > 80:
                # Wrapped the way clang-format wraps a long macro, so the
                # format check accepts the generated file.
                line = f"#define {macro_name}".ljust(79) + f"\\\n  {value}"
            f.write(line + "\n")

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
        for functions in (
            hip_functions_with_annotations,
            nccl_functions_with_annotations,
        ):
            for function, _, _, metadata in functions:
                if unsupported(function, metadata):
                    continue
                name = function.name.format()
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
            '#include "cuda_client.h"\n'
            '#include "cuda_client_memcpy.h"\n'
            '#include "events.h"\n'
            '#include "rpc.h"\n\n'
            'extern "C" void *lupine_deep_node_cache_get(CUgraphNode node, size_t slot, size_t bytes);\n'
            'extern "C" void lupine_deep_node_cache_reset(CUgraphNode node);\n'
            'std::vector<CUgraphNode> lupine_deep_cache_graph_nodes(CUgraph graph);\n\n'
            'extern "C" CUfunction lupine_translate_private_function_for_rpc(CUfunction function);\n'
            'extern "C" const char *lupine_retain_returned_string(const void *handle, const char *data, size_t size);\n\n'
            'extern "C" void lupine_release_module_retained_strings(CUmodule module);\n'
            'extern "C" void lupine_release_library_retained_strings(CUlibrary library);\n\n'
            'extern "C" void lupine_record_library_module(CUmodule module, CUlibrary library);\n\n'
            'extern "C" CUresult lupine_record_library_kernel(CUkernel kernel, CUlibrary library, const char *name, lupine_route route);\n\n'
            'extern "C" CUresult lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice,\n'
            '                                                   CUdeviceptr srcDevice,\n'
            '                                                   size_t ByteCount,\n'
            '                                                   CUstream hStream,\n'
            '                                                   bool async);\n\n'
            'extern "C" void lupine_invalidate_current_context_cache();\n'
            'extern "C" void lupine_forget_destroyed_context(CUcontext ctx);\n'
            'extern "C" void lupine_disable_local_occupancy();\n'
            'extern "C" CUresult lupine_resolve_kernel_for_route(CUkernel *kernel, lupine_route route);\n'
            'extern "C" CUresult lupine_resolve_library_for_route(CUlibrary *library, lupine_route route);\n'
            'extern "C" int lupine_read_dtoh_watermark(conn_t *conn);\n'
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

            if unsupported(function, metadata):
                write_stub(f, CUDA, function)
                if metadata.guard is not None:
                    f.write("#endif\n\n")
                continue

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
                if unanswered(function, metadata, "CUresult"):
                    raise RuntimeError(
                        f"{function.name.format()}: ALL routing cannot be fire-and-forget"
                    )

                output_name = all_output.name
                f.write(
                    f"    return lupine_lookup_device_on_all_routes({output_name},\n"
                    "        [&](lupine_route route, CUdevice *route_output) {\n"
                    f"            {all_output.type.format()} {output_name} = route_output;\n"
                )
            else:
                if (
                    metadata.routing_kind == "DEVICE"
                    and metadata.routing_parameter is not None
                    and metadata.client_call_template is not None
                    and f"virtual_{metadata.routing_parameter.name}"
                    in (
                        metadata.client_call_template.before_call
                        + metadata.client_call_template.after_call
                    )
                ):
                    f.write(
                        f"    CUdevice virtual_{metadata.routing_parameter.name} = "
                        f"{metadata.routing_parameter.name};\n"
                    )
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
                for written in metadata.device_writes:
                    f.write(f"    lupine_note_device_write({written.name});\n")
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
            local_call = f'lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("{function.name.format()}"){helper_args})'
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
                f.write(operation.client_declaration())

            unanswered_call = unanswered(function, metadata, "CUresult")
            f.write(
                f"    if (lupine_prepare_rpc(conn) < 0 ||\n"
                f"        {client_request_start(function, metadata)} < 0 ||\n"
            )

            for operation in operations:
                operation.client_rpc_write(f)

            if unanswered_call:
                f.write("        rpc_write_end(conn) < 0)\n")
            else:
                f.write("        rpc_wait_for_response(conn) < 0 ||\n")
                if metadata.synchronize:
                    f.write("        lupine_read_dtoh_watermark(conn) < 0 ||\n")
                if metadata.synchronize and metadata.synchronize.stdout:
                    f.write("        lupine_forward_remote_stdout(conn) < 0 ||\n")
                for operation in operations:
                    operation.client_rpc_read(f)
                f.write(f"        rpc_read(conn, &return_value, sizeof({function.return_type.format()})) < 0 ||\n")
                f.write("        rpc_read_end(conn) < 0)\n")
            f.write(f"        return {result_code(function.return_type.format())};\n")
            if unanswered_call:
                f.write("    return_value = CUDA_SUCCESS;\n")

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
            for function, _, _, _ in functions_with_annotations
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
        for x, y in MANUAL_REMAPPINGS:
            # ensure y exists in the function list
            if y not in function_by_name:
                print(f"Skipping manual remapping {x} -> {y}")
                continue
            f.write(
                '    {{"{x}", (void *){y}}},\n'.format(
                    x=x,
                    y=y,
                )
            )
        for alias, target, guard in FUNCTION_MAP_ALIASES:
            if target not in function_by_name:
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
            '#include "device_stdout.h"\n'
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
            if (
                metadata.disabled_server
                or unsupported(function, metadata)
                or function.name.format() in server_bindings
            ):
                continue
            write_server_handler(f, CUDA, function, operations, metadata)


def write_registry(registry_entries, guarded_declarations, guarded_handlers):
    with open("registry.cpp", "w") as f:
        f.write(
            REGISTRY_CPP_TEMPLATE.substitute(
                cuda_registry_entries=" \\\n".join(registry_entries["CUDA"]),
                nccl_registry_entries=" \\\n".join(
                    registry_entries["NCCL"]
                ),
                nvml_registry_entries=" \\\n".join(registry_entries["NVML"]),
                hip_registry_entries=" \\\n".join(registry_entries["HIP"]),
                cuda_guarded_declarations="\n".join(
                    guarded_declarations["CUDA"]
                ),
                nccl_guarded_declarations="\n".join(
                    guarded_declarations["NCCL"]
                ),
                nvml_guarded_declarations="\n".join(
                    guarded_declarations["NVML"]
                ),
                hip_guarded_declarations="\n".join(
                    guarded_declarations["HIP"]
                ),
                cuda_guarded_handlers="\n".join(guarded_handlers["CUDA"]),
                nccl_guarded_handlers="\n".join(
                    guarded_handlers["NCCL"]
                ),
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
            defines=[
                "__HIP_PLATFORM_AMD__",
                # nccl.h declares ncclResetDebugInit only for Linux builds.
                "NCCL_OS_LINUX",
            ],
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
    client_call_templates_by_target = {
        target: collect_client_call_templates(
            ANNOTATION_FILES[target],
            {
                function.name.format(): function.return_type.format()
                for function in parsed.namespace.functions
                if function.has_body
            },
        )
        for target, parsed in annotations_by_target.items()
    }
    client_call_templates = client_call_templates_by_target["cuda"]
    server_bindings = {}
    # A handler belongs to the backend whose annotation file declares it.
    for target, path in ANNOTATION_FILES.items():
        backend = target.upper()
        for name, binding in collect_server_bindings(path, backend).items():
            if name in server_bindings and server_bindings[name] != binding:
                raise RuntimeError(f"Conflicting @disabled for {name}")
            server_bindings[name] = binding
    for name in ("lupineDeviceOpen", "lupineDeviceClose", "lupineDeviceIoctl"):
        server_bindings[name] = ServerBinding(name, "CUDA", f"handle_{name}", "defined(__linux__)")
    functions = [
        function
        for function in cuda_ast.namespace.functions
        if function.name.format().startswith("cu")
    ]

    functions_with_annotations: list[
        tuple[Function, Function, list[Operation], FunctionAnnotationMetadata]
    ] = []

    seen = set()

    for function in functions:
        # ensure duplicate functions can't be written
        if function.name.format() in seen:
            continue

        seen.add(function.name.format())

        try:
            annotation = next(
                f for f in annotations.namespace.functions if f.name == function.name
            )
        except StopIteration:
            print(f"Annotation for {function.name} not found")
            continue
        try:
            metadata = parse_annotation(
                annotation.doxygen,
                function.parameters,
                disabled_client_non_nullable=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error parsing annotation for {function.name.format()}"
            ) from e
        attach_client_call_template(function, metadata, client_call_templates)
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
        metadata = parse_annotation(
            annotation.doxygen,
            annotation.parameters,
            disabled_client_non_nullable=True,
        )
        attach_client_call_template(annotation, metadata, client_call_templates)
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
        client_call_templates=client_call_templates_by_target["nvml"],
    )
    hip_functions_with_annotations = collect_backend_functions(
        annotations_by_target["hip"],
        client_call_templates=client_call_templates_by_target["hip"],
    )
    nccl_functions_with_annotations = collect_backend_functions(
        annotations_by_target["nccl"],
        client_call_templates=client_call_templates_by_target["nccl"],
    )
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
        hip_functions_with_annotations,
        nccl_functions_with_annotations,
    )

    with open("gen_nvml_client.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, operations, metadata in nvml_functions_with_annotations:
            if metadata.disabled_client and metadata.disabled_server:
                continue
            write_client_rpc(f, NVML, function, operations, metadata)
            write_client_wrapper(f, NVML, function, operations, metadata)

    with open("gen_nvml_server.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, operations, metadata in nvml_functions_with_annotations:
            if metadata.disabled_server:
                continue
            write_server_handler(f, NVML, function, operations, metadata)

    with open("gen_nvml_server.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, _, metadata in nvml_functions_with_annotations:
            if metadata.disabled_server:
                continue
            f.write(f"int handle_{function.name.format()}(conn_t *conn);\n")

    with open("gen_hip_client.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        hip_client_functions = {}
        for function, _, operations, metadata in hip_functions_with_annotations:
            if metadata.disabled_client and metadata.disabled_server:
                continue
            if not metadata.disabled_client:
                hip_client_functions[function.name.format()] = function
                if unsupported(function, metadata):
                    write_stub(f, HIP, function)
                    continue
            write_client_rpc(f, HIP, function, operations, metadata)
            write_client_wrapper(f, HIP, function, operations, metadata)
        for alias, target in HIP_MANUAL_REMAPPINGS:
            if alias in hip_client_functions:
                continue
            target_function = hip_client_functions.get(target)
            if target_function is None:
                continue
            f.write(f"#ifdef {alias}\n#undef {alias}\n#endif\n")
            f.write(
                'extern "C" {return_type} {alias}({params}) {{\n'.format(
                    return_type=target_function.return_type.format(),
                    alias=alias,
                    params=", ".join(format_function_params(target_function)),
                )
            )
            f.write(
                "  return {target}({args});\n}}\n\n".format(
                    target=target,
                    args=", ".join(format_call_args(target_function)),
                )
            )

    with open("gen_hip_server.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, operations, metadata in hip_functions_with_annotations:
            if metadata.disabled_server or unsupported(function, metadata):
                continue
            write_server_handler(f, HIP, function, operations, metadata)

    with open("gen_hip_server.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function, _, _, metadata in hip_functions_with_annotations:
            if metadata.disabled_server or unsupported(function, metadata):
                continue
            f.write(f"int handle_{function.name.format()}(conn_t *conn);\n")

    for backend, target, functions in ((NCCL, "nccl", nccl_functions_with_annotations),):
        with open(f"gen_{target}_client.inc", "w") as f:
            f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
            for function, _, operations, metadata in functions:
                if metadata.disabled_client and metadata.disabled_server:
                    continue
                if metadata.guard is not None:
                    f.write(f"#if {metadata.guard}\n")
                if unsupported(function, metadata):
                    write_stub(f, backend, function)
                else:
                    write_client_rpc(f, backend, function, operations, metadata)
                    write_client_wrapper(f, backend, function, operations, metadata)
                if metadata.guard is not None:
                    f.write("#endif\n\n")
            write_profiling_aliases(f, backend, functions)

        with open(f"gen_{target}_server.inc", "w") as f:
            f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
            write_scalar_slot(f, functions)
            write_versioned_struct_helper(f, functions)
            for function, _, operations, metadata in functions:
                if metadata.disabled_server or unsupported(function, metadata):
                    continue
                write_server_handler(f, backend, function, operations, metadata)

        with open(f"gen_{target}_server.h", "w") as f:
            f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
            for function, _, _, metadata in functions:
                if metadata.disabled_server or unsupported(function, metadata):
                    continue
                if metadata.guard is not None:
                    f.write(f"#if {metadata.guard}\n")
                f.write(f"int handle_{function.name.format()}(conn_t *conn);\n")
                if metadata.guard is not None:
                    f.write("#endif\n")

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
        and not unsupported(function, metadata)
        and function.name.format() not in server_bindings
    ]
    generated_bindings.extend(
        ServerBinding(name, "NVML", f"handle_{name}")
        for name in NVML_RPC_FUNCTIONS
        if name not in server_bindings
    )
    generated_bindings.extend(
        ServerBinding(function.name.format(), "HIP", f"handle_{function.name.format()}")
        for function, _, _, metadata in hip_functions_with_annotations
        if not metadata.disabled_server
        and not unsupported(function, metadata)
        and function.name.format() not in server_bindings
    )
    generated_bindings.extend(
        ServerBinding(
            function.name.format(),
            "NCCL",
            f"handle_{function.name.format()}",
            metadata.guard,
        )
        for function, _, _, metadata in nccl_functions_with_annotations
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
            "gen_nccl_client.inc",
            "gen_nccl_server.inc",
            "gen_nccl_server.h",
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
        "nccl": [
            "gen_nccl_client.inc",
            "gen_nccl_server.inc",
            "gen_nccl_server.h",
        ],
    }
    forbidden = {
        "cuda": ["nvml", "hip"],
        "nccl": ["nvml", "hip"],
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
        choices=("all", "cuda", "nccl", "nvml", "hip"),
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
