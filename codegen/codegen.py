from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function, Array
from typing import Optional, Union
from dataclasses import dataclass
import os
import glob
import zlib
from ops import (
    NullableOperation,
    ArrayOperation,
    InOutCountOperation,
    OptionalArrayOperation,
    DeepStructOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    DereferenceOperation,
    Operation,
    OwnerAnnotation,
    CrossServerCopyAnnotation,
    FunctionAnnotationMetadata,
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
    "cuCtxDestroy_v2",
    "cuCtxDetach",
    "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxReset_v2",
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
]

NVML_CODEGEN_FUNCTIONS = []

PRIVATE_RPC_FUNCTIONS = [
    "cuFuncGetParamLayout",
    "cuGetExportTableMetadata",
    "cuGraphAddNode_v2",
    "cuGraphConditionalHandleCreate",
    "cuMemPrefetchAsync",
    "cuPrivateGetModuleNode",
    "cuStreamBeginCaptureToGraph",
    "cuStreamGetCaptureInfo_v3",
]


def rpc_id(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def annotated_rpc_names(annotations: ParsedData) -> list[str]:
    names: set[str] = set()
    for function in annotations.namespace.functions:
        name = function.name.format()
        if len(name) > 2 and name.startswith("cu") and name[2].isupper():
            names.add(name)
    return sorted(names)


def nvml_codegen_function(name, params, client_body, server_body):
    NVML_CODEGEN_FUNCTIONS.append(
        {
            "name": name,
            "params": params,
            "client_body": client_body,
            "server_body": server_body,
        }
    )


def nvml_client_string_no_device(name, value_name):
    return f"""conn_t *_lupine_conn = connection();
nvmlReturn_t _lupine_result = rpc_error();
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &length, sizeof(length)) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    (length != 0 && rpc_read(_lupine_conn, {value_name}, length) < 0) ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
return _lupine_result;"""


def nvml_server_string_no_device(name):
    return f"""unsigned int _lupine_length = 0;
if (rpc_read(conn, &_lupine_length, sizeof(_lupine_length)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

std::vector<char> _lupine_buffer(std::max(1u, _lupine_length), '\\0');
using _lupine_fn_t = nvmlReturn_t (*)(char *, unsigned int);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr ? function_not_found()
                           : _lupine_fn(_lupine_buffer.data(), _lupine_length);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    (_lupine_length != 0 &&
     rpc_write(conn, _lupine_buffer.data(), _lupine_length) < 0) ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_int_out(name, value_name):
    return f"""conn_t *_lupine_conn = connection();
nvmlReturn_t _lupine_result = rpc_error();
int _lupine_value = 0;
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({value_name} != nullptr) {{
  *{value_name} = _lupine_value;
}}
return _lupine_result;"""


def nvml_server_int_out(name):
    return f"""int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

int _lupine_value = 0;
using _lupine_fn_t = nvmlReturn_t (*)(int *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr ? function_not_found() : _lupine_fn(&_lupine_value);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_from_string(name, value_name):
    return f"""conn_t *_lupine_conn = connection();
nvmlReturn_t _lupine_result = rpc_error();
nvmlDevice_t _lupine_device = nullptr;
unsigned int _lupine_length =
    {value_name} == nullptr ? 0 : static_cast<unsigned int>(strlen({value_name}) + 1);
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &_lupine_length, sizeof(_lupine_length)) < 0 ||
    (_lupine_length != 0 &&
     rpc_write(_lupine_conn, {value_name}, _lupine_length) < 0) ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if (device != nullptr) {{
  *device = _lupine_device;
}}
return _lupine_result;"""


def nvml_server_device_from_string(name):
    return f"""unsigned int _lupine_length = 0;
if (rpc_read(conn, &_lupine_length, sizeof(_lupine_length)) < 0) {{
  return -1;
}}
std::vector<char> _lupine_value(std::max(1u, _lupine_length), '\\0');
if (_lupine_length != 0 &&
    rpc_read(conn, _lupine_value.data(), _lupine_length) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

nvmlDevice_t _lupine_device = nullptr;
using _lupine_fn_t = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr
        ? function_not_found()
        : _lupine_fn(_lupine_value.data(), &_lupine_device);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_string(name, value_name):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_write(_lupine_conn, &length, sizeof(length)) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    (length != 0 && rpc_read(_lupine_conn, {value_name}, length) < 0) ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
return _lupine_result;"""


def nvml_server_device_string(name):
    return f"""nvmlDevice_t _lupine_device = nullptr;
unsigned int _lupine_length = 0;
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_read(conn, &_lupine_length, sizeof(_lupine_length)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

std::vector<char> _lupine_buffer(std::max(1u, _lupine_length), '\\0');
using _lupine_fn_t = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr
        ? function_not_found()
        : _lupine_fn(_lupine_device, _lupine_buffer.data(), _lupine_length);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    (_lupine_length != 0 &&
     rpc_write(conn, _lupine_buffer.data(), _lupine_length) < 0) ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_struct(name, out_type, value_name):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
{out_type} _lupine_value = {value_name} == nullptr ? {out_type}{{}} : *{value_name};
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_write(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({value_name} != nullptr) {{
  *{value_name} = _lupine_value;
}}
return _lupine_result;"""


def nvml_server_device_struct(name, out_type):
    return f"""nvmlDevice_t _lupine_device = nullptr;
{out_type} _lupine_value = {{}};
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_read(conn, &_lupine_value, sizeof(_lupine_value)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

using _lupine_fn_t = nvmlReturn_t (*)(nvmlDevice_t, {out_type} *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr ? function_not_found()
                           : _lupine_fn(_lupine_device, &_lupine_value);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_value(name, out_type, value_name):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
{out_type} _lupine_value = {{}};
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({value_name} != nullptr) {{
  *{value_name} = _lupine_value;
}}
return _lupine_result;"""


def nvml_server_device_value(name, out_type):
    return f"""nvmlDevice_t _lupine_device = nullptr;
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

{out_type} _lupine_value = {{}};
using _lupine_fn_t = nvmlReturn_t (*)(nvmlDevice_t, {out_type} *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr ? function_not_found()
                           : _lupine_fn(_lupine_device, &_lupine_value);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_arg_value(name, arg_name, out_type, value_name):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
{out_type} _lupine_value = {{}};
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_write(_lupine_conn, &{arg_name}, sizeof({arg_name})) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({value_name} != nullptr) {{
  *{value_name} = _lupine_value;
}}
return _lupine_result;"""


def nvml_server_device_arg_value(name, arg_type, out_type):
    return f"""nvmlDevice_t _lupine_device = nullptr;
{arg_type} _lupine_arg = {{}};
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_read(conn, &_lupine_arg, sizeof(_lupine_arg)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

{out_type} _lupine_value = {{}};
using _lupine_fn_t = nvmlReturn_t (*)(nvmlDevice_t, {arg_type}, {out_type} *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result = _lupine_fn == nullptr
                                   ? function_not_found()
                                   : _lupine_fn(_lupine_device, _lupine_arg,
                                                &_lupine_value);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_two_args_value(name, first_name, second_name, out_type, value_name):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
{out_type} _lupine_value = {{}};
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_write(_lupine_conn, &{first_name}, sizeof({first_name})) < 0 ||
    rpc_write(_lupine_conn, &{second_name}, sizeof({second_name})) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({value_name} != nullptr) {{
  *{value_name} = _lupine_value;
}}
return _lupine_result;"""


def nvml_server_device_two_args_value(name, first_type, second_type, out_type):
    return f"""nvmlDevice_t _lupine_device = nullptr;
{first_type} _lupine_first = {{}};
{second_type} _lupine_second = {{}};
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_read(conn, &_lupine_first, sizeof(_lupine_first)) < 0 ||
    rpc_read(conn, &_lupine_second, sizeof(_lupine_second)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

{out_type} _lupine_value = {{}};
using _lupine_fn_t =
    nvmlReturn_t (*)(nvmlDevice_t, {first_type}, {second_type}, {out_type} *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr
        ? function_not_found()
        : _lupine_fn(_lupine_device, _lupine_first, _lupine_second,
                     &_lupine_value);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_three_args_value(
    name, first_name, second_name, third_name, out_type, value_name
):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
{out_type} _lupine_value = {{}};
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_write(_lupine_conn, &{first_name}, sizeof({first_name})) < 0 ||
    rpc_write(_lupine_conn, &{second_name}, sizeof({second_name})) < 0 ||
    rpc_write(_lupine_conn, &{third_name}, sizeof({third_name})) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({value_name} != nullptr) {{
  *{value_name} = _lupine_value;
}}
return _lupine_result;"""


def nvml_server_device_three_args_value(
    name, first_type, second_type, third_type, out_type
):
    return f"""nvmlDevice_t _lupine_device = nullptr;
{first_type} _lupine_first = {{}};
{second_type} _lupine_second = {{}};
{third_type} _lupine_third = {{}};
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0 ||
    rpc_read(conn, &_lupine_first, sizeof(_lupine_first)) < 0 ||
    rpc_read(conn, &_lupine_second, sizeof(_lupine_second)) < 0 ||
    rpc_read(conn, &_lupine_third, sizeof(_lupine_third)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

{out_type} _lupine_value = {{}};
using _lupine_fn_t = nvmlReturn_t (*)(nvmlDevice_t, {first_type}, {second_type},
                                      {third_type}, {out_type} *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr
        ? function_not_found()
        : _lupine_fn(_lupine_device, _lupine_first, _lupine_second,
                     _lupine_third, &_lupine_value);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_value, sizeof(_lupine_value)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_client_device_two_values(name, first_name, second_name, first_type, second_type):
    return f"""conn_t *_lupine_conn = connection_for_device(&device);
nvmlReturn_t _lupine_result = rpc_error();
{first_type} _lupine_first = {{}};
{second_type} _lupine_second = {{}};
if (_lupine_conn == nullptr ||
    rpc_write_start_request(_lupine_conn, RPC_{name}) < 0 ||
    rpc_write(_lupine_conn, &device, sizeof(device)) < 0 ||
    rpc_wait_for_response(_lupine_conn) < 0 ||
    rpc_read(_lupine_conn, &_lupine_first, sizeof(_lupine_first)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_second, sizeof(_lupine_second)) < 0 ||
    rpc_read(_lupine_conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_read_end(_lupine_conn) < 0) {{
  return rpc_error();
}}
if ({first_name} != nullptr) {{
  *{first_name} = _lupine_first;
}}
if ({second_name} != nullptr) {{
  *{second_name} = _lupine_second;
}}
return _lupine_result;"""


def nvml_server_device_two_values(name, first_type, second_type):
    return f"""nvmlDevice_t _lupine_device = nullptr;
if (rpc_read(conn, &_lupine_device, sizeof(_lupine_device)) < 0) {{
  return -1;
}}
int _lupine_request_id = rpc_read_end(conn);
if (_lupine_request_id < 0) {{
  return -1;
}}

{first_type} _lupine_first = {{}};
{second_type} _lupine_second = {{}};
using _lupine_fn_t =
    nvmlReturn_t (*)(nvmlDevice_t, {first_type} *, {second_type} *);
_lupine_fn_t _lupine_fn = nvml_symbol<_lupine_fn_t>("{name}");
nvmlReturn_t _lupine_result =
    _lupine_fn == nullptr
        ? function_not_found()
        : _lupine_fn(_lupine_device, &_lupine_first, &_lupine_second);

if (rpc_write_start_response(conn, _lupine_request_id) < 0 ||
    rpc_write(conn, &_lupine_first, sizeof(_lupine_first)) < 0 ||
    rpc_write(conn, &_lupine_second, sizeof(_lupine_second)) < 0 ||
    rpc_write(conn, &_lupine_result, sizeof(_lupine_result)) < 0 ||
    rpc_write_end(conn) < 0) {{
  return -1;
}}
return 0;"""


def nvml_string_no_device(name, value_name):
    nvml_codegen_function(
        name,
        f"char *{value_name}, unsigned int length",
        nvml_client_string_no_device(name, value_name),
        nvml_server_string_no_device(name),
    )


def nvml_int_out(name, value_name):
    nvml_codegen_function(
        name,
        f"int *{value_name}",
        nvml_client_int_out(name, value_name),
        nvml_server_int_out(name),
    )


def nvml_device_from_string(name, value_name):
    nvml_codegen_function(
        name,
        f"const char *{value_name}, nvmlDevice_t *device",
        nvml_client_device_from_string(name, value_name),
        nvml_server_device_from_string(name),
    )


def nvml_device_string(name, value_name):
    nvml_codegen_function(
        name,
        f"nvmlDevice_t device, char *{value_name}, unsigned int length",
        nvml_client_device_string(name, value_name),
        nvml_server_device_string(name),
    )


def nvml_device_struct(name, out_type, value_name):
    nvml_codegen_function(
        name,
        f"nvmlDevice_t device, {out_type} *{value_name}",
        nvml_client_device_struct(name, out_type, value_name),
        nvml_server_device_struct(name, out_type),
    )


def nvml_device_value(name, out_type, value_name):
    nvml_codegen_function(
        name,
        f"nvmlDevice_t device, {out_type} *{value_name}",
        nvml_client_device_value(name, out_type, value_name),
        nvml_server_device_value(name, out_type),
    )


def nvml_device_arg_value(name, arg_type, arg_name, out_type, value_name):
    nvml_codegen_function(
        name,
        f"nvmlDevice_t device, {arg_type} {arg_name}, {out_type} *{value_name}",
        nvml_client_device_arg_value(name, arg_name, out_type, value_name),
        nvml_server_device_arg_value(name, arg_type, out_type),
    )


def nvml_device_two_args_value(
    name, first_type, first_name, second_type, second_name, out_type, value_name
):
    nvml_codegen_function(
        name,
        (
            f"nvmlDevice_t device, {first_type} {first_name}, "
            f"{second_type} {second_name}, {out_type} *{value_name}"
        ),
        nvml_client_device_two_args_value(
            name, first_name, second_name, out_type, value_name
        ),
        nvml_server_device_two_args_value(name, first_type, second_type, out_type),
    )


def nvml_device_three_args_value(
    name,
    first_type,
    first_name,
    second_type,
    second_name,
    third_type,
    third_name,
    out_type,
    value_name,
):
    nvml_codegen_function(
        name,
        (
            f"nvmlDevice_t device, {first_type} {first_name}, "
            f"{second_type} {second_name}, {third_type} {third_name}, "
            f"{out_type} *{value_name}"
        ),
        nvml_client_device_three_args_value(
            name, first_name, second_name, third_name, out_type, value_name
        ),
        nvml_server_device_three_args_value(
            name, first_type, second_type, third_type, out_type
        ),
    )


def nvml_device_two_values(name, first_type, first_name, second_type, second_name):
    nvml_codegen_function(
        name,
        f"nvmlDevice_t device, {first_type} *{first_name}, {second_type} *{second_name}",
        nvml_client_device_two_values(
            name, first_name, second_name, first_type, second_type
        ),
        nvml_server_device_two_values(name, first_type, second_type),
    )


nvml_string_no_device("nvmlSystemGetDriverVersion", "version")
nvml_string_no_device("nvmlSystemGetNVMLVersion", "version")
nvml_int_out("nvmlSystemGetCudaDriverVersion", "cudaDriverVersion")
nvml_int_out("nvmlSystemGetCudaDriverVersion_v2", "cudaDriverVersion")
nvml_device_from_string("nvmlDeviceGetHandleByUUID", "uuid")
nvml_device_from_string("nvmlDeviceGetHandleByPciBusId_v2", "pciBusId")
nvml_device_string("nvmlDeviceGetUUID", "uuid")
nvml_device_string("nvmlDeviceGetVbiosVersion", "version")
nvml_device_string("nvmlDeviceGetSerial", "serial")
nvml_device_string("nvmlDeviceGetBoardPartNumber", "partNumber")
nvml_device_struct("nvmlDeviceGetPciInfo_v3", "nvmlPciInfo_t", "pci")
nvml_device_struct("nvmlDeviceGetMemoryInfo", "nvmlMemory_t", "memory")
nvml_device_struct(
    "nvmlDeviceGetUtilizationRates", "nvmlUtilization_t", "utilization"
)
nvml_device_struct(
    "nvmlDeviceGetTemperatureV", "lupine_nvmlTemperature_t", "temperature"
)
nvml_device_struct("nvmlDeviceGetMemoryInfo_v2", "nvmlMemory_v2_t", "memory")
nvml_device_value("nvmlDeviceGetMinorNumber", "unsigned int", "minorNumber")
nvml_device_value("nvmlDeviceGetPowerUsage", "unsigned int", "power")
nvml_device_value("nvmlDeviceGetPowerManagementLimit", "unsigned int", "limit")
nvml_device_value("nvmlDeviceGetEnforcedPowerLimit", "unsigned int", "limit")
nvml_device_value("nvmlDeviceGetPerformanceState", "nvmlPstates_t", "pState")
nvml_device_value("nvmlDeviceGetComputeMode", "nvmlComputeMode_t", "mode")
nvml_device_value("nvmlDeviceGetPersistenceMode", "nvmlEnableState_t", "mode")
nvml_device_value("nvmlDeviceGetFanSpeed", "unsigned int", "speed")
nvml_device_value("nvmlDeviceGetBrand", "nvmlBrandType_t", "type")
nvml_device_value("nvmlDeviceGetDisplayMode", "nvmlEnableState_t", "display")
nvml_device_value("nvmlDeviceGetDisplayActive", "nvmlEnableState_t", "active")
nvml_device_value("nvmlDeviceGetCurrPcieLinkGeneration", "unsigned int", "value")
nvml_device_value("nvmlDeviceGetCurrPcieLinkWidth", "unsigned int", "value")
nvml_device_value("nvmlDeviceGetMaxPcieLinkGeneration", "unsigned int", "value")
nvml_device_value("nvmlDeviceGetMaxPcieLinkWidth", "unsigned int", "value")
nvml_device_value("nvmlDeviceGetPcieReplayCounter", "unsigned int", "value")
nvml_device_value("nvmlDeviceGetMaxMigDeviceCount", "unsigned int", "count")
nvml_device_value(
    "nvmlDeviceGetVirtualizationMode", "nvmlGpuVirtualizationMode_t", "pVirtualMode"
)
nvml_device_value("nvmlDeviceIsMigDeviceHandle", "unsigned int", "isMigDevice")
nvml_device_arg_value(
    "nvmlDeviceGetTemperature",
    "nvmlTemperatureSensors_t",
    "sensorType",
    "unsigned int",
    "temp",
)
nvml_device_arg_value(
    "nvmlDeviceGetClockInfo", "nvmlClockType_t", "type", "unsigned int", "clock"
)
nvml_device_arg_value(
    "nvmlDeviceGetMaxClockInfo", "nvmlClockType_t", "type", "unsigned int", "clock"
)
nvml_device_arg_value(
    "nvmlDeviceGetPcieThroughput",
    "nvmlPcieUtilCounter_t",
    "counter",
    "unsigned int",
    "value",
)
nvml_device_arg_value(
    "nvmlDeviceGetNvLinkRemoteDeviceType",
    "unsigned int",
    "link",
    "nvmlIntNvLinkDeviceType_t",
    "pNvLinkDeviceType",
)
nvml_device_arg_value(
    "nvmlDeviceGetNvLinkRemotePciInfo_v2",
    "unsigned int",
    "link",
    "nvmlPciInfo_t",
    "pci",
)
nvml_device_two_args_value(
    "nvmlDeviceGetTotalEccErrors",
    "nvmlMemoryErrorType_t",
    "errorType",
    "nvmlEccCounterType_t",
    "counterType",
    "unsigned long long",
    "eccCounts",
)
nvml_device_two_args_value(
    "nvmlDeviceGetDetailedEccErrors",
    "nvmlMemoryErrorType_t",
    "errorType",
    "nvmlEccCounterType_t",
    "counterType",
    "nvmlEccErrorCounts_t",
    "eccCounts",
)
nvml_device_three_args_value(
    "nvmlDeviceGetMemoryErrorCounter",
    "nvmlMemoryErrorType_t",
    "errorType",
    "nvmlEccCounterType_t",
    "counterType",
    "nvmlMemoryLocation_t",
    "locationType",
    "unsigned long long",
    "count",
)
nvml_device_two_values(
    "nvmlDeviceGetEccMode",
    "nvmlEnableState_t",
    "current",
    "nvmlEnableState_t",
    "pending",
)
nvml_device_two_values(
    "nvmlDeviceGetMigMode", "unsigned int", "currentMode", "unsigned int", "pendingMode"
)

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
        if line.startswith("@routingkey"):
            parts = line.split()
            if len(parts) < 2:
                continue
            metadata.routing_kind = parts[1].upper()
            if len(parts) >= 3:
                metadata.routing_parameter = annotation_param(params, parts[2])
            continue
        if line.startswith("@recordowner"):
            parts = line.split()
            if len(parts) < 3:
                continue
            param = annotation_param(params, parts[2])
            metadata.record_owners.append(OwnerAnnotation(parts[1].upper(), param))
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

                # validate that only one of the arguments is present
                if (
                    sum([bool(length_arg), bool(size_arg), null_terminated, nullable])
                    > 1
                ):
                    raise NotImplementedError(
                        "Only one of LENGTH, SIZE, NULL_TERMINATED, or NULLABLE can be specified"
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
                    if "OPTIONAL" in args:
                        # optional out-array sized by an in/out count param (the
                        # cuGraphGetNodes query pattern); linked to its count in
                        # the post-pass below.
                        operations.append(
                            OptionalArrayOperation(
                                parameter=param,
                                ptr=param.type,
                                count=length_param,
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
                        raise NotImplementedError(
                            "NULL_TERMINATED parameters cannot be received; use LENGTH or SIZE for output buffers"
                        )
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
    optional_ops = [op for op in operations if isinstance(op, OptionalArrayOperation)]
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
    if metadata.routing_kind is None:
        metadata.routing_kind, metadata.routing_parameter = infer_routing_key(params)
    return metadata


def client_routing_route_expr(metadata: FunctionAnnotationMetadata) -> str:
    kind = metadata.routing_kind
    param = metadata.routing_parameter
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
        return f"lupine_route_for_function({name})"
    if kind == "STREAM":
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

    if function.name.format() == "cuDevicePrimaryCtxRetain":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_note_primary_context_active(dev);\n")
    if function.name.format() == "cuDevicePrimaryCtxRelease_v2":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_invalidate_primary_context_state(dev);\n")
    if function.name.format() == "cuDevicePrimaryCtxSetFlags_v2":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_note_primary_context_flags(dev, flags);\n")
    if function.name.format() == "cuDevicePrimaryCtxReset_v2":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_invalidate_primary_context_state(dev);\n")
    if function.name.format() == "cuCtxDestroy_v2":
        f.write("    if (return_value == CUDA_SUCCESS) lupine_invalidate_current_context_cache();\n")
    if function.name.format() in KERNEL_PARAM_LAYOUT_INVALIDATORS:
        f.write("    if (return_value == CUDA_SUCCESS) lupine_invalidate_kernel_param_layout_cache();\n")
    if function.name.format() == "cuModuleGetFunction":
        f.write("    if (return_value == CUDA_SUCCESS && hfunc != nullptr) lupine_record_module_function(*hfunc, hmod, name, route);\n")
    if function.name.format() == "cuLibraryGetKernel":
        f.write("    if (return_value == CUDA_SUCCESS && pKernel != nullptr) lupine_record_library_kernel(*pKernel, library, name, route);\n")


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
        functions_with_annotations.append(
            (function, annotation, metadata.operations, metadata)
        )

    annotated_names = annotated_rpc_names(annotations)

    with open("gen_api.h", "w") as f:
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
            write_rpc_define(f"RPC_{name}", name)
        for name in annotated_names:
            write_rpc_define(f"RPC_{name}", name)
        for name in NVML_RPC_FUNCTIONS:
            write_rpc_define(f"RPC_{name}", name)
        f.write("\n")
        for name in PRIVATE_RPC_FUNCTIONS:
            write_rpc_define(f"LUPINE_RPC_{name}", name)

    with open("gen_nvml_client.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function in NVML_CODEGEN_FUNCTIONS:
            f.write(
                'extern "C" nvmlReturn_t {name}({params}) {{\n'.format(
                    name=function["name"],
                    params=function["params"],
                )
            )
            for line in function["client_body"].splitlines():
                if line:
                    f.write(f"  {line}\n")
                else:
                    f.write("\n")
            f.write("}\n\n")

    with open("gen_nvml_server.inc", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function in NVML_CODEGEN_FUNCTIONS:
            f.write(
                "int handle_{name}(conn_t *conn) {{\n".format(
                    name=function["name"],
                )
            )
            for line in function["server_body"].splitlines():
                if line:
                    f.write(f"  {line}\n")
                else:
                    f.write("\n")
            f.write("}\n\n")

    with open("gen_nvml_server.h", "w") as f:
        f.write("// Generated by codegen.py. Do not edit by hand.\n\n")
        for function in NVML_CODEGEN_FUNCTIONS:
            f.write(
                "int handle_{name}(conn_t *conn);\n".format(
                    name=function["name"],
                )
            )

    with open("gen_client.cpp", "w") as f:
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
            '#include "gen_api.h"\n\n'
            '#include "client_routing.h"\n'
            '#include "rpc.h"\n\n'
            "extern int rpc_size();\n"
            "extern conn_t *rpc_client_get_connection(unsigned int index);\n"
            "extern void rpc_close(conn_t *conn);\n"
            'extern "C" void lupine_deep_cache_reset(const void *key);\n'
            'extern "C" void *lupine_deep_cache_add(const void *key, '
            "size_t bytes);\n\n"
            'extern "C" CUresult lupine_cuInit_multi(unsigned int flags);\n'
            'extern "C" CUresult lupine_cuDeviceGetCount_multi(int *count);\n'
            'extern "C" CUresult lupine_cuDeviceGet_multi(CUdevice *device, int ordinal);\n'
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
            'extern "C" void lupine_record_library_kernel(CUkernel kernel, CUlibrary library, const char *name, lupine_route route);\n\n'
            'extern "C" void lupine_record_module_function(CUfunction function, CUmodule module, const char *name, lupine_route route);\n\n'
            'extern "C" void lupine_prepare_host_range_write(void *host, size_t size);\n'
            'extern "C" void lupine_mark_host_range_clean(void *host, size_t size);\n'
            'extern "C" bool lupine_deviceptrs_share_route(CUdeviceptr first, CUdeviceptr second);\n'
            'extern "C" CUresult lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice,\n'
            '                                                   CUdeviceptr srcDevice,\n'
            '                                                   size_t ByteCount,\n'
            '                                                   CUstream hStream,\n'
            '                                                   bool async);\n\n'
            'extern "C" CUresult lupine_cuCtxPushCurrent_virtual(CUcontext ctx);\n'
            'extern "C" CUresult lupine_cuCtxPopCurrent_virtual(CUcontext *pctx);\n'
            'extern "C" CUresult lupine_cuCtxSetCurrent_virtual(CUcontext ctx);\n'
            'extern "C" CUresult lupine_cuCtxGetCurrent_virtual(CUcontext *pctx);\n'
            'extern "C" CUresult lupine_cuCtxGetDevice_cached(CUdevice *device);\n'
            'extern "C" void lupine_invalidate_current_context_cache();\n'
            'extern "C" void lupine_invalidate_kernel_param_layout_cache();\n'
            'extern "C" CUresult lupine_cuDevicePrimaryCtxGetState_cached(CUdevice dev, unsigned int *flags, int *active);\n'
            'extern "C" void lupine_note_primary_context_active(CUdevice dev);\n'
            'extern "C" void lupine_note_primary_context_flags(CUdevice dev, unsigned int flags);\n'
            'extern "C" void lupine_invalidate_primary_context_state(CUdevice dev);\n'
            'extern "C" CUresult lupine_cuDeviceGetAttribute_cached(int *pi, CUdevice_attribute attrib, CUdevice dev);\n'
            'extern "C" CUresult lupine_cuKernelGetFunction_cached(CUfunction *pFunc, CUkernel kernel);\n'
            'extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_cached(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);\n'
            'extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cached(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);\n'
            'extern "C" CUresult lupine_cuDeviceCanAccessPeer_multi(int *canAccessPeer, CUdevice dev, CUdevice peerDev);\n'
            'extern "C" CUresult lupine_cuCtxEnablePeerAccess_multi(CUcontext peerContext, unsigned int flags);\n'
            'extern "C" CUresult lupine_cuCtxDisablePeerAccess_multi(CUcontext peerContext);\n\n'
        )
        for function, annotation, operations, metadata in functions_with_annotations:
            # We don't generate client function definitions for client-disabled
            # functions; their RPC/server definitions may still be generated.
            if metadata.disabled_client:
                continue

            joined_params = ", ".join(format_function_params(function))

            f.write(
                "{return_type} {name}({params})\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=joined_params,
                )
            )
            f.write("{\n")

            if function.name.format() == "cuInit":
                f.write("    return lupine_cuInit_multi(Flags);\n")
                f.write("}\n\n")
                continue
            if function.name.format() == "cuDeviceGet":
                f.write("    return lupine_cuDeviceGet_multi(device, ordinal);\n")
                f.write("}\n\n")
                continue
            if function.name.format() == "cuDeviceGetCount":
                f.write("    return lupine_cuDeviceGetCount_multi(count);\n")
                f.write("}\n\n")
                continue
            direct_wrappers = {
                "cuDeviceGetAttribute": "lupine_cuDeviceGetAttribute_cached(pi, attrib, dev)",
                "cuDevicePrimaryCtxGetState": "lupine_cuDevicePrimaryCtxGetState_cached(dev, flags, active)",
                "cuCtxPushCurrent_v2": "lupine_cuCtxPushCurrent_virtual(ctx)",
                "cuCtxPopCurrent_v2": "lupine_cuCtxPopCurrent_virtual(pctx)",
                "cuCtxSetCurrent": "lupine_cuCtxSetCurrent_virtual(ctx)",
                "cuCtxGetCurrent": "lupine_cuCtxGetCurrent_virtual(pctx)",
                "cuCtxGetDevice": "lupine_cuCtxGetDevice_cached(device)",
                "cuKernelGetFunction": "lupine_cuKernelGetFunction_cached(pFunc, kernel)",
                "cuOccupancyMaxActiveBlocksPerMultiprocessor": "lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_cached(numBlocks, func, blockSize, dynamicSMemSize)",
                "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": "lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cached(numBlocks, func, blockSize, dynamicSMemSize, flags)",
                "cuDeviceCanAccessPeer": "lupine_cuDeviceCanAccessPeer_multi(canAccessPeer, dev, peerDev)",
                "cuCtxEnablePeerAccess": "lupine_cuCtxEnablePeerAccess_multi(peerContext, Flags)",
                "cuCtxDisablePeerAccess": "lupine_cuCtxDisablePeerAccess_multi(peerContext)",
            }
            if function.name.format() in direct_wrappers:
                f.write("    return {call};\n".format(call=direct_wrappers[function.name.format()]))
                f.write("}\n\n")
                continue
            if function.name.format() == "cuModuleGetGlobal_v2":
                f.write("    conn_t *conn = lupine_rpc_conn_for_module(hmod);\n")
                f.write("    CUresult return_value;\n")
                f.write("    size_t remote_bytes = 0;\n")
                f.write("    std::size_t name_len = std::strlen(name) + 1;\n")
                f.write("    if (conn == nullptr ||\n")
                f.write("        rpc_write_start_request(conn, RPC_cuModuleGetGlobal_v2) < 0 ||\n")
                f.write("        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||\n")
                f.write("        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||\n")
                f.write("        rpc_write(conn, name, name_len) < 0 ||\n")
                f.write("        rpc_wait_for_response(conn) < 0 ||\n")
                f.write("        (dptr != nullptr && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||\n")
                f.write("        rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||\n")
                f.write("        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||\n")
                f.write("        rpc_read_end(conn) < 0)\n")
                f.write("        return CUDA_ERROR_DEVICE_UNAVAILABLE;\n")
                f.write("    if (bytes != nullptr) *bytes = remote_bytes;\n")
                f.write("    if (return_value == CUDA_SUCCESS && dptr != nullptr) lupine_note_deviceptr_allocation(*dptr, remote_bytes, conn);\n")
                f.write("    return return_value;\n")
                f.write("}\n\n")
                continue
            if function.name.format() in {"cuLibraryGetGlobal", "cuLibraryGetManaged"}:
                f.write("    lupine_route route = lupine_route_for_library(library);\n")
                f.write("    conn_t *conn = lupine_route_remote_conn(route);\n")
                f.write("    CUresult return_value;\n")
                f.write("    size_t remote_bytes = 0;\n")
                f.write("    std::size_t name_len = std::strlen(name) + 1;\n")
                f.write("    if (conn == nullptr ||\n")
                f.write("        rpc_write_start_request(conn, RPC_{name}) < 0 ||\n".format(name=function.name.format()))
                f.write("        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||\n")
                f.write("        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||\n")
                f.write("        rpc_write(conn, name, name_len) < 0 ||\n")
                f.write("        rpc_wait_for_response(conn) < 0 ||\n")
                f.write("        (dptr != nullptr && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||\n")
                f.write("        rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||\n")
                f.write("        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||\n")
                f.write("        rpc_read_end(conn) < 0)\n")
                f.write("        return CUDA_ERROR_DEVICE_UNAVAILABLE;\n")
                f.write("    if (bytes != nullptr) *bytes = remote_bytes;\n")
                f.write("    if (return_value == CUDA_SUCCESS && dptr != nullptr) lupine_note_deviceptr_allocation(*dptr, remote_bytes, conn);\n")
                f.write("    return return_value;\n")
                f.write("}\n\n")
                continue

            f.write(
                "    lupine_route route = {route_expr};\n".format(
                    route_expr=client_routing_route_expr(metadata)
                )
            )
            if metadata.cross_server_copy is not None:
                copy = metadata.cross_server_copy
                stream_arg = copy.stream.name if copy.stream is not None else "nullptr"
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
            f.write(
                "    using real_fn_t = {return_type} (*)({params});\n".format(
                    return_type=function.return_type.format(),
                    params=", ".join([param.type.format() for param in function.parameters]),
                )
            )
            call_args = ", ".join(format_call_args(function))
            helper_args = f", {call_args}" if call_args else ""
            f.write(
                "    if (lupine_call_local_cuda_if_routed<real_fn_t>(\n"
                "            route, \"{name}\", &return_value{args})) {{\n".format(
                    name=function.name.format(),
                    args=helper_args,
                )
            )
            write_client_post_call(f, function, metadata)
            f.write("        return return_value;\n")
            f.write("    }\n")
            f.write("    conn_t *conn = lupine_route_remote_conn(route);\n")

            for operation in operations:
                if isinstance(operation, OpaqueTypeOperation):
                    f.write(operation.client_declaration())
                if (
                    isinstance(operation, InOutCountOperation)
                    or isinstance(operation, OptionalArrayOperation)
                    or isinstance(operation, DeepStructOperation)
                ):
                    f.write(operation.client_declaration())

            # compute the strlen's for null-terminated operations.
            for operation in operations:
                if isinstance(operation, NullTerminatedOperation):
                    if operation.send:
                        f.write(
                            "    std::size_t {param_name}_len = std::strlen({param_name}) + 1;\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                    else:
                        f.write(
                            "    std::size_t {param_name}_len;\n".format(
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

            if metadata.async_fire_forget:
                # Fire-and-forget: send without waiting for a response.
                f.write(
                    "    if (conn == nullptr ||\n"
                    "        rpc_write_start_request(conn, RPC_{name}) < 0 ||\n".format(
                        name=function.name.format()
                    )
                )
                for operation in operations:
                    operation.client_rpc_write(f)
                f.write("        rpc_write_end(conn) < 0) {\n")
                f.write(
                    "        return {error_return};\n".format(
                        error_return=error_const(function.return_type.format())
                    )
                )
                f.write("    }\n")
                f.write("    return CUDA_SUCCESS;\n")
                f.write("}\n\n")
                continue

            f.write(
                "    if (conn == nullptr ||\n"
                "        rpc_write_start_request(conn, RPC_{name}) < 0 ||\n".format(
                    name=function.name.format()
                )
            )

            for operation in operations:
                operation.client_rpc_write(f)

            f.write("        rpc_wait_for_response(conn) < 0 ||\n")

            for operation in operations:
                if isinstance(operation, ArrayOperation):
                    operation.client_prepare_rpc_read(f)

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

            write_client_post_call(f, function, metadata)
            for operation in operations:
                if isinstance(operation, ArrayOperation):
                    operation.client_post_rpc_read_success(f)

            f.write("    return return_value;\n")
            f.write("}\n\n")

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
            if metadata.disabled_client:
                continue

            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.name.format()
                )
            )
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

    with open("gen_server.cpp", "w") as f:
        f.write(
            "#include <iostream>\n"
            "#include <cuda.h>\n"
            '#include "cuda_compat.h"\n'
            "\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include <vector>\n\n'
            '#include <cstdio>\n\n'
            '#include "gen_server.h"\n\n'
            '#include <cstdio>\n\n'
            '#include "rpc.h"\n\n'
            '#include "nvml_server.h"\n\n'
        )
        for function, annotation, operations, metadata in functions_with_annotations:
            if metadata.disabled_server:
                continue

            # parse the annotation doxygen
            f.write(
                "int handle_{name}(conn_t *conn)\n".format(
                    name=function.name.format(),
                )
            )
            f.write("{\n")

            defers = []

            for operation in operations:
                f.write(operation.server_declaration)

            f.write("    int request_id;\n")

            # we only generate return from non-void types
            if function.return_type.format() != "void":
                f.write(
                    "    {return_type} lupine_intercept_result;\n".format(
                        return_type=function.return_type.format()
                    )
                )
            else:
                f.write("    void* lupine_intercept_result;\n")

            f.write("    if (\n")
            for operation in operations:
                if (
                    isinstance(operation, NullTerminatedOperation)
                    or isinstance(operation, ArrayOperation)
                    or isinstance(operation, OptionalArrayOperation)
                ):
                    if error := operation.server_rpc_read(f, len(defers)):
                        defers.append(error)
                else:
                    operation.server_rpc_read(f)
            f.write("        false)\n")
            f.write("        goto ERROR_{index};\n".format(index=len(defers)))

            f.write("\n")

            f.write("    request_id = rpc_read_end(conn);\n")
            f.write("    if (request_id < 0)\n")
            f.write("        goto ERROR_{index};\n".format(index=len(defers)))

            params: list[str] = []
            # these need to be in function param order, not operation order.
            for param in function.parameters:
                for op in operations:
                    if op.parameter.name == param.name:
                        params.append(op.server_reference)

            if function.return_type.format() != "void":
                f.write(
                    "    lupine_intercept_result = {name}({params});\n\n".format(
                        name=server_call_name(function.name.format()),
                        params=", ".join(params),
                    )
                )
            else:
                f.write(
                    "    {name}({params});\n\n".format(
                        name=server_call_name(function.name.format()),
                        params=", ".join(params),
                    )
                )

            if metadata.async_fire_forget:
                # Fire-and-forget: no response is sent.
                f.write("    (void) lupine_intercept_result;\n")
                f.write("\n")
                f.write("    return 0;\n")
            else:
                f.write("    if (rpc_write_start_response(conn, request_id) < 0 ||\n")

                for operation in operations:
                    operation.server_rpc_write(f)

                f.write(
                    "        rpc_write(conn, &lupine_intercept_result, sizeof({return_type})) < 0 ||\n".format(
                        return_type=function.return_type.format()
                    )
                )
                f.write("        rpc_write_end(conn) < 0)\n")
                f.write("        goto ERROR_{index};\n".format(index=len(defers)))
                f.write("\n")
                f.write("    return 0;\n")

            for i, defer in enumerate(defers):
                f.write("ERROR_{index}:\n".format(index=len(defers) - i))
                f.write("    free((void *) {param_name});\n".format(param_name=defer))
            f.write("ERROR_0:\n")
            f.write("    return -1;\n")
            f.write("}\n\n")

        f.write("static const std::unordered_map<int, RequestHandler> opHandlers = {\n")
        for function, _, _, metadata in functions_with_annotations:
            if metadata.disabled_server:
                continue
            else:
                f.write(
                    "    {{RPC_{name}, handle_{name}}},\n".format(
                        name=function.name.format()
                    )
                )
        for name in NVML_RPC_FUNCTIONS:
            f.write("    {{RPC_{name}, handle_{name}}},\n".format(name=name))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    auto it = opHandlers.find(op);\n")
        f.write("    if (it == opHandlers.end())\n")
        f.write("        return nullptr;\n")
        f.write("    return it->second;\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
