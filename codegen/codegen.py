from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function, Array
from typing import Optional, Union
from dataclasses import dataclass
import hashlib
import os
import glob

# PROTOCOL_HASH_SALT is mixed into LUPINE_PROTOCOL_VERSION. The version is
# otherwise derived from the opcode table and the annotation signatures in
# annotations.h, so API-affecting changes there bump it automatically. Bump
# this salt manually when a wire format changes in a way codegen cannot see
# (manual handlers in manual_server.cpp / client.cpp, the handshake itself,
# or the framing in rpc.cpp).
PROTOCOL_HASH_SALT = "lupine-protocol-1"

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
    ("cuLinkCreate", "cuLinkCreate_v2"),
    ("cuLinkAddData", "cuLinkAddData_v2"),
    ("cuLinkAddFile", "cuLinkAddFile_v2"),
    ("cuMemAlloc", "cuMemAlloc_v2"),
    ("cuMemAllocPitch", "cuMemAllocPitch_v2"),
    ("cuMemFree", "cuMemFree_v2"),
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
    ("cuStreamBeginCapture", "cuStreamBeginCapture_v2"),
    ("cuGraphAddKernelNode", "cuGraphAddKernelNode_v2"),
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
    ("cuLaunchCooperativeKernel_ptsz", "cuLaunchCooperativeKernel"),
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
        if not self.send:
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )

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
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
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
                "        ({size} != 0 && rpc_write(conn, {param_name}, {size}) < 0) ||\n".format(
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
            s = (
                f"    {self.ptr.format()} {self.parameter.name};\n"
                f"    size_t {self.parameter.name}_size;\n"
            )
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
                "        ({size} != 0 && rpc_read(conn, {param_name}, {size}) < 0) ||\n".format(
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
                "        ({size} != 0 && rpc_write(conn, {param_name}, {size}) < 0) ||\n".format(
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
                "        ({size} != 0 && rpc_read(conn, {param_name}, {size}) < 0) ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
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
    disabled: bool = False
    routing_kind: Optional[str] = None
    routing_parameter: Optional[Parameter] = None
    record_owners: list[OwnerAnnotation] = None
    cross_server_copy: Optional[CrossServerCopyAnnotation] = None

    def __post_init__(self):
        if self.record_owners is None:
            self.record_owners = []


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
        if type_name == "CUdeviceptr":
            return "DEVICEPTR", param
    return None, None


# parses a function annotation. if disabled is encountered, returns True for short circuiting.
def parse_annotation(
    annotation: str, params: list[Parameter]
) -> FunctionAnnotationMetadata:
    operations: list[Operation] = []
    metadata = FunctionAnnotationMetadata(operations=operations)

    if not annotation:
        metadata.routing_kind, metadata.routing_parameter = infer_routing_key(params)
        return metadata
    for line in annotation.split("\n"):
        # we treat disabled functions a bit differently.
        # we should record that this function is disabled so that we can create the RPC method but not the actual handler.
        # this is so that we don't break API compatability by commenting/uncommenting functions from our annotations file.
        if "@disabled" in line or "@DISABLED" in line:
            # we can return instantly; the function is disabled and we don't care about operations at this time.
            metadata.disabled = True
            return metadata
        if line.startswith("/**"):
            continue
        if line.startswith("*/"):
            continue
        if line.startswith("*"):
            line = line[2:]
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
                    operations.append(
                        ArrayOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            length=length_param,
                            iter=False
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
        tuple[Function, Function, list[Operation], bool, FunctionAnnotationMetadata]
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
            (function, annotation, metadata.operations, metadata.disabled, metadata)
        )

    protocol_hash = hashlib.sha256()
    protocol_hash.update(PROTOCOL_HASH_SALT.encode())
    for i, (function, annotation, _, _, _) in enumerate(functions_with_annotations):
        protocol_hash.update(
            f"{i}:{function.name.format()}:{annotation.doxygen or ''}\n".encode()
        )
    for i, name in enumerate(NVML_RPC_FUNCTIONS, len(functions_with_annotations)):
        protocol_hash.update(f"{i}:{name}\n".encode())
    for function in NVML_CODEGEN_FUNCTIONS:
        protocol_hash.update(
            "{name}:{params}:{client_body}:{server_body}\n".format(
                **function
            ).encode()
        )
    protocol_version = int(protocol_hash.hexdigest()[:8], 16)
    if protocol_version == 0:
        protocol_version = 1

    with open("gen_api.h", "w") as f:
        f.write(
            "// Generated by codegen.py. Do not edit by hand.\n"
            "\n"
            "// LUPINE_PROTOCOL_VERSION is a hash of the RPC opcode table and the\n"
            "// annotation wire-format signatures, so any API-affecting change to\n"
            "// annotations.h produces a new version. The client sends this value in\n"
            "// an RPC_lupine_handshake request when it connects and the server\n"
            "// replies with its own version; both sides refuse mismatched\n"
            "// connections.\n"
            "#define LUPINE_PROTOCOL_VERSION 0x{version:08x}u\n"
            "\n"
            "// Reserved opcode for the connection handshake. It must never collide\n"
            "// with the generated opcodes below or the manual opcodes (1000001+)\n"
            "// in server.cpp.\n"
            "#define RPC_lupine_handshake 999999\n"
            "\n".format(version=protocol_version)
        )
        for i, (function, _, _, _, _) in enumerate(functions_with_annotations):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=function.name.format(),
                    value=i,
                )
            )
        for i, name in enumerate(NVML_RPC_FUNCTIONS, len(functions_with_annotations)):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=name,
                    value=i,
                )
            )

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
            '#include "rpc.h"\n\n'
            "extern int rpc_size();\n"
            "extern conn_t *rpc_client_get_connection(unsigned int index);\n"
            "extern void rpc_close(conn_t *conn);\n\n"
            "struct lupine_route {\n"
            "    int kind;\n"
            "    conn_t *conn;\n"
            "};\n\n"
            'extern "C" CUresult lupine_cuInit_multi(unsigned int flags);\n'
            'extern "C" CUresult lupine_cuDeviceGetCount_multi(int *count);\n'
            'extern "C" CUresult lupine_cuDeviceGet_multi(CUdevice *device, int ordinal);\n'
            'extern "C" lupine_route lupine_route_for_default();\n'
            'extern "C" lupine_route lupine_route_for_device(CUdevice *device);\n'
            'extern "C" lupine_route lupine_route_for_current_context();\n'
            'extern "C" lupine_route lupine_route_for_context(CUcontext ctx);\n'
            'extern "C" lupine_route lupine_route_for_module(CUmodule module);\n'
            'extern "C" lupine_route lupine_route_for_library(CUlibrary library);\n'
            'extern "C" lupine_route lupine_route_for_function(CUfunction function);\n'
            'extern "C" lupine_route lupine_route_for_stream(CUstream stream);\n'
            'extern "C" lupine_route lupine_route_for_event(CUevent event);\n'
            'extern "C" lupine_route lupine_route_for_deviceptr(CUdeviceptr ptr);\n'
            'extern "C" bool lupine_route_is_local(lupine_route route);\n'
            'extern "C" conn_t *lupine_route_remote_conn(lupine_route route);\n'
            'extern "C" void *lupine_real_cuda_symbol(const char *name);\n'
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
            'extern "C" void lupine_note_deviceptr_owner(CUdeviceptr ptr, conn_t *conn);\n\n'
            'extern "C" void lupine_note_deviceptr_allocation(CUdeviceptr ptr, size_t size, conn_t *conn);\n\n'
            'extern "C" void lupine_note_context_owner_route(CUcontext ctx, lupine_route route);\n'
            'extern "C" void lupine_note_module_owner_route(CUmodule module, lupine_route route);\n'
            'extern "C" void lupine_note_library_owner_route(CUlibrary library, lupine_route route);\n'
            'extern "C" void lupine_note_function_owner_route(CUfunction function, lupine_route route);\n'
            'extern "C" void lupine_note_stream_owner_route(CUstream stream, lupine_route route);\n'
            'extern "C" void lupine_note_event_owner_route(CUevent event, lupine_route route);\n'
            'extern "C" void lupine_note_deviceptr_owner_route(CUdeviceptr ptr, lupine_route route);\n\n'
            'extern "C" void lupine_note_deviceptr_allocation_route(CUdeviceptr ptr, size_t size, lupine_route route);\n\n'
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
            'extern "C" CUresult lupine_cuArrayCreate_v2_safe(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);\n'
            'extern "C" CUresult lupine_cuArray3DCreate_v2_safe(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);\n'
            'extern "C" CUresult lupine_cuLinkCreate_v2_safe(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);\n'
            'extern "C" CUresult lupine_cuLinkAddData_v2_safe(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);\n'
            'extern "C" CUresult lupine_cuLinkAddFile_v2_safe(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues);\n'
            'extern "C" CUresult lupine_cuLinkComplete_safe(CUlinkState state, void **cubinOut, size_t *sizeOut);\n'
            'extern "C" CUresult lupine_cuLinkDestroy_safe(CUlinkState state);\n'
            'extern "C" CUresult lupine_cuLaunchCooperativeKernel_safe(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);\n'
            'extern "C" CUresult lupine_cuGraphExecKernelNodeSetParams_v2_safe(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);\n'
            'extern "C" CUresult lupine_cuMemAllocManaged_safe(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);\n'
            'extern "C" CUresult lupine_cuMemFree_v2_safe(CUdeviceptr dptr);\n'
            'extern "C" CUresult lupine_cuCtxPushCurrent_virtual(CUcontext ctx);\n'
            'extern "C" CUresult lupine_cuCtxPopCurrent_virtual(CUcontext *pctx);\n'
            'extern "C" CUresult lupine_cuCtxSetCurrent_virtual(CUcontext ctx);\n'
            'extern "C" CUresult lupine_cuCtxGetCurrent_virtual(CUcontext *pctx);\n'
            'extern "C" CUresult lupine_cuCtxGetDevice_cached(CUdevice *device);\n'
            'extern "C" void lupine_invalidate_current_context_cache();\n'
            'extern "C" CUresult lupine_cuDevicePrimaryCtxGetState_cached(CUdevice dev, unsigned int *flags, int *active);\n'
            'extern "C" void lupine_note_primary_context_active(CUdevice dev);\n'
            'extern "C" void lupine_note_primary_context_flags(CUdevice dev, unsigned int flags);\n'
            'extern "C" void lupine_invalidate_primary_context_state(CUdevice dev);\n'
            'extern "C" CUresult lupine_cuDeviceGetAttribute_cached(int *pi, CUdevice_attribute attrib, CUdevice dev);\n'
            'extern "C" CUresult lupine_cuKernelGetFunction_cached(CUfunction *pFunc, CUkernel kernel);\n'
            'extern "C" CUresult lupine_cuPointerGetAttributes_safe(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);\n'
            'extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_cached(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);\n'
            'extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cached(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);\n'
            'extern "C" CUresult lupine_cuDeviceCanAccessPeer_multi(int *canAccessPeer, CUdevice dev, CUdevice peerDev);\n'
            'extern "C" CUresult lupine_cuCtxEnablePeerAccess_multi(CUcontext peerContext, unsigned int flags);\n'
            'extern "C" CUresult lupine_cuCtxDisablePeerAccess_multi(CUcontext peerContext);\n\n'
        )
        for function, annotation, operations, disabled, metadata in functions_with_annotations:
            # we don't generate client function definitions for disabled functions; only the RPC definitions.
            if disabled:
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
                "cuMemFree_v2": "lupine_cuMemFree_v2_safe(dptr)",
                "cuMemAllocManaged": "lupine_cuMemAllocManaged_safe(dptr, bytesize, flags)",
                "cuArrayCreate_v2": "lupine_cuArrayCreate_v2_safe(pHandle, pAllocateArray)",
                "cuArray3DCreate_v2": "lupine_cuArray3DCreate_v2_safe(pHandle, pAllocateArray)",
                "cuLinkCreate_v2": "lupine_cuLinkCreate_v2_safe(numOptions, options, optionValues, stateOut)",
                "cuLinkAddData_v2": "lupine_cuLinkAddData_v2_safe(state, type, data, size, name, numOptions, options, optionValues)",
                "cuLinkAddFile_v2": "lupine_cuLinkAddFile_v2_safe(state, type, path, numOptions, options, optionValues)",
                "cuLinkComplete": "lupine_cuLinkComplete_safe(state, cubinOut, sizeOut)",
                "cuLinkDestroy": "lupine_cuLinkDestroy_safe(state)",
                "cuLaunchCooperativeKernel": "lupine_cuLaunchCooperativeKernel_safe(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)",
                "cuGraphExecKernelNodeSetParams_v2": "lupine_cuGraphExecKernelNodeSetParams_v2_safe(hGraphExec, hNode, nodeParams)",
                "cuPointerGetAttributes": "lupine_cuPointerGetAttributes_safe(numAttributes, attributes, data, ptr)",
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
            f.write("    if (lupine_route_is_local(route)) {\n")
            f.write(
                "        using real_fn_t = {return_type} (*)({params});\n".format(
                    return_type=function.return_type.format(),
                    params=", ".join([param.type.format() for param in function.parameters]),
                )
            )
            f.write(
                "        auto real = reinterpret_cast<real_fn_t>(lupine_real_cuda_symbol(\"{name}\"));\n".format(
                    name=function.name.format()
                )
            )
            f.write(
                "        if (real == nullptr) return {error_return};\n".format(
                    error_return=error_const(function.return_type.format())
                )
            )
            f.write(
                "        {return_type} return_value = real({args});\n".format(
                    return_type=function.return_type.format(),
                    args=", ".join(format_call_args(function)),
                )
            )
            write_client_post_call(f, function, metadata)
            f.write("        return return_value;\n")
            f.write("    }\n")
            f.write("    conn_t *conn = lupine_route_remote_conn(route);\n")

            f.write(
                "    {return_type} return_value;\n".format(
                    return_type=function.return_type.format()
                )
            )

            for operation in operations:
                if isinstance(operation, OpaqueTypeOperation):
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
            for function, _, _, disabled, _ in functions_with_annotations
            if not disabled
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
        for function, _, _, disabled, _ in functions_with_annotations:
            if disabled:
                continue

            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.name.format()
                )
            )
        # write manual overrides
        function_names = set(
            f.name.format()
            for f, _, _, disabled, _ in functions_with_annotations
            if not disabled
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
        for function, annotation, operations, disabled, metadata in functions_with_annotations:
            if disabled:
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
                if isinstance(operation, NullTerminatedOperation) or isinstance(
                    operation, ArrayOperation
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

        f.write("static RequestHandler opHandlers[] = {\n")
        for function, _, _, disabled, _ in functions_with_annotations:
            if disabled:
                f.write("    nullptr,\n")
            else:
                f.write("    handle_{name},\n".format(name=function.name.format()))
        for name in NVML_RPC_FUNCTIONS:
            f.write("    handle_{name},\n".format(name=name))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    if (op < 0 || op >= static_cast<int>(sizeof(opHandlers) / sizeof(opHandlers[0])))\n")
        f.write("        return nullptr;\n")
        f.write("    return opHandlers[op];\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
