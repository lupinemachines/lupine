#include "cuda_compat.h"
#include <cuda.h>
#include <iostream>

#include <cstring>
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include <io.h>
#define lupine_close_fd _close
#else
#include <unistd.h>
#define lupine_close_fd close
#endif

#include "gen_api.h"

#include <vector>

#include <cstdio>

#include "gen_server.h"

#include <cstdio>

#include "ipc.h"
#include "rpc.h"

#include "nvml_server.h"

int handle_cuInit(conn_t *conn) {
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuInit(Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDriverGetVersion(conn_t *conn) {
  int driverVersion;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDriverGetVersion(&driverVersion);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &driverVersion, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGet(conn_t *conn) {
  CUdevice device;
  int ordinal;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ordinal, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGet(&device, ordinal);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetCount(conn_t *conn) {
  int count;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetCount(&count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetName(conn_t *conn) {
  int len;
  char *name;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &len, sizeof(int)) < 0 || false)
    goto ERROR_0;
  name = (char *)malloc(len * sizeof(char));
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result =
      cuDeviceGetName((len * sizeof(char) == 0 ? nullptr : name), len, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (len * sizeof(char) != 0 &&
       rpc_write(conn, name, len * sizeof(char)) < 0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuDeviceGetUuid_v2(conn_t *conn) {
  CUuuid *uuid;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;
  uuid = (CUuuid *)malloc(16 * sizeof(CUuuid));
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuDeviceGetUuid_v2(uuid, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (16 != 0 && rpc_write(conn, uuid, 16) < 0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)uuid);
ERROR_0:
  return -1;
}

int handle_cuDeviceGetLuid(conn_t *conn) {
  char *luid;
  unsigned int deviceNodeMask;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;
  luid = (char *)malloc(8 * sizeof(char));
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuDeviceGetLuid(luid, &deviceNodeMask, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (8 != 0 && rpc_write(conn, luid, 8) < 0) ||
      rpc_write(conn, &deviceNodeMask, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)luid);
ERROR_0:
  return -1;
}

int handle_cuDeviceTotalMem_v2(conn_t *conn) {
  size_t bytes;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceTotalMem_v2(&bytes, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetTexture1DLinearMaxWidth(conn_t *conn) {
  size_t maxWidthInElements;
  CUarray_format format;
  unsigned numChannels;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &format, sizeof(CUarray_format)) < 0 ||
      rpc_read(conn, &numChannels, sizeof(unsigned)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetTexture1DLinearMaxWidth(
      &maxWidthInElements, format, numChannels, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &maxWidthInElements, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetAttribute(conn_t *conn) {
  int pi;
  CUdevice_attribute attrib;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &attrib, sizeof(CUdevice_attribute)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetAttribute(&pi, attrib, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pi, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceSetMemPool(conn_t *conn) {
  CUdevice dev;
  CUmemoryPool pool;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceSetMemPool(dev, pool);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetMemPool(conn_t *conn) {
  CUmemoryPool pool;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetMemPool(&pool, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetDefaultMemPool(conn_t *conn) {
  CUmemoryPool pool_out;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetDefaultMemPool(&pool_out, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pool_out, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetExecAffinitySupport(conn_t *conn) {
  int pi;
  CUexecAffinityType type;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetExecAffinitySupport(&pi, type, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pi, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFlushGPUDirectRDMAWrites(conn_t *conn) {
  CUflushGPUDirectRDMAWritesTarget target;
  CUflushGPUDirectRDMAWritesScope scope;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0 ||
      rpc_read(conn, &scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFlushGPUDirectRDMAWrites(target, scope);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetProperties(conn_t *conn) {
  CUdevprop prop;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGetProperties(&prop, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &prop, sizeof(CUdevprop)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceComputeCapability(conn_t *conn) {
  int major;
  int minor;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceComputeCapability(&major, &minor, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &major, sizeof(int)) < 0 ||
      rpc_write(conn, &minor, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDevicePrimaryCtxRetain(conn_t *conn) {
  CUcontext pctx;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDevicePrimaryCtxRetain(&pctx, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDevicePrimaryCtxRelease_v2(conn_t *conn) {
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDevicePrimaryCtxRelease_v2(dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDevicePrimaryCtxSetFlags_v2(conn_t *conn) {
  CUdevice dev;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDevicePrimaryCtxSetFlags_v2(dev, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDevicePrimaryCtxGetState(conn_t *conn) {
  CUdevice dev;
  unsigned int flags;
  int active;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDevicePrimaryCtxGetState(dev, &flags, &active);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &active, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDevicePrimaryCtxReset_v2(conn_t *conn) {
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDevicePrimaryCtxReset_v2(dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxDestroy_v2(conn_t *conn) {
  CUcontext ctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxDestroy_v2(ctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxPushCurrent_v2(conn_t *conn) {
  CUcontext ctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxPushCurrent_v2(ctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxPopCurrent_v2(conn_t *conn) {
  CUcontext pctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxPopCurrent_v2(&pctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxSetCurrent(conn_t *conn) {
  CUcontext ctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxSetCurrent(ctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetCurrent(conn_t *conn) {
  CUcontext pctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetCurrent(&pctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetDevice(conn_t *conn) {
  CUdevice device;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetDevice(&device);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetFlags(conn_t *conn) {
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetFlags(&flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetId(conn_t *conn) {
  CUcontext ctx;
  unsigned long long ctxId;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetId(ctx, &ctxId);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ctxId, sizeof(unsigned long long)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxSetLimit(conn_t *conn) {
  CUlimit limit;
  size_t value;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &limit, sizeof(CUlimit)) < 0 ||
      rpc_read(conn, &value, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxSetLimit(limit, value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetLimit(conn_t *conn) {
  size_t pvalue;
  CUlimit limit;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &limit, sizeof(CUlimit)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetLimit(&pvalue, limit);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pvalue, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetCacheConfig(conn_t *conn) {
  CUfunc_cache pconfig;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetCacheConfig(&pconfig);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pconfig, sizeof(CUfunc_cache)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxSetCacheConfig(conn_t *conn) {
  CUfunc_cache config;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxSetCacheConfig(config);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetApiVersion(conn_t *conn) {
  CUcontext ctx;
  unsigned int version;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetApiVersion(ctx, &version);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &version, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetStreamPriorityRange(conn_t *conn) {
  int leastPriority;
  int greatestPriority;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &leastPriority, sizeof(int)) < 0 ||
      rpc_write(conn, &greatestPriority, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxResetPersistingL2Cache(conn_t *conn) {
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxResetPersistingL2Cache();

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetExecAffinity(conn_t *conn) {
  CUexecAffinityParam pExecAffinity;
  CUexecAffinityType type;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &type, sizeof(CUexecAffinityType)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetExecAffinity(&pExecAffinity, type);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pExecAffinity, sizeof(CUexecAffinityParam)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxAttach(conn_t *conn) {
  CUcontext pctx;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxAttach(&pctx, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxDetach(conn_t *conn) {
  CUcontext ctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxDetach(ctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxGetSharedMemConfig(conn_t *conn) {
  CUsharedconfig pConfig;
  int request_id;
  CUresult lupine_intercept_result;
  if (false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxGetSharedMemConfig(&pConfig);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pConfig, sizeof(CUsharedconfig)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxSetSharedMemConfig(conn_t *conn) {
  CUsharedconfig config;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &config, sizeof(CUsharedconfig)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxSetSharedMemConfig(config);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuModuleUnload(conn_t *conn) {
  CUmodule hmod;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuModuleUnload(hmod);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuModuleGetLoadingMode(conn_t *conn) {
  CUmoduleLoadingMode mode;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &mode, sizeof(CUmoduleLoadingMode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuModuleGetLoadingMode(&mode);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &mode, sizeof(CUmoduleLoadingMode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuModuleGetFunction(conn_t *conn) {
  CUfunction hfunc;
  CUmodule hmod;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuModuleGetFunction(&hfunc, hmod, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuModuleGetGlobal_v2(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytes;
  CUmodule hmod;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuModuleGetGlobal_v2(&dptr, &bytes, hmod, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuLinkCreate_v2(conn_t *conn) {
  unsigned int numOptions;
  CUjit_option options;
  void *optionValues;
  CUlinkState stateOut;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &options, sizeof(CUjit_option)) < 0 ||
      rpc_read(conn, &optionValues, sizeof(void *)) < 0 ||
      rpc_read(conn, &stateOut, sizeof(CUlinkState)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuLinkCreate_v2(numOptions, &options, &optionValues, &stateOut);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &options, sizeof(CUjit_option)) < 0 ||
      rpc_write(conn, &optionValues, sizeof(void *)) < 0 ||
      rpc_write(conn, &stateOut, sizeof(CUlinkState)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLinkAddData_v2(conn_t *conn) {
  CUlinkState state;
  CUjitInputType type;
  void *data;
  size_t size;
  const char *name;
  std::size_t name_len;
  unsigned int numOptions;
  CUjit_option *options;
  size_t options_size;
  void **optionValues;
  size_t optionValues_size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0 ||
      rpc_read(conn, &type, sizeof(CUjitInputType)) < 0 ||
      rpc_read(conn, &data, sizeof(void *)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 ||
      rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0 || false)
    goto ERROR_1;
  options_size = numOptions * sizeof(CUjit_option);
  options = (CUjit_option *)malloc(options_size);
  if (options_size != 0 && options == nullptr)
    goto ERROR_1;
  if ((options_size != 0 && rpc_read(conn, options, options_size) < 0) || false)
    goto ERROR_2;
  optionValues_size = numOptions * sizeof(void *);
  optionValues = (void **)malloc(optionValues_size);
  if (optionValues_size != 0 && optionValues == nullptr)
    goto ERROR_2;
  if ((optionValues_size != 0 &&
       rpc_read(conn, optionValues, optionValues_size) < 0) ||
      false)
    goto ERROR_3;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_3;
  lupine_intercept_result = cuLinkAddData_v2(
      state, type, data, size, name, numOptions,
      (numOptions * sizeof(CUjit_option) == 0 ? nullptr : options),
      (numOptions * sizeof(void *) == 0 ? nullptr : optionValues));

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_3;

  return 0;
ERROR_3:
  free((void *)name);
ERROR_2:
  free((void *)options);
ERROR_1:
  free((void *)optionValues);
ERROR_0:
  return -1;
}

int handle_cuLinkAddFile_v2(conn_t *conn) {
  CUlinkState state;
  CUjitInputType type;
  const char *path;
  std::size_t path_len;
  unsigned int numOptions;
  CUjit_option *options;
  size_t options_size;
  void **optionValues;
  size_t optionValues_size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0 ||
      rpc_read(conn, &type, sizeof(CUjitInputType)) < 0 ||
      rpc_read(conn, &path_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  path = (const char *)malloc(path_len);
  if (rpc_read(conn, (void *)path, path_len) < 0 ||
      rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0 || false)
    goto ERROR_1;
  options_size = numOptions * sizeof(CUjit_option);
  options = (CUjit_option *)malloc(options_size);
  if (options_size != 0 && options == nullptr)
    goto ERROR_1;
  if ((options_size != 0 && rpc_read(conn, options, options_size) < 0) || false)
    goto ERROR_2;
  optionValues_size = numOptions * sizeof(void *);
  optionValues = (void **)malloc(optionValues_size);
  if (optionValues_size != 0 && optionValues == nullptr)
    goto ERROR_2;
  if ((optionValues_size != 0 &&
       rpc_read(conn, optionValues, optionValues_size) < 0) ||
      false)
    goto ERROR_3;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_3;
  lupine_intercept_result = cuLinkAddFile_v2(
      state, type, path, numOptions,
      (numOptions * sizeof(CUjit_option) == 0 ? nullptr : options),
      (numOptions * sizeof(void *) == 0 ? nullptr : optionValues));

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_3;

  return 0;
ERROR_3:
  free((void *)path);
ERROR_2:
  free((void *)options);
ERROR_1:
  free((void *)optionValues);
ERROR_0:
  return -1;
}

int handle_cuLinkComplete(conn_t *conn) {
  CUlinkState state;
  void *cubinOut;
  size_t sizeOut;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLinkComplete(state, &cubinOut, &sizeOut);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &cubinOut, sizeof(void *)) < 0 ||
      rpc_write(conn, &sizeOut, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLinkDestroy(conn_t *conn) {
  CUlinkState state;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLinkDestroy(state);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuModuleGetTexRef(conn_t *conn) {
  CUtexref pTexRef;
  CUmodule hmod;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuModuleGetTexRef(&pTexRef, hmod, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuModuleGetSurfRef(conn_t *conn) {
  CUsurfref pSurfRef;
  CUmodule hmod;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuModuleGetSurfRef(&pSurfRef, hmod, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuLibraryLoadFromFile(conn_t *conn) {
  CUlibrary library;
  const char *fileName;
  std::size_t fileName_len;
  unsigned int numJitOptions;
  CUjit_option *jitOptions;
  size_t jitOptions_size;
  void **jitOptionsValues;
  size_t jitOptionsValues_size;
  unsigned int numLibraryOptions;
  CUlibraryOption *libraryOptions;
  size_t libraryOptions_size;
  void **libraryOptionValues;
  size_t libraryOptionValues_size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &fileName_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  fileName = (const char *)malloc(fileName_len);
  if (rpc_read(conn, (void *)fileName, fileName_len) < 0 ||
      rpc_read(conn, &numJitOptions, sizeof(unsigned int)) < 0 || false)
    goto ERROR_1;
  jitOptions_size = numJitOptions * sizeof(CUjit_option);
  jitOptions = (CUjit_option *)malloc(jitOptions_size);
  if (jitOptions_size != 0 && jitOptions == nullptr)
    goto ERROR_1;
  if ((jitOptions_size != 0 &&
       rpc_read(conn, jitOptions, jitOptions_size) < 0) ||
      false)
    goto ERROR_2;
  jitOptionsValues_size = numJitOptions * sizeof(void *);
  jitOptionsValues = (void **)malloc(jitOptionsValues_size);
  if (jitOptionsValues_size != 0 && jitOptionsValues == nullptr)
    goto ERROR_2;
  if ((jitOptionsValues_size != 0 &&
       rpc_read(conn, jitOptionsValues, jitOptionsValues_size) < 0) ||
      rpc_read(conn, &numLibraryOptions, sizeof(unsigned int)) < 0 || false)
    goto ERROR_3;
  libraryOptions_size = numLibraryOptions * sizeof(CUlibraryOption);
  libraryOptions = (CUlibraryOption *)malloc(libraryOptions_size);
  if (libraryOptions_size != 0 && libraryOptions == nullptr)
    goto ERROR_3;
  if ((libraryOptions_size != 0 &&
       rpc_read(conn, libraryOptions, libraryOptions_size) < 0) ||
      false)
    goto ERROR_4;
  libraryOptionValues_size = numLibraryOptions * sizeof(void *);
  libraryOptionValues = (void **)malloc(libraryOptionValues_size);
  if (libraryOptionValues_size != 0 && libraryOptionValues == nullptr)
    goto ERROR_4;
  if ((libraryOptionValues_size != 0 &&
       rpc_read(conn, libraryOptionValues, libraryOptionValues_size) < 0) ||
      false)
    goto ERROR_5;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_5;
  lupine_intercept_result = cuLibraryLoadFromFile(
      &library, fileName,
      (numJitOptions * sizeof(CUjit_option) == 0 ? nullptr : jitOptions),
      (numJitOptions * sizeof(void *) == 0 ? nullptr : jitOptionsValues),
      numJitOptions,
      (numLibraryOptions * sizeof(CUlibraryOption) == 0 ? nullptr
                                                        : libraryOptions),
      (numLibraryOptions * sizeof(void *) == 0 ? nullptr : libraryOptionValues),
      numLibraryOptions);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_5;

  return 0;
ERROR_5:
  free((void *)fileName);
ERROR_4:
  free((void *)jitOptions);
ERROR_3:
  free((void *)jitOptionsValues);
ERROR_2:
  free((void *)libraryOptions);
ERROR_1:
  free((void *)libraryOptionValues);
ERROR_0:
  return -1;
}

int handle_cuLibraryUnload(conn_t *conn) {
  CUlibrary library;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLibraryUnload(library);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLibraryGetKernel(conn_t *conn) {
  CUkernel pKernel;
  CUlibrary library;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuLibraryGetKernel(&pKernel, library, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pKernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuLibraryGetModule(conn_t *conn) {
  CUmodule pMod;
  CUlibrary library;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLibraryGetModule(&pMod, library);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pMod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuKernelGetFunction(conn_t *conn) {
  CUfunction pFunc;
  CUkernel kernel;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuKernelGetFunction(&pFunc, kernel);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pFunc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLibraryGetGlobal(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytes;
  CUlibrary library;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuLibraryGetGlobal(&dptr, &bytes, library, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuLibraryGetManaged(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytes;
  CUlibrary library;
  const char *name;
  std::size_t name_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &name_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  name = (const char *)malloc(name_len);
  if (rpc_read(conn, (void *)name, name_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuLibraryGetManaged(&dptr, &bytes, library, name);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)name);
ERROR_0:
  return -1;
}

int handle_cuLibraryGetUnifiedFunction(conn_t *conn) {
  void *fptr;
  CUlibrary library;
  const char *symbol;
  std::size_t symbol_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &symbol_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  symbol = (const char *)malloc(symbol_len);
  if (rpc_read(conn, (void *)symbol, symbol_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuLibraryGetUnifiedFunction(&fptr, library, symbol);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &fptr, sizeof(void *)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)symbol);
ERROR_0:
  return -1;
}

int handle_cuKernelGetAttribute(conn_t *conn) {
  int pi;
  CUfunction_attribute attrib;
  CUkernel kernel;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pi, sizeof(int)) < 0 ||
      rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_read(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuKernelGetAttribute(&pi, attrib, kernel, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pi, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuKernelSetAttribute(conn_t *conn) {
  CUfunction_attribute attrib;
  int val;
  CUkernel kernel;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_read(conn, &val, sizeof(int)) < 0 ||
      rpc_read(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuKernelSetAttribute(attrib, val, kernel, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuKernelSetCacheConfig(conn_t *conn) {
  CUkernel kernel;
  CUfunc_cache config;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuKernelSetCacheConfig(kernel, config, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemGetInfo_v2(conn_t *conn) {
  size_t free;
  size_t total;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &free, sizeof(size_t)) < 0 ||
      rpc_read(conn, &total, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemGetInfo_v2(&free, &total);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &free, sizeof(size_t)) < 0 ||
      rpc_write(conn, &total, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAlloc_v2(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytesize;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemAlloc_v2(&dptr, bytesize);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAllocPitch_v2(conn_t *conn) {
  CUdeviceptr dptr;
  size_t pPitch;
  size_t WidthInBytes;
  size_t Height;
  unsigned int ElementSizeBytes;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &pPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &WidthInBytes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_read(conn, &ElementSizeBytes, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemAllocPitch_v2(&dptr, &pPitch, WidthInBytes,
                                               Height, ElementSizeBytes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &pPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemFree_v2(conn_t *conn) {
  CUdeviceptr dptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemFree_v2(dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemGetAddressRange_v2(conn_t *conn) {
  CUdeviceptr pbase;
  size_t psize;
  CUdeviceptr dptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pbase, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &psize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemGetAddressRange_v2(&pbase, &psize, dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pbase, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &psize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAllocManaged(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytesize;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemAllocManaged(&dptr, bytesize, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetByPCIBusId(conn_t *conn) {
  CUdevice dev;
  const char *pciBusId;
  std::size_t pciBusId_len;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_read(conn, &pciBusId_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  pciBusId = (const char *)malloc(pciBusId_len);
  if (rpc_read(conn, (void *)pciBusId, pciBusId_len) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuDeviceGetByPCIBusId(&dev, pciBusId);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)pciBusId);
ERROR_0:
  return -1;
}

int handle_cuDeviceGetPCIBusId(conn_t *conn) {
  int len;
  char *pciBusId;
  CUdevice dev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &len, sizeof(int)) < 0 || false)
    goto ERROR_0;
  pciBusId = (char *)malloc(len * sizeof(char));
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuDeviceGetPCIBusId(
      (len * sizeof(char) == 0 ? nullptr : pciBusId), len, dev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (len * sizeof(char) != 0 &&
       rpc_write(conn, pciBusId, len * sizeof(char)) < 0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)pciBusId);
ERROR_0:
  return -1;
}

int handle_cuIpcGetEventHandle(conn_t *conn) {
  CUipcEventHandle pHandle;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pHandle, sizeof(CUipcEventHandle)) < 0 ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuIpcGetEventHandle(&pHandle, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pHandle, sizeof(CUipcEventHandle)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuIpcOpenEventHandle(conn_t *conn) {
  CUevent phEvent;
  CUipcEventHandle handle;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &handle, sizeof(CUipcEventHandle)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuIpcOpenEventHandle(&phEvent, handle);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuIpcGetMemHandle(conn_t *conn) {
  CUipcMemHandle pHandle;
  CUdeviceptr dptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pHandle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuIpcGetMemHandle(&pHandle, dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pHandle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuIpcOpenMemHandle_v2(conn_t *conn) {
  CUdeviceptr pdptr;
  CUipcMemHandle handle;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &handle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuIpcOpenMemHandle_v2(&pdptr, handle, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuIpcCloseMemHandle(conn_t *conn) {
  CUdeviceptr dptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuIpcCloseMemHandle(dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpy(conn_t *conn) {
  CUdeviceptr dst;
  CUdeviceptr src;
  size_t ByteCount;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dst, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &src, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemcpy(dst, src, ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyPeer(conn_t *conn) {
  CUdeviceptr dstDevice;
  CUcontext dstContext;
  CUdeviceptr srcDevice;
  CUcontext srcContext;
  size_t ByteCount;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstContext, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &srcContext, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyHtoD_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t ByteCount;
  void *srcHost;
  size_t srcHost_size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  srcHost_size = ByteCount;
  srcHost = (void *)malloc(srcHost_size);
  if (srcHost_size != 0 && srcHost == nullptr)
    goto ERROR_0;
  if ((srcHost_size != 0 &&
       rpc_read_payload(conn, srcHost, srcHost_size) < 0) ||
      false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuMemcpyHtoD_v2(
      dstDevice, (ByteCount == 0 ? nullptr : srcHost), ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)srcHost);
ERROR_0:
  return -1;
}

int handle_cuMemcpyDtoD_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  CUdeviceptr srcDevice;
  size_t ByteCount;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyDtoA_v2(conn_t *conn) {
  CUarray dstArray;
  size_t dstOffset;
  CUdeviceptr srcDevice;
  size_t ByteCount;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &dstOffset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyAtoD_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  CUarray srcArray;
  size_t srcOffset;
  size_t ByteCount;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &srcArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &srcOffset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyAtoA_v2(conn_t *conn) {
  CUarray dstArray;
  size_t dstOffset;
  CUarray srcArray;
  size_t srcOffset;
  size_t ByteCount;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &dstOffset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &srcArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &srcOffset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyPeerAsync(conn_t *conn) {
  CUdeviceptr dstDevice;
  CUcontext dstContext;
  CUdeviceptr srcDevice;
  CUcontext srcContext;
  size_t ByteCount;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstContext, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &srcContext, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice,
                                              srcContext, ByteCount, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemcpyDtoDAsync_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  CUdeviceptr srcDevice;
  size_t ByteCount;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);

  (void)lupine_intercept_result;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD8_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  unsigned char uc;
  size_t N;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_read(conn, &N, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemsetD8_v2(dstDevice, uc, N);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD16_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  unsigned short us;
  size_t N;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_read(conn, &N, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemsetD16_v2(dstDevice, us, N);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD32_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  unsigned int ui;
  size_t N;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &N, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemsetD32_v2(dstDevice, ui, N);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD2D8_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t dstPitch;
  unsigned char uc;
  size_t Width;
  size_t Height;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_read(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD2D16_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t dstPitch;
  unsigned short us;
  size_t Width;
  size_t Height;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_read(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD2D32_v2(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t dstPitch;
  unsigned int ui;
  size_t Width;
  size_t Height;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD8Async(conn_t *conn) {
  CUdeviceptr dstDevice;
  unsigned char uc;
  size_t N;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_read(conn, &N, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemsetD8Async(dstDevice, uc, N, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD16Async(conn_t *conn) {
  CUdeviceptr dstDevice;
  unsigned short us;
  size_t N;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_read(conn, &N, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemsetD16Async(dstDevice, us, N, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD32Async(conn_t *conn) {
  CUdeviceptr dstDevice;
  unsigned int ui;
  size_t N;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &N, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemsetD32Async(dstDevice, ui, N, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD2D8Async(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t dstPitch;
  unsigned char uc;
  size_t Width;
  size_t Height;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_read(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD2D16Async(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t dstPitch;
  unsigned short us;
  size_t Width;
  size_t Height;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_read(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemsetD2D32Async(conn_t *conn) {
  CUdeviceptr dstDevice;
  size_t dstPitch;
  unsigned int ui;
  size_t Width;
  size_t Height;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_read(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArrayCreate_v2(conn_t *conn) {
  CUarray pHandle;
  CUDA_ARRAY_DESCRIPTOR pAllocateArray;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pHandle, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &pAllocateArray, sizeof(const CUDA_ARRAY_DESCRIPTOR)) <
          0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuArrayCreate_v2(&pHandle, &pAllocateArray);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pHandle, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArrayGetDescriptor_v2(conn_t *conn) {
  CUDA_ARRAY_DESCRIPTOR pArrayDescriptor;
  CUarray hArray;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
      rpc_read(conn, &hArray, sizeof(CUarray)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuArrayGetDescriptor_v2(&pArrayDescriptor, hArray);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArrayGetSparseProperties(conn_t *conn) {
  CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties;
  CUarray array;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_read(conn, &array, sizeof(CUarray)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuArrayGetSparseProperties(&sparseProperties, array);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMipmappedArrayGetSparseProperties(conn_t *conn) {
  CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties;
  CUmipmappedArray mipmap;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMipmappedArrayGetSparseProperties(&sparseProperties, mipmap);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArrayGetMemoryRequirements(conn_t *conn) {
  CUDA_ARRAY_MEMORY_REQUIREMENTS memoryRequirements;
  CUarray array;
  CUdevice device;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &memoryRequirements,
               sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_read(conn, &array, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &device, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuArrayGetMemoryRequirements(&memoryRequirements, array, device);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &memoryRequirements,
                sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMipmappedArrayGetMemoryRequirements(conn_t *conn) {
  CUDA_ARRAY_MEMORY_REQUIREMENTS memoryRequirements;
  CUmipmappedArray mipmap;
  CUdevice device;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &memoryRequirements,
               sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &device, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMipmappedArrayGetMemoryRequirements(
      &memoryRequirements, mipmap, device);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &memoryRequirements,
                sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArrayGetPlane(conn_t *conn) {
  CUarray pPlaneArray;
  CUarray hArray;
  unsigned int planeIdx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pPlaneArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &planeIdx, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuArrayGetPlane(&pPlaneArray, hArray, planeIdx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pPlaneArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArrayDestroy(conn_t *conn) {
  CUarray hArray;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuArrayDestroy(hArray);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArray3DCreate_v2(conn_t *conn) {
  CUarray pHandle;
  CUDA_ARRAY3D_DESCRIPTOR pAllocateArray;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pHandle, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &pAllocateArray, sizeof(const CUDA_ARRAY3D_DESCRIPTOR)) <
          0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuArray3DCreate_v2(&pHandle, &pAllocateArray);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pHandle, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuArray3DGetDescriptor_v2(conn_t *conn) {
  CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor;
  CUarray hArray;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
      rpc_read(conn, &hArray, sizeof(CUarray)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuArray3DGetDescriptor_v2(&pArrayDescriptor, hArray);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMipmappedArrayCreate(conn_t *conn) {
  CUmipmappedArray pHandle;
  CUDA_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc;
  unsigned int numMipmapLevels;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pHandle, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &pMipmappedArrayDesc,
               sizeof(const CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
      rpc_read(conn, &numMipmapLevels, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMipmappedArrayCreate(&pHandle, &pMipmappedArrayDesc, numMipmapLevels);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pHandle, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMipmappedArrayGetLevel(conn_t *conn) {
  CUarray pLevelArray;
  CUmipmappedArray hMipmappedArray;
  unsigned int level;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pLevelArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &level, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMipmappedArrayGetLevel(&pLevelArray, hMipmappedArray, level);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pLevelArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMipmappedArrayDestroy(conn_t *conn) {
  CUmipmappedArray hMipmappedArray;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMipmappedArrayDestroy(hMipmappedArray);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAddressReserve(conn_t *conn) {
  CUdeviceptr ptr;
  size_t size;
  size_t alignment;
  CUdeviceptr addr;
  unsigned long long flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &alignment, sizeof(size_t)) < 0 ||
      rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemAddressReserve(&ptr, size, alignment, addr, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAddressFree(conn_t *conn) {
  CUdeviceptr ptr;
  size_t size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemAddressFree(ptr, size);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemCreate(conn_t *conn) {
  CUmemGenericAllocationHandle handle;
  size_t size;
  CUmemAllocationProp prop;
  unsigned long long flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &prop, sizeof(const CUmemAllocationProp)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemCreate(&handle, size, &prop, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemRelease(conn_t *conn) {
  CUmemGenericAllocationHandle handle;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemRelease(handle);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemMap(conn_t *conn) {
  CUdeviceptr ptr;
  size_t size;
  size_t offset;
  CUmemGenericAllocationHandle handle;
  unsigned long long flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &offset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemMap(ptr, size, offset, handle, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemMapArrayAsync(conn_t *conn) {
  CUarrayMapInfo mapInfoList;
  unsigned int count;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
      rpc_read(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemMapArrayAsync(&mapInfoList, count, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemUnmap(conn_t *conn) {
  CUdeviceptr ptr;
  size_t size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemUnmap(ptr, size);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemSetAccess(conn_t *conn) {
  CUdeviceptr ptr;
  size_t size;
  size_t count;
  CUmemAccessDesc *desc;
  size_t desc_size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &count, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  desc_size = count * sizeof(const CUmemAccessDesc);
  desc = (CUmemAccessDesc *)malloc(desc_size);
  if (desc_size != 0 && desc == nullptr)
    goto ERROR_0;
  if ((desc_size != 0 && rpc_read(conn, desc, desc_size) < 0) || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuMemSetAccess(
      ptr, size, (count * sizeof(const CUmemAccessDesc) == 0 ? nullptr : desc),
      count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)desc);
ERROR_0:
  return -1;
}

int handle_cuMemGetAccess(conn_t *conn) {
  unsigned long long flags;
  CUmemLocation location;
  CUdeviceptr ptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &location, sizeof(const CUmemLocation)) < 0 ||
      rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemGetAccess(&flags, &location, ptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemExportToShareableHandle(conn_t *conn) {
  CUmemGenericAllocationHandle handle;
  CUmemAllocationHandleType handleType;
  unsigned long long flags;
  int request_id;
  int shareable_fd = -1;
  lupine_ipc_token token = {};
  CUresult lupine_intercept_result = CUDA_ERROR_NOT_SUPPORTED;
  if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    lupine_intercept_result =
        cuMemExportToShareableHandle(&shareable_fd, handle, handleType, flags);
    if (lupine_intercept_result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION,
                                        &token, shareable_fd) < 0) {
        lupine_intercept_result = CUDA_ERROR_UNKNOWN;
      }
      lupine_close_fd(shareable_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemImportFromShareableHandle(conn_t *conn) {
  lupine_ipc_token token;
  CUmemAllocationHandleType shHandleType;
  CUmemGenericAllocationHandle handle = 0;
  int request_id;
  CUresult lupine_intercept_result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &shHandleType, sizeof(CUmemAllocationHandleType)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  if (shHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token);
    if (import_fd >= 0) {
      lupine_intercept_result = cuMemImportFromShareableHandle(
          &handle, reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd)),
          shHandleType);
      lupine_close_fd(import_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemGetAllocationGranularity(conn_t *conn) {
  size_t granularity;
  CUmemAllocationProp prop;
  CUmemAllocationGranularity_flags option;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &prop, sizeof(const CUmemAllocationProp)) < 0 ||
      rpc_read(conn, &option, sizeof(CUmemAllocationGranularity_flags)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemGetAllocationGranularity(&granularity, &prop, option);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &granularity, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemGetAllocationPropertiesFromHandle(conn_t *conn) {
  CUmemAllocationProp prop;
  CUmemGenericAllocationHandle handle;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &prop, sizeof(CUmemAllocationProp)) < 0 ||
      rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemGetAllocationPropertiesFromHandle(&prop, handle);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &prop, sizeof(CUmemAllocationProp)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemFreeAsync(conn_t *conn) {
  CUdeviceptr dptr;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemFreeAsync(dptr, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAllocAsync(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytesize;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemAllocAsync(&dptr, bytesize, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolTrimTo(conn_t *conn) {
  CUmemoryPool pool;
  size_t minBytesToKeep;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &minBytesToKeep, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolTrimTo(pool, minBytesToKeep);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolSetAccess(conn_t *conn) {
  CUmemoryPool pool;
  const CUmemAccessDesc *map;
  size_t count;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &map, sizeof(const CUmemAccessDesc *)) < 0 ||
      rpc_read(conn, &count, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolSetAccess(pool, map, count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolGetAccess(conn_t *conn) {
  CUmemAccess_flags flags;
  CUmemoryPool memPool;
  CUmemLocation location;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &flags, sizeof(CUmemAccess_flags)) < 0 ||
      rpc_read(conn, &memPool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &location, sizeof(CUmemLocation)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolGetAccess(&flags, memPool, &location);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(CUmemAccess_flags)) < 0 ||
      rpc_write(conn, &location, sizeof(CUmemLocation)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolCreate(conn_t *conn) {
  CUmemoryPool pool;
  CUmemPoolProps poolProps;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &poolProps, sizeof(CUmemPoolProps)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolCreate(&pool, &poolProps);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolDestroy(conn_t *conn) {
  CUmemoryPool pool;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolDestroy(pool);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemAllocFromPoolAsync(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytesize;
  CUmemoryPool pool;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuMemAllocFromPoolAsync(&dptr, bytesize, pool, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolExportToShareableHandle(conn_t *conn) {
  CUmemoryPool pool;
  CUmemAllocationHandleType handleType;
  unsigned long long flags;
  int request_id;
  int shareable_fd = -1;
  lupine_ipc_token token = {};
  CUresult lupine_intercept_result = CUDA_ERROR_NOT_SUPPORTED;
  if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    lupine_intercept_result = cuMemPoolExportToShareableHandle(
        &shareable_fd, pool, handleType, flags);
    if (lupine_intercept_result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token,
                                        shareable_fd) < 0) {
        lupine_intercept_result = CUDA_ERROR_UNKNOWN;
      }
      lupine_close_fd(shareable_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolImportFromShareableHandle(conn_t *conn) {
  lupine_ipc_token token;
  CUmemAllocationHandleType handleType;
  unsigned long long flags;
  CUmemoryPool pool = nullptr;
  int request_id;
  CUresult lupine_intercept_result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token);
    if (import_fd >= 0) {
      lupine_intercept_result = cuMemPoolImportFromShareableHandle(
          &pool, reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd)),
          handleType, flags);
      lupine_close_fd(import_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolExportPointer(conn_t *conn) {
  CUmemPoolPtrExportData shareData_out;
  CUdeviceptr ptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolExportPointer(&shareData_out, ptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemPoolImportPointer(conn_t *conn) {
  CUdeviceptr ptr_out;
  CUmemoryPool pool;
  CUmemPoolPtrExportData shareData;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &shareData, sizeof(CUmemPoolPtrExportData)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuMemRangeGetAttributes(conn_t *conn) {
  void *data;
  size_t dataSizes;
  CUmem_range_attribute attributes;
  size_t numAttributes;
  CUdeviceptr devPtr;
  size_t count;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &data, sizeof(void *)) < 0 ||
      rpc_read(conn, &dataSizes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &attributes, sizeof(CUmem_range_attribute)) < 0 ||
      rpc_read(conn, &numAttributes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &count, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuMemRangeGetAttributes(
      &data, &dataSizes, &attributes, numAttributes, devPtr, count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &data, sizeof(void *)) < 0 ||
      rpc_write(conn, &dataSizes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &attributes, sizeof(CUmem_range_attribute)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuPointerSetAttribute(conn_t *conn) {
  const void *value;
  CUpointer_attribute attribute;
  CUdeviceptr ptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &value, sizeof(const void *)) < 0 ||
      rpc_read(conn, &attribute, sizeof(CUpointer_attribute)) < 0 ||
      rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuPointerSetAttribute(value, attribute, ptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuPointerGetAttributes(conn_t *conn) {
  unsigned int numAttributes;
  CUpointer_attribute attributes;
  void *data;
  CUdeviceptr ptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &numAttributes, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &attributes, sizeof(CUpointer_attribute)) < 0 ||
      rpc_read(conn, &data, sizeof(void *)) < 0 ||
      rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuPointerGetAttributes(numAttributes, &attributes, &data, ptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &attributes, sizeof(CUpointer_attribute)) < 0 ||
      rpc_write(conn, &data, sizeof(void *)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamCreate(conn_t *conn) {
  CUstream phStream;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamCreate(&phStream, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamCreateWithPriority(conn_t *conn) {
  CUstream phStream;
  unsigned int flags;
  int priority;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &priority, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuStreamCreateWithPriority(&phStream, flags, priority);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamGetPriority(conn_t *conn) {
  CUstream hStream;
  int priority;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &priority, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamGetPriority(hStream, &priority);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &priority, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamGetFlags(conn_t *conn) {
  CUstream hStream;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamGetFlags(hStream, &flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamGetId(conn_t *conn) {
  CUstream hStream;
  unsigned long long streamId;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &streamId, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamGetId(hStream, &streamId);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &streamId, sizeof(unsigned long long)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamGetCtx(conn_t *conn) {
  CUstream hStream;
  CUcontext pctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &pctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamGetCtx(hStream, &pctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamBeginCapture_v2(conn_t *conn) {
  CUstream hStream;
  CUstreamCaptureMode mode;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamBeginCapture_v2(hStream, mode);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuThreadExchangeStreamCaptureMode(conn_t *conn) {
  CUstreamCaptureMode mode;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuThreadExchangeStreamCaptureMode(&mode);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamEndCapture(conn_t *conn) {
  CUstream hStream;
  CUgraph *phGraph_null_check;
  CUgraph phGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &phGraph_null_check, sizeof(CUgraph *)) < 0 ||
      (phGraph_null_check && rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuStreamEndCapture(hStream, phGraph_null_check ? &phGraph : nullptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraph_null_check, sizeof(CUgraph *)) < 0 ||
      (phGraph_null_check && rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamIsCapturing(conn_t *conn) {
  CUstream hStream;
  CUstreamCaptureStatus captureStatus;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamIsCapturing(hStream, &captureStatus);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamAttachMemAsync(conn_t *conn) {
  CUstream hStream;
  CUdeviceptr dptr;
  size_t length;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &length, sizeof(size_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuStreamAttachMemAsync(hStream, dptr, length, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamQuery(conn_t *conn) {
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamQuery(hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamDestroy_v2(conn_t *conn) {
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamDestroy_v2(hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamCopyAttributes(conn_t *conn) {
  CUstream dst;
  CUstream src;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dst, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &src, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamCopyAttributes(dst, src);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamGetAttribute(conn_t *conn) {
  CUstream hStream;
  CUstreamAttrID attr;
  CUstreamAttrValue value_out;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
      rpc_read(conn, &value_out, sizeof(CUstreamAttrValue)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamGetAttribute(hStream, attr, &value_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value_out, sizeof(CUstreamAttrValue)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamSetAttribute(conn_t *conn) {
  CUstream hStream;
  CUstreamAttrID attr;
  CUstreamAttrValue value;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
      rpc_read(conn, &value, sizeof(const CUstreamAttrValue)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamSetAttribute(hStream, attr, &value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuEventCreate(conn_t *conn) {
  CUevent phEvent;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuEventCreate(&phEvent, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuEventRecord(conn_t *conn) {
  CUevent hEvent;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuEventRecord(hEvent, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuEventRecordWithFlags(conn_t *conn) {
  CUevent hEvent;
  CUstream hStream;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuEventRecordWithFlags(hEvent, hStream, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuEventDestroy_v2(conn_t *conn) {
  CUevent hEvent;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuEventDestroy_v2(hEvent);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuEventElapsedTime_v2(conn_t *conn) {
  float pMilliseconds;
  CUevent hStart;
  CUevent hEnd;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pMilliseconds, sizeof(float)) < 0 ||
      rpc_read(conn, &hStart, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &hEnd, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuEventElapsedTime(&pMilliseconds, hStart, hEnd);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pMilliseconds, sizeof(float)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuImportExternalMemory(conn_t *conn) {
  CUexternalMemory extMem_out;
  const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &extMem_out, sizeof(CUexternalMemory)) < 0 ||
      rpc_read(conn, &memHandleDesc,
               sizeof(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuImportExternalMemory(&extMem_out, memHandleDesc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &extMem_out, sizeof(CUexternalMemory)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuExternalMemoryGetMappedBuffer(conn_t *conn) {
  CUdeviceptr devPtr;
  CUexternalMemory extMem;
  const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_read(conn, &bufferDesc,
               sizeof(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuExternalMemoryGetMappedBuffer(&devPtr, extMem, bufferDesc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuExternalMemoryGetMappedMipmappedArray(conn_t *conn) {
  CUmipmappedArray mipmap;
  CUexternalMemory extMem;
  const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_read(conn, &mipmapDesc,
               sizeof(const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, mipmapDesc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDestroyExternalMemory(conn_t *conn) {
  CUexternalMemory extMem;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &extMem, sizeof(CUexternalMemory)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDestroyExternalMemory(extMem);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuImportExternalSemaphore(conn_t *conn) {
  CUexternalSemaphore extSem_out;
  const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
      rpc_read(conn, &semHandleDesc,
               sizeof(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuImportExternalSemaphore(&extSem_out, semHandleDesc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuSignalExternalSemaphoresAsync(conn_t *conn) {
  const CUexternalSemaphore *extSemArray;
  const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray;
  unsigned int numExtSems;
  CUstream stream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &extSemArray, sizeof(const CUexternalSemaphore *)) < 0 ||
      rpc_read(conn, &paramsArray,
               sizeof(const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *)) < 0 ||
      rpc_read(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &stream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuSignalExternalSemaphoresAsync(
      extSemArray, paramsArray, numExtSems, stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuWaitExternalSemaphoresAsync(conn_t *conn) {
  const CUexternalSemaphore *extSemArray;
  const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray;
  unsigned int numExtSems;
  CUstream stream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &extSemArray, sizeof(const CUexternalSemaphore *)) < 0 ||
      rpc_read(conn, &paramsArray,
               sizeof(const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *)) < 0 ||
      rpc_read(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &stream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuWaitExternalSemaphoresAsync(
      extSemArray, paramsArray, numExtSems, stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDestroyExternalSemaphore(conn_t *conn) {
  CUexternalSemaphore extSem;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &extSem, sizeof(CUexternalSemaphore)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDestroyExternalSemaphore(extSem);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamWaitValue32_v2(conn_t *conn) {
  CUstream stream;
  CUdeviceptr addr;
  cuuint32_t value;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &value, sizeof(cuuint32_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamWaitValue32_v2(stream, addr, value, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamWaitValue64_v2(conn_t *conn) {
  CUstream stream;
  CUdeviceptr addr;
  cuuint64_t value;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &value, sizeof(cuuint64_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamWaitValue64_v2(stream, addr, value, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamWriteValue32_v2(conn_t *conn) {
  CUstream stream;
  CUdeviceptr addr;
  cuuint32_t value;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &value, sizeof(cuuint32_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamWriteValue32_v2(stream, addr, value, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamWriteValue64_v2(conn_t *conn) {
  CUstream stream;
  CUdeviceptr addr;
  cuuint64_t value;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &value, sizeof(cuuint64_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuStreamWriteValue64_v2(stream, addr, value, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuStreamBatchMemOp_v2(conn_t *conn) {
  CUstream stream;
  unsigned int count;
  CUstreamBatchMemOpParams paramArray;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuStreamBatchMemOp_v2(stream, count, &paramArray, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncGetAttribute(conn_t *conn) {
  int pi;
  CUfunction_attribute attrib;
  CUfunction hfunc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pi, sizeof(int)) < 0 ||
      rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncGetAttribute(&pi, attrib, hfunc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pi, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncSetAttribute(conn_t *conn) {
  CUfunction hfunc;
  CUfunction_attribute attrib;
  int value;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_read(conn, &value, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncSetAttribute(hfunc, attrib, value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncSetCacheConfig(conn_t *conn) {
  CUfunction hfunc;
  CUfunc_cache config;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncSetCacheConfig(hfunc, config);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncGetModule(conn_t *conn) {
  CUmodule hmod;
  CUfunction hfunc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncGetModule(&hmod, hfunc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLaunchCooperativeKernel(conn_t *conn) {
  CUfunction f;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  void *kernelParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &f, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &kernelParams, sizeof(void *)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLaunchCooperativeKernel(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, &kernelParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &kernelParams, sizeof(void *)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLaunchCooperativeKernelMultiDevice(conn_t *conn) {
  CUDA_LAUNCH_PARAMS launchParamsList;
  unsigned int numDevices;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
      rpc_read(conn, &numDevices, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLaunchCooperativeKernelMultiDevice(
      &launchParamsList, numDevices, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncSetBlockShape(conn_t *conn) {
  CUfunction hfunc;
  int x;
  int y;
  int z;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &x, sizeof(int)) < 0 ||
      rpc_read(conn, &y, sizeof(int)) < 0 ||
      rpc_read(conn, &z, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncSetBlockShape(hfunc, x, y, z);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncSetSharedSize(conn_t *conn) {
  CUfunction hfunc;
  unsigned int bytes;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &bytes, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncSetSharedSize(hfunc, bytes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuParamSetSize(conn_t *conn) {
  CUfunction hfunc;
  unsigned int numbytes;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &numbytes, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuParamSetSize(hfunc, numbytes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuParamSeti(conn_t *conn) {
  CUfunction hfunc;
  int offset;
  unsigned int value;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &offset, sizeof(int)) < 0 ||
      rpc_read(conn, &value, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuParamSeti(hfunc, offset, value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuParamSetf(conn_t *conn) {
  CUfunction hfunc;
  int offset;
  float value;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &offset, sizeof(int)) < 0 ||
      rpc_read(conn, &value, sizeof(float)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuParamSetf(hfunc, offset, value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLaunch(conn_t *conn) {
  CUfunction f;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &f, sizeof(CUfunction)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLaunch(f);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLaunchGrid(conn_t *conn) {
  CUfunction f;
  int grid_width;
  int grid_height;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &f, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &grid_width, sizeof(int)) < 0 ||
      rpc_read(conn, &grid_height, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuLaunchGrid(f, grid_width, grid_height);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuLaunchGridAsync(conn_t *conn) {
  CUfunction f;
  int grid_width;
  int grid_height;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &f, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &grid_width, sizeof(int)) < 0 ||
      rpc_read(conn, &grid_height, sizeof(int)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuLaunchGridAsync(f, grid_width, grid_height, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuParamSetTexRef(conn_t *conn) {
  CUfunction hfunc;
  int texunit;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &texunit, sizeof(int)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuParamSetTexRef(hfunc, texunit, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuFuncSetSharedMemConfig(conn_t *conn) {
  CUfunction hfunc;
  CUsharedconfig config;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &config, sizeof(CUsharedconfig)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuFuncSetSharedMemConfig(hfunc, config);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphCreate(conn_t *conn) {
  CUgraph phGraph;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphCreate(&phGraph, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphMemcpyNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_MEMCPY3D nodeParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphMemcpyNodeGetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphMemcpyNodeSetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_MEMCPY3D nodeParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(const CUDA_MEMCPY3D)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphMemcpyNodeSetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphMemsetNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_MEMSET_NODE_PARAMS nodeParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphMemsetNodeGetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphMemsetNodeSetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_MEMSET_NODE_PARAMS nodeParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphMemsetNodeSetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddChildGraphNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUgraph childGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &childGraph, sizeof(CUgraph)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddChildGraphNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, childGraph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphChildGraphNodeGetGraph(conn_t *conn) {
  CUgraphNode hNode;
  CUgraph phGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphChildGraphNodeGetGraph(hNode, &phGraph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddEmptyNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddEmptyNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphAddEventRecordNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddEventRecordNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphEventRecordNodeGetEvent(conn_t *conn) {
  CUgraphNode hNode;
  CUevent event_out;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &event_out, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphEventRecordNodeGetEvent(hNode, &event_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &event_out, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphEventRecordNodeSetEvent(conn_t *conn) {
  CUgraphNode hNode;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphEventRecordNodeSetEvent(hNode, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddEventWaitNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddEventWaitNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphEventWaitNodeGetEvent(conn_t *conn) {
  CUgraphNode hNode;
  CUevent event_out;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &event_out, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphEventWaitNodeGetEvent(hNode, &event_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &event_out, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphEventWaitNodeSetEvent(conn_t *conn) {
  CUgraphNode hNode;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphEventWaitNodeSetEvent(hNode, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddExternalSemaphoresSignalNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_extSemArray_buf;
  std::vector<unsigned char> nodeParams_paramsArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_extSemArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.extSemArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_extSemArray_buf.data(),
                nodeParams_extSemArray_buf.size()) < 0) ||
      ((nodeParams.extSemArray = (decltype(nodeParams.extSemArray))
                                     nodeParams_extSemArray_buf.data()),
       false) ||
      ((nodeParams_paramsArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.paramsArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_paramsArray_buf.data(),
                nodeParams_paramsArray_buf.size()) < 0) ||
      ((nodeParams.paramsArray = (decltype(nodeParams.paramsArray))
                                     nodeParams_paramsArray_buf.data()),
       false) ||
      false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddExternalSemaphoresSignalNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphExternalSemaphoresSignalNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS params_out = {};
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExternalSemaphoresSignalNodeGetParams(hNode, &params_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &params_out, sizeof(params_out)) < 0 ||
      (params_out.numExtSems != 0 &&
       rpc_write(conn, params_out.extSemArray,
                 params_out.numExtSems * sizeof(*params_out.extSemArray)) <
           0) ||
      (params_out.numExtSems != 0 &&
       rpc_write(conn, params_out.paramsArray,
                 params_out.numExtSems * sizeof(*params_out.paramsArray)) <
           0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExternalSemaphoresSignalNodeSetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_extSemArray_buf;
  std::vector<unsigned char> nodeParams_paramsArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_extSemArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.extSemArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_extSemArray_buf.data(),
                nodeParams_extSemArray_buf.size()) < 0) ||
      ((nodeParams.extSemArray = (decltype(nodeParams.extSemArray))
                                     nodeParams_extSemArray_buf.data()),
       false) ||
      ((nodeParams_paramsArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.paramsArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_paramsArray_buf.data(),
                nodeParams_paramsArray_buf.size()) < 0) ||
      ((nodeParams.paramsArray = (decltype(nodeParams.paramsArray))
                                     nodeParams_paramsArray_buf.data()),
       false) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExternalSemaphoresSignalNodeSetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddExternalSemaphoresWaitNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_extSemArray_buf;
  std::vector<unsigned char> nodeParams_paramsArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_extSemArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.extSemArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_extSemArray_buf.data(),
                nodeParams_extSemArray_buf.size()) < 0) ||
      ((nodeParams.extSemArray = (decltype(nodeParams.extSemArray))
                                     nodeParams_extSemArray_buf.data()),
       false) ||
      ((nodeParams_paramsArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.paramsArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_paramsArray_buf.data(),
                nodeParams_paramsArray_buf.size()) < 0) ||
      ((nodeParams.paramsArray = (decltype(nodeParams.paramsArray))
                                     nodeParams_paramsArray_buf.data()),
       false) ||
      false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddExternalSemaphoresWaitNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphExternalSemaphoresWaitNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS params_out = {};
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExternalSemaphoresWaitNodeGetParams(hNode, &params_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &params_out, sizeof(params_out)) < 0 ||
      (params_out.numExtSems != 0 &&
       rpc_write(conn, params_out.extSemArray,
                 params_out.numExtSems * sizeof(*params_out.extSemArray)) <
           0) ||
      (params_out.numExtSems != 0 &&
       rpc_write(conn, params_out.paramsArray,
                 params_out.numExtSems * sizeof(*params_out.paramsArray)) <
           0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExternalSemaphoresWaitNodeSetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_extSemArray_buf;
  std::vector<unsigned char> nodeParams_paramsArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_extSemArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.extSemArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_extSemArray_buf.data(),
                nodeParams_extSemArray_buf.size()) < 0) ||
      ((nodeParams.extSemArray = (decltype(nodeParams.extSemArray))
                                     nodeParams_extSemArray_buf.data()),
       false) ||
      ((nodeParams_paramsArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.paramsArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_paramsArray_buf.data(),
                nodeParams_paramsArray_buf.size()) < 0) ||
      ((nodeParams.paramsArray = (decltype(nodeParams.paramsArray))
                                     nodeParams_paramsArray_buf.data()),
       false) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExternalSemaphoresWaitNodeSetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddBatchMemOpNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUDA_BATCH_MEM_OP_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_paramArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_paramArray_buf.resize(nodeParams.count *
                                         sizeof(*nodeParams.paramArray)),
        false)) ||
      (nodeParams.count != 0 &&
       rpc_read(conn, nodeParams_paramArray_buf.data(),
                nodeParams_paramArray_buf.size()) < 0) ||
      ((nodeParams.paramArray =
            (decltype(nodeParams.paramArray))nodeParams_paramArray_buf.data()),
       false) ||
      false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddBatchMemOpNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphBatchMemOpNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_BATCH_MEM_OP_NODE_PARAMS nodeParams_out = {};
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphBatchMemOpNodeGetParams(hNode, &nodeParams_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &nodeParams_out, sizeof(nodeParams_out)) < 0 ||
      (nodeParams_out.count != 0 &&
       rpc_write(conn, nodeParams_out.paramArray,
                 nodeParams_out.count * sizeof(*nodeParams_out.paramArray)) <
           0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphBatchMemOpNodeSetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_BATCH_MEM_OP_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_paramArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_paramArray_buf.resize(nodeParams.count *
                                         sizeof(*nodeParams.paramArray)),
        false)) ||
      (nodeParams.count != 0 &&
       rpc_read(conn, nodeParams_paramArray_buf.data(),
                nodeParams_paramArray_buf.size()) < 0) ||
      ((nodeParams.paramArray =
            (decltype(nodeParams.paramArray))nodeParams_paramArray_buf.data()),
       false) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphBatchMemOpNodeSetParams(hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecBatchMemOpNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUDA_BATCH_MEM_OP_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_paramArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_paramArray_buf.resize(nodeParams.count *
                                         sizeof(*nodeParams.paramArray)),
        false)) ||
      (nodeParams.count != 0 &&
       rpc_read(conn, nodeParams_paramArray_buf.data(),
                nodeParams_paramArray_buf.size()) < 0) ||
      ((nodeParams.paramArray =
            (decltype(nodeParams.paramArray))nodeParams_paramArray_buf.data()),
       false) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddMemAllocNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUDA_MEM_ALLOC_NODE_PARAMS nodeParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddMemAllocNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphMemAllocNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUDA_MEM_ALLOC_NODE_PARAMS params_out;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphMemAllocNodeGetParams(hNode, &params_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphAddMemFreeNode(conn_t *conn) {
  CUgraphNode phGraphNode;
  CUgraph hGraph;
  size_t numDependencies;
  CUgraphNode *dependencies;
  size_t dependencies_size;
  CUdeviceptr dptr;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 || false)
    goto ERROR_0;
  dependencies_size = numDependencies * sizeof(const CUgraphNode);
  dependencies = (CUgraphNode *)malloc(dependencies_size);
  if (dependencies_size != 0 && dependencies == nullptr)
    goto ERROR_0;
  if ((dependencies_size != 0 &&
       rpc_read(conn, dependencies, dependencies_size) < 0) ||
      rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphAddMemFreeNode(
      &phGraphNode, hGraph,
      (numDependencies * sizeof(const CUgraphNode) == 0 ? nullptr
                                                        : dependencies),
      numDependencies, dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)dependencies);
ERROR_0:
  return -1;
}

int handle_cuGraphMemFreeNodeGetParams(conn_t *conn) {
  CUgraphNode hNode;
  CUdeviceptr dptr_out;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &dptr_out, sizeof(CUdeviceptr)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGraphMemTrim(conn_t *conn) {
  CUdevice device;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &device, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceGraphMemTrim(device);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphClone(conn_t *conn) {
  CUgraph phGraphClone;
  CUgraph originalGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphClone, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &originalGraph, sizeof(CUgraph)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphClone(&phGraphClone, originalGraph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphClone, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphNodeFindInClone(conn_t *conn) {
  CUgraphNode phNode;
  CUgraphNode hOriginalNode;
  CUgraph hClonedGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hOriginalNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &hClonedGraph, sizeof(CUgraph)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphNodeFindInClone(&phNode, hOriginalNode, hClonedGraph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphNodeGetType(conn_t *conn) {
  CUgraphNode hNode;
  CUgraphNodeType type;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &type, sizeof(CUgraphNodeType)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphNodeGetType(hNode, &type);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &type, sizeof(CUgraphNodeType)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphGetNodes(conn_t *conn) {
  CUgraph hGraph;
  size_t numNodes = 0;
  size_t numNodes_requested = 0;
  CUgraphNode *nodes = nullptr;
  uint8_t nodes_present = 0;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numNodes_requested, sizeof(size_t)) < 0 ||
      ((numNodes = numNodes_requested), false) ||
      rpc_read(conn, &nodes_present, sizeof(uint8_t)) < 0 || false)
    goto ERROR_0;
  if (nodes_present && numNodes_requested != 0) {
    nodes = (CUgraphNode *)malloc(numNodes_requested * sizeof(CUgraphNode));
    if (nodes == nullptr)
      goto ERROR_0;
  }
  if (false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphGetNodes(hGraph, nodes, &numNodes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &numNodes, sizeof(size_t)) < 0 ||
      (nodes_present && numNodes != 0 &&
       rpc_write(conn, nodes, numNodes * sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)nodes);
ERROR_0:
  return -1;
}

int handle_cuGraphGetRootNodes(conn_t *conn) {
  CUgraph hGraph;
  size_t numRootNodes = 0;
  size_t numRootNodes_requested = 0;
  CUgraphNode *rootNodes = nullptr;
  uint8_t rootNodes_present = 0;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &numRootNodes_requested, sizeof(size_t)) < 0 ||
      ((numRootNodes = numRootNodes_requested), false) ||
      rpc_read(conn, &rootNodes_present, sizeof(uint8_t)) < 0 || false)
    goto ERROR_0;
  if (rootNodes_present && numRootNodes_requested != 0) {
    rootNodes =
        (CUgraphNode *)malloc(numRootNodes_requested * sizeof(CUgraphNode));
    if (rootNodes == nullptr)
      goto ERROR_0;
  }
  if (false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result =
      cuGraphGetRootNodes(hGraph, rootNodes, &numRootNodes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &numRootNodes, sizeof(size_t)) < 0 ||
      (rootNodes_present && numRootNodes != 0 &&
       rpc_write(conn, rootNodes, numRootNodes * sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)rootNodes);
ERROR_0:
  return -1;
}

int handle_cuGraphDestroyNode(conn_t *conn) {
  CUgraphNode hNode;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphDestroyNode(hNode);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphInstantiateWithFlags(conn_t *conn) {
  CUgraphExec phGraphExec;
  CUgraph hGraph;
  unsigned long long flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned long long)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphInstantiateWithFlags(&phGraphExec, hGraph, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphInstantiateWithParams(conn_t *conn) {
  CUgraphExec phGraphExec;
  CUgraph hGraph;
  CUDA_GRAPH_INSTANTIATE_PARAMS instantiateParams;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &instantiateParams,
               sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphInstantiateWithParams(&phGraphExec, hGraph, &instantiateParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &instantiateParams,
                sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecGetFlags(conn_t *conn) {
  CUgraphExec hGraphExec;
  cuuint64_t flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &flags, sizeof(cuuint64_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphExecGetFlags(hGraphExec, &flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(cuuint64_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecMemcpyNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUDA_MEMCPY3D copyParams;
  CUcontext ctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &copyParams, sizeof(const CUDA_MEMCPY3D)) < 0 ||
      rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, &copyParams, ctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecMemsetNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUDA_MEMSET_NODE_PARAMS memsetParams;
  CUcontext ctx;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) <
          0 ||
      rpc_read(conn, &ctx, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, &memsetParams, ctx);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecChildGraphNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUgraph childGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &childGraph, sizeof(CUgraph)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecEventRecordNodeSetEvent(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecEventWaitNodeSetEvent(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUevent event;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &event, sizeof(CUevent)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecExternalSemaphoresSignalNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_extSemArray_buf;
  std::vector<unsigned char> nodeParams_paramsArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_extSemArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.extSemArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_extSemArray_buf.data(),
                nodeParams_extSemArray_buf.size()) < 0) ||
      ((nodeParams.extSemArray = (decltype(nodeParams.extSemArray))
                                     nodeParams_extSemArray_buf.data()),
       false) ||
      ((nodeParams_paramsArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.paramsArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_paramsArray_buf.data(),
                nodeParams_paramsArray_buf.size()) < 0) ||
      ((nodeParams.paramsArray = (decltype(nodeParams.paramsArray))
                                     nodeParams_paramsArray_buf.data()),
       false) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphExecExternalSemaphoresSignalNodeSetParams(
      hGraphExec, hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecExternalSemaphoresWaitNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS nodeParams = {};
  std::vector<unsigned char> nodeParams_extSemArray_buf;
  std::vector<unsigned char> nodeParams_paramsArray_buf;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      ((nodeParams_extSemArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.extSemArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_extSemArray_buf.data(),
                nodeParams_extSemArray_buf.size()) < 0) ||
      ((nodeParams.extSemArray = (decltype(nodeParams.extSemArray))
                                     nodeParams_extSemArray_buf.data()),
       false) ||
      ((nodeParams_paramsArray_buf.resize(nodeParams.numExtSems *
                                          sizeof(*nodeParams.paramsArray)),
        false)) ||
      (nodeParams.numExtSems != 0 &&
       rpc_read(conn, nodeParams_paramsArray_buf.data(),
                nodeParams_paramsArray_buf.size()) < 0) ||
      ((nodeParams.paramsArray = (decltype(nodeParams.paramsArray))
                                     nodeParams_paramsArray_buf.data()),
       false) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphExecExternalSemaphoresWaitNodeSetParams(
      hGraphExec, hNode, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphNodeSetEnabled(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  unsigned int isEnabled;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphNodeGetEnabled(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraphNode hNode;
  unsigned int isEnabled;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphNodeGetEnabled(hGraphExec, hNode, &isEnabled);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphUpload(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphUpload(hGraphExec, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphLaunch(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphLaunch(hGraphExec, hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecDestroy(conn_t *conn) {
  CUgraphExec hGraphExec;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphExecDestroy(hGraphExec);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphDestroy(conn_t *conn) {
  CUgraph hGraph;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphDestroy(hGraph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphExecUpdate_v2(conn_t *conn) {
  CUgraphExec hGraphExec;
  CUgraph hGraph;
  CUgraphExecUpdateResultInfo resultInfo;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphExecUpdate_v2(hGraphExec, hGraph, &resultInfo);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphKernelNodeCopyAttributes(conn_t *conn) {
  CUgraphNode dst;
  CUgraphNode src;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dst, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &src, sizeof(CUgraphNode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphKernelNodeCopyAttributes(dst, src);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphKernelNodeGetAttribute(conn_t *conn) {
  CUgraphNode hNode;
  CUkernelNodeAttrID attr;
  CUkernelNodeAttrValue value_out;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
      rpc_read(conn, &value_out, sizeof(CUkernelNodeAttrValue)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphKernelNodeGetAttribute(hNode, attr, &value_out);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphKernelNodeSetAttribute(conn_t *conn) {
  CUgraphNode hNode;
  CUkernelNodeAttrID attr;
  const CUkernelNodeAttrValue *value;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
      rpc_read(conn, &value, sizeof(const CUkernelNodeAttrValue *)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphKernelNodeSetAttribute(hNode, attr, value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphDebugDotPrint(conn_t *conn) {
  CUgraph hGraph;
  const char *path;
  std::size_t path_len;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &path_len, sizeof(std::size_t)) < 0)
    goto ERROR_0;
  path = (const char *)malloc(path_len);
  if (rpc_read(conn, (void *)path, path_len) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphDebugDotPrint(hGraph, path, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)path);
ERROR_0:
  return -1;
}

int handle_cuUserObjectRetain(conn_t *conn) {
  CUuserObject object;
  unsigned int count;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_read(conn, &count, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuUserObjectRetain(object, count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuUserObjectRelease(conn_t *conn) {
  CUuserObject object;
  unsigned int count;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_read(conn, &count, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuUserObjectRelease(object, count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphRetainUserObject(conn_t *conn) {
  CUgraph graph;
  CUuserObject object;
  unsigned int count;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &graph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_read(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphRetainUserObject(graph, object, count, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphReleaseUserObject(conn_t *conn) {
  CUgraph graph;
  CUuserObject object;
  unsigned int count;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &graph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_read(conn, &count, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphReleaseUserObject(graph, object, count);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(conn_t *conn) {
  int numBlocks;
  CUfunction func;
  int blockSize;
  size_t dynamicSMemSize;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_read(conn, &func, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &blockSize, sizeof(int)) < 0 ||
      rpc_read(conn, &dynamicSMemSize, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, func, blockSize, dynamicSMemSize);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(conn_t *conn) {
  int numBlocks;
  CUfunction func;
  int blockSize;
  size_t dynamicSMemSize;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_read(conn, &func, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &blockSize, sizeof(int)) < 0 ||
      rpc_read(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          &numBlocks, func, blockSize, dynamicSMemSize, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuOccupancyAvailableDynamicSMemPerBlock(conn_t *conn) {
  size_t dynamicSmemSize;
  CUfunction func;
  int numBlocks;
  int blockSize;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &dynamicSmemSize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &func, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_read(conn, &blockSize, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuOccupancyAvailableDynamicSMemPerBlock(
      &dynamicSmemSize, func, numBlocks, blockSize);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dynamicSmemSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuOccupancyMaxPotentialClusterSize(conn_t *conn) {
  int clusterSize;
  CUfunction func;
  const CUlaunchConfig *config;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &clusterSize, sizeof(int)) < 0 ||
      rpc_read(conn, &func, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &config, sizeof(const CUlaunchConfig *)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuOccupancyMaxPotentialClusterSize(&clusterSize, func, config);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &clusterSize, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuOccupancyMaxActiveClusters(conn_t *conn) {
  int numClusters;
  CUfunction func;
  const CUlaunchConfig *config;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &numClusters, sizeof(int)) < 0 ||
      rpc_read(conn, &func, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &config, sizeof(const CUlaunchConfig *)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuOccupancyMaxActiveClusters(&numClusters, func, config);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &numClusters, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetArray(conn_t *conn) {
  CUtexref hTexRef;
  CUarray hArray;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetArray(hTexRef, hArray, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetMipmappedArray(conn_t *conn) {
  CUtexref hTexRef;
  CUmipmappedArray hMipmappedArray;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetAddress_v2(conn_t *conn) {
  size_t ByteOffset;
  CUtexref hTexRef;
  CUdeviceptr dptr;
  size_t bytes;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &ByteOffset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &bytes, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuTexRefSetAddress_v2(&ByteOffset, hTexRef, dptr, bytes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ByteOffset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetAddress2D_v3(conn_t *conn) {
  CUtexref hTexRef;
  CUDA_ARRAY_DESCRIPTOR desc;
  CUdeviceptr dptr;
  size_t Pitch;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &desc, sizeof(const CUDA_ARRAY_DESCRIPTOR)) < 0 ||
      rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &Pitch, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuTexRefSetAddress2D_v3(hTexRef, &desc, dptr, Pitch);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetFormat(conn_t *conn) {
  CUtexref hTexRef;
  CUarray_format fmt;
  int NumPackedComponents;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &fmt, sizeof(CUarray_format)) < 0 ||
      rpc_read(conn, &NumPackedComponents, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetAddressMode(conn_t *conn) {
  CUtexref hTexRef;
  int dim;
  CUaddress_mode am;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &dim, sizeof(int)) < 0 ||
      rpc_read(conn, &am, sizeof(CUaddress_mode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetAddressMode(hTexRef, dim, am);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetFilterMode(conn_t *conn) {
  CUtexref hTexRef;
  CUfilter_mode fm;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &fm, sizeof(CUfilter_mode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetFilterMode(hTexRef, fm);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetMipmapFilterMode(conn_t *conn) {
  CUtexref hTexRef;
  CUfilter_mode fm;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &fm, sizeof(CUfilter_mode)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetMipmapFilterMode(hTexRef, fm);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetMipmapLevelBias(conn_t *conn) {
  CUtexref hTexRef;
  float bias;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &bias, sizeof(float)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetMipmapLevelBias(hTexRef, bias);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetMipmapLevelClamp(conn_t *conn) {
  CUtexref hTexRef;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &minMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_read(conn, &maxMipmapLevelClamp, sizeof(float)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetMipmapLevelClamp(
      hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetMaxAnisotropy(conn_t *conn) {
  CUtexref hTexRef;
  unsigned int maxAniso;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &maxAniso, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetMaxAnisotropy(hTexRef, maxAniso);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetBorderColor(conn_t *conn) {
  CUtexref hTexRef;
  float pBorderColor;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &pBorderColor, sizeof(float)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetBorderColor(hTexRef, &pBorderColor);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pBorderColor, sizeof(float)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefSetFlags(conn_t *conn) {
  CUtexref hTexRef;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefSetFlags(hTexRef, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetAddress_v2(conn_t *conn) {
  CUdeviceptr pdptr;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetAddress_v2(&pdptr, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetArray(conn_t *conn) {
  CUarray phArray;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetArray(&phArray, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetMipmappedArray(conn_t *conn) {
  CUmipmappedArray phMipmappedArray;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuTexRefGetMipmappedArray(&phMipmappedArray, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetAddressMode(conn_t *conn) {
  CUaddress_mode pam;
  CUtexref hTexRef;
  int dim;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pam, sizeof(CUaddress_mode)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &dim, sizeof(int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetAddressMode(&pam, hTexRef, dim);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pam, sizeof(CUaddress_mode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetFilterMode(conn_t *conn) {
  CUfilter_mode pfm;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetFilterMode(&pfm, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetFormat(conn_t *conn) {
  CUarray_format pFormat;
  int pNumChannels;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pFormat, sizeof(CUarray_format)) < 0 ||
      rpc_read(conn, &pNumChannels, sizeof(int)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetFormat(&pFormat, &pNumChannels, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pFormat, sizeof(CUarray_format)) < 0 ||
      rpc_write(conn, &pNumChannels, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetMipmapFilterMode(conn_t *conn) {
  CUfilter_mode pfm;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetMipmapFilterMode(&pfm, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetMipmapLevelBias(conn_t *conn) {
  float pbias;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pbias, sizeof(float)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetMipmapLevelBias(&pbias, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pbias, sizeof(float)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetMipmapLevelClamp(conn_t *conn) {
  float pminMipmapLevelClamp;
  float pmaxMipmapLevelClamp;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pminMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_read(conn, &pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetMipmapLevelClamp(
      &pminMipmapLevelClamp, &pmaxMipmapLevelClamp, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pminMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write(conn, &pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetMaxAnisotropy(conn_t *conn) {
  int pmaxAniso;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pmaxAniso, sizeof(int)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetMaxAnisotropy(&pmaxAniso, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pmaxAniso, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetBorderColor(conn_t *conn) {
  float pBorderColor;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pBorderColor, sizeof(float)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetBorderColor(&pBorderColor, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pBorderColor, sizeof(float)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefGetFlags(conn_t *conn) {
  unsigned int pFlags;
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefGetFlags(&pFlags, hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefCreate(conn_t *conn) {
  CUtexref pTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefCreate(&pTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexRefDestroy(conn_t *conn) {
  CUtexref hTexRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexRefDestroy(hTexRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuSurfRefSetArray(conn_t *conn) {
  CUsurfref hSurfRef;
  CUarray hArray;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_read(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuSurfRefSetArray(hSurfRef, hArray, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuSurfRefGetArray(conn_t *conn) {
  CUarray phArray;
  CUsurfref hSurfRef;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &phArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &hSurfRef, sizeof(CUsurfref)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuSurfRefGetArray(&phArray, hSurfRef);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexObjectCreate(conn_t *conn) {
  CUtexObject pTexObject;
  CUDA_RESOURCE_DESC pResDesc;
  CUDA_TEXTURE_DESC *pTexDesc_null_check;
  CUDA_TEXTURE_DESC pTexDesc;
  CUDA_RESOURCE_VIEW_DESC *pResViewDesc_null_check;
  CUDA_RESOURCE_VIEW_DESC pResViewDesc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pTexObject, sizeof(CUtexObject)) < 0 ||
      rpc_read(conn, &pResDesc, sizeof(const CUDA_RESOURCE_DESC)) < 0 ||
      rpc_read(conn, &pTexDesc_null_check, sizeof(const CUDA_TEXTURE_DESC *)) <
          0 ||
      (pTexDesc_null_check &&
       rpc_read(conn, &pTexDesc, sizeof(const CUDA_TEXTURE_DESC)) < 0) ||
      rpc_read(conn, &pResViewDesc_null_check,
               sizeof(const CUDA_RESOURCE_VIEW_DESC *)) < 0 ||
      (pResViewDesc_null_check &&
       rpc_read(conn, &pResViewDesc, sizeof(const CUDA_RESOURCE_VIEW_DESC)) <
           0) ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexObjectCreate(
      &pTexObject, &pResDesc, pTexDesc_null_check ? &pTexDesc : nullptr,
      pResViewDesc_null_check ? &pResViewDesc : nullptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pTexObject, sizeof(CUtexObject)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexObjectDestroy(conn_t *conn) {
  CUtexObject texObject;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexObjectDestroy(texObject);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexObjectGetResourceDesc(conn_t *conn) {
  CUDA_RESOURCE_DESC pResDesc;
  CUtexObject texObject;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexObjectGetResourceDesc(&pResDesc, texObject);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexObjectGetTextureDesc(conn_t *conn) {
  CUDA_TEXTURE_DESC pTexDesc;
  CUtexObject texObject;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
      rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuTexObjectGetTextureDesc(&pTexDesc, texObject);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuTexObjectGetResourceViewDesc(conn_t *conn) {
  CUDA_RESOURCE_VIEW_DESC pResViewDesc;
  CUtexObject texObject;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
      rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuTexObjectGetResourceViewDesc(&pResViewDesc, texObject);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuSurfObjectCreate(conn_t *conn) {
  CUsurfObject pSurfObject;
  CUDA_RESOURCE_DESC pResDesc;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pSurfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_read(conn, &pResDesc, sizeof(const CUDA_RESOURCE_DESC)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuSurfObjectCreate(&pSurfObject, &pResDesc);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pSurfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuSurfObjectDestroy(conn_t *conn) {
  CUsurfObject surfObject;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &surfObject, sizeof(CUsurfObject)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuSurfObjectDestroy(surfObject);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuSurfObjectGetResourceDesc(conn_t *conn) {
  CUDA_RESOURCE_DESC pResDesc;
  CUsurfObject surfObject;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_read(conn, &surfObject, sizeof(CUsurfObject)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuSurfObjectGetResourceDesc(&pResDesc, surfObject);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceCanAccessPeer(conn_t *conn) {
  int canAccessPeer;
  CUdevice dev;
  CUdevice peerDev;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &canAccessPeer, sizeof(int)) < 0 ||
      rpc_read(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_read(conn, &peerDev, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuDeviceCanAccessPeer(&canAccessPeer, dev, peerDev);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &canAccessPeer, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxEnablePeerAccess(conn_t *conn) {
  CUcontext peerContext;
  unsigned int Flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &peerContext, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxEnablePeerAccess(peerContext, Flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuCtxDisablePeerAccess(conn_t *conn) {
  CUcontext peerContext;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &peerContext, sizeof(CUcontext)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuCtxDisablePeerAccess(peerContext);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuDeviceGetP2PAttribute(conn_t *conn) {
  int value;
  CUdevice_P2PAttribute attrib;
  CUdevice srcDevice;
  CUdevice dstDevice;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &value, sizeof(int)) < 0 ||
      rpc_read(conn, &attrib, sizeof(CUdevice_P2PAttribute)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(CUdevice)) < 0 ||
      rpc_read(conn, &dstDevice, sizeof(CUdevice)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuDeviceGetP2PAttribute(&value, attrib, srcDevice, dstDevice);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(int)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphicsUnregisterResource(conn_t *conn) {
  CUgraphicsResource resource;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphicsUnregisterResource(resource);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphicsSubResourceGetMappedArray(conn_t *conn) {
  CUarray pArray;
  CUgraphicsResource resource;
  unsigned int arrayIndex;
  unsigned int mipLevel;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_read(conn, &arrayIndex, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &mipLevel, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphicsSubResourceGetMappedArray(
      &pArray, resource, arrayIndex, mipLevel);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphicsResourceGetMappedMipmappedArray(conn_t *conn) {
  CUmipmappedArray pMipmappedArray;
  CUgraphicsResource resource;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphicsResourceGetMappedMipmappedArray(&pMipmappedArray, resource);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphicsResourceGetMappedPointer_v2(conn_t *conn) {
  CUdeviceptr pDevPtr;
  size_t pSize;
  CUgraphicsResource resource;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &pDevPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &pSize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result =
      cuGraphicsResourceGetMappedPointer_v2(&pDevPtr, &pSize, resource);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pDevPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &pSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphicsResourceSetMapFlags_v2(conn_t *conn) {
  CUgraphicsResource resource;
  unsigned int flags;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  lupine_intercept_result = cuGraphicsResourceSetMapFlags_v2(resource, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cuGraphicsMapResources(conn_t *conn) {
  unsigned int count;
  CUgraphicsResource *resources;
  size_t resources_size;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &count, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;
  resources_size = count * sizeof(CUgraphicsResource);
  resources = (CUgraphicsResource *)malloc(resources_size);
  if (resources_size != 0 && resources == nullptr)
    goto ERROR_0;
  if ((resources_size != 0 && rpc_read(conn, resources, resources_size) < 0) ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphicsMapResources(
      count, (count * sizeof(CUgraphicsResource) == 0 ? nullptr : resources),
      hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)resources);
ERROR_0:
  return -1;
}

int handle_cuGraphicsUnmapResources(conn_t *conn) {
  unsigned int count;
  CUgraphicsResource *resources;
  size_t resources_size;
  CUstream hStream;
  int request_id;
  CUresult lupine_intercept_result;
  if (rpc_read(conn, &count, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;
  resources_size = count * sizeof(CUgraphicsResource);
  resources = (CUgraphicsResource *)malloc(resources_size);
  if (resources_size != 0 && resources == nullptr)
    goto ERROR_0;
  if ((resources_size != 0 && rpc_read(conn, resources, resources_size) < 0) ||
      rpc_read(conn, &hStream, sizeof(CUstream)) < 0 || false)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;
  lupine_intercept_result = cuGraphicsUnmapResources(
      count, (count * sizeof(CUgraphicsResource) == 0 ? nullptr : resources),
      hStream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &lupine_intercept_result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  free((void *)resources);
ERROR_0:
  return -1;
}

static const std::unordered_map<int, RequestHandler> opHandlers = {
    {RPC_cuInit, handle_cuInit},
    {RPC_cuDriverGetVersion, handle_cuDriverGetVersion},
    {RPC_cuDeviceGet, handle_cuDeviceGet},
    {RPC_cuDeviceGetCount, handle_cuDeviceGetCount},
    {RPC_cuDeviceGetName, handle_cuDeviceGetName},
    {RPC_cuDeviceGetUuid_v2, handle_cuDeviceGetUuid_v2},
    {RPC_cuDeviceGetLuid, handle_cuDeviceGetLuid},
    {RPC_cuDeviceTotalMem_v2, handle_cuDeviceTotalMem_v2},
    {RPC_cuDeviceGetTexture1DLinearMaxWidth,
     handle_cuDeviceGetTexture1DLinearMaxWidth},
    {RPC_cuDeviceGetAttribute, handle_cuDeviceGetAttribute},
    {RPC_cuDeviceSetMemPool, handle_cuDeviceSetMemPool},
    {RPC_cuDeviceGetMemPool, handle_cuDeviceGetMemPool},
    {RPC_cuDeviceGetDefaultMemPool, handle_cuDeviceGetDefaultMemPool},
    {RPC_cuDeviceGetExecAffinitySupport, handle_cuDeviceGetExecAffinitySupport},
    {RPC_cuFlushGPUDirectRDMAWrites, handle_cuFlushGPUDirectRDMAWrites},
    {RPC_cuDeviceGetProperties, handle_cuDeviceGetProperties},
    {RPC_cuDeviceComputeCapability, handle_cuDeviceComputeCapability},
    {RPC_cuDevicePrimaryCtxRetain, handle_cuDevicePrimaryCtxRetain},
    {RPC_cuDevicePrimaryCtxRelease_v2, handle_cuDevicePrimaryCtxRelease_v2},
    {RPC_cuDevicePrimaryCtxSetFlags_v2, handle_cuDevicePrimaryCtxSetFlags_v2},
    {RPC_cuDevicePrimaryCtxGetState, handle_cuDevicePrimaryCtxGetState},
    {RPC_cuDevicePrimaryCtxReset_v2, handle_cuDevicePrimaryCtxReset_v2},
    {RPC_cuCtxDestroy_v2, handle_cuCtxDestroy_v2},
    {RPC_cuCtxPushCurrent_v2, handle_cuCtxPushCurrent_v2},
    {RPC_cuCtxPopCurrent_v2, handle_cuCtxPopCurrent_v2},
    {RPC_cuCtxSetCurrent, handle_cuCtxSetCurrent},
    {RPC_cuCtxGetCurrent, handle_cuCtxGetCurrent},
    {RPC_cuCtxGetDevice, handle_cuCtxGetDevice},
    {RPC_cuCtxGetFlags, handle_cuCtxGetFlags},
    {RPC_cuCtxGetId, handle_cuCtxGetId},
    {RPC_cuCtxSetLimit, handle_cuCtxSetLimit},
    {RPC_cuCtxGetLimit, handle_cuCtxGetLimit},
    {RPC_cuCtxGetCacheConfig, handle_cuCtxGetCacheConfig},
    {RPC_cuCtxSetCacheConfig, handle_cuCtxSetCacheConfig},
    {RPC_cuCtxGetApiVersion, handle_cuCtxGetApiVersion},
    {RPC_cuCtxGetStreamPriorityRange, handle_cuCtxGetStreamPriorityRange},
    {RPC_cuCtxResetPersistingL2Cache, handle_cuCtxResetPersistingL2Cache},
    {RPC_cuCtxGetExecAffinity, handle_cuCtxGetExecAffinity},
    {RPC_cuCtxAttach, handle_cuCtxAttach},
    {RPC_cuCtxDetach, handle_cuCtxDetach},
    {RPC_cuCtxGetSharedMemConfig, handle_cuCtxGetSharedMemConfig},
    {RPC_cuCtxSetSharedMemConfig, handle_cuCtxSetSharedMemConfig},
    {RPC_cuModuleUnload, handle_cuModuleUnload},
    {RPC_cuModuleGetLoadingMode, handle_cuModuleGetLoadingMode},
    {RPC_cuModuleGetFunction, handle_cuModuleGetFunction},
    {RPC_cuModuleGetGlobal_v2, handle_cuModuleGetGlobal_v2},
    {RPC_cuLinkCreate_v2, handle_cuLinkCreate_v2},
    {RPC_cuLinkAddData_v2, handle_cuLinkAddData_v2},
    {RPC_cuLinkAddFile_v2, handle_cuLinkAddFile_v2},
    {RPC_cuLinkComplete, handle_cuLinkComplete},
    {RPC_cuLinkDestroy, handle_cuLinkDestroy},
    {RPC_cuModuleGetTexRef, handle_cuModuleGetTexRef},
    {RPC_cuModuleGetSurfRef, handle_cuModuleGetSurfRef},
    {RPC_cuLibraryLoadFromFile, handle_cuLibraryLoadFromFile},
    {RPC_cuLibraryUnload, handle_cuLibraryUnload},
    {RPC_cuLibraryGetKernel, handle_cuLibraryGetKernel},
    {RPC_cuLibraryGetModule, handle_cuLibraryGetModule},
    {RPC_cuKernelGetFunction, handle_cuKernelGetFunction},
    {RPC_cuLibraryGetGlobal, handle_cuLibraryGetGlobal},
    {RPC_cuLibraryGetManaged, handle_cuLibraryGetManaged},
    {RPC_cuLibraryGetUnifiedFunction, handle_cuLibraryGetUnifiedFunction},
    {RPC_cuKernelGetAttribute, handle_cuKernelGetAttribute},
    {RPC_cuKernelSetAttribute, handle_cuKernelSetAttribute},
    {RPC_cuKernelSetCacheConfig, handle_cuKernelSetCacheConfig},
    {RPC_cuMemGetInfo_v2, handle_cuMemGetInfo_v2},
    {RPC_cuMemAlloc_v2, handle_cuMemAlloc_v2},
    {RPC_cuMemAllocPitch_v2, handle_cuMemAllocPitch_v2},
    {RPC_cuMemFree_v2, handle_cuMemFree_v2},
    {RPC_cuMemGetAddressRange_v2, handle_cuMemGetAddressRange_v2},
    {RPC_cuMemAllocManaged, handle_cuMemAllocManaged},
    {RPC_cuDeviceGetByPCIBusId, handle_cuDeviceGetByPCIBusId},
    {RPC_cuDeviceGetPCIBusId, handle_cuDeviceGetPCIBusId},
    {RPC_cuIpcGetEventHandle, handle_cuIpcGetEventHandle},
    {RPC_cuIpcOpenEventHandle, handle_cuIpcOpenEventHandle},
    {RPC_cuIpcGetMemHandle, handle_cuIpcGetMemHandle},
    {RPC_cuIpcOpenMemHandle_v2, handle_cuIpcOpenMemHandle_v2},
    {RPC_cuIpcCloseMemHandle, handle_cuIpcCloseMemHandle},
    {RPC_cuMemcpy, handle_cuMemcpy},
    {RPC_cuMemcpyPeer, handle_cuMemcpyPeer},
    {RPC_cuMemcpyHtoD_v2, handle_cuMemcpyHtoD_v2},
    {RPC_cuMemcpyDtoD_v2, handle_cuMemcpyDtoD_v2},
    {RPC_cuMemcpyDtoA_v2, handle_cuMemcpyDtoA_v2},
    {RPC_cuMemcpyAtoD_v2, handle_cuMemcpyAtoD_v2},
    {RPC_cuMemcpyAtoA_v2, handle_cuMemcpyAtoA_v2},
    {RPC_cuMemcpyPeerAsync, handle_cuMemcpyPeerAsync},
    {RPC_cuMemcpyDtoDAsync_v2, handle_cuMemcpyDtoDAsync_v2},
    {RPC_cuMemsetD8_v2, handle_cuMemsetD8_v2},
    {RPC_cuMemsetD16_v2, handle_cuMemsetD16_v2},
    {RPC_cuMemsetD32_v2, handle_cuMemsetD32_v2},
    {RPC_cuMemsetD2D8_v2, handle_cuMemsetD2D8_v2},
    {RPC_cuMemsetD2D16_v2, handle_cuMemsetD2D16_v2},
    {RPC_cuMemsetD2D32_v2, handle_cuMemsetD2D32_v2},
    {RPC_cuMemsetD8Async, handle_cuMemsetD8Async},
    {RPC_cuMemsetD16Async, handle_cuMemsetD16Async},
    {RPC_cuMemsetD32Async, handle_cuMemsetD32Async},
    {RPC_cuMemsetD2D8Async, handle_cuMemsetD2D8Async},
    {RPC_cuMemsetD2D16Async, handle_cuMemsetD2D16Async},
    {RPC_cuMemsetD2D32Async, handle_cuMemsetD2D32Async},
    {RPC_cuArrayCreate_v2, handle_cuArrayCreate_v2},
    {RPC_cuArrayGetDescriptor_v2, handle_cuArrayGetDescriptor_v2},
    {RPC_cuArrayGetSparseProperties, handle_cuArrayGetSparseProperties},
    {RPC_cuMipmappedArrayGetSparseProperties,
     handle_cuMipmappedArrayGetSparseProperties},
    {RPC_cuArrayGetMemoryRequirements, handle_cuArrayGetMemoryRequirements},
    {RPC_cuMipmappedArrayGetMemoryRequirements,
     handle_cuMipmappedArrayGetMemoryRequirements},
    {RPC_cuArrayGetPlane, handle_cuArrayGetPlane},
    {RPC_cuArrayDestroy, handle_cuArrayDestroy},
    {RPC_cuArray3DCreate_v2, handle_cuArray3DCreate_v2},
    {RPC_cuArray3DGetDescriptor_v2, handle_cuArray3DGetDescriptor_v2},
    {RPC_cuMipmappedArrayCreate, handle_cuMipmappedArrayCreate},
    {RPC_cuMipmappedArrayGetLevel, handle_cuMipmappedArrayGetLevel},
    {RPC_cuMipmappedArrayDestroy, handle_cuMipmappedArrayDestroy},
    {RPC_cuMemAddressReserve, handle_cuMemAddressReserve},
    {RPC_cuMemAddressFree, handle_cuMemAddressFree},
    {RPC_cuMemCreate, handle_cuMemCreate},
    {RPC_cuMemRelease, handle_cuMemRelease},
    {RPC_cuMemMap, handle_cuMemMap},
    {RPC_cuMemMapArrayAsync, handle_cuMemMapArrayAsync},
    {RPC_cuMemUnmap, handle_cuMemUnmap},
    {RPC_cuMemSetAccess, handle_cuMemSetAccess},
    {RPC_cuMemGetAccess, handle_cuMemGetAccess},
    {RPC_cuMemExportToShareableHandle, handle_cuMemExportToShareableHandle},
    {RPC_cuMemImportFromShareableHandle, handle_cuMemImportFromShareableHandle},
    {RPC_cuMemGetAllocationGranularity, handle_cuMemGetAllocationGranularity},
    {RPC_cuMemGetAllocationPropertiesFromHandle,
     handle_cuMemGetAllocationPropertiesFromHandle},
    {RPC_cuMemFreeAsync, handle_cuMemFreeAsync},
    {RPC_cuMemAllocAsync, handle_cuMemAllocAsync},
    {RPC_cuMemPoolTrimTo, handle_cuMemPoolTrimTo},
    {RPC_cuMemPoolSetAccess, handle_cuMemPoolSetAccess},
    {RPC_cuMemPoolGetAccess, handle_cuMemPoolGetAccess},
    {RPC_cuMemPoolCreate, handle_cuMemPoolCreate},
    {RPC_cuMemPoolDestroy, handle_cuMemPoolDestroy},
    {RPC_cuMemAllocFromPoolAsync, handle_cuMemAllocFromPoolAsync},
    {RPC_cuMemPoolExportToShareableHandle,
     handle_cuMemPoolExportToShareableHandle},
    {RPC_cuMemPoolImportFromShareableHandle,
     handle_cuMemPoolImportFromShareableHandle},
    {RPC_cuMemPoolExportPointer, handle_cuMemPoolExportPointer},
    {RPC_cuMemPoolImportPointer, handle_cuMemPoolImportPointer},
    {RPC_cuMemRangeGetAttributes, handle_cuMemRangeGetAttributes},
    {RPC_cuPointerSetAttribute, handle_cuPointerSetAttribute},
    {RPC_cuPointerGetAttributes, handle_cuPointerGetAttributes},
    {RPC_cuStreamCreate, handle_cuStreamCreate},
    {RPC_cuStreamCreateWithPriority, handle_cuStreamCreateWithPriority},
    {RPC_cuStreamGetPriority, handle_cuStreamGetPriority},
    {RPC_cuStreamGetFlags, handle_cuStreamGetFlags},
    {RPC_cuStreamGetId, handle_cuStreamGetId},
    {RPC_cuStreamGetCtx, handle_cuStreamGetCtx},
    {RPC_cuStreamBeginCapture_v2, handle_cuStreamBeginCapture_v2},
    {RPC_cuThreadExchangeStreamCaptureMode,
     handle_cuThreadExchangeStreamCaptureMode},
    {RPC_cuStreamEndCapture, handle_cuStreamEndCapture},
    {RPC_cuStreamIsCapturing, handle_cuStreamIsCapturing},
    {RPC_cuStreamAttachMemAsync, handle_cuStreamAttachMemAsync},
    {RPC_cuStreamQuery, handle_cuStreamQuery},
    {RPC_cuStreamDestroy_v2, handle_cuStreamDestroy_v2},
    {RPC_cuStreamCopyAttributes, handle_cuStreamCopyAttributes},
    {RPC_cuStreamGetAttribute, handle_cuStreamGetAttribute},
    {RPC_cuStreamSetAttribute, handle_cuStreamSetAttribute},
    {RPC_cuEventCreate, handle_cuEventCreate},
    {RPC_cuEventRecord, handle_cuEventRecord},
    {RPC_cuEventRecordWithFlags, handle_cuEventRecordWithFlags},
    {RPC_cuEventDestroy_v2, handle_cuEventDestroy_v2},
    {RPC_cuEventElapsedTime_v2, handle_cuEventElapsedTime_v2},
    {RPC_cuImportExternalMemory, handle_cuImportExternalMemory},
    {RPC_cuExternalMemoryGetMappedBuffer,
     handle_cuExternalMemoryGetMappedBuffer},
    {RPC_cuExternalMemoryGetMappedMipmappedArray,
     handle_cuExternalMemoryGetMappedMipmappedArray},
    {RPC_cuDestroyExternalMemory, handle_cuDestroyExternalMemory},
    {RPC_cuImportExternalSemaphore, handle_cuImportExternalSemaphore},
    {RPC_cuSignalExternalSemaphoresAsync,
     handle_cuSignalExternalSemaphoresAsync},
    {RPC_cuWaitExternalSemaphoresAsync, handle_cuWaitExternalSemaphoresAsync},
    {RPC_cuDestroyExternalSemaphore, handle_cuDestroyExternalSemaphore},
    {RPC_cuStreamWaitValue32_v2, handle_cuStreamWaitValue32_v2},
    {RPC_cuStreamWaitValue64_v2, handle_cuStreamWaitValue64_v2},
    {RPC_cuStreamWriteValue32_v2, handle_cuStreamWriteValue32_v2},
    {RPC_cuStreamWriteValue64_v2, handle_cuStreamWriteValue64_v2},
    {RPC_cuStreamBatchMemOp_v2, handle_cuStreamBatchMemOp_v2},
    {RPC_cuFuncGetAttribute, handle_cuFuncGetAttribute},
    {RPC_cuFuncSetAttribute, handle_cuFuncSetAttribute},
    {RPC_cuFuncSetCacheConfig, handle_cuFuncSetCacheConfig},
    {RPC_cuFuncGetModule, handle_cuFuncGetModule},
    {RPC_cuLaunchCooperativeKernel, handle_cuLaunchCooperativeKernel},
    {RPC_cuLaunchCooperativeKernelMultiDevice,
     handle_cuLaunchCooperativeKernelMultiDevice},
    {RPC_cuFuncSetBlockShape, handle_cuFuncSetBlockShape},
    {RPC_cuFuncSetSharedSize, handle_cuFuncSetSharedSize},
    {RPC_cuParamSetSize, handle_cuParamSetSize},
    {RPC_cuParamSeti, handle_cuParamSeti},
    {RPC_cuParamSetf, handle_cuParamSetf},
    {RPC_cuLaunch, handle_cuLaunch},
    {RPC_cuLaunchGrid, handle_cuLaunchGrid},
    {RPC_cuLaunchGridAsync, handle_cuLaunchGridAsync},
    {RPC_cuParamSetTexRef, handle_cuParamSetTexRef},
    {RPC_cuFuncSetSharedMemConfig, handle_cuFuncSetSharedMemConfig},
    {RPC_cuGraphCreate, handle_cuGraphCreate},
    {RPC_cuGraphMemcpyNodeGetParams, handle_cuGraphMemcpyNodeGetParams},
    {RPC_cuGraphMemcpyNodeSetParams, handle_cuGraphMemcpyNodeSetParams},
    {RPC_cuGraphMemsetNodeGetParams, handle_cuGraphMemsetNodeGetParams},
    {RPC_cuGraphMemsetNodeSetParams, handle_cuGraphMemsetNodeSetParams},
    {RPC_cuGraphAddChildGraphNode, handle_cuGraphAddChildGraphNode},
    {RPC_cuGraphChildGraphNodeGetGraph, handle_cuGraphChildGraphNodeGetGraph},
    {RPC_cuGraphAddEmptyNode, handle_cuGraphAddEmptyNode},
    {RPC_cuGraphAddEventRecordNode, handle_cuGraphAddEventRecordNode},
    {RPC_cuGraphEventRecordNodeGetEvent, handle_cuGraphEventRecordNodeGetEvent},
    {RPC_cuGraphEventRecordNodeSetEvent, handle_cuGraphEventRecordNodeSetEvent},
    {RPC_cuGraphAddEventWaitNode, handle_cuGraphAddEventWaitNode},
    {RPC_cuGraphEventWaitNodeGetEvent, handle_cuGraphEventWaitNodeGetEvent},
    {RPC_cuGraphEventWaitNodeSetEvent, handle_cuGraphEventWaitNodeSetEvent},
    {RPC_cuGraphAddExternalSemaphoresSignalNode,
     handle_cuGraphAddExternalSemaphoresSignalNode},
    {RPC_cuGraphExternalSemaphoresSignalNodeGetParams,
     handle_cuGraphExternalSemaphoresSignalNodeGetParams},
    {RPC_cuGraphExternalSemaphoresSignalNodeSetParams,
     handle_cuGraphExternalSemaphoresSignalNodeSetParams},
    {RPC_cuGraphAddExternalSemaphoresWaitNode,
     handle_cuGraphAddExternalSemaphoresWaitNode},
    {RPC_cuGraphExternalSemaphoresWaitNodeGetParams,
     handle_cuGraphExternalSemaphoresWaitNodeGetParams},
    {RPC_cuGraphExternalSemaphoresWaitNodeSetParams,
     handle_cuGraphExternalSemaphoresWaitNodeSetParams},
    {RPC_cuGraphAddBatchMemOpNode, handle_cuGraphAddBatchMemOpNode},
    {RPC_cuGraphBatchMemOpNodeGetParams, handle_cuGraphBatchMemOpNodeGetParams},
    {RPC_cuGraphBatchMemOpNodeSetParams, handle_cuGraphBatchMemOpNodeSetParams},
    {RPC_cuGraphExecBatchMemOpNodeSetParams,
     handle_cuGraphExecBatchMemOpNodeSetParams},
    {RPC_cuGraphAddMemAllocNode, handle_cuGraphAddMemAllocNode},
    {RPC_cuGraphMemAllocNodeGetParams, handle_cuGraphMemAllocNodeGetParams},
    {RPC_cuGraphAddMemFreeNode, handle_cuGraphAddMemFreeNode},
    {RPC_cuGraphMemFreeNodeGetParams, handle_cuGraphMemFreeNodeGetParams},
    {RPC_cuDeviceGraphMemTrim, handle_cuDeviceGraphMemTrim},
    {RPC_cuGraphClone, handle_cuGraphClone},
    {RPC_cuGraphNodeFindInClone, handle_cuGraphNodeFindInClone},
    {RPC_cuGraphNodeGetType, handle_cuGraphNodeGetType},
    {RPC_cuGraphGetNodes, handle_cuGraphGetNodes},
    {RPC_cuGraphGetRootNodes, handle_cuGraphGetRootNodes},
    {RPC_cuGraphDestroyNode, handle_cuGraphDestroyNode},
    {RPC_cuGraphInstantiateWithFlags, handle_cuGraphInstantiateWithFlags},
    {RPC_cuGraphInstantiateWithParams, handle_cuGraphInstantiateWithParams},
    {RPC_cuGraphExecGetFlags, handle_cuGraphExecGetFlags},
    {RPC_cuGraphExecMemcpyNodeSetParams, handle_cuGraphExecMemcpyNodeSetParams},
    {RPC_cuGraphExecMemsetNodeSetParams, handle_cuGraphExecMemsetNodeSetParams},
    {RPC_cuGraphExecChildGraphNodeSetParams,
     handle_cuGraphExecChildGraphNodeSetParams},
    {RPC_cuGraphExecEventRecordNodeSetEvent,
     handle_cuGraphExecEventRecordNodeSetEvent},
    {RPC_cuGraphExecEventWaitNodeSetEvent,
     handle_cuGraphExecEventWaitNodeSetEvent},
    {RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams,
     handle_cuGraphExecExternalSemaphoresSignalNodeSetParams},
    {RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams,
     handle_cuGraphExecExternalSemaphoresWaitNodeSetParams},
    {RPC_cuGraphNodeSetEnabled, handle_cuGraphNodeSetEnabled},
    {RPC_cuGraphNodeGetEnabled, handle_cuGraphNodeGetEnabled},
    {RPC_cuGraphUpload, handle_cuGraphUpload},
    {RPC_cuGraphLaunch, handle_cuGraphLaunch},
    {RPC_cuGraphExecDestroy, handle_cuGraphExecDestroy},
    {RPC_cuGraphDestroy, handle_cuGraphDestroy},
    {RPC_cuGraphExecUpdate_v2, handle_cuGraphExecUpdate_v2},
    {RPC_cuGraphKernelNodeCopyAttributes,
     handle_cuGraphKernelNodeCopyAttributes},
    {RPC_cuGraphKernelNodeGetAttribute, handle_cuGraphKernelNodeGetAttribute},
    {RPC_cuGraphKernelNodeSetAttribute, handle_cuGraphKernelNodeSetAttribute},
    {RPC_cuGraphDebugDotPrint, handle_cuGraphDebugDotPrint},
    {RPC_cuUserObjectRetain, handle_cuUserObjectRetain},
    {RPC_cuUserObjectRelease, handle_cuUserObjectRelease},
    {RPC_cuGraphRetainUserObject, handle_cuGraphRetainUserObject},
    {RPC_cuGraphReleaseUserObject, handle_cuGraphReleaseUserObject},
    {RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor,
     handle_cuOccupancyMaxActiveBlocksPerMultiprocessor},
    {RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
     handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {RPC_cuOccupancyAvailableDynamicSMemPerBlock,
     handle_cuOccupancyAvailableDynamicSMemPerBlock},
    {RPC_cuOccupancyMaxPotentialClusterSize,
     handle_cuOccupancyMaxPotentialClusterSize},
    {RPC_cuOccupancyMaxActiveClusters, handle_cuOccupancyMaxActiveClusters},
    {RPC_cuTexRefSetArray, handle_cuTexRefSetArray},
    {RPC_cuTexRefSetMipmappedArray, handle_cuTexRefSetMipmappedArray},
    {RPC_cuTexRefSetAddress_v2, handle_cuTexRefSetAddress_v2},
    {RPC_cuTexRefSetAddress2D_v3, handle_cuTexRefSetAddress2D_v3},
    {RPC_cuTexRefSetFormat, handle_cuTexRefSetFormat},
    {RPC_cuTexRefSetAddressMode, handle_cuTexRefSetAddressMode},
    {RPC_cuTexRefSetFilterMode, handle_cuTexRefSetFilterMode},
    {RPC_cuTexRefSetMipmapFilterMode, handle_cuTexRefSetMipmapFilterMode},
    {RPC_cuTexRefSetMipmapLevelBias, handle_cuTexRefSetMipmapLevelBias},
    {RPC_cuTexRefSetMipmapLevelClamp, handle_cuTexRefSetMipmapLevelClamp},
    {RPC_cuTexRefSetMaxAnisotropy, handle_cuTexRefSetMaxAnisotropy},
    {RPC_cuTexRefSetBorderColor, handle_cuTexRefSetBorderColor},
    {RPC_cuTexRefSetFlags, handle_cuTexRefSetFlags},
    {RPC_cuTexRefGetAddress_v2, handle_cuTexRefGetAddress_v2},
    {RPC_cuTexRefGetArray, handle_cuTexRefGetArray},
    {RPC_cuTexRefGetMipmappedArray, handle_cuTexRefGetMipmappedArray},
    {RPC_cuTexRefGetAddressMode, handle_cuTexRefGetAddressMode},
    {RPC_cuTexRefGetFilterMode, handle_cuTexRefGetFilterMode},
    {RPC_cuTexRefGetFormat, handle_cuTexRefGetFormat},
    {RPC_cuTexRefGetMipmapFilterMode, handle_cuTexRefGetMipmapFilterMode},
    {RPC_cuTexRefGetMipmapLevelBias, handle_cuTexRefGetMipmapLevelBias},
    {RPC_cuTexRefGetMipmapLevelClamp, handle_cuTexRefGetMipmapLevelClamp},
    {RPC_cuTexRefGetMaxAnisotropy, handle_cuTexRefGetMaxAnisotropy},
    {RPC_cuTexRefGetBorderColor, handle_cuTexRefGetBorderColor},
    {RPC_cuTexRefGetFlags, handle_cuTexRefGetFlags},
    {RPC_cuTexRefCreate, handle_cuTexRefCreate},
    {RPC_cuTexRefDestroy, handle_cuTexRefDestroy},
    {RPC_cuSurfRefSetArray, handle_cuSurfRefSetArray},
    {RPC_cuSurfRefGetArray, handle_cuSurfRefGetArray},
    {RPC_cuTexObjectCreate, handle_cuTexObjectCreate},
    {RPC_cuTexObjectDestroy, handle_cuTexObjectDestroy},
    {RPC_cuTexObjectGetResourceDesc, handle_cuTexObjectGetResourceDesc},
    {RPC_cuTexObjectGetTextureDesc, handle_cuTexObjectGetTextureDesc},
    {RPC_cuTexObjectGetResourceViewDesc, handle_cuTexObjectGetResourceViewDesc},
    {RPC_cuSurfObjectCreate, handle_cuSurfObjectCreate},
    {RPC_cuSurfObjectDestroy, handle_cuSurfObjectDestroy},
    {RPC_cuSurfObjectGetResourceDesc, handle_cuSurfObjectGetResourceDesc},
    {RPC_cuDeviceCanAccessPeer, handle_cuDeviceCanAccessPeer},
    {RPC_cuCtxEnablePeerAccess, handle_cuCtxEnablePeerAccess},
    {RPC_cuCtxDisablePeerAccess, handle_cuCtxDisablePeerAccess},
    {RPC_cuDeviceGetP2PAttribute, handle_cuDeviceGetP2PAttribute},
    {RPC_cuGraphicsUnregisterResource, handle_cuGraphicsUnregisterResource},
    {RPC_cuGraphicsSubResourceGetMappedArray,
     handle_cuGraphicsSubResourceGetMappedArray},
    {RPC_cuGraphicsResourceGetMappedMipmappedArray,
     handle_cuGraphicsResourceGetMappedMipmappedArray},
    {RPC_cuGraphicsResourceGetMappedPointer_v2,
     handle_cuGraphicsResourceGetMappedPointer_v2},
    {RPC_cuGraphicsResourceSetMapFlags_v2,
     handle_cuGraphicsResourceSetMapFlags_v2},
    {RPC_cuGraphicsMapResources, handle_cuGraphicsMapResources},
    {RPC_cuGraphicsUnmapResources, handle_cuGraphicsUnmapResources},
    {RPC_nvmlInit_v2, handle_nvmlInit_v2},
    {RPC_nvmlInitWithFlags, handle_nvmlInitWithFlags},
    {RPC_nvmlShutdown, handle_nvmlShutdown},
    {RPC_nvmlSystemGetDriverVersion, handle_nvmlSystemGetDriverVersion},
    {RPC_nvmlSystemGetNVMLVersion, handle_nvmlSystemGetNVMLVersion},
    {RPC_nvmlSystemGetCudaDriverVersion, handle_nvmlSystemGetCudaDriverVersion},
    {RPC_nvmlSystemGetCudaDriverVersion_v2,
     handle_nvmlSystemGetCudaDriverVersion_v2},
    {RPC_nvmlDeviceGetCount_v2, handle_nvmlDeviceGetCount_v2},
    {RPC_nvmlDeviceGetHandleByIndex_v2, handle_nvmlDeviceGetHandleByIndex_v2},
    {RPC_nvmlDeviceGetHandleByUUID, handle_nvmlDeviceGetHandleByUUID},
    {RPC_nvmlDeviceGetHandleByPciBusId_v2,
     handle_nvmlDeviceGetHandleByPciBusId_v2},
    {RPC_nvmlDeviceGetName, handle_nvmlDeviceGetName},
    {RPC_nvmlDeviceGetUUID, handle_nvmlDeviceGetUUID},
    {RPC_nvmlDeviceGetIndex, handle_nvmlDeviceGetIndex},
    {RPC_nvmlDeviceGetMinorNumber, handle_nvmlDeviceGetMinorNumber},
    {RPC_nvmlDeviceGetPciInfo_v3, handle_nvmlDeviceGetPciInfo_v3},
    {RPC_nvmlDeviceGetMemoryInfo, handle_nvmlDeviceGetMemoryInfo},
    {RPC_nvmlDeviceGetUtilizationRates, handle_nvmlDeviceGetUtilizationRates},
    {RPC_nvmlDeviceGetTemperature, handle_nvmlDeviceGetTemperature},
    {RPC_nvmlDeviceGetPowerUsage, handle_nvmlDeviceGetPowerUsage},
    {RPC_nvmlDeviceGetPowerManagementLimit,
     handle_nvmlDeviceGetPowerManagementLimit},
    {RPC_nvmlDeviceGetClockInfo, handle_nvmlDeviceGetClockInfo},
    {RPC_nvmlDeviceGetMaxClockInfo, handle_nvmlDeviceGetMaxClockInfo},
    {RPC_nvmlDeviceGetPerformanceState, handle_nvmlDeviceGetPerformanceState},
    {RPC_nvmlDeviceGetComputeMode, handle_nvmlDeviceGetComputeMode},
    {RPC_nvmlDeviceGetPersistenceMode, handle_nvmlDeviceGetPersistenceMode},
    {RPC_nvmlDeviceGetFanSpeed, handle_nvmlDeviceGetFanSpeed},
    {RPC_nvmlDeviceGetBrand, handle_nvmlDeviceGetBrand},
    {RPC_nvmlDeviceGetVbiosVersion, handle_nvmlDeviceGetVbiosVersion},
    {RPC_nvmlDeviceGetSerial, handle_nvmlDeviceGetSerial},
    {RPC_nvmlDeviceGetBoardPartNumber, handle_nvmlDeviceGetBoardPartNumber},
    {RPC_nvmlDeviceGetDisplayMode, handle_nvmlDeviceGetDisplayMode},
    {RPC_nvmlDeviceGetDisplayActive, handle_nvmlDeviceGetDisplayActive},
    {RPC_nvmlDeviceGetCurrPcieLinkGeneration,
     handle_nvmlDeviceGetCurrPcieLinkGeneration},
    {RPC_nvmlDeviceGetCurrPcieLinkWidth, handle_nvmlDeviceGetCurrPcieLinkWidth},
    {RPC_nvmlDeviceGetMaxPcieLinkGeneration,
     handle_nvmlDeviceGetMaxPcieLinkGeneration},
    {RPC_nvmlDeviceGetMaxPcieLinkWidth, handle_nvmlDeviceGetMaxPcieLinkWidth},
    {RPC_nvmlDeviceGetPcieThroughput, handle_nvmlDeviceGetPcieThroughput},
    {RPC_nvmlDeviceGetPcieReplayCounter, handle_nvmlDeviceGetPcieReplayCounter},
    {RPC_nvmlDeviceGetComputeRunningProcesses,
     handle_nvmlDeviceGetComputeRunningProcesses},
    {RPC_nvmlDeviceGetComputeRunningProcesses_v2,
     handle_nvmlDeviceGetComputeRunningProcesses_v2},
    {RPC_nvmlDeviceGetGraphicsRunningProcesses,
     handle_nvmlDeviceGetGraphicsRunningProcesses},
    {RPC_nvmlDeviceGetGraphicsRunningProcesses_v2,
     handle_nvmlDeviceGetGraphicsRunningProcesses_v2},
    {RPC_nvmlDeviceGetMPSComputeRunningProcesses,
     handle_nvmlDeviceGetMPSComputeRunningProcesses},
    {RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2,
     handle_nvmlDeviceGetMPSComputeRunningProcesses_v2},
    {RPC_nvmlEventSetCreate, handle_nvmlEventSetCreate},
    {RPC_nvmlEventSetFree, handle_nvmlEventSetFree},
    {RPC_nvmlEventSetWait_v2, handle_nvmlEventSetWait_v2},
    {RPC_nvmlDeviceRegisterEvents, handle_nvmlDeviceRegisterEvents},
    {RPC_nvmlDeviceGetMaxMigDeviceCount, handle_nvmlDeviceGetMaxMigDeviceCount},
    {RPC_nvmlDeviceGetTotalEccErrors, handle_nvmlDeviceGetTotalEccErrors},
    {RPC_nvmlDeviceGetDetailedEccErrors, handle_nvmlDeviceGetDetailedEccErrors},
    {RPC_nvmlDeviceGetMemoryErrorCounter,
     handle_nvmlDeviceGetMemoryErrorCounter},
    {RPC_nvmlDeviceGetEccMode, handle_nvmlDeviceGetEccMode},
    {RPC_nvmlDeviceGetTemperatureV, handle_nvmlDeviceGetTemperatureV},
    {RPC_nvmlDeviceGetEnforcedPowerLimit,
     handle_nvmlDeviceGetEnforcedPowerLimit},
    {RPC_nvmlDeviceGetMemoryInfo_v2, handle_nvmlDeviceGetMemoryInfo_v2},
    {RPC_nvmlDeviceGetMigMode, handle_nvmlDeviceGetMigMode},
    {RPC_nvmlDeviceGetVirtualizationMode,
     handle_nvmlDeviceGetVirtualizationMode},
    {RPC_nvmlDeviceIsMigDeviceHandle, handle_nvmlDeviceIsMigDeviceHandle},
    {RPC_nvmlDeviceGetNvLinkRemoteDeviceType,
     handle_nvmlDeviceGetNvLinkRemoteDeviceType},
    {RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2,
     handle_nvmlDeviceGetNvLinkRemotePciInfo_v2},
};

RequestHandler get_handler(const int op) {
  auto it = opHandlers.find(op);
  if (it == opHandlers.end())
    return nullptr;
  return it->second;
}
