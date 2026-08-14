// Verifies that module function lookup and parameter-layout prefetch use
// separate RPCs, and that successful prefetches warm cuFuncGetParamInfo.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>

#include <sys/wait.h>
#include <unistd.h>

#include "../codegen/gen_api.h"

static const char kParamPtx[] =
    ".version 6.4\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry parameterized(\n"
    "    .param .u64 pointer,\n"
    "    .param .u32 value)\n"
    "{\n"
    "  ret;\n"
    "}\n";

static const char *error_name(CUresult result) {
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  return name == nullptr ? "unknown" : name;
}

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s (%d)\n", operation,
               error_name(result), static_cast<int>(result));
  return false;
}

static int exercise_snapshot() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuModuleLoadData(&module, kParamPtx), "cuModuleLoadData") ||
      !check(cuModuleGetFunction(&function, module, "parameterized"),
             "cuModuleGetFunction")) {
    return 1;
  }

  size_t offset = 0;
  size_t size = 0;
  if (!check(cuFuncGetParamInfo(function, 0, &offset, &size),
             "cuFuncGetParamInfo(0)") ||
      offset != 0 || size != sizeof(CUdeviceptr)) {
    std::fprintf(stderr, "unexpected parameter 0 layout: %zu/%zu\n", offset,
                 size);
    return 1;
  }
  if (!check(cuFuncGetParamInfo(function, 1, &offset, &size),
             "cuFuncGetParamInfo(1)") ||
      offset != sizeof(CUdeviceptr) || size != sizeof(unsigned int)) {
    std::fprintf(stderr, "unexpected parameter 1 layout: %zu/%zu\n", offset,
                 size);
    return 1;
  }
  if (cuFuncGetParamInfo(function, 2, &offset, &size) !=
      CUDA_ERROR_INVALID_VALUE) {
    std::fprintf(stderr, "parameter layout lacked a cached terminator\n");
    return 1;
  }

  CUfunction missing = nullptr;
  CUresult missing_result =
      cuModuleGetFunction(&missing, module, "missing_function");
  if (missing_result != CUDA_ERROR_NOT_FOUND) {
    std::fprintf(stderr, "missing function lookup returned %s (%d)\n",
                 error_name(missing_result), static_cast<int>(missing_result));
    return 1;
  }

  if (!check(cuModuleUnload(module), "cuModuleUnload") ||
      !check(cuDevicePrimaryCtxRelease(device),
             "cuDevicePrimaryCtxRelease")) {
    return 1;
  }
  return 0;
}

static bool read_rpc_count(const char *path, int wanted_op,
                           unsigned long long *count) {
  FILE *file = std::fopen(path, "r");
  if (file == nullptr) {
    return false;
  }
  int op = 0;
  unsigned long long op_count = 0;
  unsigned long long wait_ns = 0;
  *count = 0;
  while (std::fscanf(file, "%d\t%llu\t%llu", &op, &op_count, &wait_ns) == 3) {
    if (op == wanted_op) {
      *count = op_count;
      break;
    }
  }
  std::fclose(file);
  return true;
}

int main() {
  char stats_path[] = "/tmp/lupine_param_layout_stats_XXXXXX";
  int stats_fd = mkstemp(stats_path);
  if (stats_fd < 0) {
    std::perror("mkstemp");
    return 1;
  }
  close(stats_fd);
  if (setenv("LUPINE_RPC_STATS", stats_path, 1) != 0) {
    std::perror("setenv");
    unlink(stats_path);
    return 1;
  }

  pid_t child = fork();
  if (child < 0) {
    std::perror("fork");
    unlink(stats_path);
    return 1;
  }
  if (child == 0) {
    std::exit(exercise_snapshot());
  }

  int child_status = 0;
  if (waitpid(child, &child_status, 0) != child ||
      !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    std::fprintf(stderr, "snapshot exercise failed\n");
    unlink(stats_path);
    return 1;
  }

  unsigned long long lookup_count = 0;
  unsigned long long snapshot_count = 0;
  unsigned long long fallback_count = 0;
  bool read_ok =
      read_rpc_count(stats_path, RPC_cuModuleGetFunction, &lookup_count) &&
      read_rpc_count(stats_path,
                     LUPINE_RPC_lupineFunctionParamLayoutSnapshot,
                     &snapshot_count) &&
      read_rpc_count(stats_path, RPC_cuFuncGetParamInfo, &fallback_count);
  unlink(stats_path);
  if (!read_ok || lookup_count != 2 || snapshot_count != 1 ||
      fallback_count != 0) {
    std::fprintf(stderr,
                 "unexpected RPC counts: lookup=%llu snapshot=%llu "
                 "fallback=%llu\n",
                 lookup_count, snapshot_count, fallback_count);
    return 1;
  }

  std::printf("module lookup and parameter-layout snapshot are isolated; "
              "parameter queries use the warm cache\n");
  return 0;
}
