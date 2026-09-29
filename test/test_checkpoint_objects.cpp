#include "checkpoint_objects.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

static const char program[] =
    ".version 7.0\n.target sm_50\n.address_size 64\n"
    ".visible .entry write_value(.param .u64 output) {"
    ".reg .u64 p; .reg .u32 v; ld.param.u64 p,[output];"
    "mov.u32 v,42; st.global.u32 [p],v; ret; }";

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::fprintf(stderr, "failed: %s at %d\n", #expr, __LINE__);             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  setenv("LUPINE_CHECKPOINT_OBJECTS", "1", 1);
  if (argc == 1) {
    char path[] = "/tmp/lupine-objects.XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    close(fd);
    for (const char *phase : {"save", "restore"}) {
      pid_t pid = fork();
      CHECK(pid >= 0);
      if (pid == 0) {
        execl(argv[0], argv[0], phase, path, nullptr);
        _exit(1);
      }
      int status = 0;
      CHECK(waitpid(pid, &status, 0) == pid);
      if (WIFEXITED(status) && WEXITSTATUS(status) == 77) {
        unlink(path);
        return 77;
      }
      CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    unlink(path);
    return 0;
  }
  CHECK(argc == 3);
  if (cuInit(0) != CUDA_SUCCESS)
    return 77;
  int devices = 0;
  if (cuDeviceGetCount(&devices) != CUDA_SUCCESS || !devices)
    return 77;
  using namespace lupine_objects;
  if (std::strcmp(argv[1], "save") == 0) {
    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, 0) == CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(ctx) == CUDA_SUCCESS);
    context(ctx, 0, 0, true);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, program) == CUDA_SUCCESS);
    module(mod, program, sizeof(program), false);
    CUfunction fn;
    CHECK(cuModuleGetFunction(&fn, mod, "write_value") == CUDA_SUCCESS);
    function(fn, mod, "write_value");
    CUstream s;
    CHECK(cuStreamCreateWithPriority(&s, CU_STREAM_NON_BLOCKING, -1) ==
          CUDA_SUCCESS);
    stream(s, CU_STREAM_NON_BLOCKING, -1);
    CUevent e;
    CHECK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS);
    event(e, CU_EVENT_DISABLE_TIMING);
    CHECK(wire(kind::context, reinterpret_cast<uintptr_t>(ctx)) == 0x10000);
    CHECK(wire(kind::function, reinterpret_cast<uintptr_t>(fn)) == 0x10002);
    CHECK(save(argv[2]) == 0);
    return 0;
  }
  CHECK(restore(argv[2]) == 0);
  auto ctx = reinterpret_cast<CUcontext>(native(kind::context, 0x10000));
  auto fn = reinterpret_cast<CUfunction>(native(kind::function, 0x10002));
  auto s = reinterpret_cast<CUstream>(native(kind::stream, 0x10003));
  auto e = reinterpret_cast<CUevent>(native(kind::event, 0x10004));
  CHECK(cuCtxSetCurrent(ctx) == CUDA_SUCCESS);
  CHECK(wire(kind::function, reinterpret_cast<uintptr_t>(fn)) == 0x10002);
  CHECK(native(kind::stream, 1) == 1 && native(kind::stream, 2) == 2);
  CUdeviceptr output;
  CHECK(cuMemAlloc(&output, sizeof(int)) == CUDA_SUCCESS);
  void *args[] = {&output};
  CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, s, args, nullptr) ==
        CUDA_SUCCESS);
  CHECK(cuEventRecord(e, s) == CUDA_SUCCESS);
  CHECK(cuEventSynchronize(e) == CUDA_SUCCESS);
  int result = 0;
  CHECK(cuMemcpyDtoH(&result, output, sizeof(result)) == CUDA_SUCCESS &&
        result == 42);
  CHECK(cuMemFree(output) == CUDA_SUCCESS);
  CHECK(cuEventDestroy(e) == CUDA_SUCCESS);
  erase(kind::event, reinterpret_cast<uintptr_t>(e));
  CHECK(save(argv[2]) == 0);
  static const char global_image[] =
      ".version 7.0\n.target sm_50\n.address_size 64\n.global .u32 state;\n";
  CUmodule with_global = nullptr;
  CHECK(cuModuleLoadData(&with_global, global_image) == CUDA_SUCCESS);
  module(with_global, global_image, sizeof(global_image), false);
  CHECK(save(argv[2]) != 0);
  return 0;
}
