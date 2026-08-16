#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

static const char kPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry set_value(
    .param .u64 output,
    .param .u32 value
)
{
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [value];
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";

static void check(CUresult result, const char *expression, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  (void)cuGetErrorName(result, &name);
  std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expression, line,
               name == nullptr ? "unknown" : name, static_cast<int>(result));
  std::exit(EXIT_FAILURE);
}

#define CHECK(expression) check((expression), #expression, __LINE__)

struct context_state {
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUdeviceptr output = 0;
};

static context_state make_context(CUdevice device) {
  context_state state;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&state.context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&state.context, 0, device));
#endif
  CHECK(cuModuleLoadData(&state.module, kPtx));
  CHECK(cuModuleGetFunction(&state.function, state.module, "set_value"));
  CHECK(cuMemAlloc(&state.output, sizeof(int)));
  return state;
}

static void launch_and_check(const context_state &state, int value,
                             bool extended) {
  CUdeviceptr output = state.output;
  void *params[] = {&output, &value};
#if CUDA_VERSION >= 11080
  if (extended) {
    CUlaunchConfig config = {};
    config.gridDimX = 1;
    config.gridDimY = 1;
    config.gridDimZ = 1;
    config.blockDimX = 1;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    CHECK(cuLaunchKernelEx(&config, state.function, params, nullptr));
  } else
#else
  (void)extended;
#endif
  {
    CHECK(cuLaunchKernel(state.function, 1, 1, 1, 1, 1, 1, 0, nullptr, params,
                         nullptr));
  }

  int observed = 0;
  CHECK(cuMemcpyDtoH(&observed, state.output, sizeof(observed)));
  if (observed != value) {
    std::fprintf(stderr, "unexpected kernel result: got %d want %d\n", observed,
                 value);
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  context_state first = make_context(device);
  context_state second = make_context(device);

  CHECK(cuCtxSetCurrent(first.context));
  launch_and_check(first, 11, false);
  CHECK(cuCtxSetCurrent(second.context));
  launch_and_check(second, 22, true);

  CHECK(cuCtxSetCurrent(first.context));
  CHECK(cuCtxPushCurrent(second.context));
  launch_and_check(second, 33, false);
  CUcontext popped = nullptr;
  CHECK(cuCtxPopCurrent(&popped));
  if (popped != second.context) {
    std::fprintf(stderr, "cuCtxPopCurrent returned the wrong context\n");
    return EXIT_FAILURE;
  }
  launch_and_check(first, 44, true);

  std::thread first_thread([&first]() {
    CHECK(cuCtxSetCurrent(first.context));
    for (int i = 0; i < 8; ++i) {
      launch_and_check(first, 100 + i, (i & 1) != 0);
    }
    CHECK(cuCtxSetCurrent(nullptr));
  });
  std::thread second_thread([&second]() {
    CHECK(cuCtxSetCurrent(second.context));
    for (int i = 0; i < 8; ++i) {
      launch_and_check(second, 200 + i, (i & 1) == 0);
    }
    CHECK(cuCtxSetCurrent(nullptr));
  });
  first_thread.join();
  second_thread.join();

  CHECK(cuCtxSetCurrent(first.context));
  CHECK(cuMemFree(first.output));
  CHECK(cuModuleUnload(first.module));
  CHECK(cuCtxSetCurrent(second.context));
  CHECK(cuMemFree(second.output));
  CHECK(cuModuleUnload(second.module));
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuCtxDestroy(second.context));
  CHECK(cuCtxDestroy(first.context));

  std::printf("PASS: launch lanes preserve their current CUDA contexts\n");
  return EXIT_SUCCESS;
}
