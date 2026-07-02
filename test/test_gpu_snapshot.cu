#include <cuda.h>

#include <dlfcn.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using snapshot_fn = CUresult (*)(const char *);
using disconnect_fn = void (*)();

static const char *kPtx = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_one(
    .param .u64 data,
    .param .u32 count
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [data];
    ld.param.u32 %r1, [count];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE;

    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32 %f1, %f1, 0f3f800000;
    st.global.f32 [%rd3], %f1;

DONE:
    ret;
}
)ptx";

static void check(CUresult result, const char *what) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    fprintf(stderr, "%s failed: %d %s\n", what, (int)result,
            name ? name : "");
    exit(1);
  }
}

template <typename T> static T load_symbol(void *lib, const char *name) {
  void *symbol = dlsym(lib, name);
  if (symbol == nullptr) {
    fprintf(stderr, "missing symbol %s: %s\n", name, dlerror());
    exit(1);
  }
  return reinterpret_cast<T>(symbol);
}

static void maybe_wait_for_server_restart() {
  const char *ready = getenv("LUPINE_SNAPSHOT_RESTART_READY");
  const char *go = getenv("LUPINE_SNAPSHOT_RESTART_GO");
  if (ready == nullptr || go == nullptr) return;

  FILE *f = fopen(ready, "wb");
  if (f != nullptr) fclose(f);
  while (access(go, F_OK) != 0) usleep(10000);
}

static void launch_add_one(CUfunction fn, CUdeviceptr data, int count,
                           CUstream stream) {
  void *args[] = {&data, &count};
  unsigned int blocks = (count + 127) / 128;
  check(cuLaunchKernel(fn, blocks, 1, 1, 128, 1, 1, 0, stream, args, nullptr),
        "cuLaunchKernel");
}

int main() {
  void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
    return 1;
  }
  auto snapshot_save = load_symbol<snapshot_fn>(lib, "lupine_snapshot_save");
  auto snapshot_load = load_symbol<snapshot_fn>(lib, "lupine_snapshot_load");
  auto snapshot_disconnect =
      load_symbol<disconnect_fn>(lib, "lupine_snapshot_disconnect");

  check(cuInit(0), "cuInit");
  CUdevice dev = 0;
  check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx = nullptr;
  check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  CUmodule module = nullptr;
  CUfunction add_one = nullptr;
  check(cuModuleLoadData(&module, kPtx), "cuModuleLoadData");
  check(cuModuleGetFunction(&add_one, module, "add_one"), "cuModuleGetFunction");

  constexpr int count = 256;
  float host[count];
  for (int i = 0; i < count; ++i) host[i] = (float)i;

  check(snapshot_save("gpu-snapshot-regression"), "lupine_snapshot_save");
  snapshot_disconnect();
  maybe_wait_for_server_restart();
  check(snapshot_load("gpu-snapshot-regression"), "lupine_snapshot_load");

  for (int i = 0; i < count; ++i) host[i] = (float)i;
  CUdeviceptr device = 0;
  check(cuMemAlloc(&device, sizeof(host)), "cuMemAlloc after restore");
  check(cuMemcpyHtoD(device, host, sizeof(host)), "cuMemcpyHtoD after restore");

  // Reuse the cached pre-restore function handle. The restored server
  // translates it to the replacement function created before payload restore.
  launch_add_one(add_one, device, count, nullptr);
  check(cuCtxSynchronize(), "cuCtxSynchronize after restore");

  float out[count] = {};
  check(cuMemcpyDtoH(out, device, sizeof(out)), "cuMemcpyDtoH");

  for (int i = 0; i < count; ++i) {
    float expected = (float)i + 1.0f;
    if (fabsf(out[i] - expected) > 0.001f) {
      fprintf(stderr, "mismatch at %d: got %.6f expected %.6f\n", i, out[i],
              expected);
      return 1;
    }
  }
  printf("gpu snapshot regression passed\n");
  return 0;
}
