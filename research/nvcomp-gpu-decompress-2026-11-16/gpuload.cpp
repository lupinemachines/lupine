// A second connection's steady GPU workload: compute-bound FMA kernels, each
// ~5 ms alone, synchronised every 4 launches. Prints completed kernels per
// 0.25 s window with CLOCK_REALTIME stamps. gpuload <ptx> <seconds>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static double now() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  std::string ptx;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) ptx.append(buf, n);
  fclose(f);
  double seconds = atof(argv[2]);
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  int sms;
  CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CUmodule mod;
  CK(cuModuleLoadData(&mod, ptx.c_str()));
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, mod, "burn"));
  unsigned blocks = sms * 8, threads = 256;
  CUdeviceptr a;
  CK(cuMemAlloc(&a, size_t(blocks) * threads * 4));
  CK(cuMemsetD8(a, 0, size_t(blocks) * threads * 4));
  int iters = 1000;
  for (int k = 0; k < 6; k++) {
    void *args[] = {&a, &iters};
    double t0 = now();
    CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    CK(cuCtxSynchronize());
    double dt = now() - t0;
    if (k >= 2) iters = int(iters * 0.005 / dt);
  }
  fprintf(stderr, "iters=%d\n", iters);
  void *args[] = {&a, &iters};
  double end = now() + seconds, window = now();
  long done = 0;
  while (now() < end) {
    for (int i = 0; i < 4; i++)
      CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    CK(cuCtxSynchronize());
    done += 4;
    double t = now();
    if (t - window >= 0.25) {
      printf("LOAD t=%.3f kps=%.1f\n", t, done / (t - window));
      fflush(stdout);
      done = 0;
      window = t;
    }
  }
  return 0;
}
