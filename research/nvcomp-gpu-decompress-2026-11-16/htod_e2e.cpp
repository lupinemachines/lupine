// End-to-end sync pageable HtoD through Lupine: htod_e2e <file> <reps>
#include <cuda.h>

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  fseek(f, 0, SEEK_END);
  size_t n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> src(n), back(n);
  if (fread(src.data(), 1, n, f) != n)
    return 1;
  fclose(f);
  int reps = atoi(argv[2]);
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  char name[128];
  CK(cuDeviceGetName(name, sizeof name, dev));
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  CUdeviceptr d;
  CK(cuMemAlloc(&d, n));
  CK(cuMemcpyHtoD(d, src.data(), n)); // warm-up: dials bulk lanes
  std::vector<double> r;
  rusage u0, u1;
  getrusage(RUSAGE_SELF, &u0);
  for (int i = 0; i < reps; i++) {
    CK(cuMemsetD8(d, 0xab, n));
    CK(cuCtxSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    CK(cuMemcpyHtoD(d, src.data(), n));
    CK(cuCtxSynchronize());
    double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    r.push_back(n / s / (1 << 30));
  }
  getrusage(RUSAGE_SELF, &u1);
  auto secs = [](const rusage &u) {
    return u.ru_utime.tv_sec + u.ru_stime.tv_sec +
           1e-6 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec);
  };
  double client_cpu =
      reps ? (secs(u1) - secs(u0)) / (reps * (n / double(1 << 30))) : 0;
  CK(cuMemcpyDtoH(back.data(), d, n));
  bool ok = memcmp(back.data(), src.data(), n) == 0;
  std::vector<double> s = r;
  std::sort(s.begin(), s.end());
  printf(
      "E2E dev=\"%s\" file=%s nvcomp=%s gibs_med=%.3f min=%.3f max=%.3f all=",
      name, argv[1],
      getenv("LUPINE_NVCOMP_HTOD") ? getenv("LUPINE_NVCOMP_HTOD") : "off",
      reps ? s[s.size() / 2] : 0, reps ? s.front() : 0, reps ? s.back() : 0);
  for (double x : r)
    printf("%.3f,", x);
  printf(" client_cpu_s_per_gib=%.3f verify=%s\n", client_cpu,
         ok ? "ok" : "FAIL");
  return ok ? 0 : 2;
}
