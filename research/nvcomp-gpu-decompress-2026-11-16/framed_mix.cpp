// Odd-sized striped HtoD copies mixing compressible and incompressible 4 MiB
// chunks, at offsets inside an allocation, verified by readback.
#include <cuda.h>

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

int main() {
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  const size_t sizes[] = {8u << 20, (8u << 20) + 4097, (33u << 20) + 5,
                          (100u << 20) + 3};
  int failures = 0;
  for (size_t bytes : sizes) {
    for (size_t shift : {size_t{0}, size_t{4099}}) {
      std::vector<unsigned char> src(bytes), back(bytes);
      unsigned x = 12345 + bytes + shift;
      for (size_t i = 0; i < bytes; i++) {
        size_t chunk = i >> 22;
        if (chunk % 3 == 1) {
          x ^= x << 13;
          x ^= x >> 17;
          x ^= x << 5;
          src[i] = x;
        } else if (chunk % 3 == 2) {
          src[i] = (i / 4096) & 0xff;
        } else {
          src[i] = (i % 97 == 0) ? (i >> 7) : 0;
        }
      }
      CUdeviceptr d;
      CK(cuMemAlloc(&d, bytes + shift + 64));
      CK(cuMemsetD8(d, 0xee, bytes + shift + 64));
      CK(cuMemcpyHtoD(d + shift, src.data(), bytes));
      CK(cuMemcpyDtoH(back.data(), d + shift, bytes));
      unsigned char guard[64];
      CK(cuMemcpyDtoH(guard, d + shift + bytes, 64));
      bool ok = memcmp(back.data(), src.data(), bytes) == 0;
      for (unsigned char g : guard)
        ok = ok && g == 0xee;
      printf("bytes=%zu shift=%zu %s\n", bytes, shift, ok ? "ok" : "FAIL");
      failures += !ok;
      CK(cuMemFree(d));
    }
  }
  return failures != 0;
}
