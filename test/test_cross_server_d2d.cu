// Cross-server device-to-device copy integration test for #255.
//
// This is a MANUAL integration test (it cannot run in the single-GPU CI L4
// environment): start two LUPINE servers on a host that has GPUs, e.g.
//
//   LUPINE_PORT=15001 ./build/lupine_driver_server &
//   LUPINE_PORT=15002 ./build/lupine_driver_server &
//
// then run the client pointing at both. Each server exposes every local GPU,
// so with two servers the client sees [srv1 gpus..., srv2 gpus...]:
//
//   nvcc -arch=sm_89 test/test_cross_server_d2d.cu -o test_cross_server_d2d
//   LUPINE_SERVER=<host>:15001,<host>:15002 \
//     LD_PRELOAD=./build/libcuda.so.1 ./test_cross_server_d2d
//
// It allocates a buffer on server 1's device 0, fills it, allocates a buffer
// on server 2's first device, issues a device->device copy between them, and
// verifies the bytes. With LUPINE_TRACE=1 the log shows the copy going through
// the client-staged path ("cross-route D2D via client") today; the goal of
// #255 is a direct server-to-server path that avoids that hop while keeping
// this test green.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "FAIL %s: %d (%s)\n", what, (int)e,
            cudaGetErrorString(e));
    exit(1);
  }
}

int main() {
  int ndev = 0;
  check(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
  if (ndev < 2) {
    fprintf(stderr, "need >=2 devices across servers; saw %d\n", ndev);
    return 1;
  }

  // The client exposes server 1's devices first, then server 2's. With an even
  // split, the first device on server 2 is at ndev/2.
  int srcdev = 0;
  int dstdev = ndev / 2;
  printf("device count=%d; src=device%d (server 1) -> dst=device%d (server 2)\n",
         ndev, srcdev, dstdev);

  const int N = 1 << 20;  // 4 MiB
  std::vector<int> pat(N);
  std::vector<int> out(N, 0);
  for (int i = 0; i < N; i++) pat[i] = i * 7 + 3;

  int *d_src = nullptr;
  int *d_dst = nullptr;

  check(cudaSetDevice(srcdev), "cudaSetDevice(src)");
  check(cudaMalloc((void **)&d_src, N * sizeof(int)), "cudaMalloc(src)");
  check(cudaMemcpy(d_src, pat.data(), N * sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H->D (src)");

  check(cudaSetDevice(dstdev), "cudaSetDevice(dst)");
  check(cudaMalloc((void **)&d_dst, N * sizeof(int)), "cudaMalloc(dst)");

  // Cross-server device->device copy.
  check(cudaMemcpy(d_dst, d_src, N * sizeof(int),
                   cudaMemcpyDeviceToDevice),
        "cudaMemcpy D->D (cross-server)");

  check(cudaMemcpy(out.data(), d_dst, N * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy D->H (dst)");

  int mism = 0;
  for (int i = 0; i < N; i++) {
    if (out[i] != pat[i]) {
      if (mism < 5)
        printf("mismatch at %d: got %d want %d\n", i, out[i], pat[i]);
      mism++;
    }
  }

  if (mism) {
    printf("FAIL: %d/%d mismatches\n", mism, N);
    return 1;
  }

  printf("PASS: cross-server D2D verified (%d ints)\n", N);
  return 0;
}
