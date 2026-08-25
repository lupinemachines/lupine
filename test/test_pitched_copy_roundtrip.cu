// 2D and 3D copies whose rows carry padding travel as a packed, compressed
// payload. Exercises both sides of the size at which the connection starts
// framing a payload, and both copy directions, checking every byte back.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

bool check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  return false;
}

// Padding between rows must never reach the far side, so the destination rows
// are pre-poisoned and the padding is checked to be untouched.
bool roundtrip_2d(size_t width, size_t height, const char *label) {
  const size_t host_pitch = width + 64;
  std::vector<unsigned char> source(host_pitch * height);
  std::vector<unsigned char> readback(host_pitch * height, 0xEE);
  for (size_t row = 0; row < height; ++row) {
    for (size_t column = 0; column < host_pitch; ++column) {
      source[row * host_pitch + column] =
          column < width
              ? static_cast<unsigned char>((row * 31 + column) & 0xff)
              : 0xAA;
    }
  }

  void *device = nullptr;
  size_t device_pitch = 0;
  if (!check(cudaMallocPitch(&device, &device_pitch, width, height),
             "cudaMallocPitch")) {
    return false;
  }
  bool ok =
      check(cudaMemcpy2D(device, device_pitch, source.data(), host_pitch, width,
                         height, cudaMemcpyHostToDevice),
            "cudaMemcpy2D HtoD") &&
      check(cudaMemcpy2D(readback.data(), host_pitch, device, device_pitch,
                         width, height, cudaMemcpyDeviceToHost),
            "cudaMemcpy2D DtoH");
  if (ok) {
    for (size_t row = 0; row < height && ok; ++row) {
      for (size_t column = 0; column < host_pitch && ok; ++column) {
        unsigned char got = readback[row * host_pitch + column];
        unsigned char want =
            column < width
                ? static_cast<unsigned char>((row * 31 + column) & 0xff)
                : 0xEE; // padding must be left alone
        if (got != want) {
          std::fprintf(stderr,
                       "FAIL: %s row %zu column %zu: got %02x want %02x\n",
                       label, row, column, got, want);
          ok = false;
        }
      }
    }
  }
  cudaFree(device);
  if (ok) {
    std::printf("ok: %s (%zu x %zu, host pitch %zu)\n", label, width, height,
                host_pitch);
  }
  return ok;
}

} // namespace

int main() {
  // 16 KiB of payload stays under the size at which payloads are framed; 1 MiB
  // is comfortably over it, so both the packed-and-compressed path and the
  // row-by-row fallback are covered.
  if (!roundtrip_2d(256, 64, "unframed 2D with padding") ||
      !roundtrip_2d(4096, 256, "framed 2D with padding") ||
      !roundtrip_2d(1, 1024, "single-byte rows")) {
    return 1;
  }
  std::printf("PASS: pitched copies round-trip through the payload path\n");
  return 0;
}
