// Exercises the cuBLAS shim end to end against the remote device: handle
// state, scalars in host and device pointer mode, strided host copies, and
// the GEMM entry points PyTorch takes, each checked against a CPU reference.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "%s failed: %s\n", #call, cublasGetStatusName(status));  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static bool close_to(float value, double reference) {
  return std::fabs(value - reference) <= 1e-3 * (1.0 + std::fabs(reference));
}

// Column-major C = A(m x k) * B(k x n) on the host.
static std::vector<float> reference_gemm(const std::vector<float> &a,
                                         const std::vector<float> &b, int m,
                                         int k, int n, int lda, int ldb) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int j = 0; j < n; ++j) {
    for (int p = 0; p < k; ++p) {
      const float b_pj = b[static_cast<size_t>(j) * ldb + p];
      for (int i = 0; i < m; ++i) {
        c[static_cast<size_t>(j) * m + i] +=
            a[static_cast<size_t>(p) * lda + i] * b_pj;
      }
    }
  }
  return c;
}

static int compare(const float *got, const std::vector<float> &want,
                   const char *what) {
  for (size_t i = 0; i < want.size(); ++i) {
    if (!close_to(got[i], want[i])) {
      fprintf(stderr, "%s[%zu] = %g, expected %g\n", what, i, got[i], want[i]);
      return 1;
    }
  }
  return 0;
}

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  int version = 0;
  CHECK_CUBLAS(cublasGetVersion(handle, &version));
  EXPECT(version > 0);
  EXPECT(cublasGetCudartVersion() > 0);
  EXPECT(strcmp(cublasGetStatusName(CUBLAS_STATUS_SUCCESS),
                "CUBLAS_STATUS_SUCCESS") == 0);
  EXPECT(strlen(cublasGetStatusString(CUBLAS_STATUS_INVALID_VALUE)) > 0);
  printf("cuBLAS %d, runtime %zu\n", version, cublasGetCudartVersion());

  cublasPointerMode_t mode;
  CHECK_CUBLAS(cublasGetPointerMode(handle, &mode));
  EXPECT(mode == CUBLAS_POINTER_MODE_HOST);

  // Level 1: a strided host vector in, a dot product back to the host.
  constexpr int kCount = 1024;
  std::vector<float> host_x(2 * kCount, -1000.0f);
  std::vector<float> host_y(kCount);
  double dot_reference = 0.0;
  for (int i = 0; i < kCount; ++i) {
    host_x[2 * i] = 0.5f * i;
    host_y[i] = 1.0f / (i + 1);
    dot_reference += static_cast<double>(host_x[2 * i]) * host_y[i];
  }
  float *x = nullptr;
  float *y = nullptr;
  CHECK_CUDA(cudaMalloc(&x, kCount * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&y, kCount * sizeof(float)));
  CHECK_CUBLAS(cublasSetVector(kCount, sizeof(float), host_x.data(), 2, x, 1));
  CHECK_CUBLAS(cublasSetVector(kCount, sizeof(float), host_y.data(), 1, y, 1));

  float dot = 0.0f;
  CHECK_CUBLAS(cublasSdot(handle, kCount, x, 1, y, 1, &dot));
  EXPECT(close_to(dot, dot_reference));
  int max_index = 0;
  CHECK_CUBLAS(cublasIsamax(handle, kCount, x, 1, &max_index));
  EXPECT(max_index == kCount);

  // Device pointer mode: the scalar arguments and results are device memory.
  float *device_dot = nullptr;
  float *device_alpha = nullptr;
  const float two = 2.0f;
  CHECK_CUDA(cudaMalloc(&device_dot, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_alpha, sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(device_alpha, &two, sizeof(two), cudaMemcpyHostToDevice));
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CHECK_CUBLAS(cublasSdot(handle, kCount, x, 1, y, 1, device_dot));
  CHECK_CUBLAS(cublasSscal(handle, kCount, device_alpha, x, 1));
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  float dot_from_device = 0.0f;
  CHECK_CUDA(cudaMemcpy(&dot_from_device, device_dot, sizeof(float),
                        cudaMemcpyDeviceToHost));
  EXPECT(close_to(dot_from_device, dot_reference));

  // The scaled vector comes back into a strided host buffer.
  std::vector<float> readback(3 * kCount, 7.0f);
  CHECK_CUBLAS(
      cublasGetVector(kCount, sizeof(float), x, 1, readback.data(), 3));
  EXPECT(close_to(readback[3 * (kCount - 1)], 2.0 * host_x[2 * (kCount - 1)]));
  EXPECT(readback[1] == 7.0f && readback[2] == 7.0f);

  // In/out scalars: rotg rewrites a and b and produces c and s.
  float a = 3.0f;
  float b = 4.0f;
  float c = 0.0f;
  float s = 0.0f;
  CHECK_CUBLAS(cublasSrotg(handle, &a, &b, &c, &s));
  EXPECT(close_to(a, 5.0) && close_to(c, 0.6) && close_to(s, 0.8));

  // Level 3: sgemm with padded host leading dimensions on both copies.
  constexpr int m = 64;
  constexpr int k = 32;
  constexpr int n = 48;
  constexpr int host_lda = m + 5;
  constexpr int host_ldc = m + 3;
  std::vector<float> host_a(static_cast<size_t>(host_lda) * k);
  std::vector<float> host_b(static_cast<size_t>(k) * n);
  for (size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = static_cast<float>((i * 7) % 11) - 5.0f;
  }
  for (size_t i = 0; i < host_b.size(); ++i) {
    host_b[i] = static_cast<float>((i * 3) % 13) / 4.0f - 1.5f;
  }
  const std::vector<float> c_reference =
      reference_gemm(host_a, host_b, m, k, n, host_lda, k);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_c, 2 * m * n * sizeof(float)));
  CHECK_CUBLAS(cublasSetMatrix(m, k, sizeof(float), host_a.data(), host_lda,
                               device_a, m));
  CHECK_CUBLAS(
      cublasSetMatrix(k, n, sizeof(float), host_b.data(), k, device_b, k));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           device_a, m, device_b, k, &beta, device_c, m));
  std::vector<float> host_c(static_cast<size_t>(host_ldc) * n, 0.0f);
  CHECK_CUBLAS(cublasGetMatrix(m, n, sizeof(float), device_c, m, host_c.data(),
                               host_ldc));
  for (int j = 0; j < n; ++j) {
    if (compare(host_c.data() + static_cast<size_t>(j) * host_ldc,
                std::vector<float>(c_reference.begin() + j * m,
                                   c_reference.begin() + (j + 1) * m),
                "sgemm column") != 0) {
      return 1;
    }
  }

  // GemmEx sizes its void scalars by the compute type.
  CHECK_CUDA(cudaMemset(device_c, 0, m * n * sizeof(float)));
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                            device_a, CUDA_R_32F, m, device_b, CUDA_R_32F, k,
                            &beta, device_c, CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
  std::vector<float> host_c_packed(static_cast<size_t>(2) * m * n);
  CHECK_CUDA(cudaMemcpy(host_c_packed.data(), device_c, m * n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  if (compare(host_c_packed.data(), c_reference, "gemmEx") != 0) {
    return 1;
  }

  // Batched: two problems sharing A and B, once by stride and once by
  // device arrays of matrix pointers.
  CHECK_CUDA(cudaMemset(device_c, 0, 2 * m * n * sizeof(float)));
  CHECK_CUBLAS(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, device_a, m, 0,
      device_b, k, 0, &beta, device_c, m, static_cast<long long>(m) * n, 2));
  CHECK_CUDA(cudaMemcpy(host_c_packed.data(), device_c,
                        2 * m * n * sizeof(float), cudaMemcpyDeviceToHost));
  if (compare(host_c_packed.data(), c_reference, "strided batch 0") != 0 ||
      compare(host_c_packed.data() + m * n, c_reference, "strided batch 1") !=
          0) {
    return 1;
  }

  const float *a_pointers[2] = {device_a, device_a};
  const float *b_pointers[2] = {device_b, device_b};
  float *c_pointers[2] = {device_c, device_c + m * n};
  const float **device_a_pointers = nullptr;
  const float **device_b_pointers = nullptr;
  float **device_c_pointers = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a_pointers, sizeof(a_pointers)));
  CHECK_CUDA(cudaMalloc(&device_b_pointers, sizeof(b_pointers)));
  CHECK_CUDA(cudaMalloc(&device_c_pointers, sizeof(c_pointers)));
  CHECK_CUDA(cudaMemcpy(device_a_pointers, a_pointers, sizeof(a_pointers),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b_pointers, b_pointers, sizeof(b_pointers),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_c_pointers, c_pointers, sizeof(c_pointers),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(device_c, 0, 2 * m * n * sizeof(float)));
  CHECK_CUBLAS(cublasSgemmBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, device_a_pointers, m,
      device_b_pointers, k, &beta, device_c_pointers, m, 2));
  CHECK_CUDA(cudaMemcpy(host_c_packed.data(), device_c,
                        2 * m * n * sizeof(float), cudaMemcpyDeviceToHost));
  if (compare(host_c_packed.data(), c_reference, "pointer batch 0") != 0 ||
      compare(host_c_packed.data() + m * n, c_reference, "pointer batch 1") !=
          0) {
    return 1;
  }

#if CUBLAS_VERSION >= 120500
  // Grouped: two groups of one problem each, the second scaled by 2, with the
  // per-group host arrays and device pointer arrays the call takes.
  {
    const cublasOperation_t ops[2] = {CUBLAS_OP_N, CUBLAS_OP_N};
    const int ms[2] = {m, m}, ns[2] = {n, n}, ks[2] = {k, k};
    const int ldas[2] = {m, m}, ldbs[2] = {k, k}, ldcs[2] = {m, m};
    const float alphas[2] = {1.0f, 2.0f}, betas[2] = {0.0f, 0.0f};
    const int sizes[2] = {1, 1};
    CHECK_CUDA(cudaMemset(device_c, 0, 2 * m * n * sizeof(float)));
    CHECK_CUBLAS(cublasSgemmGroupedBatched(
        handle, ops, ops, ms, ns, ks, alphas, device_a_pointers, ldas,
        device_b_pointers, ldbs, betas, device_c_pointers, ldcs, 2, sizes));
    CHECK_CUDA(cudaMemcpy(host_c_packed.data(), device_c,
                          2 * m * n * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> doubled(c_reference);
    for (float &value : doubled) {
      value *= 2.0f;
    }
    if (compare(host_c_packed.data(), c_reference, "grouped batch 0") != 0 ||
        compare(host_c_packed.data() + m * n, doubled, "grouped batch 1") !=
            0) {
      return 1;
    }
  }
#endif

  // Streams: the handle's stream round-trips, and async host copies land on
  // it.
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  cudaStream_t handle_stream = nullptr;
  CHECK_CUBLAS(cublasGetStream(handle, &handle_stream));
  EXPECT(handle_stream == stream);
  CHECK_CUBLAS(cublasSetVectorAsync(kCount, sizeof(float), host_y.data(), 1, x,
                                    1, stream));
  CHECK_CUBLAS(cublasSaxpy(handle, kCount, &two, y, 1, x, 1));
  std::vector<float> axpy(kCount, 0.0f);
  CHECK_CUBLAS(cublasGetVectorAsync(kCount, sizeof(float), x, 1, axpy.data(), 1,
                                    stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (int i = 0; i < kCount; ++i) {
    EXPECT(close_to(axpy[i], 3.0 * host_y[i]));
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaFree(device_dot));
  CHECK_CUDA(cudaFree(device_alpha));
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(device_a_pointers));
  CHECK_CUDA(cudaFree(device_b_pointers));
  CHECK_CUDA(cudaFree(device_c_pointers));

  printf("test_cublas_shim passed\n");
  return 0;
}
