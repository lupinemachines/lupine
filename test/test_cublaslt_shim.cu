// Exercises the cuBLASLt shim end to end against the remote device: handles,
// descriptors that travel by value, heuristics, a matmul in host and device
// pointer modes with a bias epilogue, and a matrix transform, each checked
// against a CPU reference.
#include <cublasLt.h>
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

#define CHECK_LT(call)                                                         \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "%s failed: %s\n", #call,                                \
              cublasLtGetStatusName(status));                                  \
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

static int compare(const std::vector<float> &got, const std::vector<float> &want,
                   const char *what) {
  for (size_t i = 0; i < want.size(); ++i) {
    if (!close_to(got[i], want[i])) {
      fprintf(stderr, "%s[%zu] = %g, expected %g\n", what, i, got[i], want[i]);
      return 1;
    }
  }
  return 0;
}

// The library writes only what it reports, so bytes of the caller's buffer
// past that keep their contents: the entries past the returned algorithm
// count, the tile list of an algorithm without tiles, and a rejected get.
static int test_unwritten_buffers(cublasLtHandle_t handle) {
  constexpr int kIds = 64;
  constexpr int kUnwritten = -7;
  int ids[kIds];
  for (int &id : ids) {
    id = kUnwritten;
  }
  int count = 0;
  CHECK_LT(cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                                    CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                    CUDA_R_32F, kIds, ids, &count));
  EXPECT(count > 0 && count < kIds && ids[count] == kUnwritten);

  int tileless = 0;
  for (int i = 0; i < count; ++i) {
    cublasLtMatmulAlgo_t algo;
    if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                               CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                               ids[i], &algo) != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    size_t written = 0;
    CHECK_LT(cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &written));
    if (written != 0) {
      continue;
    }
    int tile = kUnwritten;
    CHECK_LT(cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_TILE_IDS, &tile, sizeof(tile), &written));
    EXPECT(written == 0 && tile == kUnwritten);
    ++tileless;
  }

  cublasLtMatmulDescOpaque_t desc;
  CHECK_LT(cublasLtMatmulDescInit(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const uint64_t filled = 0x7f7f7f7f7f7f7f7fULL;
  uint64_t wide = filled;
  if (cublasLtMatmulDescGetAttribute(&desc, CUBLASLT_MATMUL_DESC_TRANSA, &wide,
                                     sizeof(wide), nullptr) !=
      CUBLAS_STATUS_SUCCESS) {
    EXPECT(wide == filled);
  }
  printf("unwritten buffers: passed (%d algorithms without tiles)\n",
         tileless);
  return 0;
}

#if CUBLAS_VERSION >= 130100
// A matmul descriptor names its emulation descriptor by address: the emulated
// DGEMM must see the caller's settings, a get must return the caller's
// address, and a later change to the emulation descriptor must still apply.
static int test_emulated_dgemm(cublasLtHandle_t handle) {
  constexpr int m = 16;
  constexpr int k = 24;
  constexpr int n = 12;
  std::vector<double> a(static_cast<size_t>(m) * k);
  std::vector<double> b(static_cast<size_t>(k) * n);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<double>((i * 5) % 17) / 3.0 - 2.0;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<double>((i * 11) % 7) - 3.0;
  }
  const double alpha = 2.0;
  const double beta = 0.0;
  std::vector<double> reference(static_cast<size_t>(m) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p) {
        sum += a[static_cast<size_t>(p) * m + i] * b[static_cast<size_t>(j) * k + p];
      }
      reference[static_cast<size_t>(j) * m + i] = alpha * sum;
    }
  }

  cublasLtEmulationDescOpaque_t emulation;
  CHECK_LT(cublasLtEmulationDescInit(&emulation));
  const cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_EAGER;
  CHECK_LT(cublasLtEmulationDescSetAttribute(
      &emulation, CUBLASLT_EMULATION_DESC_STRATEGY, &strategy,
      sizeof(strategy)));
  cublasLtMatmulDescOpaque_t operation;
  CHECK_LT(cublasLtMatmulDescInit(
      &operation, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F));
  cublasLtEmulationDesc_t attached = &emulation;
  CHECK_LT(cublasLtMatmulDescSetAttribute(
      &operation, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &attached,
      sizeof(attached)));
  cublasLtEmulationDesc_t readback = nullptr;
  CHECK_LT(cublasLtMatmulDescGetAttribute(
      &operation, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &readback,
      sizeof(readback), nullptr));
  EXPECT(readback == &emulation);
  const int mantissa_bits = 55;
  CHECK_LT(cublasLtEmulationDescSetAttribute(
      &emulation, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
      &mantissa_bits, sizeof(mantissa_bits)));

  cublasLtMatrixLayoutOpaque_t layout_a, layout_b, layout_c;
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_a, CUDA_R_64F, m, k, m));
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_b, CUDA_R_64F, k, n, k));
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_c, CUDA_R_64F, m, n, m));
  constexpr size_t kWorkspace = 256u << 20;
  cublasLtMatmulPreferenceOpaque_t preference;
  CHECK_LT(cublasLtMatmulPreferenceInit(&preference));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
      &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace,
      sizeof(kWorkspace)));
  cublasLtMatmulHeuristicResult_t heuristic = {};
  int found = 0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, &operation, &layout_a,
                                          &layout_b, &layout_c, &layout_c,
                                          &preference, 1, &heuristic, &found));
  EXPECT(found == 1);

  double *device_a = nullptr;
  double *device_b = nullptr;
  double *device_c = nullptr;
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, a.size() * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&device_b, b.size() * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&device_c, reference.size() * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspace));
  CHECK_CUDA(cudaMemcpy(device_a, a.data(), a.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, b.data(), b.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_LT(cublasLtMatmul(handle, &operation, &alpha, device_a, &layout_a,
                          device_b, &layout_b, &beta, device_c, &layout_c,
                          device_c, &layout_c, &heuristic.algo, workspace,
                          kWorkspace, nullptr));
  std::vector<double> c(reference.size());
  CHECK_CUDA(cudaMemcpy(c.data(), device_c, c.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < c.size(); ++i) {
    EXPECT(std::fabs(c[i] - reference[i]) <= 1e-9 * (1.0 + std::fabs(reference[i])));
  }
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(workspace));
  printf("emulated dgemm: passed\n");
  return 0;
}
#endif

// NVIDIA's libraries name one object with two types: a cublasHandle_t is
// accepted wherever a cublasLtHandle_t is, which is how PyTorch reaches
// cuBLASLt (getCurrentCUDABlasLtHandle casts the cuBLAS handle it already
// holds). The two shims are separate objects, so this only holds while they
// agree on the owner; nothing else in this file would notice them diverging.
static int test_cublas_handle_reused_as_lt() {
  cublasHandle_t blas = nullptr;
  if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    return 1;
  }
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t result{};
  int returned = 0;
  CHECK_LT(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_LT(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, 64, 64, 64));
  CHECK_LT(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(
      reinterpret_cast<cublasLtHandle_t>(blas), operation, layout, layout,
      layout, layout, preference, 1, &result, &returned));
  EXPECT(returned == 1);
  CHECK_LT(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_LT(cublasLtMatrixLayoutDestroy(layout));
  CHECK_LT(cublasLtMatmulDescDestroy(operation));
  if (cublasDestroy(blas) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasDestroy failed\n");
    return 1;
  }
  printf("cublas handle reused as cublasLt: passed\n");
  return 0;
}

int main() {
  cublasLtHandle_t handle = nullptr;
  CHECK_LT(cublasLtCreate(&handle));
  EXPECT(cublasLtGetVersion() > 0);
  EXPECT(strcmp(cublasLtGetStatusName(CUBLAS_STATUS_SUCCESS),
                "CUBLAS_STATUS_SUCCESS") == 0);
  printf("cuBLASLt %zu, runtime %zu\n", cublasLtGetVersion(),
         cublasLtGetCudartVersion());
  if (test_unwritten_buffers(handle) != 0) {
    return 1;
  }
  if (test_cublas_handle_reused_as_lt() != 0) {
    return 1;
  }
#if CUBLAS_VERSION >= 130100
  if (test_emulated_dgemm(handle) != 0) {
    return 1;
  }
#endif

  // Column-major D = alpha * A(m x k) * B(k x n) + beta * C + bias.
  constexpr int m = 64;
  constexpr int k = 32;
  constexpr int n = 48;
  std::vector<float> host_a(static_cast<size_t>(m) * k);
  std::vector<float> host_b(static_cast<size_t>(k) * n);
  std::vector<float> host_c(static_cast<size_t>(m) * n);
  std::vector<float> host_bias(m);
  for (size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = static_cast<float>((i * 7) % 11) - 5.0f;
  }
  for (size_t i = 0; i < host_b.size(); ++i) {
    host_b[i] = static_cast<float>((i * 3) % 13) / 4.0f - 1.5f;
  }
  for (size_t i = 0; i < host_c.size(); ++i) {
    host_c[i] = static_cast<float>(i % 5) - 2.0f;
  }
  for (int i = 0; i < m; ++i) {
    host_bias[i] = 0.25f * i;
  }
  const float alpha = 1.5f;
  const float beta = 0.5f;
  std::vector<float> reference(static_cast<size_t>(m) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p) {
        sum += static_cast<double>(host_a[static_cast<size_t>(p) * m + i]) *
               host_b[static_cast<size_t>(j) * k + p];
      }
      reference[static_cast<size_t>(j) * m + i] =
          static_cast<float>(alpha * sum + beta * host_c[j * m + i] +
                             host_bias[i]);
    }
  }

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  float *device_d = nullptr;
  float *device_bias = nullptr;
  float *device_scalars = nullptr;
  void *workspace = nullptr;
  constexpr size_t kWorkspace = 4u << 20;
  CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_d, host_c.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_bias, host_bias.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_scalars, 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspace));
  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_c, host_c.data(), host_c.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_bias, host_bias.data(),
                        host_bias.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  const float scalars[2] = {alpha, beta};
  CHECK_CUDA(cudaMemcpy(device_scalars, scalars, sizeof(scalars),
                        cudaMemcpyHostToDevice));

  // Descriptors: one created by the library, the rest in caller storage.
  cublasLtMatmulDesc_t operation = nullptr;
  CHECK_LT(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  CHECK_LT(cublasLtMatmulDescSetAttribute(operation,
                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue, sizeof(epilogue)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(operation,
                                          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &device_bias, sizeof(device_bias)));
  cublasLtEpilogue_t epilogue_readback = CUBLASLT_EPILOGUE_DEFAULT;
  size_t written = 0;
  CHECK_LT(cublasLtMatmulDescGetAttribute(
      operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_readback,
      sizeof(epilogue_readback), &written));
  EXPECT(epilogue_readback == CUBLASLT_EPILOGUE_BIAS &&
         written == sizeof(epilogue_readback));

  cublasLtMatrixLayoutOpaque_t layout_a, layout_b, layout_c;
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_a, CUDA_R_32F, m, k, m));
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_b, CUDA_R_32F, k, n, k));
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_c, CUDA_R_32F, m, n, m));
  uint64_t rows = 0;
  CHECK_LT(cublasLtMatrixLayoutGetAttribute(&layout_a,
                                            CUBLASLT_MATRIX_LAYOUT_ROWS, &rows,
                                            sizeof(rows), nullptr));
  EXPECT(rows == static_cast<uint64_t>(m));

  cublasLtMatmulPreferenceOpaque_t preference;
  CHECK_LT(cublasLtMatmulPreferenceInit(&preference));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
      &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace,
      sizeof(kWorkspace)));

  cublasLtMatmulHeuristicResult_t heuristics[4] = {};
  int found = 0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, operation, &layout_a,
                                          &layout_b, &layout_c, &layout_c,
                                          &preference, 4, heuristics, &found));
  EXPECT(found > 0 && heuristics[0].state == CUBLAS_STATUS_SUCCESS);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Host pointer mode: alpha and beta travel by value.
  CHECK_LT(cublasLtMatmul(handle, operation, &alpha, device_a, &layout_a,
                          device_b, &layout_b, &beta, device_c, &layout_c,
                          device_d, &layout_c, &heuristics[0].algo, workspace,
                          kWorkspace, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<float> host_d(host_c.size());
  CHECK_CUDA(cudaMemcpy(host_d.data(), device_d, host_d.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  if (compare(host_d, reference, "matmul host scalars") != 0) {
    return 1;
  }

  // Device pointer mode: the descriptor attribute moves alpha and beta to the
  // device, and the shim sends their addresses instead. No algorithm this
  // time, so the library picks one.
  const cublasLtPointerMode_t device_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CHECK_LT(cublasLtMatmulDescSetAttribute(operation,
                                          CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                          &device_mode, sizeof(device_mode)));
  CHECK_CUDA(cudaMemset(device_d, 0, host_d.size() * sizeof(float)));
  CHECK_LT(cublasLtMatmul(handle, operation, device_scalars, device_a,
                          &layout_a, device_b, &layout_b, device_scalars + 1,
                          device_c, &layout_c, device_d, &layout_c, nullptr,
                          workspace, kWorkspace, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(host_d.data(), device_d, host_d.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  if (compare(host_d, reference, "matmul device scalars") != 0) {
    return 1;
  }

  // Transform: D = 2 * A^T, into a k x m layout.
  cublasLtMatrixTransformDesc_t transform = nullptr;
  CHECK_LT(cublasLtMatrixTransformDescCreate(&transform, CUDA_R_32F));
  const cublasOperation_t transpose = CUBLAS_OP_T;
  CHECK_LT(cublasLtMatrixTransformDescSetAttribute(
      transform, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transpose,
      sizeof(transpose)));
  cublasLtMatrixLayoutOpaque_t layout_at;
  CHECK_LT(cublasLtMatrixLayoutInit(&layout_at, CUDA_R_32F, k, m, k));
  const float two = 2.0f;
  const float zero = 0.0f;
  CHECK_LT(cublasLtMatrixTransform(handle, transform, &two, device_a,
                                   &layout_a, &zero, nullptr, nullptr,
                                   device_d, &layout_at, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<float> host_at(static_cast<size_t>(k) * m);
  CHECK_CUDA(cudaMemcpy(host_at.data(), device_d,
                        host_at.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      EXPECT(close_to(host_at[static_cast<size_t>(i) * k + p],
                      2.0 * host_a[static_cast<size_t>(p) * m + i]));
    }
  }

  CHECK_LT(cublasLtMatrixTransformDescDestroy(transform));
  CHECK_LT(cublasLtMatmulDescDestroy(operation));
  CHECK_LT(cublasLtDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(device_d));
  CHECK_CUDA(cudaFree(device_bias));
  CHECK_CUDA(cudaFree(device_scalars));
  CHECK_CUDA(cudaFree(workspace));

  printf("test_cublaslt_shim passed\n");
  return 0;
}
