// Exercises the cuBLASLt shim end to end against the remote device: handles,
// descriptors that travel by value, heuristics, a matmul in host and device
// pointer modes with a bias epilogue, and a matrix transform, each checked
// against a CPU reference.
#include <cublasLt.h>
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

int main() {
  cublasLtHandle_t handle = nullptr;
  CHECK_LT(cublasLtCreate(&handle));
  EXPECT(cublasLtGetVersion() > 0);
  EXPECT(strcmp(cublasLtGetStatusName(CUBLAS_STATUS_SUCCESS),
                "CUBLAS_STATUS_SUCCESS") == 0);
  printf("cuBLASLt %zu, runtime %zu\n", cublasLtGetVersion(),
         cublasLtGetCudartVersion());

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
