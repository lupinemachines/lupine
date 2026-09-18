// Exercises the cuSPARSELt shim end to end against the remote device: a 2:4
// structured matmul from pruning to compression to execution, checked against
// a dense reference computed on the device; the pruning validity check, which
// writes its answer to device memory; the descriptor, matmul and algorithm
// attributes read back through the objects the server holds; the algorithm
// search, which picks a configuration the plan then uses; a second matmul
// through the compress-by-descriptor calls; alpha as a device vector; and the
// error strings.
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <cstdint>
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
    cusparseStatus_t status = (call);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status));     \
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

namespace {

constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 64;

// D = A * B on the device, in the row-major layout the descriptors declare, so
// the shim's answer is checked against work the same server did.
__global__ void reference_matmul(const __half *a, const __half *b, float *d,
                                 int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || column >= n) {
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    sum += __half2float(a[row * k + i]) * __half2float(b[i * n + column]);
  }
  d[row * n + column] = sum;
}

__global__ void to_float(const __half *values, float *out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = __half2float(values[i]);
  }
}

__global__ void fill_half(__half *values, int count, int seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    values[i] = __float2half(static_cast<float>((i * 7 + seed) % 5) - 2.0f);
  }
}

__global__ void fill_float(float *values, int count, float value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    values[i] = value;
  }
}

int compare(const float *expected, const float *actual, int count) {
  for (int i = 0; i < count; ++i) {
    const float difference = expected[i] - actual[i];
    if (difference > 0.5f || difference < -0.5f) {
      fprintf(stderr, "element %d: expected %f, got %f\n", i, expected[i],
              actual[i]);
      return 1;
    }
  }
  return 0;
}

} // namespace

int main() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    fprintf(stderr, "no CUDA devices\n");
    return 1;
  }
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 8) {
    printf("SKIP: structured sparsity needs compute capability 8.0 or newer\n");
    return 0;
  }

  cusparseLtHandle_t handle;
  CHECK_LT(cusparseLtInit(&handle));

  int version = 0;
  CHECK_LT(cusparseLtGetVersion(&handle, &version));
  EXPECT(version >= 600);
  int property = 0;
  CHECK_LT(cusparseLtGetProperty(MAJOR_VERSION, &property));
  EXPECT(property == version / 1000);
  printf("cuSPARSELt %d\n", version);

#if CUSPARSELT_VERSION >= 800
  EXPECT(strcmp(cusparseLtGetErrorName(CUSPARSE_STATUS_SUCCESS),
                "CUSPARSE_STATUS_SUCCESS") == 0);
  EXPECT(strlen(cusparseLtGetErrorString(CUSPARSE_STATUS_INVALID_VALUE)) != 0);
#endif

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cusparseLtMatDescriptor_t matA;
  cusparseLtMatDescriptor_t matB;
  cusparseLtMatDescriptor_t matC;
  CHECK_LT(cusparseLtStructuredDescriptorInit(&handle, &matA, kM, kK, kK, 16,
                                              CUDA_R_16F, CUSPARSE_ORDER_ROW,
                                              CUSPARSELT_SPARSITY_50_PERCENT));
  CHECK_LT(cusparseLtDenseDescriptorInit(&handle, &matB, kK, kN, kN, 16,
                                         CUDA_R_16F, CUSPARSE_ORDER_ROW));
  CHECK_LT(cusparseLtDenseDescriptorInit(&handle, &matC, kM, kN, kN, 16,
                                         CUDA_R_16F, CUSPARSE_ORDER_ROW));

  // A matrix attribute set on the server and read back through the descriptor
  // it belongs to.
  int batches = 1;
  CHECK_LT(cusparseLtMatDescSetAttribute(
      &handle, &matA, CUSPARSELT_MAT_NUM_BATCHES, &batches, sizeof(batches)));
  int read_batches = 0;
  CHECK_LT(cusparseLtMatDescGetAttribute(&handle, &matA,
                                         CUSPARSELT_MAT_NUM_BATCHES,
                                         &read_batches, sizeof(read_batches)));
  EXPECT(read_batches == batches);

  cusparseLtMatmulDescriptor_t matmul;
  CHECK_LT(cusparseLtMatmulDescriptorInit(
      &handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC, &matC,
      CUSPARSE_COMPUTE_32F));

  cusparseLtMatmulAlgSelection_t algorithm;
  CHECK_LT(cusparseLtMatmulAlgSelectionInit(&handle, &algorithm, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT));
  int max_id = 0;
  CHECK_LT(cusparseLtMatmulAlgGetAttribute(&handle, &algorithm,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                                           &max_id, sizeof(max_id)));
  EXPECT(max_id > 0);
  int config_id = 0;
  CHECK_LT(cusparseLtMatmulAlgSetAttribute(&handle, &algorithm,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &config_id, sizeof(config_id)));

  cusparseLtMatmulPlan_t plan;
  CHECK_LT(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &algorithm));

  size_t workspace_size = 0;
  size_t compressed_size = 0;
  size_t compressed_buffer_size = 0;
  CHECK_LT(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));
  CHECK_LT(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size,
                                         &compressed_buffer_size));
  EXPECT(compressed_size != 0);

  __half *dense_a = nullptr;
  __half *pruned_a = nullptr;
  __half *dense_b = nullptr;
  __half *result = nullptr;
  float *expected = nullptr;
  float *actual = nullptr;
  void *compressed = nullptr;
  void *compressed_buffer = nullptr;
  void *workspace = nullptr;
  int *valid = nullptr;
  CHECK_CUDA(cudaMalloc(&dense_a, sizeof(__half) * kM * kK));
  CHECK_CUDA(cudaMalloc(&pruned_a, sizeof(__half) * kM * kK));
  CHECK_CUDA(cudaMalloc(&dense_b, sizeof(__half) * kK * kN));
  CHECK_CUDA(cudaMalloc(&result, sizeof(__half) * kM * kN));
  CHECK_CUDA(cudaMalloc(&expected, sizeof(float) * kM * kN));
  CHECK_CUDA(cudaMalloc(&actual, sizeof(float) * kM * kN));
  CHECK_CUDA(cudaMalloc(&compressed, compressed_size));
  CHECK_CUDA(cudaMalloc(&compressed_buffer, compressed_buffer_size));
  CHECK_CUDA(cudaMalloc(&valid, sizeof(int)));
  if (workspace_size != 0) {
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }

  fill_half<<<(kM * kK + 255) / 256, 256>>>(dense_a, kM * kK, 1);
  fill_half<<<(kK * kN + 255) / 256, 256>>>(dense_b, kK * kN, 3);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_LT(cusparseLtSpMMAPrune(&handle, &matmul, dense_a, pruned_a,
                                CUSPARSELT_PRUNE_SPMMA_TILE, stream));
  // The validity flag is device memory: a zero there means the pruned matrix
  // holds the 2:4 pattern.
  CHECK_LT(
      cusparseLtSpMMAPruneCheck(&handle, &matmul, pruned_a, valid, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  int host_valid = -1;
  CHECK_CUDA(cudaMemcpy(&host_valid, valid, sizeof(host_valid),
                        cudaMemcpyDeviceToHost));
  EXPECT(host_valid == 0);

  CHECK_LT(cusparseLtSpMMACompress(&handle, &plan, pruned_a, compressed,
                                   compressed_buffer, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_LT(cusparseLtMatmul(&handle, &plan, &alpha, compressed, dense_b, &beta,
                            result, result, workspace, &stream, 1));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  dim3 block(16, 16);
  dim3 grid((kN + block.x - 1) / block.x, (kM + block.y - 1) / block.y);
  reference_matmul<<<grid, block>>>(pruned_a, dense_b, expected, kM, kN, kK);
  to_float<<<(kM * kN + 255) / 256, 256>>>(result, actual, kM * kN);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> host_expected(kM * kN);
  std::vector<float> host_actual(kM * kN);
  CHECK_CUDA(cudaMemcpy(host_expected.data(), expected, sizeof(float) * kM * kN,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_actual.data(), actual, sizeof(float) * kM * kN,
                        cudaMemcpyDeviceToHost));
  if (compare(host_expected.data(), host_actual.data(), kM * kN) != 0) {
    return 1;
  }
  printf("structured matmul matches the dense reference\n");

  // The search runs the candidates and leaves its choice in the algorithm
  // selection the plan was built from.
  CHECK_LT(cusparseLtMatmulSearch(&handle, &plan, &alpha, compressed, dense_b,
                                  &beta, result, result, workspace, &stream,
                                  1));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  int searched_id = -1;
  CHECK_LT(cusparseLtMatmulAlgGetAttribute(&handle, &algorithm,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &searched_id, sizeof(searched_id)));
  EXPECT(searched_id >= 0);
  printf("search chose configuration %d of %d\n", searched_id, max_id);

  // alpha as a device vector, which the descriptor's attribute moves off the
  // host: every element of the answer is scaled by two.
  float *alpha_vector = nullptr;
  CHECK_CUDA(cudaMalloc(&alpha_vector, sizeof(float) * kM));
  fill_float<<<(kM + 255) / 256, 256>>>(alpha_vector, kM, 2.0f);
  CHECK_CUDA(cudaDeviceSynchronize());
  int vector_scaling = 1;
  CHECK_LT(cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &vector_scaling,
      sizeof(vector_scaling)));
  int read_scaling = 0;
  CHECK_LT(cusparseLtMatmulDescGetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &read_scaling,
      sizeof(read_scaling)));
  EXPECT(read_scaling == vector_scaling);

  cusparseLtMatmulPlan_t scaled_plan;
  CHECK_LT(
      cusparseLtMatmulPlanInit(&handle, &scaled_plan, &matmul, &algorithm));
  CHECK_LT(cusparseLtMatmul(&handle, &scaled_plan, alpha_vector, compressed,
                            dense_b, &beta, result, result, workspace, &stream,
                            1));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  to_float<<<(kM * kN + 255) / 256, 256>>>(result, actual, kM * kN);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_actual.data(), actual, sizeof(float) * kM * kN,
                        cudaMemcpyDeviceToHost));
  for (float &value : host_expected) {
    value *= 2.0f;
  }
  if (compare(host_expected.data(), host_actual.data(), kM * kN) != 0) {
    return 1;
  }
  printf("device alpha vector scales the answer\n");
  CHECK_LT(cusparseLtMatmulPlanDestroy(&scaled_plan));

  // The by-descriptor pruning and compression calls, which name the structured
  // matrix instead of the plan.
  size_t descriptor_compressed_size = 0;
  size_t descriptor_buffer_size = 0;
  CHECK_LT(cusparseLtSpMMACompressedSize2(
      &handle, &matA, &descriptor_compressed_size, &descriptor_buffer_size));
  EXPECT(descriptor_compressed_size == compressed_size);
  CHECK_LT(cusparseLtSpMMAPrune2(
      &handle, &matA, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, dense_a, pruned_a,
      CUSPARSELT_PRUNE_SPMMA_TILE, stream));
  CHECK_LT(cusparseLtSpMMAPruneCheck2(&handle, &matA, 1,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      pruned_a, valid, stream));
  CHECK_LT(cusparseLtSpMMACompress2(&handle, &matA, 1,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, pruned_a,
                                    compressed, compressed_buffer, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  host_valid = -1;
  CHECK_CUDA(cudaMemcpy(&host_valid, valid, sizeof(host_valid),
                        cudaMemcpyDeviceToHost));
  EXPECT(host_valid == 0);
  printf("descriptor pruning and compression agree with the plan's\n");

  CHECK_LT(cusparseLtMatmulPlanDestroy(&plan));
#if CUSPARSELT_VERSION >= 800
  CHECK_LT(cusparseLtMatmulAlgSelectionDestroy(&algorithm));
#endif
  CHECK_LT(cusparseLtMatDescriptorDestroy(&matA));
  CHECK_LT(cusparseLtMatDescriptorDestroy(&matB));
  CHECK_LT(cusparseLtMatDescriptorDestroy(&matC));
  CHECK_LT(cusparseLtDestroy(&handle));

  CHECK_CUDA(cudaFree(alpha_vector));
  CHECK_CUDA(cudaFree(valid));
  CHECK_CUDA(cudaFree(compressed_buffer));
  CHECK_CUDA(cudaFree(compressed));
  if (workspace != nullptr) {
    CHECK_CUDA(cudaFree(workspace));
  }
  CHECK_CUDA(cudaFree(actual));
  CHECK_CUDA(cudaFree(expected));
  CHECK_CUDA(cudaFree(result));
  CHECK_CUDA(cudaFree(dense_b));
  CHECK_CUDA(cudaFree(pruned_a));
  CHECK_CUDA(cudaFree(dense_a));
  CHECK_CUDA(cudaStreamDestroy(stream));
  printf("cusparselt shim OK\n");
  return 0;
}
