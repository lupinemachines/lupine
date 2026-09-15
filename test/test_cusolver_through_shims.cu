// cuSOLVER is not shimmed: NVIDIA's libcusolver runs on the client and calls
// the cuBLAS shim underneath, including the undocumented
// cublasSetEnvironmentMode cusolverDnCreate imports from cuBLAS 12.9 on. Solves
// a system by LU and by Cholesky and checks both against the known solution.
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUSOLVER(call)                                                   \
  do {                                                                         \
    cusolverStatus_t status = (call);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
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

static const int n = 3;
// Symmetric positive definite, so it serves both factorizations.
static const double matrix[n * n] = {4, 12, -16, 12, 37, -43, -16, -43, 98};
static const double solution[n] = {1, -2, 3};

static std::vector<double> right_hand_side() {
  std::vector<double> b(n, 0.0);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      b[row] += matrix[col * n + row] * solution[col];
    }
  }
  return b;
}

static int solve(cusolverDnHandle_t handle, bool cholesky) {
  const size_t bytes = sizeof(double) * n * n;
  double *a = nullptr;
  double *b = nullptr;
  int *pivots = nullptr;
  int *info = nullptr;
  double *work = nullptr;
  CHECK_CUDA(cudaMalloc(&a, bytes));
  CHECK_CUDA(cudaMalloc(&b, sizeof(double) * n));
  CHECK_CUDA(cudaMalloc(&pivots, sizeof(int) * n));
  CHECK_CUDA(cudaMalloc(&info, sizeof(int)));
  std::vector<double> rhs = right_hand_side();
  CHECK_CUDA(cudaMemcpy(a, matrix, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(b, rhs.data(), sizeof(double) * n, cudaMemcpyHostToDevice));

  int lwork = 0;
  if (cholesky) {
    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
                                               n, a, n, &lwork));
  } else {
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(handle, n, n, a, n, &lwork));
  }
  CHECK_CUDA(cudaMalloc(&work, sizeof(double) * (lwork > 0 ? lwork : 1)));
  if (cholesky) {
    CHECK_CUSOLVER(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, a, n,
                                    work, lwork, info));
  } else {
    CHECK_CUSOLVER(cusolverDnDgetrf(handle, n, n, a, n, work, pivots, info));
  }
  int factor_info = -1;
  CHECK_CUDA(
      cudaMemcpy(&factor_info, info, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT(factor_info == 0);
  if (cholesky) {
    CHECK_CUSOLVER(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1, a, n,
                                    b, n, info));
  } else {
    CHECK_CUSOLVER(
        cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, a, n, pivots, b, n, info));
  }
  std::vector<double> x(n);
  CHECK_CUDA(
      cudaMemcpy(x.data(), b, sizeof(double) * n, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    EXPECT(std::fabs(x[i] - solution[i]) < 1e-9);
  }
  CHECK_CUDA(cudaFree(work));
  CHECK_CUDA(cudaFree(info));
  CHECK_CUDA(cudaFree(pivots));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(a));
  return 0;
}

int main() {
  cusolverDnHandle_t handle = nullptr;
  CHECK_CUSOLVER(cusolverDnCreate(&handle));
  if (solve(handle, false) != 0 || solve(handle, true) != 0) {
    return 1;
  }
  CHECK_CUSOLVER(cusolverDnDestroy(handle));
  printf("cusolver through shims: getrf and potrf passed\n");
  return 0;
}
