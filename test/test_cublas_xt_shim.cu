// Exercises cuBLASXt, whose matrices are host memory: handle and device
// selection, tile size, pinning mode, GEMM in every precision on non-square
// matrices whose leading dimensions exceed their rows, and routines that
// update an argument in place, each checked against a CPU reference.
#include <cublasXt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
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

using zcomplex = std::complex<double>;

static double magnitude(float value) { return std::fabs(value); }
static double magnitude(double value) { return std::fabs(value); }
static double magnitude(const zcomplex &value) { return std::abs(value); }

template <typename T> static bool close_to(const T &value, const T &reference) {
  return magnitude(value - reference) <= 1e-3 * (1.0 + magnitude(reference));
}

static float sample(float, size_t i) {
  return static_cast<float>((i * 37) % 17) / 8.0f - 1.0f;
}
static double sample(double, size_t i) {
  return static_cast<double>((i * 41) % 19) / 9.0 - 1.0;
}
static zcomplex sample(zcomplex, size_t i) {
  return {static_cast<double>((i * 13) % 11) / 5.0 - 1.0,
          static_cast<double>((i * 7) % 5) / 2.0 - 1.0};
}

// Fills an ld x columns array; the rows past the matrix hold a marker that
// must survive the call untouched.
template <typename T>
static std::vector<T> matrix(size_t rows, size_t columns, size_t ld,
                             size_t seed) {
  std::vector<T> values(ld * columns);
  for (size_t j = 0; j < columns; ++j) {
    for (size_t i = 0; i < ld; ++i) {
      values[j * ld + i] =
          i < rows ? sample(T{}, seed + j * ld + i) : T(static_cast<float>(-99));
    }
  }
  return values;
}

template <typename T> static T op(const T &value, cublasOperation_t) {
  return value;
}
template <>
zcomplex op(const zcomplex &value, cublasOperation_t operation) {
  return operation == CUBLAS_OP_C ? std::conj(value) : value;
}

// Element (i, j) of op(A) for an ld-led array.
template <typename T>
static T op_element(const std::vector<T> &a, size_t ld,
                    cublasOperation_t operation, size_t i, size_t j) {
  return operation == CUBLAS_OP_N ? a[j * ld + i] : op(a[i * ld + j], operation);
}

template <typename T>
static std::vector<T> reference_gemm(cublasOperation_t transa,
                                     cublasOperation_t transb, size_t m,
                                     size_t n, size_t k, T alpha,
                                     const std::vector<T> &a, size_t lda,
                                     const std::vector<T> &b, size_t ldb,
                                     T beta, std::vector<T> c, size_t ldc) {
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      T sum{};
      for (size_t p = 0; p < k; ++p) {
        sum += op_element(a, lda, transa, i, p) *
               op_element(b, ldb, transb, p, j);
      }
      c[j * ldc + i] = alpha * sum + beta * c[j * ldc + i];
    }
  }
  return c;
}

template <typename T>
static int check_gemm(cublasXtHandle_t handle, cublasOperation_t transa,
                      cublasOperation_t transb, const char *name,
                      cublasStatus_t (*gemm)(cublasXtHandle_t,
                                             cublasOperation_t,
                                             cublasOperation_t, size_t, size_t,
                                             size_t, const T *, const T *,
                                             size_t, const T *, size_t,
                                             const T *, T *, size_t)) {
  const size_t m = 37, n = 29, k = 23;
  const size_t a_rows = transa == CUBLAS_OP_N ? m : k;
  const size_t a_columns = transa == CUBLAS_OP_N ? k : m;
  const size_t b_rows = transb == CUBLAS_OP_N ? k : n;
  const size_t b_columns = transb == CUBLAS_OP_N ? n : k;
  const size_t lda = a_rows + 4, ldb = b_rows + 2, ldc = m + 3;
  const std::vector<T> a = matrix<T>(a_rows, a_columns, lda, 1);
  const std::vector<T> b = matrix<T>(b_rows, b_columns, ldb, 2);
  std::vector<T> c = matrix<T>(m, n, ldc, 3);
  const T alpha = sample(T{}, 5) + T(static_cast<float>(1.5));
  const T beta = sample(T{}, 6);
  const std::vector<T> want = reference_gemm(transa, transb, m, n, k, alpha, a,
                                             lda, b, ldb, beta, c, ldc);
  CHECK_CUBLAS(gemm(handle, transa, transb, m, n, k, &alpha, a.data(), lda,
                    b.data(), ldb, &beta, c.data(), ldc));
  for (size_t i = 0; i < c.size(); ++i) {
    if (!close_to(c[i], want[i])) {
      fprintf(stderr, "%s mismatch at %zu\n", name, i);
      return 1;
    }
  }
  return 0;
}

int main() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  EXPECT(device_count > 0);

  int max_boards = 0;
  CHECK_CUBLAS(cublasXtMaxBoards(&max_boards));
  EXPECT(max_boards > 0);
  // The library adds the board count to the caller's value.
  int first_device = 0;
  int boards = 5;
  CHECK_CUBLAS(cublasXtGetNumBoards(1, &first_device, &boards));
  EXPECT(boards == 6);

  cublasXtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasXtCreate(&handle));
  CHECK_CUBLAS(cublasXtDeviceSelect(handle, 1, &first_device));

  CHECK_CUBLAS(cublasXtSetBlockDim(handle, 16));
  int block = 0;
  CHECK_CUBLAS(cublasXtGetBlockDim(handle, &block));
  EXPECT(block == 16);

  cublasXtPinnedMemMode_t pinning = CUBLASXT_PINNING_ENABLED;
  CHECK_CUBLAS(cublasXtGetPinningMemMode(handle, &pinning));
  EXPECT(pinning == CUBLASXT_PINNING_DISABLED);
  CHECK_CUBLAS(cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED));
  CHECK_CUBLAS(cublasXtGetPinningMemMode(handle, &pinning));
  EXPECT(pinning == CUBLASXT_PINNING_ENABLED);
  CHECK_CUBLAS(cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_DISABLED));

  CHECK_CUBLAS(cublasXtSetCpuRatio(handle, CUBLASXT_GEMM, CUBLASXT_FLOAT, 0.0f));
  // The native library records the routine; a remote library cannot call a
  // client function, so the shim refuses it.
  const cublasStatus_t routine =
      cublasXtSetCpuRoutine(handle, CUBLASXT_GEMM, CUBLASXT_FLOAT, nullptr);
  EXPECT(routine == CUBLAS_STATUS_SUCCESS ||
         routine == CUBLAS_STATUS_NOT_SUPPORTED);
  printf("cublasXtSetCpuRoutine: %s\n", cublasGetStatusName(routine));

  EXPECT(check_gemm<float>(handle, CUBLAS_OP_N, CUBLAS_OP_T, "Sgemm",
                           cublasXtSgemm) == 0);
  EXPECT(check_gemm<double>(handle, CUBLAS_OP_T, CUBLAS_OP_N, "Dgemm",
                            cublasXtDgemm) == 0);
  EXPECT(check_gemm<zcomplex>(
             handle, CUBLAS_OP_C, CUBLAS_OP_N, "Zgemm",
             [](cublasXtHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                size_t m, size_t n, size_t k, const zcomplex *alpha,
                const zcomplex *a, size_t lda, const zcomplex *b, size_t ldb,
                const zcomplex *beta, zcomplex *c, size_t ldc) {
               return cublasXtZgemm(
                   h, ta, tb, m, n, k,
                   reinterpret_cast<const cuDoubleComplex *>(alpha),
                   reinterpret_cast<const cuDoubleComplex *>(a), lda,
                   reinterpret_cast<const cuDoubleComplex *>(b), ldb,
                   reinterpret_cast<const cuDoubleComplex *>(beta),
                   reinterpret_cast<cuDoubleComplex *>(c), ldc);
             }) == 0);

  // Solve a lower triangular system in place and multiply back.
  {
    const size_t m = 31, n = 19, lda = m + 5, ldb = m + 2;
    std::vector<double> a = matrix<double>(m, m, lda, 7);
    for (size_t i = 0; i < m; ++i) {
      a[i * lda + i] = 4.0 + static_cast<double>(i % 3);
    }
    const std::vector<double> b = matrix<double>(m, n, ldb, 8);
    std::vector<double> x = b;
    const double alpha = 2.0;
    CHECK_CUBLAS(cublasXtDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                               CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                               a.data(), lda, x.data(), ldb));
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < ldb; ++i) {
        if (i >= m) {
          EXPECT(x[j * ldb + i] == b[j * ldb + i]);
          continue;
        }
        double product = 0.0;
        for (size_t p = 0; p <= i; ++p) {
          product += a[p * lda + i] * x[j * ldb + p];
        }
        EXPECT(close_to(product, alpha * b[j * ldb + i]));
      }
    }
  }

  // C = alpha * A * A^T + beta * C on the upper triangle only.
  {
    const size_t n = 27, k = 13, lda = n + 3, ldc = n + 4;
    const std::vector<float> a = matrix<float>(n, k, lda, 9);
    std::vector<float> c = matrix<float>(n, n, ldc, 10);
    const std::vector<float> before = c;
    const float alpha = 0.5f, beta = -1.25f;
    CHECK_CUBLAS(cublasXtSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n,
                               k, &alpha, a.data(), lda, &beta, c.data(), ldc));
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < ldc; ++i) {
        if (i > j) {
          EXPECT(c[j * ldc + i] == before[j * ldc + i]);
          continue;
        }
        double sum = 0.0;
        for (size_t p = 0; p < k; ++p) {
          sum += a[p * lda + i] * a[p * lda + j];
        }
        EXPECT(close_to(c[j * ldc + i],
                        static_cast<float>(alpha * sum +
                                           beta * before[j * ldc + i])));
      }
    }
  }

  CHECK_CUBLAS(cublasXtDestroy(handle));

  if (device_count >= 2) {
    int devices[2] = {0, 1};
    boards = 0;
    CHECK_CUBLAS(cublasXtGetNumBoards(2, devices, &boards));
    EXPECT(boards == 2);
    cublasXtHandle_t pair = nullptr;
    CHECK_CUBLAS(cublasXtCreate(&pair));
    CHECK_CUBLAS(cublasXtDeviceSelect(pair, 2, devices));
    CHECK_CUBLAS(cublasXtSetBlockDim(pair, 8));
    EXPECT(check_gemm<double>(pair, CUBLAS_OP_N, CUBLAS_OP_N, "Dgemm on two",
                              cublasXtDgemm) == 0);
    CHECK_CUBLAS(cublasXtDestroy(pair));
    printf("two devices: %d boards\n", boards);
  }

  printf("test_cublas_xt_shim passed\n");
  return 0;
}
