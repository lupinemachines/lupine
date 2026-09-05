#include <cublas_api.h>

// cuBLAS API. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libcublas
// by name at runtime. A handle routes to the connection that created it, and a
// call without one goes to the runtime's current device.
//
// This file is the list of what the shim supports. Anything cublas_api.h
// declares that is absent below, or declared below without saying what to do
// with its parameters, gets a stub returning CUBLAS_STATUS_NOT_SUPPORTED, so a
// consumer linking the whole library still loads.
//
// A pointer the handle's pointer mode places on the host or on the device
// (alpha, beta, a dot product's result) is `SCALAR`: the client sends the
// value in host mode and the address in device mode. A `void` scalar carries
// its width as `SIZE:<expr>`, evaluated on the client. A batched call's array
// of matrix pointers lives in device memory, so it is declared as a pointer to
// pointers and sent as an address.
//
// The `_64` entry points arrived with cuBLAS 12.0, emulation strategy with
// 12.9 and the fixed-point emulation controls with 13.1, so each carries the
// matching @guard. A grouped batched GEMM's per-group arrays are host memory
// sized by group_count, and its alpha and beta hold one scalar per group.

/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, void *result,
                            cudaDataType resultType,
                            cudaDataType executiontype);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasAsumEx_64(cublasHandle_t handle, int64_t n, const void *x,
                               cudaDataType xType, int64_t incx, void *result,
                               cudaDataType resultType,
                               cudaDataType executiontype);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(alphaType)
 * @param alphaType SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha,
                            cudaDataType alphaType, const void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy,
                            cudaDataType executiontype);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(alphaType)
 * @param alphaType SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasAxpyEx_64(cublasHandle_t handle, int64_t n,
                               const void *alpha, cudaDataType alphaType,
                               const void *x, cudaDataType xType, int64_t incx,
                               void *y, cudaDataType yType, int64_t incy,
                               cudaDataType executiontype);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle, int n,
                              const cuComplex *alpha, const cuComplex *x,
                              int incx, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCaxpy_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuComplex *alpha, const cuComplex *x,
                                 int64_t incx, cuComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x,
                              int incx, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCcopy_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuComplex *x, int64_t incx, cuComplex *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const cuComplex *A, int lda,
                           const cuComplex *x, int incx, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode,
                              int64_t m, int64_t n, const cuComplex *A,
                              int64_t lda, const cuComplex *x, int64_t incx,
                              cuComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasCdotc_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *y, int64_t incy,
                                 cuComplex *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasCdotu_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *y, int64_t incy,
                                 cuComplex *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, int kl, int ku,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *x, int incx,
                              const cuComplex *beta, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, int64_t kl, int64_t ku,
                                 const cuComplex *alpha, const cuComplex *A,
                                 int64_t lda, const cuComplex *x, int64_t incx,
                                 const cuComplex *beta, cuComplex *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *beta, const cuComplex *B, int ldb,
                           cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int64_t m, int64_t n,
                              const cuComplex *alpha, const cuComplex *A,
                              int64_t lda, const cuComplex *beta,
                              const cuComplex *B, int64_t ldb, cuComplex *C,
                              int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info RECV_ONLY
 * @param devInfoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasCgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, cuComplex *const *Aarray, int lda,
                                  cuComplex *const *Carray, int ldc, int *info,
                                  int *devInfoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const cuComplex *alpha, const cuComplex *A,
                             int lda, const cuComplex *B, int ldb,
                             const cuComplex *beta, cuComplex *C, int ldc);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemm3mBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *const *Aarray,
    int lda, const cuComplex *const *Barray, int ldb, const cuComplex *beta,
    cuComplex *const *Carray, int ldc, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemm3mBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const cuComplex *alpha,
    const cuComplex *const *Aarray, int64_t lda, const cuComplex *const *Barray,
    int64_t ldb, const cuComplex *beta, cuComplex *const *Carray, int64_t ldc,
    int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               const cuComplex *alpha, const void *A,
                               cudaDataType Atype, int lda, const void *B,
                               cudaDataType Btype, int ldb,
                               const cuComplex *beta, void *C,
                               cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t
cublasCgemm3mEx_64(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                   const cuComplex *alpha, const void *A, cudaDataType Atype,
                   int64_t lda, const void *B, cudaDataType Btype, int64_t ldb,
                   const cuComplex *beta, void *C, cudaDataType Ctype,
                   int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemm3mStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    long long int strideA, const cuComplex *B, int ldb, long long int strideB,
    const cuComplex *beta, cuComplex *C, int ldc, long long int strideC,
    int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemm3mStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A,
    int64_t lda, long long int strideA, const cuComplex *B, int64_t ldb,
    long long int strideB, const cuComplex *beta, cuComplex *C, int64_t ldc,
    long long int strideC, int64_t batchCount);
#endif
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int64_t m, int64_t n,
                                int64_t k, const cuComplex *alpha,
                                const cuComplex *A, int64_t lda,
                                const cuComplex *B, int64_t ldb,
                                const cuComplex *beta, cuComplex *C,
                                int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *const *Aarray,
    int lda, const cuComplex *const *Barray, int ldb, const cuComplex *beta,
    cuComplex *const *Carray, int ldc, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa,
                      cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                      const cuComplex *alpha, const cuComplex *const *Aarray,
                      int64_t lda, const cuComplex *const *Barray, int64_t ldb,
                      const cuComplex *beta, cuComplex *const *Carray,
                      int64_t ldc, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const cuComplex *alpha, const void *A,
                             cudaDataType Atype, int lda, const void *B,
                             cudaDataType Btype, int ldb, const cuComplex *beta,
                             void *C, cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int64_t m, int64_t n,
                                int64_t k, const cuComplex *alpha,
                                const void *A, cudaDataType Atype, int64_t lda,
                                const void *B, cudaDataType Btype, int64_t ldb,
                                const cuComplex *beta, void *C,
                                cudaDataType Ctype, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    long long int strideA, const cuComplex *B, int ldb, long long int strideB,
    const cuComplex *beta, cuComplex *C, int ldc, long long int strideC,
    int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemmStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A,
    int64_t lda, long long int strideA, const cuComplex *B, int64_t ldb,
    long long int strideB, const cuComplex *beta, cuComplex *C, int64_t ldc,
    long long int strideC, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *B, int ldb,
                              const cuComplex *beta, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t
cublasCgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                  const cuComplex *alpha, const cuComplex *A, int64_t lda,
                  const cuComplex *B, int64_t ldb, const cuComplex *beta,
                  cuComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                   const cuComplex *alpha, const cuComplex *const *Aarray,
                   int lda, const cuComplex *const *xarray, int incx,
                   const cuComplex *beta, cuComplex *const *yarray, int incy,
                   int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemvBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const cuComplex *alpha, const cuComplex *const *Aarray, int64_t lda,
    const cuComplex *const *xarray, int64_t incx, const cuComplex *beta,
    cuComplex *const *yarray, int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemvStridedBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA,
    const cuComplex *x, int incx, long long int stridex, const cuComplex *beta,
    cuComplex *y, int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const cuComplex *alpha, const cuComplex *A, int64_t lda,
    long long int strideA, const cuComplex *x, int64_t incx,
    long long int stridex, const cuComplex *beta, cuComplex *y, int64_t incy,
    long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, const cuComplex *alpha,
                              const cuComplex *A, int lda, const cuComplex *x,
                              int incx, const cuComplex *beta, cuComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, const cuComplex *alpha,
                                 const cuComplex *A, int64_t lda,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *beta, cuComplex *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param TauArray SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   cuComplex *const *Aarray, int lda,
                                   cuComplex *const *TauArray, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n,
                              const cuComplex *alpha, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n,
                                 const cuComplex *alpha, const cuComplex *x,
                                 int64_t incx, const cuComplex *y, int64_t incy,
                                 cuComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n,
                              const cuComplex *alpha, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n,
                                 const cuComplex *alpha, const cuComplex *x,
                                 int64_t incx, const cuComplex *y, int64_t incy,
                                 cuComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n,
                                   cuComplex *const *A, int lda, int *P,
                                   int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n,
                                   const cuComplex *const *A, int lda,
                                   const int *P, cuComplex *const *C, int ldc,
                                   int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const cuComplex *const *Aarray, int lda,
                                   const int *devIpiv, cuComplex *const *Barray,
                                   int ldb, int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, int k, const cuComplex *alpha,
                              const cuComplex *A, int lda, const cuComplex *x,
                              int incx, const cuComplex *beta, cuComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasChbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, int64_t k, const cuComplex *alpha,
                                 const cuComplex *A, int64_t lda,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *beta, cuComplex *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, int m, int n,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *B, int ldb,
                              const cuComplex *beta, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasChemm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int64_t m, int64_t n,
                                 const cuComplex *alpha, const cuComplex *A,
                                 int64_t lda, const cuComplex *B, int64_t ldb,
                                 const cuComplex *beta, cuComplex *C,
                                 int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *x, int incx,
                              const cuComplex *beta, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasChemv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuComplex *alpha,
                                 const cuComplex *A, int64_t lda,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *beta, cuComplex *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *alpha, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCher2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuComplex *alpha,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *y, int64_t incy, cuComplex *A,
                                 int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const cuComplex *alpha, const cuComplex *A,
                               int lda, const cuComplex *B, int ldb,
                               const float *beta, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCher2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const cuComplex *alpha, const cuComplex *A,
                                  int64_t lda, const cuComplex *B, int64_t ldb,
                                  const float *beta, cuComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const float *alpha, const cuComplex *x,
                             int incx, cuComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCher_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const float *alpha,
                                const cuComplex *x, int64_t incx, cuComplex *A,
                                int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const float *alpha, const void *A,
                               cudaDataType Atype, int lda, const float *beta,
                               void *C, cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const float *alpha, const void *A,
                                  cudaDataType Atype, int64_t lda,
                                  const float *beta, void *C,
                                  cudaDataType Ctype, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const float *alpha, const void *A,
                             cudaDataType Atype, int lda, const float *beta,
                             void *C, cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherkEx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                cublasOperation_t trans, int64_t n, int64_t k,
                                const float *alpha, const void *A,
                                cudaDataType Atype, int64_t lda,
                                const float *beta, void *C, cudaDataType Ctype,
                                int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, int n, int k,
                              const float *alpha, const cuComplex *A, int lda,
                              const float *beta, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int64_t n, int64_t k,
                                 const float *alpha, const cuComplex *A,
                                 int64_t lda, const float *beta, cuComplex *C,
                                 int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const float *beta,
                            cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int64_t n, int64_t k,
                               const cuComplex *alpha, const cuComplex *A,
                               int64_t lda, const cuComplex *B, int64_t ldb,
                               const float *beta, cuComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *alpha,
                              const cuComplex *AP, const cuComplex *x, int incx,
                              const cuComplex *beta, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasChpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuComplex *alpha,
                                 const cuComplex *AP, const cuComplex *x,
                                 int64_t incx, const cuComplex *beta,
                                 cuComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *alpha, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasChpr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuComplex *alpha,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *y, int64_t incy,
                                 cuComplex *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const float *alpha, const cuComplex *x,
                             int incx, cuComplex *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasChpr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const float *alpha,
                                const cuComplex *x, int64_t incx,
                                cuComplex *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Ainv SEND_ONLY
 * @param lda_inv SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n,
                                    const cuComplex *const *A, int lda,
                                    cuComplex *const *Ainv, int lda_inv,
                                    int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCopyEx_64(cublasHandle_t handle, int64_t n, const void *x,
                               cudaDataType xType, int64_t incx, void *y,
                               cudaDataType yType, int64_t incy);
#endif
/**
 * @recordowner HANDLE handle
 * @param handle RECV_ONLY
 */
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x,
                             int incx, cuComplex *y, int incy, const float *c,
                             const cuComplex *s);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasCrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex *x,
                                int64_t incx, cuComplex *y, int64_t incy,
                                const float *c, const cuComplex *s);
#endif
/**
 * @param handle SEND_ONLY
 * @param a SEND_RECV SCALAR
 * @param b SEND_RECV SCALAR
 * @param c RECV_ONLY SCALAR
 * @param s RECV_ONLY SCALAR
 */
cublasStatus_t cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, cuComplex *b,
                              float *c, cuComplex *s);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n,
                              const cuComplex *alpha, cuComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCscal_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuComplex *alpha, cuComplex *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x,
                              int incx, cuComplex *y, int incy, const float *c,
                              const float *s);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasCsrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex *x,
                                 int64_t incx, cuComplex *y, int64_t incy,
                                 const float *c, const float *s);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCsscal_v2(cublasHandle_t handle, int n, const float *alpha,
                               cuComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCsscal_v2_64(cublasHandle_t handle, int64_t n,
                                  const float *alpha, cuComplex *x,
                                  int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x,
                              int incx, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCswap_v2_64(cublasHandle_t handle, int64_t n, cuComplex *x,
                                 int64_t incx, cuComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, int m, int n,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *B, int ldb,
                              const cuComplex *beta, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int64_t m, int64_t n,
                                 const cuComplex *alpha, const cuComplex *A,
                                 int64_t lda, const cuComplex *B, int64_t ldb,
                                 const cuComplex *beta, cuComplex *C,
                                 int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *x, int incx,
                              const cuComplex *beta, cuComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasCsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuComplex *alpha,
                                 const cuComplex *A, int64_t lda,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *beta, cuComplex *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuComplex *alpha, const cuComplex *x,
                              int incx, const cuComplex *y, int incy,
                              cuComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuComplex *alpha,
                                 const cuComplex *x, int64_t incx,
                                 const cuComplex *y, int64_t incy, cuComplex *A,
                                 int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const cuComplex *alpha, const cuComplex *A,
                               int lda, const cuComplex *B, int ldb,
                               const cuComplex *beta, cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const cuComplex *alpha, const cuComplex *A,
                                  int64_t lda, const cuComplex *B, int64_t ldb,
                                  const cuComplex *beta, cuComplex *C,
                                  int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const cuComplex *alpha, const cuComplex *x,
                             int incx, cuComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const cuComplex *alpha,
                                const cuComplex *x, int64_t incx, cuComplex *A,
                                int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const cuComplex *alpha, const void *A,
                               cudaDataType Atype, int lda,
                               const cuComplex *beta, void *C,
                               cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const cuComplex *alpha, const void *A,
                                  cudaDataType Atype, int64_t lda,
                                  const cuComplex *beta, void *C,
                                  cudaDataType Ctype, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const cuComplex *alpha, const void *A,
                             cudaDataType Atype, int lda, const cuComplex *beta,
                             void *C, cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                cublasOperation_t trans, int64_t n, int64_t k,
                                const cuComplex *alpha, const void *A,
                                cudaDataType Atype, int64_t lda,
                                const cuComplex *beta, void *C,
                                cudaDataType Ctype, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, int n, int k,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *beta, cuComplex *C,
                              int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int64_t n, int64_t k,
                                 const cuComplex *alpha, const cuComplex *A,
                                 int64_t lda, const cuComplex *beta,
                                 cuComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const cuComplex *beta,
                            cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int64_t n, int64_t k,
                               const cuComplex *alpha, const cuComplex *A,
                               int64_t lda, const cuComplex *B, int64_t ldb,
                               const cuComplex *beta, cuComplex *C,
                               int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const cuComplex *A, int lda,
                              cuComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const cuComplex *A,
                                 int64_t lda, cuComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const cuComplex *A, int lda,
                              cuComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const cuComplex *A,
                                 int64_t lda, cuComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuComplex *AP, cuComplex *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuComplex *AP, cuComplex *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuComplex *AP, cuComplex *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuComplex *AP, cuComplex *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuComplex *AP, cuComplex *A, int lda);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, const cuComplex *B, int ldb,
                              cuComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasCtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const cuComplex *alpha, const cuComplex *A,
                                 int64_t lda, const cuComplex *B, int64_t ldb,
                                 cuComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuComplex *A, int lda, cuComplex *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuComplex *A, int64_t lda,
                                 cuComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const cuComplex *alpha,
                                  const cuComplex *const *A, int lda,
                                  cuComplex *const *B, int ldb, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasCtrsmBatched_64(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n,
    const cuComplex *alpha, const cuComplex *const *A, int64_t lda,
    cuComplex *const *B, int64_t ldb, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const cuComplex *alpha, const cuComplex *A,
                              int lda, cuComplex *B, int ldb);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasCtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const cuComplex *alpha, const cuComplex *A,
                                 int64_t lda, cuComplex *B, int64_t ldb);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuComplex *A, int lda, cuComplex *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasCtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuComplex *A, int64_t lda,
                                 cuComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuComplex *A, int lda, cuComplex *AP);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x,
                              int incx, double *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDasum_v2_64(cublasHandle_t handle, int64_t n,
                                 const double *x, int64_t incx, double *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha,
                              const double *x, int incx, double *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDaxpy_v2_64(cublasHandle_t handle, int64_t n,
                                 const double *alpha, const double *x,
                                 int64_t incx, double *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x,
                              int incx, double *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDcopy_v2_64(cublasHandle_t handle, int64_t n,
                                 const double *x, int64_t incx, double *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const double *A, int lda, const double *x,
                           int incx, double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode,
                              int64_t m, int64_t n, const double *A,
                              int64_t lda, const double *x, int64_t incx,
                              double *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x,
                             int incx, const double *y, int incy,
                             double *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDdot_v2_64(cublasHandle_t handle, int64_t n,
                                const double *x, int64_t incx, const double *y,
                                int64_t incy, double *result);
#endif
/**
 * @param handle SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, int kl, int ku, const double *alpha,
                              const double *A, int lda, const double *x,
                              int incx, const double *beta, double *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, int64_t kl, int64_t ku,
                                 const double *alpha, const double *A,
                                 int64_t lda, const double *x, int64_t incx,
                                 const double *beta, double *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *beta, const double *B, int ldb,
                           double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int64_t m, int64_t n,
                              const double *alpha, const double *A, int64_t lda,
                              const double *beta, const double *B, int64_t ldb,
                              double *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info RECV_ONLY
 * @param devInfoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasDgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, double *const *Aarray, int lda,
                                  double *const *Carray, int ldc, int *info,
                                  int *devInfoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const double *alpha, const double *const *Aarray, int lda,
                   const double *const *Barray, int ldb, const double *beta,
                   double *const *Carray, int ldc, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasDgemmBatched_64(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb, int64_t m,
                                     int64_t n, int64_t k, const double *alpha,
                                     const double *const *Aarray, int64_t lda,
                                     const double *const *Barray, int64_t ldb,
                                     const double *beta, double *const *Carray,
                                     int64_t ldc, int64_t batchCount);
#endif
#if CUBLAS_VERSION >= 120500
/**
 * @guard CUBLAS_VERSION >= 120500
 * @param handle SEND_ONLY
 * @param transa_array SEND_ONLY LENGTH:group_count
 * @param transb_array SEND_ONLY LENGTH:group_count
 * @param m_array SEND_ONLY LENGTH:group_count
 * @param n_array SEND_ONLY LENGTH:group_count
 * @param k_array SEND_ONLY LENGTH:group_count
 * @param alpha_array SEND_ONLY SCALAR SIZE:group_count*sizeof(double)
 * @param Aarray SEND_ONLY
 * @param lda_array SEND_ONLY LENGTH:group_count
 * @param Barray SEND_ONLY
 * @param ldb_array SEND_ONLY LENGTH:group_count
 * @param beta_array SEND_ONLY SCALAR SIZE:group_count*sizeof(double)
 * @param Carray SEND_ONLY
 * @param ldc_array SEND_ONLY LENGTH:group_count
 * @param group_count SEND_ONLY
 * @param group_size SEND_ONLY LENGTH:group_count
 */
cublasStatus_t cublasDgemmGroupedBatched(
    cublasHandle_t handle, const cublasOperation_t *transa_array,
    const cublasOperation_t *transb_array, const int *m_array,
    const int *n_array, const int *k_array, const double *alpha_array,
    const double *const *Aarray, const int *lda_array,
    const double *const *Barray, const int *ldb_array, const double *beta_array,
    double *const *Carray, const int *ldc_array, int group_count,
    const int *group_size);
#endif
#if CUBLAS_VERSION >= 120500
/**
 * @guard CUBLAS_VERSION >= 120500
 * @param handle SEND_ONLY
 * @param transa_array SEND_ONLY LENGTH:group_count
 * @param transb_array SEND_ONLY LENGTH:group_count
 * @param m_array SEND_ONLY LENGTH:group_count
 * @param n_array SEND_ONLY LENGTH:group_count
 * @param k_array SEND_ONLY LENGTH:group_count
 * @param alpha_array SEND_ONLY SCALAR SIZE:group_count*sizeof(double)
 * @param Aarray SEND_ONLY
 * @param lda_array SEND_ONLY LENGTH:group_count
 * @param Barray SEND_ONLY
 * @param ldb_array SEND_ONLY LENGTH:group_count
 * @param beta_array SEND_ONLY SCALAR SIZE:group_count*sizeof(double)
 * @param Carray SEND_ONLY
 * @param ldc_array SEND_ONLY LENGTH:group_count
 * @param group_count SEND_ONLY
 * @param group_size SEND_ONLY LENGTH:group_count
 */
cublasStatus_t cublasDgemmGroupedBatched_64(
    cublasHandle_t handle, const cublasOperation_t *transa_array,
    const cublasOperation_t *transb_array, const int64_t *m_array,
    const int64_t *n_array, const int64_t *k_array, const double *alpha_array,
    const double *const *Aarray, const int64_t *lda_array,
    const double *const *Barray, const int64_t *ldb_array,
    const double *beta_array, double *const *Carray, const int64_t *ldc_array,
    int64_t group_count, const int64_t *group_size);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const double *alpha, const double *A, int lda,
                          long long int strideA, const double *B, int ldb,
                          long long int strideB, const double *beta, double *C,
                          int ldc, long long int strideC, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasDgemmStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const double *alpha, const double *A,
    int64_t lda, long long int strideA, const double *B, int64_t ldb,
    long long int strideB, const double *beta, double *C, int64_t ldc,
    long long int strideC, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta,
                              double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDgemm_v2_64(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int64_t m, int64_t n,
                                 int64_t k, const double *alpha,
                                 const double *A, int64_t lda, const double *B,
                                 int64_t ldb, const double *beta, double *C,
                                 int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                   const double *alpha, const double *const *Aarray, int lda,
                   const double *const *xarray, int incx, const double *beta,
                   double *const *yarray, int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasDgemvBatched_64(cublasHandle_t handle,
                                     cublasOperation_t trans, int64_t m,
                                     int64_t n, const double *alpha,
                                     const double *const *Aarray, int64_t lda,
                                     const double *const *xarray, int64_t incx,
                                     const double *beta, double *const *yarray,
                                     int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m,
                          int n, const double *alpha, const double *A, int lda,
                          long long int strideA, const double *x, int incx,
                          long long int stridex, const double *beta, double *y,
                          int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasDgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const double *alpha, const double *A, int64_t lda, long long int strideA,
    const double *x, int64_t incx, long long int stridex, const double *beta,
    double *y, int64_t incy, long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, const double *alpha,
                              const double *A, int lda, const double *x,
                              int incx, const double *beta, double *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, const double *alpha,
                                 const double *A, int64_t lda, const double *x,
                                 int64_t incx, const double *beta, double *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param TauArray SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   double *const *Aarray, int lda,
                                   double *const *TauArray, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n,
                             const double *alpha, const double *x, int incx,
                             const double *y, int incy, double *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDger_v2_64(cublasHandle_t handle, int64_t m, int64_t n,
                                const double *alpha, const double *x,
                                int64_t incx, const double *y, int64_t incy,
                                double *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n,
                                   double *const *A, int lda, int *P, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n,
                                   const double *const *A, int lda,
                                   const int *P, double *const *C, int ldc,
                                   int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const double *const *Aarray, int lda,
                                   const int *devIpiv, double *const *Barray,
                                   int ldb, int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Ainv SEND_ONLY
 * @param lda_inv SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n,
                                    const double *const *A, int lda,
                                    double *const *Ainv, int lda_inv, int *info,
                                    int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x,
                              int incx, double *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t handle, int64_t n,
                                 const double *x, int64_t incx, double *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x,
                           cudaDataType xType, int incx, const void *y,
                           cudaDataType yType, int incy, void *result,
                           cudaDataType resultType, cudaDataType executionType);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasDotEx_64(cublasHandle_t handle, int64_t n, const void *x,
                              cudaDataType xType, int64_t incx, const void *y,
                              cudaDataType yType, int64_t incy, void *result,
                              cudaDataType resultType,
                              cudaDataType executionType);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, const void *y,
                            cudaDataType yType, int incy, void *result,
                            cudaDataType resultType,
                            cudaDataType executionType);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void *x,
                               cudaDataType xType, int64_t incx, const void *y,
                               cudaDataType yType, int64_t incy, void *result,
                               cudaDataType resultType,
                               cudaDataType executionType);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx,
                             double *y, int incy, const double *c,
                             const double *s);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasDrot_v2_64(cublasHandle_t handle, int64_t n, double *x,
                                int64_t incx, double *y, int64_t incy,
                                const double *c, const double *s);
#endif
/**
 * @param handle SEND_ONLY
 * @param a SEND_RECV SCALAR
 * @param b SEND_RECV SCALAR
 * @param c RECV_ONLY SCALAR
 * @param s RECV_ONLY SCALAR
 */
cublasStatus_t cublasDrotg_v2(cublasHandle_t handle, double *a, double *b,
                              double *c, double *s);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param param SEND_ONLY SCALAR SIZE:5*sizeof(double)
 */
cublasStatus_t cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx,
                              double *y, int incy, const double *param);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param param SEND_ONLY SCALAR SIZE:5*sizeof(double)
 */
cublasStatus_t cublasDrotm_v2_64(cublasHandle_t handle, int64_t n, double *x,
                                 int64_t incx, double *y, int64_t incy,
                                 const double *param);
#endif
/**
 * @param handle SEND_ONLY
 * @param d1 SEND_RECV SCALAR
 * @param d2 SEND_RECV SCALAR
 * @param x1 SEND_RECV SCALAR
 * @param y1 SEND_ONLY SCALAR
 * @param param RECV_ONLY SCALAR SIZE:5*sizeof(double)
 */
cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle, double *d1, double *d2,
                               double *x1, const double *y1, double *param);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, int k, const double *alpha,
                              const double *A, int lda, const double *x,
                              int incx, const double *beta, double *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, int64_t k, const double *alpha,
                                 const double *A, int64_t lda, const double *x,
                                 int64_t incx, const double *beta, double *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha,
                              double *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDscal_v2_64(cublasHandle_t handle, int64_t n,
                                 const double *alpha, double *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha, const double *AP,
                              const double *x, int incx, const double *beta,
                              double *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const double *alpha,
                                 const double *AP, const double *x,
                                 int64_t incx, const double *beta, double *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha, const double *x,
                              int incx, const double *y, int incy, double *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasDspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const double *alpha,
                                 const double *x, int64_t incx, const double *y,
                                 int64_t incy, double *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const double *alpha, const double *x,
                             int incx, double *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasDspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const double *alpha, const double *x,
                                int64_t incx, double *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx,
                              double *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDswap_v2_64(cublasHandle_t handle, int64_t n, double *x,
                                 int64_t incx, double *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, int m, int n,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, const double *beta,
                              double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int64_t m, int64_t n,
                                 const double *alpha, const double *A,
                                 int64_t lda, const double *B, int64_t ldb,
                                 const double *beta, double *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha, const double *A,
                              int lda, const double *x, int incx,
                              const double *beta, double *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasDsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const double *alpha,
                                 const double *A, int64_t lda, const double *x,
                                 int64_t incx, const double *beta, double *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const double *alpha, const double *x,
                              int incx, const double *y, int incy, double *A,
                              int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const double *alpha,
                                 const double *x, int64_t incx, const double *y,
                                 int64_t incy, double *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const double *alpha, const double *A, int lda,
                               const double *B, int ldb, const double *beta,
                               double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const double *alpha, const double *A,
                                  int64_t lda, const double *B, int64_t ldb,
                                  const double *beta, double *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const double *alpha, const double *x,
                             int incx, double *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const double *alpha, const double *x,
                                int64_t incx, double *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, int n, int k,
                              const double *alpha, const double *A, int lda,
                              const double *beta, double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int64_t n, int64_t k,
                                 const double *alpha, const double *A,
                                 int64_t lda, const double *beta, double *C,
                                 int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta,
                            double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int64_t n, int64_t k,
                               const double *alpha, const double *A,
                               int64_t lda, const double *B, int64_t ldb,
                               const double *beta, double *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const double *A, int lda, double *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const double *A,
                                 int64_t lda, double *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const double *A, int lda, double *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const double *A,
                                 int64_t lda, double *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const double *AP, double *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const double *AP, double *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const double *AP, double *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const double *AP, double *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const double *AP, double *A, int lda);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const double *alpha, const double *A, int lda,
                              const double *B, int ldb, double *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasDtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const double *alpha, const double *A,
                                 int64_t lda, const double *B, int64_t ldb,
                                 double *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const double *A, int lda, double *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const double *A, int64_t lda,
                                 double *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const double *alpha, const double *const *A,
                                  int lda, double *const *B, int ldb,
                                  int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side,
                      cublasFillMode_t uplo, cublasOperation_t trans,
                      cublasDiagType_t diag, int64_t m, int64_t n,
                      const double *alpha, const double *const *A, int64_t lda,
                      double *const *B, int64_t ldb, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const double *alpha, const double *A, int lda,
                              double *B, int ldb);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasDtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const double *alpha, const double *A,
                                 int64_t lda, double *B, int64_t ldb);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const double *A, int lda, double *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasDtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const double *A, int64_t lda,
                                 double *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const double *A, int lda, double *AP);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDzasum_v2(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx,
                               double *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDzasum_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuDoubleComplex *x, int64_t incx,
                                  double *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx,
                               double *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasDznrm2_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuDoubleComplex *x, int64_t incx,
                                  double *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param Aarray SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param Carray SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param computeType SEND_ONLY
 * @param algo SEND_ONLY
 */
cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *const *Aarray,
    cudaDataType Atype, int lda, const void *const *Barray, cudaDataType Btype,
    int ldb, const void *beta, void *const *Carray, cudaDataType Ctype, int ldc,
    int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param Aarray SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param Carray SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param computeType SEND_ONLY
 * @param algo SEND_ONLY
 */
cublasStatus_t cublasGemmBatchedEx_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const void *alpha,
    const void *const *Aarray, cudaDataType Atype, int64_t lda,
    const void *const *Barray, cudaDataType Btype, int64_t ldb,
    const void *beta, void *const *Carray, cudaDataType Ctype, int64_t ldc,
    int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 * @param computeType SEND_ONLY
 * @param algo SEND_ONLY
 */
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A,
                            cudaDataType Atype, int lda, const void *B,
                            cudaDataType Btype, int ldb, const void *beta,
                            void *C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 * @param computeType SEND_ONLY
 * @param algo SEND_ONLY
 */
cublasStatus_t cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int64_t m, int64_t n,
                               int64_t k, const void *alpha, const void *A,
                               cudaDataType Atype, int64_t lda, const void *B,
                               cudaDataType Btype, int64_t ldb,
                               const void *beta, void *C, cudaDataType Ctype,
                               int64_t ldc, cublasComputeType_t computeType,
                               cublasGemmAlgo_t algo);
#endif
#if CUBLAS_VERSION >= 120500
// clang-format off
/**
 * @guard CUBLAS_VERSION >= 120500
 * @param handle SEND_ONLY
 * @param transa_array SEND_ONLY LENGTH:group_count
 * @param transb_array SEND_ONLY LENGTH:group_count
 * @param m_array SEND_ONLY LENGTH:group_count
 * @param n_array SEND_ONLY LENGTH:group_count
 * @param k_array SEND_ONLY LENGTH:group_count
 * @param alpha_array SEND_ONLY SCALAR SIZE:group_count*compute_scalar_width(computeType,Ctype)
 * @param Aarray SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda_array SEND_ONLY LENGTH:group_count
 * @param Barray SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb_array SEND_ONLY LENGTH:group_count
 * @param beta_array SEND_ONLY SCALAR SIZE:group_count*compute_scalar_width(computeType,Ctype)
 * @param Carray SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc_array SEND_ONLY LENGTH:group_count
 * @param group_count SEND_ONLY
 * @param group_size SEND_ONLY LENGTH:group_count
 * @param computeType SEND_ONLY
 */
// clang-format on
cublasStatus_t cublasGemmGroupedBatchedEx(
    cublasHandle_t handle, const cublasOperation_t *transa_array,
    const cublasOperation_t *transb_array, const int *m_array,
    const int *n_array, const int *k_array, const void *alpha_array,
    const void *const *Aarray, cudaDataType_t Atype, const int *lda_array,
    const void *const *Barray, cudaDataType_t Btype, const int *ldb_array,
    const void *beta_array, void *const *Carray, cudaDataType_t Ctype,
    const int *ldc_array, int group_count, const int *group_size,
    cublasComputeType_t computeType);
#endif
#if CUBLAS_VERSION >= 120500
// clang-format off
/**
 * @guard CUBLAS_VERSION >= 120500
 * @param handle SEND_ONLY
 * @param transa_array SEND_ONLY LENGTH:group_count
 * @param transb_array SEND_ONLY LENGTH:group_count
 * @param m_array SEND_ONLY LENGTH:group_count
 * @param n_array SEND_ONLY LENGTH:group_count
 * @param k_array SEND_ONLY LENGTH:group_count
 * @param alpha_array SEND_ONLY SCALAR SIZE:group_count*compute_scalar_width(computeType,Ctype)
 * @param Aarray SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda_array SEND_ONLY LENGTH:group_count
 * @param Barray SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb_array SEND_ONLY LENGTH:group_count
 * @param beta_array SEND_ONLY SCALAR SIZE:group_count*compute_scalar_width(computeType,Ctype)
 * @param Carray SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc_array SEND_ONLY LENGTH:group_count
 * @param group_count SEND_ONLY
 * @param group_size SEND_ONLY LENGTH:group_count
 * @param computeType SEND_ONLY
 */
// clang-format on
cublasStatus_t cublasGemmGroupedBatchedEx_64(
    cublasHandle_t handle, const cublasOperation_t *transa_array,
    const cublasOperation_t *transb_array, const int64_t *m_array,
    const int64_t *n_array, const int64_t *k_array, const void *alpha_array,
    const void *const *Aarray, cudaDataType_t Atype, const int64_t *lda_array,
    const void *const *Barray, cudaDataType_t Btype, const int64_t *ldb_array,
    const void *beta_array, void *const *Carray, cudaDataType_t Ctype,
    const int64_t *ldc_array, int64_t group_count, const int64_t *group_size,
    cublasComputeType_t computeType);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param computeType SEND_ONLY
 * @param algo SEND_ONLY
 */
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
    int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb,
    long long int strideB, const void *beta, void *C, cudaDataType Ctype,
    int ldc, long long int strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:compute_scalar_width(computeType,Ctype)
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param computeType SEND_ONLY
 * @param algo SEND_ONLY
 */
cublasStatus_t cublasGemmStridedBatchedEx_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const void *alpha, const void *A,
    cudaDataType Atype, int64_t lda, long long int strideA, const void *B,
    cudaDataType Btype, int64_t ldb, long long int strideB, const void *beta,
    void *C, cudaDataType Ctype, int64_t ldc, long long int strideC,
    int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode RECV_ONLY
 */
cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle,
                                    cublasAtomicsMode_t *mode);
size_t cublasGetCudartVersion();
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mask RECV_ONLY
 */
cublasStatus_t
cublasGetEmulationSpecialValuesSupport(cublasHandle_t handle,
                                       cudaEmulationSpecialValuesSupport *mask);
#endif
#if CUBLAS_VERSION >= 120900
/**
 * @guard CUBLAS_VERSION >= 120900
 * @param handle SEND_ONLY
 * @param emulationStrategy RECV_ONLY
 */
cublasStatus_t
cublasGetEmulationStrategy(cublasHandle_t handle,
                           cublasEmulationStrategy_t *emulationStrategy);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mantissaBitCount RECV_ONLY
 */
cublasStatus_t
cublasGetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t handle,
                                                    int **mantissaBitCount);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mantissaBitOffset RECV_ONLY
 */
cublasStatus_t
cublasGetFixedPointEmulationMantissaBitOffset(cublasHandle_t handle,
                                              int *mantissaBitOffset);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mantissaControl RECV_ONLY
 */
cublasStatus_t cublasGetFixedPointEmulationMantissaControl(
    cublasHandle_t handle, cudaEmulationMantissaControl *mantissaControl);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param maxMantissaBitCount RECV_ONLY
 */
cublasStatus_t
cublasGetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t handle,
                                                int *maxMantissaBitCount);
#endif
cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *userCallback);
/**
 * @param handle SEND_ONLY
 * @param mode RECV_ONLY
 */
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode);
/**
 * @disabled
 */
cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A,
                               int lda, void *B, int ldb);
/**
 * @disabled
 */
cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,
                                    const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasGetMatrixAsync_64(int64_t rows, int64_t cols,
                                       int64_t elemSize, const void *A,
                                       int64_t lda, void *B, int64_t ldb,
                                       cudaStream_t stream);
#endif
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize,
                                  const void *A, int64_t lda, void *B,
                                  int64_t ldb);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode RECV_ONLY
 */
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle,
                                       cublasPointerMode_t *mode);
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value);
/**
 * @param handle SEND_ONLY
 * @param smCountTarget RECV_ONLY
 */
cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle,
                                      int *smCountTarget);
/**
 * @disabled
 */
const char *cublasGetStatusName(cublasStatus_t status);
/**
 * @disabled
 */
const char *cublasGetStatusString(cublasStatus_t status);
/**
 * @param handle SEND_ONLY
 * @param streamId RECV_ONLY
 */
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle,
                                  cudaStream_t *streamId);
/**
 * @disabled
 */
cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx,
                               void *y, int incy);
/**
 * @disabled
 */
cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr,
                                    int incx, void *hostPtr, int incy,
                                    cudaStream_t stream);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasGetVectorAsync_64(int64_t n, int64_t elemSize,
                                       const void *devicePtr, int64_t incx,
                                       void *hostPtr, int64_t incy,
                                       cudaStream_t stream);
#endif
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasGetVector_64(int64_t n, int64_t elemSize, const void *x,
                                  int64_t incx, void *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param version RECV_ONLY
 */
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSHgemvBatched(cublasHandle_t handle,
                                    cublasOperation_t trans, int m, int n,
                                    const float *alpha,
                                    const __half *const *Aarray, int lda,
                                    const __half *const *xarray, int incx,
                                    const float *beta, __half *const *yarray,
                                    int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSHgemvBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *const *Aarray, int64_t lda,
    const __half *const *xarray, int64_t incx, const float *beta,
    __half *const *yarray, int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSHgemvStridedBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float *alpha, const __half *A, int lda, long long int strideA,
    const __half *x, int incx, long long int stridex, const float *beta,
    __half *y, int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSHgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *A, int64_t lda, long long int strideA,
    const __half *x, int64_t incx, long long int stridex, const float *beta,
    __half *y, int64_t incy, long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSSgemvBatched(cublasHandle_t handle,
                                    cublasOperation_t trans, int m, int n,
                                    const float *alpha,
                                    const __half *const *Aarray, int lda,
                                    const __half *const *xarray, int incx,
                                    const float *beta, float *const *yarray,
                                    int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSSgemvBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *const *Aarray, int64_t lda,
    const __half *const *xarray, int64_t incx, const float *beta,
    float *const *yarray, int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSSgemvStridedBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float *alpha, const __half *A, int lda, long long int strideA,
    const __half *x, int incx, long long int stridex, const float *beta,
    float *y, int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHSSgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *A, int64_t lda, long long int strideA,
    const __half *x, int64_t incx, long long int stridex, const float *beta,
    float *y, int64_t incy, long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda,
                           const __half *B, int ldb, const __half *beta,
                           __half *C, int ldc);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const __half *alpha, const __half *const *Aarray, int lda,
                   const __half *const *Barray, int ldb, const __half *beta,
                   __half *const *Carray, int ldc, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHgemmBatched_64(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb, int64_t m,
                                     int64_t n, int64_t k, const __half *alpha,
                                     const __half *const *Aarray, int64_t lda,
                                     const __half *const *Barray, int64_t ldb,
                                     const __half *beta, __half *const *Carray,
                                     int64_t ldc, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const __half *alpha, const __half *A, int lda,
                          long long int strideA, const __half *B, int ldb,
                          long long int strideB, const __half *beta, __half *C,
                          int ldc, long long int strideC, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasHgemmStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const __half *alpha, const __half *A,
    int64_t lda, long long int strideA, const __half *B, int64_t ldb,
    long long int strideB, const __half *beta, __half *C, int64_t ldc,
    long long int strideC, int64_t batchCount);
#endif
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasHgemm_64(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int64_t m, int64_t n,
                              int64_t k, const __half *alpha, const __half *A,
                              int64_t lda, const __half *B, int64_t ldb,
                              const __half *beta, __half *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void *x,
                             cudaDataType xType, int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIamaxEx_64(cublasHandle_t handle, int64_t n, const void *x,
                                cudaDataType xType, int64_t incx,
                                int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void *x,
                             cudaDataType xType, int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIaminEx_64(cublasHandle_t handle, int64_t n, const void *x,
                                cudaDataType xType, int64_t incx,
                                int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x,
                               int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIcamax_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuComplex *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x,
                               int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIcamin_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuComplex *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x,
                               int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIdamax_v2_64(cublasHandle_t handle, int64_t n,
                                  const double *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x,
                               int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIdamin_v2_64(cublasHandle_t handle, int64_t n,
                                  const double *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x,
                               int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIsamax_v2_64(cublasHandle_t handle, int64_t n,
                                  const float *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x,
                               int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIsamin_v2_64(cublasHandle_t handle, int64_t n,
                                  const float *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIzamax_v2(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIzamax_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuDoubleComplex *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIzamin_v2(cublasHandle_t handle, int n,
                               const cuDoubleComplex *x, int incx, int *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasIzamin_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuDoubleComplex *x, int64_t incx,
                                  int64_t *result);
#endif
/**
 * @disabled
 */
cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut,
                                     int logToStdErr, const char *logFileName);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, void *result,
                            cudaDataType resultType,
                            cudaDataType executionType);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(resultType)
 * @param resultType SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void *x,
                               cudaDataType xType, int64_t incx, void *result,
                               cudaDataType resultType,
                               cudaDataType executionType);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR SIZE:data_type_width(csType)
 * @param s SEND_ONLY SCALAR SIZE:data_type_width(csType)
 * @param csType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void *x,
                           cudaDataType xType, int incx, void *y,
                           cudaDataType yType, int incy, const void *c,
                           const void *s, cudaDataType csType,
                           cudaDataType executiontype);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR SIZE:data_type_width(csType)
 * @param s SEND_ONLY SCALAR SIZE:data_type_width(csType)
 * @param csType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasRotEx_64(cublasHandle_t handle, int64_t n, void *x,
                              cudaDataType xType, int64_t incx, void *y,
                              cudaDataType yType, int64_t incy, const void *c,
                              const void *s, cudaDataType csType,
                              cudaDataType executiontype);
#endif
/**
 * @param handle SEND_ONLY
 * @param a SEND_RECV SCALAR SIZE:data_type_width(abType)
 * @param b SEND_RECV SCALAR SIZE:data_type_width(abType)
 * @param abType SEND_ONLY
 * @param c RECV_ONLY SCALAR SIZE:data_type_width(csType)
 * @param s RECV_ONLY SCALAR SIZE:data_type_width(csType)
 * @param csType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasRotgEx(cublasHandle_t handle, void *a, void *b,
                            cudaDataType abType, void *c, void *s,
                            cudaDataType csType, cudaDataType executiontype);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param param SEND_ONLY SCALAR SIZE:5*data_type_width(paramType)
 * @param paramType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy, const void *param,
                            cudaDataType paramType, cudaDataType executiontype);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 * @param param SEND_ONLY SCALAR SIZE:5*data_type_width(paramType)
 * @param paramType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasRotmEx_64(cublasHandle_t handle, int64_t n, void *x,
                               cudaDataType xType, int64_t incx, void *y,
                               cudaDataType yType, int64_t incy,
                               const void *param, cudaDataType paramType,
                               cudaDataType executiontype);
#endif
/**
 * @param handle SEND_ONLY
 * @param d1 SEND_RECV SCALAR SIZE:data_type_width(d1Type)
 * @param d1Type SEND_ONLY
 * @param d2 SEND_RECV SCALAR SIZE:data_type_width(d2Type)
 * @param d2Type SEND_ONLY
 * @param x1 SEND_RECV SCALAR SIZE:data_type_width(x1Type)
 * @param x1Type SEND_ONLY
 * @param y1 SEND_ONLY SCALAR SIZE:data_type_width(y1Type)
 * @param y1Type SEND_ONLY
 * @param param RECV_ONLY SCALAR SIZE:5*data_type_width(paramType)
 * @param paramType SEND_ONLY
 * @param executiontype SEND_ONLY
 */
cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void *d1,
                             cudaDataType d1Type, void *d2, cudaDataType d2Type,
                             void *x1, cudaDataType x1Type, const void *y1,
                             cudaDataType y1Type, void *param,
                             cudaDataType paramType,
                             cudaDataType executiontype);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x,
                              int incx, float *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasSasum_v2_64(cublasHandle_t handle, int64_t n,
                                 const float *x, int64_t incx, float *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha,
                              const float *x, int incx, float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSaxpy_v2_64(cublasHandle_t handle, int64_t n,
                                 const float *alpha, const float *x,
                                 int64_t incx, float *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(alphaType)
 * @param alphaType SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha,
                            cudaDataType alphaType, void *x, cudaDataType xType,
                            int incx, cudaDataType executionType);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(alphaType)
 * @param alphaType SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param executionType SEND_ONLY
 */
cublasStatus_t cublasScalEx_64(cublasHandle_t handle, int64_t n,
                               const void *alpha, cudaDataType alphaType,
                               void *x, cudaDataType xType, int64_t incx,
                               cudaDataType executionType);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x,
                               int incx, float *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasScasum_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuComplex *x, int64_t incx,
                                  float *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex *x,
                               int incx, float *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasScnrm2_v2_64(cublasHandle_t handle, int64_t n,
                                  const cuComplex *x, int64_t incx,
                                  float *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x,
                              int incx, float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasScopy_v2_64(cublasHandle_t handle, int64_t n,
                                 const float *x, int64_t incx, float *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const float *A, int lda, const float *x,
                           int incx, float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode,
                              int64_t m, int64_t n, const float *A, int64_t lda,
                              const float *x, int64_t incx, float *C,
                              int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x,
                             int incx, const float *y, int incy, float *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasSdot_v2_64(cublasHandle_t handle, int64_t n,
                                const float *x, int64_t incx, const float *y,
                                int64_t incy, float *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 */
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle,
                                    cublasAtomicsMode_t mode);
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mask SEND_ONLY
 */
cublasStatus_t
cublasSetEmulationSpecialValuesSupport(cublasHandle_t handle,
                                       cudaEmulationSpecialValuesSupport mask);
#endif
#if CUBLAS_VERSION >= 120900
/**
 * @guard CUBLAS_VERSION >= 120900
 * @param handle SEND_ONLY
 * @param emulationStrategy SEND_ONLY
 */
cublasStatus_t
cublasSetEmulationStrategy(cublasHandle_t handle,
                           cublasEmulationStrategy_t emulationStrategy);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mantissaBitCount SEND_ONLY
 */
cublasStatus_t
cublasSetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t handle,
                                                    int *mantissaBitCount);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mantissaBitOffset SEND_ONLY
 */
cublasStatus_t
cublasSetFixedPointEmulationMantissaBitOffset(cublasHandle_t handle,
                                              int mantissaBitOffset);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param mantissaControl SEND_ONLY
 */
cublasStatus_t cublasSetFixedPointEmulationMantissaControl(
    cublasHandle_t handle, cudaEmulationMantissaControl mantissaControl);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param handle SEND_ONLY
 * @param maxMantissaBitCount SEND_ONLY
 */
cublasStatus_t
cublasSetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t handle,
                                                int maxMantissaBitCount);
#endif
cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 */
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
/**
 * @disabled
 */
cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A,
                               int lda, void *B, int ldb);
/**
 * @disabled
 */
cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,
                                    const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasSetMatrixAsync_64(int64_t rows, int64_t cols,
                                       int64_t elemSize, const void *A,
                                       int64_t lda, void *B, int64_t ldb,
                                       cudaStream_t stream);
#endif
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize,
                                  const void *A, int64_t lda, void *B,
                                  int64_t ldb);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
    cublasPointerMode_t mode) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUBLAS_STATUS_SUCCESS) {
    note_pointer_mode(handle, mode);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param smCountTarget SEND_ONLY
 */
cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget);
/**
 * @param handle SEND_ONLY
 * @param streamId SEND_ONLY
 */
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
/**
 * @disabled
 */
cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx,
                               void *devicePtr, int incy);
/**
 * @disabled
 */
cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr,
                                    int incx, void *devicePtr, int incy,
                                    cudaStream_t stream);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasSetVectorAsync_64(int64_t n, int64_t elemSize,
                                       const void *hostPtr, int64_t incx,
                                       void *devicePtr, int64_t incy,
                                       cudaStream_t stream);
#endif
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @disabled
 */
cublasStatus_t cublasSetVector_64(int64_t n, int64_t elemSize, const void *x,
                                  int64_t incx, void *devicePtr, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param workspace SEND_ONLY
 * @param workspaceSizeInBytes SEND_ONLY
 */
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace,
                                     size_t workspaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, int kl, int ku, const float *alpha,
                              const float *A, int lda, const float *x, int incx,
                              const float *beta, float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, int64_t kl, int64_t ku,
                                 const float *alpha, const float *A,
                                 int64_t lda, const float *x, int64_t incx,
                                 const float *beta, float *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *beta, const float *B, int ldb, float *C,
                           int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int64_t m, int64_t n,
                              const float *alpha, const float *A, int64_t lda,
                              const float *beta, const float *B, int64_t ldb,
                              float *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info RECV_ONLY
 * @param devInfoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasSgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, float *const *Aarray, int lda,
                                  float *const *Carray, int ldc, int *info,
                                  int *devInfoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *const *Aarray, int lda,
                   const float *const *Barray, int ldb, const float *beta,
                   float *const *Carray, int ldc, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasSgemmBatched_64(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb, int64_t m,
                                     int64_t n, int64_t k, const float *alpha,
                                     const float *const *Aarray, int64_t lda,
                                     const float *const *Barray, int64_t ldb,
                                     const float *beta, float *const *Carray,
                                     int64_t ldc, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const float *alpha, const void *A,
                             cudaDataType Atype, int lda, const void *B,
                             cudaDataType Btype, int ldb, const float *beta,
                             void *C, cudaDataType Ctype, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param Atype SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param Btype SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int64_t m, int64_t n,
                                int64_t k, const float *alpha, const void *A,
                                cudaDataType Atype, int64_t lda, const void *B,
                                cudaDataType Btype, int64_t ldb,
                                const float *beta, void *C, cudaDataType Ctype,
                                int64_t ldc);
#endif
#if CUBLAS_VERSION >= 120500
/**
 * @guard CUBLAS_VERSION >= 120500
 * @param handle SEND_ONLY
 * @param transa_array SEND_ONLY LENGTH:group_count
 * @param transb_array SEND_ONLY LENGTH:group_count
 * @param m_array SEND_ONLY LENGTH:group_count
 * @param n_array SEND_ONLY LENGTH:group_count
 * @param k_array SEND_ONLY LENGTH:group_count
 * @param alpha_array SEND_ONLY SCALAR SIZE:group_count*sizeof(float)
 * @param Aarray SEND_ONLY
 * @param lda_array SEND_ONLY LENGTH:group_count
 * @param Barray SEND_ONLY
 * @param ldb_array SEND_ONLY LENGTH:group_count
 * @param beta_array SEND_ONLY SCALAR SIZE:group_count*sizeof(float)
 * @param Carray SEND_ONLY
 * @param ldc_array SEND_ONLY LENGTH:group_count
 * @param group_count SEND_ONLY
 * @param group_size SEND_ONLY LENGTH:group_count
 */
cublasStatus_t cublasSgemmGroupedBatched(
    cublasHandle_t handle, const cublasOperation_t *transa_array,
    const cublasOperation_t *transb_array, const int *m_array,
    const int *n_array, const int *k_array, const float *alpha_array,
    const float *const *Aarray, const int *lda_array,
    const float *const *Barray, const int *ldb_array, const float *beta_array,
    float *const *Carray, const int *ldc_array, int group_count,
    const int *group_size);
#endif
#if CUBLAS_VERSION >= 120500
/**
 * @guard CUBLAS_VERSION >= 120500
 * @param handle SEND_ONLY
 * @param transa_array SEND_ONLY LENGTH:group_count
 * @param transb_array SEND_ONLY LENGTH:group_count
 * @param m_array SEND_ONLY LENGTH:group_count
 * @param n_array SEND_ONLY LENGTH:group_count
 * @param k_array SEND_ONLY LENGTH:group_count
 * @param alpha_array SEND_ONLY SCALAR SIZE:group_count*sizeof(float)
 * @param Aarray SEND_ONLY
 * @param lda_array SEND_ONLY LENGTH:group_count
 * @param Barray SEND_ONLY
 * @param ldb_array SEND_ONLY LENGTH:group_count
 * @param beta_array SEND_ONLY SCALAR SIZE:group_count*sizeof(float)
 * @param Carray SEND_ONLY
 * @param ldc_array SEND_ONLY LENGTH:group_count
 * @param group_count SEND_ONLY
 * @param group_size SEND_ONLY LENGTH:group_count
 */
cublasStatus_t cublasSgemmGroupedBatched_64(
    cublasHandle_t handle, const cublasOperation_t *transa_array,
    const cublasOperation_t *transb_array, const int64_t *m_array,
    const int64_t *n_array, const int64_t *k_array, const float *alpha_array,
    const float *const *Aarray, const int64_t *lda_array,
    const float *const *Barray, const int64_t *ldb_array,
    const float *beta_array, float *const *Carray, const int64_t *ldc_array,
    int64_t group_count, const int64_t *group_size);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const float *alpha, const float *A, int lda,
                          long long int strideA, const float *B, int ldb,
                          long long int strideB, const float *beta, float *C,
                          int ldc, long long int strideC, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasSgemmStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const float *alpha, const float *A,
    int64_t lda, long long int strideA, const float *B, int64_t ldb,
    long long int strideB, const float *beta, float *C, int64_t ldc,
    long long int strideC, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgemm_v2_64(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int64_t m, int64_t n,
                                 int64_t k, const float *alpha, const float *A,
                                 int64_t lda, const float *B, int64_t ldb,
                                 const float *beta, float *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                   const float *alpha, const float *const *Aarray, int lda,
                   const float *const *xarray, int incx, const float *beta,
                   float *const *yarray, int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasSgemvBatched_64(cublasHandle_t handle,
                                     cublasOperation_t trans, int64_t m,
                                     int64_t n, const float *alpha,
                                     const float *const *Aarray, int64_t lda,
                                     const float *const *xarray, int64_t incx,
                                     const float *beta, float *const *yarray,
                                     int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m,
                          int n, const float *alpha, const float *A, int lda,
                          long long int strideA, const float *x, int incx,
                          long long int stridex, const float *beta, float *y,
                          int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasSgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const float *A, int64_t lda, long long int strideA,
    const float *x, int64_t incx, long long int stridex, const float *beta,
    float *y, int64_t incy, long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, const float *alpha, const float *A,
                              int lda, const float *x, int incx,
                              const float *beta, float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, const float *alpha,
                                 const float *A, int64_t lda, const float *x,
                                 int64_t incx, const float *beta, float *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param TauArray SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   float *const *Aarray, int lda,
                                   float *const *TauArray, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n,
                             const float *alpha, const float *x, int incx,
                             const float *y, int incy, float *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasSger_v2_64(cublasHandle_t handle, int64_t m, int64_t n,
                                const float *alpha, const float *x,
                                int64_t incx, const float *y, int64_t incy,
                                float *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n,
                                   float *const *A, int lda, int *P, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n,
                                   const float *const *A, int lda, const int *P,
                                   float *const *C, int ldc, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const float *const *Aarray, int lda,
                                   const int *devIpiv, float *const *Barray,
                                   int ldb, int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Ainv SEND_ONLY
 * @param lda_inv SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n,
                                    const float *const *A, int lda,
                                    float *const *Ainv, int lda_inv, int *info,
                                    int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x,
                              int incx, float *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t handle, int64_t n,
                                 const float *x, int64_t incx, float *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx,
                             float *y, int incy, const float *c,
                             const float *s);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasSrot_v2_64(cublasHandle_t handle, int64_t n, float *x,
                                int64_t incx, float *y, int64_t incy,
                                const float *c, const float *s);
#endif
/**
 * @param handle SEND_ONLY
 * @param a SEND_RECV SCALAR
 * @param b SEND_RECV SCALAR
 * @param c RECV_ONLY SCALAR
 * @param s RECV_ONLY SCALAR
 */
cublasStatus_t cublasSrotg_v2(cublasHandle_t handle, float *a, float *b,
                              float *c, float *s);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param param SEND_ONLY SCALAR SIZE:5*sizeof(float)
 */
cublasStatus_t cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx,
                              float *y, int incy, const float *param);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param param SEND_ONLY SCALAR SIZE:5*sizeof(float)
 */
cublasStatus_t cublasSrotm_v2_64(cublasHandle_t handle, int64_t n, float *x,
                                 int64_t incx, float *y, int64_t incy,
                                 const float *param);
#endif
/**
 * @param handle SEND_ONLY
 * @param d1 SEND_RECV SCALAR
 * @param d2 SEND_RECV SCALAR
 * @param x1 SEND_RECV SCALAR
 * @param y1 SEND_ONLY SCALAR
 * @param param RECV_ONLY SCALAR SIZE:5*sizeof(float)
 */
cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle, float *d1, float *d2,
                               float *x1, const float *y1, float *param);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, int k, const float *alpha, const float *A,
                              int lda, const float *x, int incx,
                              const float *beta, float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, int64_t k, const float *alpha,
                                 const float *A, int64_t lda, const float *x,
                                 int64_t incx, const float *beta, float *y,
                                 int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha,
                              float *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasSscal_v2_64(cublasHandle_t handle, int64_t n,
                                 const float *alpha, float *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *alpha, const float *AP,
                              const float *x, int incx, const float *beta,
                              float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const float *alpha, const float *AP,
                                 const float *x, int64_t incx,
                                 const float *beta, float *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *alpha, const float *x,
                              int incx, const float *y, int incy, float *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasSspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const float *alpha, const float *x,
                                 int64_t incx, const float *y, int64_t incy,
                                 float *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const float *alpha, const float *x,
                             int incx, float *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasSspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const float *alpha, const float *x,
                                int64_t incx, float *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx,
                              float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSswap_v2_64(cublasHandle_t handle, int64_t n, float *x,
                                 int64_t incx, float *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, int m, int n,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int64_t m, int64_t n,
                                 const float *alpha, const float *A,
                                 int64_t lda, const float *B, int64_t ldb,
                                 const float *beta, float *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *alpha, const float *A,
                              int lda, const float *x, int incx,
                              const float *beta, float *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const float *alpha, const float *A,
                                 int64_t lda, const float *x, int64_t incx,
                                 const float *beta, float *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const float *alpha, const float *x,
                              int incx, const float *y, int incy, float *A,
                              int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasSsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const float *alpha, const float *x,
                                 int64_t incx, const float *y, int64_t incy,
                                 float *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const float *alpha, const float *A, int lda,
                               const float *B, int ldb, const float *beta,
                               float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const float *alpha, const float *A,
                                  int64_t lda, const float *B, int64_t ldb,
                                  const float *beta, float *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const float *alpha, const float *x,
                             int incx, float *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasSsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const float *alpha, const float *x,
                                int64_t incx, float *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *beta, float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int64_t n, int64_t k,
                                 const float *alpha, const float *A,
                                 int64_t lda, const float *beta, float *C,
                                 int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const float *alpha, const float *A, int lda,
                            const float *B, int ldb, const float *beta,
                            float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int64_t n, int64_t k,
                               const float *alpha, const float *A, int64_t lda,
                               const float *B, int64_t ldb, const float *beta,
                               float *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const float *A, int lda, float *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const float *A,
                                 int64_t lda, float *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const float *A, int lda, float *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const float *A,
                                 int64_t lda, float *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const float *AP, float *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const float *AP, float *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const float *AP, float *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const float *AP, float *x,
                                 int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const float *AP, float *A, int lda);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, float *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasStrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const float *alpha, const float *A,
                                 int64_t lda, const float *B, int64_t ldb,
                                 float *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const float *A, int lda, float *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const float *A, int64_t lda,
                                 float *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const float *alpha, const float *const *A,
                                  int lda, float *const *B, int ldb,
                                  int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t
cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side,
                      cublasFillMode_t uplo, cublasOperation_t trans,
                      cublasDiagType_t diag, int64_t m, int64_t n,
                      const float *alpha, const float *const *A, int64_t lda,
                      float *const *B, int64_t ldb, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const float *alpha, const float *A, int lda,
                              float *B, int ldb);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasStrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const float *alpha, const float *A,
                                 int64_t lda, float *B, int64_t ldb);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const float *A, int lda, float *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasStrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const float *A, int64_t lda,
                                 float *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const float *A, int lda, float *AP);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param xType SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param yType SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasSwapEx_64(cublasHandle_t handle, int64_t n, void *x,
                               cudaDataType xType, int64_t incx, void *y,
                               cudaDataType yType, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSSgemvBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float *alpha, const __nv_bfloat16 *const *Aarray, int lda,
    const __nv_bfloat16 *const *xarray, int incx, const float *beta,
    float *const *yarray, int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSSgemvBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __nv_bfloat16 *const *Aarray, int64_t lda,
    const __nv_bfloat16 *const *xarray, int64_t incx, const float *beta,
    float *const *yarray, int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSSgemvStridedBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float *alpha, const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *x, int incx, long long int stridex, const float *beta,
    float *y, int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSSgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __nv_bfloat16 *A, int64_t lda,
    long long int strideA, const __nv_bfloat16 *x, int64_t incx,
    long long int stridex, const float *beta, float *y, int64_t incy,
    long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSTgemvBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float *alpha, const __nv_bfloat16 *const *Aarray, int lda,
    const __nv_bfloat16 *const *xarray, int incx, const float *beta,
    __nv_bfloat16 *const *yarray, int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSTgemvBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __nv_bfloat16 *const *Aarray, int64_t lda,
    const __nv_bfloat16 *const *xarray, int64_t incx, const float *beta,
    __nv_bfloat16 *const *yarray, int64_t incy, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSTgemvStridedBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float *alpha, const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *x, int incx, long long int stridex, const float *beta,
    __nv_bfloat16 *y, int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasTSTgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __nv_bfloat16 *A, int64_t lda,
    long long int strideA, const __nv_bfloat16 *x, int64_t incx,
    long long int stridex, const float *beta, __nv_bfloat16 *y, int64_t incy,
    long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param transc SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param A_bias SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param B_bias SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param C_bias SEND_ONLY
 * @param ldc SEND_ONLY
 * @param C_mult SEND_ONLY
 * @param C_shift SEND_ONLY
 */
cublasStatus_t cublasUint8gemmBias(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    cublasOperation_t transc, int m, int n, int k, const unsigned char *A,
    int A_bias, int lda, const unsigned char *B, int B_bias, int ldb,
    unsigned char *C, int C_bias, int ldc, int C_mult, int C_shift);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZaxpy_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x, int64_t incx,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n,
                              const cuDoubleComplex *x, int incx,
                              cuDoubleComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZcopy_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuDoubleComplex *x, int64_t incx,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode,
                              int64_t m, int64_t n, const cuDoubleComplex *A,
                              int64_t lda, const cuDoubleComplex *x,
                              int64_t incx, cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasZdotc_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *result);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param result RECV_ONLY SCALAR
 */
cublasStatus_t cublasZdotu_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *result);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x,
                              int incx, cuDoubleComplex *y, int incy,
                              const double *c, const double *s);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasZdrot_v2_64(cublasHandle_t handle, int64_t n,
                                 cuDoubleComplex *x, int64_t incx,
                                 cuDoubleComplex *y, int64_t incy,
                                 const double *c, const double *s);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n,
                               const double *alpha, cuDoubleComplex *x,
                               int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZdscal_v2_64(cublasHandle_t handle, int64_t n,
                                  const double *alpha, cuDoubleComplex *x,
                                  int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, int kl, int ku,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param kl SEND_ONLY
 * @param ku SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n, int64_t kl, int64_t ku,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *beta,
                           const cuDoubleComplex *B, int ldb,
                           cuDoubleComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int64_t m, int64_t n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int64_t lda,
                              const cuDoubleComplex *beta,
                              const cuDoubleComplex *B, int64_t ldb,
                              cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info RECV_ONLY
 * @param devInfoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasZgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, cuDoubleComplex *const *Aarray,
                                  int lda, cuDoubleComplex *const *Carray,
                                  int ldc, int *info, int *devInfoArray,
                                  int batchSize);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const cuDoubleComplex *alpha,
                             const cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *B, int ldb,
                             const cuDoubleComplex *beta, cuDoubleComplex *C,
                             int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int64_t m, int64_t n,
                                int64_t k, const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int64_t lda,
                                const cuDoubleComplex *B, int64_t ldb,
                                const cuDoubleComplex *beta, cuDoubleComplex *C,
                                int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *const *Aarray, int lda,
    const cuDoubleComplex *const *Barray, int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *const *Carray, int ldc, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param Carray SEND_ONLY
 * @param ldc SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemmBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *const *Aarray, int64_t lda,
    const cuDoubleComplex *const *Barray, int64_t ldb,
    const cuDoubleComplex *beta, cuDoubleComplex *const *Carray, int64_t ldc,
    int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, long long int strideA, const cuDoubleComplex *B, int ldb,
    long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C,
    int ldc, long long int strideC, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param strideB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param strideC SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemmStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int64_t lda, long long int strideA,
    const cuDoubleComplex *B, int64_t ldb, long long int strideB,
    const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc,
    long long int strideC, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *B, int ldb,
                              const cuDoubleComplex *beta, cuDoubleComplex *C,
                              int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t
cublasZgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, int64_t m, int64_t n, int64_t k,
                  const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                  int64_t lda, const cuDoubleComplex *B, int64_t ldb,
                  const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemvBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *const *Aarray, int lda,
    const cuDoubleComplex *const *xarray, int incx, const cuDoubleComplex *beta,
    cuDoubleComplex *const *yarray, int incy, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param xarray SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param yarray SEND_ONLY
 * @param incy SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemvBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *const *Aarray,
    int64_t lda, const cuDoubleComplex *const *xarray, int64_t incx,
    const cuDoubleComplex *beta, cuDoubleComplex *const *yarray, int64_t incy,
    int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemvStridedBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
    long long int strideA, const cuDoubleComplex *x, int incx,
    long long int stridex, const cuDoubleComplex *beta, cuDoubleComplex *y,
    int incy, long long int stridey, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param stridex SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param stridey SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZgemvStridedBatched_64(
    cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda,
    long long int strideA, const cuDoubleComplex *x, int64_t incx,
    long long int stridex, const cuDoubleComplex *beta, cuDoubleComplex *y,
    int64_t incy, long long int stridey, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans,
                              int m, int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans,
                                 int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param TauArray SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   cuDoubleComplex *const *Aarray, int lda,
                                   cuDoubleComplex *const *TauArray, int *info,
                                   int batchSize);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n,
                                   cuDoubleComplex *const *A, int lda, int *P,
                                   int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param P SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n,
                                   const cuDoubleComplex *const *A, int lda,
                                   const int *P, cuDoubleComplex *const *C,
                                   int ldc, int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param Barray SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info RECV_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const cuDoubleComplex *const *Aarray,
                                   int lda, const int *devIpiv,
                                   cuDoubleComplex *const *Barray, int ldb,
                                   int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, int k, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZhbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, int64_t k,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *B, int ldb,
                              const cuDoubleComplex *beta, cuDoubleComplex *C,
                              int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZhemm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *B, int64_t ldb,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZhemv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZher2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const double *beta, cuDoubleComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZher2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const cuDoubleComplex *alpha,
                                  const cuDoubleComplex *A, int64_t lda,
                                  const cuDoubleComplex *B, int64_t ldb,
                                  const double *beta, cuDoubleComplex *C,
                                  int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const double *alpha,
                             const cuDoubleComplex *x, int incx,
                             cuDoubleComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZher_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const double *alpha,
                                const cuDoubleComplex *x, int64_t incx,
                                cuDoubleComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, int n, int k,
                              const double *alpha, const cuDoubleComplex *A,
                              int lda, const double *beta, cuDoubleComplex *C,
                              int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZherk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int64_t n, int64_t k,
                                 const double *alpha, const cuDoubleComplex *A,
                                 int64_t lda, const double *beta,
                                 cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const double *beta, cuDoubleComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int64_t n, int64_t k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int64_t lda,
                               const cuDoubleComplex *B, int64_t ldb,
                               const double *beta, cuDoubleComplex *C,
                               int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *AP,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZhpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *AP,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasZhpr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const double *alpha,
                             const cuDoubleComplex *x, int incx,
                             cuDoubleComplex *AP);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasZhpr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const double *alpha,
                                const cuDoubleComplex *x, int64_t incx,
                                cuDoubleComplex *AP);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Ainv SEND_ONLY
 * @param lda_inv SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n,
                                    const cuDoubleComplex *const *A, int lda,
                                    cuDoubleComplex *const *Ainv, int lda_inv,
                                    int *info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x,
                             int incx, cuDoubleComplex *y, int incy,
                             const double *c, const cuDoubleComplex *s);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 */
cublasStatus_t cublasZrot_v2_64(cublasHandle_t handle, int64_t n,
                                cuDoubleComplex *x, int64_t incx,
                                cuDoubleComplex *y, int64_t incy,
                                const double *c, const cuDoubleComplex *s);
#endif
/**
 * @param handle SEND_ONLY
 * @param a SEND_RECV SCALAR
 * @param b SEND_RECV SCALAR
 * @param c RECV_ONLY SCALAR
 * @param s RECV_ONLY SCALAR
 */
cublasStatus_t cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex *a,
                              cuDoubleComplex *b, double *c,
                              cuDoubleComplex *s);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n,
                              const cuDoubleComplex *alpha, cuDoubleComplex *x,
                              int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZscal_v2_64(cublasHandle_t handle, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x,
                              int incx, cuDoubleComplex *y, int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZswap_v2_64(cublasHandle_t handle, int64_t n,
                                 cuDoubleComplex *x, int64_t incx,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *B, int ldb,
                              const cuDoubleComplex *beta, cuDoubleComplex *C,
                              int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *B, int64_t ldb,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *beta, cuDoubleComplex *y,
                              int incy);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 */
cublasStatus_t cublasZsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *y, int64_t incy);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              int n, const cuDoubleComplex *alpha,
                              const cuDoubleComplex *x, int incx,
                              const cuDoubleComplex *y, int incy,
                              cuDoubleComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param y SEND_ONLY
 * @param incy SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int64_t n, const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *x, int64_t incx,
                                 const cuDoubleComplex *y, int64_t incy,
                                 cuDoubleComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int lda,
                               const cuDoubleComplex *B, int ldb,
                               const cuDoubleComplex *beta, cuDoubleComplex *C,
                               int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int64_t n, int64_t k,
                                  const cuDoubleComplex *alpha,
                                  const cuDoubleComplex *A, int64_t lda,
                                  const cuDoubleComplex *B, int64_t ldb,
                                  const cuDoubleComplex *beta,
                                  cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                             int n, const cuDoubleComplex *alpha,
                             const cuDoubleComplex *x, int incx,
                             cuDoubleComplex *A, int lda);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                int64_t n, const cuDoubleComplex *alpha,
                                const cuDoubleComplex *x, int64_t incx,
                                cuDoubleComplex *A, int64_t lda);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, int n, int k,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *beta, cuDoubleComplex *C,
                              int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int64_t n, int64_t k,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *beta,
                                 cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const cuDoubleComplex *beta, cuDoubleComplex *C,
                            int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int64_t n, int64_t k,
                               const cuDoubleComplex *alpha,
                               const cuDoubleComplex *A, int64_t lda,
                               const cuDoubleComplex *B, int64_t ldb,
                               const cuDoubleComplex *beta, cuDoubleComplex *C,
                               int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const cuDoubleComplex *A, int lda,
                              cuDoubleComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const cuDoubleComplex *A,
                                 int64_t lda, cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, int k, const cuDoubleComplex *A, int lda,
                              cuDoubleComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, int64_t k, const cuDoubleComplex *A,
                                 int64_t lda, cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuDoubleComplex *AP,
                              cuDoubleComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuDoubleComplex *AP,
                                 cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuDoubleComplex *AP,
                              cuDoubleComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuDoubleComplex *AP,
                                 cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param AP SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuDoubleComplex *AP, cuDoubleComplex *A,
                            int lda);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              const cuDoubleComplex *B, int ldb,
                              cuDoubleComplex *C, int ldc);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasZtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 const cuDoubleComplex *B, int64_t ldb,
                                 cuDoubleComplex *C, int64_t ldc);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuDoubleComplex *A, int lda,
                              cuDoubleComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuDoubleComplex *A,
                                 int64_t lda, cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZtrsmBatched(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *const *A, int lda,
    cuDoubleComplex *const *B, int ldb, int batchCount);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cublasStatus_t cublasZtrsmBatched_64(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *const *A, int64_t lda,
    cuDoubleComplex *const *B, int64_t ldb, int64_t batchCount);
#endif
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int m, int n,
                              const cuDoubleComplex *alpha,
                              const cuDoubleComplex *A, int lda,
                              cuDoubleComplex *B, int ldb);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 */
cublasStatus_t cublasZtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int64_t m, int64_t n,
                                 const cuDoubleComplex *alpha,
                                 const cuDoubleComplex *A, int64_t lda,
                                 cuDoubleComplex *B, int64_t ldb);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int n, const cuDoubleComplex *A, int lda,
                              cuDoubleComplex *x, int incx);
#if CUBLAS_VERSION >= 120000
/**
 * @guard CUBLAS_VERSION >= 120000
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param x SEND_ONLY
 * @param incx SEND_ONLY
 */
cublasStatus_t cublasZtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, cublasDiagType_t diag,
                                 int64_t n, const cuDoubleComplex *A,
                                 int64_t lda, cuDoubleComplex *x, int64_t incx);
#endif
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param AP SEND_ONLY
 */
cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuDoubleComplex *A, int lda,
                            cuDoubleComplex *AP);
