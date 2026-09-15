// The experimental block holds the SpMVOp calls, which the shim forwards too.
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#include <cusparse.h>

// cuSPARSE API. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libcusparse
// by name at runtime. A handle, descriptor, plan, or info structure is an
// address in the server's library and routes to the connection that created
// it; a call with none goes to the runtime's current device.
//
// Matrix and vector data, index arrays, and work buffers are device pointers
// and travel as addresses, so descriptors are opaque handles the client never
// reads. A scalar the handle's pointer mode places on the host or on the
// device (alpha, beta, a threshold, a returned nnz or pivot) is `SCALAR`. A
// generic call sizes it by its compute type; cusparseAxpby, cusparseRot and
// cusparseSpMVOp have none, so the client uses the value type it recorded when
// the vector descriptor was created. LTO IR and user data buffers are host
// bytes sized by their length parameter.
//
// The logger calls that take a client callback or FILE are declared without
// @param lines, so each gets a stub returning CUSPARSE_STATUS_NOT_SUPPORTED.
// The header's bufferSizeExt and gebsr2csr declarations the library does not
// export are left out, so the shim exports what the library does.
//
// Each @guard names the first cuSPARSE release with that signature. The 12.0
// headers retyped read-only descriptor parameters as const; the shim aliases
// those names on 11.x, so the retyped calls carry no guard.
//
// The calls at the end of this file are the ones cuSPARSE 12.0 removed. They
// are declared here as the 11.x headers declare them, since this file is read
// against a newer header that no longer does, and each is guarded to 11.x.

/**
 * @param handle RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param version RECV_ONLY
 */
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int *version);
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int *value);
/**
 * @disabled
 */
const char *cusparseGetErrorName(cusparseStatus_t status);
/**
 * @disabled
 */
const char *cusparseGetErrorString(cusparseStatus_t status);
/**
 * @param handle SEND_ONLY
 * @param streamId SEND_ONLY
 */
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle,
                                   cudaStream_t streamId);
/**
 * @param handle SEND_ONLY
 * @param streamId RECV_ONLY
 */
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle,
                                   cudaStream_t *streamId);
/**
 * @param handle SEND_ONLY
 * @param mode RECV_ONLY
 */
cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle,
                                        cusparsePointerMode_t *mode);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_pointer_mode(handle, mode);
  }
  return return_value;
}
// clang-format on
cusparseStatus_t cusparseLoggerSetCallback(cusparseLoggerCallback_t callback);
cusparseStatus_t cusparseLoggerSetFile(FILE *file);
/**
 * @param logFile SEND_ONLY NULL_TERMINATED
 */
cusparseStatus_t cusparseLoggerOpenFile(const char *logFile);
/**
 * @param level SEND_ONLY
 */
cusparseStatus_t cusparseLoggerSetLevel(int level);
/**
 * @param mask SEND_ONLY
 */
cusparseStatus_t cusparseLoggerSetMask(int mask);
/**
 */
cusparseStatus_t cusparseLoggerForceDisable();
/**
 * @param descrA RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *descrA);
  }
  return return_value;
}
// clang-format on
/**
 * @param descrA SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(descrA);
  return return_value;
}
// clang-format on
/**
 * @param descrA SEND_ONLY
 * @param type SEND_ONLY
 */
cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA,
                                    cusparseMatrixType_t type);
/**
 * @param descrA SEND_ONLY
 */
cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA);
/**
 * @param descrA SEND_ONLY
 * @param fillMode SEND_ONLY
 */
cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                                        cusparseFillMode_t fillMode);
/**
 * @param descrA SEND_ONLY
 */
cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
/**
 * @param descrA SEND_ONLY
 * @param diagType SEND_ONLY
 */
cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                                        cusparseDiagType_t diagType);
/**
 * @param descrA SEND_ONLY
 */
cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA);
/**
 * @param descrA SEND_ONLY
 * @param base SEND_ONLY
 */
cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA,
                                         cusparseIndexBase_t base);
/**
 * @param descrA SEND_ONLY
 */
cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateBsrsv2Info(bsrsv2Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyBsrsv2Info(bsrsv2Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateBsrsm2Info(bsrsm2Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyBsrsm2Info(bsrsm2Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsru2csrInfo(csru2csrInfo_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyCsru2csrInfo(csru2csrInfo_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateColorInfo(cusparseColorInfo_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyColorInfo(cusparseColorInfo_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreatePruneInfo(pruneInfo_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyPruneInfo(pruneInfo_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgemvi(cusparseHandle_t handle,
                                cusparseOperation_t transA, int m, int n,
                                const float *alpha, const float *A, int lda,
                                int nnz, const float *xVal, const int *xInd,
                                const float *beta, float *y,
                                cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSgemvi_bufferSize(cusparseHandle_t handle,
                                           cusparseOperation_t transA, int m,
                                           int n, int nnz, int *pBufferSize);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgemvi(cusparseHandle_t handle,
                                cusparseOperation_t transA, int m, int n,
                                const double *alpha, const double *A, int lda,
                                int nnz, const double *xVal, const int *xInd,
                                const double *beta, double *y,
                                cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseDgemvi_bufferSize(cusparseHandle_t handle,
                                           cusparseOperation_t transA, int m,
                                           int n, int nnz, int *pBufferSize);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgemvi(cusparseHandle_t handle,
                                cusparseOperation_t transA, int m, int n,
                                const cuComplex *alpha, const cuComplex *A,
                                int lda, int nnz, const cuComplex *xVal,
                                const int *xInd, const cuComplex *beta,
                                cuComplex *y, cusparseIndexBase_t idxBase,
                                void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseCgemvi_bufferSize(cusparseHandle_t handle,
                                           cusparseOperation_t transA, int m,
                                           int n, int nnz, int *pBufferSize);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgemvi(cusparseHandle_t handle,
                                cusparseOperation_t transA, int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda, int nnz,
                                const cuDoubleComplex *xVal, const int *xInd,
                                const cuDoubleComplex *beta, cuDoubleComplex *y,
                                cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseZgemvi_bufferSize(cusparseHandle_t handle,
                                           cusparseOperation_t transA, int m,
                                           int n, int nnz, int *pBufferSize);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, int mb, int nb, int nnzb,
               const float *alpha, const cusparseMatDescr_t descrA,
               const float *bsrSortedValA, const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA, int blockDim, const float *x,
               const float *beta, float *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, int mb, int nb, int nnzb,
               const double *alpha, const cusparseMatDescr_t descrA,
               const double *bsrSortedValA, const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA, int blockDim, const double *x,
               const double *beta, double *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, int mb, int nb, int nnzb,
               const cuComplex *alpha, const cusparseMatDescr_t descrA,
               const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA, int blockDim, const cuComplex *x,
               const cuComplex *beta, cuComplex *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, int mb, int nb, int nnzb,
               const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
               const cuDoubleComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
               int blockDim, const cuDoubleComplex *x,
               const cuDoubleComplex *beta, cuDoubleComplex *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param sizeOfMask SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedMaskPtrA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedEndPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseSbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
                const float *bsrSortedValA, const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA, int blockDim, const float *x,
                const float *beta, float *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param sizeOfMask SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedMaskPtrA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedEndPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseDbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
                const double *bsrSortedValA, const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA, int blockDim, const double *x,
                const double *beta, double *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param sizeOfMask SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedMaskPtrA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedEndPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t
cusparseCbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const cuComplex *alpha,
                const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
                const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA,
                const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
                int blockDim, const cuComplex *x, const cuComplex *beta,
                cuComplex *y);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param sizeOfMask SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedMaskPtrA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedEndPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param y SEND_ONLY
 */
cusparseStatus_t cusparseZbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
    const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
    const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x,
    const cuDoubleComplex *beta, cuDoubleComplex *y);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle,
                                           bsrsv2Info_t info, int *position);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseSbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                           cusparseOperation_t transA, int mb, int nnzb,
                           const cusparseMatDescr_t descrA,
                           float *bsrSortedValA, const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA, int blockDim,
                           bsrsv2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseDbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                           cusparseOperation_t transA, int mb, int nnzb,
                           const cusparseMatDescr_t descrA,
                           double *bsrSortedValA, const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA, int blockDim,
                           bsrsv2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                      cusparseOperation_t transA, int mb, int nnzb,
                      const float *alpha, const cusparseMatDescr_t descrA,
                      const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA, int blockDim,
                      bsrsv2Info_t info, const float *f, float *x,
                      cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                      cusparseOperation_t transA, int mb, int nnzb,
                      const double *alpha, const cusparseMatDescr_t descrA,
                      const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA, int blockDim,
                      bsrsv2Info_t info, const double *f, double *x,
                      cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const cuComplex *f, cuComplex *x,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, int mb, int nnzb, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const cuDoubleComplex *f, cuDoubleComplex *x,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param kb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t
cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, cusparseOperation_t transB, int mb,
               int n, int kb, int nnzb, const float *alpha,
               const cusparseMatDescr_t descrA, const float *bsrSortedValA,
               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
               const int blockSize, const float *B, const int ldb,
               const float *beta, float *C, int ldc);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param kb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t
cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, cusparseOperation_t transB, int mb,
               int n, int kb, int nnzb, const double *alpha,
               const cusparseMatDescr_t descrA, const double *bsrSortedValA,
               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
               const int blockSize, const double *B, const int ldb,
               const double *beta, double *C, int ldc);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param kb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t
cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, cusparseOperation_t transB, int mb,
               int n, int kb, int nnzb, const cuComplex *alpha,
               const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
               const int blockSize, const cuComplex *B, const int ldb,
               const cuComplex *beta, cuComplex *C, int ldc);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param kb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t
cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
               cusparseOperation_t transA, cusparseOperation_t transB, int mb,
               int n, int kb, int nnzb, const cuDoubleComplex *alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
               const int blockSize, const cuDoubleComplex *B, const int ldb,
               const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle,
                                           bsrsm2Info_t info, int *position);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param X SEND_ONLY
 * @param ldx SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                      cusparseOperation_t transA, cusparseOperation_t transXY,
                      int mb, int n, int nnzb, const float *alpha,
                      const cusparseMatDescr_t descrA,
                      const float *bsrSortedVal, const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd, int blockSize,
                      bsrsm2Info_t info, const float *B, int ldb, float *X,
                      int ldx, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param X SEND_ONLY
 * @param ldx SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA,
                      cusparseOperation_t transA, cusparseOperation_t transXY,
                      int mb, int n, int nnzb, const double *alpha,
                      const cusparseMatDescr_t descrA,
                      const double *bsrSortedVal, const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd, int blockSize,
                      bsrsm2Info_t info, const double *B, int ldb, double *X,
                      int ldx, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param X SEND_ONLY
 * @param ldx SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const cuComplex *B, int ldb, cuComplex *X, int ldx,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param transA SEND_ONLY
 * @param transXY SEND_ONLY
 * @param mb SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param info SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param X SEND_ONLY
 * @param ldx SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA,
    cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n,
    int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const cuDoubleComplex *B, int ldb, cuDoubleComplex *X, int ldx,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                float *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                double *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                cuComplex *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                cuDoubleComplex *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle,
                                             csrilu02Info_t info,
                                             int *position);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseScsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseMatDescr_t descrA,
                                   float *csrSortedValA_valM,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrilu02Info_t info,
                                   cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseMatDescr_t descrA,
                                   double *csrSortedValA_valM,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrilu02Info_t info,
                                   cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseMatDescr_t descrA,
                                   cuComplex *csrSortedValA_valM,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrilu02Info_t info,
                                   cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseMatDescr_t descrA,
                                   cuDoubleComplex *csrSortedValA_valM,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrilu02Info_t info,
                                   cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                float *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                double *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                cuComplex *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param enable_boost SEND_ONLY
 * @param tol SEND_ONLY SCALAR
 * @param boost_val SEND_ONLY SCALAR
 */
cusparseStatus_t cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t info,
                                                int enable_boost, double *tol,
                                                cuDoubleComplex *boost_val);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                                             bsrilu02Info_t info,
                                             int *position);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseSbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                             int mb, int nnzb, const cusparseMatDescr_t descrA,
                             float *bsrSortedVal, const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd, int blockDim,
                             bsrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseDbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                             int mb, int nnzb, const cusparseMatDescr_t descrA,
                             double *bsrSortedVal, const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd, int blockDim,
                             bsrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                  int nnzb, const cusparseMatDescr_t descrA,
                  float *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                  int nnzb, const cusparseMatDescr_t descrA,
                  double *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseCbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                  int nnzb, const cusparseMatDescr_t descrA,
                  cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseZbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                  int nnzb, const cusparseMatDescr_t descrA,
                  cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                                            csric02Info_t info, int *position);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseScsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsric02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csric02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsric02(cusparseHandle_t handle, int m, int nnz,
                                  const cusparseMatDescr_t descrA,
                                  float *csrSortedValA_valM,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  csric02Info_t info,
                                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsric02(cusparseHandle_t handle, int m, int nnz,
                                  const cusparseMatDescr_t descrA,
                                  double *csrSortedValA_valM,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  csric02Info_t info,
                                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsric02(cusparseHandle_t handle, int m, int nnz,
                                  const cusparseMatDescr_t descrA,
                                  cuComplex *csrSortedValA_valM,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  csric02Info_t info,
                                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA_valM SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsric02(cusparseHandle_t handle, int m, int nnz,
                                  const cusparseMatDescr_t descrA,
                                  cuDoubleComplex *csrSortedValA_valM,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  csric02Info_t info,
                                  cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                                            bsric02Info_t info, int *position);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseSbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                            int mb, int nnzb, const cusparseMatDescr_t descrA,
                            float *bsrSortedVal, const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd, int blockDim,
                            bsric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseDbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                            int mb, int nnzb, const cusparseMatDescr_t descrA,
                            double *bsrSortedVal, const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd, int blockDim,
                            bsric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseCbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA,
                            int mb, int nnzb, const cusparseMatDescr_t descrA,
                            cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd, int blockDim,
                            bsric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZbsric02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pInputBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pInputBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pInputBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pInputBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal,
                 const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                 int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy,
                 void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nnzb, const cusparseMatDescr_t descrA,
                 double *bsrSortedVal, const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                 cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseCbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nnzb, const cusparseMatDescr_t descrA,
                 cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                 cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseZbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nnzb, const cusparseMatDescr_t descrA,
                 cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                 cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m,
                                              int n, const float *dl,
                                              const float *d, const float *du,
                                              const float *B, int ldb,
                                              size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m,
                                              int n, const double *dl,
                                              const double *d, const double *du,
                                              const double *B, int ldb,
                                              size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m,
                                              int n, const cuComplex *dl,
                                              const cuComplex *d,
                                              const cuComplex *du,
                                              const cuComplex *B, int ldb,
                                              size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m,
                                              int n, const cuDoubleComplex *dl,
                                              const cuDoubleComplex *d,
                                              const cuDoubleComplex *du,
                                              const cuDoubleComplex *B, int ldb,
                                              size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgtsv2(cusparseHandle_t handle, int m, int n,
                                const float *dl, const float *d,
                                const float *du, float *B, int ldb,
                                void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgtsv2(cusparseHandle_t handle, int m, int n,
                                const double *dl, const double *d,
                                const double *du, double *B, int ldb,
                                void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgtsv2(cusparseHandle_t handle, int m, int n,
                                const cuComplex *dl, const cuComplex *d,
                                const cuComplex *du, cuComplex *B, int ldb,
                                void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgtsv2(cusparseHandle_t handle, int m, int n,
                                const cuDoubleComplex *dl,
                                const cuDoubleComplex *d,
                                const cuDoubleComplex *du, cuDoubleComplex *B,
                                int ldb, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *dl, const float *d,
    const float *du, const float *B, int ldb, size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *dl, const double *d,
    const double *du, const double *B, int ldb, size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                     const cuComplex *dl, const cuComplex *d,
                                     const cuComplex *du, const cuComplex *B,
                                     int ldb, size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                        const float *dl, const float *d,
                                        const float *du, float *B, int ldb,
                                        void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                        const double *dl, const double *d,
                                        const double *du, double *B, int ldb,
                                        void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                        const cuComplex *dl, const cuComplex *d,
                                        const cuComplex *du, cuComplex *B,
                                        int ldb, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                        const cuDoubleComplex *dl,
                                        const cuDoubleComplex *d,
                                        const cuDoubleComplex *du,
                                        cuDoubleComplex *B, int ldb,
                                        void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const float *dl, const float *d,
    const float *du, const float *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const double *dl, const double *d,
    const double *du, const double *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuComplex *dl, const cuComplex *d,
    const cuComplex *du, const cuComplex *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m,
                                            const float *dl, const float *d,
                                            const float *du, float *x,
                                            int batchCount, int batchStride,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m,
                                            const double *dl, const double *d,
                                            const double *du, double *x,
                                            int batchCount, int batchStride,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m,
                                            const cuComplex *dl,
                                            const cuComplex *d,
                                            const cuComplex *du, cuComplex *x,
                                            int batchCount, int batchStride,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m,
                                            const cuDoubleComplex *dl,
                                            const cuDoubleComplex *d,
                                            const cuDoubleComplex *du,
                                            cuDoubleComplex *x, int batchCount,
                                            int batchStride, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const float *dl, const float *d,
    const float *du, const float *x, int batchCount,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const double *dl, const double *d,
    const double *du, const double *x, int batchCount,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuComplex *dl,
    const cuComplex *d, const cuComplex *du, const cuComplex *x, int batchCount,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgtsvInterleavedBatch(cusparseHandle_t handle,
                                               int algo, int m, float *dl,
                                               float *d, float *du, float *x,
                                               int batchCount, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgtsvInterleavedBatch(cusparseHandle_t handle,
                                               int algo, int m, double *dl,
                                               double *d, double *du, double *x,
                                               int batchCount, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgtsvInterleavedBatch(cusparseHandle_t handle,
                                               int algo, int m, cuComplex *dl,
                                               cuComplex *d, cuComplex *du,
                                               cuComplex *x, int batchCount,
                                               void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                              cuDoubleComplex *dl, cuDoubleComplex *d,
                              cuDoubleComplex *du, cuDoubleComplex *x,
                              int batchCount, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, const float *x,
    int batchCount, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const double *ds,
    const double *dl, const double *d, const double *du, const double *dw,
    const double *x, int batchCount, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuComplex *ds,
    const cuComplex *dl, const cuComplex *d, const cuComplex *du,
    const cuComplex *dw, const cuComplex *x, int batchCount,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *ds,
    const cuDoubleComplex *dl, const cuDoubleComplex *d,
    const cuDoubleComplex *du, const cuDoubleComplex *dw,
    const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgpsvInterleavedBatch(cusparseHandle_t handle,
                                               int algo, int m, float *ds,
                                               float *dl, float *d, float *du,
                                               float *dw, float *x,
                                               int batchCount, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgpsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, double *ds, double *dl, double *d,
    double *du, double *dw, double *x, int batchCount, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgpsvInterleavedBatch(cusparseHandle_t handle,
                                               int algo, int m, cuComplex *ds,
                                               cuComplex *dl, cuComplex *d,
                                               cuComplex *du, cuComplex *dw,
                                               cuComplex *x, int batchCount,
                                               void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param m SEND_ONLY
 * @param ds SEND_ONLY
 * @param dl SEND_ONLY
 * @param d SEND_ONLY
 * @param du SEND_ONLY
 * @param dw SEND_ONLY
 * @param x SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgpsvInterleavedBatch(
    cusparseHandle_t handle, int algo, int m, cuDoubleComplex *ds,
    cuDoubleComplex *dl, cuDoubleComplex *d, cuDoubleComplex *du,
    cuDoubleComplex *dw, cuDoubleComplex *x, int batchCount, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cusparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *beta, const cusparseMatDescr_t descrB, int nnzB,
    const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuComplex *beta, const cusparseMatDescr_t descrB, int nnzB,
    const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuComplex *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrB, int nnzB,
    const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param workspace SEND_ONLY
 */
cusparseStatus_t cusparseXcsrgeam2Nnz(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrgeam2(
    cusparseHandle_t handle, int m, int n, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double *alpha,
                  const cusparseMatDescr_t descrA, int nnzA,
                  const double *csrSortedValA, const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA, const double *beta,
                  const cusparseMatDescr_t descrB, int nnzB,
                  const double *csrSortedValB, const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                  double *csrSortedValC, int *csrSortedRowPtrC,
                  int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex *alpha,
                  const cusparseMatDescr_t descrA, int nnzA,
                  const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA, const cuComplex *beta,
                  const cusparseMatDescr_t descrB, int nnzB,
                  const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                  cuComplex *csrSortedValC, int *csrSortedRowPtrC,
                  int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseZcsrgeam2(cusparseHandle_t handle, int m, int n,
                  const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                  int nnzA, const cuDoubleComplex *csrSortedValA,
                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const cuDoubleComplex *beta, const cusparseMatDescr_t descrB,
                  int nnzB, const cuDoubleComplex *csrSortedValB,
                  const int *csrSortedRowPtrB, const int *csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
                  int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param fractionToColor SEND_ONLY SCALAR
 * @param ncolors RECV_ONLY SCALAR
 * @param coloring SEND_ONLY
 * @param reordering SEND_ONLY
 * @param info SEND_ONLY
 */
cusparseStatus_t
cusparseScsrcolor(cusparseHandle_t handle, int m, int nnz,
                  const cusparseMatDescr_t descrA, const float *csrSortedValA,
                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const float *fractionToColor, int *ncolors, int *coloring,
                  int *reordering, const cusparseColorInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param fractionToColor SEND_ONLY SCALAR
 * @param ncolors RECV_ONLY SCALAR
 * @param coloring SEND_ONLY
 * @param reordering SEND_ONLY
 * @param info SEND_ONLY
 */
cusparseStatus_t
cusparseDcsrcolor(cusparseHandle_t handle, int m, int nnz,
                  const cusparseMatDescr_t descrA, const double *csrSortedValA,
                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const double *fractionToColor, int *ncolors, int *coloring,
                  int *reordering, const cusparseColorInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param fractionToColor SEND_ONLY SCALAR
 * @param ncolors RECV_ONLY SCALAR
 * @param coloring SEND_ONLY
 * @param reordering SEND_ONLY
 * @param info SEND_ONLY
 */
cusparseStatus_t cusparseCcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param fractionToColor SEND_ONLY SCALAR
 * @param ncolors RECV_ONLY SCALAR
 * @param coloring SEND_ONLY
 * @param reordering SEND_ONLY
 * @param info SEND_ONLY
 */
cusparseStatus_t cusparseZcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRowCol SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA,
                              int m, int n, const cusparseMatDescr_t descrA,
                              const float *A, int lda, int *nnzPerRowCol,
                              int *nnzTotalDevHostPtr);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRowCol SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA,
                              int m, int n, const cusparseMatDescr_t descrA,
                              const double *A, int lda, int *nnzPerRowCol,
                              int *nnzTotalDevHostPtr);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRowCol SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA,
                              int m, int n, const cusparseMatDescr_t descrA,
                              const cuComplex *A, int lda, int *nnzPerRowCol,
                              int *nnzTotalDevHostPtr);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRowCol SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA,
                              int m, int n, const cusparseMatDescr_t descrA,
                              const cuDoubleComplex *A, int lda,
                              int *nnzPerRowCol, int *nnzTotalDevHostPtr);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param descr SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param nnzC RECV_ONLY SCALAR
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseSnnz_compress(cusparseHandle_t handle, int m,
                                       const cusparseMatDescr_t descr,
                                       const float *csrSortedValA,
                                       const int *csrSortedRowPtrA,
                                       int *nnzPerRow, int *nnzC, float tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param descr SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param nnzC RECV_ONLY SCALAR
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseDnnz_compress(cusparseHandle_t handle, int m,
                                       const cusparseMatDescr_t descr,
                                       const double *csrSortedValA,
                                       const int *csrSortedRowPtrA,
                                       int *nnzPerRow, int *nnzC, double tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param descr SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param nnzC RECV_ONLY SCALAR
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseCnnz_compress(cusparseHandle_t handle, int m,
                                       const cusparseMatDescr_t descr,
                                       const cuComplex *csrSortedValA,
                                       const int *csrSortedRowPtrA,
                                       int *nnzPerRow, int *nnzC,
                                       cuComplex tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param descr SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param nnzC RECV_ONLY SCALAR
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseZnnz_compress(cusparseHandle_t handle, int m,
                                       const cusparseMatDescr_t descr,
                                       const cuDoubleComplex *csrSortedValA,
                                       const int *csrSortedRowPtrA,
                                       int *nnzPerRow, int *nnzC,
                                       cuDoubleComplex tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseScsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    float *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    float tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseDcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    double *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    double tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseCcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    cuComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    cuComplex tol);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param tol SEND_ONLY
 */
cusparseStatus_t cusparseZcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    cuDoubleComplex *csrSortedValC, int *csrSortedColIndC,
    int *csrSortedRowPtrC, cuDoubleComplex tol);
/**
 * @param handle SEND_ONLY
 * @param cooRowInd SEND_ONLY
 * @param nnz SEND_ONLY
 * @param m SEND_ONLY
 * @param csrSortedRowPtr SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int *cooRowInd,
                                  int nnz, int m, int *csrSortedRowPtr,
                                  cusparseIndexBase_t idxBase);
/**
 * @param handle SEND_ONLY
 * @param csrSortedRowPtr SEND_ONLY
 * @param nnz SEND_ONLY
 * @param m SEND_ONLY
 * @param cooRowInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t handle,
                                  const int *csrSortedRowPtr, int nnz, int m,
                                  int *cooRowInd, cusparseIndexBase_t idxBase);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 */
cusparseStatus_t
cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                    int n, const cusparseMatDescr_t descrA,
                    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                    int blockDim, const cusparseMatDescr_t descrC,
                    int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 */
cusparseStatus_t
cusparseScsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                 int n, const cusparseMatDescr_t descrA,
                 const float *csrSortedValA, const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA, int blockDim,
                 const cusparseMatDescr_t descrC, float *bsrSortedValC,
                 int *bsrSortedRowPtrC, int *bsrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 */
cusparseStatus_t
cusparseDcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                 int n, const cusparseMatDescr_t descrA,
                 const double *csrSortedValA, const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA, int blockDim,
                 const cusparseMatDescr_t descrC, double *bsrSortedValC,
                 int *bsrSortedRowPtrC, int *bsrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 */
cusparseStatus_t
cusparseCcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                 int n, const cusparseMatDescr_t descrA,
                 const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA, int blockDim,
                 const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
                 int *bsrSortedRowPtrC, int *bsrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 */
cusparseStatus_t cusparseZcsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t
cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nb, const cusparseMatDescr_t descrA,
                 const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA, int blockDim,
                 const cusparseMatDescr_t descrC, float *csrSortedValC,
                 int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t
cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nb, const cusparseMatDescr_t descrA,
                 const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA, int blockDim,
                 const cusparseMatDescr_t descrC, double *csrSortedValC,
                 int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t
cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                 int nb, const cusparseMatDescr_t descrA,
                 const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA, int blockDim,
                 const cusparseMatDescr_t descrC, cuComplex *csrSortedValC,
                 int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t cusparseZbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb,
                                int nnzb, const float *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int rowBlockDim,
                                int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb,
                                int nnzb, const double *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int rowBlockDim,
                                int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb,
                                int nnzb, const cuComplex *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int rowBlockDim,
                                int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb,
                                int nnzb, const cuDoubleComplex *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int rowBlockDim,
                                int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param bscVal SEND_ONLY
 * @param bscRowInd SEND_ONLY
 * @param bscColPtr SEND_ONLY
 * @param copyValues SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                     const float *bsrSortedVal, const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd, int rowBlockDim,
                     int colBlockDim, float *bscVal, int *bscRowInd,
                     int *bscColPtr, cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param bscVal SEND_ONLY
 * @param bscRowInd SEND_ONLY
 * @param bscColPtr SEND_ONLY
 * @param copyValues SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                     const double *bsrSortedVal, const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd, int rowBlockDim,
                     int colBlockDim, double *bscVal, int *bscRowInd,
                     int *bscColPtr, cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param bscVal SEND_ONLY
 * @param bscRowInd SEND_ONLY
 * @param bscColPtr SEND_ONLY
 * @param copyValues SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                     const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd, int rowBlockDim,
                     int colBlockDim, cuComplex *bscVal, int *bscRowInd,
                     int *bscColPtr, cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param bsrSortedVal SEND_ONLY
 * @param bsrSortedRowPtr SEND_ONLY
 * @param bsrSortedColInd SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param bscVal SEND_ONLY
 * @param bscRowInd SEND_ONLY
 * @param bscColPtr SEND_ONLY
 * @param copyValues SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    cuDoubleComplex *bscVal, int *bscRowInd, int *bscColPtr,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t cusparseSgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t cusparseDgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, double *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t cusparseCgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, cuComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 */
cusparseStatus_t cusparseZgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
    int *csrSortedColIndC);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseScsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                      int n, const cusparseMatDescr_t descrA,
                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                      const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC,
                      int rowBlockDim, int colBlockDim, int *nnzTotalDevHostPtr,
                      void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                   int n, const cusparseMatDescr_t descrA,
                   const float *csrSortedValA, const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA, const cusparseMatDescr_t descrC,
                   float *bsrSortedValC, int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC, int rowBlockDim, int colBlockDim,
                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                   int n, const cusparseMatDescr_t descrA,
                   const double *csrSortedValA, const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA, const cusparseMatDescr_t descrC,
                   double *bsrSortedValC, int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC, int rowBlockDim, int colBlockDim,
                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                   int n, const cusparseMatDescr_t descrA,
                   const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA, const cusparseMatDescr_t descrC,
                   cuComplex *bsrSortedValC, int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC, int rowBlockDim, int colBlockDim,
                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
    int colBlockDim, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseXgebsr2gebsrNnz(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
    const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC, int rowBlockDimC,
    int colBlockDimC, int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, float *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, double *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param dirA SEND_ONLY
 * @param mb SEND_ONLY
 * @param nb SEND_ONLY
 * @param nnzb SEND_ONLY
 * @param descrA SEND_ONLY
 * @param bsrSortedValA SEND_ONLY
 * @param bsrSortedRowPtrA SEND_ONLY
 * @param bsrSortedColIndA SEND_ONLY
 * @param rowBlockDimA SEND_ONLY
 * @param colBlockDimA SEND_ONLY
 * @param descrC SEND_ONLY
 * @param bsrSortedValC SEND_ONLY
 * @param bsrSortedRowPtrC SEND_ONLY
 * @param bsrSortedColIndC SEND_ONLY
 * @param rowBlockDimC SEND_ONLY
 * @param colBlockDimC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC,
    cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param p SEND_ONLY
 */
cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle,
                                                   int n, int *p);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cooRowsA SEND_ONLY
 * @param cooColsA SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m,
                                                int n, int nnz,
                                                const int *cooRowsA,
                                                const int *cooColsA,
                                                size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cooRowsA SEND_ONLY
 * @param cooColsA SEND_ONLY
 * @param P SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n,
                                       int nnz, int *cooRowsA, int *cooColsA,
                                       int *P, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cooRowsA SEND_ONLY
 * @param cooColsA SEND_ONLY
 * @param P SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n,
                                          int nnz, int *cooRowsA, int *cooColsA,
                                          int *P, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m,
                                                int n, int nnz,
                                                const int *csrRowPtrA,
                                                const int *csrColIndA,
                                                size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param P SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n,
                                  int nnz, const cusparseMatDescr_t descrA,
                                  const int *csrRowPtrA, int *csrColIndA,
                                  int *P, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cscColPtrA SEND_ONLY
 * @param cscRowIndA SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m,
                                                int n, int nnz,
                                                const int *cscColPtrA,
                                                const int *cscRowIndA,
                                                size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param cscColPtrA SEND_ONLY
 * @param cscRowIndA SEND_ONLY
 * @param P SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseXcscsort(cusparseHandle_t handle, int m, int n,
                                  int nnz, const cusparseMatDescr_t descrA,
                                  const int *cscColPtrA, int *cscRowIndA,
                                  int *P, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle, int m,
                                                 int n, int nnz, float *csrVal,
                                                 const int *csrRowPtr,
                                                 int *csrColInd,
                                                 csru2csrInfo_t info,
                                                 size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m,
                                                 int n, int nnz, double *csrVal,
                                                 const int *csrRowPtr,
                                                 int *csrColInd,
                                                 csru2csrInfo_t info,
                                                 size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                cuComplex *csrVal, const int *csrRowPtr,
                                int *csrColInd, csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                cuDoubleComplex *csrVal, const int *csrRowPtr,
                                int *csrColInd, csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsru2csr(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   float *csrVal, const int *csrRowPtr,
                                   int *csrColInd, csru2csrInfo_t info,
                                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsru2csr(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   double *csrVal, const int *csrRowPtr,
                                   int *csrColInd, csru2csrInfo_t info,
                                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsru2csr(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   cuComplex *csrVal, const int *csrRowPtr,
                                   int *csrColInd, csru2csrInfo_t info,
                                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsru2csr(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   cuDoubleComplex *csrVal,
                                   const int *csrRowPtr, int *csrColInd,
                                   csru2csrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsr2csru(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   float *csrVal, const int *csrRowPtr,
                                   int *csrColInd, csru2csrInfo_t info,
                                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsr2csru(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   double *csrVal, const int *csrRowPtr,
                                   int *csrColInd, csru2csrInfo_t info,
                                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsr2csru(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   cuComplex *csrVal, const int *csrRowPtr,
                                   int *csrColInd, csru2csrInfo_t info,
                                   void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsr2csru(cusparseHandle_t handle, int m, int n,
                                   int nnz, const cusparseMatDescr_t descrA,
                                   cuDoubleComplex *csrVal,
                                   const int *csrRowPtr, int *csrColInd,
                                   csru2csrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseHpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda,
    const __half *threshold, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    const float *threshold, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseHpruneDense2csrNnz(cusparseHandle_t handle, int m, int n,
                           const __half *A, int lda, const __half *threshold,
                           const cusparseMatDescr_t descrC, int *csrRowPtrC,
                           int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpruneDense2csrNnz(cusparseHandle_t handle, int m, int n,
                           const float *A, int lda, const float *threshold,
                           const cusparseMatDescr_t descrC, int *csrRowPtrC,
                           int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneDense2csrNnz(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseHpruneDense2csr(cusparseHandle_t handle, int m, int n,
                                         const __half *A, int lda,
                                         const __half *threshold,
                                         const cusparseMatDescr_t descrC,
                                         __half *csrSortedValC,
                                         const int *csrSortedRowPtrC,
                                         int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpruneDense2csr(cusparseHandle_t handle, int m, int n,
                                         const float *A, int lda,
                                         const float *threshold,
                                         const cusparseMatDescr_t descrC,
                                         float *csrSortedValC,
                                         const int *csrSortedRowPtrC,
                                         int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneDense2csr(cusparseHandle_t handle, int m, int n,
                                         const double *A, int lda,
                                         const double *threshold,
                                         const cusparseMatDescr_t descrC,
                                         double *csrSortedValC,
                                         const int *csrSortedRowPtrC,
                                         int *csrSortedColIndC, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseHpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const __half *threshold, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseHpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const __half *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseHpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA,
                      const cusparseMatDescr_t descrA,
                      const __half *csrSortedValA, const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA, const __half *threshold,
                      const cusparseMatDescr_t descrC, __half *csrSortedValC,
                      const int *csrSortedRowPtrC, int *csrSortedColIndC,
                      void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA,
                      const cusparseMatDescr_t descrA,
                      const float *csrSortedValA, const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA, const float *threshold,
                      const cusparseMatDescr_t descrC, float *csrSortedValC,
                      const int *csrSortedRowPtrC, int *csrSortedColIndC,
                      void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param threshold SEND_ONLY SCALAR
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseDpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA,
                      const cusparseMatDescr_t descrA,
                      const double *csrSortedValA, const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA, const double *threshold,
                      const cusparseMatDescr_t descrC, double *csrSortedValC,
                      const int *csrSortedRowPtrC, int *csrSortedColIndC,
                      void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseHpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda,
    float percentage, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseHpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseHpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, __half *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const cusparseMatDescr_t descrC, double *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, const __half *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, const double *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseHpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseHpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, __half *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param percentage SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const cusparseMatDescr_t descrC, double *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param cscVal SEND_ONLY
 * @param cscColPtr SEND_ONLY
 * @param cscRowInd SEND_ONLY
 * @param valType SEND_ONLY
 * @param copyValues SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param alg SEND_ONLY
 * @param buffer SEND_ONLY
 */
cusparseStatus_t cusparseCsr2cscEx2(
    cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
    const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr,
    int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void *buffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param cscVal SEND_ONLY
 * @param cscColPtr SEND_ONLY
 * @param cscRowInd SEND_ONLY
 * @param valType SEND_ONLY
 * @param copyValues SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseCsr2cscEx2_bufferSize(
    cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
    const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr,
    int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t *bufferSize);
/**
 * @param spVecDescr RECV_ONLY
 * @param size SEND_ONLY
 * @param nnz SEND_ONLY
 * @param indices SEND_ONLY
 * @param values SEND_ONLY
 * @param idxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spVecDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spVecDescr RECV_ONLY
 * @param size SEND_ONLY
 * @param nnz SEND_ONLY
 * @param indices SEND_ONLY
 * @param values SEND_ONLY
 * @param idxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstSpVec(cusparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, const void* indices, const void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spVecDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param spVecDescr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(spVecDescr);
  return return_value;
}
// clang-format on
/**
 * @param spVecDescr SEND_ONLY
 * @param size RECV_ONLY
 * @param nnz RECV_ONLY
 * @param indices RECV_ONLY
 * @param values RECV_ONLY
 * @param idxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr,
                                  int64_t *size, int64_t *nnz, void **indices,
                                  void **values, cusparseIndexType_t *idxType,
                                  cusparseIndexBase_t *idxBase,
                                  cudaDataType *valueType);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spVecDescr SEND_ONLY
 * @param size RECV_ONLY
 * @param nnz RECV_ONLY
 * @param indices RECV_ONLY
 * @param values RECV_ONLY
 * @param idxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t
cusparseConstSpVecGet(cusparseConstSpVecDescr_t spVecDescr, int64_t *size,
                      int64_t *nnz, const void **indices, const void **values,
                      cusparseIndexType_t *idxType,
                      cusparseIndexBase_t *idxBase, cudaDataType *valueType);
#endif
/**
 * @param spVecDescr SEND_ONLY
 * @param idxBase RECV_ONLY
 */
cusparseStatus_t cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr,
                                           cusparseIndexBase_t *idxBase);
/**
 * @param spVecDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr,
                                        void **values);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spVecDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t
cusparseConstSpVecGetValues(cusparseConstSpVecDescr_t spVecDescr,
                            const void **values);
#endif
/**
 * @param spVecDescr SEND_ONLY
 * @param values SEND_ONLY
 */
cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr,
                                        void *values);
/**
 * @param dnVecDescr RECV_ONLY
 * @param size SEND_ONLY
 * @param values SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *dnVecDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param dnVecDescr RECV_ONLY
 * @param size SEND_ONLY
 * @param values SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstDnVec(cusparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *dnVecDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param dnVecDescr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(dnVecDescr);
  return return_value;
}
// clang-format on
/**
 * @param dnVecDescr SEND_ONLY
 * @param size RECV_ONLY
 * @param values RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                                  int64_t *size, void **values,
                                  cudaDataType *valueType);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param dnVecDescr SEND_ONLY
 * @param size RECV_ONLY
 * @param values RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseConstDnVecGet(cusparseConstDnVecDescr_t dnVecDescr,
                                       int64_t *size, const void **values,
                                       cudaDataType *valueType);
#endif
/**
 * @param dnVecDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr,
                                        void **values);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param dnVecDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t
cusparseConstDnVecGetValues(cusparseConstDnVecDescr_t dnVecDescr,
                            const void **values);
#endif
/**
 * @param dnVecDescr SEND_ONLY
 * @param values SEND_ONLY
 */
cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr,
                                        void *values);
/**
 * @param spMatDescr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(spMatDescr);
  return return_value;
}
// clang-format on
/**
 * @param spMatDescr SEND_ONLY
 * @param format RECV_ONLY
 */
cusparseStatus_t cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr,
                                        cusparseFormat_t *format);
/**
 * @param spMatDescr SEND_ONLY
 * @param idxBase RECV_ONLY
 */
cusparseStatus_t cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr,
                                           cusparseIndexBase_t *idxBase);
/**
 * @param spMatDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr,
                                        void **values);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t
cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t spMatDescr,
                            const void **values);
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param values SEND_ONLY
 */
cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr,
                                        void *values);
/**
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 */
cusparseStatus_t cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr,
                                      int64_t *rows, int64_t *cols,
                                      int64_t *nnz);
/**
 * @param spMatDescr SEND_ONLY
 * @param batchCount RECV_ONLY
 */
cusparseStatus_t
cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr,
                             int *batchCount);
/**
 * @param spMatDescr SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 */
cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                            int batchCount,
                                            int64_t batchStride);
/**
 * @param spMatDescr SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param offsetsBatchStride SEND_ONLY
 * @param columnsValuesBatchStride SEND_ONLY
 */
cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                            int batchCount,
                                            int64_t offsetsBatchStride,
                                            int64_t columnsValuesBatchStride);
#if CUSPARSE_VERSION >= 12100
/**
 * @guard CUSPARSE_VERSION >= 12100
 * @param spMatDescr SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param offsetsBatchStride SEND_ONLY
 * @param columnsBatchStride SEND_ONLY
 * @param ValuesBatchStride SEND_ONLY
 */
cusparseStatus_t cusparseBsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                            int batchCount,
                                            int64_t offsetsBatchStride,
                                            int64_t columnsBatchStride,
                                            int64_t ValuesBatchStride);
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param attribute SEND_ONLY
 * @param data RECV_ONLY LENGTH:dataSize
 * @param dataSize SEND_ONLY
 */
cusparseStatus_t cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr,
                                           cusparseSpMatAttribute_t attribute,
                                           void *data, size_t dataSize);
/**
 * @param spMatDescr SEND_ONLY
 * @param attribute SEND_ONLY
 * @param data SEND_ONLY LENGTH:dataSize
 * @param dataSize SEND_ONLY
 */
cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                           cusparseSpMatAttribute_t attribute,
                                           void *data, size_t dataSize);
/**
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrRowOffsets SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param csrValues SEND_ONLY
 * @param csrRowOffsetsType SEND_ONLY
 * @param csrColIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param csrRowOffsets SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param csrValues SEND_ONLY
 * @param csrRowOffsetsType SEND_ONLY
 * @param csrColIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* csrRowOffsets, const void* csrColInd, const void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cscColOffsets SEND_ONLY
 * @param cscRowInd SEND_ONLY
 * @param cscValues SEND_ONLY
 * @param cscColOffsetsType SEND_ONLY
 * @param cscRowIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cscColOffsets SEND_ONLY
 * @param cscRowInd SEND_ONLY
 * @param cscValues SEND_ONLY
 * @param cscColOffsetsType SEND_ONLY
 * @param cscRowIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cscColOffsets, const void* cscRowInd, const void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param csrRowOffsets RECV_ONLY
 * @param csrColInd RECV_ONLY
 * @param csrValues RECV_ONLY
 * @param csrRowOffsetsType RECV_ONLY
 * @param csrColIndType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t
cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
               int64_t *nnz, void **csrRowOffsets, void **csrColInd,
               void **csrValues, cusparseIndexType_t *csrRowOffsetsType,
               cusparseIndexType_t *csrColIndType, cusparseIndexBase_t *idxBase,
               cudaDataType *valueType);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param csrRowOffsets RECV_ONLY
 * @param csrColInd RECV_ONLY
 * @param csrValues RECV_ONLY
 * @param csrRowOffsetsType RECV_ONLY
 * @param csrColIndType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t
cusparseConstCsrGet(cusparseConstSpMatDescr_t spMatDescr, int64_t *rows,
                    int64_t *cols, int64_t *nnz, const void **csrRowOffsets,
                    const void **csrColInd, const void **csrValues,
                    cusparseIndexType_t *csrRowOffsetsType,
                    cusparseIndexType_t *csrColIndType,
                    cusparseIndexBase_t *idxBase, cudaDataType *valueType);
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param cscColOffsets RECV_ONLY
 * @param cscRowInd RECV_ONLY
 * @param cscValues RECV_ONLY
 * @param cscColOffsetsType RECV_ONLY
 * @param cscRowIndType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t
cusparseCscGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
               int64_t *nnz, void **cscColOffsets, void **cscRowInd,
               void **cscValues, cusparseIndexType_t *cscColOffsetsType,
               cusparseIndexType_t *cscRowIndType, cusparseIndexBase_t *idxBase,
               cudaDataType *valueType);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param cscColOffsets RECV_ONLY
 * @param cscRowInd RECV_ONLY
 * @param cscValues RECV_ONLY
 * @param cscColOffsetsType RECV_ONLY
 * @param cscRowIndType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t
cusparseConstCscGet(cusparseConstSpMatDescr_t spMatDescr, int64_t *rows,
                    int64_t *cols, int64_t *nnz, const void **cscColOffsets,
                    const void **cscRowInd, const void **cscValues,
                    cusparseIndexType_t *cscColOffsetsType,
                    cusparseIndexType_t *cscRowIndType,
                    cusparseIndexBase_t *idxBase, cudaDataType *valueType);
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param csrRowOffsets SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param csrValues SEND_ONLY
 */
cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                        void *csrRowOffsets, void *csrColInd,
                                        void *csrValues);
/**
 * @param spMatDescr SEND_ONLY
 * @param cscColOffsets SEND_ONLY
 * @param cscRowInd SEND_ONLY
 * @param cscValues SEND_ONLY
 */
cusparseStatus_t cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr,
                                        void *cscColOffsets, void *cscRowInd,
                                        void *cscValues);
#if CUSPARSE_VERSION >= 12100
/**
 * @guard CUSPARSE_VERSION >= 12100
 * @param spMatDescr RECV_ONLY
 * @param brows SEND_ONLY
 * @param bcols SEND_ONLY
 * @param bnnz SEND_ONLY
 * @param rowBlockSize SEND_ONLY
 * @param colBlockSize SEND_ONLY
 * @param bsrRowOffsets SEND_ONLY
 * @param bsrColInd SEND_ONLY
 * @param bsrValues SEND_ONLY
 * @param bsrRowOffsetsType SEND_ONLY
 * @param bsrColIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 * @param order SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateBsr(cusparseSpMatDescr_t* spMatDescr, int64_t brows, int64_t bcols, int64_t bnnz, int64_t rowBlockSize, int64_t colBlockSize, void* bsrRowOffsets, void* bsrColInd, void* bsrValues, cusparseIndexType_t bsrRowOffsetsType, cusparseIndexType_t bsrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType, cusparseOrder_t order) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12100
/**
 * @guard CUSPARSE_VERSION >= 12100
 * @param spMatDescr RECV_ONLY
 * @param brows SEND_ONLY
 * @param bcols SEND_ONLY
 * @param bnnz SEND_ONLY
 * @param rowBlockDim SEND_ONLY
 * @param colBlockDim SEND_ONLY
 * @param bsrRowOffsets SEND_ONLY
 * @param bsrColInd SEND_ONLY
 * @param bsrValues SEND_ONLY
 * @param bsrRowOffsetsType SEND_ONLY
 * @param bsrColIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 * @param order SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstBsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t brows, int64_t bcols, int64_t bnnz, int64_t rowBlockDim, int64_t colBlockDim, const void* bsrRowOffsets, const void* bsrColInd, const void* bsrValues, cusparseIndexType_t bsrRowOffsetsType, cusparseIndexType_t bsrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType, cusparseOrder_t order) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cooRowInd SEND_ONLY
 * @param cooColInd SEND_ONLY
 * @param cooValues SEND_ONLY
 * @param cooIdxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cooRowInd SEND_ONLY
 * @param cooColInd SEND_ONLY
 * @param cooValues SEND_ONLY
 * @param cooIdxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstCoo(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cooRowInd, const void* cooColInd, const void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param cooRowInd RECV_ONLY
 * @param cooColInd RECV_ONLY
 * @param cooValues RECV_ONLY
 * @param idxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                                int64_t *cols, int64_t *nnz, void **cooRowInd,
                                void **cooColInd, void **cooValues,
                                cusparseIndexType_t *idxType,
                                cusparseIndexBase_t *idxBase,
                                cudaDataType *valueType);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param cooRowInd RECV_ONLY
 * @param cooColInd RECV_ONLY
 * @param cooValues RECV_ONLY
 * @param idxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t
cusparseConstCooGet(cusparseConstSpMatDescr_t spMatDescr, int64_t *rows,
                    int64_t *cols, int64_t *nnz, const void **cooRowInd,
                    const void **cooColInd, const void **cooValues,
                    cusparseIndexType_t *idxType, cusparseIndexBase_t *idxBase,
                    cudaDataType *valueType);
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param cooRows SEND_ONLY
 * @param cooColumns SEND_ONLY
 * @param cooValues SEND_ONLY
 */
cusparseStatus_t cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr,
                                        void *cooRows, void *cooColumns,
                                        void *cooValues);
/**
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ellBlockSize SEND_ONLY
 * @param ellCols SEND_ONLY
 * @param ellColInd SEND_ONLY
 * @param ellValue SEND_ONLY
 * @param ellIdxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateBlockedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ellBlockSize SEND_ONLY
 * @param ellCols SEND_ONLY
 * @param ellColInd SEND_ONLY
 * @param ellValue SEND_ONLY
 * @param ellIdxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstBlockedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, const void* ellColInd, const void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param ellBlockSize RECV_ONLY
 * @param ellCols RECV_ONLY
 * @param ellColInd RECV_ONLY
 * @param ellValue RECV_ONLY
 * @param ellIdxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr,
                                       int64_t *rows, int64_t *cols,
                                       int64_t *ellBlockSize, int64_t *ellCols,
                                       void **ellColInd, void **ellValue,
                                       cusparseIndexType_t *ellIdxType,
                                       cusparseIndexBase_t *idxBase,
                                       cudaDataType *valueType);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param ellBlockSize RECV_ONLY
 * @param ellCols RECV_ONLY
 * @param ellColInd RECV_ONLY
 * @param ellValue RECV_ONLY
 * @param ellIdxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseConstBlockedEllGet(
    cusparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *ellBlockSize, int64_t *ellCols, const void **ellColInd,
    const void **ellValue, cusparseIndexType_t *ellIdxType,
    cusparseIndexBase_t *idxBase, cudaDataType *valueType);
#endif
#if CUSPARSE_VERSION >= 12100
/**
 * @guard CUSPARSE_VERSION >= 12100
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param sellValuesSize SEND_ONLY
 * @param sliceSize SEND_ONLY
 * @param sellSliceOffsets SEND_ONLY
 * @param sellColInd SEND_ONLY
 * @param sellValues SEND_ONLY
 * @param sellSliceOffsetsType SEND_ONLY
 * @param sellColIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateSlicedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, void* sellSliceOffsets, void* sellColInd, void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12102
/**
 * @guard CUSPARSE_VERSION >= 12102
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param sellValuesSize SEND_ONLY
 * @param sliceSize SEND_ONLY
 * @param sellSliceOffsets SEND_ONLY
 * @param sellColInd SEND_ONLY
 * @param sellValues SEND_ONLY
 * @param sellSliceOffsetsType SEND_ONLY
 * @param sellColIndType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstSlicedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, const void* sellSliceOffsets, const void* sellColInd, const void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param dnMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ld SEND_ONLY
 * @param values SEND_ONLY
 * @param valueType SEND_ONLY
 * @param order SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *dnMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param dnMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ld SEND_ONLY
 * @param values SEND_ONLY
 * @param valueType SEND_ONLY
 * @param order SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateConstDnMat(cusparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, const void* values, cudaDataType valueType, cusparseOrder_t order) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *dnMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @param dnMatDescr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(dnMatDescr);
  return return_value;
}
// clang-format on
/**
 * @param dnMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param ld RECV_ONLY
 * @param values RECV_ONLY
 * @param type RECV_ONLY
 * @param order RECV_ONLY
 */
cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr,
                                  int64_t *rows, int64_t *cols, int64_t *ld,
                                  void **values, cudaDataType *type,
                                  cusparseOrder_t *order);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param dnMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param ld RECV_ONLY
 * @param values RECV_ONLY
 * @param type RECV_ONLY
 * @param order RECV_ONLY
 */
cusparseStatus_t cusparseConstDnMatGet(cusparseConstDnMatDescr_t dnMatDescr,
                                       int64_t *rows, int64_t *cols,
                                       int64_t *ld, const void **values,
                                       cudaDataType *type,
                                       cusparseOrder_t *order);
#endif
/**
 * @param dnMatDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr,
                                        void **values);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param dnMatDescr SEND_ONLY
 * @param values RECV_ONLY
 */
cusparseStatus_t
cusparseConstDnMatGetValues(cusparseConstDnMatDescr_t dnMatDescr,
                            const void **values);
#endif
/**
 * @param dnMatDescr SEND_ONLY
 * @param values SEND_ONLY
 */
cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr,
                                        void *values);
/**
 * @param dnMatDescr SEND_ONLY
 * @param batchCount SEND_ONLY
 * @param batchStride SEND_ONLY
 */
cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                                              int batchCount,
                                              int64_t batchStride);
/**
 * @param dnMatDescr SEND_ONLY
 * @param batchCount RECV_ONLY
 * @param batchStride RECV_ONLY
 */
cusparseStatus_t
cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr,
                             int *batchCount, int64_t *batchStride);
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:descriptor_value_width(vecY)
 * @param vecX SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:descriptor_value_width(vecY)
 * @param vecY SEND_ONLY
 */
cusparseStatus_t cusparseAxpby(cusparseHandle_t handle, const void *alpha,
                               cusparseConstSpVecDescr_t vecX, const void *beta,
                               cusparseDnVecDescr_t vecY);
/**
 * @param handle SEND_ONLY
 * @param vecY SEND_ONLY
 * @param vecX SEND_ONLY
 */
cusparseStatus_t cusparseGather(cusparseHandle_t handle,
                                cusparseConstDnVecDescr_t vecY,
                                cusparseSpVecDescr_t vecX);
/**
 * @param handle SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 */
cusparseStatus_t cusparseScatter(cusparseHandle_t handle,
                                 cusparseConstSpVecDescr_t vecX,
                                 cusparseDnVecDescr_t vecY);
/**
 * @param handle SEND_ONLY
 * @param c_coeff SEND_ONLY SCALAR SIZE:descriptor_value_width(vecX)
 * @param s_coeff SEND_ONLY SCALAR SIZE:descriptor_value_width(vecX)
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 */
cusparseStatus_t cusparseRot(cusparseHandle_t handle, const void *c_coeff,
                             const void *s_coeff, cusparseSpVecDescr_t vecX,
                             cusparseDnVecDescr_t vecY);
/**
 * @param handle SEND_ONLY
 * @param opX SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param result SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t
cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX,
                        cusparseConstSpVecDescr_t vecX,
                        cusparseConstDnVecDescr_t vecY, const void *result,
                        cudaDataType computeType, size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param opX SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param result RECV_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param computeType SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX,
                              cusparseConstSpVecDescr_t vecX,
                              cusparseConstDnVecDescr_t vecY, void *result,
                              cudaDataType computeType, void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSparseToDense_bufferSize(
    cusparseHandle_t handle, cusparseConstSpMatDescr_t matA,
    cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg,
    size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle,
                                       cusparseConstSpMatDescr_t matA,
                                       cusparseDnMatDescr_t matB,
                                       cusparseSparseToDenseAlg_t alg,
                                       void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseDenseToSparse_bufferSize(
    cusparseHandle_t handle, cusparseConstDnMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg,
    size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle,
                                                cusparseConstDnMatDescr_t matA,
                                                cusparseSpMatDescr_t matB,
                                                cusparseDenseToSparseAlg_t alg,
                                                void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t handle,
                                               cusparseConstDnMatDescr_t matA,
                                               cusparseSpMatDescr_t matB,
                                               cusparseDenseToSparseAlg_t alg,
                                               void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param vecY SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA,
                              const void *alpha, cusparseConstSpMatDescr_t matA,
                              cusparseConstDnVecDescr_t vecX, const void *beta,
                              cusparseDnVecDescr_t vecY,
                              cudaDataType computeType, cusparseSpMVAlg_t alg,
                              void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param vecY SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t
cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                        const void *alpha, cusparseConstSpMatDescr_t matA,
                        cusparseConstDnVecDescr_t vecX, const void *beta,
                        cusparseDnVecDescr_t vecY, cudaDataType computeType,
                        cusparseSpMVAlg_t alg, size_t *bufferSize);
#if CUSPARSE_VERSION >= 12300
/**
 * @guard CUSPARSE_VERSION >= 12300
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param vecY SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpMV_preprocess(cusparseHandle_t handle, cusparseOperation_t opA,
                        const void *alpha, cusparseConstSpMatDescr_t matA,
                        cusparseConstDnVecDescr_t vecX, const void *beta,
                        cusparseDnVecDescr_t vecY, cudaDataType computeType,
                        cusparseSpMVAlg_t alg, void *externalBuffer);
#endif
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param vecZ SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSpMVOp_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA,
    cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
    cusparseDnVecDescr_t vecY, cusparseDnVecDescr_t vecZ,
    cudaDataType computeType, cusparseSpMVOpAlg_t alg, size_t *bufferSize);
#endif
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param handle SEND_ONLY
 * @param desc RECV_ONLY
 * @param opA SEND_ONLY
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param vecZ SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param buffer SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpMVOp_createDescr(cusparseHandle_t handle, cusparseSpMVOpDescr_t* desc, cusparseOperation_t opA, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cusparseDnVecDescr_t vecZ, cudaDataType computeType, cusparseSpMVOpAlg_t alg, void* buffer) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *desc);
  }
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12702
/**
 * @guard CUSPARSE_VERSION >= 12702
 * @param desc SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpMVOp_destroyDescr(cusparseSpMVOpDescr_t desc) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(desc);
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12702
/**
 * @guard CUSPARSE_VERSION >= 12702
 * @param handle SEND_ONLY
 * @param desc SEND_ONLY
 * @param plan RECV_ONLY
 * @param code SEND_ONLY LENGTH:codeSize
 * @param codeSize SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpMVOp_createPlan(cusparseHandle_t handle, cusparseSpMVOpDescr_t desc, cusparseSpMVOpPlan_t* plan, const void* code, size_t codeSize) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *plan);
  }
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12702
/**
 * @guard CUSPARSE_VERSION >= 12702
 * @param plan SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpMVOp_destroyPlan(cusparseSpMVOpPlan_t plan) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(plan);
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12702
/**
 * @guard CUSPARSE_VERSION >= 12702
 * @param handle SEND_ONLY
 * @param plan SEND_ONLY
 * @param global_data_name SEND_ONLY NULL_TERMINATED
 * @param input_data SEND_ONLY LENGTH:data_size
 * @param data_size SEND_ONLY
 */
cusparseStatus_t cusparseSpMVOp_setGlobalUserData(cusparseHandle_t handle,
                                                  cusparseSpMVOpPlan_t plan,
                                                  const char *global_data_name,
                                                  void *input_data,
                                                  size_t data_size);
#endif
#if CUSPARSE_VERSION >= 12702
/**
 * @guard CUSPARSE_VERSION >= 12702
 * @param handle SEND_ONLY
 * @param plan SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:descriptor_value_width(vecZ)
 * @param beta SEND_ONLY SCALAR SIZE:descriptor_value_width(vecZ)
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param vecZ SEND_ONLY
 */
cusparseStatus_t cusparseSpMVOp(cusparseHandle_t handle,
                                cusparseSpMVOpPlan_t plan, const void *alpha,
                                const void *beta,
                                cusparseConstDnVecDescr_t vecX,
                                cusparseConstDnVecDescr_t vecY,
                                cusparseDnVecDescr_t vecZ);
#endif
/**
 * @param descr RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *descr);
  }
  return return_value;
}
// clang-format on
/**
 * @param descr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(descr);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spsvDescr SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSpSV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha,
    cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr, size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spsvDescr SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA,
                      const void *alpha, cusparseConstSpMatDescr_t matA,
                      cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                      cudaDataType computeType, cusparseSpSVAlg_t alg,
                      cusparseSpSVDescr_t spsvDescr, void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param vecX SEND_ONLY
 * @param vecY SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spsvDescr SEND_ONLY
 */
cusparseStatus_t
cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA,
                   const void *alpha, cusparseConstSpMatDescr_t matA,
                   cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                   cudaDataType computeType, cusparseSpSVAlg_t alg,
                   cusparseSpSVDescr_t spsvDescr);
#if CUSPARSE_VERSION >= 12100
/**
 * @guard CUSPARSE_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param spsvDescr SEND_ONLY
 * @param newValues SEND_ONLY
 * @param updatePart SEND_ONLY
 */
cusparseStatus_t cusparseSpSV_updateMatrix(cusparseHandle_t handle,
                                           cusparseSpSVDescr_t spsvDescr,
                                           void *newValues,
                                           cusparseSpSVUpdate_t updatePart);
#endif
/**
 * @param descr RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *descr);
  }
  return return_value;
}
// clang-format on
/**
 * @param descr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(descr);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spsmDescr SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSpSM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
    cudaDataType computeType, cusparseSpSMAlg_t alg,
    cusparseSpSMDescr_t spsmDescr, size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spsmDescr SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA,
                      cusparseOperation_t opB, const void *alpha,
                      cusparseConstSpMatDescr_t matA,
                      cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                      cudaDataType computeType, cusparseSpSMAlg_t alg,
                      cusparseSpSMDescr_t spsmDescr, void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spsmDescr SEND_ONLY
 */
cusparseStatus_t
cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA,
                   cusparseOperation_t opB, const void *alpha,
                   cusparseConstSpMatDescr_t matA,
                   cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                   cudaDataType computeType, cusparseSpSMAlg_t alg,
                   cusparseSpSMDescr_t spsmDescr);
#if CUSPARSE_VERSION >= 12300
/**
 * @guard CUSPARSE_VERSION >= 12300
 * @param handle SEND_ONLY
 * @param spsmDescr SEND_ONLY
 * @param newValues SEND_ONLY
 * @param updatePart SEND_ONLY
 */
cusparseStatus_t cusparseSpSM_updateMatrix(cusparseHandle_t handle,
                                           cusparseSpSMDescr_t spsmDescr,
                                           void *newValues,
                                           cusparseSpSMUpdate_t updatePart);
#endif
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSpMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstDnMatDescr_t matB, const void *beta, cusparseDnMatDescr_t matC,
    cudaDataType computeType, cusparseSpMMAlg_t alg, size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstDnMatDescr_t matB, const void *beta, cusparseDnMatDescr_t matC,
    cudaDataType computeType, cusparseSpMMAlg_t alg, void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void *alpha,
                              cusparseConstSpMatDescr_t matA,
                              cusparseConstDnMatDescr_t matB, const void *beta,
                              cusparseDnMatDescr_t matC,
                              cudaDataType computeType, cusparseSpMMAlg_t alg,
                              void *externalBuffer);
/**
 * @param descr RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *descr);
  }
  return return_value;
}
// clang-format on
/**
 * @param descr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(descr);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 * @param bufferSize1 SEND_RECV
 * @param externalBuffer1 SEND_ONLY
 */
cusparseStatus_t cusparseSpGEMM_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize1,
    void *externalBuffer1);
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param spgemmDescr SEND_ONLY
 * @param num_prods RECV_ONLY
 */
cusparseStatus_t
cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr_t spgemmDescr,
                              int64_t *num_prods);
#endif
#if CUSPARSE_VERSION >= 12000
/**
 * @guard CUSPARSE_VERSION >= 12000
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 * @param chunk_fraction SEND_ONLY
 * @param bufferSize3 SEND_RECV
 * @param externalBuffer3 SEND_ONLY
 * @param bufferSize2 SEND_RECV NULLABLE
 */
cusparseStatus_t cusparseSpGEMM_estimateMemory(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, float chunk_fraction,
    size_t *bufferSize3, void *externalBuffer3, size_t *bufferSize2);
#endif
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 * @param bufferSize2 SEND_RECV
 * @param externalBuffer2 SEND_ONLY
 */
cusparseStatus_t cusparseSpGEMM_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2,
    void *externalBuffer2);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 */
cusparseStatus_t
cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA,
                    cusparseOperation_t opB, const void *alpha,
                    cusparseConstSpMatDescr_t matA,
                    cusparseConstSpMatDescr_t matB, const void *beta,
                    cusparseSpMatDescr_t matC, cudaDataType computeType,
                    cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 * @param bufferSize1 SEND_RECV
 * @param externalBuffer1 SEND_ONLY
 */
cusparseStatus_t cusparseSpGEMMreuse_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
    cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize1,
    void *externalBuffer1);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 * @param bufferSize2 SEND_RECV
 * @param externalBuffer2 SEND_ONLY
 * @param bufferSize3 SEND_RECV
 * @param externalBuffer3 SEND_ONLY
 * @param bufferSize4 SEND_RECV
 * @param externalBuffer4 SEND_ONLY
 */
cusparseStatus_t cusparseSpGEMMreuse_nnz(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
    cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2,
    void *externalBuffer2, size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize4, void *externalBuffer4);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 * @param bufferSize5 SEND_RECV
 * @param externalBuffer5 SEND_ONLY
 */
cusparseStatus_t cusparseSpGEMMreuse_copy(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
    cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize5,
    void *externalBuffer5);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgemmDescr SEND_ONLY
 */
cusparseStatus_t cusparseSpGEMMreuse_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstSpMatDescr_t matA,
    cusparseConstSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr);
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param descr RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpGEAM_createDescr(cusparseSpGEAMDescr_t* descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *descr);
  }
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param descr SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpGEAM_destroyDescr(cusparseSpGEAMDescr_t descr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(descr);
  return return_value;
}
// clang-format on
#endif
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgeamDescr SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSpGEAM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseSpMatDescr_t matA, const void *beta,
    cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEAMAlg_t alg,
    cusparseSpGEAMDescr_t spgeamDescr, size_t *bufferSize);
#endif
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgeamDescr SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpGEAM_nnz(cusparseHandle_t handle, cusparseOperation_t opA,
                   cusparseOperation_t opB, const void *alpha,
                   cusparseSpMatDescr_t matA, const void *beta,
                   cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC,
                   cudaDataType computeType, cusparseSpGEAMAlg_t alg,
                   cusparseSpGEAMDescr_t spgeamDescr, void *externalBuffer);
#endif
#if CUSPARSE_VERSION >= 12801
/**
 * @guard CUSPARSE_VERSION >= 12801
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param spgeamDescr SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseSpGEAM(cusparseHandle_t handle, cusparseOperation_t opA,
               cusparseOperation_t opB, const void *alpha,
               cusparseSpMatDescr_t matA, const void *beta,
               cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC,
               cudaDataType computeType, cusparseSpGEAMAlg_t alg,
               cusparseSpGEAMDescr_t spgeamDescr, void *externalBuffer);
#endif
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseSDDMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstDnMatDescr_t matA,
    cusparseConstDnMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t *bufferSize);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSDDMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseConstDnMatDescr_t matA,
    cusparseConstDnMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSDDMMAlg_t alg, void *externalBuffer);
/**
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA,
                               cusparseOperation_t opB, const void *alpha,
                               cusparseConstDnMatDescr_t matA,
                               cusparseConstDnMatDescr_t matB, const void *beta,
                               cusparseSpMatDescr_t matC,
                               cudaDataType computeType, cusparseSDDMMAlg_t alg,
                               void *externalBuffer);
// clang-format off
/**
 * @param handle SEND_ONLY
 * @param plan RECV_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param alg SEND_ONLY
 * @param addOperationLtoirBuffer SEND_ONLY NULLABLE LENGTH:addOperationBufferSize
 * @param addOperationBufferSize SEND_ONLY
 * @param mulOperationLtoirBuffer SEND_ONLY NULLABLE LENGTH:mulOperationBufferSize
 * @param mulOperationBufferSize SEND_ONLY
 * @param epilogueLtoirBuffer SEND_ONLY NULLABLE LENGTH:epilogueBufferSize
 * @param epilogueBufferSize SEND_ONLY
 * @param SpMMWorkspaceSize RECV_ONLY
 */
cusparseStatus_t cusparseSpMMOp_createPlan(cusparseHandle_t handle, cusparseSpMMOpPlan_t* plan, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMOpAlg_t alg, const void* addOperationLtoirBuffer, size_t addOperationBufferSize, const void* mulOperationLtoirBuffer, size_t mulOperationBufferSize, const void* epilogueLtoirBuffer, size_t epilogueBufferSize, size_t* SpMMWorkspaceSize) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *plan);
  }
  return return_value;
}
// clang-format on
/**
 * @param plan SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t cusparseSpMMOp(cusparseSpMMOpPlan_t plan,
                                void *externalBuffer);
/**
 * @param plan SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseSpMMOp_destroyPlan(cusparseSpMMOpPlan_t plan) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(plan);
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseCaxpyi(cusparseHandle_t handle, int nnz,
                                const cuComplex *alpha, const cuComplex *xVal,
                                const int *xInd, cuComplex *y,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseCcsc2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *cscSortedValA,
                                    const int *cscSortedRowIndA,
                                    const int *cscSortedColPtrA, cuComplex *A,
                                    int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseCcsr2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA, cuComplex *A,
                                    int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedValD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cuComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const cuComplex *csrSortedValD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    cuComplex *csrSortedValC, const int *csrSortedRowPtrC,
    int *csrSortedColIndC, const csrgemm2Info_t info, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cuComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseCcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, size_t *pBufferSize);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCcsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseCcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const cuComplex *f,
    cuComplex *x, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerCol SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 */
cusparseStatus_t
cusparseCdense2csc(cusparseHandle_t handle, int m, int n,
                   const cusparseMatDescr_t descrA, const cuComplex *A, int lda,
                   const int *nnzPerCol, cuComplex *cscSortedValA,
                   int *cscSortedRowIndA, int *cscSortedColPtrA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 */
cusparseStatus_t
cusparseCdense2csr(cusparseHandle_t handle, int m, int n,
                   const cusparseMatDescr_t descrA, const cuComplex *A, int lda,
                   const int *nnzPerRow, cuComplex *csrSortedValA,
                   int *csrSortedRowPtrA, int *csrSortedColIndA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param cscValB SEND_ONLY
 * @param cscColPtrB SEND_ONLY
 * @param cscRowIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t cusparseCgemmi(cusparseHandle_t handle, int m, int n, int k,
                                int nnz, const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                const cuComplex *cscValB, const int *cscColPtrB,
                                const int *cscRowIndB, const cuComplex *beta,
                                cuComplex *C, int ldc);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseCgthr(cusparseHandle_t handle, int nnz,
                               const cuComplex *y, cuComplex *xVal,
                               const int *xInd, cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseCgthrz(cusparseHandle_t handle, int nnz, cuComplex *y,
                                cuComplex *xVal, const int *xInd,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param externalBuffer SEND_ONLY
 */
cusparseStatus_t
cusparseConstrainedGeMM(cusparseHandle_t handle, cusparseOperation_t opA,
                        cusparseOperation_t opB, const void *alpha,
                        cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
                        const void *beta, cusparseSpMatDescr_t matC,
                        cudaDataType computeType, void *externalBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matA SEND_ONLY
 * @param matB SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(computeType)
 * @param matC SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
cusparseStatus_t cusparseConstrainedGeMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
    const void *beta, cusparseSpMatDescr_t matC, cudaDataType computeType,
    size_t *bufferSize);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param spMatDescr SEND_ONLY
 * @param rows RECV_ONLY
 * @param cols RECV_ONLY
 * @param nnz RECV_ONLY
 * @param cooInd RECV_ONLY
 * @param cooValues RECV_ONLY
 * @param idxType RECV_ONLY
 * @param idxBase RECV_ONLY
 * @param valueType RECV_ONLY
 */
cusparseStatus_t cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr,
                                   int64_t *rows, int64_t *cols, int64_t *nnz,
                                   void **cooInd, void **cooValues,
                                   cusparseIndexType_t *idxType,
                                   cusparseIndexBase_t *idxBase,
                                   cudaDataType *valueType);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param spMatDescr RECV_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param nnz SEND_ONLY
 * @param cooInd SEND_ONLY
 * @param cooValues SEND_ONLY
 * @param cooIdxType SEND_ONLY
 * @param idxBase SEND_ONLY
 * @param valueType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *spMatDescr, valueType);
  }
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsrgemm2Info(csrgemm2Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsrsm2Info(csrsm2Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param info RECV_ONLY
 */
// clang-format off
cusparseStatus_t cusparseCreateCsrsv2Info(csrsv2Info_t* info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseCsctr(cusparseHandle_t handle, int nnz,
                               const cuComplex *xVal, const int *xInd,
                               cuComplex *y, cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param alg SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(alphatype)
 * @param alphatype SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrValAtype SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param x SEND_ONLY
 * @param xtype SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(betatype)
 * @param betatype SEND_ONLY
 * @param y SEND_ONLY
 * @param ytype SEND_ONLY
 * @param executiontype SEND_ONLY
 * @param buffer SEND_ONLY
 */
cusparseStatus_t cusparseCsrmvEx(
    cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
    int m, int n, int nnz, const void *alpha, cudaDataType alphatype,
    const cusparseMatDescr_t descrA, const void *csrValA,
    cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
    const void *x, cudaDataType xtype, const void *beta, cudaDataType betatype,
    void *y, cudaDataType ytype, cudaDataType executiontype, void *buffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param alg SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR SIZE:data_type_width(alphatype)
 * @param alphatype SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrValAtype SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param x SEND_ONLY
 * @param xtype SEND_ONLY
 * @param beta SEND_ONLY SCALAR SIZE:data_type_width(betatype)
 * @param betatype SEND_ONLY
 * @param y SEND_ONLY
 * @param ytype SEND_ONLY
 * @param executiontype SEND_ONLY
 * @param bufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseCsrmvEx_bufferSize(
    cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA,
    int m, int n, int nnz, const void *alpha, cudaDataType alphatype,
    const cusparseMatDescr_t descrA, const void *csrValA,
    cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
    const void *x, cudaDataType xtype, const void *beta, cudaDataType betatype,
    void *y, cudaDataType ytype, cudaDataType executiontype,
    size_t *bufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseDaxpyi(cusparseHandle_t handle, int nnz,
                                const double *alpha, const double *xVal,
                                const int *xInd, double *y,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseDcsc2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *cscSortedValA,
                                    const int *cscSortedRowIndA,
                                    const int *cscSortedColPtrA, double *A,
                                    int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseDcsr2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA, double *A,
                                    int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedValD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const double *beta, const cusparseMatDescr_t descrD, int nnzD,
    const double *csrSortedValD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseDcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const double *beta, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseDcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
    size_t *pBufferSize);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, double *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseDcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA,
                           int m, int nnz, const cusparseMatDescr_t descrA,
                           double *csrSortedValA, const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA, csrsv2Info_t info,
                           int *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseDcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const double *alpha, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const double *f, double *x,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerCol SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 */
cusparseStatus_t cusparseDdense2csc(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *A, int lda,
                                    const int *nnzPerCol, double *cscSortedValA,
                                    int *cscSortedRowIndA,
                                    int *cscSortedColPtrA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 */
cusparseStatus_t cusparseDdense2csr(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const double *A, int lda,
                                    const int *nnzPerRow, double *csrSortedValA,
                                    int *csrSortedRowPtrA,
                                    int *csrSortedColIndA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyCsrgemm2Info(csrgemm2Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyCsrsm2Info(csrsm2Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param info SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseDestroyCsrsv2Info(csrsv2Info_t info) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param cscValB SEND_ONLY
 * @param cscColPtrB SEND_ONLY
 * @param cscRowIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t cusparseDgemmi(cusparseHandle_t handle, int m, int n, int k,
                                int nnz, const double *alpha, const double *A,
                                int lda, const double *cscValB,
                                const int *cscColPtrB, const int *cscRowIndB,
                                const double *beta, double *C, int ldc);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseDgthr(cusparseHandle_t handle, int nnz,
                               const double *y, double *xVal, const int *xInd,
                               cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseDgthrz(cusparseHandle_t handle, int nnz, double *y,
                                double *xVal, const int *xInd,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseDroti(cusparseHandle_t handle, int nnz, double *xVal,
                               const int *xInd, double *y, const double *c,
                               const double *s, cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseDsctr(cusparseHandle_t handle, int nnz,
                               const double *xVal, const int *xInd, double *y,
                               cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseSaxpyi(cusparseHandle_t handle, int nnz,
                                const float *alpha, const float *xVal,
                                const int *xInd, float *y,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseScsc2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *cscSortedValA,
                                    const int *cscSortedRowIndA,
                                    const int *cscSortedColPtrA, float *A,
                                    int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseScsr2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA, float *A,
                                    int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedValD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const float *beta,
    const cusparseMatDescr_t descrD, int nnzD, const float *csrSortedValD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const float *beta,
    const cusparseMatDescr_t descrD, int nnzD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseScsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
    int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
    size_t *pBufferSize);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float *B, int ldb,
    csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t
cusparseScsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA,
                           int m, int nnz, const cusparseMatDescr_t descrA,
                           float *csrSortedValA, const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA, csrsv2Info_t info,
                           int *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseScsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const float *alpha, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const float *f, float *x,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerCol SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 */
cusparseStatus_t cusparseSdense2csc(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *A, int lda,
                                    const int *nnzPerCol, float *cscSortedValA,
                                    int *cscSortedRowIndA,
                                    int *cscSortedColPtrA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 */
cusparseStatus_t cusparseSdense2csr(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const float *A, int lda,
                                    const int *nnzPerRow, float *csrSortedValA,
                                    int *csrSortedRowPtrA,
                                    int *csrSortedColIndA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param cscValB SEND_ONLY
 * @param cscColPtrB SEND_ONLY
 * @param cscRowIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t cusparseSgemmi(cusparseHandle_t handle, int m, int n, int k,
                                int nnz, const float *alpha, const float *A,
                                int lda, const float *cscValB,
                                const int *cscColPtrB, const int *cscRowIndB,
                                const float *beta, float *C, int ldc);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseSgthr(cusparseHandle_t handle, int nnz, const float *y,
                               float *xVal, const int *xInd,
                               cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseSgthrz(cusparseHandle_t handle, int nnz, float *y,
                                float *xVal, const int *xInd,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param spMatDescr SEND_ONLY
 * @param batchCount SEND_ONLY
 */
cusparseStatus_t cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                              int batchCount);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param c SEND_ONLY SCALAR
 * @param s SEND_ONLY SCALAR
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseSroti(cusparseHandle_t handle, int nnz, float *xVal,
                               const int *xInd, float *y, const float *c,
                               const float *s, cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseSsctr(cusparseHandle_t handle, int nnz,
                               const float *xVal, const int *xInd, float *y,
                               cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param nnzTotalDevHostPtr RECV_ONLY SCALAR
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseXcsrgemm2Nnz(
    cusparseHandle_t handle, int m, int n, int k,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cusparseMatDescr_t descrD, int nnzD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, const csrgemm2Info_t info,
    void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                                           csrsm2Info_t info, int *position);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param position RECV_ONLY SCALAR
 */
cusparseStatus_t cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                                           csrsv2Info_t info, int *position);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseZaxpyi(cusparseHandle_t handle, int nnz,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *xVal, const int *xInd,
                                cuDoubleComplex *y,
                                cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseZcsc2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *cscSortedValA,
                                    const int *cscSortedRowIndA,
                                    const int *cscSortedColPtrA,
                                    cuDoubleComplex *A, int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 */
cusparseStatus_t cusparseZcsr2dense(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA,
                                    cuDoubleComplex *A, int lda);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedValB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedValD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param descrC SEND_ONLY
 * @param csrSortedValC SEND_ONLY
 * @param csrSortedRowPtrC SEND_ONLY
 * @param csrSortedColIndC SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrD, int nnzD,
    const cuDoubleComplex *csrSortedValD, const int *csrSortedRowPtrD,
    const int *csrSortedColIndD, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC,
    int *csrSortedColIndC, const csrgemm2Info_t info, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param descrB SEND_ONLY
 * @param nnzB SEND_ONLY
 * @param csrSortedRowPtrB SEND_ONLY
 * @param csrSortedColIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param descrD SEND_ONLY
 * @param nnzD SEND_ONLY
 * @param csrSortedRowPtrD SEND_ONLY
 * @param csrSortedColIndD SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const cuDoubleComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb,
    csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseZcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb,
    csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t *pBufferSize);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param transA SEND_ONLY
 * @param transB SEND_ONLY
 * @param m SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, cuDoubleComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBufferSizeInBytes RECV_ONLY
 */
cusparseStatus_t cusparseZcsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param transA SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param descrA SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param f SEND_ONLY
 * @param x SEND_ONLY
 * @param policy SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusparseStatus_t cusparseZcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const cuDoubleComplex *f,
    cuDoubleComplex *x, cusparseSolvePolicy_t policy, void *pBuffer);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerCol SEND_ONLY
 * @param cscSortedValA SEND_ONLY
 * @param cscSortedRowIndA SEND_ONLY
 * @param cscSortedColPtrA SEND_ONLY
 */
cusparseStatus_t cusparseZdense2csc(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *A, int lda,
                                    const int *nnzPerCol,
                                    cuDoubleComplex *cscSortedValA,
                                    int *cscSortedRowIndA,
                                    int *cscSortedColPtrA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param descrA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param nnzPerRow SEND_ONLY
 * @param csrSortedValA SEND_ONLY
 * @param csrSortedRowPtrA SEND_ONLY
 * @param csrSortedColIndA SEND_ONLY
 */
cusparseStatus_t cusparseZdense2csr(cusparseHandle_t handle, int m, int n,
                                    const cusparseMatDescr_t descrA,
                                    const cuDoubleComplex *A, int lda,
                                    const int *nnzPerRow,
                                    cuDoubleComplex *csrSortedValA,
                                    int *csrSortedRowPtrA,
                                    int *csrSortedColIndA);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param nnz SEND_ONLY
 * @param alpha SEND_ONLY SCALAR
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param cscValB SEND_ONLY
 * @param cscColPtrB SEND_ONLY
 * @param cscRowIndB SEND_ONLY
 * @param beta SEND_ONLY SCALAR
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cusparseStatus_t cusparseZgemmi(cusparseHandle_t handle, int m, int n, int k,
                                int nnz, const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *cscValB,
                                const int *cscColPtrB, const int *cscRowIndB,
                                const cuDoubleComplex *beta, cuDoubleComplex *C,
                                int ldc);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseZgthr(cusparseHandle_t handle, int nnz,
                               const cuDoubleComplex *y, cuDoubleComplex *xVal,
                               const int *xInd, cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param y SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseZgthrz(cusparseHandle_t handle, int nnz,
                                cuDoubleComplex *y, cuDoubleComplex *xVal,
                                const int *xInd, cusparseIndexBase_t idxBase);
/**
 * @guard CUSPARSE_VERSION < 12000
 * @param handle SEND_ONLY
 * @param nnz SEND_ONLY
 * @param xVal SEND_ONLY
 * @param xInd SEND_ONLY
 * @param y SEND_ONLY
 * @param idxBase SEND_ONLY
 */
cusparseStatus_t cusparseZsctr(cusparseHandle_t handle, int nnz,
                               const cuDoubleComplex *xVal, const int *xInd,
                               cuDoubleComplex *y, cusparseIndexBase_t idxBase);
