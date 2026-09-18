#include <cusparseLt.h>

// cuSPARSELt API. Every call forwards over the CUDA driver shim's connections
// like cuBLAS and cuBLASLt; a handle routes to the connection that created it,
// and a call without one goes to the runtime's current device.
//
// A cuSPARSELt object (handle, matrix descriptor, matmul descriptor, algorithm
// selection, plan) is an opaque handle the caller keeps in storage of its own:
// the library fills 512 caller-owned bytes with state it links to its other
// objects by address, so a plan holds the addresses of the matmul descriptor
// and algorithm selection it was built from, and that descriptor holds the
// addresses of its matrix descriptors. The storage cannot travel, so the object
// lives on the server, where those addresses stay valid, and the caller's
// storage holds its address there (`REMOTE`). The initializing call allocates
// it and answers with its address, which the client records against the
// connection once the call succeeds; the matching Destroy frees it, and an
// object whose Destroy never arrives, a failed Init among them, is freed with
// the rest of the session when the connection's child process exits.
//
// alpha and beta are host scalars unless the matmul descriptor sets vector
// scaling, which makes them device addresses; the plan names the descriptor
// that decides (`SCALAR:plan`), and the client mirrors both attributes as
// descriptors are set. Every other pointer a call takes is device memory whose
// address travels unchanged, the pruning validity flags included.
//
// cusparseLtGetErrorName and cusparseLtGetErrorString return static strings
// rather than a status, so they are hand-written in cusparselt_client.cpp and
// cusparselt_server.cpp.

/**
 * @param handle RECV_ONLY REMOTE
 */
// clang-format off
cusparseStatus_t cusparseLtInit(cusparseLtHandle_t *handle) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_object(conn, handle);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 */
// clang-format off
cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t *handle) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_object(handle);
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param version RECV_ONLY
 */
cusparseStatus_t cusparseLtGetVersion(const cusparseLtHandle_t *handle,
                                      int *version);
/**
 * @param propertyType SEND_ONLY
 * @param value RECV_ONLY
 */
cusparseStatus_t cusparseLtGetProperty(libraryPropertyType propertyType,
                                       int *value);
/**
 * @disabled
 * @guard CUSPARSELT_VERSION >= 800
 */
const char *cusparseLtGetErrorName(cusparseStatus_t status);
/**
 * @disabled
 * @guard CUSPARSELT_VERSION >= 800
 */
const char *cusparseLtGetErrorString(cusparseStatus_t status);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matDescr RECV_ONLY REMOTE
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ld SEND_ONLY
 * @param alignment SEND_ONLY
 * @param valueType SEND_ONLY
 * @param order SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseLtDenseDescriptorInit(
    const cusparseLtHandle_t *handle, cusparseLtMatDescriptor_t *matDescr,
    int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
    cudaDataType valueType, cusparseOrder_t order) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_object(conn, matDescr);
  }
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matDescr RECV_ONLY REMOTE
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ld SEND_ONLY
 * @param alignment SEND_ONLY
 * @param valueType SEND_ONLY
 * @param order SEND_ONLY
 * @param sparsity SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseLtStructuredDescriptorInit(
    const cusparseLtHandle_t *handle, cusparseLtMatDescriptor_t *matDescr,
    int64_t rows, int64_t cols, int64_t ld, uint32_t alignment,
    cudaDataType valueType, cusparseOrder_t order,
    cusparseLtSparsity_t sparsity) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_object(conn, matDescr);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @routingkey HANDLE matDescr
 * @param matDescr SEND_ONLY REMOTE
 */
// clang-format off
cusparseStatus_t cusparseLtMatDescriptorDestroy(
    const cusparseLtMatDescriptor_t *matDescr) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_object(matDescr);
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param matAttribute SEND_ONLY
 * @param dataSize SEND_ONLY
 * @param data SEND_ONLY LENGTH:dataSize
 */
cusparseStatus_t
cusparseLtMatDescSetAttribute(const cusparseLtHandle_t *handle,
                              cusparseLtMatDescriptor_t *matmulDescr,
                              cusparseLtMatDescAttribute_t matAttribute,
                              const void *data, size_t dataSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param matAttribute SEND_ONLY
 * @param dataSize SEND_ONLY
 * @param data RECV_ONLY LENGTH:dataSize
 */
cusparseStatus_t
cusparseLtMatDescGetAttribute(const cusparseLtHandle_t *handle,
                              const cusparseLtMatDescriptor_t *matmulDescr,
                              cusparseLtMatDescAttribute_t matAttribute,
                              void *data, size_t dataSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr RECV_ONLY REMOTE
 * @param opA SEND_ONLY
 * @param opB SEND_ONLY
 * @param matA SEND_ONLY REMOTE
 * @param matB SEND_ONLY REMOTE
 * @param matC SEND_ONLY REMOTE
 * @param matD SEND_ONLY REMOTE
 * @param computeType SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseLtMatmulDescriptorInit(
    const cusparseLtHandle_t *handle,
    cusparseLtMatmulDescriptor_t *matmulDescr, cusparseOperation_t opA,
    cusparseOperation_t opB, const cusparseLtMatDescriptor_t *matA,
    const cusparseLtMatDescriptor_t *matB,
    const cusparseLtMatDescriptor_t *matC,
    const cusparseLtMatDescriptor_t *matD, cusparseComputeType computeType) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_object(conn, matmulDescr);
  }
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param matmulAttribute SEND_ONLY
 * @param dataSize SEND_ONLY
 * @param data SEND_ONLY LENGTH:dataSize
 */
// clang-format off
cusparseStatus_t cusparseLtMatmulDescSetAttribute(
    const cusparseLtHandle_t *handle,
    cusparseLtMatmulDescriptor_t *matmulDescr,
    cusparseLtMatmulDescAttribute_t matmulAttribute, const void *data,
    size_t dataSize) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_scaling(matmulDescr, matmulAttribute, data, dataSize);
  }
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param matmulAttribute SEND_ONLY
 * @param dataSize SEND_ONLY
 * @param data RECV_ONLY LENGTH:dataSize
 */
cusparseStatus_t cusparseLtMatmulDescGetAttribute(
    const cusparseLtHandle_t *handle,
    const cusparseLtMatmulDescriptor_t *matmulDescr,
    cusparseLtMatmulDescAttribute_t matmulAttribute, void *data,
    size_t dataSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param algSelection RECV_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param alg SEND_ONLY
 */
// clang-format off
cusparseStatus_t cusparseLtMatmulAlgSelectionInit(
    const cusparseLtHandle_t *handle,
    cusparseLtMatmulAlgSelection_t *algSelection,
    const cusparseLtMatmulDescriptor_t *matmulDescr,
    cusparseLtMatmulAlg_t alg) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_object(conn, algSelection);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @guard CUSPARSELT_VERSION >= 800
 * @routingkey HANDLE algSelection
 * @param algSelection SEND_ONLY REMOTE
 */
// clang-format off
cusparseStatus_t cusparseLtMatmulAlgSelectionDestroy(
    const cusparseLtMatmulAlgSelection_t *algSelection) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_object(algSelection);
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param algSelection SEND_ONLY REMOTE
 * @param attribute SEND_ONLY
 * @param dataSize SEND_ONLY
 * @param data SEND_ONLY LENGTH:dataSize
 */
cusparseStatus_t
cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t *handle,
                                cusparseLtMatmulAlgSelection_t *algSelection,
                                cusparseLtMatmulAlgAttribute_t attribute,
                                const void *data, size_t dataSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param algSelection SEND_ONLY REMOTE
 * @param attribute SEND_ONLY
 * @param dataSize SEND_ONLY
 * @param data RECV_ONLY LENGTH:dataSize
 */
cusparseStatus_t cusparseLtMatmulAlgGetAttribute(
    const cusparseLtHandle_t *handle,
    const cusparseLtMatmulAlgSelection_t *algSelection,
    cusparseLtMatmulAlgAttribute_t attribute, void *data, size_t dataSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param plan SEND_ONLY REMOTE
 * @param workspaceSize RECV_ONLY
 */
cusparseStatus_t
cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t *handle,
                             const cusparseLtMatmulPlan_t *plan,
                             size_t *workspaceSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param plan RECV_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param algSelection SEND_ONLY REMOTE
 */
// clang-format off
cusparseStatus_t cusparseLtMatmulPlanInit(
    const cusparseLtHandle_t *handle, cusparseLtMatmulPlan_t *plan,
    const cusparseLtMatmulDescriptor_t *matmulDescr,
    const cusparseLtMatmulAlgSelection_t *algSelection) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSPARSE_STATUS_SUCCESS) {
    note_object(conn, plan);
    note_plan_descriptor(plan, matmulDescr);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled server
 * @routingkey HANDLE plan
 * @param plan SEND_ONLY REMOTE
 */
// clang-format off
cusparseStatus_t cusparseLtMatmulPlanDestroy(
    const cusparseLtMatmulPlan_t *plan) {
  cusparseStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_object(plan);
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param plan SEND_ONLY REMOTE
 * @param alpha SEND_ONLY SCALAR:plan SIZE:sizeof(float)
 * @param d_A SEND_ONLY
 * @param d_B SEND_ONLY
 * @param beta SEND_ONLY SCALAR:plan SIZE:sizeof(float)
 * @param d_C SEND_ONLY
 * @param d_D SEND_ONLY
 * @param workspace SEND_ONLY
 * @param numStreams SEND_ONLY
 * @param streams SEND_ONLY NULLABLE LENGTH:numStreams
 */
cusparseStatus_t cusparseLtMatmul(const cusparseLtHandle_t *handle,
                                  const cusparseLtMatmulPlan_t *plan,
                                  const void *alpha, const void *d_A,
                                  const void *d_B, const void *beta,
                                  const void *d_C, void *d_D, void *workspace,
                                  cudaStream_t *streams, int32_t numStreams);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param plan SEND_ONLY REMOTE
 * @param alpha SEND_ONLY SCALAR:plan SIZE:sizeof(float)
 * @param d_A SEND_ONLY
 * @param d_B SEND_ONLY
 * @param beta SEND_ONLY SCALAR:plan SIZE:sizeof(float)
 * @param d_C SEND_ONLY
 * @param d_D SEND_ONLY
 * @param workspace SEND_ONLY
 * @param numStreams SEND_ONLY
 * @param streams SEND_ONLY NULLABLE LENGTH:numStreams
 */
cusparseStatus_t cusparseLtMatmulSearch(const cusparseLtHandle_t *handle,
                                        cusparseLtMatmulPlan_t *plan,
                                        const void *alpha, const void *d_A,
                                        const void *d_B, const void *beta,
                                        const void *d_C, void *d_D,
                                        void *workspace, cudaStream_t *streams,
                                        int32_t numStreams);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param d_in SEND_ONLY
 * @param d_out SEND_ONLY
 * @param pruneAlg SEND_ONLY
 * @param stream SEND_ONLY
 */
cusparseStatus_t
cusparseLtSpMMAPrune(const cusparseLtHandle_t *handle,
                     const cusparseLtMatmulDescriptor_t *matmulDescr,
                     const void *d_in, void *d_out,
                     cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param matmulDescr SEND_ONLY REMOTE
 * @param d_in SEND_ONLY
 * @param valid SEND_ONLY
 * @param stream SEND_ONLY
 */
cusparseStatus_t
cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t *handle,
                          const cusparseLtMatmulDescriptor_t *matmulDescr,
                          const void *d_in, int *valid, cudaStream_t stream);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param sparseMatDescr SEND_ONLY REMOTE
 * @param isSparseA SEND_ONLY
 * @param op SEND_ONLY
 * @param d_in SEND_ONLY
 * @param d_out SEND_ONLY
 * @param pruneAlg SEND_ONLY
 * @param stream SEND_ONLY
 */
cusparseStatus_t
cusparseLtSpMMAPrune2(const cusparseLtHandle_t *handle,
                      const cusparseLtMatDescriptor_t *sparseMatDescr,
                      int isSparseA, cusparseOperation_t op, const void *d_in,
                      void *d_out, cusparseLtPruneAlg_t pruneAlg,
                      cudaStream_t stream);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param sparseMatDescr SEND_ONLY REMOTE
 * @param isSparseA SEND_ONLY
 * @param op SEND_ONLY
 * @param d_in SEND_ONLY
 * @param d_valid SEND_ONLY
 * @param stream SEND_ONLY
 */
cusparseStatus_t
cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t *handle,
                           const cusparseLtMatDescriptor_t *sparseMatDescr,
                           int isSparseA, cusparseOperation_t op,
                           const void *d_in, int *d_valid, cudaStream_t stream);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param plan SEND_ONLY REMOTE
 * @param compressedSize RECV_ONLY
 * @param compressedBufferSize RECV_ONLY
 */
cusparseStatus_t cusparseLtSpMMACompressedSize(
    const cusparseLtHandle_t *handle, const cusparseLtMatmulPlan_t *plan,
    size_t *compressedSize, size_t *compressedBufferSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param plan SEND_ONLY REMOTE
 * @param d_dense SEND_ONLY
 * @param d_compressed SEND_ONLY
 * @param d_compressed_buffer SEND_ONLY
 * @param stream SEND_ONLY
 */
cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t *handle,
                                         const cusparseLtMatmulPlan_t *plan,
                                         const void *d_dense,
                                         void *d_compressed,
                                         void *d_compressed_buffer,
                                         cudaStream_t stream);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param sparseMatDescr SEND_ONLY REMOTE
 * @param compressedSize RECV_ONLY
 * @param compressedBufferSize RECV_ONLY
 */
cusparseStatus_t
cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t *handle,
                               const cusparseLtMatDescriptor_t *sparseMatDescr,
                               size_t *compressedSize,
                               size_t *compressedBufferSize);
/**
 * @routingkey HANDLE handle
 * @param handle SEND_ONLY REMOTE
 * @param sparseMatDescr SEND_ONLY REMOTE
 * @param isSparseA SEND_ONLY
 * @param op SEND_ONLY
 * @param d_dense SEND_ONLY
 * @param d_compressed SEND_ONLY
 * @param d_compressed_buffer SEND_ONLY
 * @param stream SEND_ONLY
 */
cusparseStatus_t
cusparseLtSpMMACompress2(const cusparseLtHandle_t *handle,
                         const cusparseLtMatDescriptor_t *sparseMatDescr,
                         int isSparseA, cusparseOperation_t op,
                         const void *d_dense, void *d_compressed,
                         void *d_compressed_buffer, cudaStream_t stream);
