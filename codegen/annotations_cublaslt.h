#include <cublasLt.h>

// cuBLASLt API. Like cuBLAS, every call forwards over the CUDA driver shim's
// connections to the server child that runs the library; a handle routes to
// the connection that created it, and a call without one goes to the
// runtime's current device.
//
// A descriptor (matmul, layout, transform, preference, emulation) is a
// fixed-size opaque struct the library fills in, so it travels by value: the
// caller's storage is sent with a call and refilled from the answer, and
// nothing about it lives on the server between calls. `Init_internal`
// receives the storage, `SetAttribute` sends and receives it, and a call that
// reads it sends it, nullable where the API allows a null descriptor. The
// `Create` and `Destroy` pairs are client-local: `Create` allocates the struct
// on the client and runs the generated `Init_internal`.
//
// alpha and beta follow the pointer mode of the matmul or transform descriptor
// (`SCALAR:<desc>`), and their width is that descriptor's scale type; the
// client mirrors both attributes as descriptors are initialized and set.
//
// The experimental grouped matrix layouts (13.x) take host arrays of an
// undocumented element type and are left to the stubs, as is anything else
// cublasLt.h declares that is absent below.

/**
 * @recordowner HANDLE lightHandle
 * @param lightHandle RECV_ONLY
 */
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle);
/**
 * @param lightHandle SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(lightHandle);
  return return_value;
}
// clang-format on
#if CUBLAS_VERSION >= 120900
/**
 * @guard CUBLAS_VERSION >= 120900
 * @param mask SEND_ONLY
 */
unsigned cublasLtDisableCpuInstructionsSetMask(unsigned mask);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @disabled local
 */
cublasStatus_t
cublasLtEmulationDescCreate(cublasLtEmulationDesc_t *emulationDesc);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @disabled local
 */
cublasStatus_t
cublasLtEmulationDescDestroy(cublasLtEmulationDescOpaque_t *emulationDesc);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param emulationDesc SEND_ONLY NULLABLE
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t
cublasLtEmulationDescGetAttribute(cublasLtEmulationDescOpaque_t *emulationDesc,
                                  cublasLtEmulationDescAttributes_t attr,
                                  void *buf, size_t sizeInBytes,
                                  size_t *sizeWritten);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param emulationDesc RECV_ONLY
 * @param size SEND_ONLY
 */
cublasStatus_t
cublasLtEmulationDescInit_internal(cublasLtEmulationDescOpaque_t *emulationDesc,
                                   size_t size);
#endif
#if CUBLAS_VERSION >= 130100
/**
 * @guard CUBLAS_VERSION >= 130100
 * @param emulationDesc SEND_RECV DEREF
 * @param attr SEND_ONLY
 * @param buf SEND_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 */
cublasStatus_t
cublasLtEmulationDescSetAttribute(cublasLtEmulationDescOpaque_t *emulationDesc,
                                  cublasLtEmulationDescAttributes_t attr,
                                  const void *buf, size_t sizeInBytes);
#endif
size_t cublasLtGetCudartVersion();
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int *value);
/**
 * @disabled
 */
const char *cublasLtGetStatusName(cublasStatus_t status);
/**
 * @disabled
 */
const char *cublasLtGetStatusString(cublasStatus_t status);
size_t cublasLtGetVersion();
/**
 * @param capacity RECV_ONLY
 */
cublasStatus_t cublasLtHeuristicsCacheGetCapacity(size_t *capacity);
/**
 * @param capacity SEND_ONLY
 */
cublasStatus_t cublasLtHeuristicsCacheSetCapacity(size_t capacity);
cublasStatus_t cublasLtLoggerForceDisable();
/**
 * @param logFile SEND_ONLY NULL_TERMINATED
 */
cublasStatus_t cublasLtLoggerOpenFile(const char *logFile);
cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback);
cublasStatus_t cublasLtLoggerSetFile(FILE *file);
/**
 * @param level SEND_ONLY
 */
cublasStatus_t cublasLtLoggerSetLevel(int level);
/**
 * @param mask SEND_ONLY
 */
cublasStatus_t cublasLtLoggerSetMask(int mask);
/**
 * @param lightHandle SEND_ONLY
 * @param computeDesc SEND_ONLY NULLABLE
 * @param alpha SEND_ONLY SCALAR:computeDesc SIZE:scalar_width(computeDesc)
 * @param A SEND_ONLY
 * @param Adesc SEND_ONLY NULLABLE
 * @param B SEND_ONLY
 * @param Bdesc SEND_ONLY NULLABLE
 * @param beta SEND_ONLY SCALAR:computeDesc SIZE:scalar_width(computeDesc)
 * @param C SEND_ONLY
 * @param Cdesc SEND_ONLY NULLABLE
 * @param D SEND_ONLY
 * @param Ddesc SEND_ONLY NULLABLE
 * @param algo SEND_ONLY NULLABLE
 * @param workspace SEND_ONLY
 * @param workspaceSizeInBytes SEND_ONLY
 * @param stream SEND_ONLY
 */
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDescOpaque_t *computeDesc,
    const void *alpha, const void *A, cublasLtMatrixLayoutOpaque_t *Adesc,
    const void *B, cublasLtMatrixLayoutOpaque_t *Bdesc, const void *beta,
    const void *C, cublasLtMatrixLayoutOpaque_t *Cdesc, void *D,
    cublasLtMatrixLayoutOpaque_t *Ddesc, const cublasLtMatmulAlgo_t *algo,
    void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream);
/**
 * @param algo SEND_ONLY DEREF
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoCapAttributes_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten);
/**
 * @param lightHandle SEND_ONLY
 * @param operationDesc SEND_ONLY NULLABLE
 * @param Adesc SEND_ONLY NULLABLE
 * @param Bdesc SEND_ONLY NULLABLE
 * @param Cdesc SEND_ONLY NULLABLE
 * @param Ddesc SEND_ONLY NULLABLE
 * @param algo SEND_ONLY DEREF
 * @param result RECV_ONLY
 */
cublasStatus_t cublasLtMatmulAlgoCheck(
    cublasLtHandle_t lightHandle, cublasLtMatmulDescOpaque_t *operationDesc,
    cublasLtMatrixLayoutOpaque_t *Adesc, cublasLtMatrixLayoutOpaque_t *Bdesc,
    cublasLtMatrixLayoutOpaque_t *Cdesc, cublasLtMatrixLayoutOpaque_t *Ddesc,
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulHeuristicResult_t *result);
#if CUBLAS_VERSION >= 130300
/**
 * @guard CUBLAS_VERSION >= 130300
 * @param lightHandle SEND_ONLY
 * @param operationDesc SEND_ONLY NULLABLE
 * @param Adesc SEND_ONLY NULLABLE
 * @param Bdesc SEND_ONLY NULLABLE
 * @param Cdesc SEND_ONLY NULLABLE
 * @param Ddesc SEND_ONLY NULLABLE
 * @param algo SEND_ONLY DEREF
 * @param result RECV_ONLY
 * @param stream SEND_ONLY
 */
cublasStatus_t cublasLtMatmulAlgoCheckForStream(
    cublasLtHandle_t lightHandle, cublasLtMatmulDescOpaque_t *operationDesc,
    cublasLtMatrixLayoutOpaque_t *Adesc, cublasLtMatrixLayoutOpaque_t *Bdesc,
    cublasLtMatrixLayoutOpaque_t *Cdesc, cublasLtMatrixLayoutOpaque_t *Ddesc,
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulHeuristicResult_t *result,
    cudaStream_t stream);
#endif
/**
 * @param algo SEND_ONLY DEREF
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr,
    void *buf, size_t sizeInBytes, size_t *sizeWritten);
/**
 * @param algo SEND_RECV DEREF
 * @param attr SEND_ONLY
 * @param buf SEND_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 */
cublasStatus_t
cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t *algo,
                                     cublasLtMatmulAlgoConfigAttributes_t attr,
                                     const void *buf, size_t sizeInBytes);
/**
 * @param lightHandle SEND_ONLY
 * @param operationDesc SEND_ONLY NULLABLE
 * @param Adesc SEND_ONLY NULLABLE
 * @param Bdesc SEND_ONLY NULLABLE
 * @param Cdesc SEND_ONLY NULLABLE
 * @param Ddesc SEND_ONLY NULLABLE
 * @param preference SEND_ONLY NULLABLE
 * @param requestedAlgoCount SEND_ONLY
 * @param heuristicResultsArray RECV_ONLY LENGTH:requestedAlgoCount
 * @param returnAlgoCount RECV_ONLY
 */
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDescOpaque_t *operationDesc,
    cublasLtMatrixLayoutOpaque_t *Adesc, cublasLtMatrixLayoutOpaque_t *Bdesc,
    cublasLtMatrixLayoutOpaque_t *Cdesc, cublasLtMatrixLayoutOpaque_t *Ddesc,
    cublasLtMatmulPreferenceOpaque_t *preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
    int *returnAlgoCount);
#if CUBLAS_VERSION >= 130300
/**
 * @guard CUBLAS_VERSION >= 130300
 * @param lightHandle SEND_ONLY
 * @param operationDesc SEND_ONLY NULLABLE
 * @param Adesc SEND_ONLY NULLABLE
 * @param Bdesc SEND_ONLY NULLABLE
 * @param Cdesc SEND_ONLY NULLABLE
 * @param Ddesc SEND_ONLY NULLABLE
 * @param preference SEND_ONLY NULLABLE
 * @param requestedAlgoCount SEND_ONLY
 * @param heuristicResultsArray RECV_ONLY LENGTH:requestedAlgoCount
 * @param returnAlgoCount RECV_ONLY
 * @param stream SEND_ONLY
 */
cublasStatus_t cublasLtMatmulAlgoGetHeuristicForStream(
    cublasLtHandle_t lightHandle, cublasLtMatmulDescOpaque_t *operationDesc,
    cublasLtMatrixLayoutOpaque_t *Adesc, cublasLtMatrixLayoutOpaque_t *Bdesc,
    cublasLtMatrixLayoutOpaque_t *Cdesc, cublasLtMatrixLayoutOpaque_t *Ddesc,
    cublasLtMatmulPreferenceOpaque_t *preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
    int *returnAlgoCount, cudaStream_t stream);
#endif
/**
 * @param lightHandle SEND_ONLY
 * @param computeType SEND_ONLY
 * @param scaleType SEND_ONLY
 * @param Atype SEND_ONLY
 * @param Btype SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param Dtype SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param algoIdsArray RECV_ONLY LENGTH:requestedAlgoCount
 * @param returnAlgoCount RECV_ONLY
 */
cublasStatus_t cublasLtMatmulAlgoGetIds(
    cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype,
    cudaDataType_t Ctype, cudaDataType_t Dtype, int requestedAlgoCount,
    int *algoIdsArray, int *returnAlgoCount);
/**
 * @param lightHandle SEND_ONLY
 * @param computeType SEND_ONLY
 * @param scaleType SEND_ONLY
 * @param Atype SEND_ONLY
 * @param Btype SEND_ONLY
 * @param Ctype SEND_ONLY
 * @param Dtype SEND_ONLY
 * @param algoId SEND_ONLY
 * @param algo RECV_ONLY
 */
cublasStatus_t cublasLtMatmulAlgoInit(
    cublasLtHandle_t lightHandle, cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype,
    cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId,
    cublasLtMatmulAlgo_t *algo);
/**
 * @disabled local
 */
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cudaDataType_t scaleType);
/**
 * @disabled local
 */
cublasStatus_t
cublasLtMatmulDescDestroy(cublasLtMatmulDescOpaque_t *matmulDesc);
/**
 * @param matmulDesc SEND_ONLY NULLABLE
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t
cublasLtMatmulDescGetAttribute(cublasLtMatmulDescOpaque_t *matmulDesc,
                               cublasLtMatmulDescAttributes_t attr, void *buf,
                               size_t sizeInBytes, size_t *sizeWritten);
/**
 * @param matmulDesc RECV_ONLY
 * @param size SEND_ONLY
 * @param computeType SEND_ONLY
 * @param scaleType SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasLtMatmulDescInit_internal(
    cublasLtMatmulDescOpaque_t *matmulDesc, size_t size,
    cublasComputeType_t computeType, cudaDataType_t scaleType) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUBLAS_STATUS_SUCCESS) {
    note_descriptor(matmulDesc, scaleType);
  }
  return return_value;
}
// clang-format on
/**
 * @param matmulDesc SEND_RECV DEREF
 * @param attr SEND_ONLY
 * @param buf SEND_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDescOpaque_t *matmulDesc, cublasLtMatmulDescAttributes_t attr,
    const void *buf, size_t sizeInBytes) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUBLAS_STATUS_SUCCESS) {
    note_attribute(matmulDesc, attr, buf, sizeInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled local
 */
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref);
/**
 * @disabled local
 */
cublasStatus_t
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreferenceOpaque_t *pref);
/**
 * @param pref SEND_ONLY NULLABLE
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t
cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreferenceOpaque_t *pref,
                                     cublasLtMatmulPreferenceAttributes_t attr,
                                     void *buf, size_t sizeInBytes,
                                     size_t *sizeWritten);
/**
 * @param pref RECV_ONLY
 * @param size SEND_ONLY
 */
cublasStatus_t
cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreferenceOpaque_t *pref,
                                      size_t size);
/**
 * @param pref SEND_RECV DEREF
 * @param attr SEND_ONLY
 * @param buf SEND_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 */
cublasStatus_t
cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreferenceOpaque_t *pref,
                                     cublasLtMatmulPreferenceAttributes_t attr,
                                     const void *buf, size_t sizeInBytes);
/**
 * @disabled local
 */
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                          cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld);
/**
 * @disabled local
 */
cublasStatus_t
cublasLtMatrixLayoutDestroy(cublasLtMatrixLayoutOpaque_t *matLayout);
/**
 * @param matLayout SEND_ONLY NULLABLE
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t
cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayoutOpaque_t *matLayout,
                                 cublasLtMatrixLayoutAttribute_t attr,
                                 void *buf, size_t sizeInBytes,
                                 size_t *sizeWritten);
/**
 * @param matLayout RECV_ONLY
 * @param size SEND_ONLY
 * @param type SEND_ONLY
 * @param rows SEND_ONLY
 * @param cols SEND_ONLY
 * @param ld SEND_ONLY
 */
cublasStatus_t
cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayoutOpaque_t *matLayout,
                                  size_t size, cudaDataType type, uint64_t rows,
                                  uint64_t cols, int64_t ld);
/**
 * @param matLayout SEND_RECV DEREF
 * @param attr SEND_ONLY
 * @param buf SEND_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 */
cublasStatus_t
cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayoutOpaque_t *matLayout,
                                 cublasLtMatrixLayoutAttribute_t attr,
                                 const void *buf, size_t sizeInBytes);
/**
 * @param lightHandle SEND_ONLY
 * @param transformDesc SEND_ONLY NULLABLE
 * @param alpha SEND_ONLY SCALAR:transformDesc SIZE:scalar_width(transformDesc)
 * @param A SEND_ONLY
 * @param Adesc SEND_ONLY NULLABLE
 * @param beta SEND_ONLY SCALAR:transformDesc SIZE:scalar_width(transformDesc)
 * @param B SEND_ONLY
 * @param Bdesc SEND_ONLY NULLABLE
 * @param C SEND_ONLY
 * @param Cdesc SEND_ONLY NULLABLE
 * @param stream SEND_ONLY
 */
cublasStatus_t cublasLtMatrixTransform(
    cublasLtHandle_t lightHandle,
    cublasLtMatrixTransformDescOpaque_t *transformDesc, const void *alpha,
    const void *A, cublasLtMatrixLayoutOpaque_t *Adesc, const void *beta,
    const void *B, cublasLtMatrixLayoutOpaque_t *Bdesc, void *C,
    cublasLtMatrixLayoutOpaque_t *Cdesc, cudaStream_t stream);
/**
 * @disabled local
 */
cublasStatus_t
cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t *transformDesc,
                                  cudaDataType scaleType);
/**
 * @disabled local
 */
cublasStatus_t cublasLtMatrixTransformDescDestroy(
    cublasLtMatrixTransformDescOpaque_t *transformDesc);
/**
 * @param transformDesc SEND_ONLY NULLABLE
 * @param attr SEND_ONLY
 * @param buf RECV_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 * @param sizeWritten RECV_ONLY NULLABLE
 */
cublasStatus_t cublasLtMatrixTransformDescGetAttribute(
    cublasLtMatrixTransformDescOpaque_t *transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr, void *buf, size_t sizeInBytes,
    size_t *sizeWritten);
/**
 * @param transformDesc RECV_ONLY
 * @param size SEND_ONLY
 * @param scaleType SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasLtMatrixTransformDescInit_internal(
    cublasLtMatrixTransformDescOpaque_t *transformDesc, size_t size,
    cudaDataType scaleType) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUBLAS_STATUS_SUCCESS) {
    note_descriptor(transformDesc, scaleType);
  }
  return return_value;
}
// clang-format on
/**
 * @param transformDesc SEND_RECV DEREF
 * @param attr SEND_ONLY
 * @param buf SEND_ONLY LENGTH:sizeInBytes
 * @param sizeInBytes SEND_ONLY
 */
// clang-format off
cublasStatus_t cublasLtMatrixTransformDescSetAttribute(
    cublasLtMatrixTransformDescOpaque_t *transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr, const void *buf,
    size_t sizeInBytes) {
  cublasStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUBLAS_STATUS_SUCCESS) {
    note_attribute(transformDesc, attr, buf, sizeInBytes);
  }
  return return_value;
}
// clang-format on
