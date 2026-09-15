#include <cudnn.h>

// cuDNN 9 API. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libcudnn by
// name at runtime. A handle or descriptor is a pointer the server's library
// hands out; it routes to the connection that created it, and a call without
// one goes to the runtime's current device.
//
// Tensor data, workspaces, reserve spaces, dropout states and the device
// sequence-length arrays are device pointers and travel as addresses.
// Descriptors travel by value and live on the server. Dimension arrays are
// `LENGTH:<count>`. Scaling factors are host scalars (`SCALAR`) whose width
// follows the data type of the descriptor they scale: double for
// CUDNN_DATA_DOUBLE and float otherwise, as the library reads them. The
// client records each descriptor's data type as it is set and asks the server
// for the ones the library derives.
//
// Hand-written: cudnnGetErrorString (a static string), the backend attribute
// accessors (the element width depends on the attribute type), and the CTC
// loss and multi-head attention calls whose host arrays are sized by a
// descriptor. Declared below without saying what to do with their
// parameters, and so stubs returning CUDNN_STATUS_NOT_SUPPORTED: the callback
// accessors, which carry a client function, and the fused-ops parameter pack
// attributes, whose host pointers the pack keeps.
//
// Calls that arrived after cuDNN 9.0 carry the matching @guard.

size_t cudnnGetVersion();
size_t cudnnGetMaxDeviceVersion();
size_t cudnnGetCudartVersion();
/**
 * @disabled
 */
const char *cudnnGetErrorString(cudnnStatus_t status);
/**
 * @param message RECV_ONLY LENGTH:max_size
 * @param max_size SEND_ONLY
 */
void cudnnGetLastErrorString(char *message, size_t max_size);
/**
 * @param handle SEND_ONLY
 * @param rstatus RECV_ONLY
 * @param mode SEND_ONLY
 * @param tag SEND_ONLY
 */
cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t handle,
                                     cudnnStatus_t *rstatus,
                                     cudnnErrQueryMode_t mode,
                                     cudnnRuntimeTag_t *tag);
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int *value);
/**
 * @param handle RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param streamId SEND_ONLY
 */
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
/**
 * @param handle SEND_ONLY
 * @param streamId RECV_ONLY
 */
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);
cudnnStatus_t cudnnSetCallback(unsigned mask, void *udata,
                               cudnnCallback_t fptr);
cudnnStatus_t cudnnGetCallback(unsigned *mask, void **udata,
                               cudnnCallback_t *fptr);
cudnnStatus_t cudnnGraphVersionCheck();
/**
 * @param descriptorType SEND_ONLY
 * @param descriptor RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t *descriptor) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *descriptor);
  }
  return return_value;
}
// clang-format on
/**
 * @param descriptor SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnBackendDestroyDescriptor(
    cudnnBackendDescriptor_t descriptor) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(descriptor);
  return return_value;
}
// clang-format on
/**
 * @param descriptor SEND_ONLY
 */
cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor);
/**
 * @param descriptor SEND_ONLY
 */
cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor);
/**
 * @disabled
 */
cudnnStatus_t
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t elementCount, const void *arrayOfElements);
/**
 * @disabled
 */
cudnnStatus_t
cudnnBackendGetAttribute(const cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount, int64_t *elementCount,
                         void *arrayOfElements);
#if CUDNN_VERSION >= 92300
/**
 * @guard CUDNN_VERSION >= 92300
 * @param handle SEND_ONLY
 * @param executionPlan SEND_ONLY
 * @param variantPack SEND_ONLY
 * @param workspaceSizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetExecutionPlanWorkspaceSize(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack, size_t *workspaceSizeInBytes);
#endif
/**
 * @param handle SEND_ONLY
 * @param executionPlan SEND_ONLY
 * @param variantPack SEND_ONLY
 */
cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle,
                                  cudnnBackendDescriptor_t executionPlan,
                                  cudnnBackendDescriptor_t variantPack);
#if CUDNN_VERSION >= 90500
/**
 * @guard CUDNN_VERSION >= 90500
 * @param handle SEND_ONLY
 * @param executionPlan SEND_ONLY
 * @param variantPack SEND_ONLY
 * @param graph SEND_ONLY
 */
cudnnStatus_t cudnnBackendPopulateCudaGraph(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack, cudaGraph_t graph);
#endif
#if CUDNN_VERSION >= 90500
/**
 * @guard CUDNN_VERSION >= 90500
 * @param handle SEND_ONLY
 * @param executionPlan SEND_ONLY
 * @param variantPack SEND_ONLY
 * @param graph SEND_ONLY
 */
cudnnStatus_t cudnnBackendUpdateCudaGraph(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
    cudnnBackendDescriptor_t variantPack, cudaGraph_t graph);
#endif
/**
 * @param tensorDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *tensorDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param tensorDesc SEND_ONLY
 * @param format SEND_ONLY
 * @param dataType SEND_ONLY
 * @param n SEND_ONLY
 * @param c SEND_ONLY
 * @param h SEND_ONLY
 * @param w SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int n, int c, int h, int w) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    const int dims[] = {n, c, h, w};
    note_data_type(tensorDesc, dataType, 4, dims);
  }
  return return_value;
}
// clang-format on
/**
 * @param tensorDesc SEND_ONLY
 * @param dataType SEND_ONLY
 * @param n SEND_ONLY
 * @param c SEND_ONLY
 * @param h SEND_ONLY
 * @param w SEND_ONLY
 * @param nStride SEND_ONLY
 * @param cStride SEND_ONLY
 * @param hStride SEND_ONLY
 * @param wStride SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnSetTensor4dDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c,
    int h, int w, int nStride, int cStride, int hStride, int wStride) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    const int dims[] = {n, c, h, w};
    note_data_type(tensorDesc, dataType, 4, dims);
  }
  return return_value;
}
// clang-format on
/**
 * @param tensorDesc SEND_ONLY
 * @param dataType RECV_ONLY
 * @param n RECV_ONLY
 * @param c RECV_ONLY
 * @param h RECV_ONLY
 * @param w RECV_ONLY
 * @param nStride RECV_ONLY
 * @param cStride RECV_ONLY
 * @param hStride RECV_ONLY
 * @param wStride RECV_ONLY
 */
cudnnStatus_t
cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t *dataType, int *n, int *c, int *h,
                           int *w, int *nStride, int *cStride, int *hStride,
                           int *wStride);
/**
 * @param tensorDesc SEND_ONLY
 * @param dataType SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param dimA SEND_ONLY LENGTH:nbDims
 * @param strideA SEND_ONLY LENGTH:nbDims
 */
// clang-format off
cudnnStatus_t cudnnSetTensorNdDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
    const int *dimA, const int *strideA) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_data_type(tensorDesc, dataType, nbDims, dimA);
  }
  return return_value;
}
// clang-format on
/**
 * @param tensorDesc SEND_ONLY
 * @param format SEND_ONLY
 * @param dataType SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param dimA SEND_ONLY LENGTH:nbDims
 */
// clang-format off
cudnnStatus_t cudnnSetTensorNdDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
    cudnnDataType_t dataType, int nbDims, const int *dimA) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_data_type(tensorDesc, dataType, nbDims, dimA);
  }
  return return_value;
}
// clang-format on
/**
 * @param tensorDesc SEND_ONLY
 * @param nbDimsRequested SEND_ONLY
 * @param dataType RECV_ONLY
 * @param nbDims RECV_ONLY
 * @param dimA RECV_ONLY LENGTH:nbDimsRequested
 * @param strideA RECV_ONLY LENGTH:nbDimsRequested
 */
cudnnStatus_t
cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested, cudnnDataType_t *dataType,
                           int *nbDims, int *dimA, int *strideA);
/**
 * @param tensorDesc SEND_ONLY
 * @param size RECV_ONLY
 */
cudnnStatus_t
cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc,
                          size_t *size);
/**
 * @param tensorDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(tensorDesc);
  return return_value;
}
// clang-format on
/**
 * @param transformDesc SEND_ONLY
 * @param srcDesc SEND_ONLY
 * @param destDesc SEND_ONLY
 * @param destSizeInBytes RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnInitTransformDest(
    const cudnnTensorTransformDescriptor_t transformDesc,
    const cudnnTensorDescriptor_t srcDesc, cudnnTensorDescriptor_t destDesc,
    size_t *destSizeInBytes) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_data_type(destDesc);
  return return_value;
}
// clang-format on
/**
 * @param transformDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t *transformDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *transformDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param transformDesc SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param destFormat SEND_ONLY
 * @param padBeforeA SEND_ONLY NULLABLE LENGTH:nbDims
 * @param padAfterA SEND_ONLY NULLABLE LENGTH:nbDims
 * @param foldA SEND_ONLY NULLABLE LENGTH:nbDims
 * @param direction SEND_ONLY
 */
cudnnStatus_t cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t *padBeforeA,
    const int32_t *padAfterA, const uint32_t *foldA,
    const cudnnFoldingDirection_t direction);
/**
 * @param transformDesc SEND_ONLY
 * @param nbDimsRequested SEND_ONLY
 * @param destFormat RECV_ONLY
 * @param padBeforeA RECV_ONLY NULLABLE LENGTH:nbDimsRequested
 * @param padAfterA RECV_ONLY NULLABLE LENGTH:nbDimsRequested
 * @param foldA RECV_ONLY NULLABLE LENGTH:nbDimsRequested
 * @param direction RECV_ONLY
 */
cudnnStatus_t cudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t *padBeforeA, int32_t *padAfterA,
    uint32_t *foldA, cudnnFoldingDirection_t *direction);
/**
 * @param transformDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(transformDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnTransformTensor(cudnnHandle_t handle, const void *alpha,
                                   const cudnnTensorDescriptor_t xDesc,
                                   const void *x, const void *beta,
                                   const cudnnTensorDescriptor_t yDesc,
                                   void *y);
/**
 * @param handle SEND_ONLY
 * @param transDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:destDesc SIZE:scaling_width(destDesc)
 * @param srcDesc SEND_ONLY
 * @param srcData SEND_ONLY
 * @param beta SEND_ONLY SCALAR:destDesc SIZE:scaling_width(destDesc)
 * @param destDesc SEND_ONLY
 * @param destData SEND_ONLY
 */
cudnnStatus_t
cudnnTransformTensorEx(cudnnHandle_t handle,
                       const cudnnTensorTransformDescriptor_t transDesc,
                       const void *alpha, const cudnnTensorDescriptor_t srcDesc,
                       const void *srcData, const void *beta,
                       const cudnnTensorDescriptor_t destDesc, void *destData);
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param aDesc SEND_ONLY
 * @param A SEND_ONLY
 * @param beta SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param cDesc SEND_ONLY
 * @param C SEND_ONLY
 */
cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void *alpha,
                             const cudnnTensorDescriptor_t aDesc, const void *A,
                             const void *beta,
                             const cudnnTensorDescriptor_t cDesc, void *C);
/**
 * @param opTensorDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateOpTensorDescriptor(
    cudnnOpTensorDescriptor_t *opTensorDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *opTensorDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param opTensorDesc SEND_ONLY
 * @param opTensorOp SEND_ONLY
 * @param opTensorCompType SEND_ONLY
 * @param opTensorNanOpt SEND_ONLY
 */
cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                         cudnnOpTensorOp_t opTensorOp,
                                         cudnnDataType_t opTensorCompType,
                                         cudnnNanPropagation_t opTensorNanOpt);
/**
 * @param opTensorDesc SEND_ONLY
 * @param opTensorOp RECV_ONLY
 * @param opTensorCompType RECV_ONLY
 * @param opTensorNanOpt RECV_ONLY
 */
cudnnStatus_t cudnnGetOpTensorDescriptor(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt);
/**
 * @param opTensorDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyOpTensorDescriptor(
    cudnnOpTensorDescriptor_t opTensorDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(opTensorDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param opTensorDesc SEND_ONLY
 * @param alpha1 SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param aDesc SEND_ONLY
 * @param A SEND_ONLY
 * @param alpha2 SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param bDesc SEND_ONLY
 * @param B SEND_ONLY
 * @param beta SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param cDesc SEND_ONLY
 * @param C SEND_ONLY
 */
cudnnStatus_t cudnnOpTensor(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc, void *C);
/**
 * @param reduceTensorDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *reduceTensorDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param reduceTensorDesc SEND_ONLY
 * @param reduceTensorOp SEND_ONLY
 * @param reduceTensorCompType SEND_ONLY
 * @param reduceTensorNanOpt SEND_ONLY
 * @param reduceTensorIndices SEND_ONLY
 * @param reduceTensorIndicesType SEND_ONLY
 */
cudnnStatus_t
cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t reduceTensorOp,
                               cudnnDataType_t reduceTensorCompType,
                               cudnnNanPropagation_t reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t reduceTensorIndices,
                               cudnnIndicesType_t reduceTensorIndicesType);
/**
 * @param reduceTensorDesc SEND_ONLY
 * @param reduceTensorOp RECV_ONLY
 * @param reduceTensorCompType RECV_ONLY
 * @param reduceTensorNanOpt RECV_ONLY
 * @param reduceTensorIndices RECV_ONLY
 * @param reduceTensorIndicesType RECV_ONLY
 */
cudnnStatus_t cudnnGetReduceTensorDescriptor(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType);
/**
 * @param reduceTensorDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(reduceTensorDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param reduceTensorDesc SEND_ONLY
 * @param aDesc SEND_ONLY
 * @param cDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetReductionIndicesSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param reduceTensorDesc SEND_ONLY
 * @param aDesc SEND_ONLY
 * @param cDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param reduceTensorDesc SEND_ONLY
 * @param indices SEND_ONLY
 * @param indicesSizeInBytes SEND_ONLY
 * @param workspace SEND_ONLY
 * @param workspaceSizeInBytes SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param aDesc SEND_ONLY
 * @param A SEND_ONLY
 * @param beta SEND_ONLY SCALAR:cDesc SIZE:scaling_width(cDesc)
 * @param cDesc SEND_ONLY
 * @param C SEND_ONLY
 */
cudnnStatus_t cudnnReduceTensor(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes, void *workspace,
    size_t workspaceSizeInBytes, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C);
/**
 * @param handle SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param valuePtr SEND_ONLY SCALAR:yDesc SIZE:element_width(yDesc)
 */
cudnnStatus_t cudnnSetTensor(cudnnHandle_t handle,
                             const cudnnTensorDescriptor_t yDesc, void *y,
                             const void *valuePtr);
/**
 * @param handle SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 */
cudnnStatus_t cudnnScaleTensor(cudnnHandle_t handle,
                               const cudnnTensorDescriptor_t yDesc, void *y,
                               const void *alpha);
/**
 * @param filterDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *filterDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param filterDesc SEND_ONLY
 * @param dataType SEND_ONLY
 * @param format SEND_ONLY
 * @param k SEND_ONLY
 * @param c SEND_ONLY
 * @param h SEND_ONLY
 * @param w SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int k, int c, int h, int w) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_data_type(filterDesc, dataType, 0, nullptr);
  }
  return return_value;
}
// clang-format on
/**
 * @param filterDesc SEND_ONLY
 * @param dataType RECV_ONLY
 * @param format RECV_ONLY
 * @param k RECV_ONLY
 * @param c RECV_ONLY
 * @param h RECV_ONLY
 * @param w RECV_ONLY
 */
cudnnStatus_t cudnnGetFilter4dDescriptor(
    const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w);
/**
 * @param filterDesc SEND_ONLY
 * @param dataType SEND_ONLY
 * @param format SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param filterDimA SEND_ONLY LENGTH:nbDims
 */
// clang-format off
cudnnStatus_t cudnnSetFilterNdDescriptor(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
    cudnnTensorFormat_t format, int nbDims, const int *filterDimA) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_data_type(filterDesc, dataType, 0, nullptr);
  }
  return return_value;
}
// clang-format on
/**
 * @param filterDesc SEND_ONLY
 * @param nbDimsRequested SEND_ONLY
 * @param dataType RECV_ONLY
 * @param format RECV_ONLY
 * @param nbDims RECV_ONLY
 * @param filterDimA RECV_ONLY LENGTH:nbDimsRequested
 */
cudnnStatus_t
cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           int nbDimsRequested, cudnnDataType_t *dataType,
                           cudnnTensorFormat_t *format, int *nbDims,
                           int *filterDimA);
/**
 * @param filterDesc SEND_ONLY
 * @param size RECV_ONLY
 */
cudnnStatus_t
cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc,
                          size_t *size);
/**
 * @param handle SEND_ONLY
 * @param transDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:destDesc SIZE:scaling_width(destDesc)
 * @param srcDesc SEND_ONLY
 * @param srcData SEND_ONLY
 * @param beta SEND_ONLY SCALAR:destDesc SIZE:scaling_width(destDesc)
 * @param destDesc SEND_ONLY
 * @param destData SEND_ONLY
 */
cudnnStatus_t
cudnnTransformFilter(cudnnHandle_t handle,
                     const cudnnTensorTransformDescriptor_t transDesc,
                     const void *alpha, const cudnnFilterDescriptor_t srcDesc,
                     const void *srcData, const void *beta,
                     const cudnnFilterDescriptor_t destDesc, void *destData);
/**
 * @param filterDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(filterDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param mode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle,
                                  cudnnSoftmaxAlgorithm_t algo,
                                  cudnnSoftmaxMode_t mode, const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc,
                                  const void *x, const void *beta,
                                  const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param poolingDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreatePoolingDescriptor(
    cudnnPoolingDescriptor_t *poolingDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *poolingDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param poolingDesc SEND_ONLY
 * @param mode SEND_ONLY
 * @param maxpoolingNanOpt SEND_ONLY
 * @param windowHeight SEND_ONLY
 * @param windowWidth SEND_ONLY
 * @param verticalPadding SEND_ONLY
 * @param horizontalPadding SEND_ONLY
 * @param verticalStride SEND_ONLY
 * @param horizontalStride SEND_ONLY
 */
cudnnStatus_t cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride);
/**
 * @param poolingDesc SEND_ONLY
 * @param mode RECV_ONLY
 * @param maxpoolingNanOpt RECV_ONLY
 * @param windowHeight RECV_ONLY
 * @param windowWidth RECV_ONLY
 * @param verticalPadding RECV_ONLY
 * @param horizontalPadding RECV_ONLY
 * @param verticalStride RECV_ONLY
 * @param horizontalStride RECV_ONLY
 */
cudnnStatus_t cudnnGetPooling2dDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride);
/**
 * @param poolingDesc SEND_ONLY
 * @param mode SEND_ONLY
 * @param maxpoolingNanOpt SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param windowDimA SEND_ONLY LENGTH:nbDims
 * @param paddingA SEND_ONLY LENGTH:nbDims
 * @param strideA SEND_ONLY LENGTH:nbDims
 */
cudnnStatus_t cudnnSetPoolingNdDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int *windowDimA, const int *paddingA, const int *strideA);
/**
 * @param poolingDesc SEND_ONLY
 * @param nbDimsRequested SEND_ONLY
 * @param mode RECV_ONLY
 * @param maxpoolingNanOpt RECV_ONLY
 * @param nbDims RECV_ONLY
 * @param windowDimA RECV_ONLY LENGTH:nbDimsRequested
 * @param paddingA RECV_ONLY LENGTH:nbDimsRequested
 * @param strideA RECV_ONLY LENGTH:nbDimsRequested
 */
cudnnStatus_t cudnnGetPoolingNdDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int *windowDimA, int *paddingA, int *strideA);
/**
 * @param poolingDesc SEND_ONLY
 * @param inputTensorDesc SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param outputTensorDimA RECV_ONLY LENGTH:nbDims
 */
cudnnStatus_t
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims, int *outputTensorDimA);
/**
 * @param poolingDesc SEND_ONLY
 * @param inputTensorDesc SEND_ONLY
 * @param n RECV_ONLY
 * @param c RECV_ONLY
 * @param h RECV_ONLY
 * @param w RECV_ONLY
 */
cudnnStatus_t
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n, int *c, int *h, int *w);
/**
 * @param poolingDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyPoolingDescriptor(
    cudnnPoolingDescriptor_t poolingDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(poolingDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param poolingDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle,
                                  const cudnnPoolingDescriptor_t poolingDesc,
                                  const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc,
                                  const void *x, const void *beta,
                                  const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param activationDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateActivationDescriptor(
    cudnnActivationDescriptor_t *activationDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *activationDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param activationDesc SEND_ONLY
 * @param mode SEND_ONLY
 * @param reluNanOpt SEND_ONLY
 * @param coef SEND_ONLY
 */
cudnnStatus_t
cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t mode,
                             cudnnNanPropagation_t reluNanOpt, double coef);
/**
 * @param activationDesc SEND_ONLY
 * @param mode RECV_ONLY
 * @param reluNanOpt RECV_ONLY
 * @param coef RECV_ONLY
 */
cudnnStatus_t
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt, double *coef);
/**
 * @param activationDesc SEND_ONLY
 * @param swish_beta SEND_ONLY
 */
cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(
    cudnnActivationDescriptor_t activationDesc, double swish_beta);
/**
 * @param activationDesc SEND_ONLY
 * @param swish_beta RECV_ONLY
 */
cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(
    cudnnActivationDescriptor_t activationDesc, double *swish_beta);
/**
 * @param activationDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyActivationDescriptor(
    cudnnActivationDescriptor_t activationDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(activationDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param normDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *normDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param normDesc SEND_ONLY
 * @param lrnN SEND_ONLY
 * @param lrnAlpha SEND_ONLY
 * @param lrnBeta SEND_ONLY
 * @param lrnK SEND_ONLY
 */
cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                    unsigned lrnN, double lrnAlpha,
                                    double lrnBeta, double lrnK);
/**
 * @param normDesc SEND_ONLY
 * @param lrnN RECV_ONLY
 * @param lrnAlpha RECV_ONLY
 * @param lrnBeta RECV_ONLY
 * @param lrnK RECV_ONLY
 */
cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                    unsigned *lrnN, double *lrnAlpha,
                                    double *lrnBeta, double *lrnK);
/**
 * @param lrnDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(lrnDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param normDesc SEND_ONLY
 * @param lrnMode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnLRNCrossChannelForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param handle SEND_ONLY
 * @param normDesc SEND_ONLY
 * @param mode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param means SEND_ONLY
 * @param temp SEND_ONLY
 * @param temp2 SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means,
    void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param derivedBnDesc SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param mode SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDeriveBNTensorDescriptor(
    cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc,
    cudnnBatchNormMode_t mode) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_data_type(derivedBnDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param bnScaleBiasMeanVarDesc SEND_ONLY
 * @param bnScale SEND_ONLY
 * @param bnBias SEND_ONLY
 * @param estimatedMean SEND_ONLY
 * @param estimatedVariance SEND_ONLY
 * @param epsilon SEND_ONLY
 */
cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon);
/**
 * @param derivedNormScaleBiasDesc SEND_ONLY
 * @param derivedNormMeanVarDesc SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param mode SEND_ONLY
 * @param groupCnt SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDeriveNormTensorDescriptor(
    cudnnTensorDescriptor_t derivedNormScaleBiasDesc,
    cudnnTensorDescriptor_t derivedNormMeanVarDesc,
    const cudnnTensorDescriptor_t xDesc, cudnnNormMode_t mode, int groupCnt) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_data_type(derivedNormScaleBiasDesc);
  forget_data_type(derivedNormMeanVarDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param normOps SEND_ONLY
 * @param algo SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param normScaleBiasDesc SEND_ONLY
 * @param normScale SEND_ONLY
 * @param normBias SEND_ONLY
 * @param normMeanVarDesc SEND_ONLY
 * @param estimatedMean SEND_ONLY
 * @param estimatedVariance SEND_ONLY
 * @param zDesc SEND_ONLY
 * @param z SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param groupCnt SEND_ONLY
 */
cudnnStatus_t cudnnNormalizationForwardInference(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y, double epsilon, int groupCnt);
/**
 * @param stDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t *stDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *stDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param stDesc SEND_ONLY
 * @param samplerType SEND_ONLY
 * @param dataType SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param dimA SEND_ONLY LENGTH:nbDims
 */
cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims, const int *dimA);
/**
 * @param stDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(stDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param stDesc SEND_ONLY
 * @param theta SEND_ONLY
 * @param grid SEND_ONLY
 */
cudnnStatus_t cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid);
/**
 * @param handle SEND_ONLY
 * @param stDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param grid SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnSpatialTfSamplerForward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *grid, const void *beta, cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param dropoutDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateDropoutDescriptor(
    cudnnDropoutDescriptor_t *dropoutDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *dropoutDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param dropoutDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(dropoutDesc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                        size_t *sizeInBytes);
/**
 * @param xdesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc,
                                              size_t *sizeInBytes);
/**
 * @param dropoutDesc SEND_ONLY
 * @param handle SEND_ONLY
 * @param dropout SEND_ONLY
 * @param states SEND_ONLY
 * @param stateSizeInBytes SEND_ONLY
 * @param seed SEND_ONLY
 */
cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                        cudnnHandle_t handle, float dropout,
                                        void *states, size_t stateSizeInBytes,
                                        unsigned long long seed);
/**
 * @param dropoutDesc SEND_ONLY
 * @param handle SEND_ONLY
 * @param dropout SEND_ONLY
 * @param states SEND_ONLY
 * @param stateSizeInBytes SEND_ONLY
 * @param seed SEND_ONLY
 */
cudnnStatus_t
cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                              cudnnHandle_t handle, float dropout, void *states,
                              size_t stateSizeInBytes, unsigned long long seed);
/**
 * @param dropoutDesc SEND_ONLY
 * @param handle SEND_ONLY
 * @param dropout RECV_ONLY
 * @param states RECV_ONLY
 * @param seed RECV_ONLY
 */
cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                        cudnnHandle_t handle, float *dropout,
                                        void **states,
                                        unsigned long long *seed);
/**
 * @param handle SEND_ONLY
 * @param dropoutDesc SEND_ONLY
 * @param xdesc SEND_ONLY
 * @param x SEND_ONLY
 * @param ydesc SEND_ONLY
 * @param y SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,
                                  const cudnnDropoutDescriptor_t dropoutDesc,
                                  const cudnnTensorDescriptor_t xdesc,
                                  const void *x,
                                  const cudnnTensorDescriptor_t ydesc, void *y,
                                  void *reserveSpace,
                                  size_t reserveSpaceSizeInBytes);
cudnnStatus_t cudnnOpsVersionCheck();
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param mode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 */
cudnnStatus_t cudnnSoftmaxBackward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
/**
 * @param handle SEND_ONLY
 * @param poolingDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 */
cudnnStatus_t cudnnPoolingBackward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
/**
 * @param handle SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 */
cudnnStatus_t cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
/**
 * @param handle SEND_ONLY
 * @param normDesc SEND_ONLY
 * @param lrnMode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 */
cudnnStatus_t cudnnLRNCrossChannelBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
/**
 * @param handle SEND_ONLY
 * @param normDesc SEND_ONLY
 * @param mode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dXdMeansDesc SIZE:scaling_width(dXdMeansDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param means SEND_ONLY
 * @param dy SEND_ONLY
 * @param temp SEND_ONLY
 * @param temp2 SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dXdMeansDesc SIZE:scaling_width(dXdMeansDesc)
 * @param dXdMeansDesc SEND_ONLY
 * @param dx SEND_ONLY
 * @param dMeans SEND_ONLY
 */
cudnnStatus_t cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means,
    const void *dy, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, void *dx, void *dMeans);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param bnOps SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param zDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param bnScaleBiasMeanVarDesc SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param bnOps SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dzDesc SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param dBnScaleBiasDesc SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param bnOps SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param bnScaleBiasMeanVarDesc SEND_ONLY
 * @param bnScale SEND_ONLY
 * @param bnBias SEND_ONLY
 * @param exponentialAverageFactor SEND_ONLY
 * @param resultRunningMean SEND_ONLY
 * @param resultRunningVariance SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param resultSaveMean SEND_ONLY
 * @param resultSaveInvVariance SEND_ONLY
 */
cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param bnOps SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param xData SEND_ONLY
 * @param zDesc SEND_ONLY
 * @param zData SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param yData SEND_ONLY
 * @param bnScaleBiasMeanVarDesc SEND_ONLY
 * @param bnScale SEND_ONLY
 * @param bnBias SEND_ONLY
 * @param exponentialAverageFactor SEND_ONLY
 * @param resultRunningMean SEND_ONLY
 * @param resultRunningVariance SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param resultSaveMean SEND_ONLY
 * @param resultSaveInvVariance SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param workspace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);
// clang-format off
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param alphaDataDiff SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param betaDataDiff SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param alphaParamDiff SEND_ONLY SCALAR:dBnScaleBiasDesc SIZE:scaling_width(dBnScaleBiasDesc)
 * @param betaParamDiff SEND_ONLY SCALAR:dBnScaleBiasDesc SIZE:scaling_width(dBnScaleBiasDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 * @param dBnScaleBiasDesc SEND_ONLY
 * @param bnScale SEND_ONLY
 * @param dBnScaleResult SEND_ONLY
 * @param dBnBiasResult SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param savedMean SEND_ONLY
 * @param savedInvVariance SEND_ONLY
 */
// clang-format on
cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult, double epsilon,
    const void *savedMean, const void *savedInvVariance);
// clang-format off
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param bnOps SEND_ONLY
 * @param alphaDataDiff SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param betaDataDiff SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param alphaParamDiff SEND_ONLY SCALAR:dBnScaleBiasDesc SIZE:scaling_width(dBnScaleBiasDesc)
 * @param betaParamDiff SEND_ONLY SCALAR:dBnScaleBiasDesc SIZE:scaling_width(dBnScaleBiasDesc)
 * @param xDesc SEND_ONLY
 * @param xData SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param yData SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dyData SEND_ONLY
 * @param dzDesc SEND_ONLY
 * @param dzData SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param dxData SEND_ONLY
 * @param dBnScaleBiasDesc SEND_ONLY
 * @param bnScaleData SEND_ONLY
 * @param bnBiasData SEND_ONLY
 * @param dBnScaleData SEND_ONLY
 * @param dBnBiasData SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param savedMean SEND_ONLY
 * @param savedInvVariance SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 */
// clang-format on
cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScaleData,
    const void *bnBiasData, void *dBnScaleData, void *dBnBiasData,
    double epsilon, const void *savedMean, const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param normOps SEND_ONLY
 * @param algo SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param zDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param normScaleBiasDesc SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param normMeanVarDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 * @param groupCnt SEND_ONLY
 */
cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc, size_t *sizeInBytes,
    int groupCnt);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param normOps SEND_ONLY
 * @param algo SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dzDesc SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param dNormScaleBiasDesc SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param normMeanVarDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 * @param groupCnt SEND_ONLY
 */
cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t normMeanVarDesc, size_t *sizeInBytes,
    int groupCnt);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param normOps SEND_ONLY
 * @param algo SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 * @param groupCnt SEND_ONLY
 */
cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo, const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes, int groupCnt);
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param normOps SEND_ONLY
 * @param algo SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param xData SEND_ONLY
 * @param normScaleBiasDesc SEND_ONLY
 * @param normScale SEND_ONLY
 * @param normBias SEND_ONLY
 * @param exponentialAverageFactor SEND_ONLY
 * @param normMeanVarDesc SEND_ONLY
 * @param resultRunningMean SEND_ONLY
 * @param resultRunningVariance SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param resultSaveMean SEND_ONLY
 * @param resultSaveInvVariance SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param zDesc SEND_ONLY
 * @param zData SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param yData SEND_ONLY
 * @param workspace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 * @param groupCnt SEND_ONLY
 */
cudnnStatus_t cudnnNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t normScaleBiasDesc, const void *normScale,
    const void *normBias, double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc, void *resultRunningMean,
    void *resultRunningVariance, double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance, cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes, int groupCnt);
// clang-format off
/**
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 * @param normOps SEND_ONLY
 * @param algo SEND_ONLY
 * @param alphaDataDiff SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param betaDataDiff SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param alphaParamDiff SEND_ONLY SCALAR:dNormScaleBiasDesc SIZE:scaling_width(dNormScaleBiasDesc)
 * @param betaParamDiff SEND_ONLY SCALAR:dNormScaleBiasDesc SIZE:scaling_width(dNormScaleBiasDesc)
 * @param xDesc SEND_ONLY
 * @param xData SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param yData SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dyData SEND_ONLY
 * @param dzDesc SEND_ONLY
 * @param dzData SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param dxData SEND_ONLY
 * @param dNormScaleBiasDesc SEND_ONLY
 * @param normScaleData SEND_ONLY
 * @param normBiasData SEND_ONLY
 * @param dNormScaleData SEND_ONLY
 * @param dNormBiasData SEND_ONLY
 * @param epsilon SEND_ONLY
 * @param normMeanVarDesc SEND_ONLY
 * @param savedMean SEND_ONLY
 * @param savedInvVariance SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 * @param groupCnt SEND_ONLY
 */
// clang-format on
cudnnStatus_t cudnnNormalizationBackward(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo, const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dNormScaleBiasDesc, const void *normScaleData,
    const void *normBiasData, void *dNormScaleData, void *dNormBiasData,
    double epsilon, const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *savedMean, const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes, int groupCnt);
/**
 * @param handle SEND_ONLY
 * @param stDesc SEND_ONLY
 * @param dgrid SEND_ONLY
 * @param dtheta SEND_ONLY
 */
cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta);
/**
 * @param handle SEND_ONLY
 * @param stDesc SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 * @param alphaDgrid SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param grid SEND_ONLY
 * @param betaDgrid SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dgrid SEND_ONLY
 */
cudnnStatus_t cudnnSpatialTfSamplerBackward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid, const void *betaDgrid, void *dgrid);
/**
 * @param handle SEND_ONLY
 * @param dropoutDesc SEND_ONLY
 * @param dydesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param dxdesc SEND_ONLY
 * @param dx SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,
                                   const cudnnDropoutDescriptor_t dropoutDesc,
                                   const cudnnTensorDescriptor_t dydesc,
                                   const void *dy,
                                   const cudnnTensorDescriptor_t dxdesc,
                                   void *dx, void *reserveSpace,
                                   size_t reserveSpaceSizeInBytes);
/**
 * @param convDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateConvolutionDescriptor(
    cudnnConvolutionDescriptor_t *convDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *convDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param convDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyConvolutionDescriptor(
    cudnnConvolutionDescriptor_t convDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(convDesc);
  return return_value;
}
// clang-format on
/**
 * @param convDesc SEND_ONLY
 * @param mathType SEND_ONLY
 */
cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                          cudnnMathType_t mathType);
/**
 * @param convDesc SEND_ONLY
 * @param mathType RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                          cudnnMathType_t *mathType);
/**
 * @param convDesc SEND_ONLY
 * @param groupCount SEND_ONLY
 */
cudnnStatus_t
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                              int groupCount);
/**
 * @param convDesc SEND_ONLY
 * @param groupCount RECV_ONLY
 */
cudnnStatus_t
cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                              int *groupCount);
/**
 * @param convDesc SEND_ONLY
 * @param reorderType SEND_ONLY
 */
cudnnStatus_t
cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                               cudnnReorderType_t reorderType);
/**
 * @param convDesc SEND_ONLY
 * @param reorderType RECV_ONLY
 */
cudnnStatus_t
cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                               cudnnReorderType_t *reorderType);
/**
 * @param convDesc SEND_ONLY
 * @param pad_h SEND_ONLY
 * @param pad_w SEND_ONLY
 * @param u SEND_ONLY
 * @param v SEND_ONLY
 * @param dilation_h SEND_ONLY
 * @param dilation_w SEND_ONLY
 * @param mode SEND_ONLY
 * @param computeType SEND_ONLY
 */
cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v,
    int dilation_h, int dilation_w, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType);
/**
 * @param convDesc SEND_ONLY
 * @param pad_h RECV_ONLY
 * @param pad_w RECV_ONLY
 * @param u RECV_ONLY
 * @param v RECV_ONLY
 * @param dilation_h RECV_ONLY
 * @param dilation_w RECV_ONLY
 * @param mode RECV_ONLY
 * @param computeType RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolution2dDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w, int *u,
    int *v, int *dilation_h, int *dilation_w, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType);
/**
 * @param convDesc SEND_ONLY
 * @param arrayLength SEND_ONLY
 * @param padA SEND_ONLY LENGTH:arrayLength
 * @param filterStrideA SEND_ONLY LENGTH:arrayLength
 * @param dilationA SEND_ONLY LENGTH:arrayLength
 * @param mode SEND_ONLY
 * @param computeType SEND_ONLY
 */
cudnnStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int *padA,
    const int *filterStrideA, const int *dilationA, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType);
/**
 * @param convDesc SEND_ONLY
 * @param arrayLengthRequested SEND_ONLY
 * @param arrayLength RECV_ONLY
 * @param padA RECV_ONLY LENGTH:arrayLengthRequested
 * @param strideA RECV_ONLY LENGTH:arrayLengthRequested
 * @param dilationA RECV_ONLY LENGTH:arrayLengthRequested
 * @param mode RECV_ONLY
 * @param computeType RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolutionNdDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int *padA, int *strideA, int *dilationA,
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType);
/**
 * @param convDesc SEND_ONLY
 * @param inputTensorDesc SEND_ONLY
 * @param filterDesc SEND_ONLY
 * @param n RECV_ONLY
 * @param c RECV_ONLY
 * @param h RECV_ONLY
 * @param w RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w);
/**
 * @param convDesc SEND_ONLY
 * @param inputTensorDesc SEND_ONLY
 * @param filterDesc SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param tensorOuputDimA RECV_ONLY LENGTH:nbDims
 */
cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims, int *tensorOuputDimA);
/**
 * @param handle SEND_ONLY
 * @param count RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle,
                                                          int *count);
/**
 * @param handle SEND_ONLY
 * @param srcDesc SEND_ONLY
 * @param filterDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param destDesc SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 */
cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 */
cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param w SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param colBuffer SEND_ONLY
 */
cudnnStatus_t cudnnIm2Col(cudnnHandle_t handle,
                          const cudnnTensorDescriptor_t xDesc, const void *x,
                          const cudnnFilterDescriptor_t wDesc,
                          const cudnnConvolutionDescriptor_t convDesc,
                          void *colBuffer);
/**
 * @param handle SEND_ONLY
 * @param filterDesc SEND_ONLY
 * @param reorderType SEND_ONLY
 * @param filterData SEND_ONLY
 * @param reorderedFilterData SEND_ONLY
 * @param reorderBias SEND_ONLY
 * @param biasData SEND_ONLY
 * @param reorderedBiasData SEND_ONLY
 */
cudnnStatus_t cudnnReorderFilterAndBias(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias, const void *biasData,
    void *reorderedBiasData);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param w SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param beta SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param handle SEND_ONLY
 * @param alpha1 SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param w SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param alpha2 SEND_ONLY SCALAR:yDesc SIZE:scaling_width(yDesc)
 * @param zDesc SEND_ONLY
 * @param z SEND_ONLY
 * @param biasDesc SEND_ONLY
 * @param bias SEND_ONLY
 * @param activationDesc SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 */
cudnnStatus_t cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y);
/**
 * @param handle SEND_ONLY
 * @param count RECV_ONLY
 */
cudnnStatus_t
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                 int *count);
/**
 * @param handle SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 */
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults);
/**
 * @param handle SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param w SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param filterDesc SEND_ONLY
 * @param diffDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param gradDesc SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 */
cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults);
/**
 * @param handle SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param dxDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param wDesc SEND_ONLY
 * @param w SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dxDesc SIZE:scaling_width(dxDesc)
 * @param dxDesc SEND_ONLY
 * @param dx SEND_ONLY
 */
cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
/**
 * @param handle SEND_ONLY
 * @param filterDesc SEND_ONLY
 * @param diffDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param gradDesc SEND_ONLY
 * @param transformFormat SEND_ONLY
 * @param foldedFilterDesc SEND_ONLY
 * @param paddedDiffDesc SEND_ONLY
 * @param foldedConvDesc SEND_ONLY
 * @param foldedGradDesc SEND_ONLY
 * @param filterFoldTransDesc SEND_ONLY
 * @param diffPadTransDesc SEND_ONLY
 * @param gradFoldTransDesc SEND_ONLY
 * @param gradUnfoldTransDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_data_type(foldedFilterDesc);
  forget_data_type(paddedDiffDesc);
  forget_data_type(foldedGradDesc);
  return return_value;
}
// clang-format on
cudnnStatus_t cudnnCnnVersionCheck();
/**
 * @param handle SEND_ONLY
 * @param count RECV_ONLY
 */
cudnnStatus_t
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle,
                                                   int *count);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param dwDesc SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 */
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param dwDesc SEND_ONLY
 * @param dw SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param srcDesc SEND_ONLY
 * @param diffDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param gradDesc SEND_ONLY
 * @param requestedAlgoCount SEND_ONLY
 * @param returnedAlgoCount RECV_ONLY
 * @param perfResults RECV_ONLY LENGTH:requestedAlgoCount
 */
cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);
/**
 * @param handle SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param gradDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dwDesc SIZE:scaling_width(dwDesc)
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param convDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dwDesc SIZE:scaling_width(dwDesc)
 * @param dwDesc SEND_ONLY
 * @param dw SEND_ONLY
 */
cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw);
/**
 * @param handle SEND_ONLY
 * @param alpha SEND_ONLY SCALAR:dbDesc SIZE:scaling_width(dbDesc)
 * @param dyDesc SEND_ONLY
 * @param dy SEND_ONLY
 * @param beta SEND_ONLY SCALAR:dbDesc SIZE:scaling_width(dbDesc)
 * @param dbDesc SEND_ONLY
 * @param db SEND_ONLY
 */
cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                           const void *alpha,
                                           const cudnnTensorDescriptor_t dyDesc,
                                           const void *dy, const void *beta,
                                           const cudnnTensorDescriptor_t dbDesc,
                                           void *db);
/**
 * @param constPack RECV_ONLY
 * @param ops SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateFusedOpsConstParamPack(
    cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *constPack);
  }
  return return_value;
}
// clang-format on
/**
 * @param constPack SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(
    cudnnFusedOpsConstParamPack_t constPack) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(constPack);
  return return_value;
}
// clang-format on
cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(
    cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel, const void *param);
cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(
    const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel, void *param, int *isNULL);
/**
 * @param varPack RECV_ONLY
 * @param ops SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(
    cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *varPack);
  }
  return return_value;
}
// clang-format on
/**
 * @param varPack SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(
    cudnnFusedOpsVariantParamPack_t varPack) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(varPack);
  return return_value;
}
// clang-format on
cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(
    cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr);
cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(
    const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr);
/**
 * @param plan RECV_ONLY
 * @param ops SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateFusedOpsPlan(
    cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *plan);
  }
  return return_value;
}
// clang-format on
/**
 * @param plan SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(plan);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param plan SEND_ONLY
 * @param constPack SEND_ONLY
 * @param workspaceSizeInBytes RECV_ONLY
 */
cudnnStatus_t
cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack,
                      size_t *workspaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param plan SEND_ONLY
 * @param varPack SEND_ONLY
 */
cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t handle,
                                   const cudnnFusedOpsPlan_t plan,
                                   cudnnFusedOpsVariantParamPack_t varPack);
/**
 * @param rnnDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *rnnDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param rnnDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(rnnDesc);
  return return_value;
}
// clang-format on
/**
 * @param rnnDesc SEND_ONLY
 * @param algo SEND_ONLY
 * @param cellMode SEND_ONLY
 * @param biasMode SEND_ONLY
 * @param dirMode SEND_ONLY
 * @param inputMode SEND_ONLY
 * @param dataType SEND_ONLY
 * @param mathPrec SEND_ONLY
 * @param mathType SEND_ONLY
 * @param inputSize SEND_ONLY
 * @param hiddenSize SEND_ONLY
 * @param projSize SEND_ONLY
 * @param numLayers SEND_ONLY
 * @param dropoutDesc SEND_ONLY
 * @param auxFlags SEND_ONLY
 */
cudnnStatus_t cudnnSetRNNDescriptor_v8(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo, cudnnRNNMode_t cellMode,
    cudnnRNNBiasMode_t biasMode, cudnnDirectionMode_t dirMode,
    cudnnRNNInputMode_t inputMode, cudnnDataType_t dataType,
    cudnnDataType_t mathPrec, cudnnMathType_t mathType, int32_t inputSize,
    int32_t hiddenSize, int32_t projSize, int32_t numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, uint32_t auxFlags);
/**
 * @param rnnDesc SEND_ONLY
 * @param algo RECV_ONLY
 * @param cellMode RECV_ONLY
 * @param biasMode RECV_ONLY
 * @param dirMode RECV_ONLY
 * @param inputMode RECV_ONLY
 * @param dataType RECV_ONLY
 * @param mathPrec RECV_ONLY
 * @param mathType RECV_ONLY
 * @param inputSize RECV_ONLY
 * @param hiddenSize RECV_ONLY
 * @param projSize RECV_ONLY
 * @param numLayers RECV_ONLY
 * @param dropoutDesc RECV_ONLY
 * @param auxFlags RECV_ONLY
 */
cudnnStatus_t cudnnGetRNNDescriptor_v8(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo,
    cudnnRNNMode_t *cellMode, cudnnRNNBiasMode_t *biasMode,
    cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType, cudnnDataType_t *mathPrec,
    cudnnMathType_t *mathType, int32_t *inputSize, int32_t *hiddenSize,
    int32_t *projSize, int32_t *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc, uint32_t *auxFlags);
/**
 * @param rnnDesc SEND_ONLY
 * @param clipMode SEND_ONLY
 * @param clipNanOpt SEND_ONLY
 * @param lclip SEND_ONLY
 * @param rclip SEND_ONLY
 */
cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t rnnDesc,
                                 cudnnRNNClipMode_t clipMode,
                                 cudnnNanPropagation_t clipNanOpt, double lclip,
                                 double rclip);
/**
 * @param rnnDesc SEND_ONLY
 * @param clipMode SEND_ONLY
 * @param lclip SEND_ONLY
 * @param rclip SEND_ONLY
 */
cudnnStatus_t cudnnRNNSetClip_v9(cudnnRNNDescriptor_t rnnDesc,
                                 cudnnRNNClipMode_t clipMode, double lclip,
                                 double rclip);
/**
 * @param rnnDesc SEND_ONLY
 * @param clipMode RECV_ONLY
 * @param clipNanOpt RECV_ONLY
 * @param lclip RECV_ONLY
 * @param rclip RECV_ONLY
 */
cudnnStatus_t cudnnRNNGetClip_v8(cudnnRNNDescriptor_t rnnDesc,
                                 cudnnRNNClipMode_t *clipMode,
                                 cudnnNanPropagation_t *clipNanOpt,
                                 double *lclip, double *rclip);
/**
 * @param rnnDesc SEND_ONLY
 * @param clipMode RECV_ONLY
 * @param lclip RECV_ONLY
 * @param rclip RECV_ONLY
 */
cudnnStatus_t cudnnRNNGetClip_v9(cudnnRNNDescriptor_t rnnDesc,
                                 cudnnRNNClipMode_t *clipMode, double *lclip,
                                 double *rclip);
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param miniBatch SEND_ONLY
 */
cudnnStatus_t cudnnBuildRNNDynamic(cudnnHandle_t handle,
                                   cudnnRNNDescriptor_t rnnDesc, int miniBatch);
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param fwdMode SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param workSpaceSize RECV_ONLY NULLABLE
 * @param reserveSpaceSize RECV_ONLY NULLABLE
 */
cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t handle,
                                        cudnnRNNDescriptor_t rnnDesc,
                                        cudnnForwardMode_t fwdMode,
                                        cudnnRNNDataDescriptor_t xDesc,
                                        size_t *workSpaceSize,
                                        size_t *reserveSpaceSize);
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param weightSpaceSize RECV_ONLY
 */
cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle,
                                         cudnnRNNDescriptor_t rnnDesc,
                                         size_t *weightSpaceSize);
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param pseudoLayer SEND_ONLY
 * @param weightSpaceSize SEND_ONLY
 * @param weightSpace SEND_ONLY
 * @param linLayerID SEND_ONLY
 * @param mDesc SEND_ONLY
 * @param mAddr RECV_ONLY NULLABLE
 * @param bDesc SEND_ONLY
 * @param bAddr RECV_ONLY NULLABLE
 */
// clang-format off
cudnnStatus_t cudnnGetRNNWeightParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int32_t pseudoLayer,
    size_t weightSpaceSize, const void *weightSpace, int32_t linLayerID,
    cudnnTensorDescriptor_t mDesc, void **mAddr, cudnnTensorDescriptor_t bDesc,
    void **bAddr) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_data_type(mDesc);
  forget_data_type(bDesc);
  return return_value;
}
// clang-format on
/**
 * @param rnnDataDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateRNNDataDescriptor(
    cudnnRNNDataDescriptor_t *rnnDataDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *rnnDataDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param rnnDataDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(rnnDataDesc);
  return return_value;
}
// clang-format on
// clang-format off
/**
 * @param rnnDataDesc SEND_ONLY
 * @param dataType SEND_ONLY
 * @param layout SEND_ONLY
 * @param maxSeqLength SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param vectorSize SEND_ONLY
 * @param seqLengthArray SEND_ONLY LENGTH:batchSize
 * @param paddingFill SEND_ONLY SCALAR:rnnDataDesc SIZE:data_type_width(dataType)
 */
cudnnStatus_t cudnnSetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize,
    int vectorSize, const int *seqLengthArray, void *paddingFill) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_data_type(rnnDataDesc, dataType, 0, nullptr);
  }
  return return_value;
}
// clang-format on
// clang-format off
/**
 * @param rnnDataDesc SEND_ONLY
 * @param dataType RECV_ONLY
 * @param layout RECV_ONLY
 * @param maxSeqLength RECV_ONLY
 * @param batchSize RECV_ONLY
 * @param vectorSize RECV_ONLY
 * @param arrayLengthRequested SEND_ONLY
 * @param seqLengthArray RECV_ONLY LENGTH:arrayLengthRequested
 * @param paddingFill RECV_ONLY SCALAR:rnnDataDesc SIZE:recorded_width(rnnDataDesc)
 */
// clang-format on
cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                                        cudnnDataType_t *dataType,
                                        cudnnRNNDataLayout_t *layout,
                                        int *maxSeqLength, int *batchSize,
                                        int *vectorSize,
                                        int arrayLengthRequested,
                                        int *seqLengthArray, void *paddingFill);
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param fwdMode SEND_ONLY
 * @param devSeqLengths SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param hDesc SEND_ONLY
 * @param hx SEND_ONLY
 * @param hy SEND_ONLY
 * @param cDesc SEND_ONLY
 * @param cx SEND_ONLY
 * @param cy SEND_ONLY
 * @param weightSpaceSize SEND_ONLY
 * @param weightSpace SEND_ONLY
 * @param workSpaceSize SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param reserveSpaceSize SEND_ONLY
 * @param reserveSpace SEND_ONLY
 */
cudnnStatus_t
cudnnRNNForward(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                cudnnForwardMode_t fwdMode, const int32_t *devSeqLengths,
                cudnnRNNDataDescriptor_t xDesc, const void *x,
                cudnnRNNDataDescriptor_t yDesc, void *y,
                cudnnTensorDescriptor_t hDesc, const void *hx, void *hy,
                cudnnTensorDescriptor_t cDesc, const void *cx, void *cy,
                size_t weightSpaceSize, const void *weightSpace,
                size_t workSpaceSize, void *workSpace, size_t reserveSpaceSize,
                void *reserveSpace);
/**
 * @param seqDataDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateSeqDataDescriptor(
    cudnnSeqDataDescriptor_t *seqDataDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *seqDataDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param seqDataDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroySeqDataDescriptor(
    cudnnSeqDataDescriptor_t seqDataDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(seqDataDesc);
  return return_value;
}
// clang-format on
// clang-format off
/**
 * @param seqDataDesc SEND_ONLY
 * @param dataType SEND_ONLY
 * @param nbDims SEND_ONLY
 * @param dimA SEND_ONLY LENGTH:nbDims
 * @param axes SEND_ONLY LENGTH:nbDims
 * @param seqLengthArraySize SEND_ONLY
 * @param seqLengthArray SEND_ONLY LENGTH:seqLengthArraySize
 * @param paddingFill SEND_ONLY SCALAR:seqDataDesc SIZE:data_type_width(dataType)
 */
cudnnStatus_t cudnnSetSeqDataDescriptor(
    cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType, int nbDims,
    const int *dimA, const cudnnSeqDataAxis_t *axes, size_t seqLengthArraySize,
    const int *seqLengthArray, void *paddingFill) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_data_type(seqDataDesc, dataType, 0, nullptr);
  }
  return return_value;
}
// clang-format on
// clang-format off
/**
 * @param seqDataDesc SEND_ONLY
 * @param dataType RECV_ONLY
 * @param nbDims RECV_ONLY
 * @param nbDimsRequested SEND_ONLY
 * @param dimA RECV_ONLY LENGTH:nbDimsRequested
 * @param axes RECV_ONLY LENGTH:nbDimsRequested
 * @param seqLengthArraySize RECV_ONLY
 * @param seqLengthSizeRequested SEND_ONLY
 * @param seqLengthArray RECV_ONLY LENGTH:seqLengthSizeRequested
 * @param paddingFill RECV_ONLY SCALAR:seqDataDesc SIZE:recorded_width(seqDataDesc)
 */
// clang-format on
cudnnStatus_t cudnnGetSeqDataDescriptor(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested, int *dimA, cudnnSeqDataAxis_t *axes,
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int *seqLengthArray, void *paddingFill);
/**
 * @param attnDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *attnDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param attnDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(attnDesc);
  return return_value;
}
// clang-format on
/**
 * @param attnDesc SEND_ONLY
 * @param attnMode SEND_ONLY
 * @param nHeads SEND_ONLY
 * @param smScaler SEND_ONLY
 * @param dataType SEND_ONLY
 * @param computePrec SEND_ONLY
 * @param mathType SEND_ONLY
 * @param attnDropoutDesc SEND_ONLY
 * @param postDropoutDesc SEND_ONLY
 * @param qSize SEND_ONLY
 * @param kSize SEND_ONLY
 * @param vSize SEND_ONLY
 * @param qProjSize SEND_ONLY
 * @param kProjSize SEND_ONLY
 * @param vProjSize SEND_ONLY
 * @param oProjSize SEND_ONLY
 * @param qoMaxSeqLength SEND_ONLY
 * @param kvMaxSeqLength SEND_ONLY
 * @param maxBatchSize SEND_ONLY
 * @param maxBeamSize SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnSetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads,
    double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize, int vSize,
    int qProjSize, int kProjSize, int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_attn_windows(attnDesc, qoMaxSeqLength);
  }
  return return_value;
}
// clang-format on
/**
 * @param attnDesc SEND_ONLY
 * @param attnMode RECV_ONLY
 * @param nHeads RECV_ONLY
 * @param smScaler RECV_ONLY
 * @param dataType RECV_ONLY
 * @param computePrec RECV_ONLY
 * @param mathType RECV_ONLY
 * @param attnDropoutDesc RECV_ONLY
 * @param postDropoutDesc RECV_ONLY
 * @param qSize RECV_ONLY
 * @param kSize RECV_ONLY
 * @param vSize RECV_ONLY
 * @param qProjSize RECV_ONLY
 * @param kProjSize RECV_ONLY
 * @param vProjSize RECV_ONLY
 * @param oProjSize RECV_ONLY
 * @param qoMaxSeqLength RECV_ONLY
 * @param kvMaxSeqLength RECV_ONLY
 * @param maxBatchSize RECV_ONLY
 * @param maxBeamSize RECV_ONLY
 */
cudnnStatus_t cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads,
    double *smScaler, cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize,
    int *vSize, int *qProjSize, int *kProjSize, int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize,
    int *maxBeamSize);
/**
 * @param handle SEND_ONLY
 * @param attnDesc SEND_ONLY
 * @param weightSizeInBytes RECV_ONLY NULLABLE
 * @param workSpaceSizeInBytes RECV_ONLY NULLABLE
 * @param reserveSpaceSizeInBytes RECV_ONLY NULLABLE
 */
cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                                           const cudnnAttnDescriptor_t attnDesc,
                                           size_t *weightSizeInBytes,
                                           size_t *workSpaceSizeInBytes,
                                           size_t *reserveSpaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param attnDesc SEND_ONLY
 * @param wKind SEND_ONLY
 * @param weightSizeInBytes SEND_ONLY
 * @param weights SEND_ONLY
 * @param wDesc SEND_ONLY
 * @param wAddr RECV_ONLY NULLABLE
 */
// clang-format off
cudnnStatus_t cudnnGetMultiHeadAttnWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *weights, cudnnTensorDescriptor_t wDesc, void **wAddr) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_data_type(wDesc);
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
cudnnStatus_t cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx,
    const int *loWinIdx, const int *hiWinIdx, const int *devSeqLengthsQO,
    const int *devSeqLengthsKV, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace);
cudnnStatus_t cudnnAdvVersionCheck();
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param devSeqLengths SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param dy SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param dx SEND_ONLY
 * @param hDesc SEND_ONLY
 * @param hx SEND_ONLY
 * @param dhy SEND_ONLY
 * @param dhx SEND_ONLY
 * @param cDesc SEND_ONLY
 * @param cx SEND_ONLY
 * @param dcy SEND_ONLY
 * @param dcx SEND_ONLY
 * @param weightSpaceSize SEND_ONLY
 * @param weightSpace SEND_ONLY
 * @param workSpaceSize SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param reserveSpaceSize SEND_ONLY
 * @param reserveSpace SEND_ONLY
 */
cudnnStatus_t cudnnRNNBackwardData_v8(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    const int32_t *devSeqLengths, cudnnRNNDataDescriptor_t yDesc, const void *y,
    const void *dy, cudnnRNNDataDescriptor_t xDesc, void *dx,
    cudnnTensorDescriptor_t hDesc, const void *hx, const void *dhy, void *dhx,
    cudnnTensorDescriptor_t cDesc, const void *cx, const void *dcy, void *dcx,
    size_t weightSpaceSize, const void *weightSpace, size_t workSpaceSize,
    void *workSpace, size_t reserveSpaceSize, void *reserveSpace);
/**
 * @param handle SEND_ONLY
 * @param rnnDesc SEND_ONLY
 * @param addGrad SEND_ONLY
 * @param devSeqLengths SEND_ONLY
 * @param xDesc SEND_ONLY
 * @param x SEND_ONLY
 * @param hDesc SEND_ONLY
 * @param hx SEND_ONLY
 * @param yDesc SEND_ONLY
 * @param y SEND_ONLY
 * @param weightSpaceSize SEND_ONLY
 * @param dweightSpace SEND_ONLY
 * @param workSpaceSize SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param reserveSpaceSize SEND_ONLY
 * @param reserveSpace SEND_ONLY
 */
cudnnStatus_t cudnnRNNBackwardWeights_v8(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
    cudnnWgradMode_t addGrad, const int32_t *devSeqLengths,
    cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t hDesc, const void *hx,
    cudnnRNNDataDescriptor_t yDesc, const void *y, size_t weightSpaceSize,
    void *dweightSpace, size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace);
/**
 * @disabled
 */
cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int *loWinIdx, const int *hiWinIdx, const int *devSeqLengthsDQDO,
    const int *devSeqLengthsDKDV, const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc, void *dkeys,
    const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace);
/**
 * @param handle SEND_ONLY
 * @param attnDesc SEND_ONLY
 * @param addGrad SEND_ONLY
 * @param qDesc SEND_ONLY
 * @param queries SEND_ONLY
 * @param kDesc SEND_ONLY
 * @param keys SEND_ONLY
 * @param vDesc SEND_ONLY
 * @param values SEND_ONLY
 * @param doDesc SEND_ONLY
 * @param dout SEND_ONLY
 * @param weightSizeInBytes SEND_ONLY
 * @param weights SEND_ONLY
 * @param dweights SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param workSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 */
cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace);
/**
 * @param ctcLossDesc RECV_ONLY
 */
// clang-format off
cudnnStatus_t cudnnCreateCTCLossDescriptor(
    cudnnCTCLossDescriptor_t *ctcLossDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDNN_STATUS_SUCCESS) {
    note_handle_owner(conn, *ctcLossDesc);
  }
  return return_value;
}
// clang-format on
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType SEND_ONLY
 */
cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                        cudnnDataType_t compType);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType SEND_ONLY
 * @param normMode SEND_ONLY
 * @param gradMode SEND_ONLY
 */
cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                                          cudnnDataType_t compType,
                                          cudnnLossNormalizationMode_t normMode,
                                          cudnnNanPropagation_t gradMode);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType SEND_ONLY
 * @param normMode SEND_ONLY
 * @param gradMode SEND_ONLY
 * @param maxLabelLength SEND_ONLY
 */
cudnnStatus_t cudnnSetCTCLossDescriptor_v8(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode,
    int maxLabelLength);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType SEND_ONLY
 * @param normMode SEND_ONLY
 * @param ctcGradMode SEND_ONLY
 * @param maxLabelLength SEND_ONLY
 */
cudnnStatus_t cudnnSetCTCLossDescriptor_v9(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnCTCGradMode_t ctcGradMode,
    int maxLabelLength);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType RECV_ONLY
 */
cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                        cudnnDataType_t *compType);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType RECV_ONLY
 * @param normMode RECV_ONLY
 * @param gradMode RECV_ONLY
 */
cudnnStatus_t cudnnGetCTCLossDescriptorEx(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType RECV_ONLY
 * @param normMode RECV_ONLY
 * @param gradMode RECV_ONLY
 * @param maxLabelLength RECV_ONLY
 */
cudnnStatus_t cudnnGetCTCLossDescriptor_v8(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode,
    int *maxLabelLength);
/**
 * @param ctcLossDesc SEND_ONLY
 * @param compType RECV_ONLY
 * @param normMode RECV_ONLY
 * @param ctcGradMode RECV_ONLY
 * @param maxLabelLength RECV_ONLY
 */
cudnnStatus_t cudnnGetCTCLossDescriptor_v9(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnCTCGradMode_t *ctcGradMode,
    int *maxLabelLength);
/**
 * @param ctcLossDesc SEND_ONLY
 */
// clang-format off
cudnnStatus_t cudnnDestroyCTCLossDescriptor(
    cudnnCTCLossDescriptor_t ctcLossDesc) {
  cudnnStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(ctcLossDesc);
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
cudnnStatus_t cudnnCTCLoss(cudnnHandle_t handle,
                           const cudnnTensorDescriptor_t probsDesc,
                           const void *probs, const int *hostLabels,
                           const int *hostLabelLengths,
                           const int *hostInputLengths, void *costs,
                           const cudnnTensorDescriptor_t gradientsDesc,
                           void *gradients, cudnnCTCLossAlgo_t algo,
                           cudnnCTCLossDescriptor_t ctcLossDesc,
                           void *workspace, size_t workSpaceSizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param ctcLossDesc SEND_ONLY
 * @param probsDesc SEND_ONLY
 * @param probs SEND_ONLY
 * @param labels SEND_ONLY
 * @param labelLengths SEND_ONLY
 * @param inputLengths SEND_ONLY
 * @param costs SEND_ONLY
 * @param gradientsDesc SEND_ONLY
 * @param gradients SEND_ONLY
 * @param workSpaceSizeInBytes SEND_ONLY
 * @param workspace SEND_ONLY
 */
cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
                              cudnnCTCLossDescriptor_t ctcLossDesc,
                              const cudnnTensorDescriptor_t probsDesc,
                              const void *probs, const int *labels,
                              const int *labelLengths, const int *inputLengths,
                              void *costs,
                              const cudnnTensorDescriptor_t gradientsDesc,
                              void *gradients, size_t workSpaceSizeInBytes,
                              void *workspace);
/**
 * @disabled
 */
cudnnStatus_t cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes);
/**
 * @param handle SEND_ONLY
 * @param algo SEND_ONLY
 * @param ctcLossDesc SEND_ONLY
 * @param probsDesc SEND_ONLY
 * @param gradientsDesc SEND_ONLY
 * @param sizeInBytes RECV_ONLY
 */
cudnnStatus_t
cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
                                cudnnCTCLossDescriptor_t ctcLossDesc,
                                const cudnnTensorDescriptor_t probsDesc,
                                const cudnnTensorDescriptor_t gradientsDesc,
                                size_t *sizeInBytes);
#if CUDNN_VERSION >= 92200
/**
 * @guard CUDNN_VERSION >= 92200
 */
cudnnStatus_t cudnnSubquadraticOpsVersionCheck();
#endif
#if CUDNN_VERSION >= 92200
/**
 * @guard CUDNN_VERSION >= 92200
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param bias SEND_ONLY
 * @param y SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param activation SEND_ONLY
 */
cudnnStatus_t
cudnnCausalConv1dForward(cudaStream_t stream, const void *x, const void *weight,
                         const void *bias, void *y, int batch, int dim,
                         int seqLen, int kernelSize, cudnnDataType_t dataType,
                         cudnnCausalConv1dActivation_t activation);
#endif
#if CUDNN_VERSION >= 92200
/**
 * @guard CUDNN_VERSION >= 92200
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param bias SEND_ONLY
 * @param dy SEND_ONLY
 * @param dx SEND_ONLY
 * @param dweight SEND_ONLY
 * @param dbias SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param dwDataType SEND_ONLY
 * @param activation SEND_ONLY
 */
cudnnStatus_t cudnnCausalConv1dBackward(
    cudaStream_t stream, const void *x, const void *weight, const void *bias,
    const void *dy, void *dx, void *dweight, void *dbias, int batch, int dim,
    int seqLen, int kernelSize, cudnnDataType_t dataType,
    cudnnDataType_t dwDataType, cudnnCausalConv1dActivation_t activation);
#endif
#if CUDNN_VERSION >= 92400
/**
 * @guard CUDNN_VERSION >= 92400
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param bias SEND_ONLY
 * @param y SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param activation SEND_ONLY
 */
cudnnStatus_t cudnnCausalConv1dNwhForward(
    cudaStream_t stream, const void *x, const void *weight, const void *bias,
    void *y, int batch, int dim, int seqLen, int kernelSize,
    cudnnDataType_t dataType, cudnnCausalConv1dActivation_t activation);
#endif
#if CUDNN_VERSION >= 92400
/**
 * @guard CUDNN_VERSION >= 92400
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param bias SEND_ONLY
 * @param dy SEND_ONLY
 * @param dx SEND_ONLY
 * @param dweight SEND_ONLY
 * @param dbias SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param dwDataType SEND_ONLY
 * @param activation SEND_ONLY
 */
cudnnStatus_t cudnnCausalConv1dNwhBackward(
    cudaStream_t stream, const void *x, const void *weight, const void *bias,
    const void *dy, void *dx, void *dweight, void *dbias, int batch, int dim,
    int seqLen, int kernelSize, cudnnDataType_t dataType,
    cudnnDataType_t dwDataType, cudnnCausalConv1dActivation_t activation);
#endif
#if CUDNN_VERSION >= 92400
/**
 * @guard CUDNN_VERSION >= 92400
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weightsProj SEND_ONLY
 * @param weightsMixer SEND_ONLY
 * @param skipBias SEND_ONLY
 * @param y SEND_ONLY
 * @param yGated SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSizeProj SEND_ONLY
 * @param kernelSizeMixer SEND_ONLY
 * @param dataType SEND_ONLY
 */
cudnnStatus_t
cudnnB2BCausalConv1dForward(cudaStream_t stream, const void *x,
                            const void *weightsProj, const void *weightsMixer,
                            const void *skipBias, void *y, void *yGated,
                            int batch, int dim, int seqLen, int kernelSizeProj,
                            int kernelSizeMixer, cudnnDataType_t dataType);
#endif
#if CUDNN_VERSION >= 92400
/**
 * @guard CUDNN_VERSION >= 92400
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weightsProj SEND_ONLY
 * @param weightsMixer SEND_ONLY
 * @param skipBias SEND_ONLY
 * @param y SEND_ONLY
 * @param dy SEND_ONLY
 * @param dx SEND_ONLY
 * @param dweightsProj SEND_ONLY
 * @param dweightsMixer SEND_ONLY
 * @param dskipBias SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSizeProj SEND_ONLY
 * @param kernelSizeMixer SEND_ONLY
 * @param dataType SEND_ONLY
 * @param dwDataType SEND_ONLY
 */
cudnnStatus_t cudnnB2BCausalConv1dBackward(
    cudaStream_t stream, const void *x, const void *weightsProj,
    const void *weightsMixer, const void *skipBias, const void *y,
    const void *dy, void *dx, void *dweightsProj, void *dweightsMixer,
    void *dskipBias, int batch, int dim, int seqLen, int kernelSizeProj,
    int kernelSizeMixer, cudnnDataType_t dataType, cudnnDataType_t dwDataType);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param y SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 */
cudnnStatus_t cudnnFFTCausalConv1dForward(cudaStream_t stream, const void *x,
                                          const void *weight, void *y,
                                          int batch, int dim, int seqLen,
                                          int kernelSize,
                                          cudnnDataType_t dataType);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param dy SEND_ONLY
 * @param dx SEND_ONLY
 * @param dweight SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 */
cudnnStatus_t cudnnFFTCausalConv1dBackward(cudaStream_t stream, const void *x,
                                           const void *weight, const void *dy,
                                           void *dx, void *dweight, int batch,
                                           int dim, int seqLen, int kernelSize,
                                           cudnnDataType_t dataType);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param workspaceSizeInBytes RECV_ONLY
 * @param reserveSpaceSizeInBytes RECV_ONLY
 */
cudnnStatus_t cudnnLongFFTCausalConv1dGetBufferSizes(
    int batch, int dim, int seqLen, int kernelSize, cudnnDataType_t dataType,
    size_t *workspaceSizeInBytes, size_t *reserveSpaceSizeInBytes);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param x SEND_ONLY
 * @param weight SEND_ONLY
 * @param y SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param workspace SEND_ONLY
 * @param workspaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnLongFFTCausalConv1dForward(
    cudaStream_t stream, const void *x, const void *weight, void *y, int batch,
    int dim, int seqLen, int kernelSize, cudnnDataType_t dataType,
    void *workspace, size_t workspaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param dy SEND_ONLY
 * @param dx SEND_ONLY
 * @param dweight SEND_ONLY
 * @param batch SEND_ONLY
 * @param dim SEND_ONLY
 * @param seqLen SEND_ONLY
 * @param kernelSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param workspace SEND_ONLY
 * @param workspaceSizeInBytes SEND_ONLY
 * @param reserveSpace SEND_ONLY
 * @param reserveSpaceSizeInBytes SEND_ONLY
 */
cudnnStatus_t cudnnLongFFTCausalConv1dBackward(
    cudaStream_t stream, const void *dy, void *dx, void *dweight, int batch,
    int dim, int seqLen, int kernelSize, cudnnDataType_t dataType,
    void *workspace, size_t workspaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param graph SEND_ONLY DEREF
 * @param nodeFeatures SEND_ONLY
 * @param edgeFeatures SEND_ONLY
 * @param concatFeatures SEND_ONLY
 * @param output SEND_ONLY
 * @param outPositions SEND_ONLY
 * @param nodeFeatDim SEND_ONLY
 * @param edgeFeatDim SEND_ONLY
 * @param concatFeatDim SEND_ONLY
 * @param dataType SEND_ONLY
 * @param aggOp SEND_ONLY
 */
cudnnStatus_t
cudnnGnnAggSimpleForward(cudaStream_t stream, const cudnnGnnCscGraph_t *graph,
                         const void *nodeFeatures, const void *edgeFeatures,
                         const void *concatFeatures, void *output,
                         void *outPositions, int nodeFeatDim, int edgeFeatDim,
                         int concatFeatDim, cudnnDataType_t dataType,
                         cudnnGnnAggOp_t aggOp);
#endif
#if CUDNN_VERSION >= 92600
/**
 * @guard CUDNN_VERSION >= 92600
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param graph SEND_ONLY DEREF
 * @param gradOutput SEND_ONLY
 * @param outPositions SEND_ONLY
 * @param gradNodeFeatures SEND_ONLY
 * @param gradEdgeFeatures SEND_ONLY
 * @param gradConcatFeatures SEND_ONLY
 * @param nodeFeatDim SEND_ONLY
 * @param edgeFeatDim SEND_ONLY
 * @param concatFeatDim SEND_ONLY
 * @param dataType SEND_ONLY
 * @param aggOp SEND_ONLY
 */
cudnnStatus_t
cudnnGnnAggSimpleBackward(cudaStream_t stream, const cudnnGnnCscGraph_t *graph,
                          const void *gradOutput, const void *outPositions,
                          void *gradNodeFeatures, void *gradEdgeFeatures,
                          void *gradConcatFeatures, int nodeFeatDim,
                          int edgeFeatDim, int concatFeatDim,
                          cudnnDataType_t dataType, cudnnGnnAggOp_t aggOp);
#endif
