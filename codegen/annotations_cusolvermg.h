#include <cusolverMg.h>

// cuSOLVERMg API. A cuSOLVERMg handle, device grid and matrix descriptor are
// addresses in one server's library and route to the connection that created
// them. Device ordinals in a device selection or grid are the caller's
// virtual ordinals, rewritten to the server's before they travel; a handle or
// grid cannot span servers. Each distributed matrix is a host array of device
// pointers, one per device of the descriptor's grid, and a call's workspace
// array is sized the same way. W, info and the workspace size are host
// memory.

/**
 * @param handle RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverMgCreate(cusolverMgHandle_t* handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *handle);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverMgDestroy(cusolverMgHandle_t handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param nbDevices SEND_ONLY
 * @param deviceId SEND_ONLY LENGTH:nbDevices
 */
// clang-format off
cusolverStatus_t cusolverMgDeviceSelect(cusolverMgHandle_t handle, int nbDevices, int *deviceId) {
  std::vector<int> server_ids = server_ordinals(deviceId, nbDevices);
  deviceId = server_ids.data();
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled client
 * @param grid RECV_ONLY
 * @param numRowDevices SEND_ONLY
 * @param numColDevices SEND_ONLY
 * @param deviceId SEND_ONLY LENGTH:numRowDevices*numColDevices
 * @param mapping SEND_ONLY
 */
cusolverStatus_t cusolverMgCreateDeviceGrid(cudaLibMgGrid_t *grid,
                                            int32_t numRowDevices,
                                            int32_t numColDevices,
                                            const int32_t *deviceId,
                                            cusolverMgGridMapping_t mapping);
/**
 * @param grid SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverMgDestroyGrid(cudaLibMgGrid_t grid) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(grid);
  return return_value;
}
// clang-format on
/**
 * @param desc RECV_ONLY
 * @param numRows SEND_ONLY
 * @param numCols SEND_ONLY
 * @param rowBlockSize SEND_ONLY
 * @param colBlockSize SEND_ONLY
 * @param dataType SEND_ONLY
 * @param grid SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverMgCreateMatrixDesc(cudaLibMgMatrixDesc_t* desc, int64_t numRows, int64_t numCols, int64_t rowBlockSize, int64_t colBlockSize, cudaDataType dataType, const cudaLibMgGrid_t grid) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *desc);
    note_matrix_grid(*desc, grid);
  }
  return return_value;
}
// clang-format on
/**
 * @param desc SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverMgDestroyMatrixDesc(cudaLibMgMatrixDesc_t desc) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(desc);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param W SEND_ONLY LENGTH:(W!=nullptr)*N*data_type_width(dataTypeW)
 * @param dataTypeW SEND_ONLY
 * @param computeType SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverMgSyevd_bufferSize(
    cusolverMgHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int N, void **array_d_A, int IA, int JA, cudaLibMgMatrixDesc_t descrA,
    void *W, cudaDataType dataTypeW, cudaDataType computeType, int64_t *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param W SEND_RECV LENGTH:N*data_type_width(dataTypeW)
 * @param dataTypeW SEND_ONLY
 * @param computeType SEND_ONLY
 * @param array_d_work SEND_ONLY LENGTH:mg_devices(descrA)
 * @param lwork SEND_ONLY
 * @param info SEND_RECV
 */
cusolverStatus_t cusolverMgSyevd(cusolverMgHandle_t handle,
                                 cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                 int N, void **array_d_A, int IA, int JA,
                                 cudaLibMgMatrixDesc_t descrA, void *W,
                                 cudaDataType dataTypeW,
                                 cudaDataType computeType, void **array_d_work,
                                 int64_t lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param M SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param array_d_IPIV SEND_ONLY LENGTH:mg_devices(descrA)
 * @param computeType SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverMgGetrf_bufferSize(cusolverMgHandle_t handle, int M, int N,
                           void **array_d_A, int IA, int JA,
                           cudaLibMgMatrixDesc_t descrA, int **array_d_IPIV,
                           cudaDataType computeType, int64_t *lwork);
/**
 * @param handle SEND_ONLY
 * @param M SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param array_d_IPIV SEND_ONLY LENGTH:mg_devices(descrA)
 * @param computeType SEND_ONLY
 * @param array_d_work SEND_ONLY LENGTH:mg_devices(descrA)
 * @param lwork SEND_ONLY
 * @param info SEND_RECV
 */
cusolverStatus_t cusolverMgGetrf(cusolverMgHandle_t handle, int M, int N,
                                 void **array_d_A, int IA, int JA,
                                 cudaLibMgMatrixDesc_t descrA,
                                 int **array_d_IPIV, cudaDataType computeType,
                                 void **array_d_work, int64_t lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param TRANS SEND_ONLY
 * @param N SEND_ONLY
 * @param NRHS SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param array_d_IPIV SEND_ONLY LENGTH:mg_devices(descrA)
 * @param array_d_B SEND_ONLY LENGTH:mg_devices(descrB)
 * @param IB SEND_ONLY
 * @param JB SEND_ONLY
 * @param descrB SEND_ONLY
 * @param computeType SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverMgGetrs_bufferSize(
    cusolverMgHandle_t handle, cublasOperation_t TRANS, int N, int NRHS,
    void **array_d_A, int IA, int JA, cudaLibMgMatrixDesc_t descrA,
    int **array_d_IPIV, void **array_d_B, int IB, int JB,
    cudaLibMgMatrixDesc_t descrB, cudaDataType computeType, int64_t *lwork);
/**
 * @param handle SEND_ONLY
 * @param TRANS SEND_ONLY
 * @param N SEND_ONLY
 * @param NRHS SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param array_d_IPIV SEND_ONLY LENGTH:mg_devices(descrA)
 * @param array_d_B SEND_ONLY LENGTH:mg_devices(descrB)
 * @param IB SEND_ONLY
 * @param JB SEND_ONLY
 * @param descrB SEND_ONLY
 * @param computeType SEND_ONLY
 * @param array_d_work SEND_ONLY LENGTH:mg_devices(descrA)
 * @param lwork SEND_ONLY
 * @param info SEND_RECV
 */
cusolverStatus_t cusolverMgGetrs(cusolverMgHandle_t handle,
                                 cublasOperation_t TRANS, int N, int NRHS,
                                 void **array_d_A, int IA, int JA,
                                 cudaLibMgMatrixDesc_t descrA,
                                 int **array_d_IPIV, void **array_d_B, int IB,
                                 int JB, cudaLibMgMatrixDesc_t descrB,
                                 cudaDataType computeType, void **array_d_work,
                                 int64_t lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param computeType SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverMgPotrf_bufferSize(cusolverMgHandle_t handle,
                                            cublasFillMode_t uplo, int N,
                                            void **array_d_A, int IA, int JA,
                                            cudaLibMgMatrixDesc_t descrA,
                                            cudaDataType computeType,
                                            int64_t *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param computeType SEND_ONLY
 * @param array_d_work SEND_ONLY LENGTH:mg_devices(descrA)
 * @param lwork SEND_ONLY
 * @param h_info SEND_RECV
 */
cusolverStatus_t cusolverMgPotrf(cusolverMgHandle_t handle,
                                 cublasFillMode_t uplo, int N, void **array_d_A,
                                 int IA, int JA, cudaLibMgMatrixDesc_t descrA,
                                 cudaDataType computeType, void **array_d_work,
                                 int64_t lwork, int *h_info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param array_d_B SEND_ONLY LENGTH:mg_devices(descrB)
 * @param IB SEND_ONLY
 * @param JB SEND_ONLY
 * @param descrB SEND_ONLY
 * @param computeType SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverMgPotrs_bufferSize(cusolverMgHandle_t handle, cublasFillMode_t uplo,
                           int n, int nrhs, void **array_d_A, int IA, int JA,
                           cudaLibMgMatrixDesc_t descrA, void **array_d_B,
                           int IB, int JB, cudaLibMgMatrixDesc_t descrB,
                           cudaDataType computeType, int64_t *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param array_d_B SEND_ONLY LENGTH:mg_devices(descrB)
 * @param IB SEND_ONLY
 * @param JB SEND_ONLY
 * @param descrB SEND_ONLY
 * @param computeType SEND_ONLY
 * @param array_d_work SEND_ONLY LENGTH:mg_devices(descrA)
 * @param lwork SEND_ONLY
 * @param h_info SEND_RECV
 */
cusolverStatus_t cusolverMgPotrs(cusolverMgHandle_t handle,
                                 cublasFillMode_t uplo, int n, int nrhs,
                                 void **array_d_A, int IA, int JA,
                                 cudaLibMgMatrixDesc_t descrA, void **array_d_B,
                                 int IB, int JB, cudaLibMgMatrixDesc_t descrB,
                                 cudaDataType computeType, void **array_d_work,
                                 int64_t lwork, int *h_info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param computeType SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverMgPotri_bufferSize(cusolverMgHandle_t handle,
                                            cublasFillMode_t uplo, int N,
                                            void **array_d_A, int IA, int JA,
                                            cudaLibMgMatrixDesc_t descrA,
                                            cudaDataType computeType,
                                            int64_t *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param N SEND_ONLY
 * @param array_d_A SEND_ONLY LENGTH:mg_devices(descrA)
 * @param IA SEND_ONLY
 * @param JA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param computeType SEND_ONLY
 * @param array_d_work SEND_ONLY LENGTH:mg_devices(descrA)
 * @param lwork SEND_ONLY
 * @param h_info SEND_RECV
 */
cusolverStatus_t cusolverMgPotri(cusolverMgHandle_t handle,
                                 cublasFillMode_t uplo, int N, void **array_d_A,
                                 int IA, int JA, cudaLibMgMatrixDesc_t descrA,
                                 cudaDataType computeType, void **array_d_work,
                                 int64_t lwork, int *h_info);
