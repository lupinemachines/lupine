#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

// cuSOLVER API (cusolverDn, cusolverSp, the low-level cusolverSp preview and
// cusolverRf). The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libcusolver
// by name at runtime. A handle, parameter set, info structure or IRS object
// is an address in the server's library and routes to the connection that
// created it; a call with none goes to the runtime's current device.
//
// Matrices, vectors, pivots, workspaces and devInfo are device pointers and
// travel as addresses. Host memory is marshalled by size: workspace sizes,
// eigenvalue counts, sweeps and residuals come back as values; a generic
// call's host workspace (bufferOnHost) is its workspaceInBytesOnHost bytes;
// vl and vu are one value of dataTypeW. A batched call's array of matrix
// pointers is itself device memory. A cusolverSp routine named *Host takes CSR
// arrays, right-hand sides and results in host memory sized by its dimensions
// and nnz. The low-level *Host routines size the arrays they do not bound by
// parameter (the Extract outputs, the workspace) from what the client recorded
// at analysis, buffer-info and nnz time for that info structure. cusolverRf's
// batched zero-pivot output is one host entry per batch entry recorded at
// setup. Outputs travel both ways,
// so bytes the library does not write keep the caller's contents.
//
// Each @guard names the first cuSOLVER release that declares the call with
// its current signature. The header's logger declarations are left out: the
// library does not export them.

/**
 * @param type SEND_ONLY
 * @param value SEND_RECV
 */
cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int *value);
/**
 * @param version SEND_RECV
 */
cusolverStatus_t cusolverGetVersion(int *version);
/**
 * @param handle RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
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
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param streamId SEND_ONLY
 */
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle,
                                     cudaStream_t streamId);
/**
 * @param handle SEND_ONLY
 * @param streamId RECV_ONLY
 */
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle,
                                     cudaStream_t *streamId);
/**
 * @guard CUSOLVER_VERSION >= 11500
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 */
cusolverStatus_t
cusolverDnSetDeterministicMode(cusolverDnHandle_t handle,
                               cusolverDeterministicMode_t mode);
/**
 * @guard CUSOLVER_VERSION >= 11500
 * @param handle SEND_ONLY
 * @param mode SEND_RECV
 */
cusolverStatus_t
cusolverDnGetDeterministicMode(cusolverDnHandle_t handle,
                               cusolverDeterministicMode_t *mode);
/**
 * @guard CUSOLVER_VERSION >= 12003
 * @param handle SEND_ONLY
 * @param mode SEND_ONLY
 */
cusolverStatus_t cusolverDnSetMathMode(cusolverDnHandle_t handle,
                                       cusolverMathMode_t mode);
/**
 * @guard CUSOLVER_VERSION >= 12003
 * @param handle SEND_ONLY
 * @param mode SEND_RECV
 */
cusolverStatus_t cusolverDnGetMathMode(cusolverDnHandle_t handle,
                                       cusolverMathMode_t *mode);
/**
 * @guard CUSOLVER_VERSION >= 12003
 * @param handle SEND_ONLY
 * @param strategy SEND_ONLY
 */
cusolverStatus_t
cusolverDnSetEmulationStrategy(cusolverDnHandle_t handle,
                               cudaEmulationStrategy_t strategy);
/**
 * @guard CUSOLVER_VERSION >= 12003
 * @param handle SEND_ONLY
 * @param strategy SEND_RECV
 */
cusolverStatus_t
cusolverDnGetEmulationStrategy(cusolverDnHandle_t handle,
                               cudaEmulationStrategy_t *strategy);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param control SEND_ONLY
 */
cusolverStatus_t cusolverDnSetFixedPointEmulationMantissaControl(
    cusolverDnHandle_t handle, cudaEmulationMantissaControl_t control);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param control SEND_RECV
 */
cusolverStatus_t cusolverDnGetFixedPointEmulationMantissaControl(
    cusolverDnHandle_t handle, cudaEmulationMantissaControl_t *control);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param mantissaBitCount SEND_ONLY
 */
cusolverStatus_t
cusolverDnSetFixedPointEmulationMaxMantissaBitCount(cusolverDnHandle_t handle,
                                                    int mantissaBitCount);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param mantissaBitCount SEND_RECV
 */
cusolverStatus_t
cusolverDnGetFixedPointEmulationMaxMantissaBitCount(cusolverDnHandle_t handle,
                                                    int *mantissaBitCount);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param mantissaBitOffset SEND_ONLY
 */
cusolverStatus_t
cusolverDnSetFixedPointEmulationMantissaBitOffset(cusolverDnHandle_t handle,
                                                  int mantissaBitOffset);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param mantissaBitOffset SEND_RECV
 */
cusolverStatus_t
cusolverDnGetFixedPointEmulationMantissaBitOffset(cusolverDnHandle_t handle,
                                                  int *mantissaBitOffset);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param mask SEND_ONLY
 */
cusolverStatus_t cusolverDnSetEmulationSpecialValuesSupport(
    cusolverDnHandle_t handle, cudaEmulationSpecialValuesSupport_t mask);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param mask SEND_RECV
 */
cusolverStatus_t cusolverDnGetEmulationSpecialValuesSupport(
    cusolverDnHandle_t handle, cudaEmulationSpecialValuesSupport_t *mask);
/**
 * @param params_ptr RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnIRSParamsCreate(cusolverDnIRSParams_t* params_ptr) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *params_ptr);
  }
  return return_value;
}
// clang-format on
/**
 * @param params SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(params);
  return return_value;
}
// clang-format on
/**
 * @param params SEND_ONLY
 * @param refinement_solver SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetRefinementSolver(
    cusolverDnIRSParams_t params, cusolverIRSRefinement_t refinement_solver);
/**
 * @param params SEND_ONLY
 * @param solver_main_precision SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetSolverMainPrecision(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision);
/**
 * @param params SEND_ONLY
 * @param solver_lowest_precision SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetSolverLowestPrecision(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_lowest_precision);
/**
 * @param params SEND_ONLY
 * @param solver_main_precision SEND_ONLY
 * @param solver_lowest_precision SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetSolverPrecisions(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision,
    cusolverPrecType_t solver_lowest_precision);
/**
 * @param params SEND_ONLY
 * @param val SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params,
                                           double val);
/**
 * @param params SEND_ONLY
 * @param val SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params,
                                                double val);
/**
 * @param params SEND_ONLY
 * @param maxiters SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t params,
                                                cusolver_int_t maxiters);
/**
 * @param params SEND_ONLY
 * @param maxiters_inner SEND_ONLY
 */
cusolverStatus_t
cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t params,
                                    cusolver_int_t maxiters_inner);
/**
 * @param params SEND_ONLY
 * @param maxiters SEND_RECV
 */
cusolverStatus_t cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t params,
                                                cusolver_int_t *maxiters);
/**
 * @param params SEND_ONLY
 */
cusolverStatus_t
cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params);
/**
 * @param params SEND_ONLY
 */
cusolverStatus_t
cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params);
/**
 * @param infos SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(infos);
  return return_value;
}
// clang-format on
/**
 * @param infos_ptr RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t* infos_ptr) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *infos_ptr);
  }
  return return_value;
}
// clang-format on
/**
 * @param infos SEND_ONLY
 * @param niters SEND_RECV
 */
cusolverStatus_t cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t infos,
                                             cusolver_int_t *niters);
/**
 * @param infos SEND_ONLY
 * @param outer_niters SEND_RECV
 */
cusolverStatus_t cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t infos,
                                                  cusolver_int_t *outer_niters);
/**
 * @param infos SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos);
/**
 * @disabled
 */
cusolverStatus_t
cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos,
                                     void **residual_history);
/**
 * @param infos SEND_ONLY
 * @param maxiters SEND_RECV
 */
cusolverStatus_t cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t infos,
                                               cusolver_int_t *maxiters);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZZgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuDoubleComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZCgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuDoubleComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZKgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuDoubleComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZEgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuDoubleComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZYgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuDoubleComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCCgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCEgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCKgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCYgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, cuComplex *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDDgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, double *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  double *dB, cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDSgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, double *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  double *dB, cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDHgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, double *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  double *dB, cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDBgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, double *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  double *dB, cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDXgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, double *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  double *dB, cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSSgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, float *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  float *dB, cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSHgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, float *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  float *dB, cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSBgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, float *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  float *dB, cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSXgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                  cusolver_int_t nrhs, float *dA,
                                  cusolver_int_t ldda, cusolver_int_t *dipiv,
                                  float *dB, cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZZgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZCgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZKgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZEgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZYgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuDoubleComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCCgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCKgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCEgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCYgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    cuComplex *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDDgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDSgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDHgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDBgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDXgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs,
    double *dA, cusolver_int_t ldda, cusolver_int_t *dipiv, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnSSgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnSHgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnSBgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dipiv SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnSXgesv_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float *dA,
    cusolver_int_t ldda, cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
    float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZZgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuDoubleComplex *dA, cusolver_int_t ldda,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZCgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuDoubleComplex *dA, cusolver_int_t ldda,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZKgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuDoubleComplex *dA, cusolver_int_t ldda,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZEgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuDoubleComplex *dA, cusolver_int_t ldda,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnZYgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuDoubleComplex *dA, cusolver_int_t ldda,
                                  cuDoubleComplex *dB, cusolver_int_t lddb,
                                  cuDoubleComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCCgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuComplex *dA, cusolver_int_t ldda,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCKgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuComplex *dA, cusolver_int_t ldda,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCEgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuComplex *dA, cusolver_int_t ldda,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnCYgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  cuComplex *dA, cusolver_int_t ldda,
                                  cuComplex *dB, cusolver_int_t lddb,
                                  cuComplex *dX, cusolver_int_t lddx,
                                  void *dWorkspace, size_t lwork_bytes,
                                  cusolver_int_t *iter, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDDgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  double *dA, cusolver_int_t ldda, double *dB,
                                  cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDSgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  double *dA, cusolver_int_t ldda, double *dB,
                                  cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDHgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  double *dA, cusolver_int_t ldda, double *dB,
                                  cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDBgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  double *dA, cusolver_int_t ldda, double *dB,
                                  cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnDXgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  double *dA, cusolver_int_t ldda, double *dB,
                                  cusolver_int_t lddb, double *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSSgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  float *dA, cusolver_int_t ldda, float *dB,
                                  cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSHgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  float *dA, cusolver_int_t ldda, float *dB,
                                  cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSBgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  float *dA, cusolver_int_t ldda, float *dB,
                                  cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param iter SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnSXgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                  cusolver_int_t n, cusolver_int_t nrhs,
                                  float *dA, cusolver_int_t ldda, float *dB,
                                  cusolver_int_t lddb, float *dX,
                                  cusolver_int_t lddx, void *dWorkspace,
                                  size_t lwork_bytes, cusolver_int_t *iter,
                                  cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZZgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZCgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZKgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZEgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnZYgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCCgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCKgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCEgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnCYgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda, cuComplex *dB,
    cusolver_int_t lddb, cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDDgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDSgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDHgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDBgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnDXgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, double *dA, cusolver_int_t ldda, double *dB,
    cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t
cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                            cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                            cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                            float *dX, cusolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t
cusolverDnSHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                            cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                            cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                            float *dX, cusolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t
cusolverDnSBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                            cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                            cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                            float *dX, cusolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t
cusolverDnSXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                            cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                            cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                            float *dX, cusolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param gesv_irs_params SEND_ONLY
 * @param gesv_irs_infos SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param niters SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSXgesv(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params,
    cusolverDnIRSInfos_t gesv_irs_infos, cusolver_int_t n, cusolver_int_t nrhs,
    void *dA, cusolver_int_t ldda, void *dB, cusolver_int_t lddb, void *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    cusolver_int_t *niters, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t handle,
                                               cusolverDnIRSParams_t params,
                                               cusolver_int_t n,
                                               cusolver_int_t nrhs,
                                               size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param gels_irs_params SEND_ONLY
 * @param gels_irs_infos SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dA SEND_ONLY
 * @param ldda SEND_ONLY
 * @param dB SEND_ONLY
 * @param lddb SEND_ONLY
 * @param dX SEND_ONLY
 * @param lddx SEND_ONLY
 * @param dWorkspace SEND_ONLY
 * @param lwork_bytes SEND_ONLY
 * @param niters SEND_RECV
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnIRSXgels(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params,
    cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n,
    cusolver_int_t nrhs, void *dA, cusolver_int_t ldda, void *dB,
    cusolver_int_t lddb, void *dX, cusolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, cusolver_int_t *niters, cusolver_int_t *d_info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param lwork_bytes SEND_RECV
 */
cusolverStatus_t cusolverDnIRSXgels_bufferSize(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t m,
    cusolver_int_t n, cusolver_int_t nrhs, size_t *lwork_bytes);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             float *A, int lda, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             double *A, int lda, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuComplex *A, int lda, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuDoubleComplex *A, int lda,
                                             int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, float *Workspace, int Lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, double *Workspace, int Lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, cuComplex *Workspace, int Lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *Workspace, int Lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, int nrhs,
                                  const float *A, int lda, float *B, int ldb,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, int nrhs,
                                  const double *A, int lda, double *B, int ldb,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCpotrs(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, int nrhs,
                                  const cuComplex *A, int lda, cuComplex *B,
                                  int ldb, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, int nrhs,
                                  const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param infoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n,
                                         float **Aarray, int lda,
                                         int *infoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param infoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n,
                                         double **Aarray, int lda,
                                         int *infoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param infoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n,
                                         cuComplex **Aarray, int lda,
                                         int *infoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param Aarray SEND_ONLY
 * @param lda SEND_ONLY
 * @param infoArray SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n,
                                         cuDoubleComplex **Aarray, int lda,
                                         int *infoArray, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param d_info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSpotrsBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n, int nrhs,
                                         float **A, int lda, float **B, int ldb,
                                         int *d_info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param d_info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDpotrsBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n, int nrhs,
                                         double **A, int lda, double **B,
                                         int ldb, int *d_info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param d_info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCpotrsBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n, int nrhs,
                                         cuComplex **A, int lda, cuComplex **B,
                                         int ldb, int *d_info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param d_info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnZpotrsBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo, int n, int nrhs,
                                         cuDoubleComplex **A, int lda,
                                         cuDoubleComplex **B, int ldb,
                                         int *d_info, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             float *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             double *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuComplex *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuDoubleComplex *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSpotri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, float *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDpotri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, double *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCpotri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, cuComplex *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZpotri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXtrtri_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag,
    int64_t n, cudaDataType dataTypeA, void *A, int64_t lda,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param diag SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnXtrtri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, cublasDiagType_t diag,
                                  int64_t n, cudaDataType dataTypeA, void *A,
                                  int64_t lda, void *bufferOnDevice,
                                  size_t workspaceInBytesOnDevice,
                                  void *bufferOnHost,
                                  size_t workspaceInBytesOnHost, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSlauum_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             float *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDlauum_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             double *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnClauum_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuComplex *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZlauum_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuDoubleComplex *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSlauum(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, float *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDlauum(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, double *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnClauum(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, cuComplex *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZlauum(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, float *A, int lda,
                                             int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, double *A, int lda,
                                             int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, cuComplex *A, int lda,
                                             int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, cuDoubleComplex *A, int lda,
                                             int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n,
                                  float *A, int lda, float *Workspace,
                                  int *devIpiv, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n,
                                  double *A, int lda, double *Workspace,
                                  int *devIpiv, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n,
                                  cuComplex *A, int lda, cuComplex *Workspace,
                                  int *devIpiv, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n,
                                  cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *Workspace, int *devIpiv,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param k1 SEND_ONLY
 * @param k2 SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param incx SEND_ONLY
 */
cusolverStatus_t cusolverDnSlaswp(cusolverDnHandle_t handle, int n, float *A,
                                  int lda, int k1, int k2, const int *devIpiv,
                                  int incx);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param k1 SEND_ONLY
 * @param k2 SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param incx SEND_ONLY
 */
cusolverStatus_t cusolverDnDlaswp(cusolverDnHandle_t handle, int n, double *A,
                                  int lda, int k1, int k2, const int *devIpiv,
                                  int incx);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param k1 SEND_ONLY
 * @param k2 SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param incx SEND_ONLY
 */
cusolverStatus_t cusolverDnClaswp(cusolverDnHandle_t handle, int n,
                                  cuComplex *A, int lda, int k1, int k2,
                                  const int *devIpiv, int incx);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param k1 SEND_ONLY
 * @param k2 SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param incx SEND_ONLY
 */
cusolverStatus_t cusolverDnZlaswp(cusolverDnHandle_t handle, int n,
                                  cuDoubleComplex *A, int lda, int k1, int k2,
                                  const int *devIpiv, int incx);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans, int n, int nrhs,
                                  const float *A, int lda, const int *devIpiv,
                                  float *B, int ldb, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans, int n, int nrhs,
                                  const double *A, int lda, const int *devIpiv,
                                  double *B, int ldb, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans, int n, int nrhs,
                                  const cuComplex *A, int lda,
                                  const int *devIpiv, cuComplex *B, int ldb,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param devIpiv SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans, int n, int nrhs,
                                  const cuDoubleComplex *A, int lda,
                                  const int *devIpiv, cuDoubleComplex *B,
                                  int ldb, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, float *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, double *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, cuComplex *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, cuDoubleComplex *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param TAU SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  float *A, int lda, float *TAU,
                                  float *Workspace, int Lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param TAU SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  double *A, int lda, double *TAU,
                                  double *Workspace, int Lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param TAU SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  cuComplex *A, int lda, cuComplex *TAU,
                                  cuComplex *Workspace, int Lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param TAU SEND_ONLY
 * @param Workspace SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *TAU,
                                  cuDoubleComplex *Workspace, int Lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int k, const float *A,
                                             int lda, const float *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int k, const double *A,
                                             int lda, const double *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int k, const cuComplex *A,
                                             int lda, const cuComplex *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int k,
                                             const cuDoubleComplex *A, int lda,
                                             const cuDoubleComplex *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n,
                                  int k, float *A, int lda, const float *tau,
                                  float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n,
                                  int k, double *A, int lda, const double *tau,
                                  double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n,
                                  int k, cuComplex *A, int lda,
                                  const cuComplex *tau, cuComplex *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n,
                                  int k, cuDoubleComplex *A, int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                            cublasOperation_t trans, int m, int n, int k,
                            const float *A, int lda, const float *tau,
                            const float *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                            cublasOperation_t trans, int m, int n, int k,
                            const double *A, int lda, const double *tau,
                            const double *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                            cublasOperation_t trans, int m, int n, int k,
                            const cuComplex *A, int lda, const cuComplex *tau,
                            const cuComplex *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZunmqr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans, int m, int n, int k,
                                  const float *A, int lda, const float *tau,
                                  float *C, int ldc, float *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans, int m, int n, int k,
                                  const double *A, int lda, const double *tau,
                                  double *C, int ldc, double *work, int lwork,
                                  int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCunmqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans, int m, int n, int k,
                                  const cuComplex *A, int lda,
                                  const cuComplex *tau, cuComplex *C, int ldc,
                                  cuComplex *work, int lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t
cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                 cublasOperation_t trans, int m, int n, int k,
                 const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau,
                 cuDoubleComplex *C, int ldc, cuDoubleComplex *work, int lwork,
                 int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n,
                                             float *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n,
                                             double *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n,
                                             cuComplex *A, int lda, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n,
                                             cuDoubleComplex *A, int lda,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSsytrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, int *ipiv, float *work, int lwork,
                                  int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDsytrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, int *ipiv, double *work, int lwork,
                                  int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCsytrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, int *ipiv, cuComplex *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZsytrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda, int *ipiv,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXsytrs_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs,
    cudaDataType dataTypeA, const void *A, int64_t lda, const int64_t *ipiv,
    cudaDataType dataTypeB, void *B, int64_t ldb,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXsytrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n,
                 int64_t nrhs, cudaDataType dataTypeA, const void *A,
                 int64_t lda, const int64_t *ipiv, cudaDataType dataTypeB,
                 void *B, int64_t ldb, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSsytri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             float *A, int lda, const int *ipiv,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDsytri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             double *A, int lda,
                                             const int *ipiv, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCsytri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuComplex *A, int lda,
                                             const int *ipiv, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZsytri_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             cuDoubleComplex *A, int lda,
                                             const int *ipiv, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSsytri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, const int *ipiv, float *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDsytri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, const int *ipiv, double *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCsytri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, const int *ipiv, cuComplex *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZsytri(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda, const int *ipiv,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param Lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *Lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param D SEND_ONLY
 * @param E SEND_ONLY
 * @param TAUQ SEND_ONLY
 * @param TAUP SEND_ONLY
 * @param Work SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n,
                                  float *A, int lda, float *D, float *E,
                                  float *TAUQ, float *TAUP, float *Work,
                                  int Lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param D SEND_ONLY
 * @param E SEND_ONLY
 * @param TAUQ SEND_ONLY
 * @param TAUP SEND_ONLY
 * @param Work SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n,
                                  double *A, int lda, double *D, double *E,
                                  double *TAUQ, double *TAUP, double *Work,
                                  int Lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param D SEND_ONLY
 * @param E SEND_ONLY
 * @param TAUQ SEND_ONLY
 * @param TAUP SEND_ONLY
 * @param Work SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n,
                                  cuComplex *A, int lda, float *D, float *E,
                                  cuComplex *TAUQ, cuComplex *TAUP,
                                  cuComplex *Work, int Lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param D SEND_ONLY
 * @param E SEND_ONLY
 * @param TAUQ SEND_ONLY
 * @param TAUP SEND_ONLY
 * @param Work SEND_ONLY
 * @param Lwork SEND_ONLY
 * @param devInfo SEND_ONLY
 */
cusolverStatus_t cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n,
                                  cuDoubleComplex *A, int lda, double *D,
                                  double *E, cuDoubleComplex *TAUQ,
                                  cuDoubleComplex *TAUP, cuDoubleComplex *Work,
                                  int Lwork, int *devInfo);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side, int m,
                                             int n, int k, const float *A,
                                             int lda, const float *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side, int m,
                                             int n, int k, const double *A,
                                             int lda, const double *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side, int m,
                                             int n, int k, const cuComplex *A,
                                             int lda, const cuComplex *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                            int m, int n, int k, const cuDoubleComplex *A,
                            int lda, const cuDoubleComplex *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSorgbr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, int m, int n, int k,
                                  float *A, int lda, const float *tau,
                                  float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDorgbr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, int m, int n, int k,
                                  double *A, int lda, const double *tau,
                                  double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCungbr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, int m, int n, int k,
                                  cuComplex *A, int lda, const cuComplex *tau,
                                  cuComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZungbr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, int m, int n, int k,
                                  cuDoubleComplex *A, int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const float *A, int lda,
                                             const float *d, const float *e,
                                             const float *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const double *A, int lda,
                                             const double *d, const double *e,
                                             const double *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const cuComplex *A, int lda,
                                             const float *d, const float *e,
                                             const cuComplex *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const cuDoubleComplex *A, int lda,
                                             const double *d, const double *e,
                                             const cuDoubleComplex *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSsytrd(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, float *d, float *e, float *tau,
                                  float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDsytrd(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, double *d, double *e, double *tau,
                                  double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnChetrd(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, float *d, float *e, cuComplex *tau,
                                  cuComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param d SEND_ONLY
 * @param e SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZhetrd(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda, double *d,
                                  double *e, cuDoubleComplex *tau,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const float *A, int lda,
                                             const float *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const double *A, int lda,
                                             const double *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const cuComplex *A, int lda,
                                             const cuComplex *tau, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             const cuDoubleComplex *A, int lda,
                                             const cuDoubleComplex *tau,
                                             int *lwork);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSorgtr(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float *A,
                                  int lda, const float *tau, float *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDorgtr(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double *A,
                                  int lda, const double *tau, double *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCungtr(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, cuComplex *A,
                                  int lda, const cuComplex *tau,
                                  cuComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZungtr(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  cuDoubleComplex *A, int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSormtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const float *A, int lda,
    const float *tau, const float *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDormtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const double *A, int lda,
    const double *tau, const double *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCunmtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const cuComplex *A, int lda,
    const cuComplex *tau, const cuComplex *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZunmtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, int m, int n, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSormtr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int m, int n,
                                  float *A, int lda, float *tau, float *C,
                                  int ldc, float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDormtr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int m, int n,
                                  double *A, int lda, double *tau, double *C,
                                  int ldc, double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCunmtr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int m, int n,
                                  cuComplex *A, int lda, cuComplex *tau,
                                  cuComplex *C, int ldc, cuComplex *work,
                                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param side SEND_ONLY
 * @param uplo SEND_ONLY
 * @param trans SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param tau SEND_ONLY
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZunmtr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, int m, int n,
                                  cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *tau, cuDoubleComplex *C,
                                  int ldc, cuDoubleComplex *work, int lwork,
                                  int *info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobvt SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param VT SEND_ONLY
 * @param ldvt SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param rwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n, float *A,
                                  int lda, float *S, float *U, int ldu,
                                  float *VT, int ldvt, float *work, int lwork,
                                  float *rwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobvt SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param VT SEND_ONLY
 * @param ldvt SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param rwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n, double *A,
                                  int lda, double *S, double *U, int ldu,
                                  double *VT, int ldvt, double *work, int lwork,
                                  double *rwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobvt SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param VT SEND_ONLY
 * @param ldvt SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param rwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n, cuComplex *A,
                                  int lda, float *S, cuComplex *U, int ldu,
                                  cuComplex *VT, int ldvt, cuComplex *work,
                                  int lwork, float *rwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobvt SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param VT SEND_ONLY
 * @param ldvt SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param rwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
                 int m, int n, cuDoubleComplex *A, int lda, double *S,
                 cuDoubleComplex *U, int ldu, cuDoubleComplex *VT, int ldvt,
                 cuDoubleComplex *work, int lwork, double *rwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const float *A, int lda,
                                             const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const double *A, int lda,
                                             const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const cuComplex *A, int lda,
                                             const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const cuDoubleComplex *A, int lda,
                                             const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, float *A, int lda, float *W,
                                  float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, double *A, int lda, double *W,
                                  double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuComplex *A, int lda, float *W,
                                  cuComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuDoubleComplex *A, int lda, double *W,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const float *A, int lda, float vl, float vu,
    int il, int iu, int *meig, const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const double *A, int lda, double vl,
    double vu, int il, int iu, int *meig, const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnCheevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const cuComplex *A, int lda, float vl,
    float vu, int il, int iu, int *meig, const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZheevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, double vl,
    double vu, int il, int iu, int *meig, const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                  float *A, int lda, float vl, float vu, int il, int iu,
                  int *meig, float *W, float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                  double *A, int lda, double vl, double vu, int il, int iu,
                  int *meig, double *W, double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                  cuComplex *A, int lda, float vl, float vu, int il, int iu,
                  int *meig, float *W, cuComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                  cuDoubleComplex *A, int lda, double vl, double vu, int il,
                  int iu, int *meig, double *W, cuDoubleComplex *work,
                  int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnSsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float *A,
    int lda, const float *B, int ldb, float vl, float vu, int il, int iu,
    int *meig, const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnDsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double *A,
    int lda, const double *B, int ldb, double vl, double vu, int il, int iu,
    int *meig, const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnChegvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex *A,
    int lda, const cuComplex *B, int ldb, float vl, float vu, int il, int iu,
    int *meig, const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZhegvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
    double vl, double vu, int il, int iu, int *meig, const double *W,
    int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                  cusolverEigMode_t jobz, cusolverEigRange_t range,
                  cublasFillMode_t uplo, int n, float *A, int lda, float *B,
                  int ldb, float vl, float vu, int il, int iu, int *meig,
                  float *W, float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                  cusolverEigMode_t jobz, cusolverEigRange_t range,
                  cublasFillMode_t uplo, int n, double *A, int lda, double *B,
                  int ldb, double vl, double vu, int il, int iu, int *meig,
                  double *W, double *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                  cusolverEigMode_t jobz, cusolverEigRange_t range,
                  cublasFillMode_t uplo, int n, cuComplex *A, int lda,
                  cuComplex *B, int ldb, float vl, float vu, int il, int iu,
                  int *meig, float *W, cuComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY
 * @param vu SEND_ONLY
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZhegvdx(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex *A,
    int lda, cuDoubleComplex *B, int ldb, double vl, double vu, int il, int iu,
    int *meig, double *W, cuDoubleComplex *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype,
                            cusolverEigMode_t jobz, cublasFillMode_t uplo,
                            int n, const float *A, int lda, const float *B,
                            int ldb, const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t
cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype,
                            cusolverEigMode_t jobz, cublasFillMode_t uplo,
                            int n, const double *A, int lda, const double *B,
                            int ldb, const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnChegvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const float *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 */
cusolverStatus_t cusolverDnZhegvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb, const double *W, int *lwork);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnSsygvd(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, float *A, int lda, float *B, int ldb,
                                  float *W, float *work, int lwork, int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnDsygvd(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, double *A, int lda, double *B, int ldb,
                                  double *W, double *work, int lwork,
                                  int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnChegvd(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuComplex *A, int lda, cuComplex *B,
                                  int ldb, float *W, cuComplex *work, int lwork,
                                  int *info);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnZhegvd(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb, double *W,
                                  cuDoubleComplex *work, int lwork, int *info);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param d_B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param d_W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXsygvd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *d_A, int64_t lda,
    cudaDataType dataTypeB, const void *d_B, int64_t ldb,
    cudaDataType dataTypeW, const void *d_W, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param d_B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param d_W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param d_info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXsygvd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cusolverEigType_t itype, cusolverEigMode_t jobz,
                 cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                 void *d_A, int64_t lda, cudaDataType dataTypeB, void *d_B,
                 int64_t ldb, cudaDataType dataTypeW, void *d_W,
                 cudaDataType computeType, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *d_info);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param d_B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY LENGTH:(vl!=nullptr)*data_type_width(dataTypeW)
 * @param vu SEND_ONLY LENGTH:(vu!=nullptr)*data_type_width(dataTypeW)
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param dataTypeW SEND_ONLY
 * @param d_W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *d_A, int64_t lda,
    cudaDataType dataTypeB, const void *d_B, int64_t ldb, void *vl, void *vu,
    int64_t il, int64_t iu, int64_t *meig, cudaDataType dataTypeW,
    const void *d_W, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 12100
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param d_B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param vl SEND_ONLY LENGTH:(vl!=nullptr)*data_type_width(dataTypeW)
 * @param vu SEND_ONLY LENGTH:(vu!=nullptr)*data_type_width(dataTypeW)
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig SEND_RECV
 * @param dataTypeW SEND_ONLY
 * @param d_W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnXsygvdx(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *d_A,
    int64_t lda, cudaDataType dataTypeB, void *d_B, int64_t ldb, void *vl,
    void *vu, int64_t il, int64_t iu, int64_t *meig, cudaDataType dataTypeW,
    void *d_W, cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *d_info);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 * @param tolerance SEND_ONLY
 */
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info,
                                              double tolerance);
/**
 * @param info SEND_ONLY
 * @param max_sweeps SEND_ONLY
 */
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps);
/**
 * @param info SEND_ONLY
 * @param sort_eig SEND_ONLY
 */
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param residual SEND_RECV
 */
cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle,
                                             syevjInfo_t info,
                                             double *residual);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param executed_sweeps SEND_RECV
 */
cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle,
                                           syevjInfo_t info,
                                           int *executed_sweeps);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const float *A, int lda, const float *W, int *lwork,
    syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double *A, int lda, const double *W, int *lwork,
    syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCheevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuComplex *A, int lda, const float *W, int *lwork,
    syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnZheevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const cuDoubleComplex *A, int lda, const double *W, int *lwork,
    syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n, float *A,
                                         int lda, float *W, float *work,
                                         int lwork, int *info,
                                         syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n,
                                         double *A, int lda, double *W,
                                         double *work, int lwork, int *info,
                                         syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCheevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n,
                                         cuComplex *A, int lda, float *W,
                                         cuComplex *work, int lwork, int *info,
                                         syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t
cusolverDnZheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                        cublasFillMode_t uplo, int n, cuDoubleComplex *A,
                        int lda, double *W, cuDoubleComplex *work, int lwork,
                        int *info, syevjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const float *A, int lda,
                                             const float *W, int *lwork,
                                             syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const double *A, int lda,
                                             const double *W, int *lwork,
                                             syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const cuComplex *A, int lda,
                                             const float *W, int *lwork,
                                             syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             const cuDoubleComplex *A, int lda,
                                             const double *W, int *lwork,
                                             syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnSsyevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, float *A, int lda, float *W,
                                  float *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnDsyevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, double *A, int lda, double *W,
                                  double *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnCheevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuComplex *A, int lda, float *W,
                                  cuComplex *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnZheevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuDoubleComplex *A, int lda, double *W,
                                  cuDoubleComplex *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnSsygvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const float *A, int lda, const float *B,
    int ldb, const float *W, int *lwork, syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnDsygvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, const double *A, int lda, const double *B,
    int ldb, const double *W, int *lwork, syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t
cusolverDnChegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype,
                            cusolverEigMode_t jobz, cublasFillMode_t uplo,
                            int n, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const float *W,
                            int *lwork, syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t
cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype,
                            cusolverEigMode_t jobz, cublasFillMode_t uplo,
                            int n, const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb, const double *W,
                            int *lwork, syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnSsygvj(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, float *A, int lda, float *B, int ldb,
                                  float *W, float *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnDsygvj(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, double *A, int lda, double *B, int ldb,
                                  double *W, double *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnChegvj(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuComplex *A, int lda, cuComplex *B,
                                  int ldb, float *W, cuComplex *work, int lwork,
                                  int *info, syevjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param itype SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param W SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnZhegvj(cusolverDnHandle_t handle,
                                  cusolverEigType_t itype,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb, double *W,
                                  cuDoubleComplex *work, int lwork, int *info,
                                  syevjInfo_t params);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 * @param tolerance SEND_ONLY
 */
cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info,
                                               double tolerance);
/**
 * @param info SEND_ONLY
 * @param max_sweeps SEND_ONLY
 */
cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info,
                                               int max_sweeps);
/**
 * @param info SEND_ONLY
 * @param sort_svd SEND_ONLY
 */
cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param residual SEND_RECV
 */
cusolverStatus_t cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle,
                                              gesvdjInfo_t info,
                                              double *residual);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param executed_sweeps SEND_RECV
 */
cusolverStatus_t cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle,
                                            gesvdjInfo_t info,
                                            int *executed_sweeps);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const float *A, int lda, const float *S, const float *U, int ldu,
    const float *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const double *A, int lda, const double *S, const double *U, int ldu,
    const double *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const cuComplex *A, int lda, const float *S, const cuComplex *U, int ldu,
    const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params,
    int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
    const cuDoubleComplex *A, int lda, const double *S,
    const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv,
    int *lwork, gesvdjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvdjBatched(cusolverDnHandle_t handle,
                                          cusolverEigMode_t jobz, int m, int n,
                                          float *A, int lda, float *S, float *U,
                                          int ldu, float *V, int ldv,
                                          float *work, int lwork, int *info,
                                          gesvdjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t
cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                         int m, int n, double *A, int lda, double *S, double *U,
                         int ldu, double *V, int ldv, double *work, int lwork,
                         int *info, gesvdjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvdjBatched(cusolverDnHandle_t handle,
                                          cusolverEigMode_t jobz, int m, int n,
                                          cuComplex *A, int lda, float *S,
                                          cuComplex *U, int ldu, cuComplex *V,
                                          int ldv, cuComplex *work, int lwork,
                                          int *info, gesvdjInfo_t params,
                                          int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t
cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                         int m, int n, cuDoubleComplex *A, int lda, double *S,
                         cuDoubleComplex *U, int ldu, cuDoubleComplex *V,
                         int ldv, cuDoubleComplex *work, int lwork, int *info,
                         gesvdjInfo_t params, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const float *A, int lda, const float *S, const float *U, int ldu,
    const float *V, int ldv, int *lwork, gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnDgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const double *A, int lda, const double *S, const double *U, int ldu,
    const double *V, int ldv, int *lwork, gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
    const cuComplex *A, int lda, const float *S, const cuComplex *U, int ldu,
    const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param lwork SEND_RECV
 * @param params SEND_ONLY
 */
cusolverStatus_t
cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                             int econ, int m, int n, const cuDoubleComplex *A,
                             int lda, const double *S, const cuDoubleComplex *U,
                             int ldu, const cuDoubleComplex *V, int ldv,
                             int *lwork, gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvdj(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz, int econ, int m,
                                   int n, float *A, int lda, float *S, float *U,
                                   int ldu, float *V, int ldv, float *work,
                                   int lwork, int *info, gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnDgesvdj(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz, int econ, int m,
                                   int n, double *A, int lda, double *S,
                                   double *U, int ldu, double *V, int ldv,
                                   double *work, int lwork, int *info,
                                   gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvdj(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz, int econ, int m,
                                   int n, cuComplex *A, int lda, float *S,
                                   cuComplex *U, int ldu, cuComplex *V, int ldv,
                                   cuComplex *work, int lwork, int *info,
                                   gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param S SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param info SEND_ONLY
 * @param params SEND_ONLY
 */
cusolverStatus_t cusolverDnZgesvdj(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz, int econ, int m,
                                   int n, cuDoubleComplex *A, int lda,
                                   double *S, cuDoubleComplex *U, int ldu,
                                   cuDoubleComplex *V, int ldv,
                                   cuDoubleComplex *work, int lwork, int *info,
                                   gesvdjInfo_t params);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param lwork SEND_RECV
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const float *d_A, int lda, long long int strideA, const float *d_S,
    long long int strideS, const float *d_U, int ldu, long long int strideU,
    const float *d_V, int ldv, long long int strideV, int *lwork,
    int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param lwork SEND_RECV
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const double *d_A, int lda, long long int strideA, const double *d_S,
    long long int strideS, const double *d_U, int ldu, long long int strideU,
    const double *d_V, int ldv, long long int strideV, int *lwork,
    int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param lwork SEND_RECV
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuComplex *d_A, int lda, long long int strideA, const float *d_S,
    long long int strideS, const cuComplex *d_U, int ldu, long long int strideU,
    const cuComplex *d_V, int ldv, long long int strideV, int *lwork,
    int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param lwork SEND_RECV
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuDoubleComplex *d_A, int lda, long long int strideA,
    const double *d_S, long long int strideS, const cuDoubleComplex *d_U,
    int ldu, long long int strideU, const cuDoubleComplex *d_V, int ldv,
    long long int strideV, int *lwork, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param d_work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param d_info SEND_ONLY
 * @param h_R_nrmF SEND_RECV LENGTH:batchSize
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnSgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const float *d_A, int lda, long long int strideA, float *d_S,
    long long int strideS, float *d_U, int ldu, long long int strideU,
    float *d_V, int ldv, long long int strideV, float *d_work, int lwork,
    int *d_info, double *h_R_nrmF, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param d_work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param d_info SEND_ONLY
 * @param h_R_nrmF SEND_RECV LENGTH:batchSize
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnDgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const double *d_A, int lda, long long int strideA, double *d_S,
    long long int strideS, double *d_U, int ldu, long long int strideU,
    double *d_V, int ldv, long long int strideV, double *d_work, int lwork,
    int *d_info, double *h_R_nrmF, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param d_work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param d_info SEND_ONLY
 * @param h_R_nrmF SEND_RECV LENGTH:batchSize
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnCgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuComplex *d_A, int lda, long long int strideA, float *d_S,
    long long int strideS, cuComplex *d_U, int ldu, long long int strideU,
    cuComplex *d_V, int ldv, long long int strideV, cuComplex *d_work,
    int lwork, int *d_info, double *h_R_nrmF, int batchSize);
/**
 * @param handle SEND_ONLY
 * @param jobz SEND_ONLY
 * @param rank SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param d_A SEND_ONLY
 * @param lda SEND_ONLY
 * @param strideA SEND_ONLY
 * @param d_S SEND_ONLY
 * @param strideS SEND_ONLY
 * @param d_U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param strideU SEND_ONLY
 * @param d_V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param strideV SEND_ONLY
 * @param d_work SEND_ONLY
 * @param lwork SEND_ONLY
 * @param d_info SEND_ONLY
 * @param h_R_nrmF SEND_RECV LENGTH:batchSize
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnZgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,
    const cuDoubleComplex *d_A, int lda, long long int strideA, double *d_S,
    long long int strideS, cuDoubleComplex *d_U, int ldu, long long int strideU,
    cuDoubleComplex *d_V, int ldv, long long int strideV,
    cuDoubleComplex *d_work, int lwork, int *d_info, double *h_R_nrmF,
    int batchSize);
/**
 * @param params RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *params);
  }
  return return_value;
}
// clang-format on
/**
 * @param params SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(params);
  return return_value;
}
// clang-format on
/**
 * @param params SEND_ONLY
 * @param function SEND_ONLY
 * @param algo SEND_ONLY
 */
cusolverStatus_t cusolverDnSetAdvOptions(cusolverDnParams_t params,
                                         cusolverDnFunction_t function,
                                         cusolverAlgMode_t algo);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXpotrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                 void *A, int64_t lda, cudaDataType computeType,
                 void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                 void *bufferOnHost, size_t workspaceInBytesOnHost, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cublasFillMode_t uplo, int64_t n, int64_t nrhs,
                 cudaDataType dataTypeA, const void *A, int64_t lda,
                 cudaDataType dataTypeB, void *B, int64_t ldb, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeTau SEND_ONLY
 * @param tau SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXgeqrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeTau, const void *tau, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeTau SEND_ONLY
 * @param tau SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                 int64_t lda, cudaDataType dataTypeTau, void *tau,
                 cudaDataType computeType, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                 int64_t lda, int64_t *ipiv, cudaDataType computeType,
                 void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                 void *bufferOnHost, size_t workspaceInBytesOnHost, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param trans SEND_ONLY
 * @param n SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param ipiv SEND_ONLY
 * @param dataTypeB SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle,
                                  cusolverDnParams_t params,
                                  cublasOperation_t trans, int64_t n,
                                  int64_t nrhs, cudaDataType dataTypeA,
                                  const void *A, int64_t lda,
                                  const int64_t *ipiv, cudaDataType dataTypeB,
                                  void *B, int64_t ldb, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                 cudaDataType dataTypeA, void *A, int64_t lda,
                 cudaDataType dataTypeW, void *W, cudaDataType computeType,
                 void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                 void *bufferOnHost, size_t workspaceInBytesOnHost, int *info);
/**
 * @guard CUSOLVER_VERSION >= 12200
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param compz SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeDE SEND_ONLY
 * @param D SEND_ONLY
 * @param E SEND_ONLY
 * @param dataTypeZ SEND_ONLY
 * @param Z SEND_ONLY
 * @param ldz SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXstedc_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigComp_t compz, int64_t n, cudaDataType dataTypeDE, const void *D,
    const void *E, cudaDataType dataTypeZ, const void *Z, int64_t ldz,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 12200
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param compz SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeDE SEND_ONLY
 * @param D SEND_ONLY
 * @param E SEND_ONLY
 * @param dataTypeZ SEND_ONLY
 * @param Z SEND_ONLY
 * @param ldz SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXstedc(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cusolverEigComp_t compz, int64_t n, cudaDataType dataTypeDE,
                 void *D, void *E, cudaDataType dataTypeZ, void *Z, int64_t ldz,
                 cudaDataType computeType, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *info);
/**
 * @guard CUSOLVER_VERSION >= 11701
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost, int64_t batchSize);
/**
 * @guard CUSOLVER_VERSION >= 11701
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 * @param batchSize SEND_ONLY
 */
cusolverStatus_t cusolverDnXsyevBatched(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeW,
    void *W, cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *info, int64_t batchSize);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY LENGTH:(vl!=nullptr)*data_type_width(dataTypeW)
 * @param vu SEND_ONLY LENGTH:(vu!=nullptr)*data_type_width(dataTypeW)
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param h_meig SEND_RECV
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo,
    int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, void *vl,
    void *vu, int64_t il, int64_t iu, int64_t *h_meig, cudaDataType dataTypeW,
    const void *W, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param range SEND_ONLY
 * @param uplo SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param vl SEND_ONLY LENGTH:(vl!=nullptr)*data_type_width(dataTypeW)
 * @param vu SEND_ONLY LENGTH:(vu!=nullptr)*data_type_width(dataTypeW)
 * @param il SEND_ONLY
 * @param iu SEND_ONLY
 * @param meig64 SEND_RECV
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXsyevdx(cusolverDnHandle_t handle, cusolverDnParams_t params,
                  cusolverEigMode_t jobz, cusolverEigRange_t range,
                  cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                  void *A, int64_t lda, void *vl, void *vu, int64_t il,
                  int64_t iu, int64_t *meig64, cudaDataType dataTypeW, void *W,
                  cudaDataType computeType, void *bufferOnDevice,
                  size_t workspaceInBytesOnDevice, void *bufferOnHost,
                  size_t workspaceInBytesOnHost, int *info);
/**
 * @guard CUSOLVER_VERSION >= 11701
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobvl SEND_ONLY
 * @param jobvr SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param dataTypeVL SEND_ONLY
 * @param VL SEND_ONLY
 * @param ldvl SEND_ONLY
 * @param dataTypeVR SEND_ONLY
 * @param VR SEND_ONLY
 * @param ldvr SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXgeev_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW,
    const void *W, cudaDataType dataTypeVL, const void *VL, int64_t ldvl,
    cudaDataType dataTypeVR, const void *VR, int64_t ldvr,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 11701
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobvl SEND_ONLY
 * @param jobvr SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeW SEND_ONLY
 * @param W SEND_ONLY
 * @param dataTypeVL SEND_ONLY
 * @param VL SEND_ONLY
 * @param ldvl SEND_ONLY
 * @param dataTypeVR SEND_ONLY
 * @param VR SEND_ONLY
 * @param ldvr SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXgeev(cusolverDnHandle_t handle, cusolverDnParams_t params,
                cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n,
                cudaDataType dataTypeA, void *A, int64_t lda,
                cudaDataType dataTypeW, void *W, cudaDataType dataTypeVL,
                void *VL, int64_t ldvl, cudaDataType dataTypeVR, void *VR,
                int64_t ldvr, cudaDataType computeType, void *bufferOnDevice,
                size_t workspaceInBytesOnDevice, void *bufferOnHost,
                size_t workspaceInBytesOnHost, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobvt SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeS SEND_ONLY
 * @param S SEND_ONLY
 * @param dataTypeU SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param dataTypeVT SEND_ONLY
 * @param VT SEND_ONLY
 * @param ldvt SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXgesvd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA,
    const void *A, int64_t lda, cudaDataType dataTypeS, const void *S,
    cudaDataType dataTypeU, const void *U, int64_t ldu, cudaDataType dataTypeVT,
    const void *VT, int64_t ldvt, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobvt SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeS SEND_ONLY
 * @param S SEND_ONLY
 * @param dataTypeU SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param dataTypeVT SEND_ONLY
 * @param VT SEND_ONLY
 * @param ldvt SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXgesvd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 signed char jobu, signed char jobvt, int64_t m, int64_t n,
                 cudaDataType dataTypeA, void *A, int64_t lda,
                 cudaDataType dataTypeS, void *S, cudaDataType dataTypeU,
                 void *U, int64_t ldu, cudaDataType dataTypeVT, void *VT,
                 int64_t ldvt, cudaDataType computeType, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *info);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeS SEND_ONLY
 * @param S SEND_ONLY
 * @param dataTypeU SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param dataTypeV SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXgesvdp_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeS,
    const void *S, cudaDataType dataTypeU, const void *U, int64_t ldu,
    cudaDataType dataTypeV, const void *V, int64_t ldv,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobz SEND_ONLY
 * @param econ SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeS SEND_ONLY
 * @param S SEND_ONLY
 * @param dataTypeU SEND_ONLY
 * @param U SEND_ONLY
 * @param ldu SEND_ONLY
 * @param dataTypeV SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param d_info SEND_ONLY
 * @param h_err_sigma SEND_RECV
 */
cusolverStatus_t cusolverDnXgesvdp(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeS,
    void *S, cudaDataType dataTypeU, void *U, int64_t ldu,
    cudaDataType dataTypeV, void *V, int64_t ldv, cudaDataType computeType,
    void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *d_info, double *h_err_sigma);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobv SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param p SEND_ONLY
 * @param niters SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeSrand SEND_ONLY
 * @param Srand SEND_ONLY
 * @param dataTypeUrand SEND_ONLY
 * @param Urand SEND_ONLY
 * @param ldUrand SEND_ONLY
 * @param dataTypeVrand SEND_ONLY
 * @param Vrand SEND_ONLY
 * @param ldVrand SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXgesvdr_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu,
    signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p,
    int64_t niters, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeSrand, const void *Srand, cudaDataType dataTypeUrand,
    const void *Urand, int64_t ldUrand, cudaDataType dataTypeVrand,
    const void *Vrand, int64_t ldVrand, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);
/**
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param jobu SEND_ONLY
 * @param jobv SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param p SEND_ONLY
 * @param niters SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeSrand SEND_ONLY
 * @param Srand SEND_ONLY
 * @param dataTypeUrand SEND_ONLY
 * @param Urand SEND_ONLY
 * @param ldUrand SEND_ONLY
 * @param dataTypeVrand SEND_ONLY
 * @param Vrand SEND_ONLY
 * @param ldVrand SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param d_info SEND_ONLY
 */
cusolverStatus_t
cusolverDnXgesvdr(cusolverDnHandle_t handle, cusolverDnParams_t params,
                  signed char jobu, signed char jobv, int64_t m, int64_t n,
                  int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA,
                  void *A, int64_t lda, cudaDataType dataTypeSrand, void *Srand,
                  cudaDataType dataTypeUrand, void *Urand, int64_t ldUrand,
                  cudaDataType dataTypeVrand, void *Vrand, int64_t ldVrand,
                  cudaDataType computeType, void *bufferOnDevice,
                  size_t workspaceInBytesOnDevice, void *bufferOnHost,
                  size_t workspaceInBytesOnHost, int *d_info);
/**
 * @guard CUSOLVER_VERSION >= 11600
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param direct SEND_ONLY
 * @param storev SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param dataTypeV SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param dataTypeTau SEND_ONLY
 * @param tau SEND_ONLY
 * @param dataTypeT SEND_ONLY
 * @param T SEND_ONLY
 * @param ldt SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXlarft_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params,
    cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n,
    int64_t k, cudaDataType dataTypeV, const void *V, int64_t ldv,
    cudaDataType dataTypeTau, const void *tau, cudaDataType dataTypeT, void *T,
    int64_t ldt, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 11600
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param direct SEND_ONLY
 * @param storev SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param dataTypeV SEND_ONLY
 * @param V SEND_ONLY
 * @param ldv SEND_ONLY
 * @param dataTypeTau SEND_ONLY
 * @param tau SEND_ONLY
 * @param dataTypeT SEND_ONLY
 * @param T SEND_ONLY
 * @param ldt SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 */
cusolverStatus_t
cusolverDnXlarft(cusolverDnHandle_t handle, cusolverDnParams_t params,
                 cusolverDirectMode_t direct, cusolverStorevMode_t storev,
                 int64_t n, int64_t k, cudaDataType dataTypeV, const void *V,
                 int64_t ldv, cudaDataType dataTypeTau, const void *tau,
                 cudaDataType dataTypeT, void *T, int64_t ldt,
                 cudaDataType computeType, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 12200
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param uplo SEND_ONLY
 * @param M SEND_ONLY
 * @param N SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeH SEND_ONLY
 * @param H SEND_ONLY
 * @param ldh SEND_ONLY
 * @param computeType SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_RECV
 * @param workspaceInBytesOnHost SEND_RECV
 */
cusolverStatus_t cusolverDnXpolar_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t M, int64_t N, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeH, const void *H, int64_t ldh,
    cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);
/**
 * @guard CUSOLVER_VERSION >= 12200
 * @param handle SEND_ONLY
 * @param params SEND_ONLY
 * @param uplo SEND_ONLY
 * @param M SEND_ONLY
 * @param N SEND_ONLY
 * @param dataTypeA SEND_ONLY
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param dataTypeH SEND_ONLY
 * @param H SEND_ONLY
 * @param ldh SEND_ONLY
 * @param computeType SEND_ONLY
 * @param bufferOnDevice SEND_ONLY
 * @param workspaceInBytesOnDevice SEND_ONLY
 * @param bufferOnHost SEND_RECV LENGTH:workspaceInBytesOnHost
 * @param workspaceInBytesOnHost SEND_ONLY
 * @param d_res_nrm SEND_ONLY
 * @param d_A_nrmF SEND_ONLY
 * @param d_rcond SEND_ONLY
 * @param d_info SEND_ONLY
 */
cusolverStatus_t cusolverDnXpolar(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo,
    int64_t M, int64_t N, cudaDataType dataTypeA, void *A, int64_t lda,
    cudaDataType dataTypeH, void *H, int64_t ldh, cudaDataType computeType,
    void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, double *d_res_nrm, double *d_A_nrmF,
    double *d_rcond, int *d_info);
/**
 * @param handle RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t* handle) {
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
cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param streamId SEND_ONLY
 */
cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle,
                                     cudaStream_t streamId);
/**
 * @param handle SEND_ONLY
 * @param streamId RECV_ONLY
 */
cusolverStatus_t cusolverSpGetStream(cusolverSpHandle_t handle,
                                     cudaStream_t *streamId);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:m
 * @param csrEndPtrA SEND_ONLY LENGTH:m
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param issym SEND_RECV
 */
cusolverStatus_t cusolverSpXcsrissymHost(cusolverSpHandle_t handle, int m,
                                         int nnzA,
                                         const cusparseMatDescr_t descrA,
                                         const int *csrRowPtrA,
                                         const int *csrEndPtrA,
                                         const int *csrColIndA, int *issym);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param b SEND_ONLY LENGTH:n
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:n
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpScsrlsvluHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const float *b, float tol, int reorder, float *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param b SEND_ONLY LENGTH:n
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:n
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrlsvluHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const double *b, double tol, int reorder, double *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param b SEND_ONLY LENGTH:n
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:n
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrlsvluHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param b SEND_ONLY LENGTH:n
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:n
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrlsvluHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, const cuDoubleComplex *b, double tol, int reorder,
    cuDoubleComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpScsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                     const cusparseMatDescr_t descrA,
                                     const float *csrVal, const int *csrRowPtr,
                                     const int *csrColInd, const float *b,
                                     float tol, int reorder, float *x,
                                     int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                     const cusparseMatDescr_t descrA,
                                     const double *csrVal, const int *csrRowPtr,
                                     const int *csrColInd, const double *b,
                                     double tol, int reorder, double *x,
                                     int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                     const cusparseMatDescr_t descrA,
                                     const cuComplex *csrVal,
                                     const int *csrRowPtr, const int *csrColInd,
                                     const cuComplex *b, float tol, int reorder,
                                     cuComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                     const cusparseMatDescr_t descrA,
                                     const cuDoubleComplex *csrVal,
                                     const int *csrRowPtr, const int *csrColInd,
                                     const cuDoubleComplex *b, double tol,
                                     int reorder, cuDoubleComplex *x,
                                     int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpScsrlsvqrHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const float *b, float tol, int reorder, float *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrlsvqrHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const double *b, double tol, int reorder, double *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrlsvqrHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrlsvqrHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, const cuDoubleComplex *b, double tol, int reorder,
    cuDoubleComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY LENGTH:nnz
 * @param csrRowPtr SEND_ONLY LENGTH:m+1
 * @param csrColInd SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpScsrlsvcholHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrVal, const int *csrRowPtr, const int *csrColInd,
    const float *b, float tol, int reorder, float *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY LENGTH:nnz
 * @param csrRowPtr SEND_ONLY LENGTH:m+1
 * @param csrColInd SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrlsvcholHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrVal, const int *csrRowPtr, const int *csrColInd,
    const double *b, double tol, int reorder, double *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY LENGTH:nnz
 * @param csrRowPtr SEND_ONLY LENGTH:m+1
 * @param csrColInd SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrlsvcholHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrVal, const int *csrRowPtr, const int *csrColInd,
    const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY LENGTH:nnz
 * @param csrRowPtr SEND_ONLY LENGTH:m+1
 * @param csrColInd SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_RECV LENGTH:m
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrlsvcholHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal, const int *csrRowPtr, const int *csrColInd,
    const cuDoubleComplex *b, double tol, int reorder, cuDoubleComplex *x,
    int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpScsrlsvchol(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrVal, const int *csrRowPtr, const int *csrColInd,
    const float *b, float tol, int reorder, float *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrlsvchol(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrVal, const int *csrRowPtr, const int *csrColInd,
    const double *b, double tol, int reorder, double *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrlsvchol(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrVal, const int *csrRowPtr, const int *csrColInd,
    const cuComplex *b, float tol, int reorder, cuComplex *x, int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param b SEND_ONLY
 * @param tol SEND_ONLY
 * @param reorder SEND_ONLY
 * @param x SEND_ONLY
 * @param singularity SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrlsvchol(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal, const int *csrRowPtr, const int *csrColInd,
    const cuDoubleComplex *b, double tol, int reorder, cuDoubleComplex *x,
    int *singularity);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param rankA SEND_RECV
 * @param x SEND_RECV LENGTH:n
 * @param p SEND_RECV LENGTH:n
 * @param min_norm SEND_RECV
 */
cusolverStatus_t
cusolverSpScsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                         const cusparseMatDescr_t descrA, const float *csrValA,
                         const int *csrRowPtrA, const int *csrColIndA,
                         const float *b, float tol, int *rankA, float *x,
                         int *p, float *min_norm);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param rankA SEND_RECV
 * @param x SEND_RECV LENGTH:n
 * @param p SEND_RECV LENGTH:n
 * @param min_norm SEND_RECV
 */
cusolverStatus_t
cusolverSpDcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                         const cusparseMatDescr_t descrA, const double *csrValA,
                         const int *csrRowPtrA, const int *csrColIndA,
                         const double *b, double tol, int *rankA, double *x,
                         int *p, double *min_norm);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param rankA SEND_RECV
 * @param x SEND_RECV LENGTH:n
 * @param p SEND_RECV LENGTH:n
 * @param min_norm SEND_RECV
 */
cusolverStatus_t
cusolverSpCcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                         const cusparseMatDescr_t descrA,
                         const cuComplex *csrValA, const int *csrRowPtrA,
                         const int *csrColIndA, const cuComplex *b, float tol,
                         int *rankA, cuComplex *x, int *p, float *min_norm);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param b SEND_ONLY LENGTH:m
 * @param tol SEND_ONLY
 * @param rankA SEND_RECV
 * @param x SEND_RECV LENGTH:n
 * @param p SEND_RECV LENGTH:n
 * @param min_norm SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrlsqvqrHost(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *b,
    double tol, int *rankA, cuDoubleComplex *x, int *p, double *min_norm);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY LENGTH:m
 * @param maxite SEND_ONLY
 * @param tol SEND_ONLY
 * @param mu SEND_RECV
 * @param x SEND_RECV LENGTH:m
 */
cusolverStatus_t cusolverSpScsreigvsiHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    float mu0, const float *x0, int maxite, float tol, float *mu, float *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY LENGTH:m
 * @param maxite SEND_ONLY
 * @param tol SEND_ONLY
 * @param mu SEND_RECV
 * @param x SEND_RECV LENGTH:m
 */
cusolverStatus_t
cusolverSpDcsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz,
                         const cusparseMatDescr_t descrA, const double *csrValA,
                         const int *csrRowPtrA, const int *csrColIndA,
                         double mu0, const double *x0, int maxite, double tol,
                         double *mu, double *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY LENGTH:m
 * @param maxite SEND_ONLY
 * @param tol SEND_ONLY
 * @param mu SEND_RECV
 * @param x SEND_RECV LENGTH:m
 */
cusolverStatus_t cusolverSpCcsreigvsiHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    cuComplex mu0, const cuComplex *x0, int maxite, float tol, cuComplex *mu,
    cuComplex *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY LENGTH:m
 * @param maxite SEND_ONLY
 * @param tol SEND_ONLY
 * @param mu SEND_RECV
 * @param x SEND_RECV LENGTH:m
 */
cusolverStatus_t cusolverSpZcsreigvsiHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, cuDoubleComplex mu0, const cuDoubleComplex *x0,
    int maxite, double tol, cuDoubleComplex *mu, cuDoubleComplex *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY
 * @param maxite SEND_ONLY
 * @param eps SEND_ONLY
 * @param mu SEND_ONLY
 * @param x SEND_ONLY
 */
cusolverStatus_t cusolverSpScsreigvsi(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    float mu0, const float *x0, int maxite, float eps, float *mu, float *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY
 * @param maxite SEND_ONLY
 * @param eps SEND_ONLY
 * @param mu SEND_ONLY
 * @param x SEND_ONLY
 */
cusolverStatus_t cusolverSpDcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const double *csrValA,
                                      const int *csrRowPtrA,
                                      const int *csrColIndA, double mu0,
                                      const double *x0, int maxite, double eps,
                                      double *mu, double *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY
 * @param maxite SEND_ONLY
 * @param eps SEND_ONLY
 * @param mu SEND_ONLY
 * @param x SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const cuComplex *csrValA,
                                      const int *csrRowPtrA,
                                      const int *csrColIndA, cuComplex mu0,
                                      const cuComplex *x0, int maxite,
                                      float eps, cuComplex *mu, cuComplex *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu0 SEND_ONLY
 * @param x0 SEND_ONLY
 * @param maxite SEND_ONLY
 * @param eps SEND_ONLY
 * @param mu SEND_ONLY
 * @param x SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsreigvsi(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, cuDoubleComplex mu0, const cuDoubleComplex *x0,
    int maxite, double eps, cuDoubleComplex *mu, cuDoubleComplex *x);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param left_bottom_corner SEND_ONLY
 * @param right_upper_corner SEND_ONLY
 * @param num_eigs SEND_RECV
 */
cusolverStatus_t cusolverSpScsreigsHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    cuComplex left_bottom_corner, cuComplex right_upper_corner, int *num_eigs);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param left_bottom_corner SEND_ONLY
 * @param right_upper_corner SEND_ONLY
 * @param num_eigs SEND_RECV
 */
cusolverStatus_t
cusolverSpDcsreigsHost(cusolverSpHandle_t handle, int m, int nnz,
                       const cusparseMatDescr_t descrA, const double *csrValA,
                       const int *csrRowPtrA, const int *csrColIndA,
                       cuDoubleComplex left_bottom_corner,
                       cuDoubleComplex right_upper_corner, int *num_eigs);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param left_bottom_corner SEND_ONLY
 * @param right_upper_corner SEND_ONLY
 * @param num_eigs SEND_RECV
 */
cusolverStatus_t cusolverSpCcsreigsHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    cuComplex left_bottom_corner, cuComplex right_upper_corner, int *num_eigs);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param left_bottom_corner SEND_ONLY
 * @param right_upper_corner SEND_ONLY
 * @param num_eigs SEND_RECV
 */
cusolverStatus_t cusolverSpZcsreigsHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, cuDoubleComplex left_bottom_corner,
    cuDoubleComplex right_upper_corner, int *num_eigs);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param p SEND_RECV LENGTH:n
 */
cusolverStatus_t cusolverSpXcsrsymrcmHost(cusolverSpHandle_t handle, int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA, int *p);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param p SEND_RECV LENGTH:n
 */
cusolverStatus_t cusolverSpXcsrsymmdqHost(cusolverSpHandle_t handle, int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA, int *p);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param p SEND_RECV LENGTH:n
 */
cusolverStatus_t cusolverSpXcsrsymamdHost(cusolverSpHandle_t handle, int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA, int *p);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param options SEND_ONLY LENGTH:(options!=nullptr)*40
 * @param p SEND_RECV LENGTH:n
 */
cusolverStatus_t cusolverSpXcsrmetisndHost(cusolverSpHandle_t handle, int n,
                                           int nnzA,
                                           const cusparseMatDescr_t descrA,
                                           const int *csrRowPtrA,
                                           const int *csrColIndA,
                                           const int64_t *options, int *p);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param P SEND_RECV LENGTH:n
 * @param numnz SEND_RECV
 */
cusolverStatus_t cusolverSpScsrzfdHost(cusolverSpHandle_t handle, int n,
                                       int nnz, const cusparseMatDescr_t descrA,
                                       const float *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, int *P,
                                       int *numnz);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param P SEND_RECV LENGTH:n
 * @param numnz SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrzfdHost(cusolverSpHandle_t handle, int n,
                                       int nnz, const cusparseMatDescr_t descrA,
                                       const double *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, int *P,
                                       int *numnz);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param P SEND_RECV LENGTH:n
 * @param numnz SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrzfdHost(cusolverSpHandle_t handle, int n,
                                       int nnz, const cusparseMatDescr_t descrA,
                                       const cuComplex *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, int *P,
                                       int *numnz);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnz
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnz
 * @param P SEND_RECV LENGTH:n
 * @param numnz SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrzfdHost(cusolverSpHandle_t handle, int n,
                                       int nnz, const cusparseMatDescr_t descrA,
                                       const cuDoubleComplex *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, int *P,
                                       int *numnz);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param p SEND_ONLY LENGTH:m
 * @param q SEND_ONLY LENGTH:n
 * @param bufferSizeInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpXcsrperm_bufferSizeHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, const int* p, const int* q, size_t* bufferSizeInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_csrperm_buffer(handle, *bufferSizeInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_RECV LENGTH:m+1
 * @param csrColIndA SEND_RECV LENGTH:nnzA
 * @param p SEND_ONLY LENGTH:m
 * @param q SEND_ONLY LENGTH:n
 * @param map SEND_RECV LENGTH:nnzA
 * @param pBuffer SEND_ONLY LENGTH:csrperm_buffer_bytes(handle)
 */
cusolverStatus_t cusolverSpXcsrpermHost(cusolverSpHandle_t handle, int m, int n,
                                        int nnzA,
                                        const cusparseMatDescr_t descrA,
                                        int *csrRowPtrA, int *csrColIndA,
                                        const int *p, const int *q, int *map,
                                        void *pBuffer);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpCreateCsrqrInfo(csrqrInfo_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpDestroyCsrqrInfo(csrqrInfo_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpXcsrqrAnalysisBatched(cusolverSpHandle_t handle, int m, int n,
                                int nnzA, const cusparseMatDescr_t descrA,
                                const int *csrRowPtrA, const int *csrColIndA,
                                csrqrInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpScsrqrBufferInfoBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const float *csrVal, const int *csrRowPtr,
    const int *csrColInd, int batchSize, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const double *csrVal, const int *csrRowPtr,
    const int *csrColInd, int batchSize, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const cuComplex *csrVal,
    const int *csrRowPtr, const int *csrColInd, int batchSize, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrVal SEND_ONLY
 * @param csrRowPtr SEND_ONLY
 * @param csrColInd SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrVal,
    const int *csrRowPtr, const int *csrColInd, int batchSize, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t
cusolverSpScsrqrsvBatched(cusolverSpHandle_t handle, int m, int n, int nnz,
                          const cusparseMatDescr_t descrA, const float *csrValA,
                          const int *csrRowPtrA, const int *csrColIndA,
                          const float *b, float *x, int batchSize,
                          csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t
cusolverSpDcsrqrsvBatched(cusolverSpHandle_t handle, int m, int n, int nnz,
                          const cusparseMatDescr_t descrA,
                          const double *csrValA, const int *csrRowPtrA,
                          const int *csrColIndA, const double *b, double *x,
                          int batchSize, csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrqrsvBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const cuComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const cuComplex *b,
    cuComplex *x, int batchSize, csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnz SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param batchSize SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrqrsvBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *b,
    cuDoubleComplex *x, int batchSize, csrqrInfo_t info, void *pBuffer);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpCreateCsrluInfoHost(csrluInfoHost_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpDestroyCsrluInfoHost(csrluInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpXcsrluAnalysisHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, csrluInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_dims(info, n);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpScsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrluInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpDcsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrluInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpCcsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrluInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpZcsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrluInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pivot_threshold SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrluFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrluInfoHost_t info, float pivot_threshold, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pivot_threshold SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrluFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrluInfoHost_t info, double pivot_threshold, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pivot_threshold SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrluFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrluInfoHost_t info, float pivot_threshold, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pivot_threshold SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrluFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, csrluInfoHost_t info, double pivot_threshold,
    void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpScsrluZeroPivotHost(cusolverSpHandle_t handle,
                                               csrluInfoHost_t info, float tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrluZeroPivotHost(cusolverSpHandle_t handle,
                                               csrluInfoHost_t info, double tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrluZeroPivotHost(cusolverSpHandle_t handle,
                                               csrluInfoHost_t info, float tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrluZeroPivotHost(cusolverSpHandle_t handle,
                                               csrluInfoHost_t info, double tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrluSolveHost(cusolverSpHandle_t handle, int n,
                                           const float *b, float *x,
                                           csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrluSolveHost(cusolverSpHandle_t handle, int n,
                                           const double *b, double *x,
                                           csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrluSolveHost(cusolverSpHandle_t handle, int n,
                                           const cuComplex *b, cuComplex *x,
                                           csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrluSolveHost(cusolverSpHandle_t handle, int n,
                                           const cuDoubleComplex *b,
                                           cuDoubleComplex *x,
                                           csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param nnzLRef SEND_RECV
 * @param nnzURef SEND_RECV
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpXcsrluNnzHost(cusolverSpHandle_t handle, int* nnzLRef, int* nnzURef, csrluInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_nnz(info, *nnzLRef, *nnzURef);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param P SEND_RECV LENGTH:sp_host_n(info)
 * @param Q SEND_RECV LENGTH:sp_host_n(info)
 * @param descrL SEND_ONLY
 * @param csrValL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param csrRowPtrL SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param descrU SEND_ONLY
 * @param csrValU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param csrRowPtrU SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL,
    float *csrValL, int *csrRowPtrL, int *csrColIndL,
    const cusparseMatDescr_t descrU, float *csrValU, int *csrRowPtrU,
    int *csrColIndU, csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param P SEND_RECV LENGTH:sp_host_n(info)
 * @param Q SEND_RECV LENGTH:sp_host_n(info)
 * @param descrL SEND_ONLY
 * @param csrValL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param csrRowPtrL SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param descrU SEND_ONLY
 * @param csrValU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param csrRowPtrU SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL,
    double *csrValL, int *csrRowPtrL, int *csrColIndL,
    const cusparseMatDescr_t descrU, double *csrValU, int *csrRowPtrU,
    int *csrColIndU, csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param P SEND_RECV LENGTH:sp_host_n(info)
 * @param Q SEND_RECV LENGTH:sp_host_n(info)
 * @param descrL SEND_ONLY
 * @param csrValL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param csrRowPtrL SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param descrU SEND_ONLY
 * @param csrValU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param csrRowPtrU SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL,
    cuComplex *csrValL, int *csrRowPtrL, int *csrColIndL,
    const cusparseMatDescr_t descrU, cuComplex *csrValU, int *csrRowPtrU,
    int *csrColIndU, csrluInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param P SEND_RECV LENGTH:sp_host_n(info)
 * @param Q SEND_RECV LENGTH:sp_host_n(info)
 * @param descrL SEND_ONLY
 * @param csrValL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param csrRowPtrL SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndL SEND_RECV LENGTH:sp_host_nnz_l(info)
 * @param descrU SEND_ONLY
 * @param csrValU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param csrRowPtrU SEND_RECV LENGTH:sp_host_n(info)+1
 * @param csrColIndU SEND_RECV LENGTH:sp_host_nnz_u(info)
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL,
    cuDoubleComplex *csrValL, int *csrRowPtrL, int *csrColIndL,
    const cusparseMatDescr_t descrU, cuDoubleComplex *csrValU, int *csrRowPtrU,
    int *csrColIndU, csrluInfoHost_t info, void *pBuffer);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpCreateCsrqrInfoHost(csrqrInfoHost_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpDestroyCsrqrInfoHost(csrqrInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpXcsrqrAnalysisHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, csrqrInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_dims(info, n);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpScsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrqrInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpDcsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrqrInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpCcsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrqrInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpZcsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrqrInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpScsrqrSetupHost(cusolverSpHandle_t handle, int m, int n, int nnzA,
                          const cusparseMatDescr_t descrA, const float *csrValA,
                          const int *csrRowPtrA, const int *csrColIndA,
                          float mu, csrqrInfoHost_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverSpDcsrqrSetupHost(cusolverSpHandle_t handle, int m,
                                           int n, int nnzA,
                                           const cusparseMatDescr_t descrA,
                                           const double *csrValA,
                                           const int *csrRowPtrA,
                                           const int *csrColIndA, double mu,
                                           csrqrInfoHost_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrqrSetupHost(cusolverSpHandle_t handle, int m,
                                           int n, int nnzA,
                                           const cusparseMatDescr_t descrA,
                                           const cuComplex *csrValA,
                                           const int *csrRowPtrA,
                                           const int *csrColIndA, cuComplex mu,
                                           csrqrInfoHost_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:m+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpZcsrqrSetupHost(cusolverSpHandle_t handle, int m, int n, int nnzA,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex *csrValA, const int *csrRowPtrA,
                          const int *csrColIndA, cuDoubleComplex mu,
                          csrqrInfoHost_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrqrFactorHost(cusolverSpHandle_t handle, int m,
                                            int n, int nnzA, float *b, float *x,
                                            csrqrInfoHost_t info,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrqrFactorHost(cusolverSpHandle_t handle, int m,
                                            int n, int nnzA, double *b,
                                            double *x, csrqrInfoHost_t info,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrqrFactorHost(cusolverSpHandle_t handle, int m,
                                            int n, int nnzA, cuComplex *b,
                                            cuComplex *x, csrqrInfoHost_t info,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrqrFactorHost(cusolverSpHandle_t handle, int m,
                                            int n, int nnzA, cuDoubleComplex *b,
                                            cuDoubleComplex *x,
                                            csrqrInfoHost_t info,
                                            void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpScsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                               csrqrInfoHost_t info, float tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                               csrqrInfoHost_t info, double tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                               csrqrInfoHost_t info, float tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                               csrqrInfoHost_t info, double tol,
                                               int *position);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrqrSolveHost(cusolverSpHandle_t handle, int m,
                                           int n, float *b, float *x,
                                           csrqrInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrqrSolveHost(cusolverSpHandle_t handle, int m,
                                           int n, double *b, double *x,
                                           csrqrInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrqrSolveHost(cusolverSpHandle_t handle, int m,
                                           int n, cuComplex *b, cuComplex *x,
                                           csrqrInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_RECV LENGTH:(b!=nullptr)*m
 * @param x SEND_RECV LENGTH:(x!=nullptr)*n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrqrSolveHost(cusolverSpHandle_t handle, int m,
                                           int n, cuDoubleComplex *b,
                                           cuDoubleComplex *x,
                                           csrqrInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpXcsrqrAnalysis(cusolverSpHandle_t handle, int m, int n, int nnzA,
                         const cusparseMatDescr_t descrA, const int *csrRowPtrA,
                         const int *csrColIndA, csrqrInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpScsrqrBufferInfo(
    cusolverSpHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const float *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrqrBufferInfo(
    cusolverSpHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const double *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrqrBufferInfo(
    cusolverSpHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const cuComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrqrBufferInfo(
    cusolverSpHandle_t handle, int m, int n, int nnzA,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, csrqrInfo_t info,
    size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpScsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                      const cusparseMatDescr_t descrA, const float *csrValA,
                      const int *csrRowPtrA, const int *csrColIndA, float mu,
                      csrqrInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpDcsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                      const cusparseMatDescr_t descrA, const double *csrValA,
                      const int *csrRowPtrA, const int *csrColIndA, double mu,
                      csrqrInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t
cusolverSpCcsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                      const cusparseMatDescr_t descrA, const cuComplex *csrValA,
                      const int *csrRowPtrA, const int *csrColIndA,
                      cuComplex mu, csrqrInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param mu SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrqrSetup(cusolverSpHandle_t handle, int m, int n,
                                       int nnzA,
                                       const cusparseMatDescr_t descrA,
                                       const cuDoubleComplex *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA,
                                       cuDoubleComplex mu, csrqrInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpScsrqrFactor(cusolverSpHandle_t handle, int m, int n,
                                        int nnzA, float *b, float *x,
                                        csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpDcsrqrFactor(cusolverSpHandle_t handle, int m, int n,
                                        int nnzA, double *b, double *x,
                                        csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrqrFactor(cusolverSpHandle_t handle, int m, int n,
                                        int nnzA, cuComplex *b, cuComplex *x,
                                        csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrqrFactor(cusolverSpHandle_t handle, int m, int n,
                                        int nnzA, cuDoubleComplex *b,
                                        cuDoubleComplex *x, csrqrInfo_t info,
                                        void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpScsrqrZeroPivot(cusolverSpHandle_t handle,
                                           csrqrInfo_t info, float tol,
                                           int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrqrZeroPivot(cusolverSpHandle_t handle,
                                           csrqrInfo_t info, double tol,
                                           int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrqrZeroPivot(cusolverSpHandle_t handle,
                                           csrqrInfo_t info, float tol,
                                           int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrqrZeroPivot(cusolverSpHandle_t handle,
                                           csrqrInfo_t info, double tol,
                                           int *position);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpScsrqrSolve(cusolverSpHandle_t handle, int m, int n,
                                       float *b, float *x, csrqrInfo_t info,
                                       void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpDcsrqrSolve(cusolverSpHandle_t handle, int m, int n,
                                       double *b, double *x, csrqrInfo_t info,
                                       void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrqrSolve(cusolverSpHandle_t handle, int m, int n,
                                       cuComplex *b, cuComplex *x,
                                       csrqrInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrqrSolve(cusolverSpHandle_t handle, int m, int n,
                                       cuDoubleComplex *b, cuDoubleComplex *x,
                                       csrqrInfo_t info, void *pBuffer);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpCreateCsrcholInfoHost(csrcholInfoHost_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpDestroyCsrcholInfoHost(csrcholInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpXcsrcholAnalysisHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, csrcholInfoHost_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_dims(info, n);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpScsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrcholInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpDcsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrcholInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpCcsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrcholInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
// clang-format off
cusolverStatus_t cusolverSpZcsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, csrcholInfoHost_t info, size_t* internalDataInBytes, size_t* workspaceInBytes) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_sp_host_workspace(info, *workspaceInBytes);
  }
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrcholFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrcholFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrcholFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY LENGTH:nnzA
 * @param csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param csrColIndA SEND_ONLY LENGTH:nnzA
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrcholFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, csrcholInfoHost_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpScsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                 csrcholInfoHost_t info,
                                                 float tol, int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                 csrcholInfoHost_t info,
                                                 double tol, int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                 csrcholInfoHost_t info,
                                                 float tol, int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                 csrcholInfoHost_t info,
                                                 double tol, int *position);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpScsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                             const float *b, float *x,
                                             csrcholInfoHost_t info,
                                             void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpDcsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                             const double *b, double *x,
                                             csrcholInfoHost_t info,
                                             void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpCcsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                             const cuComplex *b, cuComplex *x,
                                             csrcholInfoHost_t info,
                                             void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY LENGTH:n
 * @param x SEND_RECV LENGTH:n
 * @param info SEND_ONLY
 * @param pBuffer SEND_RECV LENGTH:sp_host_workspace(info)
 */
cusolverStatus_t cusolverSpZcsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                             const cuDoubleComplex *b,
                                             cuDoubleComplex *x,
                                             csrcholInfoHost_t info,
                                             void *pBuffer);
/**
 * @param info RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpCreateCsrcholInfo(csrcholInfo_t* info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_handle_owner(conn, *info);
  }
  return return_value;
}
// clang-format on
/**
 * @param info SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverSpDestroyCsrcholInfo(csrcholInfo_t info) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(info);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 */
cusolverStatus_t cusolverSpXcsrcholAnalysis(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const int *csrRowPtrA, const int *csrColIndA, csrcholInfo_t info);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpScsrcholBufferInfo(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrcholBufferInfo(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrcholBufferInfo(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param internalDataInBytes SEND_RECV
 * @param workspaceInBytes SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrcholBufferInfo(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, csrcholInfo_t info, size_t *internalDataInBytes,
    size_t *workspaceInBytes);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t
cusolverSpScsrcholFactor(cusolverSpHandle_t handle, int n, int nnzA,
                         const cusparseMatDescr_t descrA, const float *csrValA,
                         const int *csrRowPtrA, const int *csrColIndA,
                         csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t
cusolverSpDcsrcholFactor(cusolverSpHandle_t handle, int n, int nnzA,
                         const cusparseMatDescr_t descrA, const double *csrValA,
                         const int *csrRowPtrA, const int *csrColIndA,
                         csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrcholFactor(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param descrA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrcholFactor(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA, const int *csrRowPtrA,
    const int *csrColIndA, csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpScsrcholZeroPivot(cusolverSpHandle_t handle,
                                             csrcholInfo_t info, float tol,
                                             int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpDcsrcholZeroPivot(cusolverSpHandle_t handle,
                                             csrcholInfo_t info, double tol,
                                             int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpCcsrcholZeroPivot(cusolverSpHandle_t handle,
                                             csrcholInfo_t info, float tol,
                                             int *position);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param tol SEND_ONLY
 * @param position SEND_RECV
 */
cusolverStatus_t cusolverSpZcsrcholZeroPivot(cusolverSpHandle_t handle,
                                             csrcholInfo_t info, double tol,
                                             int *position);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpScsrcholSolve(cusolverSpHandle_t handle, int n,
                                         const float *b, float *x,
                                         csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpDcsrcholSolve(cusolverSpHandle_t handle, int n,
                                         const double *b, double *x,
                                         csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrcholSolve(cusolverSpHandle_t handle, int n,
                                         const cuComplex *b, cuComplex *x,
                                         csrcholInfo_t info, void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param n SEND_ONLY
 * @param b SEND_ONLY
 * @param x SEND_ONLY
 * @param info SEND_ONLY
 * @param pBuffer SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrcholSolve(cusolverSpHandle_t handle, int n,
                                         const cuDoubleComplex *b,
                                         cuDoubleComplex *x, csrcholInfo_t info,
                                         void *pBuffer);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param diag SEND_ONLY
 */
cusolverStatus_t cusolverSpScsrcholDiag(cusolverSpHandle_t handle,
                                        csrcholInfo_t info, float *diag);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param diag SEND_ONLY
 */
cusolverStatus_t cusolverSpDcsrcholDiag(cusolverSpHandle_t handle,
                                        csrcholInfo_t info, double *diag);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param diag SEND_ONLY
 */
cusolverStatus_t cusolverSpCcsrcholDiag(cusolverSpHandle_t handle,
                                        csrcholInfo_t info, float *diag);
/**
 * @param handle SEND_ONLY
 * @param info SEND_ONLY
 * @param diag SEND_ONLY
 */
cusolverStatus_t cusolverSpZcsrcholDiag(cusolverSpHandle_t handle,
                                        csrcholInfo_t info, double *diag);
/**
 * @param handle RECV_ONLY
 */
// clang-format off
cusolverStatus_t cusolverRfCreate(cusolverRfHandle_t* handle) {
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
cusolverStatus_t cusolverRfDestroy(cusolverRfHandle_t handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  forget_handle_owner(handle);
  return return_value;
}
// clang-format on
/**
 * @param handle SEND_ONLY
 * @param format SEND_RECV
 * @param diag SEND_RECV
 */
cusolverStatus_t cusolverRfGetMatrixFormat(cusolverRfHandle_t handle,
                                           cusolverRfMatrixFormat_t *format,
                                           cusolverRfUnitDiagonal_t *diag);
/**
 * @param handle SEND_ONLY
 * @param format SEND_ONLY
 * @param diag SEND_ONLY
 */
cusolverStatus_t cusolverRfSetMatrixFormat(cusolverRfHandle_t handle,
                                           cusolverRfMatrixFormat_t format,
                                           cusolverRfUnitDiagonal_t diag);
/**
 * @param handle SEND_ONLY
 * @param zero SEND_ONLY
 * @param boost SEND_ONLY
 */
cusolverStatus_t cusolverRfSetNumericProperties(cusolverRfHandle_t handle,
                                                double zero, double boost);
/**
 * @param handle SEND_ONLY
 * @param zero SEND_RECV
 * @param boost SEND_RECV
 */
cusolverStatus_t cusolverRfGetNumericProperties(cusolverRfHandle_t handle,
                                                double *zero, double *boost);
/**
 * @param handle SEND_ONLY
 * @param report SEND_RECV
 */
cusolverStatus_t
cusolverRfGetNumericBoostReport(cusolverRfHandle_t handle,
                                cusolverRfNumericBoostReport_t *report);
/**
 * @param handle SEND_ONLY
 * @param factAlg SEND_ONLY
 * @param solveAlg SEND_ONLY
 */
cusolverStatus_t cusolverRfSetAlgs(cusolverRfHandle_t handle,
                                   cusolverRfFactorization_t factAlg,
                                   cusolverRfTriangularSolve_t solveAlg);
/**
 * @param handle SEND_ONLY
 * @param factAlg SEND_RECV
 * @param solveAlg SEND_RECV
 */
cusolverStatus_t cusolverRfGetAlgs(cusolverRfHandle_t handle,
                                   cusolverRfFactorization_t *factAlg,
                                   cusolverRfTriangularSolve_t *solveAlg);
/**
 * @param handle SEND_ONLY
 * @param fastMode SEND_RECV
 */
cusolverStatus_t
cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle,
                                 cusolverRfResetValuesFastMode_t *fastMode);
/**
 * @param handle SEND_ONLY
 * @param fastMode SEND_ONLY
 */
cusolverStatus_t
cusolverRfSetResetValuesFastMode(cusolverRfHandle_t handle,
                                 cusolverRfResetValuesFastMode_t fastMode);
/**
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param h_csrRowPtrA SEND_ONLY LENGTH:n+1
 * @param h_csrColIndA SEND_ONLY LENGTH:nnzA
 * @param h_csrValA SEND_ONLY LENGTH:nnzA
 * @param nnzL SEND_ONLY
 * @param h_csrRowPtrL SEND_ONLY LENGTH:n+1
 * @param h_csrColIndL SEND_ONLY LENGTH:nnzL
 * @param h_csrValL SEND_ONLY LENGTH:nnzL
 * @param nnzU SEND_ONLY
 * @param h_csrRowPtrU SEND_ONLY LENGTH:n+1
 * @param h_csrColIndU SEND_ONLY LENGTH:nnzU
 * @param h_csrValU SEND_ONLY LENGTH:nnzU
 * @param h_P SEND_ONLY LENGTH:n
 * @param h_Q SEND_ONLY LENGTH:n
 * @param handle SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverRfSetupHost(int n, int nnzA, int* h_csrRowPtrA, int* h_csrColIndA, double* h_csrValA, int nnzL, int* h_csrRowPtrL, int* h_csrColIndL, double* h_csrValL, int nnzU, int* h_csrRowPtrU, int* h_csrColIndU, double* h_csrValU, int* h_P, int* h_Q, cusolverRfHandle_t handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_rf_dims(handle, n, 1);
  }
  return return_value;
}
// clang-format on
/**
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param nnzL SEND_ONLY
 * @param csrRowPtrL SEND_ONLY
 * @param csrColIndL SEND_ONLY
 * @param csrValL SEND_ONLY
 * @param nnzU SEND_ONLY
 * @param csrRowPtrU SEND_ONLY
 * @param csrColIndU SEND_ONLY
 * @param csrValU SEND_ONLY
 * @param P SEND_ONLY
 * @param Q SEND_ONLY
 * @param handle SEND_ONLY
 */
// clang-format off
cusolverStatus_t cusolverRfSetupDevice(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int nnzL, int* csrRowPtrL, int* csrColIndL, double* csrValL, int nnzU, int* csrRowPtrU, int* csrColIndU, double* csrValU, int* P, int* Q, cusolverRfHandle_t handle) {
  cusolverStatus_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUSOLVER_STATUS_SUCCESS) {
    note_rf_dims(handle, n, 1);
  }
  return return_value;
}
// clang-format on
/**
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param csrValA SEND_ONLY
 * @param P SEND_ONLY
 * @param Q SEND_ONLY
 * @param handle SEND_ONLY
 */
cusolverStatus_t cusolverRfResetValues(int n, int nnzA, int *csrRowPtrA,
                                       int *csrColIndA, double *csrValA, int *P,
                                       int *Q, cusolverRfHandle_t handle);
/**
 * @param handle SEND_ONLY
 */
cusolverStatus_t cusolverRfAnalyze(cusolverRfHandle_t handle);
/**
 * @param handle SEND_ONLY
 */
cusolverStatus_t cusolverRfRefactor(cusolverRfHandle_t handle);
/**
 * @param handle SEND_ONLY
 * @param nnzM SEND_RECV
 * @param Mp RECV_ONLY
 * @param Mi RECV_ONLY
 * @param Mx RECV_ONLY
 */
cusolverStatus_t cusolverRfAccessBundledFactorsDevice(cusolverRfHandle_t handle,
                                                      int *nnzM, int **Mp,
                                                      int **Mi, double **Mx);
/**
 * @disabled
 */
cusolverStatus_t cusolverRfExtractBundledFactorsHost(cusolverRfHandle_t handle,
                                                     int *h_nnzM, int **h_Mp,
                                                     int **h_Mi, double **h_Mx);
/**
 * @disabled
 */
cusolverStatus_t cusolverRfExtractSplitFactorsHost(
    cusolverRfHandle_t handle, int *h_nnzL, int **h_csrRowPtrL,
    int **h_csrColIndL, double **h_csrValL, int *h_nnzU, int **h_csrRowPtrU,
    int **h_csrColIndU, double **h_csrValU);
/**
 * @param handle SEND_ONLY
 * @param P SEND_ONLY
 * @param Q SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Temp SEND_ONLY
 * @param ldt SEND_ONLY
 * @param XF SEND_ONLY
 * @param ldxf SEND_ONLY
 */
cusolverStatus_t cusolverRfSolve(cusolverRfHandle_t handle, int *P, int *Q,
                                 int nrhs, double *Temp, int ldt, double *XF,
                                 int ldxf);
/**
 * @disabled
 */
cusolverStatus_t cusolverRfBatchSetupHost(
    int batchSize, int n, int nnzA, int *h_csrRowPtrA, int *h_csrColIndA,
    double **h_csrValA_array, int nnzL, int *h_csrRowPtrL, int *h_csrColIndL,
    double *h_csrValL, int nnzU, int *h_csrRowPtrU, int *h_csrColIndU,
    double *h_csrValU, int *h_P, int *h_Q, cusolverRfHandle_t handle);
/**
 * @param batchSize SEND_ONLY
 * @param n SEND_ONLY
 * @param nnzA SEND_ONLY
 * @param csrRowPtrA SEND_ONLY
 * @param csrColIndA SEND_ONLY
 * @param csrValA_array SEND_ONLY
 * @param P SEND_ONLY
 * @param Q SEND_ONLY
 * @param handle SEND_ONLY
 */
cusolverStatus_t cusolverRfBatchResetValues(int batchSize, int n, int nnzA,
                                            int *csrRowPtrA, int *csrColIndA,
                                            double **csrValA_array, int *P,
                                            int *Q, cusolverRfHandle_t handle);
/**
 * @param handle SEND_ONLY
 */
cusolverStatus_t cusolverRfBatchAnalyze(cusolverRfHandle_t handle);
/**
 * @param handle SEND_ONLY
 */
cusolverStatus_t cusolverRfBatchRefactor(cusolverRfHandle_t handle);
/**
 * @param handle SEND_ONLY
 * @param P SEND_ONLY
 * @param Q SEND_ONLY
 * @param nrhs SEND_ONLY
 * @param Temp SEND_ONLY
 * @param ldt SEND_ONLY
 * @param XF_array SEND_ONLY
 * @param ldxf SEND_ONLY
 */
cusolverStatus_t cusolverRfBatchSolve(cusolverRfHandle_t handle, int *P, int *Q,
                                      int nrhs, double *Temp, int ldt,
                                      double **XF_array, int ldxf);
/**
 * @param handle SEND_ONLY
 * @param position SEND_RECV LENGTH:rf_batch_size(handle)
 */
cusolverStatus_t cusolverRfBatchZeroPivot(cusolverRfHandle_t handle,
                                          int *position);
