#include <cufftXt.h>

// cuFFT API. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libcufft by
// name at runtime. A plan is an integer handle the server's library hands out;
// it routes to the connection that created it, and a call without one goes to
// the runtime's current device.
//
// This file is the list of what the shim supports. Anything cufftXt.h declares
// that is absent below, or declared below without saying what to do with its
// parameters, gets a stub returning CUFFT_NOT_SUPPORTED, so a consumer linking
// the whole library still loads.
//
// The layout arrays of the *Many calls are `LENGTH:rank`; the embeds may be
// null, which `NULLABLE` carries. Transform data and work areas are device
// pointers and travel as addresses. The multi-GPU descriptor calls
// (cufftXtMalloc, cufftXtMemcpy, cufftXtFree) and the per-GPU pointer arrays
// (cufftXtSetWorkArea, the callback setters) are hand-written: the client
// mirrors each descriptor so a caller can read it, and sizes the arrays by
// the plan's GPU count, which no annotation can describe.
//
// The plan properties arrived with cuFFT 11.2 and the JIT (LTO) callback with
// 11.3, where the header binds it to a versioned symbol until 12.0, so each
// carries the matching @guard.

/**
 * @recordowner HANDLE handle
 * @param handle RECV_ONLY
 */
cufftResult cufftCreate(cufftHandle *handle);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 */
cufftResult cufftDestroy(cufftHandle plan);
/**
 * @param nx SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftEstimate1d(int nx, cufftType type, int batch,
                            size_t *workSize);
/**
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param type SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize);
/**
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param nz SEND_ONLY
 * @param type SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type,
                            size_t *workSize);
/**
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftEstimateMany(int rank, int *n, int *inembed, int istride,
                              int idist, int *onembed, int ostride, int odist,
                              cufftType type, int batch, size_t *workSize);
/**
 * @param plan SEND_ONLY
 * @param idata SEND_ONLY
 * @param odata SEND_ONLY
 * @param direction SEND_ONLY
 */
cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata,
                         cufftComplex *odata, int direction);
/**
 * @param plan SEND_ONLY
 * @param idata SEND_ONLY
 * @param odata SEND_ONLY
 */
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata,
                         cufftReal *odata);
/**
 * @param plan SEND_ONLY
 * @param idata SEND_ONLY
 * @param odata SEND_ONLY
 */
cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
                         cufftDoubleComplex *odata);
/**
 * @param plan SEND_ONLY
 * @param idata SEND_ONLY
 * @param odata SEND_ONLY
 */
cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata,
                         cufftComplex *odata);
/**
 * @param plan SEND_ONLY
 * @param idata SEND_ONLY
 * @param odata SEND_ONLY
 */
cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
                         cufftDoubleReal *odata);
/**
 * @param plan SEND_ONLY
 * @param idata SEND_ONLY
 * @param odata SEND_ONLY
 * @param direction SEND_ONLY
 */
cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
                         cufftDoubleComplex *odata, int direction);
#if CUFFT_VERSION >= 11200
/**
 * @guard CUFFT_VERSION >= 11200
 * @param plan SEND_ONLY
 * @param property SEND_ONLY
 * @param returnPtrValue RECV_ONLY
 */
cufftResult cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property,
                                      long long int *returnPtrValue);
#endif
/**
 * @param type SEND_ONLY
 * @param value RECV_ONLY
 */
cufftResult cufftGetProperty(libraryPropertyType type, int *value);
/**
 * @param handle SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftGetSize(cufftHandle handle, size_t *workSize);
/**
 * @param handle SEND_ONLY
 * @param nx SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type,
                           int batch, size_t *workSize);
/**
 * @param handle SEND_ONLY
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param type SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type,
                           size_t *workSize);
/**
 * @param handle SEND_ONLY
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param nz SEND_ONLY
 * @param type SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz,
                           cufftType type, size_t *workSize);
/**
 * @param handle SEND_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workArea RECV_ONLY
 */
cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed,
                             int istride, int idist, int *onembed, int ostride,
                             int odist, cufftType type, int batch,
                             size_t *workArea);
/**
 * @param plan SEND_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n,
                               long long int *inembed, long long int istride,
                               long long int idist, long long int *onembed,
                               long long int ostride, long long int odist,
                               cufftType type, long long int batch,
                               size_t *workSize);
/**
 * @param version RECV_ONLY
 */
cufftResult cufftGetVersion(int *version);
/**
 * @param plan SEND_ONLY
 * @param nx SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                            size_t *workSize);
/**
 * @param plan SEND_ONLY
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param type SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                            size_t *workSize);
/**
 * @param plan SEND_ONLY
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param nz SEND_ONLY
 * @param type SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz,
                            cufftType type, size_t *workSize);
/**
 * @param plan SEND_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
                              int istride, int idist, int *onembed, int ostride,
                              int odist, cufftType type, int batch,
                              size_t *workSize);
/**
 * @param plan SEND_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 */
cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n,
                                long long int *inembed, long long int istride,
                                long long int idist, long long int *onembed,
                                long long int ostride, long long int odist,
                                cufftType type, long long int batch,
                                size_t *workSize);
/**
 * @recordowner HANDLE plan
 * @param plan RECV_ONLY
 * @param nx SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 */
cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
/**
 * @recordowner HANDLE plan
 * @param plan RECV_ONLY
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param type SEND_ONLY
 */
cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
/**
 * @recordowner HANDLE plan
 * @param plan RECV_ONLY
 * @param nx SEND_ONLY
 * @param ny SEND_ONLY
 * @param nz SEND_ONLY
 * @param type SEND_ONLY
 */
cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz,
                        cufftType type);
/**
 * @recordowner HANDLE plan
 * @param plan RECV_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param type SEND_ONLY
 * @param batch SEND_ONLY
 */
cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
                          int istride, int idist, int *onembed, int ostride,
                          int odist, cufftType type, int batch);
#if CUFFT_VERSION >= 11200
/**
 * @guard CUFFT_VERSION >= 11200
 * @param plan SEND_ONLY
 * @param property SEND_ONLY
 */
cufftResult cufftResetPlanProperty(cufftHandle plan, cufftProperty property);
#endif
/**
 * @param plan SEND_ONLY
 * @param autoAllocate SEND_ONLY
 */
cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);
#if CUFFT_VERSION >= 11200
/**
 * @guard CUFFT_VERSION >= 11200
 * @param plan SEND_ONLY
 * @param property SEND_ONLY
 * @param inputValueInt SEND_ONLY
 */
cufftResult cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property,
                                      const long long int inputValueInt);
#endif
/**
 * @param plan SEND_ONLY
 * @param stream SEND_ONLY
 */
cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
/**
 * @param plan SEND_ONLY
 * @param workArea SEND_ONLY
 */
cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea);
/**
 * @param plan SEND_ONLY
 * @param cbType SEND_ONLY
 */
cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType);
/**
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 * @param direction SEND_ONLY
 */
cufftResult cufftXtExec(cufftHandle plan, void *input, void *output,
                        int direction);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 * @param direction SEND_ONLY
 */
cufftResult cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc *input,
                                  cudaLibXtDesc *output, int direction);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 * @param direction SEND_ONLY
 */
cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc *input,
                                     cudaLibXtDesc *output, int direction);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 */
cufftResult cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc *input,
                                     cudaLibXtDesc *output);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 */
cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc *input,
                                     cudaLibXtDesc *output);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 */
cufftResult cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc *input,
                                     cudaLibXtDesc *output);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 */
cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc *input,
                                     cudaLibXtDesc *output);
/**
 * @disabled client forwards
 * @param plan SEND_ONLY
 * @param input SEND_ONLY
 * @param output SEND_ONLY
 * @param direction SEND_ONLY
 */
cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc *input,
                                     cudaLibXtDesc *output, int direction);
/**
 * @disabled
 */
cufftResult cufftXtFree(cudaLibXtDesc *descriptor);
/**
 * @param plan SEND_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param inputtype SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param outputtype SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 * @param executiontype SEND_ONLY
 */
cufftResult cufftXtGetSizeMany(cufftHandle plan, int rank, long long int *n,
                               long long int *inembed, long long int istride,
                               long long int idist, cudaDataType inputtype,
                               long long int *onembed, long long int ostride,
                               long long int odist, cudaDataType outputtype,
                               long long int batch, size_t *workSize,
                               cudaDataType executiontype);
/**
 * @param plan SEND_ONLY
 * @param rank SEND_ONLY
 * @param n SEND_ONLY LENGTH:rank
 * @param inembed SEND_ONLY NULLABLE LENGTH:rank
 * @param istride SEND_ONLY
 * @param idist SEND_ONLY
 * @param inputtype SEND_ONLY
 * @param onembed SEND_ONLY NULLABLE LENGTH:rank
 * @param ostride SEND_ONLY
 * @param odist SEND_ONLY
 * @param outputtype SEND_ONLY
 * @param batch SEND_ONLY
 * @param workSize RECV_ONLY
 * @param executiontype SEND_ONLY
 */
cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n,
                                long long int *inembed, long long int istride,
                                long long int idist, cudaDataType inputtype,
                                long long int *onembed, long long int ostride,
                                long long int odist, cudaDataType outputtype,
                                long long int batch, size_t *workSize,
                                cudaDataType executiontype);
/**
 * @disabled
 */
cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor,
                          cufftXtSubFormat format);
/**
 * @disabled
 */
cufftResult cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer,
                          cufftXtCopyType type);
// The only query is CUFFT_QUERY_1D_FACTORS, which fills a cufftXt1dFactors:
// ten long longs and four ints.
/**
 * @param plan SEND_ONLY
 * @param queryStruct RECV_ONLY SIZE:96
 * @param queryType SEND_ONLY
 */
cufftResult cufftXtQueryPlan(cufftHandle plan, void *queryStruct,
                             cufftXtQueryType queryType);
/**
 * @disabled
 */
cufftResult cufftXtSetCallback(cufftHandle plan, void **callback_routine,
                               cufftXtCallbackType cbType, void **caller_info);
/**
 * @param plan SEND_ONLY
 * @param cbType SEND_ONLY
 * @param sharedSize SEND_ONLY
 */
cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan,
                                         cufftXtCallbackType cbType,
                                         size_t sharedSize);
/**
 * @disabled client forwards
 * @param handle SEND_ONLY
 * @param nGPUs SEND_ONLY
 * @param whichGPUs SEND_ONLY LENGTH:nGPUs
 */
cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs, int *whichGPUs);
#if CUFFT_VERSION >= 12000
/**
 * @guard CUFFT_VERSION >= 12000
 * @disabled
 */
cufftResult cufftXtSetJITCallback(cufftHandle plan,
                                  const char *lto_callback_symbol_name,
                                  const void *lto_callback_fatbin,
                                  size_t lto_callback_fatbin_size,
                                  cufftXtCallbackType type, void **caller_info);
#endif
// The 11.3 to 11.4 headers declare the JIT callback under this name and bind
// cufftXtSetJITCallback to it with a static inline, so it is the symbol a
// consumer built against those headers references. Not wrapped in #if: this
// file is read against the newest header, which no longer declares it.
/**
 * @guard CUFFT_VERSION >= 11300 && CUFFT_VERSION < 12000
 * @disabled
 */
cufftResult __cufftXtSetJITCallback_12_7(cufftHandle plan,
                                         const char *lto_callback_symbol_name,
                                         const void *lto_callback_fatbin,
                                         size_t lto_callback_fatbin_size,
                                         cufftXtCallbackType type,
                                         void **caller_info);
/**
 * @disabled
 */
cufftResult cufftXtSetWorkArea(cufftHandle plan, void **workArea);
/**
 * @param plan SEND_ONLY
 * @param policy SEND_ONLY
 * @param workSize SEND_ONLY NULLABLE
 */
cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan,
                                     cufftXtWorkAreaPolicy policy,
                                     size_t *workSize);
