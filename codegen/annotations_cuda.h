#include <cuda.h>

/**
 * @disabled - the driver returns a static string of unknown length, which the
 * generated marshaller cannot receive; manual client caches, manual server
 * forwards length-prefixed bytes
 * @param error SEND_ONLY
 * @param pStr SEND_RECV
 */
CUresult cuGetErrorString(CUresult error, const char **pStr);
/**
 * @disabled - see cuGetErrorString
 * @param error SEND_ONLY
 * @param pStr SEND_RECV
 */
CUresult cuGetErrorName(CUresult error, const char **pStr);
/**
 * @disabled client - manual client initializes every configured route
 * @param Flags SEND_ONLY
 */
CUresult cuInit(unsigned int Flags);
/**
 * @param driverVersion RECV_ONLY
 */
CUresult cuDriverGetVersion(int *driverVersion);
/**
 * @disabled client - manual client maps the virtual device ordinal
 * @param device RECV_ONLY
 * @param ordinal SEND_ONLY
 */
CUresult cuDeviceGet(CUdevice *device, int ordinal);
/**
 * @disabled client - manual client reports the virtual device table size
 * @param count RECV_ONLY
 */
CUresult cuDeviceGetCount(int *count);
/**
 * @disabled client - manual client serves the device snapshot cache
 * @param len SEND_ONLY
 * @param name RECV_ONLY LENGTH:len
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
/**
 * @disabled client - manual client serves the device snapshot cache
 * @param uuid RECV_ONLY SIZE:16
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
/**
 * @disabled client - manual client serves the device snapshot cache
 * @param uuid RECV_ONLY SIZE:16
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
/**
 * @param luid RECV_ONLY SIZE:8
 * @param deviceNodeMask RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                         CUdevice dev);
/**
 * @disabled client - manual client serves the device snapshot cache
 * @param bytes RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
/**
 * @param maxWidthInElements RECV_ONLY
 * @param format SEND_ONLY
 * @param numChannels SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                            CUarray_format format,
                                            unsigned numChannels, CUdevice dev);
/**
 * @disabled client - manual client caches attributes per device
 * @param pi RECV_ONLY
 * @param attrib SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
/**
 * @param dev SEND_ONLY
 * @param pool SEND_ONLY
 */
CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
/**
 * @recordowner MEMORY_POOL pool
 * @param pool RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev);
/**
 * @recordowner MEMORY_POOL pool_out
 * @param pool_out RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev);
/**
 * @param pi RECV_ONLY
 * @param type SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type,
                                        CUdevice dev);
/**
 * @param target SEND_ONLY
 * @param scope SEND_ONLY
 */
CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target,
                                    CUflushGPUDirectRDMAWritesScope scope);
/**
 * @param prop RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
/**
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
/**
 * @disabled - manual client caches primary context state; manual server
 * coordinates retained staging lifecycle
 * @recordowner CONTEXT pctx
 * @param pctx RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
/**
 * @disabled - manual client caches primary context state; manual server
 * coordinates retained staging lifecycle
 * @param dev SEND_ONLY
 */
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev);
/**
 * @disabled client - manual client caches primary context state
 * @param dev SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);
/**
 * @disabled client - manual client caches primary context state
 * @param dev SEND_ONLY
 * @param flags RECV_ONLY
 * @param active RECV_ONLY
 */
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                    int *active);
/**
 * @disabled - manual client caches primary context state; manual server
 * coordinates retained staging lifecycle
 * @param dev SEND_ONLY
 */
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev);
/**
 * @disabled - manual client and server maintain context ownership state
 * @recordowner CONTEXT pctx
 * @param pctx RECV_ONLY
 * @param flags SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
/**
 * @param pctx RECV_ONLY
 * @param numParams SEND_ONLY
 * @param paramsArray RECV_ONLY LENGTH:numParams
 * @param flags SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray,
                        int numParams, unsigned int flags, CUdevice dev);
/**
 * @disabled server - manual server coordinates retained staging lifecycle
 * @param ctx SEND_ONLY
 */
CUresult cuCtxDestroy_v2(CUcontext ctx) {
  CUcontext lupine_current_before_destroy = nullptr;
  if (cuCtxGetCurrent(&lupine_current_before_destroy) == CUDA_SUCCESS &&
      lupine_current_before_destroy == ctx) {
    cuCtxSetCurrent(nullptr);
  }
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_forget_destroyed_context(ctx);
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  return return_value;
}
/**
 * @disabled client - manual client maintains the virtual context stack
 * @param ctx SEND_ONLY
 */
CUresult cuCtxPushCurrent_v2(CUcontext ctx);
/**
 * @disabled client - manual client maintains the virtual context stack
 * @param pctx RECV_ONLY
 */
CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
/**
 * @disabled client - manual client maintains the virtual current context
 * @param ctx SEND_ONLY
 */
CUresult cuCtxSetCurrent(CUcontext ctx);
/**
 * @disabled client - manual client maintains the virtual current context
 * @param pctx RECV_ONLY
 */
CUresult cuCtxGetCurrent(CUcontext *pctx);
/**
 * @disabled client - manual client resolves the device of the virtual context
 * @param device RECV_ONLY
 */
CUresult cuCtxGetDevice(CUdevice *device);
/**
 * @guard CUDA_VERSION >= 13000
 * @disabled client - manual client resolves the device of the supplied context
 * @routingkey CONTEXT ctx
 * @param device RECV_ONLY
 * @param ctx SEND_ONLY
 */
CUresult cuCtxGetDevice_v2(CUdevice *device, CUcontext ctx);
/**
 * @param flags RECV_ONLY
 */
CUresult cuCtxGetFlags(unsigned int *flags);
/**
 * @param ctx SEND_ONLY
 * @param ctxId RECV_ONLY
 */
CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId);
/**
 * @guard CUDA_VERSION >= 12050
 * @param hCtx SEND_ONLY
 * @param hEvent SEND_ONLY
 */
CUresult cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent);
/**
 * @guard CUDA_VERSION >= 12050
 * @param hCtx SEND_ONLY
 * @param hEvent SEND_ONLY
 */
CUresult cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent);
/**
 * @disabled server
 * @synchronize DEFERRED_DTOH STDOUT
 * @routingkey CURRENT_CONTEXT
 */
CUresult cuCtxSynchronize();
/**
 * @guard CUDA_VERSION >= 13000
 * @disabled server
 * @synchronize DEFERRED_DTOH STDOUT
 * @routingkey CONTEXT ctx
 * @param ctx SEND_ONLY
 */
CUresult cuCtxSynchronize_v2(CUcontext ctx);
/**
 * @param limit SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuCtxSetLimit(CUlimit limit, size_t value);
/**
 * @param pvalue RECV_ONLY
 * @param limit SEND_ONLY
 */
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit);
/**
 * @param pconfig RECV_ONLY
 */
CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig);
/**
 * @param config SEND_ONLY
 */
CUresult cuCtxSetCacheConfig(CUfunc_cache config);
/**
 * @param pConfig RECV_ONLY
 */
CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig);
/**
 * @param config SEND_ONLY
 */
CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);
/**
 * @param ctx SEND_ONLY
 * @param version RECV_ONLY
 */
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
/**
 * @disabled client - manual client handles nullable output pointers
 * @param leastPriority RECV_ONLY
 * @param greatestPriority RECV_ONLY
 */
CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
/**
 */
CUresult cuCtxResetPersistingL2Cache();
/**
 * @param pExecAffinity RECV_ONLY
 * @param type SEND_ONLY
 */
CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity,
                              CUexecAffinityType type);
/**
 * @disabled server - manual server coordinates retained staging lifecycle
 * @param pctx RECV_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags);
/**
 * @disabled server - manual server coordinates retained staging lifecycle
 * @param ctx SEND_ONLY
 */
CUresult cuCtxDetach(CUcontext ctx) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  return return_value;
}
/**
 * @disabled - manual client sends mapped file bytes to server
 * @param module RECV_ONLY
 * @param fname SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleLoad(CUmodule *module, const char *fname);
/**
 * @disabled
 * @param module RECV_ONLY
 * @param image SEND_ONLY
 */
CUresult cuModuleLoadData(CUmodule *module, const void *image);
/**
 * @disabled
 * @param module RECV_ONLY
 * @param image SEND_ONLY NULL_TERMINATED
 * @param numOptions SEND_ONLY
 * @param options SEND_ONLY LENGTH:numOptions
 * @param optionValues SEND_RECV
 */
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues);
/**
 * @routingkey MODULE hmod
 * @release MODULE hmod
 * @param hmod SEND_ONLY
 */
CUresult cuModuleUnload(CUmodule hmod) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_function_caches();
  return return_value;
}
/**
 * @param mode SEND_RECV
 */
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode);
/**
 * @disabled client - manual client follows lookup with a layout snapshot
 * @routingkey MODULE hmod
 * @recordowner FUNCTION hfunc
 * @param hfunc RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name);
/**
 * @routingkey MODULE hmod
 * @recordowner DEVICEPTR dptr
 * @param dptr RECV_ONLY NULLABLE
 * @param bytes RECV_ONLY NULLABLE
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 * @disabled server
 */
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                              const char *name) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr && bytes != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
  return return_value;
}
/**
 * @disabled - manual client handles JIT option values
 * @param numOptions SEND_ONLY
 * @param options SEND_RECV
 * @param optionValues SEND_RECV
 * @param stateOut SEND_RECV
 */
CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                         void **optionValues, CUlinkState *stateOut);
/**
 * @disabled - manual client handles JIT option values and input bytes
 * @param state SEND_ONLY
 * @param type SEND_ONLY
 * @param data SEND_ONLY
 * @param size SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 * @param numOptions SEND_ONLY
 * @param options SEND_ONLY LENGTH:numOptions
 * @param optionValues SEND_ONLY LENGTH:numOptions
 */
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data,
                          size_t size, const char *name,
                          unsigned int numOptions, CUjit_option *options,
                          void **optionValues);
/**
 * @disabled - manual client handles JIT option values and file bytes
 * @param state SEND_ONLY
 * @param type SEND_ONLY
 * @param path SEND_ONLY NULL_TERMINATED
 * @param numOptions SEND_ONLY
 * @param options SEND_ONLY LENGTH:numOptions
 * @param optionValues SEND_ONLY LENGTH:numOptions
 */
CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type,
                          const char *path, unsigned int numOptions,
                          CUjit_option *options, void **optionValues);
/**
 * @disabled - manual client owns returned cubin storage
 * @param state SEND_ONLY
 * @param cubinOut RECV_ONLY
 * @param sizeOut RECV_ONLY
 */
CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
/**
 * @disabled - manual client clears JIT state
 * @param state SEND_ONLY
 */
CUresult cuLinkDestroy(CUlinkState state);
/**
 * @param pTexRef RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
/**
 * @param pSurfRef RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                            const char *name);
/**
 * @disabled
 * @param library RECV_ONLY
 * @param code SEND_ONLY
 * @param numJitOptions SEND_ONLY
 * @param jitOptions SEND_ONLY LENGTH:numJitOptions
 * @param jitOptionsValues SEND_ONLY LENGTH:numJitOptions
 * @param numLibraryOptions SEND_ONLY
 * @param libraryOptions SEND_ONLY LENGTH:numLibraryOptions
 * @param libraryOptionValues SEND_ONLY LENGTH:numLibraryOptions
 */
CUresult cuLibraryLoadData(CUlibrary *library, const void *code,
                           CUjit_option *jitOptions, void **jitOptionsValues,
                           unsigned int numJitOptions,
                           CUlibraryOption *libraryOptions,
                           void **libraryOptionValues,
                           unsigned int numLibraryOptions);
/**
 * @recordowner LIBRARY library
 * @param library RECV_ONLY
 * @param fileName SEND_ONLY NULL_TERMINATED
 * @param numJitOptions SEND_ONLY
 * @param jitOptions SEND_ONLY LENGTH:numJitOptions
 * @param jitOptionsValues SEND_ONLY LENGTH:numJitOptions
 * @param numLibraryOptions SEND_ONLY
 * @param libraryOptions SEND_ONLY LENGTH:numLibraryOptions
 * @param libraryOptionValues SEND_ONLY LENGTH:numLibraryOptions
 */
CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName,
                               CUjit_option *jitOptions,
                               void **jitOptionsValues,
                               unsigned int numJitOptions,
                               CUlibraryOption *libraryOptions,
                               void **libraryOptionValues,
                               unsigned int numLibraryOptions);
/**
 * @disabled server - manual server keeps the library loaded, see the handler
 * @async
 * @routingkey LIBRARY library
 * @release LIBRARY library
 * @param library SEND_ONLY
 */
CUresult cuLibraryUnload(CUlibrary library) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_function_caches();
  return return_value;
}
/**
 * @disabled client - manual client serves the library kernel table
 * @routingkey LIBRARY library
 * @param pKernel RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                            const char *name);
/**
 * @routingkey LIBRARY library
 * @recordowner MODULE pMod
 * @recordparent LIBRARY pMod library
 * @param pMod RECV_ONLY
 * @param library SEND_ONLY
 * @disabled server
 */
CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library);
/**
 * @disabled client - manual client caches functions per kernel and context
 * @param pFunc RECV_ONLY
 * @param kernel SEND_ONLY
 */
CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel);
/**
 * @disabled client
 * @param pLib RECV_ONLY
 * @param kernel SEND_ONLY
 */
CUresult cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel);
/**
 * @disabled client - manual client caches the parameter layout per kernel
 * @routingkey FUNCTION kernel
 * @param kernel SEND_ONLY
 * @param paramIndex SEND_ONLY
 * @param paramOffset RECV_ONLY
 * @param paramSize RECV_ONLY
 */
CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex,
                              size_t *paramOffset, size_t *paramSize);
/**
 * @disabled client - manual client caches the parameter layout per function
 * @routingkey FUNCTION func
 * @param func SEND_ONLY
 * @param paramIndex SEND_ONLY
 * @param paramOffset RECV_ONLY
 * @param paramSize RECV_ONLY
 */
CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex,
                            size_t *paramOffset, size_t *paramSize);
/**
 * @routingkey LIBRARY library
 * @recordowner DEVICEPTR dptr
 * @param dptr RECV_ONLY NULLABLE
 * @param bytes RECV_ONLY NULLABLE
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library,
                            const char *name) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr && bytes != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
  return return_value;
}
/**
 * @routingkey LIBRARY library
 * @recordowner DEVICEPTR dptr
 * @param dptr RECV_ONLY NULLABLE
 * @param bytes RECV_ONLY NULLABLE
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                             CUlibrary library, const char *name) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr && bytes != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
  return return_value;
}
/**
 * @routingkey LIBRARY library
 * @param fptr RECV_ONLY
 * @param library SEND_ONLY
 * @param symbol SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library,
                                     const char *symbol);
/**
 * @disabled client - manual client caches attributes per kernel and device
 * @param pi SEND_RECV
 * @param attrib SEND_ONLY
 * @param kernel SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                              CUkernel kernel, CUdevice dev);
/**
 * @async
 * @param attrib SEND_ONLY
 * @param val SEND_ONLY
 * @param kernel SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val,
                              CUkernel kernel, CUdevice dev) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_kernel_attribute_cache_erase(lupine_route_identity(route), kernel,
                                        (int)attrib, (int)dev);
  return return_value;
}
/**
 * @param kernel SEND_ONLY
 * @param config SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config,
                                CUdevice dev);
/**
 * @guard CUDA_VERSION >= 12030
 * @routingkey FUNCTION hfunc
 * @retain name hfunc
 * @param name RECV_ONLY NULL_TERMINATED
 * @param hfunc SEND_ONLY
 */
CUresult cuKernelGetName(const char **name, CUkernel hfunc);
/**
 * @routingkey CURRENT_CONTEXT
 * @param free SEND_RECV
 * @param total SEND_RECV
 */
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
/**
 * @routingkey CURRENT_CONTEXT
 * @recordowner DEVICEPTR dptr
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return return_value;
}
/**
 * @routingkey CURRENT_CONTEXT
 * @recordowner DEVICEPTR dptr
 * @param dptr SEND_RECV
 * @param pPitch SEND_RECV
 * @param WidthInBytes SEND_ONLY
 * @param Height SEND_ONLY
 * @param ElementSizeBytes SEND_ONLY
 */
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    size_t allocation_size = 0;
    if (pPitch != nullptr)
      allocation_size = (*pPitch) * Height;
    else
      allocation_size = WidthInBytes * Height;
    lupine_note_deviceptr_allocation_route(*dptr, allocation_size, route);
  }
  return return_value;
}
/**
 * @disabled - manual client handles managed host alias; manual server releases
 * identity-mapped allocations
 * @param dptr SEND_ONLY
 */
CUresult cuMemFree_v2(CUdeviceptr dptr);
/**
 * @routingkey DEVICEPTR dptr
 * @param pbase SEND_RECV
 * @param psize SEND_RECV
 * @param dptr SEND_ONLY
 */
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize,
                                 CUdeviceptr dptr);
/**
 * @disabled client - manual client substitutes a local faulting address
 * @param pp SEND_RECV
 * @param bytesize SEND_ONLY
 */
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize);
/**
 * @disabled - manual client frees the substituted local address; manual server
 * releases identity-mapped allocations
 * @param p SEND_ONLY
 */
CUresult cuMemFreeHost(void *p);
/**
 * @disabled - manual client substitutes a local faulting address; manual server
 * returns the mapped device alias
 * @param pp SEND_RECV
 * @param bytesize SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
/**
 * @disabled client - manual client translates local host pointers
 * @param pdptr SEND_RECV
 * @param p SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p,
                                      unsigned int Flags);
/**
 * @disabled - client-local host allocation
 * @param pFlags SEND_RECV
 * @param p SEND_ONLY
 */
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p);
/**
 * @disabled - manual client creates managed host alias; manual server allocates
 * in the negotiated VA arena
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags);
/**
 * @routingkey ALL dev
 * @param dev RECV_ONLY NULLABLE
 * @param pciBusId SEND_ONLY NULL_TERMINATED
 */
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);
/**
 * @param len SEND_ONLY
 * @param pciBusId RECV_ONLY LENGTH:len
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
/**
 * @param pHandle SEND_RECV
 * @param event SEND_ONLY
 */
CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event);
/**
 * @param phEvent SEND_RECV
 * @param handle SEND_ONLY
 */
CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle);
/**
 * @param pHandle SEND_RECV
 * @param dptr SEND_ONLY
 */
CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr);
/**
 * @param pdptr SEND_RECV
 * @param handle SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                               unsigned int Flags);
/**
 * @param dptr SEND_ONLY
 */
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);
/**
 * @param p SEND_RECV
 * @param bytesize SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags);
/**
 * @param p SEND_RECV
 */
CUresult cuMemHostUnregister(void *p);
/**
 * @disabled client
 * @routingkey DEVICEPTR dst
 * @crossservercopy dst src ByteCount
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
/**
 * @crossservercopy dstDevice srcDevice ByteCount
 * @param dstDevice SEND_ONLY
 * @param dstContext SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param srcContext SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext,
                      size_t ByteCount);
/**
 * @disabled - manual client/server pipeline large host-to-device copies
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param srcHost SEND_ONLY LENGTH:ByteCount
 */
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount);
/**
 * @disabled - manual client/server chunk large host-copy responses
 * @routingkey DEVICEPTR srcDevice
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param dstHost RECV_ONLY LENGTH:ByteCount
 */
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @crossservercopy dstDevice srcDevice ByteCount
 * @param dstDevice SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t ByteCount);
/**
 * @routingkey DEVICEPTR srcDevice
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                         CUdeviceptr srcDevice, size_t ByteCount);
/**
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                         size_t srcOffset, size_t ByteCount);
/**
 * @disabled - manual client/server chunk large host-copy responses
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param dstHost RECV_ONLY LENGTH:ByteCount
 */
CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset,
                         size_t ByteCount);
/**
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                         size_t srcOffset, size_t ByteCount);
/**
 * @disabled
 * @param pCopy SEND_ONLY DEREF
 */
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy);
/**
 * @disabled
 * @param pCopy SEND_ONLY DEREF
 */
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 */
CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 */
CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy);
/**
 * @crossservercopy dstDevice srcDevice ByteCount STREAM:hStream ASYNC
 * @param dstDevice SEND_ONLY
 * @param dstContext SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param srcContext SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount, CUstream hStream);
/**
 * @disabled - manual client matches manual server stream-before-payload framing
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param srcHost SEND_ONLY LENGTH:ByteCount
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream);
/**
 * @disabled - manual client sends destination host pointer for capture
 * callbacks
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param dstHost RECV_ONLY LENGTH:ByteCount
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @crossservercopy dstDevice srcDevice ByteCount STREAM:hStream ASYNC
 * @param dstDevice SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream);
/**
 * @param dstHost SEND_RECV
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount, CUstream hStream);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param uc SEND_ONLY
 * @param N SEND_ONLY
 */
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param us SEND_ONLY
 * @param N SEND_ONLY
 */
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param ui SEND_ONLY
 * @param N SEND_ONLY
 */
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param uc SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 */
CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                         unsigned char uc, size_t Width, size_t Height);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param us SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 */
CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned short us, size_t Width, size_t Height);
/**
 * @synchronize
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param ui SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 */
CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned int ui, size_t Width, size_t Height);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param uc SEND_ONLY
 * @param N SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                         CUstream hStream);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param us SEND_ONLY
 * @param N SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                          CUstream hStream);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param ui SEND_ONLY
 * @param N SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                          CUstream hStream);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param uc SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                           unsigned char uc, size_t Width, size_t Height,
                           CUstream hStream);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param us SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned short us, size_t Width, size_t Height,
                            CUstream hStream);
/**
 * @async
 * @routingkey DEVICEPTR dstDevice
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param ui SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned int ui, size_t Width, size_t Height,
                            CUstream hStream);
/**
 * @param pHandle SEND_RECV
 * @param pAllocateArray SEND_ONLY DEREF
 */
CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
/**
 * @param pArrayDescriptor SEND_RECV
 * @param hArray SEND_ONLY
 */
CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                 CUarray hArray);
/**
 * @param sparseProperties SEND_RECV
 * @param array SEND_ONLY
 */
CUresult
cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties,
                           CUarray array);
/**
 * @param sparseProperties SEND_RECV
 * @param mipmap SEND_ONLY
 */
CUresult cuMipmappedArrayGetSparseProperties(
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap);
/**
 * @param memoryRequirements SEND_RECV
 * @param array SEND_ONLY
 * @param device SEND_ONLY
 */
CUresult
cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements,
                             CUarray array, CUdevice device);
/**
 * @param memoryRequirements SEND_RECV
 * @param mipmap SEND_ONLY
 * @param device SEND_ONLY
 */
CUresult cuMipmappedArrayGetMemoryRequirements(
    CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap,
    CUdevice device);
/**
 * @param pPlaneArray SEND_RECV
 * @param hArray SEND_ONLY
 * @param planeIdx SEND_ONLY
 */
CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray,
                         unsigned int planeIdx);
/**
 * @param hArray SEND_ONLY
 */
CUresult cuArrayDestroy(CUarray hArray);
/**
 * @param pHandle SEND_RECV
 * @param pAllocateArray SEND_ONLY DEREF
 */
CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
/**
 * @param pArrayDescriptor SEND_RECV
 * @param hArray SEND_ONLY
 */
CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                   CUarray hArray);
/**
 * @param pHandle SEND_RECV
 * @param pMipmappedArrayDesc SEND_ONLY DEREF
 * @param numMipmapLevels SEND_ONLY
 */
CUresult
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels);
/**
 * @param pLevelArray SEND_RECV
 * @param hMipmappedArray SEND_ONLY
 * @param level SEND_ONLY
 */
CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                  CUmipmappedArray hMipmappedArray,
                                  unsigned int level);
/**
 * @param hMipmappedArray SEND_ONLY
 */
CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
/**
 * @param handle SEND_RECV
 * @param dptr SEND_ONLY
 * @param size SEND_ONLY
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr,
                                       size_t size,
                                       CUmemRangeHandleType handleType,
                                       unsigned long long flags);
/**
 * @param ptr SEND_RECV
 * @param size SEND_ONLY
 * @param alignment SEND_ONLY
 * @param addr SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 */
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
/**
 * @param handle RECV_ONLY
 * @param size SEND_ONLY
 * @param prop SEND_ONLY DEREF
 * @param flags SEND_ONLY
 */
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags);
/**
 * @param handle SEND_ONLY
 */
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 * @param offset SEND_ONLY
 * @param handle SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags);
/**
 * @param mapInfoList SEND_RECV
 * @param count SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count,
                            CUstream hStream);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 */
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 * @param count SEND_ONLY
 * @param desc SEND_ONLY LENGTH:count
 */
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count);
/**
 * @param flags RECV_ONLY
 * @param location SEND_ONLY DEREF
 * @param ptr SEND_ONLY
 */
CUresult cuMemGetAccess(unsigned long long *flags,
                        const CUmemLocation *location, CUdeviceptr ptr);
/**
 * @disabled - POSIX fds cross the wire via the IPC fd broker
 */
CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                      CUmemGenericAllocationHandle handle,
                                      CUmemAllocationHandleType handleType,
                                      unsigned long long flags);
/**
 * @disabled - POSIX fds cross the wire via the IPC fd broker
 */
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                                        void *osHandle,
                                        CUmemAllocationHandleType shHandleType);
/**
 * @param granularity RECV_ONLY
 * @param prop SEND_ONLY DEREF
 * @param option SEND_ONLY
 */
CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                       const CUmemAllocationProp *prop,
                                       CUmemAllocationGranularity_flags option);
/**
 * @param prop SEND_RECV
 * @param handle SEND_ONLY
 */
CUresult
cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop,
                                       CUmemGenericAllocationHandle handle);
/**
 * @param handle SEND_RECV
 * @param addr SEND_RECV
 */
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle,
                                     void *addr);
/**
 * @routingkey DEVICEPTR dptr
 * @param dptr SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_forget_deviceptr_owner(dptr);
  return return_value;
}
/**
 * @routingkey STREAM hStream
 * @recordowner DEVICEPTR dptr
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return return_value;
}
/**
 * @param pool SEND_ONLY
 * @param minBytesToKeep SEND_ONLY
 */
CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
/**
 * @param pool SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 * @disabled server
 */
CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr,
                               void *value);
/**
 * @param pool SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 * @disabled server
 */
CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr,
                               void *value);
/**
 * @param pool SEND_ONLY
 * @param map SEND_ONLY LENGTH:count
 * @param count SEND_ONLY
 */
CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map,
                            size_t count);
/**
 * @param flags SEND_RECV
 * @param memPool SEND_ONLY
 * @param location SEND_RECV
 */
CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool,
                            CUmemLocation *location);
/**
 * @recordowner MEMORY_POOL pool
 * @param pool SEND_RECV
 * @param poolProps SEND_ONLY DEREF
 */
CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
/**
 * @param pool SEND_ONLY
 */
CUresult cuMemPoolDestroy(CUmemoryPool pool);
/**
 * @routingkey STREAM hStream
 * @recordowner DEVICEPTR dptr
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 * @param pool SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                 CUmemoryPool pool, CUstream hStream) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return return_value;
}
/**
 * @disabled - POSIX fds cross the wire via the IPC fd broker
 */
CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool,
                                          CUmemAllocationHandleType handleType,
                                          unsigned long long flags);
/**
 * @disabled - POSIX fds cross the wire via the IPC fd broker
 */
CUresult
cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle,
                                   CUmemAllocationHandleType handleType,
                                   unsigned long long flags);
/**
 * @param shareData_out SEND_RECV
 * @param ptr SEND_ONLY
 */
CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out,
                                CUdeviceptr ptr);
/**
 * @param ptr_out SEND_RECV
 * @param pool SEND_ONLY
 * @param shareData SEND_RECV
 */
CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool,
                                CUmemPoolPtrExportData *shareData);
/**
 * @param data SEND_RECV
 * @param attribute SEND_ONLY
 * @param ptr SEND_ONLY
 * @disabled server
 */
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                               CUdeviceptr ptr);
#ifdef cuMemPrefetchAsync
#undef cuMemPrefetchAsync
#endif
/**
 * @disabled client - manual client handles managed-pointer and target routing
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param dstDevice SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                            CUdevice dstDevice, CUstream hStream);
/**
 * @guard CUDA_VERSION >= 12020
 * @disabled client - manual client handles managed-pointer and target routing
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param location SEND_ONLY
 * @param flags SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count,
                               CUmemLocation location, unsigned int flags,
                               CUstream hStream);
#ifdef cuMemAdvise
#undef cuMemAdvise
#endif
/**
 * @disabled client - manual client handles managed-pointer and target routing
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param device SEND_ONLY
 */
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
                     CUdevice device);
/**
 * @guard CUDA_VERSION >= 12020
 * @disabled client - manual client handles managed-pointer and target routing
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param location SEND_ONLY
 */
CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
                        CUmemLocation location);
/**
 * @routingkey DEVICEPTR devPtr
 * @param dataSize SEND_ONLY
 * @param data RECV_ONLY LENGTH:dataSize
 * @param attribute SEND_ONLY
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuMemRangeGetAttribute(void *data, size_t dataSize,
                                CUmem_range_attribute attribute,
                                CUdeviceptr devPtr, size_t count);
/**
 * @disabled - manual client and server marshal each variable-sized result
 * @param data SEND_RECV
 * @param dataSizes SEND_RECV
 * @param attributes SEND_RECV
 * @param numAttributes SEND_ONLY
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                 CUmem_range_attribute *attributes,
                                 size_t numAttributes, CUdeviceptr devPtr,
                                 size_t count);
/**
 * @disabled - manual client sends the attribute value, not the caller's pointer
 * @param value SEND_ONLY
 * @param attribute SEND_ONLY
 * @param ptr SEND_ONLY
 */
CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                               CUdeviceptr ptr);
/**
 * @disabled - manual client handles managed host aliases
 * @param numAttributes SEND_ONLY
 * @param attributes SEND_RECV
 * @param data SEND_RECV
 * @param ptr SEND_ONLY
 */
CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                CUpointer_attribute *attributes, void **data,
                                CUdeviceptr ptr);
/**
 * @routingkey CURRENT_CONTEXT
 * @recordowner STREAM phStream
 * @param phStream SEND_RECV
 * @param Flags SEND_ONLY
 */
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
/**
 * @routingkey CURRENT_CONTEXT
 * @recordowner STREAM phStream
 * @param phStream SEND_RECV
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                    int priority);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param priority SEND_RECV
 */
CUresult cuStreamGetPriority(CUstream hStream, int *priority);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param flags SEND_RECV
 */
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param streamId SEND_RECV
 */
CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param pctx SEND_RECV
 */
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
/**
 * @guard CUDA_VERSION >= 13010
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
CUresult cuStreamGetDevResource(CUstream hStream, CUdevResource *resource,
                                CUdevResourceType type);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param phCtx RECV_ONLY
 */
CUresult cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx *phCtx);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @param hCtx SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
CUresult cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource *resource,
                                  CUdevResourceType type);
/**
 * @guard CUDA_VERSION >= 12040
 * @param device SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
CUresult cuDeviceGetDevResource(CUdevice device, CUdevResource *resource,
                                CUdevResourceType type);
/**
 * @guard CUDA_VERSION >= 12040
 * @param hCtx SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
CUresult cuCtxGetDevResource(CUcontext hCtx, CUdevResource *resource,
                             CUdevResourceType type);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @param nbGroups SEND_RECV
 * @param result RECV_ONLY NULLABLE LENGTH:nbGroups
 * @param input SEND_ONLY DEREF
 * @param remainder RECV_ONLY NULLABLE
 * @param flags SEND_ONLY
 * @param minCount SEND_ONLY
 */
CUresult cuDevSmResourceSplitByCount(CUdevResource *result,
                                     unsigned int *nbGroups,
                                     const CUdevResource *input,
                                     CUdevResource *remainder,
                                     unsigned int flags,
                                     unsigned int minCount);
/**
 * @guard CUDA_VERSION >= 13010
 * @routingkey CURRENT_CONTEXT
 * @param nbGroups SEND_ONLY
 * @param result RECV_ONLY NULLABLE LENGTH:nbGroups
 * @param input SEND_ONLY DEREF
 * @param remainder RECV_ONLY NULLABLE
 * @param flags SEND_ONLY
 * @param groupParams SEND_RECV LENGTH:nbGroups
 */
CUresult cuDevSmResourceSplit(
    CUdevResource *result, unsigned int nbGroups,
    const CUdevResource *input, CUdevResource *remainder, unsigned int flags,
    CU_DEV_SM_RESOURCE_GROUP_PARAMS *groupParams);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @param nbResources SEND_ONLY
 * @param resources SEND_ONLY LENGTH:nbResources
 * @param phDesc RECV_ONLY
 */
CUresult cuDevResourceGenerateDesc(CUdevResourceDesc *phDesc,
                                   CUdevResource *resources,
                                   unsigned int nbResources);
/**
 * @guard CUDA_VERSION >= 12040
 * @param phCtx RECV_ONLY
 * @param desc SEND_ONLY
 * @param dev SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuGreenCtxCreate(CUgreenCtx *phCtx, CUdevResourceDesc desc,
                          CUdevice dev, unsigned int flags);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @recordowner CONTEXT pContext
 * @param pContext RECV_ONLY
 * @param hCtx SEND_ONLY
 */
CUresult cuCtxFromGreenCtx(CUcontext *pContext, CUgreenCtx hCtx) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && pContext != nullptr)
    lupine_mark_context_green(*pContext);
  return return_value;
}
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @param hCtx SEND_ONLY
 */
CUresult cuGreenCtxDestroy(CUgreenCtx hCtx) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_forget_destroyed_context(reinterpret_cast<CUcontext>(hCtx));
  return return_value;
}
/**
 * @guard CUDA_VERSION >= 12050
 * @routingkey CURRENT_CONTEXT
 * @recordowner STREAM phStream
 * @param phStream RECV_ONLY
 * @param greenCtx SEND_ONLY
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
CUresult cuGreenCtxStreamCreate(CUstream *phStream, CUgreenCtx greenCtx,
                                unsigned int flags, int priority);
/**
 * @guard CUDA_VERSION >= 13000
 * @routingkey CURRENT_CONTEXT
 * @param greenCtx SEND_ONLY
 * @param greenCtxId RECV_ONLY
 */
CUresult cuGreenCtxGetId(CUgreenCtx greenCtx, unsigned long long *greenCtxId);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @param hCtx SEND_ONLY
 * @param hEvent SEND_ONLY
 */
CUresult cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent);
/**
 * @guard CUDA_VERSION >= 12040
 * @routingkey CURRENT_CONTEXT
 * @param hCtx SEND_ONLY
 * @param hEvent SEND_ONLY
 */
CUresult cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent);
/**
 * @disabled - manual client handles cross-server event waits
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param hEvent SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                           unsigned int Flags);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param callback SEND_ONLY
 * @param userData SEND_RECV
 * @param flags SEND_ONLY
 * @disabled server
 */
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback,
                             void *userData, unsigned int flags);
/**
 * @disabled handle_cuStreamBeginCapture - manual client coordinates checkpoint
 * capture admission
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param mode SEND_ONLY
 */
CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);
/**
 * @param mode SEND_RECV
 */
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode);
/**
 * @disabled - manual client coordinates checkpoint capture completion
 * @recordowner GRAPH phGraph
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param phGraph SEND_RECV NULLABLE
 */
CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph);
/**
 * @routingkey STREAM hStream
 * @disabled client - manual client answers from the local capture count
 * @param hStream SEND_ONLY
 * @param captureStatus SEND_RECV
 */
CUresult cuStreamIsCapturing(CUstream hStream,
                             CUstreamCaptureStatus *captureStatus);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamUpdateCaptureDependencies(CUstream hStream,
                                           CUgraphNode *dependencies,
                                           size_t numDependencies,
                                           unsigned int flags);
/**
 * @routingkey STREAM hStream
 * @routingfallback DEVICEPTR dptr
 * @param hStream SEND_ONLY
 * @param dptr SEND_ONLY
 * @param length SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                size_t length, unsigned int flags);
/**
 * @synchronize
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 */
CUresult cuStreamQuery(CUstream hStream);
/**
 * @disabled server
 * @synchronize DEFERRED_DTOH STDOUT
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 */
CUresult cuStreamSynchronize(CUstream hStream);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 */
CUresult cuStreamDestroy_v2(CUstream hStream) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS)
    lupine_forget_stream_owner(hStream);
  return return_value;
}
/**
 * @routingkey STREAM dst
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
CUresult cuStreamCopyAttributes(CUstream dst, CUstream src);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out SEND_RECV
 */
CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                              CUstreamAttrValue *value_out);
/**
 * @routingkey STREAM hStream
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY DEREF
 */
CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                              const CUstreamAttrValue *value);
/**
 * @routingkey CURRENT_CONTEXT
 * @recordowner EVENT phEvent
 * @param phEvent SEND_RECV
 * @param Flags SEND_ONLY
 */
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
/**
 * @async
 * @disabled - manual client tracks which events have work outstanding
 * @routingkey STREAM hStream
 * @param hEvent SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
/**
 * @async
 * @disabled - manual client tracks which events have work outstanding
 * @routingkey STREAM hStream
 * @param hEvent SEND_ONLY
 * @param hStream SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                unsigned int flags);
/**
 * @disabled - manual client caches completions and separately prefetches events
 * @synchronize DEFERRED_DTOH
 * @routingkey EVENT hEvent
 * @param hEvent SEND_ONLY
 */
CUresult cuEventQuery(CUevent hEvent);
/**
 * @disabled server
 * @synchronize DEFERRED_DTOH STDOUT
 * @routingkey EVENT hEvent
 * @param hEvent SEND_ONLY
 */
CUresult cuEventSynchronize(CUevent hEvent);
/**
 * @disabled - manual server clears deferred-copy event bookkeeping
 * @routingkey EVENT hEvent
 * @param hEvent SEND_ONLY
 */
CUresult cuEventDestroy_v2(CUevent hEvent);
/**
 * @param pMilliseconds SEND_RECV
 * @param hStart SEND_ONLY
 * @param hEnd SEND_ONLY
 */
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
/**
 * @param extMem_out SEND_RECV
 * @param memHandleDesc SEND_ONLY DEREF
 */
CUresult
cuImportExternalMemory(CUexternalMemory *extMem_out,
                       const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc);
/**
 * @param devPtr SEND_RECV
 * @param extMem SEND_ONLY
 * @param bufferDesc SEND_ONLY DEREF
 */
CUresult cuExternalMemoryGetMappedBuffer(
    CUdeviceptr *devPtr, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc);
/**
 * @param mipmap SEND_RECV
 * @param extMem SEND_ONLY
 * @param mipmapDesc SEND_ONLY DEREF
 */
CUresult cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray *mipmap, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc);
/**
 * @param extMem SEND_ONLY
 */
CUresult cuDestroyExternalMemory(CUexternalMemory extMem);
/**
 * @param extSem_out SEND_RECV
 * @param semHandleDesc SEND_ONLY DEREF
 */
CUresult cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out,
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc);
/**
 * @param extSemArray SEND_ONLY LENGTH:numExtSems
 * @param paramsArray SEND_ONLY LENGTH:numExtSems
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
CUresult cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream);
/**
 * @param extSemArray SEND_ONLY LENGTH:numExtSems
 * @param paramsArray SEND_ONLY LENGTH:numExtSems
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
CUresult cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream);
/**
 * @param extSem SEND_ONLY
 */
CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr,
                                cuuint32_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr,
                                cuuint64_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint32_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint64_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param count SEND_ONLY
 * @param paramArray SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count,
                               CUstreamBatchMemOpParams *paramArray,
                               unsigned int flags);
/**
 * @disabled client - manual client caches successful attribute queries
 * @routingkey FUNCTION hfunc
 * @param pi SEND_RECV
 * @param attrib SEND_ONLY
 * @param hfunc SEND_ONLY
 */
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                            CUfunction hfunc);
/**
 * @routingkey FUNCTION hfunc
 * @param hfunc SEND_ONLY
 * @param attrib SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                            int value) {
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS) {
    lupine_invalidate_kernel_attribute_cache();
    lupine_invalidate_function_attribute_cache();
  }
  return return_value;
}
/**
 * @routingkey FUNCTION hfunc
 * @param hfunc SEND_ONLY
 * @param config SEND_ONLY
 */
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
/**
 * @routingkey FUNCTION hfunc
 * @param hfunc SEND_ONLY
 * @param config SEND_ONLY
 */
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
/**
 * @routingkey FUNCTION hfunc
 * @recordowner MODULE hmod
 * @param hmod SEND_RECV
 * @param hfunc SEND_ONLY
 */
CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc);
/**
 * @guard CUDA_VERSION >= 12030
 * @routingkey FUNCTION hfunc
 * @retain name hfunc
 * @param name RECV_ONLY NULL_TERMINATED
 * @param hfunc SEND_ONLY
 */
CUresult cuFuncGetName(const char **name, CUfunction hfunc);

/**
 * @disabled
 * @param f SEND_ONLY
 * @param gridDimX SEND_ONLY
 * @param gridDimY SEND_ONLY
 * @param gridDimZ SEND_ONLY
 * @param blockDimX SEND_ONLY
 * @param blockDimY SEND_ONLY
 * @param blockDimZ SEND_ONLY
 * @param sharedMemBytes SEND_ONLY
 * @param hStream SEND_ONLY
 * @param kernelParams SEND_ONLY
 * @param extra SEND_ONLY
 */
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra);
/**
 * @disabled
 * @param config SEND_ONLY DEREFERENCE
 * @param f SEND_ONLY
 * @param kernelParams SEND_ONLY
 * @param extra SEND_ONLY
 */
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra);
/**
 * @disabled - manual kernel parameter packing
 * @param f SEND_ONLY
 * @param gridDimX SEND_ONLY
 * @param gridDimY SEND_ONLY
 * @param gridDimZ SEND_ONLY
 * @param blockDimX SEND_ONLY
 * @param blockDimY SEND_ONLY
 * @param blockDimZ SEND_ONLY
 * @param sharedMemBytes SEND_ONLY
 * @param hStream SEND_ONLY
 * @param kernelParams SEND_RECV
 */
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams);
/**
 * @param launchParamsList SEND_RECV
 * @param numDevices SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult
cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList,
                                     unsigned int numDevices,
                                     unsigned int flags);
/**
 * @disabled - manual host callback forwarding
 * @param hStream SEND_ONLY
 * @param fn SEND_ONLY
 * @param userData SEND_RECV
 */
CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);
/**
 * @param hfunc SEND_ONLY
 * @param x SEND_ONLY
 * @param y SEND_ONLY
 * @param z SEND_ONLY
 */
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
/**
 * @param hfunc SEND_ONLY
 * @param bytes SEND_ONLY
 */
CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
/**
 * @param hfunc SEND_ONLY
 * @param numbytes SEND_ONLY
 */
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
/**
 * @param hfunc SEND_ONLY
 * @param offset SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value);
/**
 * @param hfunc SEND_ONLY
 * @param offset SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuParamSetf(CUfunction hfunc, int offset, float value);
/**
 * @param hfunc SEND_ONLY
 * @param offset SEND_ONLY
 * @param ptr SEND_RECV
 * @param numbytes SEND_ONLY
 */
CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr,
                     unsigned int numbytes);
/**
 * @param f SEND_ONLY
 */
CUresult cuLaunch(CUfunction f);
/**
 * @param f SEND_ONLY
 * @param grid_width SEND_ONLY
 * @param grid_height SEND_ONLY
 */
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
/**
 * @param f SEND_ONLY
 * @param grid_width SEND_ONLY
 * @param grid_height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream);
/**
 * @param hfunc SEND_ONLY
 * @param texunit SEND_ONLY
 * @param hTexRef SEND_ONLY
 */
CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
/**
 * @recordowner GRAPH phGraph
 * @param phGraph SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags);
/**
 * @disabled handle_cuGraphAddKernelNode - manual kernel parameter packing
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                                 const CUgraphNode *dependencies,
                                 size_t numDependencies,
                                 const CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @disabled handle_cuGraphKernelNodeGetParams - manual kernel parameter packing
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode,
                                       CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @disabled handle_cuGraphKernelNodeSetParams - manual kernel parameter packing
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult
cuGraphKernelNodeSetParams_v2(CUgraphNode hNode,
                              const CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @disabled - manual host pointer serialization
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param copyParams SEND_RECV
 * @param ctx SEND_ONLY
 */
CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                              const CUgraphNode *dependencies,
                              size_t numDependencies,
                              const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode,
                                    CUDA_MEMCPY3D *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode,
                                    const CUDA_MEMCPY3D *nodeParams);
/**
 * @disabled - manual nested struct serialization
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param memsetParams SEND_RECV
 * @param ctx SEND_ONLY
 */
CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                              const CUgraphNode *dependencies,
                              size_t numDependencies,
                              const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                              CUcontext ctx);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode,
                                    CUDA_MEMSET_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode,
                                    const CUDA_MEMSET_NODE_PARAMS *nodeParams);
/**
 * @disabled - manual callback forwarding
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                            const CUgraphNode *dependencies,
                            size_t numDependencies,
                            const CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @disabled - manual host callback trampoline (see cuGraphAddHostNode)
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphHostNodeGetParams(CUgraphNode hNode,
                                  CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @disabled - manual host callback trampoline (see cuGraphAddHostNode)
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphHostNodeSetParams(CUgraphNode hNode,
                                  const CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param childGraph SEND_ONLY
 */
CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                  const CUgraphNode *dependencies,
                                  size_t numDependencies, CUgraph childGraph);
/**
 * @recordowner GRAPH phGraph
 * @param hNode SEND_ONLY
 * @param phGraph SEND_RECV
 */
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 */
CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                             const CUgraphNode *dependencies,
                             size_t numDependencies);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param event SEND_ONLY
 */
CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                   const CUgraphNode *dependencies,
                                   size_t numDependencies, CUevent event);
/**
 * @param hNode SEND_ONLY
 * @param event_out SEND_RECV
 */
CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
/**
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param event SEND_ONLY
 */
CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                 const CUgraphNode *dependencies,
                                 size_t numDependencies, CUevent event);
/**
 * @param hNode SEND_ONLY
 * @param event_out SEND_RECV
 */
CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
/**
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams extSemArray numExtSems
 * @deeparray nodeParams paramsArray numExtSems
 */
CUresult cuGraphAddExternalSemaphoresSignalNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out RECV_ONLY
 * @deeparray params_out extSemArray numExtSems
 * @deeparray params_out paramsArray numExtSems
 */
CUresult cuGraphExternalSemaphoresSignalNodeGetParams(
    CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams extSemArray numExtSems
 * @deeparray nodeParams paramsArray numExtSems
 */
CUresult cuGraphExternalSemaphoresSignalNodeSetParams(
    CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams extSemArray numExtSems
 * @deeparray nodeParams paramsArray numExtSems
 */
CUresult cuGraphAddExternalSemaphoresWaitNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out RECV_ONLY
 * @deeparray params_out extSemArray numExtSems
 * @deeparray params_out paramsArray numExtSems
 */
CUresult cuGraphExternalSemaphoresWaitNodeGetParams(
    CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams extSemArray numExtSems
 * @deeparray nodeParams paramsArray numExtSems
 */
CUresult cuGraphExternalSemaphoresWaitNodeSetParams(
    CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams paramArray count
 */
CUresult cuGraphAddBatchMemOpNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams_out RECV_ONLY
 * @deeparray nodeParams_out paramArray count
 */
CUresult
cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode,
                               CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams paramArray count
 */
CUresult
cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode,
                               const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams paramArray count
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecBatchMemOpNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                const CUgraphNode *dependencies,
                                size_t numDependencies,
                                CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out SEND_RECV
 */
CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode,
                                      CUDA_MEM_ALLOC_NODE_PARAMS *params_out);
/**
 * @recordowner GRAPH_NODE phGraphNode
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dptr SEND_ONLY
 */
CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                               const CUgraphNode *dependencies,
                               size_t numDependencies, CUdeviceptr dptr);
/**
 * @param hNode SEND_ONLY
 * @param dptr_out SEND_RECV
 */
CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out);
/**
 * @param device SEND_ONLY
 */
CUresult cuDeviceGraphMemTrim(CUdevice device);
/**
 * @param device SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 * @disabled server
 */
CUresult cuDeviceGetGraphMemAttribute(CUdevice device,
                                      CUgraphMem_attribute attr, void *value);
/**
 * @param device SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 * @disabled server
 */
CUresult cuDeviceSetGraphMemAttribute(CUdevice device,
                                      CUgraphMem_attribute attr, void *value);
/**
 * @recordowner GRAPH phGraphClone
 * @param phGraphClone SEND_RECV
 * @param originalGraph SEND_ONLY
 * @disabled server
 */
CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph);
/**
 * @recordowner GRAPH_NODE phNode
 * @param phNode SEND_RECV
 * @param hOriginalNode SEND_ONLY
 * @param hClonedGraph SEND_ONLY
 */
CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode,
                                CUgraph hClonedGraph);
/**
 * @param hNode SEND_ONLY
 * @param type SEND_RECV
 */
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type);
/**
 * @guard CUDA_VERSION >= 13010
 * @recordowner GRAPH phGraph
 * @param hNode SEND_ONLY
 * @param phGraph RECV_ONLY
 */
CUresult cuGraphNodeGetContainingGraph(CUgraphNode hNode, CUgraph *phGraph);
/**
 * @guard CUDA_VERSION >= 13010
 * @param hNode SEND_ONLY
 * @param nodeId RECV_ONLY
 */
CUresult cuGraphNodeGetLocalId(CUgraphNode hNode, unsigned int *nodeId);
/**
 * @guard CUDA_VERSION >= 13010
 * @param hNode SEND_ONLY
 * @param toolsNodeId RECV_ONLY
 */
CUresult cuGraphNodeGetToolsId(CUgraphNode hNode,
                               unsigned long long *toolsNodeId);
/**
 * @guard CUDA_VERSION >= 13010
 * @param hGraph SEND_ONLY
 * @param graphId RECV_ONLY
 */
CUresult cuGraphGetId(CUgraph hGraph, unsigned int *graphId);
/**
 * @guard CUDA_VERSION >= 13010
 * @param hGraphExec SEND_ONLY
 * @param graphId RECV_ONLY
 */
CUresult cuGraphExecGetId(CUgraphExec hGraphExec, unsigned int *graphId);
/**
 * @param hGraph SEND_ONLY
 * @param numNodes SEND_RECV
 * @param nodes RECV_ONLY NULLABLE LENGTH:numNodes
 */
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
/**
 * @param hGraph SEND_ONLY
 * @param numRootNodes SEND_RECV
 * @param rootNodes RECV_ONLY NULLABLE LENGTH:numRootNodes
 */
CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes,
                             size_t *numRootNodes);
/**
 * @param hGraph SEND_ONLY
 * @param numEdges SEND_RECV
 * @param from RECV_ONLY NULLABLE LENGTH:numEdges
 * @param to RECV_ONLY NULLABLE LENGTH:numEdges
 * @param edgeData RECV_ONLY NULLABLE LENGTH:numEdges
 */
CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to,
                         CUgraphEdgeData *edgeData, size_t *numEdges);
/**
 * @param hNode SEND_ONLY
 * @param numDependencies SEND_RECV
 * @param dependencies RECV_ONLY NULLABLE LENGTH:numDependencies
 * @param edgeData RECV_ONLY NULLABLE LENGTH:numDependencies
 */
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode,
                                    CUgraphNode *dependencies,
                                    CUgraphEdgeData *edgeData,
                                    size_t *numDependencies);
/**
 * @param hNode SEND_ONLY
 * @param numDependentNodes SEND_RECV
 * @param dependentNodes RECV_ONLY NULLABLE LENGTH:numDependentNodes
 * @param edgeData RECV_ONLY NULLABLE LENGTH:numDependentNodes
 */
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                      CUgraphNode *dependentNodes,
                                      CUgraphEdgeData *edgeData,
                                      size_t *numDependentNodes);
/**
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 */
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from,
                                const CUgraphNode *to, size_t numDependencies);
/**
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 */
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from,
                                   const CUgraphNode *to,
                                   size_t numDependencies);
/**
 * @param hNode SEND_ONLY
 */
CUresult cuGraphDestroyNode(CUgraphNode hNode);
/**
 * @disabled server - manual server retains graph staging resources
 * @recordowner GRAPH_EXEC phGraphExec
 * @param phGraphExec RECV_ONLY
 * @param hGraph SEND_ONLY
 * @param phErrorNode RECV_ONLY NULLABLE
 * @param bufferSize SEND_ONLY
 * @param logBuffer RECV_ONLY NULLABLE LENGTH:bufferSize ON_ERROR
 */
CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph,
                               CUgraphNode *phErrorNode, char *logBuffer,
                               size_t bufferSize);
/**
 * @recordowner GRAPH_EXEC phGraphExec
 * @param phGraphExec SEND_RECV
 * @param hGraph SEND_ONLY
 * @param flags SEND_ONLY
 * @disabled server
 */
CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph,
                                     unsigned long long flags);
/**
 * @recordowner GRAPH_EXEC phGraphExec
 * @param phGraphExec SEND_RECV
 * @param hGraph SEND_ONLY
 * @param instantiateParams SEND_RECV
 * @disabled server
 */
CUresult
cuGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph,
                             CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param flags SEND_RECV
 */
CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t *flags);
/**
 * @guard CUDA_VERSION >= 12020
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult cuGraphNodeSetParams(CUgraphNode hNode,
                              CUgraphNodeParams *nodeParams);
/**
 * @guard CUDA_VERSION >= 12020
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                  CUgraphNodeParams *nodeParams);
/**
 * @disabled handle_cuGraphExecKernelNodeSetParams - manual kernel parameter
 * packing
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult
cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode,
                                  const CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param copyParams SEND_ONLY DEREF
 * @param ctx SEND_ONLY
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec,
                                        CUgraphNode hNode,
                                        const CUDA_MEMCPY3D *copyParams,
                                        CUcontext ctx);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param memsetParams SEND_ONLY DEREF
 * @param ctx SEND_ONLY
 * @graphexecnode hGraphExec hNode
 */
CUresult
cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                               const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                               CUcontext ctx);
/**
 * @disabled - manual host callback trampoline (see cuGraphAddHostNode)
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY DEREF
 */
CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                      const CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param childGraph SEND_ONLY
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec,
                                            CUgraphNode hNode,
                                            CUgraph childGraph);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec,
                                            CUgraphNode hNode, CUevent event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec,
                                          CUgraphNode hNode, CUevent event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams extSemArray numExtSems
 * @deeparray nodeParams paramsArray numExtSems
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams extSemArray numExtSems
 * @deeparray nodeParams paramsArray numExtSems
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_ONLY
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode,
                               unsigned int isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_RECV
 * @graphexecnode hGraphExec hNode
 */
CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode,
                               unsigned int *isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
/**
 * @param hGraphExec SEND_ONLY
 * @param hStream SEND_ONLY
 * @disabled server
 */
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
/**
 * @param hGraphExec SEND_ONLY
 * @disabled server
 */
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);
/**
 * @param hGraph SEND_ONLY
 * @disabled server
 */
CUresult cuGraphDestroy(CUgraph hGraph);
#ifdef cuGraphExecUpdate
#undef cuGraphExecUpdate
#endif
/**
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param hErrorNode_out RECV_ONLY NULLABLE
 * @param updateResult_out RECV_ONLY NULLABLE
 */
CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                           CUgraphNode *hErrorNode_out,
                           CUgraphExecUpdateResult *updateResult_out);
/**
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param resultInfo SEND_RECV
 */
CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph,
                              CUgraphExecUpdateResultInfo *resultInfo);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out SEND_RECV
 */
CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode,
                                       CUkernelNodeAttrID attr,
                                       CUkernelNodeAttrValue *value_out);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY DEREF
 */
CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode,
                                       CUkernelNodeAttrID attr,
                                       const CUkernelNodeAttrValue *value);
/**
 * @param hGraph SEND_ONLY
 * @param path SEND_ONLY NULL_TERMINATED
 * @param flags SEND_ONLY
 */
CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path,
                              unsigned int flags);
/**
 * @param object_out SEND_RECV
 * @param ptr SEND_RECV
 * @param destroy SEND_ONLY
 * @param initialRefcount SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr,
                            CUhostFn destroy, unsigned int initialRefcount,
                            unsigned int flags);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuUserObjectRetain(CUuserObject object, unsigned int count);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuUserObjectRelease(CUuserObject object, unsigned int count);
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object,
                                 unsigned int count, unsigned int flags);
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object,
                                  unsigned int count);
/**
 * @disabled client - manual client caches occupancy per function
 * @param numBlocks SEND_RECV
 * @param func SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize);
/**
 * @disabled client - manual client caches occupancy per function
 * @param numBlocks SEND_RECV
 * @param func SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags);
/**
 * @disabled
 * @param minGridSize SEND_RECV
 * @param blockSize SEND_RECV
 * @param func SEND_ONLY
 * @param blockSizeToDynamicSMemSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param blockSizeLimit SEND_ONLY
 */
CUresult
cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                 CUfunction func,
                                 CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                 size_t dynamicSMemSize, int blockSizeLimit);
/**
 * @disabled
 * @param minGridSize SEND_RECV
 * @param blockSize SEND_RECV
 * @param func SEND_ONLY
 * @param blockSizeToDynamicSMemSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param blockSizeLimit SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags);
/**
 * @param dynamicSmemSize SEND_RECV
 * @param func SEND_ONLY
 * @param numBlocks SEND_ONLY
 * @param blockSize SEND_ONLY
 */
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                 CUfunction func, int numBlocks,
                                                 int blockSize);
/**
 * @param clusterSize SEND_RECV
 * @param func SEND_ONLY
 * @param config SEND_ONLY
 * @deeparray config attrs numAttrs
 */
CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func,
                                            const CUlaunchConfig *config);
/**
 * @param numClusters SEND_RECV
 * @param func SEND_ONLY
 * @param config SEND_ONLY
 * @deeparray config attrs numAttrs
 */
CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func,
                                      const CUlaunchConfig *config);
/**
 * @param hTexRef SEND_ONLY
 * @param hArray SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
/**
 * @param hTexRef SEND_ONLY
 * @param hMipmappedArray SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                   CUmipmappedArray hMipmappedArray,
                                   unsigned int Flags);
/**
 * @param ByteOffset SEND_RECV
 * @param hTexRef SEND_ONLY
 * @param dptr SEND_ONLY
 * @param bytes SEND_ONLY
 */
CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef,
                               CUdeviceptr dptr, size_t bytes);
/**
 * @param hTexRef SEND_ONLY
 * @param desc SEND_ONLY DEREF
 * @param dptr SEND_ONLY
 * @param Pitch SEND_ONLY
 */
CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef,
                                 const CUDA_ARRAY_DESCRIPTOR *desc,
                                 CUdeviceptr dptr, size_t Pitch);
/**
 * @param hTexRef SEND_ONLY
 * @param fmt SEND_ONLY
 * @param NumPackedComponents SEND_ONLY
 */
CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                           int NumPackedComponents);
/**
 * @param hTexRef SEND_ONLY
 * @param dim SEND_ONLY
 * @param am SEND_ONLY
 */
CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
/**
 * @param hTexRef SEND_ONLY
 * @param fm SEND_ONLY
 */
CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
/**
 * @param hTexRef SEND_ONLY
 * @param fm SEND_ONLY
 */
CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
/**
 * @param hTexRef SEND_ONLY
 * @param bias SEND_ONLY
 */
CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
/**
 * @param hTexRef SEND_ONLY
 * @param minMipmapLevelClamp SEND_ONLY
 * @param maxMipmapLevelClamp SEND_ONLY
 */
CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                     float minMipmapLevelClamp,
                                     float maxMipmapLevelClamp);
/**
 * @param hTexRef SEND_ONLY
 * @param maxAniso SEND_ONLY
 */
CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
/**
 * @param hTexRef SEND_ONLY
 * @param pBorderColor SEND_RECV
 */
CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor);
/**
 * @param hTexRef SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
/**
 * @param pdptr SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef);
/**
 * @param phArray SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef);
/**
 * @param phMipmappedArray SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                   CUtexref hTexRef);
/**
 * @param pam SEND_RECV
 * @param hTexRef SEND_ONLY
 * @param dim SEND_ONLY
 */
CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim);
/**
 * @param pfm SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
/**
 * @param pFormat SEND_RECV
 * @param pNumChannels SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                           CUtexref hTexRef);
/**
 * @param pfm SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
/**
 * @param pbias SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef);
/**
 * @param pminMipmapLevelClamp SEND_RECV
 * @param pmaxMipmapLevelClamp SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                     float *pmaxMipmapLevelClamp,
                                     CUtexref hTexRef);
/**
 * @param pmaxAniso SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef);
/**
 * @param pBorderColor SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef);
/**
 * @param pFlags SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef);
/**
 * @param pTexRef SEND_RECV
 */
CUresult cuTexRefCreate(CUtexref *pTexRef);
/**
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefDestroy(CUtexref hTexRef);
/**
 * @param hSurfRef SEND_ONLY
 * @param hArray SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                           unsigned int Flags);
/**
 * @param phArray SEND_RECV
 * @param hSurfRef SEND_ONLY
 */
CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef);
/**
 * @param pTexObject SEND_RECV
 * @param pResDesc SEND_ONLY DEREF
 * @param pTexDesc SEND_ONLY NULLABLE
 * @param pResViewDesc SEND_ONLY NULLABLE
 */
CUresult cuTexObjectCreate(CUtexObject *pTexObject,
                           const CUDA_RESOURCE_DESC *pResDesc,
                           const CUDA_TEXTURE_DESC *pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);
/**
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectDestroy(CUtexObject texObject);
/**
 * @param pResDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                    CUtexObject texObject);
/**
 * @param pTexDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                   CUtexObject texObject);
/**
 * @param pResViewDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                        CUtexObject texObject);
/**
 * @param pSurfObject SEND_RECV
 * @param pResDesc SEND_ONLY DEREF
 */
CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject,
                            const CUDA_RESOURCE_DESC *pResDesc);
/**
 * @param surfObject SEND_ONLY
 */
CUresult cuSurfObjectDestroy(CUsurfObject surfObject);
/**
 * @param pResDesc SEND_RECV
 * @param surfObject SEND_ONLY
 */
CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                     CUsurfObject surfObject);
/**
 * @disabled
 * @param tensorMap SEND_RECV
 * @param tensorDataType SEND_ONLY
 * @param tensorRank SEND_ONLY
 * @param globalAddress SEND_ONLY
 * @param globalDim SEND_RECV
 * @param globalStrides SEND_RECV
 * @param boxDim SEND_RECV
 * @param elementStrides SEND_RECV
 * @param interleave SEND_ONLY
 * @param swizzle SEND_ONLY
 * @param l2Promotion SEND_ONLY
 * @param oobFill SEND_ONLY
 * @guard CUDA_VERSION >= 12000
 */
CUresult cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);
/**
 * @param tensorMap SEND_RECV
 * @param tensorDataType SEND_ONLY
 * @param tensorRank SEND_ONLY
 * @param globalAddress SEND_RECV
 * @param globalDim SEND_RECV
 * @param globalStrides SEND_RECV
 * @param pixelBoxLowerCorner SEND_RECV
 * @param pixelBoxUpperCorner SEND_RECV
 * @param channelsPerPixel SEND_ONLY
 * @param pixelsPerColumn SEND_ONLY
 * @param elementStrides SEND_RECV
 * @param interleave SEND_ONLY
 * @param swizzle SEND_ONLY
 * @param l2Promotion SEND_ONLY
 * @param oobFill SEND_ONLY
 */
CUresult cuTensorMapEncodeIm2col(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
/**
 * @param tensorMap SEND_RECV
 * @param globalAddress SEND_RECV
 */
CUresult cuTensorMapReplaceAddress(CUtensorMap *tensorMap, void *globalAddress);
/**
 * @disabled client - manual client handles cross-route peer devices
 * @param canAccessPeer SEND_RECV
 * @param dev SEND_ONLY
 * @param peerDev SEND_ONLY
 */
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                               CUdevice peerDev);
/**
 * @disabled client - manual client validates the peer context route
 * @param peerContext SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
/**
 * @disabled client - manual client validates the peer context route
 * @param peerContext SEND_ONLY
 */
CUresult cuCtxDisablePeerAccess(CUcontext peerContext);
/**
 * @disabled client - manual client translates both devices to one backend
 * @param value SEND_RECV
 * @param attrib SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param dstDevice SEND_ONLY
 */
CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                 CUdevice srcDevice, CUdevice dstDevice);
/**
 * @param resource SEND_ONLY
 */
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);
/**
 * @param pArray SEND_RECV
 * @param resource SEND_ONLY
 * @param arrayIndex SEND_ONLY
 * @param mipLevel SEND_ONLY
 */
CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray,
                                             CUgraphicsResource resource,
                                             unsigned int arrayIndex,
                                             unsigned int mipLevel);
/**
 * @param pMipmappedArray SEND_RECV
 * @param resource SEND_ONLY
 */
CUresult
cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray,
                                          CUgraphicsResource resource);
/**
 * @param pDevPtr SEND_RECV
 * @param pSize SEND_RECV
 * @param resource SEND_ONLY
 */
CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr,
                                               size_t *pSize,
                                               CUgraphicsResource resource);
/**
 * @param resource SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                          unsigned int flags);
/**
 * @param count SEND_ONLY
 * @param resources SEND_ONLY LENGTH:count
 * @param hStream SEND_ONLY
 */
CUresult cuGraphicsMapResources(unsigned int count,
                                CUgraphicsResource *resources,
                                CUstream hStream);
/**
 * @param count SEND_ONLY
 * @param resources SEND_ONLY LENGTH:count
 * @param hStream SEND_ONLY
 */
CUresult cuGraphicsUnmapResources(unsigned int count,
                                  CUgraphicsResource *resources,
                                  CUstream hStream);
/**
 * @guard CUDA_VERSION >= 12090
 * @disabled - manual persistent callback forwarding and handle routing
 */
CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData,
                                CUlogsCallbackHandle *callback_out);
/**
 * @guard CUDA_VERSION >= 12090
 * @disabled - manual callback handle routing and lifecycle management
 */
CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback);
/**
 * @guard CUDA_VERSION >= 12090
 * @param iterator_out RECV_ONLY NULLABLE
 * @param flags SEND_ONLY
 */
CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags);
/**
 * @guard CUDA_VERSION >= 12090
 * @param iterator SEND_RECV NULLABLE
 * @param pathToFile SEND_ONLY NULL_TERMINATED
 * @param flags SEND_ONLY
 */
CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile,
                          unsigned int flags) {
  if (pathToFile == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
/**
 * @guard CUDA_VERSION >= 12090
 * @param iterator SEND_RECV NULLABLE
 * @param size SEND_RECV
 * @param buffer RECV_ONLY NULLABLE LENGTH:size
 * @param flags SEND_ONLY
 */
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size,
                            unsigned int flags) {
  if (size == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  size_t buffer_capacity = *size;
  CUresult return_value = LUPINE_GENERATED_CALL();
  if (return_value == CUDA_SUCCESS && buffer != nullptr &&
      buffer_capacity > *size)
    buffer[*size] = '\0';
  return return_value;
}
/**
 * @guard CUDA_VERSION >= 12010
 * @param attrib SEND_ONLY
 * @param size SEND_RECV
 * @param value RECV_ONLY LENGTH:size
 */
CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void *value,
                                      size_t *size);
/**
 * @guard CUDA_VERSION >= 12010
 * @param attrib SEND_ONLY
 * @param size SEND_RECV
 * @param value SEND_ONLY LENGTH:size
 */
CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void *value,
                                      size_t *size);

// Registry-only operations without API declarations above. The code generator
// reads these annotations directly; the C++ parser intentionally ignores them.
#if 0
/** @disabled */
void cuGetExportTableMetadata();
/** @disabled */
void cuPrivateGetModuleNode();
/** @disabled */
void lupineFunctionParamLayoutSnapshot();
/** @disabled */
void lupineFunctionAttributeSnapshot();
/** @disabled */
void lupineLibrarySnapshot();
/** @disabled */
void lupineLibraryAttributeSnapshot();
/** @disabled */
void cuGraphConditionalHandleCreate();
/** @disabled handle_cuGraphAddNode */
void cuGraphAddNode_v2();
/** @disabled */
void lupineEventQueryBatch();
/** @disabled */
void cuStreamBeginCaptureToGraph();
/** @disabled handle_cuStreamUpdateCaptureDependencies */
void cuStreamUpdateCaptureDependencies_v2();
/** @disabled handle_cuStreamGetCaptureInfo */
void cuStreamGetCaptureInfo_v3();
/** @disabled */
void lupineManagedHostFlush();
/** @disabled */
void lupineDeviceSnapshot();
#endif
