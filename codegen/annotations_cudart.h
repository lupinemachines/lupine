#include <cuda_runtime_api.h>

// CUDA runtime API. The client forwards each call over the CUDA driver shim's
// connections, so the same server child runs both APIs; the server handlers
// resolve the machine's libcudart by name at runtime. A device ordinal routes
// with `@routingkey DEVICE <param>` through the driver shim's virtual device
// table, and a stream or event routes to the connection that created it.
//
// This file is the list of what the shim supports. Anything cuda_runtime_api.h
// declares that is absent below, or declared below without saying what to do
// with its parameters, gets a stub returning cudaErrorNotSupported, so a
// consumer linking the whole library still loads.
//
// `@disabled client forwards` keeps the generated request builder for a call
// whose entry point the manual client writes itself.
//
// Every result comes back as a return value, so the manual client keeps the
// sticky error: calls it answers itself set it, and cudaGetLastError never
// has to ask the server.

/**
 * @param desc RECV_ONLY
 * @param extent RECV_ONLY
 * @param flags RECV_ONLY
 * @param array SEND_ONLY
 */
cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc,
                             struct cudaExtent *extent, unsigned int *flags,
                             cudaArray_t array);
/**
 * @routingkey DEVICE device
 * @param memoryRequirements RECV_ONLY
 * @param array SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaArrayGetMemoryRequirements(
    struct cudaArrayMemoryRequirements *memoryRequirements, cudaArray_t array,
    int device);
/**
 * @param pPlaneArray RECV_ONLY
 * @param hArray SEND_ONLY
 * @param planeIdx SEND_ONLY
 */
cudaError_t cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray,
                              unsigned int planeIdx);
/**
 * @param sparseProperties RECV_ONLY
 * @param array SEND_ONLY
 */
cudaError_t
cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties,
                             cudaArray_t array);
/**
 * @param device RECV_ONLY
 * @param prop SEND_ONLY DEREF
 */
cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
/**
 * @disabled - the result is a struct, not an error code
 */
struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
/**
 * @param pSurfObject RECV_ONLY
 * @param pResDesc SEND_ONLY DEREF
 */
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                    const struct cudaResourceDesc *pResDesc);
/**
 * @param pTexObject RECV_ONLY
 * @param pResDesc SEND_ONLY DEREF
 * @param pTexDesc SEND_ONLY NULLABLE
 * @param pResViewDesc SEND_ONLY NULLABLE
 */
cudaError_t
cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                        const struct cudaResourceDesc *pResDesc,
                        const struct cudaTextureDesc *pTexDesc,
                        const struct cudaResourceViewDesc *pResViewDesc);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param pTexObject RECV_ONLY
 * @param pResDesc SEND_ONLY DEREF
 * @param pTexDesc SEND_ONLY NULLABLE
 * @param pResViewDesc SEND_ONLY NULLABLE
 */
cudaError_t
cudaCreateTextureObject_v2(cudaTextureObject_t *pTexObject,
                           const struct cudaResourceDesc *pResDesc,
                           const struct cudaTextureDesc_v2 *pTexDesc,
                           const struct cudaResourceViewDesc *pResViewDesc);
#endif
cudaError_t cudaCtxResetPersistingL2Cache(void);
/**
 * @param extMem SEND_ONLY
 */
cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem);
/**
 * @param extSem SEND_ONLY
 */
cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);
/**
 * @param surfObject SEND_ONLY
 */
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
/**
 * @param texObject SEND_ONLY
 */
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t *phDesc,
                                        cudaDevResource *resources,
                                        unsigned int nbResources);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaDevSmResourceSplit(cudaDevResource *result,
                                   unsigned int nbGroups,
                                   const cudaDevResource *input,
                                   cudaDevResource *remainder,
                                   unsigned int flags,
                                   cudaDevSmResourceGroupParams *groupParams);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource *result,
                                          unsigned int *nbGroups,
                                          const cudaDevResource *input,
                                          cudaDevResource *remaining,
                                          unsigned int flags,
                                          unsigned int minCount);
#endif
/**
 * @param canAccessPeer RECV_ONLY
 * @param device SEND_ONLY
 * @param peerDevice SEND_ONLY
 */
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                    int peerDevice);
/**
 * @param peerDevice SEND_ONLY
 */
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
/**
 * @param peerDevice SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
/**
 * @param target SEND_ONLY
 * @param scope SEND_ONLY
 */
cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(
    enum cudaFlushGPUDirectRDMAWritesTarget target,
    enum cudaFlushGPUDirectRDMAWritesScope scope);
/**
 * @routingkey DEVICE device
 * @param value RECV_ONLY
 * @param attr SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr,
                                   int device);
/**
 * @param device RECV_ONLY
 * @param pciBusId SEND_ONLY NULL_TERMINATED
 */
cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
/**
 * @param pCacheConfig RECV_ONLY
 */
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
/**
 * @routingkey DEVICE device
 * @param memPool RECV_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @routingkey DEVICE device
 * @param device SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource *resource,
                                     enum cudaDevResourceType type);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @routingkey DEVICE device
 * @param ctx RECV_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetExecutionCtx(cudaExecutionContext_t *ctx, int device);
#endif
/**
 * @disabled - the value width depends on the attribute
 */
cudaError_t cudaDeviceGetGraphMemAttribute(int device,
                                           enum cudaGraphMemAttributeType attr,
                                           void *value);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @routingkey DEVICE device
 * @param capabilities RECV_ONLY LENGTH:count
 * @param operations SEND_ONLY LENGTH:count
 * @param count SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t
cudaDeviceGetHostAtomicCapabilities(unsigned int *capabilities,
                                    const enum cudaAtomicOperation *operations,
                                    unsigned int count, int device);
#endif
/**
 * @param pValue RECV_ONLY
 * @param limit SEND_ONLY
 */
cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
/**
 * @routingkey DEVICE device
 * @param memPool RECV_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param capabilities RECV_ONLY LENGTH:count
 * @param operations SEND_ONLY LENGTH:count
 * @param count SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param dstDevice SEND_ONLY
 */
cudaError_t cudaDeviceGetP2PAtomicCapabilities(
    unsigned int *capabilities, const enum cudaAtomicOperation *operations,
    unsigned int count, int srcDevice, int dstDevice);
#endif
/**
 * @param value RECV_ONLY
 * @param attr SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param dstDevice SEND_ONLY
 */
cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr,
                                      int srcDevice, int dstDevice);
/**
 * @routingkey DEVICE device
 * @param pciBusId RECV_ONLY LENGTH:len
 * @param len SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param pConfig RECV_ONLY
 */
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
#endif
/**
 * @param leastPriority RECV_ONLY
 * @param greatestPriority RECV_ONLY
 */
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority);
/**
 * @routingkey DEVICE device
 * @param maxWidthInElements RECV_ONLY
 * @param fmtDesc SEND_ONLY DEREF
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(
    size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc,
    int device);
/**
 * @routingkey DEVICE device
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGraphMemTrim(int device);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t
cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc,
                                    void *userData,
                                    cudaAsyncCallbackHandle_t *callback);
#endif
/**
 * @disabled client forwards - manual client drops its per-thread state first
 */
cudaError_t cudaDeviceReset(void);
/**
 * @param cacheConfig SEND_ONLY
 */
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
/**
 * @disabled - the value width depends on the attribute
 */
cudaError_t cudaDeviceSetGraphMemAttribute(int device,
                                           enum cudaGraphMemAttributeType attr,
                                           void *value);
/**
 * @param limit SEND_ONLY
 * @param value SEND_ONLY
 */
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
/**
 * @routingkey DEVICE device
 * @param device SEND_ONLY
 * @param memPool SEND_ONLY
 */
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param config SEND_ONLY
 */
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
#endif
cudaError_t cudaDeviceSynchronize(void);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t
cudaDeviceUnregisterAsyncNotification(int device,
                                      cudaAsyncCallbackHandle_t callback);
#endif
/**
 * @param driverVersion RECV_ONLY
 */
cudaError_t cudaDriverGetVersion(int *driverVersion);
/**
 * @recordowner EVENT event
 * @param event RECV_ONLY
 */
cudaError_t cudaEventCreate(cudaEvent_t *event);
/**
 * @recordowner EVENT event
 * @param event RECV_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
/**
 * @routingkey EVENT event
 * @param event SEND_ONLY
 */
cudaError_t cudaEventDestroy(cudaEvent_t event);
/**
 * @routingkey EVENT start
 * @param ms RECV_ONLY
 * @param start SEND_ONLY
 * @param end SEND_ONLY
 */
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
/**
 * @routingkey EVENT event
 * @param event SEND_ONLY
 */
cudaError_t cudaEventQuery(cudaEvent_t event);
/**
 * @routingkey EVENT event
 * @param event SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
/**
 * @routingkey EVENT event
 * @param event SEND_ONLY
 * @param stream SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                     unsigned int flags);
/**
 * @routingkey EVENT event
 * @param event SEND_ONLY
 */
cudaError_t cudaEventSynchronize(cudaEvent_t event);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param ctx SEND_ONLY
 */
cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param ctx SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx,
                                           cudaDevResource *resource,
                                           enum cudaDevResourceType type);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param device RECV_ONLY
 * @param ctx SEND_ONLY
 */
cudaError_t cudaExecutionCtxGetDevice(int *device, cudaExecutionContext_t ctx);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param ctx SEND_ONLY
 * @param ctxId RECV_ONLY
 */
cudaError_t cudaExecutionCtxGetId(cudaExecutionContext_t ctx,
                                  unsigned long long *ctxId);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param ctx SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx,
                                        cudaEvent_t event);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @recordowner STREAM phStream
 * @param phStream RECV_ONLY
 * @param ctx SEND_ONLY
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
cudaError_t cudaExecutionCtxStreamCreate(cudaStream_t *phStream,
                                         cudaExecutionContext_t ctx,
                                         unsigned int flags, int priority);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param ctx SEND_ONLY
 */
cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param ctx SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx,
                                      cudaEvent_t event);
#endif
/**
 * @param devPtr RECV_ONLY
 * @param extMem SEND_ONLY
 * @param bufferDesc SEND_ONLY DEREF
 */
cudaError_t cudaExternalMemoryGetMappedBuffer(
    void **devPtr, cudaExternalMemory_t extMem,
    const struct cudaExternalMemoryBufferDesc *bufferDesc);
/**
 * @param mipmap RECV_ONLY
 * @param extMem SEND_ONLY
 * @param mipmapDesc SEND_ONLY DEREF
 */
cudaError_t cudaExternalMemoryGetMappedMipmappedArray(
    cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem,
    const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc);
/**
 * @param devPtr SEND_ONLY
 */
cudaError_t cudaFree(void *devPtr);
/**
 * @param array SEND_ONLY
 */
cudaError_t cudaFreeArray(cudaArray_t array);
/**
 * @param devPtr SEND_ONLY
 * @param hStream SEND_ONLY
 */
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream);
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaFreeHost(void *ptr);
/**
 * @param mipmappedArray SEND_ONLY
 */
cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
/**
 * @param attr RECV_ONLY
 * @param func SEND_ONLY
 */
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                  const void *func);
#if CUDART_VERSION >= 12000
/**
 * @disabled - the runtime returns a static string; manual client caches a copy
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t cudaFuncGetName(const char **name, const void *func);
#endif
#if CUDART_VERSION >= 13000
/**
 * @param func SEND_ONLY
 * @param paramCount RECV_ONLY
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaFuncGetParamCount(const void *func, size_t *paramCount);
#endif
#if CUDART_VERSION >= 12000
/**
 * @param func SEND_ONLY
 * @param paramIndex SEND_ONLY
 * @param paramOffset RECV_ONLY
 * @param paramSize RECV_ONLY
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t cudaFuncGetParamInfo(const void *func, size_t paramIndex,
                                 size_t *paramOffset, size_t *paramSize);
#endif
/**
 * @param func SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY
 */
cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr,
                                 int value);
/**
 * @param func SEND_ONLY
 * @param cacheConfig SEND_ONLY
 */
cudaError_t cudaFuncSetCacheConfig(const void *func,
                                   enum cudaFuncCache cacheConfig);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param func SEND_ONLY
 * @param config SEND_ONLY
 */
cudaError_t cudaFuncSetSharedMemConfig(const void *func,
                                       enum cudaSharedMemConfig config);
#endif
/**
 * @param desc RECV_ONLY
 * @param array SEND_ONLY
 */
cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
                               cudaArray_const_t array);
/**
 * @disabled client - the virtual device table is client state
 * @param device RECV_ONLY
 */
cudaError_t cudaGetDevice(int *device);
/**
 * @disabled client - the virtual device table is client state
 * @param count RECV_ONLY
 */
cudaError_t cudaGetDeviceCount(int *count);
/**
 * @param flags RECV_ONLY
 */
cudaError_t cudaGetDeviceFlags(unsigned int *flags);
/**
 * @routingkey DEVICE device
 * @param prop RECV_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
#if CUDART_VERSION < 12000
/**
 * the result is a function pointer into the server's driver
 * @guard CUDART_VERSION < 12000
 */
cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                                    unsigned long long flags);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * the result is a function pointer into the server's driver
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 */
cudaError_t
cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                        unsigned long long flags,
                        enum cudaDriverEntryPointQueryResult *driverStatus);
#endif
#if CUDART_VERSION >= 13000
/**
 * the result is a function pointer into the server's driver
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t
cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                        unsigned long long flags,
                        enum cudaDriverEntryPointQueryResult *driverStatus);
#endif
#if CUDART_VERSION >= 13000
/**
 * the result is a function pointer into the server's driver
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaGetDriverEntryPointByVersion(
    const char *symbol, void **funcPtr, unsigned int cudaVersion,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus);
#endif
/**
 * @disabled - the runtime returns a static string of unknown length, which the
 * generated marshaller cannot receive; manual client caches, manual server
 * forwards length-prefixed bytes
 */
const char *cudaGetErrorName(cudaError_t error);
/**
 * @disabled - see cudaGetErrorName
 */
const char *cudaGetErrorString(cudaError_t error);
/**
 * the result is a pointer into the server's driver
 */
cudaError_t cudaGetExportTable(const void **ppExportTable,
                               const cudaUUID_t *pExportTableId);
#if CUDART_VERSION < 12000
/**
 * @param functionPtr RECV_ONLY
 * @param symbolPtr SEND_ONLY
 * @guard CUDART_VERSION < 12000
 */
cudaError_t CUDARTAPI_CDECL cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                                const void *symbolPtr);
#endif
#if CUDART_VERSION >= 13000
/**
 * @param functionPtr RECV_ONLY
 * @param symbolPtr SEND_ONLY
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                const void *symbolPtr);
#endif
#if CUDART_VERSION >= 12000
/**
 * @param kernelPtr RECV_ONLY
 * @param entryFuncAddr SEND_ONLY
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t cudaGetKernel(cudaKernel_t *kernelPtr, const void *entryFuncAddr);
#endif
/**
 * @disabled client - the sticky error is kept on the client
 */
cudaError_t cudaGetLastError(void);
/**
 * @param levelArray RECV_ONLY
 * @param mipmappedArray SEND_ONLY
 * @param level SEND_ONLY
 */
cudaError_t
cudaGetMipmappedArrayLevel(cudaArray_t *levelArray,
                           cudaMipmappedArray_const_t mipmappedArray,
                           unsigned int level);
/**
 * @param pResDesc RECV_ONLY
 * @param surfObject SEND_ONLY
 */
cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                             cudaSurfaceObject_t surfObject);
/**
 * @param devPtr RECV_ONLY
 * @param symbol SEND_ONLY
 */
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
/**
 * @param size RECV_ONLY
 * @param symbol SEND_ONLY
 */
cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
/**
 * @param pResDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                             cudaTextureObject_t texObject);
/**
 * @param pResViewDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
cudaError_t
cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc,
                                     cudaTextureObject_t texObject);
/**
 * @param pTexDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc,
                                            cudaTextureObject_t texObject);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param pTexDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
cudaError_t
cudaGetTextureObjectTextureDesc_v2(struct cudaTextureDesc_v2 *pTexDesc,
                                   cudaTextureObject_t texObject);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @routingkey DEVICE device
 * @param phCtx RECV_ONLY
 * @param desc SEND_ONLY
 * @param device SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t *phCtx,
                               cudaDevResourceDesc_t desc, int device,
                               unsigned int flags);
#endif
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                     unsigned int flags);
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaHostUnregister(void *ptr);
cudaError_t cudaImportExternalMemory(
    cudaExternalMemory_t *extMem_out,
    const struct cudaExternalMemoryHandleDesc *memHandleDesc);
cudaError_t cudaImportExternalSemaphore(
    cudaExternalSemaphore_t *extSem_out,
    const struct cudaExternalSemaphoreHandleDesc *semHandleDesc);
#if CUDART_VERSION >= 12000
/**
 * @disabled client forwards - manual client maps the virtual ordinal
 * @guard CUDART_VERSION >= 12000
 * @param device SEND_ONLY
 * @param deviceFlags SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaInitDevice(int device, unsigned int deviceFlags,
                           unsigned int flags);
#endif
/**
 * @param devPtr SEND_ONLY
 */
cudaError_t cudaIpcCloseMemHandle(void *devPtr);
/**
 * @routingkey EVENT event
 * @param handle RECV_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
                                  cudaEvent_t event);
/**
 * @param handle RECV_ONLY
 * @param devPtr SEND_ONLY
 */
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
/**
 * @recordowner EVENT event
 * @param event RECV_ONLY
 * @param handle SEND_ONLY
 */
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
                                   cudaIpcEventHandle_t handle);
/**
 * @param devPtr RECV_ONLY
 * @param handle SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
                                 unsigned int flags);
#if CUDART_VERSION >= 13000
/**
 * @routingkey DEVICE device
 * @param kernel SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY
 * @param device SEND_ONLY
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel,
                                            enum cudaFuncAttribute attr,
                                            int value, int device);
#endif
/**
 * @disabled - manual client packs the arguments by the server's parameter
 * layout
 */
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream);
/**
 * the callback is a client function
 */
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                               void *userData);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn,
                                  void *userData, unsigned int syncMode);
#endif
/**
 * @disabled - manual client packs the arguments by the server's parameter
 * layout
 */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream);
/**
 * @disabled - manual client packs the arguments by the server's parameter
 * layout
 */
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                const void *func, void **args);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t *kernels,
                                        unsigned int numKernels,
                                        cudaLibrary_t lib);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param dptr RECV_ONLY
 * @param bytes RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
cudaError_t cudaLibraryGetGlobal(void **dptr, size_t *bytes,
                                 cudaLibrary_t library, const char *name);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLibraryGetKernel(cudaKernel_t *pKernel, cudaLibrary_t library,
                                 const char *name);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param count RECV_ONLY
 * @param lib SEND_ONLY
 */
cudaError_t cudaLibraryGetKernelCount(unsigned int *count, cudaLibrary_t lib);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param dptr RECV_ONLY
 * @param bytes RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
cudaError_t cudaLibraryGetManaged(void **dptr, size_t *bytes,
                                  cudaLibrary_t library, const char *name);
#endif
#if CUDART_VERSION >= 13000
/**
 * the result is a host function pointer
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLibraryGetUnifiedFunction(void **fptr, cudaLibrary_t library,
                                          const char *symbol);
#endif
#if CUDART_VERSION >= 13000
/**
 * the image size is not part of the call
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLibraryLoadData(cudaLibrary_t *library, const void *code,
                                enum cudaJitOption *jitOptions,
                                void **jitOptionsValues,
                                unsigned int numJitOptions,
                                enum cudaLibraryOption *libraryOptions,
                                void **libraryOptionValues,
                                unsigned int numLibraryOptions);
#endif
#if CUDART_VERSION >= 13000
/**
 * the file is a client path
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLibraryLoadFromFile(
    cudaLibrary_t *library, const char *fileName,
    enum cudaJitOption *jitOptions, void **jitOptionsValues,
    unsigned int numJitOptions, enum cudaLibraryOption *libraryOptions,
    void **libraryOptionValues, unsigned int numLibraryOptions);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param library SEND_ONLY
 */
cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param iterator_out RECV_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaLogsCurrent(cudaLogIterator *iterator_out, unsigned int flags);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param iterator SEND_RECV
 * @param pathToFile SEND_ONLY NULL_TERMINATED
 * @param flags SEND_ONLY
 */
cudaError_t cudaLogsDumpToFile(cudaLogIterator *iterator,
                               const char *pathToFile, unsigned int flags);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param iterator SEND_RECV
 * @param size SEND_RECV
 * @param buffer RECV_ONLY NULLABLE LENGTH:size
 * @param flags SEND_ONLY
 */
cudaError_t cudaLogsDumpToMemory(cudaLogIterator *iterator, char *buffer,
                                 size_t *size, unsigned int flags);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc,
                                     void *userData,
                                     cudaLogsCallbackHandle *callback_out);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param callback SEND_ONLY
 */
cudaError_t cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback);
#endif
/**
 * @param devPtr RECV_ONLY
 * @param size SEND_ONLY
 */
cudaError_t cudaMalloc(void **devPtr, size_t size);
/**
 * @param pitchedDevPtr RECV_ONLY
 * @param extent SEND_ONLY
 */
cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
                         struct cudaExtent extent);
/**
 * @param array RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param extent SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                              const struct cudaChannelFormatDesc *desc,
                              struct cudaExtent extent, unsigned int flags);
/**
 * @param array RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMallocArray(cudaArray_t *array,
                            const struct cudaChannelFormatDesc *desc,
                            size_t width, size_t height, unsigned int flags);
/**
 * @param devPtr RECV_ONLY
 * @param size SEND_ONLY
 * @param hStream SEND_ONLY
 */
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);
/**
 * @param ptr RECV_ONLY
 * @param size SEND_ONLY
 * @param memPool SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size,
                                    cudaMemPool_t memPool, cudaStream_t stream);
/**
 * host memory the server allocated is not addressable from the client
 */
cudaError_t cudaMallocHost(void **ptr, size_t size);
/**
 * @param devPtr RECV_ONLY
 * @param size SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
/**
 * @param mipmappedArray RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param extent SEND_ONLY
 * @param numLevels SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                                     const struct cudaChannelFormatDesc *desc,
                                     struct cudaExtent extent,
                                     unsigned int numLevels,
                                     unsigned int flags);
/**
 * @param devPtr RECV_ONLY
 * @param pitch RECV_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
                            size_t height);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                          enum cudaMemoryAdvise advice, int device);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param location SEND_ONLY
 */
cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param location SEND_ONLY
 */
cudaError_t cudaMemAdvise_v2(const void *devPtr, size_t count,
                             enum cudaMemoryAdvise advice,
                             struct cudaMemLocation location);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(
    void **dptrs, size_t *sizes, size_t count,
    struct cudaMemLocation *prefetchLocs, size_t *prefetchLocIdxs,
    size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemDiscardBatchAsync(void **dptrs, size_t *sizes, size_t count,
                                     unsigned long long flags,
                                     cudaStream_t stream);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param memPool RECV_ONLY
 * @param location SEND_ONLY DEREF
 * @param type SEND_ONLY
 */
cudaError_t cudaMemGetDefaultMemPool(cudaMemPool_t *memPool,
                                     struct cudaMemLocation *location,
                                     enum cudaMemAllocationType type);
#endif
/**
 * @param free RECV_ONLY
 * @param total RECV_ONLY
 */
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param memPool RECV_ONLY
 * @param location SEND_ONLY DEREF
 * @param type SEND_ONLY
 */
cudaError_t cudaMemGetMemPool(cudaMemPool_t *memPool,
                              struct cudaMemLocation *location,
                              enum cudaMemAllocationType type);
#endif
/**
 * @param memPool RECV_ONLY
 * @param poolProps SEND_ONLY DEREF
 */
cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool,
                              const struct cudaMemPoolProps *poolProps);
/**
 * @param memPool SEND_ONLY
 */
cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool);
/**
 * @param exportData RECV_ONLY
 * @param ptr SEND_ONLY
 */
cudaError_t
cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData,
                         void *ptr);
/**
 * @param flags RECV_ONLY
 * @param memPool SEND_ONLY
 * @param location SEND_ONLY DEREF
 */
cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags,
                                 cudaMemPool_t memPool,
                                 struct cudaMemLocation *location);
/**
 * @disabled - the value width depends on the attribute
 */
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                                    enum cudaMemPoolAttr attr, void *value);
/**
 * @param ptr RECV_ONLY
 * @param memPool SEND_ONLY
 * @param exportData SEND_ONLY DEREF
 */
cudaError_t
cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool,
                         struct cudaMemPoolPtrExportData *exportData);
/**
 * @param memPool SEND_ONLY
 * @param descList SEND_ONLY LENGTH:count
 * @param count SEND_ONLY
 */
cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool,
                                 const struct cudaMemAccessDesc *descList,
                                 size_t count);
/**
 * @disabled - the value width depends on the attribute
 */
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                    enum cudaMemPoolAttr attr, void *value);
/**
 * @param memPool SEND_ONLY
 * @param minBytesToKeep SEND_ONLY
 */
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param dstDevice SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                 int dstDevice, cudaStream_t stream);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param location SEND_ONLY
 * @param flags SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags, cudaStream_t stream);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param location SEND_ONLY
 * @param flags SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemPrefetchAsync_v2(const void *devPtr, size_t count,
                                    struct cudaMemLocation location,
                                    unsigned int flags, cudaStream_t stream);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemPrefetchBatchAsync(void **dptrs, size_t *sizes, size_t count,
                                      struct cudaMemLocation *prefetchLocs,
                                      size_t *prefetchLocIdxs,
                                      size_t numPrefetchLocs,
                                      unsigned long long flags,
                                      cudaStream_t stream);
#endif
/**
 * @param data RECV_ONLY LENGTH:dataSize
 * @param dataSize SEND_ONLY
 * @param attribute SEND_ONLY
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize,
                                     enum cudaMemRangeAttribute attribute,
                                     const void *devPtr, size_t count);
/**
 * each attribute writes a caller buffer of its own width
 */
cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes,
                                      enum cudaMemRangeAttribute *attributes,
                                      size_t numAttributes, const void *devPtr,
                                      size_t count);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param location SEND_ONLY DEREF
 * @param type SEND_ONLY
 * @param memPool SEND_ONLY
 */
cudaError_t cudaMemSetMemPool(struct cudaMemLocation *location,
                              enum cudaMemAllocationType type,
                              cudaMemPool_t memPool);
#endif
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         enum cudaMemcpyKind kind);
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * the copy parameters carry host pointers
 */
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
/**
 * the copy parameters carry host pointers
 */
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
                              cudaStream_t stream);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemcpy3DBatchAsync(size_t numOps,
                                   struct cudaMemcpy3DBatchOp *opList,
                                   unsigned long long flags,
                                   cudaStream_t stream);
#endif
/**
 * the copy parameters carry host pointers
 */
cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
/**
 * the copy parameters carry host pointers
 */
cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p,
                                  cudaStream_t stream);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemcpy3DWithAttributesAsync(struct cudaMemcpy3DBatchOp *op,
                                            unsigned long long flags,
                                            cudaStream_t stream);
#endif
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemcpyBatchAsync(void *const *dsts, const void *const *srcs,
                                 const size_t *sizes, size_t count,
                                 struct cudaMemcpyAttributes *attrs,
                                 size_t *attrsIdxs, size_t numAttrs,
                                 cudaStream_t stream);
#endif
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
                                 size_t offset, enum cudaMemcpyKind kind);
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                      size_t count, size_t offset,
                                      enum cudaMemcpyKind kind,
                                      cudaStream_t stream);
/**
 * @disabled client forwards - manual client maps both virtual ordinals
 * @param dst SEND_ONLY
 * @param dstDevice SEND_ONLY
 * @param src SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                           int srcDevice, size_t count);
/**
 * @disabled client forwards - manual client maps both virtual ordinals
 * @param dst SEND_ONLY
 * @param dstDevice SEND_ONLY
 * @param src SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream);
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                               size_t count, size_t offset,
                               enum cudaMemcpyKind kind);
/**
 * @disabled - manual client and server carry the host side of the copy
 */
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
                                    size_t count, size_t offset,
                                    enum cudaMemcpyKind kind,
                                    cudaStream_t stream);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaMemcpyWithAttributesAsync(void *dst, const void *src,
                                          size_t size,
                                          struct cudaMemcpyAttributes *attr,
                                          cudaStream_t stream);
#endif
/**
 * @param devPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
/**
 * @param devPtr SEND_ONLY
 * @param pitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width,
                         size_t height);
/**
 * @param devPtr SEND_ONLY
 * @param pitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
                              size_t width, size_t height, cudaStream_t stream);
/**
 * @param pitchedDevPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param extent SEND_ONLY
 */
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value,
                         struct cudaExtent extent);
/**
 * @param pitchedDevPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param extent SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value,
                              struct cudaExtent extent, cudaStream_t stream);
/**
 * @param devPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream);
/**
 * @routingkey DEVICE device
 * @param memoryRequirements RECV_ONLY
 * @param mipmap SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaMipmappedArrayGetMemoryRequirements(
    struct cudaArrayMemoryRequirements *memoryRequirements,
    cudaMipmappedArray_t mipmap, int device);
/**
 * @param sparseProperties RECV_ONLY
 * @param mipmap SEND_ONLY
 */
cudaError_t cudaMipmappedArrayGetSparseProperties(
    struct cudaArraySparseProperties *sparseProperties,
    cudaMipmappedArray_t mipmap);
/**
 * @param dynamicSmemSize RECV_ONLY
 * @param func SEND_ONLY
 * @param numBlocks SEND_ONLY
 * @param blockSize SEND_ONLY
 */
cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                      const void *func,
                                                      int numBlocks,
                                                      int blockSize);
/**
 * @param numBlocks RECV_ONLY
 * @param func SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
/**
 * @param numBlocks RECV_ONLY
 * @param func SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags);
/**
 * @disabled - manual client sends the attributes behind the launch config
 */
cudaError_t
cudaOccupancyMaxActiveClusters(int *numClusters, const void *func,
                               const cudaLaunchConfig_t *launchConfig);
/**
 * @disabled - manual client sends the attributes behind the launch config
 */
cudaError_t
cudaOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func,
                                     const cudaLaunchConfig_t *launchConfig);
/**
 * @disabled client - the sticky error is kept on the client
 */
cudaError_t cudaPeekAtLastError(void);
/**
 * @param attributes RECV_ONLY
 * @param ptr SEND_ONLY
 */
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                     const void *ptr);
/**
 * @param runtimeVersion RECV_ONLY
 */
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
/**
 * @disabled client forwards - maps the virtual ordinal, then records it
 * @param device SEND_ONLY
 */
cudaError_t cudaSetDevice(int device);
/**
 * @param flags SEND_ONLY
 */
cudaError_t cudaSetDeviceFlags(unsigned int flags);
/**
 * @disabled client forwards - manual client maps every virtual ordinal
 * @param device_arr SEND_ONLY LENGTH:len
 * @param len SEND_ONLY
 */
cudaError_t cudaSetValidDevices(int *device_arr, int len);
/**
 * @param extSemArray SEND_ONLY LENGTH:numExtSems
 * @param paramsArray SEND_ONLY LENGTH:numExtSems
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaSignalExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSemArray,
    const struct cudaExternalSemaphoreSignalParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param extSemArray SEND_ONLY DEREF
 * @param paramsArray SEND_ONLY DEREF
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaSignalExternalSemaphoresAsync_ptsz(
    const cudaExternalSemaphore_t *extSemArray,
    const struct cudaExternalSemaphoreSignalParams_v1 *paramsArray,
    unsigned int numExtSems, cudaStream_t stream);
#endif
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param extSemArray SEND_ONLY LENGTH:numExtSems
 * @param paramsArray SEND_ONLY LENGTH:numExtSems
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaSignalExternalSemaphoresAsync_v2(
    const cudaExternalSemaphore_t *extSemArray,
    const struct cudaExternalSemaphoreSignalParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream);
#endif
/**
 * the callback is a client function
 */
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback, void *userData,
                                  unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param devPtr SEND_ONLY
 * @param length SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr,
                                     size_t length, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param mode SEND_ONLY
 */
cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                   enum cudaStreamCaptureMode mode);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param stream SEND_ONLY
 * @param graph SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dependencyData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param mode SEND_ONLY
 */
cudaError_t cudaStreamBeginCaptureToGraph(
    cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *dependencyData, size_t numDependencies,
    enum cudaStreamCaptureMode mode);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaStreamBeginRecaptureToGraph(
    cudaStream_t stream, enum cudaStreamCaptureMode mode, cudaGraph_t graph,
    struct cudaGraphRecaptureCallbackData *callbackData);
#endif
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);
/**
 * @recordowner STREAM pStream
 * @param pStream RECV_ONLY
 */
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
/**
 * @recordowner STREAM pStream
 * @param pStream RECV_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags);
/**
 * @recordowner STREAM pStream
 * @param pStream RECV_ONLY
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority);
/**
 * @param stream SEND_ONLY
 */
cudaError_t cudaStreamDestroy(cudaStream_t stream);
/**
 * @param stream SEND_ONLY
 * @param pGraph RECV_ONLY
 */
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
/**
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out RECV_ONLY
 */
cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr,
                                   cudaStreamAttrValue *value_out);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 */
cudaError_t
cudaStreamGetCaptureInfo(cudaStream_t stream,
                         enum cudaStreamCaptureStatus *pCaptureStatus,
                         unsigned long long *pId);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 */
cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out,
    const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param stream SEND_ONLY
 * @param captureStatus_out RECV_ONLY
 * @param id_out RECV_ONLY
 */
cudaError_t
cudaStreamGetCaptureInfo_ptsz(cudaStream_t stream,
                              enum cudaStreamCaptureStatus *captureStatus_out,
                              unsigned long long *id_out);
#endif
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 */
cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 */
cudaError_t cudaStreamGetCaptureInfo_v3(
    cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out,
    const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hStream SEND_ONLY
 * @param resource RECV_ONLY
 * @param type SEND_ONLY
 */
cudaError_t cudaStreamGetDevResource(cudaStream_t hStream,
                                     cudaDevResource *resource,
                                     enum cudaDevResourceType type);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hStream SEND_ONLY
 * @param device RECV_ONLY
 */
cudaError_t cudaStreamGetDevice(cudaStream_t hStream, int *device);
#endif
/**
 * @param hStream SEND_ONLY
 * @param flags RECV_ONLY
 */
cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param hStream SEND_ONLY
 * @param streamId RECV_ONLY
 */
cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long *streamId);
#endif
/**
 * @param hStream SEND_ONLY
 * @param priority RECV_ONLY
 */
cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
/**
 * @param stream SEND_ONLY
 * @param pCaptureStatus RECV_ONLY
 */
cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                  enum cudaStreamCaptureStatus *pCaptureStatus);
/**
 * @param stream SEND_ONLY
 */
cudaError_t cudaStreamQuery(cudaStream_t stream);
/**
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY DEREF
 */
cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr,
                                   const cudaStreamAttrValue *value);
/**
 * @param stream SEND_ONLY
 */
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param stream SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                                cudaGraphNode_t *dependencies,
                                                size_t numDependencies,
                                                unsigned int flags);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param stream SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dependencyData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t
cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                    cudaGraphNode_t *dependencies,
                                    const cudaGraphEdgeData *dependencyData,
                                    size_t numDependencies, unsigned int flags);
#endif
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param stream SEND_ONLY
 * @param dependencies RECV_ONLY
 * @param numDependencies SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamUpdateCaptureDependencies_ptsz(
    cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies,
    unsigned int flags);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param stream SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dependencyData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamUpdateCaptureDependencies_v2(
    cudaStream_t stream, cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags);
#endif
/**
 * @routingkey STREAM stream
 * @param stream SEND_ONLY
 * @param event SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags);
/**
 * @param mode SEND_RECV
 */
cudaError_t
cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode);
cudaError_t cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr,
                                 cudaHostFn_t destroy,
                                 unsigned int initialRefcount,
                                 unsigned int flags);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count);
/**
 * @param extSemArray SEND_ONLY LENGTH:numExtSems
 * @param paramsArray SEND_ONLY LENGTH:numExtSems
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSemArray,
    const struct cudaExternalSemaphoreWaitParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param extSemArray SEND_ONLY DEREF
 * @param paramsArray SEND_ONLY DEREF
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaWaitExternalSemaphoresAsync_ptsz(
    const cudaExternalSemaphore_t *extSemArray,
    const struct cudaExternalSemaphoreWaitParams_v1 *paramsArray,
    unsigned int numExtSems, cudaStream_t stream);
#endif
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param extSemArray SEND_ONLY LENGTH:numExtSems
 * @param paramsArray SEND_ONLY LENGTH:numExtSems
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaWaitExternalSemaphoresAsync_v2(
    const cudaExternalSemaphore_t *extSemArray,
    const struct cudaExternalSemaphoreWaitParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream);
#endif
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param childGraph SEND_ONLY
 */
cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode,
                                       cudaGraph_t graph,
                                       const cudaGraphNode_t *pDependencies,
                                       size_t numDependencies,
                                       cudaGraph_t childGraph);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param graph SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphAddDependencies(cudaGraph_t graph,
                                     const cudaGraphNode_t *from,
                                     const cudaGraphNode_t *to,
                                     size_t numDependencies);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param graph SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 * @param edgeData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphAddDependencies(cudaGraph_t graph,
                                     const cudaGraphNode_t *from,
                                     const cudaGraphNode_t *to,
                                     const cudaGraphEdgeData *edgeData,
                                     size_t numDependencies);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param graph SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 * @param edgeData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphAddDependencies_v2(cudaGraph_t graph,
                                        const cudaGraphNode_t *from,
                                        const cudaGraphNode_t *to,
                                        const cudaGraphEdgeData *edgeData,
                                        size_t numDependencies);
#endif
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode,
                                  cudaGraph_t graph,
                                  const cudaGraphNode_t *pDependencies,
                                  size_t numDependencies);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode,
                                        cudaGraph_t graph,
                                        const cudaGraphNode_t *pDependencies,
                                        size_t numDependencies,
                                        cudaEvent_t event);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode,
                                      cudaGraph_t graph,
                                      const cudaGraphNode_t *pDependencies,
                                      size_t numDependencies,
                                      cudaEvent_t event);
cudaError_t cudaGraphAddExternalSemaphoresSignalNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies, size_t numDependencies,
    const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
cudaError_t cudaGraphAddExternalSemaphoresWaitNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies, size_t numDependencies,
    const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const struct cudaHostNodeParams *pNodeParams);
cudaError_t
cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaKernelNodeParams *pNodeParams);
cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode,
                                     cudaGraph_t graph,
                                     const cudaGraphNode_t *pDependencies,
                                     size_t numDependencies,
                                     struct cudaMemAllocNodeParams *nodeParams);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param dptr SEND_ONLY
 */
cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode,
                                    cudaGraph_t graph,
                                    const cudaGraphNode_t *pDependencies,
                                    size_t numDependencies, void *dptr);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 * @param pMemsetParams SEND_ONLY DEREF
 */
cudaError_t
cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaMemsetParams *pMemsetParams);
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 */
cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                             const cudaGraphNode_t *pDependencies,
                             size_t numDependencies,
                             struct cudaGraphNodeParams *nodeParams);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                             const cudaGraphNode_t *pDependencies,
                             const cudaGraphEdgeData *dependencyData,
                             size_t numDependencies,
                             struct cudaGraphNodeParams *nodeParams);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 */
cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                const cudaGraphNode_t *pDependencies,
                                const cudaGraphEdgeData *dependencyData,
                                size_t numDependencies,
                                struct cudaGraphNodeParams *nodeParams);
#endif
/**
 * @param node SEND_ONLY
 * @param pGraph RECV_ONLY
 */
cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node,
                                            cudaGraph_t *pGraph);
/**
 * @param pGraphClone RECV_ONLY
 * @param originalGraph SEND_ONLY
 */
cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param pHandle_out RECV_ONLY
 * @param graph SEND_ONLY
 * @param defaultLaunchValue SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphConditionalHandleCreate(
    cudaGraphConditionalHandle *pHandle_out, cudaGraph_t graph,
    unsigned int defaultLaunchValue, unsigned int flags);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param pHandle_out RECV_ONLY
 * @param graph SEND_ONLY
 * @param ctx SEND_ONLY
 * @param defaultLaunchValue SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphConditionalHandleCreate_v2(
    cudaGraphConditionalHandle *pHandle_out, cudaGraph_t graph,
    cudaExecutionContext_t ctx, unsigned int defaultLaunchValue,
    unsigned int flags);
#endif
/**
 * @param pGraph RECV_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags);
/**
 * @param graph SEND_ONLY
 * @param path SEND_ONLY NULL_TERMINATED
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path,
                                   unsigned int flags);
/**
 * @param graph SEND_ONLY
 */
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
/**
 * @param node SEND_ONLY
 */
cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node);
/**
 * @param node SEND_ONLY
 * @param event_out RECV_ONLY
 */
cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node,
                                             cudaEvent_t *event_out);
/**
 * @param node SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node,
                                             cudaEvent_t event);
/**
 * @param node SEND_ONLY
 * @param event_out RECV_ONLY
 */
cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node,
                                           cudaEvent_t *event_out);
/**
 * @param node SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node,
                                           cudaEvent_t event);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param childGraph SEND_ONLY
 */
cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec,
                                                 cudaGraphNode_t node,
                                                 cudaGraph_t childGraph);
/**
 * @param graphExec SEND_ONLY
 */
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec,
                                                 cudaGraphNode_t hNode,
                                                 cudaEvent_t event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec,
                                               cudaGraphNode_t hNode,
                                               cudaEvent_t event);
cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode,
    const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode,
    const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param graphExec SEND_ONLY
 * @param flags RECV_ONLY
 */
cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec,
                                  unsigned long long *flags);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hGraphExec SEND_ONLY
 * @param graphID RECV_ONLY
 */
cudaError_t cudaGraphExecGetId(cudaGraphExec_t hGraphExec,
                               unsigned int *graphID);
#endif
cudaError_t
cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                               const struct cudaHostNodeParams *pNodeParams);
cudaError_t cudaGraphExecKernelNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
    const struct cudaKernelNodeParams *pNodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param pNodeParams SEND_ONLY DEREF
 */
cudaError_t
cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec,
                                 cudaGraphNode_t node,
                                 const struct cudaMemsetParams *pNodeParams);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec,
                                       cudaGraphNode_t node,
                                       struct cudaGraphNodeParams *nodeParams);
#endif
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param hErrorNode_out RECV_ONLY
 * @param updateResult_out RECV_ONLY
 */
cudaError_t
cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                    cudaGraphNode_t *hErrorNode_out,
                    enum cudaGraphExecUpdateResult *updateResult_out);
#endif
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param resultInfo SEND_RECV DEREF
 */
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                cudaGraphExecUpdateResultInfo *resultInfo);
#endif
cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(
    cudaGraphNode_t hNode,
    struct cudaExternalSemaphoreSignalNodeParams *params_out);
cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(
    cudaGraphNode_t hNode,
    const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(
    cudaGraphNode_t hNode,
    struct cudaExternalSemaphoreWaitNodeParams *params_out);
cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(
    cudaGraphNode_t hNode,
    const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param graph SEND_ONLY
 * @param from RECV_ONLY NULLABLE LENGTH:numEdges
 * @param to RECV_ONLY NULLABLE LENGTH:numEdges
 * @param numEdges SEND_RECV
 */
cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from,
                              cudaGraphNode_t *to, size_t *numEdges);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param graph SEND_ONLY
 * @param from RECV_ONLY NULLABLE LENGTH:numEdges
 * @param to RECV_ONLY NULLABLE LENGTH:numEdges
 * @param edgeData RECV_ONLY NULLABLE LENGTH:numEdges
 * @param numEdges SEND_RECV
 */
cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from,
                              cudaGraphNode_t *to, cudaGraphEdgeData *edgeData,
                              size_t *numEdges);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param graph SEND_ONLY
 * @param from RECV_ONLY NULLABLE LENGTH:numEdges
 * @param to RECV_ONLY NULLABLE LENGTH:numEdges
 * @param edgeData RECV_ONLY NULLABLE LENGTH:numEdges
 * @param numEdges SEND_RECV
 */
cudaError_t cudaGraphGetEdges_v2(cudaGraph_t graph, cudaGraphNode_t *from,
                                 cudaGraphNode_t *to,
                                 cudaGraphEdgeData *edgeData, size_t *numEdges);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hGraph SEND_ONLY
 * @param graphID RECV_ONLY
 */
cudaError_t cudaGraphGetId(cudaGraph_t hGraph, unsigned int *graphID);
#endif
/**
 * @param graph SEND_ONLY
 * @param nodes RECV_ONLY NULLABLE LENGTH:numNodes
 * @param numNodes SEND_RECV
 */
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                              size_t *numNodes);
/**
 * @param graph SEND_ONLY
 * @param pRootNodes RECV_ONLY NULLABLE LENGTH:pNumRootNodes
 * @param pNumRootNodes SEND_RECV
 */
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph,
                                  cudaGraphNode_t *pRootNodes,
                                  size_t *pNumRootNodes);
cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node,
                                       struct cudaHostNodeParams *pNodeParams);
cudaError_t
cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                           const struct cudaHostNodeParams *pNodeParams);
#if CUDART_VERSION < 12000
/**
 * @guard CUDART_VERSION < 12000
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param pErrorNode RECV_ONLY
 * @param pLogBuffer RECV_ONLY LENGTH:bufferSize
 * @param bufferSize SEND_ONLY
 */
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                 cudaGraphNode_t *pErrorNode, char *pLogBuffer,
                                 size_t bufferSize);
#endif
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                 unsigned long long flags);
#endif
/**
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec,
                                          cudaGraph_t graph,
                                          unsigned long long flags);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param instantiateParams SEND_RECV DEREF
 */
cudaError_t
cudaGraphInstantiateWithParams(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                               cudaGraphInstantiateParams *instantiateParams);
#endif
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param hSrc SEND_ONLY
 * @param hDst SEND_ONLY
 */
cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc,
                                              cudaGraphNode_t hDst);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hDst SEND_ONLY
 * @param hSrc SEND_ONLY
 */
cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst,
                                              cudaGraphNode_t hSrc);
#endif
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out RECV_ONLY
 */
cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode,
                                            cudaKernelNodeAttrID attr,
                                            cudaKernelNodeAttrValue *value_out);
cudaError_t
cudaGraphKernelNodeGetParams(cudaGraphNode_t node,
                             struct cudaKernelNodeParams *pNodeParams);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY DEREF
 */
cudaError_t
cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode,
                                cudaKernelNodeAttrID attr,
                                const cudaKernelNodeAttrValue *value);
cudaError_t
cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                             const struct cudaKernelNodeParams *pNodeParams);
/**
 * @param graphExec SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t
cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node,
                               struct cudaMemAllocNodeParams *params_out);
/**
 * @param node SEND_ONLY
 * @param dptr_out RECV_ONLY SIZE:8
 */
cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out);
/**
 * @param node SEND_ONLY
 * @param pNodeParams RECV_ONLY
 */
cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,
                                         struct cudaMemsetParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_ONLY DEREF
 */
cudaError_t
cudaGraphMemsetNodeSetParams(cudaGraphNode_t node,
                             const struct cudaMemsetParams *pNodeParams);
/**
 * @param pNode RECV_ONLY
 * @param originalNode SEND_ONLY
 * @param clonedGraph SEND_ONLY
 */
cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode,
                                     cudaGraphNode_t originalNode,
                                     cudaGraph_t clonedGraph);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hNode SEND_ONLY
 * @param phGraph RECV_ONLY
 */
cudaError_t cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode,
                                            cudaGraph_t *phGraph);
#endif
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param node SEND_ONLY
 * @param pDependencies RECV_ONLY NULLABLE LENGTH:pNumDependencies
 * @param pNumDependencies SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                                         cudaGraphNode_t *pDependencies,
                                         size_t *pNumDependencies);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param node SEND_ONLY
 * @param pDependencies RECV_ONLY NULLABLE LENGTH:pNumDependencies
 * @param edgeData RECV_ONLY NULLABLE LENGTH:pNumDependencies
 * @param pNumDependencies SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                                         cudaGraphNode_t *pDependencies,
                                         cudaGraphEdgeData *edgeData,
                                         size_t *pNumDependencies);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param node SEND_ONLY
 * @param pDependencies RECV_ONLY NULLABLE LENGTH:pNumDependencies
 * @param edgeData RECV_ONLY NULLABLE LENGTH:pNumDependencies
 * @param pNumDependencies SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependencies_v2(cudaGraphNode_t node,
                                            cudaGraphNode_t *pDependencies,
                                            cudaGraphEdgeData *edgeData,
                                            size_t *pNumDependencies);
#endif
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param node SEND_ONLY
 * @param pDependentNodes RECV_ONLY NULLABLE LENGTH:pNumDependentNodes
 * @param pNumDependentNodes SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                                           cudaGraphNode_t *pDependentNodes,
                                           size_t *pNumDependentNodes);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param node SEND_ONLY
 * @param pDependentNodes RECV_ONLY NULLABLE LENGTH:pNumDependentNodes
 * @param edgeData RECV_ONLY NULLABLE LENGTH:pNumDependentNodes
 * @param pNumDependentNodes SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                                           cudaGraphNode_t *pDependentNodes,
                                           cudaGraphEdgeData *edgeData,
                                           size_t *pNumDependentNodes);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param node SEND_ONLY
 * @param pDependentNodes RECV_ONLY NULLABLE LENGTH:pNumDependentNodes
 * @param edgeData RECV_ONLY NULLABLE LENGTH:pNumDependentNodes
 * @param pNumDependentNodes SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependentNodes_v2(cudaGraphNode_t node,
                                              cudaGraphNode_t *pDependentNodes,
                                              cudaGraphEdgeData *edgeData,
                                              size_t *pNumDependentNodes);
#endif
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled RECV_ONLY
 */
cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec,
                                    cudaGraphNode_t hNode,
                                    unsigned int *isEnabled);
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hNode SEND_ONLY
 * @param nodeId RECV_ONLY
 */
cudaError_t cudaGraphNodeGetLocalId(cudaGraphNode_t hNode,
                                    unsigned int *nodeId);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 */
cudaError_t cudaGraphNodeGetParams(cudaGraphNode_t node,
                                   struct cudaGraphNodeParams *nodeParams);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param hNode SEND_ONLY
 * @param toolsNodeId RECV_ONLY
 */
cudaError_t cudaGraphNodeGetToolsId(cudaGraphNode_t hNode,
                                    unsigned long long *toolsNodeId);
#endif
/**
 * @param node SEND_ONLY
 * @param pType RECV_ONLY
 */
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,
                                 enum cudaGraphNodeType *pType);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_ONLY
 */
cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec,
                                    cudaGraphNode_t hNode,
                                    unsigned int isEnabled);
#if CUDART_VERSION >= 12000
/**
 * @guard CUDART_VERSION >= 12000
 */
cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node,
                                   struct cudaGraphNodeParams *nodeParams);
#endif
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph,
                                       cudaUserObject_t object,
                                       unsigned int count);
#if CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION < 13000
 * @param graph SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph,
                                        const cudaGraphNode_t *from,
                                        const cudaGraphNode_t *to,
                                        size_t numDependencies);
#endif
#if CUDART_VERSION >= 13000
/**
 * @guard CUDART_VERSION >= 13000
 * @param graph SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 * @param edgeData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph,
                                        const cudaGraphNode_t *from,
                                        const cudaGraphNode_t *to,
                                        const cudaGraphEdgeData *edgeData,
                                        size_t numDependencies);
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
/**
 * @guard CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
 * @param graph SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 * @param edgeData SEND_ONLY LENGTH:numDependencies
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphRemoveDependencies_v2(cudaGraph_t graph,
                                           const cudaGraphNode_t *from,
                                           const cudaGraphNode_t *to,
                                           const cudaGraphEdgeData *edgeData,
                                           size_t numDependencies);
#endif
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph,
                                      cudaUserObject_t object,
                                      unsigned int count, unsigned int flags);
/**
 * @param graphExec SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
/**
 * @param count SEND_ONLY
 * @param resources SEND_ONLY LENGTH:count
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphicsMapResources(int count,
                                     cudaGraphicsResource_t *resources,
                                     cudaStream_t stream);
/**
 * @param mipmappedArray RECV_ONLY
 * @param resource SEND_ONLY
 */
cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(
    cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource);
/**
 * @param devPtr RECV_ONLY
 * @param size RECV_ONLY
 * @param resource SEND_ONLY
 */
cudaError_t
cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
                                     cudaGraphicsResource_t resource);
/**
 * @param resource SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                            unsigned int flags);
/**
 * @param array RECV_ONLY
 * @param resource SEND_ONLY
 * @param arrayIndex SEND_ONLY
 * @param mipLevel SEND_ONLY
 */
cudaError_t cudaGraphicsSubResourceGetMappedArray(
    cudaArray_t *array, cudaGraphicsResource_t resource,
    unsigned int arrayIndex, unsigned int mipLevel);
/**
 * @param count SEND_ONLY
 * @param resources SEND_ONLY LENGTH:count
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphicsUnmapResources(int count,
                                       cudaGraphicsResource_t *resources,
                                       cudaStream_t stream);
/**
 * @param resource SEND_ONLY
 */
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);

// Registry-only operations without API declarations above. The code generator
// reads these annotations directly; the C++ parser intentionally ignores them.
#if 0
/** @disabled */
void __cudaRegisterFatBinary();
/** @disabled */
void __cudaRegisterFatBinaryEnd();
/** @disabled */
void __cudaUnregisterFatBinary();
/** @disabled */
void __cudaRegisterFunction();
/** @disabled */
void __cudaRegisterVar();
/** @disabled */
void __cudaRegisterManagedVar();
/**
 * @disabled
 * @guard CUDART_VERSION >= 13000
 */
void __cudaGetKernel();
/**
 * @disabled
 * @guard CUDART_VERSION >= 13000
 */
void __cudaLaunchKernel();
/** @disabled */
void lupineCudartFuncParamLayout();
#endif
