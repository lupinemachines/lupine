#include <hip/hip_runtime_api.h>

// HIP runtime API forwarded call-for-call to the server's libamdhip64.
//
// Every generated call routes to the calling thread's current device, which the
// manual hipSetDevice selects; handles are not yet tracked per server, so a
// multi-server layout is only correct while every handle a thread uses belongs
// to its current device. Device ordinals other than the routing key travel
// untranslated. Entry points whose arguments cannot be marshalled by
// annotation are bare declarations at the end of this file and generate
// not-supported stubs.

/**
 * @disabled client - manual client initializes every configured route
 * @param flags SEND_ONLY
 */
hipError_t hipInit(unsigned int flags);
/**
 * @disabled client - manual client reports the virtual device table size
 * @param count RECV_ONLY
 */
hipError_t hipGetDeviceCount(int *count);
/**
 * @disabled client - manual client maps the virtual device ordinal
 * @param device RECV_ONLY
 * @param ordinal SEND_ONLY
 */
hipError_t hipDeviceGet(int *device, int ordinal);
/**
 * @disabled client - manual client maps the virtual device ordinal and keeps
 * the thread's current device for routing
 * @param deviceId SEND_ONLY
 */
hipError_t hipSetDevice(int deviceId);
/**
 * @disabled client - the current device is client state
 * @param deviceId RECV_ONLY
 */
hipError_t hipGetDevice(int *deviceId);
hipError_t hipGetLastError(void);
hipError_t hipExtGetLastError(void);
hipError_t hipPeekAtLastError(void);
/**
 * @param prop RECV_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 * @clearfields prop hdpMemFlushCntl hdpRegFlushCntl
 */
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600 *prop,
                                       int deviceId);
/**
 * @param name RECV_ONLY LENGTH:len
 * @param len SEND_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceGetName(char *name, int len, int deviceId);
/**
 * @param bytes RECV_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceTotalMem(size_t *bytes, int deviceId);
/**
 * @param pi RECV_ONLY
 * @param attr SEND_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId);
/**
 * @param driverVersion RECV_ONLY
 */
hipError_t hipDriverGetVersion(int *driverVersion);
/**
 * @param runtimeVersion RECV_ONLY
 */
hipError_t hipRuntimeGetVersion(int *runtimeVersion);
/**
 * @routingkey HIP_DEVICE device
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipDeviceComputeCapability(int *major, int *minor,
                                      hipDevice_t device);
/**
 * @routingkey HIP_DEVICE device
 * @param uuid RECV_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipDeviceGetUuid(hipUUID *uuid, hipDevice_t device);
/**
 * @param value RECV_ONLY
 * @param attr SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param dstDevice SEND_ONLY
 */
hipError_t hipDeviceGetP2PAttribute(int *value, hipDeviceP2PAttr attr,
                                    int srcDevice, int dstDevice);
/**
 * @routingkey HIP_DEVICE device
 * @param pciBusId RECV_ONLY LENGTH:len
 * @param len SEND_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int device);
/**
 * @param device RECV_ONLY
 * @param pciBusId SEND_ONLY NULL_TERMINATED
 */
hipError_t hipDeviceGetByPCIBusId(int *device, const char *pciBusId);
hipError_t hipDeviceSynchronize();
hipError_t hipDeviceReset();
/**
 * @routingkey HIP_DEVICE device
 * @param mem_pool RECV_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t *mem_pool, int device);
/**
 * @routingkey HIP_DEVICE device
 * @param device SEND_ONLY
 * @param mem_pool SEND_ONLY
 */
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool);
/**
 * @routingkey HIP_DEVICE device
 * @param mem_pool RECV_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipDeviceGetMemPool(hipMemPool_t *mem_pool, int device);
/**
 * @routingkey HIP_DEVICE device
 * @param max_width RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param device SEND_ONLY
 */
hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t *max_width,
                                               const hipChannelFormatDesc *desc,
                                               int device);
/**
 * @param cacheConfig SEND_ONLY
 */
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
/**
 * @param cacheConfig RECV_ONLY
 */
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig);
/**
 * @param pValue RECV_ONLY
 * @param limit SEND_ONLY
 */
hipError_t hipDeviceGetLimit(size_t *pValue, enum hipLimit_t limit);
/**
 * @param limit SEND_ONLY
 * @param value SEND_ONLY
 */
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value);
/**
 * @param pConfig RECV_ONLY
 */
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *pConfig);
/**
 * @param flags RECV_ONLY
 */
hipError_t hipGetDeviceFlags(unsigned int *flags);
/**
 * @param config SEND_ONLY
 */
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
/**
 * @param flags SEND_ONLY
 */
hipError_t hipSetDeviceFlags(unsigned flags);
/**
 * @param device RECV_ONLY
 * @param prop SEND_ONLY DEREF
 */
hipError_t hipChooseDeviceR0600(int *device, const hipDeviceProp_tR0600 *prop);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param linktype RECV_ONLY
 * @param hopcount RECV_ONLY
 */
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2,
                                        uint32_t *linktype, uint32_t *hopcount);
/**
 * @param leastPriority RECV_ONLY
 * @param greatestPriority RECV_ONLY
 */
hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority,
                                           int *greatestPriority);
/**
 * @routingkey HIP_DEVICE device
 * @param device SEND_ONLY
 */
hipError_t hipDeviceGraphMemTrim(int device);
hipError_t hipProfilerStart();
hipError_t hipProfilerStop();
/**
 * @param handle RECV_ONLY
 * @param devPtr SEND_ONLY
 */
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t *handle, void *devPtr);
/**
 * @param devPtr RECV_ONLY
 * @param handle SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipIpcOpenMemHandle(void **devPtr, hipIpcMemHandle_t handle,
                               unsigned int flags);
/**
 * @param devPtr SEND_ONLY
 */
hipError_t hipIpcCloseMemHandle(void *devPtr);
/**
 * @param handle RECV_ONLY
 * @param event SEND_ONLY
 */
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t *handle, hipEvent_t event);
/**
 * @param event RECV_ONLY
 * @param handle SEND_ONLY
 */
hipError_t hipIpcOpenEventHandle(hipEvent_t *event, hipIpcEventHandle_t handle);
/**
 * @param stream RECV_ONLY
 */
hipError_t hipStreamCreate(hipStream_t *stream);
/**
 * @param stream RECV_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags);
/**
 * @param stream RECV_ONLY
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags,
                                       int priority);
/**
 * @param stream SEND_ONLY
 */
hipError_t hipStreamDestroy(hipStream_t stream);
/**
 * @param stream SEND_ONLY
 */
hipError_t hipStreamQuery(hipStream_t stream);
/**
 * @param stream SEND_ONLY
 */
hipError_t hipStreamSynchronize(hipStream_t stream);
/**
 * @param stream SEND_ONLY
 * @param event SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param flags RECV_ONLY
 */
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags);
/**
 * @param stream SEND_ONLY
 * @param streamId RECV_ONLY
 */
hipError_t hipStreamGetId(hipStream_t stream, unsigned long long *streamId);
/**
 * @param stream SEND_ONLY
 * @param priority RECV_ONLY
 */
hipError_t hipStreamGetPriority(hipStream_t stream, int *priority);
/**
 * @param stream SEND_ONLY
 * @param device RECV_ONLY
 */
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t *device);
/**
 * @param stream RECV_ONLY
 * @param cuMaskSize SEND_ONLY
 * @param cuMask SEND_ONLY LENGTH:cuMaskSize
 */
hipError_t hipExtStreamCreateWithCUMask(hipStream_t *stream,
                                        uint32_t cuMaskSize,
                                        const uint32_t *cuMask);
/**
 * @param stream SEND_ONLY
 * @param cuMaskSize SEND_ONLY
 * @param cuMask RECV_ONLY LENGTH:cuMaskSize
 */
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize,
                                 uint32_t *cuMask);
/**
 * @param stream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_ONLY DEREF
 */
hipError_t hipStreamSetAttribute(hipStream_t stream, hipLaunchAttributeID attr,
                                 const hipLaunchAttributeValue *value);
/**
 * @param stream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out RECV_ONLY
 */
hipError_t hipStreamGetAttribute(hipStream_t stream, hipLaunchAttributeID attr,
                                 hipLaunchAttributeValue *value_out);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src);
/**
 * @param stream SEND_ONLY
 * @param ptr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 * @param mask SEND_ONLY
 */
hipError_t hipStreamWaitValue32(hipStream_t stream, void *ptr, uint32_t value,
                                unsigned int flags, uint32_t mask);
/**
 * @param stream SEND_ONLY
 * @param ptr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 * @param mask SEND_ONLY
 */
hipError_t hipStreamWaitValue64(hipStream_t stream, void *ptr, uint64_t value,
                                unsigned int flags, uint64_t mask);
/**
 * @param stream SEND_ONLY
 * @param ptr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipStreamWriteValue32(hipStream_t stream, void *ptr, uint32_t value,
                                 unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param ptr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipStreamWriteValue64(hipStream_t stream, void *ptr, uint64_t value,
                                 unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param count SEND_ONLY
 * @param paramArray SEND_ONLY LENGTH:count
 * @param flags SEND_ONLY
 */
hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams *paramArray,
                               unsigned int flags);
/**
 * @param phGraphNode RECV_ONLY
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams paramArray count
 */
hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t *phGraphNode,
                                     hipGraph_t hGraph,
                                     const hipGraphNode_t *dependencies,
                                     size_t numDependencies,
                                     const hipBatchMemOpNodeParams *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams paramArray count
 */
hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_ONLY
 * @deeparray nodeParams paramArray count
 */
hipError_t
hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec,
                                    hipGraphNode_t hNode,
                                    const hipBatchMemOpNodeParams *nodeParams);
/**
 * @param event RECV_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned flags);
/**
 * @param event RECV_ONLY
 */
hipError_t hipEventCreate(hipEvent_t *event);
/**
 * @param event SEND_ONLY
 * @param stream SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream,
                                   unsigned int flags);
/**
 * @param event SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
/**
 * @param event SEND_ONLY
 */
hipError_t hipEventDestroy(hipEvent_t event);
/**
 * @param event SEND_ONLY
 */
hipError_t hipEventSynchronize(hipEvent_t event);
/**
 * @param ms RECV_ONLY
 * @param start SEND_ONLY
 * @param stop SEND_ONLY
 */
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);
/**
 * @param event SEND_ONLY
 */
hipError_t hipEventQuery(hipEvent_t event);
/**
 * @param attributes RECV_ONLY
 * @param ptr SEND_ONLY
 */
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes,
                                   const void *ptr);
/**
 * @param ptr RECV_ONLY
 * @param size SEND_ONLY
 */
hipError_t hipMalloc(void **ptr, size_t size);
/**
 * @param ptr RECV_ONLY
 * @param sizeBytes SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipExtMallocWithFlags(void **ptr, size_t sizeBytes,
                                 unsigned int flags);
/**
 * @param dev_ptr RECV_ONLY
 * @param size SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipMallocManaged(void **dev_ptr, size_t size, unsigned int flags);
/**
 * @param dev_ptr SEND_ONLY
 * @param count SEND_ONLY
 * @param device SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemPrefetchAsync(const void *dev_ptr, size_t count, int device,
                               hipStream_t stream);
/**
 * @param dev_ptr SEND_ONLY
 * @param count SEND_ONLY
 * @param location SEND_ONLY
 * @param flags SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemPrefetchAsync_v2(const void *dev_ptr, size_t count,
                                  hipMemLocation location, unsigned int flags,
                                  hipStream_t stream);
/**
 * @param dev_ptr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipMemAdvise(const void *dev_ptr, size_t count,
                        hipMemoryAdvise advice, int device);
/**
 * @param dev_ptr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param location SEND_ONLY
 */
hipError_t hipMemAdvise_v2(const void *dev_ptr, size_t count,
                           hipMemoryAdvise advice, hipMemLocation location);
/**
 * @param data_size SEND_ONLY
 * @param data RECV_ONLY LENGTH:data_size
 * @param attribute SEND_ONLY
 * @param dev_ptr SEND_ONLY
 * @param count SEND_ONLY
 */
hipError_t hipMemRangeGetAttribute(void *data, size_t data_size,
                                   hipMemRangeAttribute attribute,
                                   const void *dev_ptr, size_t count);
/**
 * @param stream SEND_ONLY
 * @param dev_ptr SEND_ONLY
 * @param length SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void *dev_ptr,
                                   size_t length, unsigned int flags);
/**
 * @param dev_ptr RECV_ONLY
 * @param size SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream);
/**
 * @param dev_ptr SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipFreeAsync(void *dev_ptr, hipStream_t stream);
/**
 * @param mem_pool SEND_ONLY
 * @param min_bytes_to_hold SEND_ONLY
 */
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
/**
 * @param mem_pool SEND_ONLY
 * @param desc_list SEND_ONLY LENGTH:count
 * @param count SEND_ONLY
 */
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool,
                               const hipMemAccessDesc *desc_list, size_t count);
/**
 * @param flags RECV_ONLY
 * @param mem_pool SEND_ONLY
 * @param location SEND_ONLY DEREF
 */
hipError_t hipMemPoolGetAccess(hipMemAccessFlags *flags, hipMemPool_t mem_pool,
                               hipMemLocation *location);
/**
 * @param mem_pool RECV_ONLY
 * @param pool_props SEND_ONLY DEREF
 */
hipError_t hipMemPoolCreate(hipMemPool_t *mem_pool,
                            const hipMemPoolProps *pool_props);
/**
 * @param mem_pool SEND_ONLY
 */
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
/**
 * @param dev_ptr RECV_ONLY
 * @param size SEND_ONLY
 * @param mem_pool SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size,
                                  hipMemPool_t mem_pool, hipStream_t stream);
/**
 * @param export_data RECV_ONLY
 * @param dev_ptr SEND_ONLY
 */
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData *export_data,
                                   void *dev_ptr);
/**
 * @param dev_ptr RECV_ONLY
 * @param mem_pool SEND_ONLY
 * @param export_data SEND_ONLY DEREF
 */
hipError_t hipMemPoolImportPointer(void **dev_ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData *export_data);
/**
 * @param ptr RECV_ONLY
 * @param pitch RECV_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height);
/**
 * @param dptr RECV_ONLY
 * @param pitch RECV_ONLY
 * @param widthInBytes SEND_ONLY
 * @param height SEND_ONLY
 * @param elementSizeBytes SEND_ONLY
 */
hipError_t hipMemAllocPitch(hipDeviceptr_t *dptr, size_t *pitch,
                            size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes);
/**
 * @param ptr SEND_ONLY
 */
hipError_t hipFree(void *ptr);
/**
 * @param dst SEND_ONLY
 * @param sizeBytes SEND_ONLY
 * @param src SEND_ONLY LENGTH:sizeBytes
 */
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void *src, size_t sizeBytes);
/**
 * @param src SEND_ONLY
 * @param sizeBytes SEND_ONLY
 * @param dst RECV_ONLY LENGTH:sizeBytes
 */
hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 * @param sizeBytes SEND_ONLY
 */
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                         size_t sizeBytes);
/**
 * @param dstDevice SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray,
                         size_t srcOffset, size_t ByteCount);
/**
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset,
                         hipDeviceptr_t srcDevice, size_t ByteCount);
/**
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset,
                         hipArray_t srcArray, size_t srcOffset,
                         size_t ByteCount);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 * @param sizeBytes SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                              size_t sizeBytes, hipStream_t stream);
/**
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param count SEND_ONLY
 * @param dst RECV_ONLY LENGTH:count
 */
hipError_t hipMemcpyAtoH(void *dst, hipArray_t srcArray, size_t srcOffset,
                         size_t count);
/**
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param count SEND_ONLY
 * @param srcHost SEND_ONLY LENGTH:count
 */
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset,
                         const void *srcHost, size_t count);
/**
 * @param dst SEND_ONLY
 * @param wOffsetDst SEND_ONLY
 * @param hOffsetDst SEND_ONLY
 * @param src SEND_ONLY
 * @param wOffsetSrc SEND_ONLY
 * @param hOffsetSrc SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 */
hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst,
                                   size_t hOffsetDst, hipArray_const_t src,
                                   size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t width, size_t height,
                                   hipMemcpyKind kind);
/**
 * @param dst SEND_ONLY
 * @param dstDeviceId SEND_ONLY
 * @param src SEND_ONLY
 * @param srcDeviceId SEND_ONLY
 * @param sizeBytes SEND_ONLY
 */
hipError_t hipMemcpyPeer(void *dst, int dstDeviceId, const void *src,
                         int srcDeviceId, size_t sizeBytes);
/**
 * @param dst SEND_ONLY
 * @param dstDeviceId SEND_ONLY
 * @param src SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param sizeBytes SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src,
                              int srcDevice, size_t sizeBytes,
                              hipStream_t stream);
/**
 * @param dptr RECV_ONLY NULLABLE
 * @param bytes RECV_ONLY NULLABLE
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t hmod, const char *name);
/**
 * @param dst SEND_ONLY
 * @param value SEND_ONLY
 * @param sizeBytes SEND_ONLY
 */
hipError_t hipMemset(void *dst, int value, size_t sizeBytes);
/**
 * @param dest SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 */
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count);
/**
 * @param dest SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value,
                            size_t count, hipStream_t stream);
/**
 * @param dest SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 */
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value,
                        size_t count);
/**
 * @param dest SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value,
                             size_t count, hipStream_t stream);
/**
 * @param dest SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 */
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
/**
 * @param dst SEND_ONLY
 * @param value SEND_ONLY
 * @param sizeBytes SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes,
                          hipStream_t stream);
/**
 * @param dst SEND_ONLY
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream);
/**
 * @param dst SEND_ONLY
 * @param pitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
hipError_t hipMemset2D(void *dst, size_t pitch, int value, size_t width,
                       size_t height);
/**
 * @param dst SEND_ONLY
 * @param pitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width,
                            size_t height, hipStream_t stream);
/**
 * @param pitchedDevPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param extent SEND_ONLY
 */
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value,
                       hipExtent extent);
/**
 * @param pitchedDevPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param extent SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                            hipExtent extent, hipStream_t stream);
/**
 * @param dst SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch,
                         unsigned char value, size_t width, size_t height);
/**
 * @param dst SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch,
                              unsigned char value, size_t width, size_t height,
                              hipStream_t stream);
/**
 * @param dst SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch,
                          unsigned short value, size_t width, size_t height);
/**
 * @param dst SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch,
                               unsigned short value, size_t width,
                               size_t height, hipStream_t stream);
/**
 * @param dst SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch,
                          unsigned int value, size_t width, size_t height);
/**
 * @param dst SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch,
                               unsigned int value, size_t width, size_t height,
                               hipStream_t stream);
/**
 * @param free RECV_ONLY
 * @param total RECV_ONLY
 */
hipError_t hipMemGetInfo(size_t *free, size_t *total);
/**
 * @param ptr SEND_ONLY
 * @param size RECV_ONLY
 */
hipError_t hipMemPtrGetInfo(void *ptr, size_t *size);
/**
 * @param pbase RECV_ONLY NULLABLE
 * @param psize RECV_ONLY NULLABLE
 * @param dptr SEND_ONLY
 */
hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize,
                                 hipDeviceptr_t dptr);
/**
 * @param array RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipMallocArray(hipArray_t *array, const hipChannelFormatDesc *desc,
                          size_t width, size_t height, unsigned int flags);
/**
 * @param pHandle RECV_ONLY
 * @param pAllocateArray SEND_ONLY DEREF
 */
hipError_t hipArrayCreate(hipArray_t *pHandle,
                          const HIP_ARRAY_DESCRIPTOR *pAllocateArray);
/**
 * @param array SEND_ONLY
 */
hipError_t hipArrayDestroy(hipArray_t array);
/**
 * @param array RECV_ONLY
 * @param pAllocateArray SEND_ONLY DEREF
 */
hipError_t hipArray3DCreate(hipArray_t *array,
                            const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray);
/**
 * @param pitchedDevPtr RECV_ONLY
 * @param extent SEND_ONLY
 */
hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent);
/**
 * @param array SEND_ONLY
 */
hipError_t hipFreeArray(hipArray_t array);
/**
 * @param array RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param extent SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipMalloc3DArray(hipArray_t *array,
                            const struct hipChannelFormatDesc *desc,
                            struct hipExtent extent, unsigned int flags);
/**
 * @param desc RECV_ONLY NULLABLE
 * @param extent RECV_ONLY NULLABLE
 * @param flags RECV_ONLY NULLABLE
 * @param array SEND_ONLY
 */
hipError_t hipArrayGetInfo(hipChannelFormatDesc *desc, hipExtent *extent,
                           unsigned int *flags, hipArray_t array);
/**
 * @param pArrayDescriptor RECV_ONLY
 * @param array SEND_ONLY
 */
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                 hipArray_t array);
/**
 * @param pArrayDescriptor RECV_ONLY
 * @param array SEND_ONLY
 */
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                   hipArray_t array);
/**
 * @param mipmappedArray RECV_ONLY
 * @param desc SEND_ONLY DEREF
 * @param extent SEND_ONLY
 * @param numLevels SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t *mipmappedArray,
                                   const struct hipChannelFormatDesc *desc,
                                   struct hipExtent extent,
                                   unsigned int numLevels, unsigned int flags);
/**
 * @param mipmappedArray SEND_ONLY
 */
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);
/**
 * @param levelArray RECV_ONLY
 * @param mipmappedArray SEND_ONLY
 * @param level SEND_ONLY
 */
hipError_t hipGetMipmappedArrayLevel(hipArray_t *levelArray,
                                     hipMipmappedArray_const_t mipmappedArray,
                                     unsigned int level);
/**
 * @param pHandle RECV_ONLY
 * @param pMipmappedArrayDesc SEND_ONLY DEREF
 * @param numMipmapLevels SEND_ONLY
 */
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t *pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels);
/**
 * @param hMipmappedArray SEND_ONLY
 */
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);
/**
 * @param pLevelArray RECV_ONLY
 * @param hMipMappedArray SEND_ONLY
 * @param level SEND_ONLY
 */
hipError_t hipMipmappedArrayGetLevel(hipArray_t *pLevelArray,
                                     hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level);
/**
 * @param canAccessPeer RECV_ONLY
 * @param deviceId SEND_ONLY
 * @param peerDeviceId SEND_ONLY
 */
hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId,
                                  int peerDeviceId);
/**
 * @param peerDeviceId SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
/**
 * @param peerDeviceId SEND_ONLY
 */
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
/**
 * @routingkey HIP_DEVICE device
 * @param ctx RECV_ONLY
 * @param flags SEND_ONLY
 * @param device SEND_ONLY
 */
hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device);
/**
 * @param ctx SEND_ONLY
 */
hipError_t hipCtxDestroy(hipCtx_t ctx);
/**
 * @param ctx RECV_ONLY NULLABLE
 */
hipError_t hipCtxPopCurrent(hipCtx_t *ctx);
/**
 * @param ctx SEND_ONLY
 */
hipError_t hipCtxPushCurrent(hipCtx_t ctx);
/**
 * @param ctx SEND_ONLY
 */
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
/**
 * @param ctx RECV_ONLY
 */
hipError_t hipCtxGetCurrent(hipCtx_t *ctx);
/**
 * @param device RECV_ONLY
 */
hipError_t hipCtxGetDevice(hipDevice_t *device);
/**
 * @param ctx SEND_ONLY
 * @param apiVersion RECV_ONLY
 */
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int *apiVersion);
/**
 * @param cacheConfig RECV_ONLY
 */
hipError_t hipCtxGetCacheConfig(hipFuncCache_t *cacheConfig);
/**
 * @param cacheConfig SEND_ONLY
 */
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
/**
 * @param config SEND_ONLY
 */
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
/**
 * @param pConfig RECV_ONLY
 */
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig *pConfig);
hipError_t hipCtxSynchronize();
/**
 * @param flags RECV_ONLY
 */
hipError_t hipCtxGetFlags(unsigned int *flags);
/**
 * @param peerCtx SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
/**
 * @param peerCtx SEND_ONLY
 */
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);
/**
 * @routingkey HIP_DEVICE dev
 * @param dev SEND_ONLY
 * @param flags RECV_ONLY
 * @param active RECV_ONLY
 */
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int *flags,
                                       int *active);
/**
 * @routingkey HIP_DEVICE dev
 * @param dev SEND_ONLY
 */
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
/**
 * @routingkey HIP_DEVICE dev
 * @param pctx RECV_ONLY
 * @param dev SEND_ONLY
 */
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t dev);
/**
 * @routingkey HIP_DEVICE dev
 * @param dev SEND_ONLY
 */
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
/**
 * @routingkey HIP_DEVICE dev
 * @param dev SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
/**
 * @param module SEND_ONLY
 */
hipError_t hipModuleUnload(hipModule_t module);
/**
 * @param function RECV_ONLY
 * @param module SEND_ONLY
 * @param kname SEND_ONLY NULL_TERMINATED
 */
hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module,
                                const char *kname);
/**
 * @param count RECV_ONLY
 * @param mod SEND_ONLY
 */
hipError_t hipModuleGetFunctionCount(unsigned int *count, hipModule_t mod);
/**
 * @param library SEND_ONLY
 */
hipError_t hipLibraryUnload(hipLibrary_t library);
/**
 * @param pKernel RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
hipError_t hipLibraryGetKernel(hipKernel_t *pKernel, hipLibrary_t library,
                               const char *name);
/**
 * @param count RECV_ONLY
 * @param library SEND_ONLY
 */
hipError_t hipLibraryGetKernelCount(unsigned int *count, hipLibrary_t library);
/**
 * @param numKernels SEND_ONLY
 * @param kernels RECV_ONLY LENGTH:numKernels
 * @param library SEND_ONLY
 */
hipError_t hipLibraryEnumerateKernels(hipKernel_t *kernels,
                                      unsigned int numKernels,
                                      hipLibrary_t library);
/**
 * @param library RECV_ONLY
 * @param kernel SEND_ONLY
 */
hipError_t hipKernelGetLibrary(hipLibrary_t *library, hipKernel_t kernel);
/**
 * @param value RECV_ONLY
 * @param attrib SEND_ONLY
 * @param hfunc SEND_ONLY
 */
hipError_t hipFuncGetAttribute(int *value, hipFunction_attribute attrib,
                               hipFunction_t hfunc);
/**
 * @param gridSize RECV_ONLY
 * @param blockSize RECV_ONLY
 * @param f SEND_ONLY
 * @param dynSharedMemPerBlk SEND_ONLY
 * @param blockSizeLimit SEND_ONLY
 */
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int *gridSize,
                                                   int *blockSize,
                                                   hipFunction_t f,
                                                   size_t dynSharedMemPerBlk,
                                                   int blockSizeLimit);
/**
 * @param gridSize RECV_ONLY
 * @param blockSize RECV_ONLY
 * @param f SEND_ONLY
 * @param dynSharedMemPerBlk SEND_ONLY
 * @param blockSizeLimit SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int *gridSize, int *blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit, unsigned int flags);
/**
 * @param numBlocks RECV_ONLY
 * @param f SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynSharedMemPerBlk SEND_ONLY
 */
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk);
/**
 * @param numBlocks RECV_ONLY
 * @param f SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynSharedMemPerBlk SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk,
    unsigned int flags);
/**
 * @param pTexObject RECV_ONLY
 * @param pResDesc SEND_ONLY DEREF
 * @param pTexDesc SEND_ONLY NULLABLE
 * @param pResViewDesc SEND_ONLY NULLABLE
 */
hipError_t
hipCreateTextureObject(hipTextureObject_t *pTexObject,
                       const hipResourceDesc *pResDesc,
                       const hipTextureDesc *pTexDesc,
                       const struct hipResourceViewDesc *pResViewDesc);
/**
 * @param textureObject SEND_ONLY
 */
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);
/**
 * @param desc RECV_ONLY
 * @param array SEND_ONLY
 */
hipError_t hipGetChannelDesc(hipChannelFormatDesc *desc,
                             hipArray_const_t array);
/**
 * @param pResDesc RECV_ONLY
 * @param textureObject SEND_ONLY
 */
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc *pResDesc,
                                           hipTextureObject_t textureObject);
/**
 * @param pResViewDesc RECV_ONLY
 * @param textureObject SEND_ONLY
 */
hipError_t
hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc *pResViewDesc,
                                    hipTextureObject_t textureObject);
/**
 * @param pTexDesc RECV_ONLY
 * @param textureObject SEND_ONLY
 */
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc *pTexDesc,
                                          hipTextureObject_t textureObject);
/**
 * @param pTexObject RECV_ONLY
 * @param pResDesc SEND_ONLY DEREF
 * @param pTexDesc SEND_ONLY NULLABLE
 * @param pResViewDesc SEND_ONLY NULLABLE
 */
hipError_t hipTexObjectCreate(hipTextureObject_t *pTexObject,
                              const HIP_RESOURCE_DESC *pResDesc,
                              const HIP_TEXTURE_DESC *pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC *pResViewDesc);
/**
 * @param texObject SEND_ONLY
 */
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject);
/**
 * @param pResDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC *pResDesc,
                                       hipTextureObject_t texObject);
/**
 * @param pResViewDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC *pResViewDesc,
                                           hipTextureObject_t texObject);
/**
 * @param pTexDesc RECV_ONLY
 * @param texObject SEND_ONLY
 */
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC *pTexDesc,
                                      hipTextureObject_t texObject);
/**
 * @param pSurfObject RECV_ONLY
 * @param pResDesc SEND_ONLY DEREF
 */
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t *pSurfObject,
                                  const hipResourceDesc *pResDesc);
/**
 * @param surfaceObject SEND_ONLY
 */
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);
/**
 * @param stream SEND_ONLY
 * @param mode SEND_ONLY
 */
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
/**
 * @param stream SEND_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dependencyData SEND_ONLY LENGTH:numDependencies
 * @param mode SEND_ONLY
 */
hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                        const hipGraphNode_t *dependencies,
                                        const hipGraphEdgeData *dependencyData,
                                        size_t numDependencies,
                                        hipStreamCaptureMode mode);
/**
 * @param stream SEND_ONLY
 * @param pGraph RECV_ONLY
 */
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t *pGraph);
/**
 * @param stream SEND_ONLY
 * @param pCaptureStatus RECV_ONLY
 * @param pId RECV_ONLY NULLABLE
 */
hipError_t hipStreamGetCaptureInfo(hipStream_t stream,
                                   hipStreamCaptureStatus *pCaptureStatus,
                                   unsigned long long *pId);
/**
 * @param stream SEND_ONLY
 * @param pCaptureStatus RECV_ONLY
 */
hipError_t hipStreamIsCapturing(hipStream_t stream,
                                hipStreamCaptureStatus *pCaptureStatus);
/**
 * @param stream SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param flags SEND_ONLY
 */
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream,
                                              hipGraphNode_t *dependencies,
                                              size_t numDependencies,
                                              unsigned int flags);
/**
 * @param mode SEND_RECV
 */
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode *mode);
/**
 * @param pGraph RECV_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipGraphCreate(hipGraph_t *pGraph, unsigned int flags);
/**
 * @param graph SEND_ONLY
 */
hipError_t hipGraphDestroy(hipGraph_t graph);
/**
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 */
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t *from,
                                   const hipGraphNode_t *to,
                                   size_t numDependencies);
/**
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param from SEND_ONLY LENGTH:numDependencies
 * @param to SEND_ONLY LENGTH:numDependencies
 */
hipError_t hipGraphRemoveDependencies(hipGraph_t graph,
                                      const hipGraphNode_t *from,
                                      const hipGraphNode_t *to,
                                      size_t numDependencies);
/**
 * @param graph SEND_ONLY
 * @param numEdges SEND_RECV
 * @param from RECV_ONLY NULLABLE LENGTH:numEdges
 * @param to RECV_ONLY NULLABLE LENGTH:numEdges
 */
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t *from,
                            hipGraphNode_t *to, size_t *numEdges);
/**
 * @param graph SEND_ONLY
 * @param numNodes SEND_RECV
 * @param nodes RECV_ONLY NULLABLE LENGTH:numNodes
 */
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t *nodes,
                            size_t *numNodes);
/**
 * @param graph SEND_ONLY
 * @param pNumRootNodes SEND_RECV
 * @param pRootNodes RECV_ONLY NULLABLE LENGTH:pNumRootNodes
 */
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t *pRootNodes,
                                size_t *pNumRootNodes);
/**
 * @param node SEND_ONLY
 * @param pNumDependencies SEND_RECV
 * @param pDependencies RECV_ONLY NULLABLE LENGTH:pNumDependencies
 */
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node,
                                       hipGraphNode_t *pDependencies,
                                       size_t *pNumDependencies);
/**
 * @param node SEND_ONLY
 * @param pNumDependentNodes SEND_RECV
 * @param pDependentNodes RECV_ONLY NULLABLE LENGTH:pNumDependentNodes
 */
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node,
                                         hipGraphNode_t *pDependentNodes,
                                         size_t *pNumDependentNodes);
/**
 * @param node SEND_ONLY
 * @param pType RECV_ONLY
 */
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType *pType);
/**
 * @param node SEND_ONLY
 */
hipError_t hipGraphDestroyNode(hipGraphNode_t node);
/**
 * @param pGraphClone RECV_ONLY
 * @param originalGraph SEND_ONLY
 */
hipError_t hipGraphClone(hipGraph_t *pGraphClone, hipGraph_t originalGraph);
/**
 * @param pNode RECV_ONLY
 * @param originalNode SEND_ONLY
 * @param clonedGraph SEND_ONLY
 */
hipError_t hipGraphNodeFindInClone(hipGraphNode_t *pNode,
                                   hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph);
/**
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param pErrorNode RECV_ONLY NULLABLE
 * @param bufferSize SEND_ONLY
 * @param pLogBuffer RECV_ONLY NULLABLE LENGTH:bufferSize
 */
hipError_t hipGraphInstantiate(hipGraphExec_t *pGraphExec, hipGraph_t graph,
                               hipGraphNode_t *pErrorNode, char *pLogBuffer,
                               size_t bufferSize);
/**
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t *pGraphExec,
                                        hipGraph_t graph,
                                        unsigned long long flags);
/**
 * @param pGraphExec RECV_ONLY
 * @param graph SEND_ONLY
 * @param instantiateParams SEND_RECV DEREF
 */
hipError_t
hipGraphInstantiateWithParams(hipGraphExec_t *pGraphExec, hipGraph_t graph,
                              hipGraphInstantiateParams *instantiateParams);
/**
 * @param graphExec SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
/**
 * @param graphExec SEND_ONLY
 * @param stream SEND_ONLY
 */
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream);
/**
 * @param graphExec SEND_ONLY
 * @param flags RECV_ONLY
 */
hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec,
                                unsigned long long *flags);
/**
 * @param graphExec SEND_ONLY
 */
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);
/**
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param hErrorNode_out RECV_ONLY NULLABLE
 * @param updateResult_out RECV_ONLY
 */
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t *hErrorNode_out,
                              hipGraphExecUpdateResult *updateResult_out);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param pMemsetParams SEND_ONLY DEREF
 */
hipError_t hipGraphAddMemsetNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const hipMemsetParams *pMemsetParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams RECV_ONLY
 */
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node,
                                       hipMemsetParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_ONLY DEREF
 */
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node,
                                       const hipMemsetParams *pNodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param pNodeParams SEND_ONLY DEREF
 */
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec,
                                           hipGraphNode_t node,
                                           const hipMemsetParams *pNodeParams);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param childGraph SEND_ONLY
 */
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t *pGraphNode,
                                     hipGraph_t graph,
                                     const hipGraphNode_t *pDependencies,
                                     size_t numDependencies,
                                     hipGraph_t childGraph);
/**
 * @param node SEND_ONLY
 * @param pGraph RECV_ONLY
 */
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node,
                                          hipGraph_t *pGraph);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param childGraph SEND_ONLY
 */
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec,
                                               hipGraphNode_t node,
                                               hipGraph_t childGraph);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 */
hipError_t hipGraphAddEmptyNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t *pDependencies,
                                size_t numDependencies);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param event SEND_ONLY
 */
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t *pGraphNode,
                                      hipGraph_t graph,
                                      const hipGraphNode_t *pDependencies,
                                      size_t numDependencies, hipEvent_t event);
/**
 * @param node SEND_ONLY
 * @param event_out RECV_ONLY
 */
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node,
                                           hipEvent_t *event_out);
/**
 * @param node SEND_ONLY
 * @param event SEND_ONLY
 */
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node,
                                           hipEvent_t event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec,
                                               hipGraphNode_t hNode,
                                               hipEvent_t event);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param event SEND_ONLY
 */
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t *pGraphNode,
                                    hipGraph_t graph,
                                    const hipGraphNode_t *pDependencies,
                                    size_t numDependencies, hipEvent_t event);
/**
 * @param node SEND_ONLY
 * @param event_out RECV_ONLY
 */
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node,
                                         hipEvent_t *event_out);
/**
 * @param node SEND_ONLY
 * @param event SEND_ONLY
 */
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,
                                             hipGraphNode_t hNode,
                                             hipEvent_t event);
/**
 * @param pGraphNode RECV_ONLY
 * @param graph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param pDependencies SEND_ONLY LENGTH:numDependencies
 * @param dev_ptr SEND_ONLY
 */
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t *pDependencies,
                                  size_t numDependencies, void *dev_ptr);
/**
 * @param node SEND_ONLY
 * @param dev_ptr RECV_ONLY SIZE:8
 */
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void *dev_ptr);
/**
 * @param hSrc SEND_ONLY
 * @param hDst SEND_ONLY
 */
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc,
                                            hipGraphNode_t hDst);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_ONLY
 */
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec,
                                  hipGraphNode_t hNode, unsigned int isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled RECV_ONLY
 */
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec,
                                  hipGraphNode_t hNode,
                                  unsigned int *isEnabled);
/**
 * @param phGraphNode RECV_ONLY
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param memsetParams SEND_ONLY DEREF
 * @param ctx SEND_ONLY
 */
hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t *phGraphNode,
                                    hipGraph_t hGraph,
                                    const hipGraphNode_t *dependencies,
                                    size_t numDependencies,
                                    const hipMemsetParams *memsetParams,
                                    hipCtx_t ctx);
/**
 * @param phGraphNode RECV_ONLY
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dptr SEND_ONLY
 */
hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t *phGraphNode,
                                     hipGraph_t hGraph,
                                     const hipGraphNode_t *dependencies,
                                     size_t numDependencies,
                                     hipDeviceptr_t dptr);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param memsetParams SEND_ONLY DEREF
 * @param ctx SEND_ONLY
 */
hipError_t hipDrvGraphExecMemsetNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipMemsetParams *memsetParams, hipCtx_t ctx);
/**
 * @param devPtr SEND_ONLY
 * @param size SEND_ONLY
 */
hipError_t hipMemAddressFree(void *devPtr, size_t size);
/**
 * @param ptr RECV_ONLY
 * @param size SEND_ONLY
 * @param alignment SEND_ONLY
 * @param addr SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipMemAddressReserve(void **ptr, size_t size, size_t alignment,
                                void *addr, unsigned long long flags);
/**
 * @param handle RECV_ONLY
 * @param size SEND_ONLY
 * @param prop SEND_ONLY DEREF
 * @param flags SEND_ONLY
 */
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t *handle, size_t size,
                        const hipMemAllocationProp *prop,
                        unsigned long long flags);
/**
 * @param flags RECV_ONLY
 * @param location SEND_ONLY DEREF
 * @param ptr SEND_ONLY
 */
hipError_t hipMemGetAccess(unsigned long long *flags,
                           const hipMemLocation *location, void *ptr);
/**
 * @param granularity RECV_ONLY
 * @param prop SEND_ONLY DEREF
 * @param option SEND_ONLY
 */
hipError_t
hipMemGetAllocationGranularity(size_t *granularity,
                               const hipMemAllocationProp *prop,
                               hipMemAllocationGranularity_flags option);
/**
 * @param prop RECV_ONLY
 * @param handle SEND_ONLY
 */
hipError_t
hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp *prop,
                                        hipMemGenericAllocationHandle_t handle);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 * @param offset SEND_ONLY
 * @param handle SEND_ONLY
 * @param flags SEND_ONLY
 */
hipError_t hipMemMap(void *ptr, size_t size, size_t offset,
                     hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags);
/**
 * @param count SEND_ONLY
 * @param mapInfoList SEND_ONLY LENGTH:count
 * @param stream SEND_ONLY
 */
hipError_t hipMemMapArrayAsync(hipArrayMapInfo *mapInfoList, unsigned int count,
                               hipStream_t stream);
/**
 * @param handle SEND_ONLY
 */
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
/**
 * @param handle RECV_ONLY
 * @param addr SEND_ONLY
 */
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t *handle,
                                        void *addr);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 * @param count SEND_ONLY
 * @param desc SEND_ONLY LENGTH:count
 */
hipError_t hipMemSetAccess(void *ptr, size_t size, const hipMemAccessDesc *desc,
                           size_t count);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 */
hipError_t hipMemUnmap(void *ptr, size_t size);

// Host memory the server allocates, host function and code-object addresses,
// kernel parameter sizes, copies whose wire layout depends on hipMemcpyKind,
// and asynchronous copies through the request buffer.
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void *src,
                              size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream);
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyWithStream(void *dst, const void *src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream);
hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags);
hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags);
hipError_t hipModuleLoadData(hipModule_t *module, const void *image);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void **kernelParams, void **extra);
hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks,
                           dim3 dimBlocks, void **args, size_t sharedMemBytes,
                           hipStream_t stream);
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t *pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t *pDependencies,
                                   size_t numDependencies,
                                   hipMemAllocNodeParams *pNodeParams);
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node,
                                         hipMemAllocNodeParams *pNodeParams);
hipError_t
hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode,
                                hipBatchMemOpNodeParams *nodeParams_out);
