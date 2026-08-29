#include "rpc_server.h"

#ifdef LUPINE_BUILD_CUDA_BACKEND
#include <cuda.h>
#endif
#include "gen_rpc_ids.h"

// clang-format off
#define LUPINE_CUDA_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cuGetErrorString, handle_cuGetErrorString, rpc_backend::cuda) \
  HANDLER(RPC_cuGetErrorName, handle_cuGetErrorName, rpc_backend::cuda) \
  HANDLER(RPC_cuDevicePrimaryCtxRetain, handle_cuDevicePrimaryCtxRetain, rpc_backend::cuda) \
  HANDLER(RPC_cuDevicePrimaryCtxRelease_v2, handle_cuDevicePrimaryCtxRelease_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuDevicePrimaryCtxReset_v2, handle_cuDevicePrimaryCtxReset_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxCreate_v2, handle_cuCtxCreate_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxDestroy_v2, handle_cuCtxDestroy_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxSynchronize, handle_cuCtxSynchronize, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxAttach, handle_cuCtxAttach, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxDetach, handle_cuCtxDetach, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleLoad, handle_cuModuleLoad, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleGetGlobal_v2, handle_cuModuleGetGlobal_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuLinkCreate_v2, handle_cuLinkCreate_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuLinkAddData_v2, handle_cuLinkAddData_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuLinkAddFile_v2, handle_cuLinkAddFile_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuLinkComplete, handle_cuLinkComplete, rpc_backend::cuda) \
  HANDLER(RPC_cuLinkDestroy, handle_cuLinkDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryLoadData, handle_cuLibraryLoadData, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryUnload, handle_cuLibraryUnload, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryGetModule, handle_cuLibraryGetModule, rpc_backend::cuda) \
  HANDLER(RPC_cuMemFree_v2, handle_cuMemFree_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemFreeHost, handle_cuMemFreeHost, rpc_backend::cuda) \
  HANDLER(RPC_cuMemHostAlloc, handle_cuMemHostAlloc, rpc_backend::cuda) \
  HANDLER(RPC_cuMemHostGetFlags, handle_cuMemHostGetFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAllocManaged, handle_cuMemAllocManaged, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyHtoD_v2, handle_cuMemcpyHtoD_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyDtoH_v2, handle_cuMemcpyDtoH_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyAtoH_v2, handle_cuMemcpyAtoH_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy2D_v2, handle_cuMemcpy2D_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy2DUnaligned_v2, handle_cuMemcpy2DUnaligned_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy3D_v2, handle_cuMemcpy3D_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy3DPeer, handle_cuMemcpy3DPeer, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyHtoDAsync_v2, handle_cuMemcpyHtoDAsync_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyDtoHAsync_v2, handle_cuMemcpyDtoHAsync_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy2DAsync_v2, handle_cuMemcpy2DAsync_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy3DAsync_v2, handle_cuMemcpy3DAsync_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy3DPeerAsync, handle_cuMemcpy3DPeerAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemExportToShareableHandle, handle_cuMemExportToShareableHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuMemImportFromShareableHandle, handle_cuMemImportFromShareableHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolSetAttribute, handle_cuMemPoolSetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolGetAttribute, handle_cuMemPoolGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolExportToShareableHandle, handle_cuMemPoolExportToShareableHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolImportFromShareableHandle, handle_cuMemPoolImportFromShareableHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuPointerGetAttribute, handle_cuPointerGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuMemRangeGetAttributes, handle_cuMemRangeGetAttributes, rpc_backend::cuda) \
  HANDLER(RPC_cuPointerSetAttribute, handle_cuPointerSetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuPointerGetAttributes, handle_cuPointerGetAttributes, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamWaitEvent, handle_cuStreamWaitEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamAddCallback, handle_cuStreamAddCallback, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamBeginCapture_v2, handle_cuStreamBeginCapture, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamEndCapture, handle_cuStreamEndCapture, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamSynchronize, handle_cuStreamSynchronize, rpc_backend::cuda) \
  HANDLER(RPC_cuEventRecord, handle_cuEventRecord, rpc_backend::cuda) \
  HANDLER(RPC_cuEventRecordWithFlags, handle_cuEventRecordWithFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuEventQuery, handle_cuEventQuery, rpc_backend::cuda) \
  HANDLER(RPC_cuEventSynchronize, handle_cuEventSynchronize, rpc_backend::cuda) \
  HANDLER(RPC_cuEventDestroy_v2, handle_cuEventDestroy_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchKernel, handle_cuLaunchKernel, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchKernelEx, handle_cuLaunchKernelEx, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchCooperativeKernel, handle_cuLaunchCooperativeKernel, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchHostFunc, handle_cuLaunchHostFunc, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddKernelNode_v2, handle_cuGraphAddKernelNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphKernelNodeGetParams_v2, handle_cuGraphKernelNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphKernelNodeSetParams_v2, handle_cuGraphKernelNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddMemcpyNode, handle_cuGraphAddMemcpyNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddMemsetNode, handle_cuGraphAddMemsetNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddHostNode, handle_cuGraphAddHostNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphHostNodeGetParams, handle_cuGraphHostNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphHostNodeSetParams, handle_cuGraphHostNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetGraphMemAttribute, handle_cuDeviceGetGraphMemAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceSetGraphMemAttribute, handle_cuDeviceSetGraphMemAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphClone, handle_cuGraphClone, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphInstantiate_v2, handle_cuGraphInstantiate_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphInstantiateWithFlags, handle_cuGraphInstantiateWithFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphInstantiateWithParams, handle_cuGraphInstantiateWithParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecKernelNodeSetParams_v2, handle_cuGraphExecKernelNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecHostNodeSetParams, handle_cuGraphExecHostNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphLaunch, handle_cuGraphLaunch, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecDestroy, handle_cuGraphExecDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphDestroy, handle_cuGraphDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyMaxPotentialBlockSize, handle_cuOccupancyMaxPotentialBlockSize, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyMaxPotentialBlockSizeWithFlags, handle_cuOccupancyMaxPotentialBlockSizeWithFlags, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuGetExportTableMetadata, handle_cuGetExportTableMetadata, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuPrivateGetModuleNode, handle_cuPrivateGetModuleNode, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineFunctionParamLayoutSnapshot, handle_lupineFunctionParamLayoutSnapshot, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineFunctionAttributeSnapshot, handle_lupineFunctionAttributeSnapshot, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineLibrarySnapshot, handle_lupineLibrarySnapshot, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineLibraryAttributeSnapshot, handle_lupineLibraryAttributeSnapshot, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuGraphConditionalHandleCreate, handle_cuGraphConditionalHandleCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddNode_v2, handle_cuGraphAddNode, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineEventQueryBatch, handle_lupineEventQueryBatch, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuStreamBeginCaptureToGraph, handle_cuStreamBeginCaptureToGraph, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamUpdateCaptureDependencies_v2, handle_cuStreamUpdateCaptureDependencies, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuStreamGetCaptureInfo_v3, handle_cuStreamGetCaptureInfo, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineManagedHostFlush, handle_lupineManagedHostFlush, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineDeviceSnapshot, handle_lupineDeviceSnapshot, rpc_backend::cuda) \
  HANDLER(RPC_cuInit, handle_cuInit, rpc_backend::cuda) \
  HANDLER(RPC_cuDriverGetVersion, handle_cuDriverGetVersion, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGet, handle_cuDeviceGet, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetCount, handle_cuDeviceGetCount, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetName, handle_cuDeviceGetName, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetUuid_v2, handle_cuDeviceGetUuid_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetLuid, handle_cuDeviceGetLuid, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceTotalMem_v2, handle_cuDeviceTotalMem_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetTexture1DLinearMaxWidth, handle_cuDeviceGetTexture1DLinearMaxWidth, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetAttribute, handle_cuDeviceGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceSetMemPool, handle_cuDeviceSetMemPool, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetMemPool, handle_cuDeviceGetMemPool, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetDefaultMemPool, handle_cuDeviceGetDefaultMemPool, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetExecAffinitySupport, handle_cuDeviceGetExecAffinitySupport, rpc_backend::cuda) \
  HANDLER(RPC_cuFlushGPUDirectRDMAWrites, handle_cuFlushGPUDirectRDMAWrites, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetProperties, handle_cuDeviceGetProperties, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceComputeCapability, handle_cuDeviceComputeCapability, rpc_backend::cuda) \
  HANDLER(RPC_cuDevicePrimaryCtxSetFlags_v2, handle_cuDevicePrimaryCtxSetFlags_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuDevicePrimaryCtxGetState, handle_cuDevicePrimaryCtxGetState, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxPushCurrent_v2, handle_cuCtxPushCurrent_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxPopCurrent_v2, handle_cuCtxPopCurrent_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxSetCurrent, handle_cuCtxSetCurrent, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetCurrent, handle_cuCtxGetCurrent, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetDevice, handle_cuCtxGetDevice, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetFlags, handle_cuCtxGetFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetId, handle_cuCtxGetId, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxSetLimit, handle_cuCtxSetLimit, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetLimit, handle_cuCtxGetLimit, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetCacheConfig, handle_cuCtxGetCacheConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxSetCacheConfig, handle_cuCtxSetCacheConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetApiVersion, handle_cuCtxGetApiVersion, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetStreamPriorityRange, handle_cuCtxGetStreamPriorityRange, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxResetPersistingL2Cache, handle_cuCtxResetPersistingL2Cache, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetExecAffinity, handle_cuCtxGetExecAffinity, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxGetSharedMemConfig, handle_cuCtxGetSharedMemConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxSetSharedMemConfig, handle_cuCtxSetSharedMemConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleLoadData, handle_cuModuleLoadData, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleLoadDataEx, handle_cuModuleLoadDataEx, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleLoadFatBinary, handle_cuModuleLoadFatBinary, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleUnload, handle_cuModuleUnload, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleGetLoadingMode, handle_cuModuleGetLoadingMode, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleGetFunction, handle_cuModuleGetFunction, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleGetTexRef, handle_cuModuleGetTexRef, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleGetSurfRef, handle_cuModuleGetSurfRef, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryLoadFromFile, handle_cuLibraryLoadFromFile, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryGetKernel, handle_cuLibraryGetKernel, rpc_backend::cuda) \
  HANDLER(RPC_cuKernelGetFunction, handle_cuKernelGetFunction, rpc_backend::cuda) \
  HANDLER(RPC_cuKernelGetLibrary, handle_cuKernelGetLibrary, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryGetGlobal, handle_cuLibraryGetGlobal, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryGetManaged, handle_cuLibraryGetManaged, rpc_backend::cuda) \
  HANDLER(RPC_cuLibraryGetUnifiedFunction, handle_cuLibraryGetUnifiedFunction, rpc_backend::cuda) \
  HANDLER(RPC_cuKernelGetAttribute, handle_cuKernelGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuKernelSetAttribute, handle_cuKernelSetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuKernelSetCacheConfig, handle_cuKernelSetCacheConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuKernelGetParamInfo, handle_cuKernelGetParamInfo, rpc_backend::cuda) \
  HANDLER(RPC_cuMemGetInfo_v2, handle_cuMemGetInfo_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAlloc_v2, handle_cuMemAlloc_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAllocPitch_v2, handle_cuMemAllocPitch_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemGetAddressRange_v2, handle_cuMemGetAddressRange_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAllocHost_v2, handle_cuMemAllocHost_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemHostGetDevicePointer_v2, handle_cuMemHostGetDevicePointer_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetByPCIBusId, handle_cuDeviceGetByPCIBusId, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetPCIBusId, handle_cuDeviceGetPCIBusId, rpc_backend::cuda) \
  HANDLER(RPC_cuIpcGetEventHandle, handle_cuIpcGetEventHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuIpcOpenEventHandle, handle_cuIpcOpenEventHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuIpcGetMemHandle, handle_cuIpcGetMemHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuIpcOpenMemHandle_v2, handle_cuIpcOpenMemHandle_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuIpcCloseMemHandle, handle_cuIpcCloseMemHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpy, handle_cuMemcpy, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyPeer, handle_cuMemcpyPeer, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyDtoD_v2, handle_cuMemcpyDtoD_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyDtoA_v2, handle_cuMemcpyDtoA_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyAtoD_v2, handle_cuMemcpyAtoD_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyAtoA_v2, handle_cuMemcpyAtoA_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyPeerAsync, handle_cuMemcpyPeerAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemcpyDtoDAsync_v2, handle_cuMemcpyDtoDAsync_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD8_v2, handle_cuMemsetD8_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD16_v2, handle_cuMemsetD16_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD32_v2, handle_cuMemsetD32_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD2D8_v2, handle_cuMemsetD2D8_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD2D16_v2, handle_cuMemsetD2D16_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD2D32_v2, handle_cuMemsetD2D32_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD8Async, handle_cuMemsetD8Async, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD16Async, handle_cuMemsetD16Async, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD32Async, handle_cuMemsetD32Async, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD2D8Async, handle_cuMemsetD2D8Async, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD2D16Async, handle_cuMemsetD2D16Async, rpc_backend::cuda) \
  HANDLER(RPC_cuMemsetD2D32Async, handle_cuMemsetD2D32Async, rpc_backend::cuda) \
  HANDLER(RPC_cuArrayCreate_v2, handle_cuArrayCreate_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuArrayGetDescriptor_v2, handle_cuArrayGetDescriptor_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuArrayGetSparseProperties, handle_cuArrayGetSparseProperties, rpc_backend::cuda) \
  HANDLER(RPC_cuMipmappedArrayGetSparseProperties, handle_cuMipmappedArrayGetSparseProperties, rpc_backend::cuda) \
  HANDLER(RPC_cuArrayGetMemoryRequirements, handle_cuArrayGetMemoryRequirements, rpc_backend::cuda) \
  HANDLER(RPC_cuMipmappedArrayGetMemoryRequirements, handle_cuMipmappedArrayGetMemoryRequirements, rpc_backend::cuda) \
  HANDLER(RPC_cuArrayGetPlane, handle_cuArrayGetPlane, rpc_backend::cuda) \
  HANDLER(RPC_cuArrayDestroy, handle_cuArrayDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuArray3DCreate_v2, handle_cuArray3DCreate_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuArray3DGetDescriptor_v2, handle_cuArray3DGetDescriptor_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuMipmappedArrayCreate, handle_cuMipmappedArrayCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuMipmappedArrayGetLevel, handle_cuMipmappedArrayGetLevel, rpc_backend::cuda) \
  HANDLER(RPC_cuMipmappedArrayDestroy, handle_cuMipmappedArrayDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAddressReserve, handle_cuMemAddressReserve, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAddressFree, handle_cuMemAddressFree, rpc_backend::cuda) \
  HANDLER(RPC_cuMemCreate, handle_cuMemCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuMemRelease, handle_cuMemRelease, rpc_backend::cuda) \
  HANDLER(RPC_cuMemMap, handle_cuMemMap, rpc_backend::cuda) \
  HANDLER(RPC_cuMemMapArrayAsync, handle_cuMemMapArrayAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemUnmap, handle_cuMemUnmap, rpc_backend::cuda) \
  HANDLER(RPC_cuMemSetAccess, handle_cuMemSetAccess, rpc_backend::cuda) \
  HANDLER(RPC_cuMemGetAccess, handle_cuMemGetAccess, rpc_backend::cuda) \
  HANDLER(RPC_cuMemGetAllocationGranularity, handle_cuMemGetAllocationGranularity, rpc_backend::cuda) \
  HANDLER(RPC_cuMemGetAllocationPropertiesFromHandle, handle_cuMemGetAllocationPropertiesFromHandle, rpc_backend::cuda) \
  HANDLER(RPC_cuMemFreeAsync, handle_cuMemFreeAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAllocAsync, handle_cuMemAllocAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolTrimTo, handle_cuMemPoolTrimTo, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolSetAccess, handle_cuMemPoolSetAccess, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolGetAccess, handle_cuMemPoolGetAccess, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolCreate, handle_cuMemPoolCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolDestroy, handle_cuMemPoolDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAllocFromPoolAsync, handle_cuMemAllocFromPoolAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolExportPointer, handle_cuMemPoolExportPointer, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPoolImportPointer, handle_cuMemPoolImportPointer, rpc_backend::cuda) \
  HANDLER(RPC_cuMemRangeGetAttribute, handle_cuMemRangeGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamCreate, handle_cuStreamCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamCreateWithPriority, handle_cuStreamCreateWithPriority, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamGetPriority, handle_cuStreamGetPriority, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamGetFlags, handle_cuStreamGetFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamGetId, handle_cuStreamGetId, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamGetCtx, handle_cuStreamGetCtx, rpc_backend::cuda) \
  HANDLER(RPC_cuThreadExchangeStreamCaptureMode, handle_cuThreadExchangeStreamCaptureMode, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamIsCapturing, handle_cuStreamIsCapturing, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamAttachMemAsync, handle_cuStreamAttachMemAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamQuery, handle_cuStreamQuery, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamDestroy_v2, handle_cuStreamDestroy_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamCopyAttributes, handle_cuStreamCopyAttributes, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamGetAttribute, handle_cuStreamGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamSetAttribute, handle_cuStreamSetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuEventCreate, handle_cuEventCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuEventElapsedTime_v2, handle_cuEventElapsedTime_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuImportExternalMemory, handle_cuImportExternalMemory, rpc_backend::cuda) \
  HANDLER(RPC_cuExternalMemoryGetMappedBuffer, handle_cuExternalMemoryGetMappedBuffer, rpc_backend::cuda) \
  HANDLER(RPC_cuExternalMemoryGetMappedMipmappedArray, handle_cuExternalMemoryGetMappedMipmappedArray, rpc_backend::cuda) \
  HANDLER(RPC_cuDestroyExternalMemory, handle_cuDestroyExternalMemory, rpc_backend::cuda) \
  HANDLER(RPC_cuImportExternalSemaphore, handle_cuImportExternalSemaphore, rpc_backend::cuda) \
  HANDLER(RPC_cuSignalExternalSemaphoresAsync, handle_cuSignalExternalSemaphoresAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuWaitExternalSemaphoresAsync, handle_cuWaitExternalSemaphoresAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuDestroyExternalSemaphore, handle_cuDestroyExternalSemaphore, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamWaitValue32_v2, handle_cuStreamWaitValue32_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamWaitValue64_v2, handle_cuStreamWaitValue64_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamWriteValue32_v2, handle_cuStreamWriteValue32_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamWriteValue64_v2, handle_cuStreamWriteValue64_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamBatchMemOp_v2, handle_cuStreamBatchMemOp_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncGetAttribute, handle_cuFuncGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncSetAttribute, handle_cuFuncSetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncSetCacheConfig, handle_cuFuncSetCacheConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncGetModule, handle_cuFuncGetModule, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncGetParamInfo, handle_cuFuncGetParamInfo, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncSetBlockShape, handle_cuFuncSetBlockShape, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncSetSharedSize, handle_cuFuncSetSharedSize, rpc_backend::cuda) \
  HANDLER(RPC_cuParamSetSize, handle_cuParamSetSize, rpc_backend::cuda) \
  HANDLER(RPC_cuParamSeti, handle_cuParamSeti, rpc_backend::cuda) \
  HANDLER(RPC_cuParamSetf, handle_cuParamSetf, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunch, handle_cuLaunch, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchGrid, handle_cuLaunchGrid, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchGridAsync, handle_cuLaunchGridAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuLaunchCooperativeKernelMultiDevice, handle_cuLaunchCooperativeKernelMultiDevice, rpc_backend::cuda) \
  HANDLER(RPC_cuParamSetTexRef, handle_cuParamSetTexRef, rpc_backend::cuda) \
  HANDLER(RPC_cuFuncSetSharedMemConfig, handle_cuFuncSetSharedMemConfig, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphCreate, handle_cuGraphCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphMemcpyNodeGetParams, handle_cuGraphMemcpyNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphMemcpyNodeSetParams, handle_cuGraphMemcpyNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphMemsetNodeGetParams, handle_cuGraphMemsetNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphMemsetNodeSetParams, handle_cuGraphMemsetNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddChildGraphNode, handle_cuGraphAddChildGraphNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphChildGraphNodeGetGraph, handle_cuGraphChildGraphNodeGetGraph, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddEmptyNode, handle_cuGraphAddEmptyNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddEventRecordNode, handle_cuGraphAddEventRecordNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphEventRecordNodeGetEvent, handle_cuGraphEventRecordNodeGetEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphEventRecordNodeSetEvent, handle_cuGraphEventRecordNodeSetEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddEventWaitNode, handle_cuGraphAddEventWaitNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphEventWaitNodeGetEvent, handle_cuGraphEventWaitNodeGetEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphEventWaitNodeSetEvent, handle_cuGraphEventWaitNodeSetEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddExternalSemaphoresSignalNode, handle_cuGraphAddExternalSemaphoresSignalNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExternalSemaphoresSignalNodeGetParams, handle_cuGraphExternalSemaphoresSignalNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExternalSemaphoresSignalNodeSetParams, handle_cuGraphExternalSemaphoresSignalNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddExternalSemaphoresWaitNode, handle_cuGraphAddExternalSemaphoresWaitNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExternalSemaphoresWaitNodeGetParams, handle_cuGraphExternalSemaphoresWaitNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExternalSemaphoresWaitNodeSetParams, handle_cuGraphExternalSemaphoresWaitNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddBatchMemOpNode, handle_cuGraphAddBatchMemOpNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphBatchMemOpNodeGetParams, handle_cuGraphBatchMemOpNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphBatchMemOpNodeSetParams, handle_cuGraphBatchMemOpNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecBatchMemOpNodeSetParams, handle_cuGraphExecBatchMemOpNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddMemAllocNode, handle_cuGraphAddMemAllocNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphMemAllocNodeGetParams, handle_cuGraphMemAllocNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddMemFreeNode, handle_cuGraphAddMemFreeNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphMemFreeNodeGetParams, handle_cuGraphMemFreeNodeGetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGraphMemTrim, handle_cuDeviceGraphMemTrim, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphNodeFindInClone, handle_cuGraphNodeFindInClone, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphNodeGetType, handle_cuGraphNodeGetType, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphGetNodes, handle_cuGraphGetNodes, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphGetRootNodes, handle_cuGraphGetRootNodes, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphGetEdges_v2, handle_cuGraphGetEdges_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphNodeGetDependencies_v2, handle_cuGraphNodeGetDependencies_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphNodeGetDependentNodes_v2, handle_cuGraphNodeGetDependentNodes_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphDestroyNode, handle_cuGraphDestroyNode, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecGetFlags, handle_cuGraphExecGetFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecMemcpyNodeSetParams, handle_cuGraphExecMemcpyNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecMemsetNodeSetParams, handle_cuGraphExecMemsetNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecChildGraphNodeSetParams, handle_cuGraphExecChildGraphNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecEventRecordNodeSetEvent, handle_cuGraphExecEventRecordNodeSetEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecEventWaitNodeSetEvent, handle_cuGraphExecEventWaitNodeSetEvent, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams, handle_cuGraphExecExternalSemaphoresSignalNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams, handle_cuGraphExecExternalSemaphoresWaitNodeSetParams, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphNodeSetEnabled, handle_cuGraphNodeSetEnabled, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphNodeGetEnabled, handle_cuGraphNodeGetEnabled, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphUpload, handle_cuGraphUpload, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecUpdate_v2, handle_cuGraphExecUpdate_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphKernelNodeCopyAttributes, handle_cuGraphKernelNodeCopyAttributes, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphKernelNodeGetAttribute, handle_cuGraphKernelNodeGetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphKernelNodeSetAttribute, handle_cuGraphKernelNodeSetAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphDebugDotPrint, handle_cuGraphDebugDotPrint, rpc_backend::cuda) \
  HANDLER(RPC_cuUserObjectRetain, handle_cuUserObjectRetain, rpc_backend::cuda) \
  HANDLER(RPC_cuUserObjectRelease, handle_cuUserObjectRelease, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphRetainUserObject, handle_cuGraphRetainUserObject, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphReleaseUserObject, handle_cuGraphReleaseUserObject, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor, handle_cuOccupancyMaxActiveBlocksPerMultiprocessor, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyAvailableDynamicSMemPerBlock, handle_cuOccupancyAvailableDynamicSMemPerBlock, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyMaxPotentialClusterSize, handle_cuOccupancyMaxPotentialClusterSize, rpc_backend::cuda) \
  HANDLER(RPC_cuOccupancyMaxActiveClusters, handle_cuOccupancyMaxActiveClusters, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetArray, handle_cuTexRefSetArray, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetMipmappedArray, handle_cuTexRefSetMipmappedArray, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetAddress_v2, handle_cuTexRefSetAddress_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetAddress2D_v3, handle_cuTexRefSetAddress2D_v3, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetFormat, handle_cuTexRefSetFormat, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetAddressMode, handle_cuTexRefSetAddressMode, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetFilterMode, handle_cuTexRefSetFilterMode, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetMipmapFilterMode, handle_cuTexRefSetMipmapFilterMode, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetMipmapLevelBias, handle_cuTexRefSetMipmapLevelBias, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetMipmapLevelClamp, handle_cuTexRefSetMipmapLevelClamp, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetMaxAnisotropy, handle_cuTexRefSetMaxAnisotropy, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetBorderColor, handle_cuTexRefSetBorderColor, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefSetFlags, handle_cuTexRefSetFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetAddress_v2, handle_cuTexRefGetAddress_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetArray, handle_cuTexRefGetArray, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetMipmappedArray, handle_cuTexRefGetMipmappedArray, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetAddressMode, handle_cuTexRefGetAddressMode, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetFilterMode, handle_cuTexRefGetFilterMode, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetFormat, handle_cuTexRefGetFormat, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetMipmapFilterMode, handle_cuTexRefGetMipmapFilterMode, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetMipmapLevelBias, handle_cuTexRefGetMipmapLevelBias, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetMipmapLevelClamp, handle_cuTexRefGetMipmapLevelClamp, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetMaxAnisotropy, handle_cuTexRefGetMaxAnisotropy, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetBorderColor, handle_cuTexRefGetBorderColor, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefGetFlags, handle_cuTexRefGetFlags, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefCreate, handle_cuTexRefCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuTexRefDestroy, handle_cuTexRefDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuSurfRefSetArray, handle_cuSurfRefSetArray, rpc_backend::cuda) \
  HANDLER(RPC_cuSurfRefGetArray, handle_cuSurfRefGetArray, rpc_backend::cuda) \
  HANDLER(RPC_cuTexObjectCreate, handle_cuTexObjectCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuTexObjectDestroy, handle_cuTexObjectDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuTexObjectGetResourceDesc, handle_cuTexObjectGetResourceDesc, rpc_backend::cuda) \
  HANDLER(RPC_cuTexObjectGetTextureDesc, handle_cuTexObjectGetTextureDesc, rpc_backend::cuda) \
  HANDLER(RPC_cuTexObjectGetResourceViewDesc, handle_cuTexObjectGetResourceViewDesc, rpc_backend::cuda) \
  HANDLER(RPC_cuSurfObjectCreate, handle_cuSurfObjectCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuSurfObjectDestroy, handle_cuSurfObjectDestroy, rpc_backend::cuda) \
  HANDLER(RPC_cuSurfObjectGetResourceDesc, handle_cuSurfObjectGetResourceDesc, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceCanAccessPeer, handle_cuDeviceCanAccessPeer, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxEnablePeerAccess, handle_cuCtxEnablePeerAccess, rpc_backend::cuda) \
  HANDLER(RPC_cuCtxDisablePeerAccess, handle_cuCtxDisablePeerAccess, rpc_backend::cuda) \
  HANDLER(RPC_cuDeviceGetP2PAttribute, handle_cuDeviceGetP2PAttribute, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsUnregisterResource, handle_cuGraphicsUnregisterResource, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsSubResourceGetMappedArray, handle_cuGraphicsSubResourceGetMappedArray, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsResourceGetMappedMipmappedArray, handle_cuGraphicsResourceGetMappedMipmappedArray, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsResourceGetMappedPointer_v2, handle_cuGraphicsResourceGetMappedPointer_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsResourceSetMapFlags_v2, handle_cuGraphicsResourceSetMapFlags_v2, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsMapResources, handle_cuGraphicsMapResources, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphicsUnmapResources, handle_cuGraphicsUnmapResources, rpc_backend::cuda) \
  HANDLER(RPC_cuMemPrefetchAsync, handle_cuMemPrefetchAsync, rpc_backend::cuda) \
  HANDLER(RPC_cuMemAdvise, handle_cuMemAdvise, rpc_backend::cuda)
#define LUPINE_NVML_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_nvmlDeviceGetComputeRunningProcesses, handle_nvmlDeviceGetComputeRunningProcesses, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetComputeRunningProcesses_v2, handle_nvmlDeviceGetComputeRunningProcesses_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetGraphicsRunningProcesses, handle_nvmlDeviceGetGraphicsRunningProcesses, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetGraphicsRunningProcesses_v2, handle_nvmlDeviceGetGraphicsRunningProcesses_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMPSComputeRunningProcesses, handle_nvmlDeviceGetMPSComputeRunningProcesses, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2, handle_nvmlDeviceGetMPSComputeRunningProcesses_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlInit_v2, handle_nvmlInit_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlInitWithFlags, handle_nvmlInitWithFlags, rpc_backend::nvml) \
  HANDLER(RPC_nvmlShutdown, handle_nvmlShutdown, rpc_backend::nvml) \
  HANDLER(RPC_nvmlSystemGetDriverVersion, handle_nvmlSystemGetDriverVersion, rpc_backend::nvml) \
  HANDLER(RPC_nvmlSystemGetNVMLVersion, handle_nvmlSystemGetNVMLVersion, rpc_backend::nvml) \
  HANDLER(RPC_nvmlSystemGetCudaDriverVersion, handle_nvmlSystemGetCudaDriverVersion, rpc_backend::nvml) \
  HANDLER(RPC_nvmlSystemGetCudaDriverVersion_v2, handle_nvmlSystemGetCudaDriverVersion_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetCount_v2, handle_nvmlDeviceGetCount_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetHandleByIndex_v2, handle_nvmlDeviceGetHandleByIndex_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetHandleByUUID, handle_nvmlDeviceGetHandleByUUID, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetHandleByPciBusId_v2, handle_nvmlDeviceGetHandleByPciBusId_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetName, handle_nvmlDeviceGetName, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetUUID, handle_nvmlDeviceGetUUID, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetIndex, handle_nvmlDeviceGetIndex, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMinorNumber, handle_nvmlDeviceGetMinorNumber, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPciInfo_v3, handle_nvmlDeviceGetPciInfo_v3, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMemoryInfo, handle_nvmlDeviceGetMemoryInfo, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetUtilizationRates, handle_nvmlDeviceGetUtilizationRates, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetTemperature, handle_nvmlDeviceGetTemperature, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPowerUsage, handle_nvmlDeviceGetPowerUsage, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPowerManagementLimit, handle_nvmlDeviceGetPowerManagementLimit, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetClockInfo, handle_nvmlDeviceGetClockInfo, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMaxClockInfo, handle_nvmlDeviceGetMaxClockInfo, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPerformanceState, handle_nvmlDeviceGetPerformanceState, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetComputeMode, handle_nvmlDeviceGetComputeMode, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPersistenceMode, handle_nvmlDeviceGetPersistenceMode, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetFanSpeed, handle_nvmlDeviceGetFanSpeed, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetBrand, handle_nvmlDeviceGetBrand, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetVbiosVersion, handle_nvmlDeviceGetVbiosVersion, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetSerial, handle_nvmlDeviceGetSerial, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetBoardPartNumber, handle_nvmlDeviceGetBoardPartNumber, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetDisplayMode, handle_nvmlDeviceGetDisplayMode, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetDisplayActive, handle_nvmlDeviceGetDisplayActive, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetCurrPcieLinkGeneration, handle_nvmlDeviceGetCurrPcieLinkGeneration, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetCurrPcieLinkWidth, handle_nvmlDeviceGetCurrPcieLinkWidth, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMaxPcieLinkGeneration, handle_nvmlDeviceGetMaxPcieLinkGeneration, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMaxPcieLinkWidth, handle_nvmlDeviceGetMaxPcieLinkWidth, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPcieThroughput, handle_nvmlDeviceGetPcieThroughput, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetPcieReplayCounter, handle_nvmlDeviceGetPcieReplayCounter, rpc_backend::nvml) \
  HANDLER(RPC_nvmlEventSetCreate, handle_nvmlEventSetCreate, rpc_backend::nvml) \
  HANDLER(RPC_nvmlEventSetFree, handle_nvmlEventSetFree, rpc_backend::nvml) \
  HANDLER(RPC_nvmlEventSetWait_v2, handle_nvmlEventSetWait_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceRegisterEvents, handle_nvmlDeviceRegisterEvents, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMaxMigDeviceCount, handle_nvmlDeviceGetMaxMigDeviceCount, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetTotalEccErrors, handle_nvmlDeviceGetTotalEccErrors, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetDetailedEccErrors, handle_nvmlDeviceGetDetailedEccErrors, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMemoryErrorCounter, handle_nvmlDeviceGetMemoryErrorCounter, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetEccMode, handle_nvmlDeviceGetEccMode, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetTemperatureV, handle_nvmlDeviceGetTemperatureV, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetEnforcedPowerLimit, handle_nvmlDeviceGetEnforcedPowerLimit, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMemoryInfo_v2, handle_nvmlDeviceGetMemoryInfo_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetMigMode, handle_nvmlDeviceGetMigMode, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetVirtualizationMode, handle_nvmlDeviceGetVirtualizationMode, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceIsMigDeviceHandle, handle_nvmlDeviceIsMigDeviceHandle, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetNvLinkRemoteDeviceType, handle_nvmlDeviceGetNvLinkRemoteDeviceType, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2, handle_nvmlDeviceGetNvLinkRemotePciInfo_v2, rpc_backend::nvml) \
  HANDLER(RPC_nvmlDeviceGetCudaComputeCapability, handle_nvmlDeviceGetCudaComputeCapability, rpc_backend::nvml)
#define LUPINE_HIP_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_hipInit, handle_hipInit, rpc_backend::hip) \
  HANDLER(RPC_hipGetDeviceCount, handle_hipGetDeviceCount, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGet, handle_hipDeviceGet, rpc_backend::hip) \
  HANDLER(RPC_hipGetDevicePropertiesR0600, handle_hipGetDevicePropertiesR0600, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetName, handle_hipDeviceGetName, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceTotalMem, handle_hipDeviceTotalMem, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetAttribute, handle_hipDeviceGetAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipDriverGetVersion, handle_hipDriverGetVersion, rpc_backend::hip) \
  HANDLER(RPC_hipRuntimeGetVersion, handle_hipRuntimeGetVersion, rpc_backend::hip)
// clang-format on

#define LUPINE_DECLARE_HANDLER(operation, handler, backend)                    \
  int handler(conn_t *conn);
#ifdef LUPINE_BUILD_CUDA_BACKEND
LUPINE_CUDA_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUDA_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cuCtxSynchronize_v2, handle_cuCtxSynchronize_v2,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cuTensorMapEncodeTiled,
                       handle_cuTensorMapEncodeTiled, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cuCtxGetDevice_v2, handle_cuCtxGetDevice_v2,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12050
LUPINE_DECLARE_HANDLER(RPC_cuCtxRecordEvent, handle_cuCtxRecordEvent,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12050
LUPINE_DECLARE_HANDLER(RPC_cuCtxWaitEvent, handle_cuCtxWaitEvent,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12030
LUPINE_DECLARE_HANDLER(RPC_cuKernelGetName, handle_cuKernelGetName,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
LUPINE_DECLARE_HANDLER(RPC_cuMemPrefetchAsync_v2, handle_cuMemPrefetchAsync_v2,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
LUPINE_DECLARE_HANDLER(RPC_cuMemAdvise_v2, handle_cuMemAdvise_v2,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12030
LUPINE_DECLARE_HANDLER(RPC_cuFuncGetName, handle_cuFuncGetName,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuGraphNodeGetContainingGraph,
                       handle_cuGraphNodeGetContainingGraph, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuGraphNodeGetLocalId, handle_cuGraphNodeGetLocalId,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuGraphNodeGetToolsId, handle_cuGraphNodeGetToolsId,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuGraphGetId, handle_cuGraphGetId, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuGraphExecGetId, handle_cuGraphExecGetId,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
LUPINE_DECLARE_HANDLER(RPC_cuGraphNodeSetParams, handle_cuGraphNodeSetParams,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
LUPINE_DECLARE_HANDLER(RPC_cuGraphExecNodeSetParams,
                       handle_cuGraphExecNodeSetParams, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12010
LUPINE_DECLARE_HANDLER(RPC_cuCoredumpGetAttributeGlobal,
                       handle_cuCoredumpGetAttributeGlobal, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12010
LUPINE_DECLARE_HANDLER(RPC_cuCoredumpSetAttributeGlobal,
                       handle_cuCoredumpSetAttributeGlobal, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxCreate, handle_cuGreenCtxCreate,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxDestroy, handle_cuGreenCtxDestroy,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuCtxFromGreenCtx, handle_cuCtxFromGreenCtx,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuDeviceGetDevResource,
                       handle_cuDeviceGetDevResource, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuCtxGetDevResource, handle_cuCtxGetDevResource,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxGetDevResource,
                       handle_cuGreenCtxGetDevResource, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuDevSmResourceSplitByCount,
                       handle_cuDevSmResourceSplitByCount, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuDevSmResourceSplit, handle_cuDevSmResourceSplit,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuDevResourceGenerateDesc,
                       handle_cuDevResourceGenerateDesc, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxRecordEvent, handle_cuGreenCtxRecordEvent,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxWaitEvent, handle_cuGreenCtxWaitEvent,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
LUPINE_DECLARE_HANDLER(RPC_cuStreamGetGreenCtx, handle_cuStreamGetGreenCtx,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12050
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxStreamCreate,
                       handle_cuGreenCtxStreamCreate, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cuGreenCtxGetId, handle_cuGreenCtxGetId,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
LUPINE_DECLARE_HANDLER(RPC_cuStreamGetDevResource,
                       handle_cuStreamGetDevResource, rpc_backend::cuda)
#endif
#endif
#ifdef LUPINE_BUILD_NVML_BACKEND
LUPINE_NVML_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)

#endif
#ifdef LUPINE_BUILD_HIP_BACKEND
LUPINE_HIP_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)

#endif
#undef LUPINE_DECLARE_HANDLER

const rpc_handler_registry &lupine_rpc_handlers() {
#define LUPINE_REGISTER_HANDLER(operation, handler, backend)                   \
  {operation, {handler, backend}},
  // clang-format treats registrations separated by preprocessor guards as a
  // single continued expression, adding another indentation level per guard.
  // clang-format off
  static const rpc_handler_registry handlers = {
#ifdef LUPINE_BUILD_CUDA_BACKEND
      LUPINE_CUDA_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUDA_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cuCtxSynchronize_v2, handle_cuCtxSynchronize_v2, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cuTensorMapEncodeTiled, handle_cuTensorMapEncodeTiled, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cuCtxGetDevice_v2, handle_cuCtxGetDevice_v2, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12050
      LUPINE_REGISTER_HANDLER(RPC_cuCtxRecordEvent, handle_cuCtxRecordEvent, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12050
      LUPINE_REGISTER_HANDLER(RPC_cuCtxWaitEvent, handle_cuCtxWaitEvent, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12030
      LUPINE_REGISTER_HANDLER(RPC_cuKernelGetName, handle_cuKernelGetName, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
      LUPINE_REGISTER_HANDLER(RPC_cuMemPrefetchAsync_v2, handle_cuMemPrefetchAsync_v2, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
      LUPINE_REGISTER_HANDLER(RPC_cuMemAdvise_v2, handle_cuMemAdvise_v2, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12030
      LUPINE_REGISTER_HANDLER(RPC_cuFuncGetName, handle_cuFuncGetName, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuGraphNodeGetContainingGraph, handle_cuGraphNodeGetContainingGraph, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuGraphNodeGetLocalId, handle_cuGraphNodeGetLocalId, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuGraphNodeGetToolsId, handle_cuGraphNodeGetToolsId, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuGraphGetId, handle_cuGraphGetId, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuGraphExecGetId, handle_cuGraphExecGetId, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
      LUPINE_REGISTER_HANDLER(RPC_cuGraphNodeSetParams, handle_cuGraphNodeSetParams, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12020
      LUPINE_REGISTER_HANDLER(RPC_cuGraphExecNodeSetParams, handle_cuGraphExecNodeSetParams, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12010
      LUPINE_REGISTER_HANDLER(RPC_cuCoredumpGetAttributeGlobal, handle_cuCoredumpGetAttributeGlobal, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12010
      LUPINE_REGISTER_HANDLER(RPC_cuCoredumpSetAttributeGlobal, handle_cuCoredumpSetAttributeGlobal, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxCreate, handle_cuGreenCtxCreate, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxDestroy, handle_cuGreenCtxDestroy, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuCtxFromGreenCtx, handle_cuCtxFromGreenCtx, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuDeviceGetDevResource, handle_cuDeviceGetDevResource, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuCtxGetDevResource, handle_cuCtxGetDevResource, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxGetDevResource, handle_cuGreenCtxGetDevResource, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuDevSmResourceSplitByCount, handle_cuDevSmResourceSplitByCount, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuDevSmResourceSplit, handle_cuDevSmResourceSplit, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuDevResourceGenerateDesc, handle_cuDevResourceGenerateDesc, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxRecordEvent, handle_cuGreenCtxRecordEvent, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxWaitEvent, handle_cuGreenCtxWaitEvent, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12040
      LUPINE_REGISTER_HANDLER(RPC_cuStreamGetGreenCtx, handle_cuStreamGetGreenCtx, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12050
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxStreamCreate, handle_cuGreenCtxStreamCreate, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cuGreenCtxGetId, handle_cuGreenCtxGetId, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 13010
      LUPINE_REGISTER_HANDLER(RPC_cuStreamGetDevResource, handle_cuStreamGetDevResource, rpc_backend::cuda)
#endif
#endif
#ifdef LUPINE_BUILD_NVML_BACKEND
      LUPINE_NVML_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)

#endif
#ifdef LUPINE_BUILD_HIP_BACKEND
      LUPINE_HIP_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)

#endif
  };
  // clang-format on
#undef LUPINE_REGISTER_HANDLER
  return handlers;
}

#undef LUPINE_CUDA_RPC_HANDLERS
#undef LUPINE_NVML_RPC_HANDLERS
#undef LUPINE_HIP_RPC_HANDLERS
