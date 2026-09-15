#include "rpc_server.h"

#ifdef LUPINE_BUILD_CUDA_BACKEND
#include <cuda.h>
#endif
#ifdef LUPINE_BUILD_CUDART_BACKEND
#include <cuda_runtime_api.h>
#endif
#ifdef LUPINE_BUILD_CUBLAS_BACKEND
#include <cublasLt.h>
#include <cublas_v2.h>
#endif
#ifdef LUPINE_BUILD_CUFFT_BACKEND
#include <cufftXt.h>
#endif
#ifdef LUPINE_BUILD_CUDNN_BACKEND
#include <cudnn.h>
#endif
#ifdef LUPINE_BUILD_CURAND_BACKEND
#include <curand.h>
#endif
#ifdef LUPINE_BUILD_CUSPARSE_BACKEND
#include <cusparse.h>
#endif
#ifdef LUPINE_BUILD_CUSOLVER_BACKEND
#include <cusolver_common.h>
#endif
#ifdef LUPINE_BUILD_NVRTC_BACKEND
#include <nvrtc.h>
#endif
#ifdef LUPINE_BUILD_NCCL_BACKEND
#include <nccl.h>
#endif
#ifdef LUPINE_BUILD_NVJITLINK_BACKEND
#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
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
  HANDLER(RPC_cuModuleLoadData, handle_cuModuleLoadData, rpc_backend::cuda) \
  HANDLER(RPC_cuModuleLoadDataEx, handle_cuModuleLoadDataEx, rpc_backend::cuda) \
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
  HANDLER(LUPINE_RPC_cuGraphConditionalHandleCreate, handle_cuGraphConditionalHandleCreate, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphAddNode_v2, handle_cuGraphAddNode, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineEventQueryBatch, handle_lupineEventQueryBatch, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineStreamPoolInit, handle_lupineStreamPoolInit, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuStreamBeginCaptureToGraph, handle_cuStreamBeginCaptureToGraph, rpc_backend::cuda) \
  HANDLER(RPC_cuStreamUpdateCaptureDependencies_v2, handle_cuStreamUpdateCaptureDependencies, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_cuStreamGetCaptureInfo_v3, handle_cuStreamGetCaptureInfo, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineManagedHostFlush, handle_lupineManagedHostFlush, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineMemcpyDtoHAsyncPinned, handle_lupineMemcpyDtoHAsyncPinned, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineDeviceSnapshot, handle_lupineDeviceSnapshot, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineBulkChunk, handle_lupineBulkChunk, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineMemcpyHtoDBulk, handle_lupineMemcpyHtoDBulk, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineBulkRead, handle_lupineBulkRead, rpc_backend::cuda) \
  HANDLER(LUPINE_RPC_lupineMemcpyDtoHBulk, handle_lupineMemcpyDtoHBulk, rpc_backend::cuda) \
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
  HANDLER(RPC_cuMemAdvise, handle_cuMemAdvise, rpc_backend::cuda) \
  HANDLER(RPC_cuGraphExecUpdate, handle_cuGraphExecUpdate, rpc_backend::cuda)
#define LUPINE_CUDART_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cudaDeviceSynchronize, handle_cudaDeviceSynchronize, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventDestroy, handle_cudaEventDestroy, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventQuery, handle_cudaEventQuery, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventRecord, handle_cudaEventRecord, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventRecordWithFlags, handle_cudaEventRecordWithFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventSynchronize, handle_cudaEventSynchronize, rpc_backend::cudart) \
  HANDLER(RPC_cudaFree, handle_cudaFree, rpc_backend::cudart) \
  HANDLER(RPC_cudaFreeHost, handle_cudaFreeHost, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetDriverEntryPoint, handle_cudaGetDriverEntryPoint, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetErrorName, handle_cudaGetErrorName, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetErrorString, handle_cudaGetErrorString, rpc_backend::cudart) \
  HANDLER(RPC_cudaHostAlloc, handle_cudaHostAlloc, rpc_backend::cudart) \
  HANDLER(RPC_cudaHostGetFlags, handle_cudaHostGetFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaHostRegister, handle_cudaHostRegister, rpc_backend::cudart) \
  HANDLER(RPC_cudaHostUnregister, handle_cudaHostUnregister, rpc_backend::cudart) \
  HANDLER(RPC_cudaLaunchCooperativeKernel, handle_cudaLaunchCooperativeKernel, rpc_backend::cudart) \
  HANDLER(RPC_cudaLaunchHostFunc, handle_cudaLaunchHostFunc, rpc_backend::cudart) \
  HANDLER(RPC_cudaLaunchKernel, handle_cudaLaunchKernel, rpc_backend::cudart) \
  HANDLER(RPC_cudaLaunchKernelExC, handle_cudaLaunchKernelExC, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocHost, handle_cudaMallocHost, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocManaged, handle_cudaMallocManaged, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemAdvise, handle_cudaMemAdvise, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolExportToShareableHandle, handle_cudaMemPoolExportToShareableHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolImportFromShareableHandle, handle_cudaMemPoolImportFromShareableHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPrefetchAsync, handle_cudaMemPrefetchAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamAddCallback, handle_cudaStreamAddCallback, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamBeginCapture, handle_cudaStreamBeginCapture, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamEndCapture, handle_cudaStreamEndCapture, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamGetCaptureInfo, handle_cudaStreamGetCaptureInfo, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamQuery, handle_cudaStreamQuery, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamSynchronize, handle_cudaStreamSynchronize, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamUpdateCaptureDependencies, handle_cudaStreamUpdateCaptureDependencies, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddHostNode, handle_cudaGraphAddHostNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddKernelNode, handle_cudaGraphAddKernelNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphClone, handle_cudaGraphClone, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphDestroy, handle_cudaGraphDestroy, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecDestroy, handle_cudaGraphExecDestroy, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecKernelNodeSetParams, handle_cudaGraphExecKernelNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecUpdate, handle_cudaGraphExecUpdate, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphInstantiate, handle_cudaGraphInstantiate, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphInstantiateWithFlags, handle_cudaGraphInstantiateWithFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphKernelNodeSetParams, handle_cudaGraphKernelNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphLaunch, handle_cudaGraphLaunch, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterFunction, handle___cudaRegisterFunction, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterVar, handle___cudaRegisterVar, rpc_backend::cudart) \
  HANDLER(RPC___cudaUnregisterFatBinary, handle___cudaUnregisterFatBinary, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterFatBinary, handle___cudaRegisterFatBinary, rpc_backend::cudart) \
  HANDLER(RPC_lupineCudartFuncParamLayout, handle_lupineCudartFuncParamLayout, rpc_backend::cudart) \
  HANDLER(RPC_cudaArrayGetInfo, handle_cudaArrayGetInfo, rpc_backend::cudart) \
  HANDLER(RPC_cudaArrayGetMemoryRequirements, handle_cudaArrayGetMemoryRequirements, rpc_backend::cudart) \
  HANDLER(RPC_cudaArrayGetPlane, handle_cudaArrayGetPlane, rpc_backend::cudart) \
  HANDLER(RPC_cudaArrayGetSparseProperties, handle_cudaArrayGetSparseProperties, rpc_backend::cudart) \
  HANDLER(RPC_cudaChooseDevice, handle_cudaChooseDevice, rpc_backend::cudart) \
  HANDLER(RPC_cudaCreateChannelDesc, handle_cudaCreateChannelDesc, rpc_backend::cudart) \
  HANDLER(RPC_cudaCreateSurfaceObject, handle_cudaCreateSurfaceObject, rpc_backend::cudart) \
  HANDLER(RPC_cudaCreateTextureObject, handle_cudaCreateTextureObject, rpc_backend::cudart) \
  HANDLER(RPC_cudaCtxResetPersistingL2Cache, handle_cudaCtxResetPersistingL2Cache, rpc_backend::cudart) \
  HANDLER(RPC_cudaDestroyExternalMemory, handle_cudaDestroyExternalMemory, rpc_backend::cudart) \
  HANDLER(RPC_cudaDestroyExternalSemaphore, handle_cudaDestroyExternalSemaphore, rpc_backend::cudart) \
  HANDLER(RPC_cudaDestroySurfaceObject, handle_cudaDestroySurfaceObject, rpc_backend::cudart) \
  HANDLER(RPC_cudaDestroyTextureObject, handle_cudaDestroyTextureObject, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceCanAccessPeer, handle_cudaDeviceCanAccessPeer, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceDisablePeerAccess, handle_cudaDeviceDisablePeerAccess, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceEnablePeerAccess, handle_cudaDeviceEnablePeerAccess, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceFlushGPUDirectRDMAWrites, handle_cudaDeviceFlushGPUDirectRDMAWrites, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetAttribute, handle_cudaDeviceGetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetByPCIBusId, handle_cudaDeviceGetByPCIBusId, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetCacheConfig, handle_cudaDeviceGetCacheConfig, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetDefaultMemPool, handle_cudaDeviceGetDefaultMemPool, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetGraphMemAttribute, handle_cudaDeviceGetGraphMemAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetLimit, handle_cudaDeviceGetLimit, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetMemPool, handle_cudaDeviceGetMemPool, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetP2PAttribute, handle_cudaDeviceGetP2PAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetPCIBusId, handle_cudaDeviceGetPCIBusId, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetStreamPriorityRange, handle_cudaDeviceGetStreamPriorityRange, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGetTexture1DLinearMaxWidth, handle_cudaDeviceGetTexture1DLinearMaxWidth, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceGraphMemTrim, handle_cudaDeviceGraphMemTrim, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceReset, handle_cudaDeviceReset, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceSetCacheConfig, handle_cudaDeviceSetCacheConfig, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceSetGraphMemAttribute, handle_cudaDeviceSetGraphMemAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceSetLimit, handle_cudaDeviceSetLimit, rpc_backend::cudart) \
  HANDLER(RPC_cudaDeviceSetMemPool, handle_cudaDeviceSetMemPool, rpc_backend::cudart) \
  HANDLER(RPC_cudaDriverGetVersion, handle_cudaDriverGetVersion, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventCreate, handle_cudaEventCreate, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventCreateWithFlags, handle_cudaEventCreateWithFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaEventElapsedTime, handle_cudaEventElapsedTime, rpc_backend::cudart) \
  HANDLER(RPC_cudaExternalMemoryGetMappedBuffer, handle_cudaExternalMemoryGetMappedBuffer, rpc_backend::cudart) \
  HANDLER(RPC_cudaExternalMemoryGetMappedMipmappedArray, handle_cudaExternalMemoryGetMappedMipmappedArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaFreeArray, handle_cudaFreeArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaFreeAsync, handle_cudaFreeAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaFreeMipmappedArray, handle_cudaFreeMipmappedArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaFuncGetAttributes, handle_cudaFuncGetAttributes, rpc_backend::cudart) \
  HANDLER(RPC_cudaFuncSetAttribute, handle_cudaFuncSetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaFuncSetCacheConfig, handle_cudaFuncSetCacheConfig, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetChannelDesc, handle_cudaGetChannelDesc, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetDevice, handle_cudaGetDevice, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetDeviceCount, handle_cudaGetDeviceCount, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetDeviceFlags, handle_cudaGetDeviceFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsGLRegisterBuffer, handle_cudaGraphicsGLRegisterBuffer, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsGLRegisterImage, handle_cudaGraphicsGLRegisterImage, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetDeviceProperties, handle_cudaGetDeviceProperties, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetExportTable, handle_cudaGetExportTable, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetFuncBySymbol, handle_cudaGetFuncBySymbol, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetLastError, handle_cudaGetLastError, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetMipmappedArrayLevel, handle_cudaGetMipmappedArrayLevel, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetSurfaceObjectResourceDesc, handle_cudaGetSurfaceObjectResourceDesc, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetSymbolAddress, handle_cudaGetSymbolAddress, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetSymbolSize, handle_cudaGetSymbolSize, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetTextureObjectResourceDesc, handle_cudaGetTextureObjectResourceDesc, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetTextureObjectResourceViewDesc, handle_cudaGetTextureObjectResourceViewDesc, rpc_backend::cudart) \
  HANDLER(RPC_cudaGetTextureObjectTextureDesc, handle_cudaGetTextureObjectTextureDesc, rpc_backend::cudart) \
  HANDLER(RPC_cudaHostGetDevicePointer, handle_cudaHostGetDevicePointer, rpc_backend::cudart) \
  HANDLER(RPC_cudaImportExternalMemory, handle_cudaImportExternalMemory, rpc_backend::cudart) \
  HANDLER(RPC_cudaImportExternalSemaphore, handle_cudaImportExternalSemaphore, rpc_backend::cudart) \
  HANDLER(RPC_cudaIpcCloseMemHandle, handle_cudaIpcCloseMemHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaIpcGetEventHandle, handle_cudaIpcGetEventHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaIpcGetMemHandle, handle_cudaIpcGetMemHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaIpcOpenEventHandle, handle_cudaIpcOpenEventHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaIpcOpenMemHandle, handle_cudaIpcOpenMemHandle, rpc_backend::cudart) \
  HANDLER(RPC_cudaMalloc, handle_cudaMalloc, rpc_backend::cudart) \
  HANDLER(RPC_cudaMalloc3D, handle_cudaMalloc3D, rpc_backend::cudart) \
  HANDLER(RPC_cudaMalloc3DArray, handle_cudaMalloc3DArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocArray, handle_cudaMallocArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocAsync, handle_cudaMallocAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocFromPoolAsync, handle_cudaMallocFromPoolAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocMipmappedArray, handle_cudaMallocMipmappedArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaMallocPitch, handle_cudaMallocPitch, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemGetInfo, handle_cudaMemGetInfo, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolCreate, handle_cudaMemPoolCreate, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolDestroy, handle_cudaMemPoolDestroy, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolExportPointer, handle_cudaMemPoolExportPointer, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolGetAccess, handle_cudaMemPoolGetAccess, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolGetAttribute, handle_cudaMemPoolGetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolImportPointer, handle_cudaMemPoolImportPointer, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolSetAccess, handle_cudaMemPoolSetAccess, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolSetAttribute, handle_cudaMemPoolSetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemPoolTrimTo, handle_cudaMemPoolTrimTo, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemRangeGetAttribute, handle_cudaMemRangeGetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemRangeGetAttributes, handle_cudaMemRangeGetAttributes, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemcpy3DPeer, handle_cudaMemcpy3DPeer, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemcpy3DPeerAsync, handle_cudaMemcpy3DPeerAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemset, handle_cudaMemset, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemset2D, handle_cudaMemset2D, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemset2DAsync, handle_cudaMemset2DAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemset3D, handle_cudaMemset3D, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemset3DAsync, handle_cudaMemset3DAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaMemsetAsync, handle_cudaMemsetAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaMipmappedArrayGetMemoryRequirements, handle_cudaMipmappedArrayGetMemoryRequirements, rpc_backend::cudart) \
  HANDLER(RPC_cudaMipmappedArrayGetSparseProperties, handle_cudaMipmappedArrayGetSparseProperties, rpc_backend::cudart) \
  HANDLER(RPC_cudaOccupancyAvailableDynamicSMemPerBlock, handle_cudaOccupancyAvailableDynamicSMemPerBlock, rpc_backend::cudart) \
  HANDLER(RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessor, handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor, rpc_backend::cudart) \
  HANDLER(RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaOccupancyMaxActiveClusters, handle_cudaOccupancyMaxActiveClusters, rpc_backend::cudart) \
  HANDLER(RPC_cudaOccupancyMaxPotentialClusterSize, handle_cudaOccupancyMaxPotentialClusterSize, rpc_backend::cudart) \
  HANDLER(RPC_cudaPeekAtLastError, handle_cudaPeekAtLastError, rpc_backend::cudart) \
  HANDLER(RPC_cudaPointerGetAttributes, handle_cudaPointerGetAttributes, rpc_backend::cudart) \
  HANDLER(RPC_cudaProfilerStart, handle_cudaProfilerStart, rpc_backend::cudart) \
  HANDLER(RPC_cudaProfilerStop, handle_cudaProfilerStop, rpc_backend::cudart) \
  HANDLER(RPC_cudaRuntimeGetVersion, handle_cudaRuntimeGetVersion, rpc_backend::cudart) \
  HANDLER(RPC_cudaSetDevice, handle_cudaSetDevice, rpc_backend::cudart) \
  HANDLER(RPC_cudaSetDeviceFlags, handle_cudaSetDeviceFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaSetValidDevices, handle_cudaSetValidDevices, rpc_backend::cudart) \
  HANDLER(RPC_cudaSignalExternalSemaphoresAsync, handle_cudaSignalExternalSemaphoresAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamAttachMemAsync, handle_cudaStreamAttachMemAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamCopyAttributes, handle_cudaStreamCopyAttributes, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamCreate, handle_cudaStreamCreate, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamCreateWithFlags, handle_cudaStreamCreateWithFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamCreateWithPriority, handle_cudaStreamCreateWithPriority, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamDestroy, handle_cudaStreamDestroy, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamGetAttribute, handle_cudaStreamGetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamGetFlags, handle_cudaStreamGetFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamGetPriority, handle_cudaStreamGetPriority, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamIsCapturing, handle_cudaStreamIsCapturing, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamSetAttribute, handle_cudaStreamSetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaStreamWaitEvent, handle_cudaStreamWaitEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaThreadExchangeStreamCaptureMode, handle_cudaThreadExchangeStreamCaptureMode, rpc_backend::cudart) \
  HANDLER(RPC_cudaUserObjectCreate, handle_cudaUserObjectCreate, rpc_backend::cudart) \
  HANDLER(RPC_cudaUserObjectRelease, handle_cudaUserObjectRelease, rpc_backend::cudart) \
  HANDLER(RPC_cudaUserObjectRetain, handle_cudaUserObjectRetain, rpc_backend::cudart) \
  HANDLER(RPC_cudaWaitExternalSemaphoresAsync, handle_cudaWaitExternalSemaphoresAsync, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddChildGraphNode, handle_cudaGraphAddChildGraphNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddEmptyNode, handle_cudaGraphAddEmptyNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddEventRecordNode, handle_cudaGraphAddEventRecordNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddEventWaitNode, handle_cudaGraphAddEventWaitNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddExternalSemaphoresSignalNode, handle_cudaGraphAddExternalSemaphoresSignalNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddExternalSemaphoresWaitNode, handle_cudaGraphAddExternalSemaphoresWaitNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddMemAllocNode, handle_cudaGraphAddMemAllocNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddMemFreeNode, handle_cudaGraphAddMemFreeNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphAddMemsetNode, handle_cudaGraphAddMemsetNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphChildGraphNodeGetGraph, handle_cudaGraphChildGraphNodeGetGraph, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphCreate, handle_cudaGraphCreate, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphDebugDotPrint, handle_cudaGraphDebugDotPrint, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphDestroyNode, handle_cudaGraphDestroyNode, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphEventRecordNodeGetEvent, handle_cudaGraphEventRecordNodeGetEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphEventRecordNodeSetEvent, handle_cudaGraphEventRecordNodeSetEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphEventWaitNodeGetEvent, handle_cudaGraphEventWaitNodeGetEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphEventWaitNodeSetEvent, handle_cudaGraphEventWaitNodeSetEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecChildGraphNodeSetParams, handle_cudaGraphExecChildGraphNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecEventRecordNodeSetEvent, handle_cudaGraphExecEventRecordNodeSetEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecEventWaitNodeSetEvent, handle_cudaGraphExecEventWaitNodeSetEvent, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecExternalSemaphoresSignalNodeSetParams, handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecExternalSemaphoresWaitNodeSetParams, handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecHostNodeSetParams, handle_cudaGraphExecHostNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExecMemsetNodeSetParams, handle_cudaGraphExecMemsetNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExternalSemaphoresSignalNodeGetParams, handle_cudaGraphExternalSemaphoresSignalNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExternalSemaphoresSignalNodeSetParams, handle_cudaGraphExternalSemaphoresSignalNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExternalSemaphoresWaitNodeGetParams, handle_cudaGraphExternalSemaphoresWaitNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphExternalSemaphoresWaitNodeSetParams, handle_cudaGraphExternalSemaphoresWaitNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphGetNodes, handle_cudaGraphGetNodes, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphGetRootNodes, handle_cudaGraphGetRootNodes, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphHostNodeGetParams, handle_cudaGraphHostNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphHostNodeSetParams, handle_cudaGraphHostNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphKernelNodeGetAttribute, handle_cudaGraphKernelNodeGetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphKernelNodeGetParams, handle_cudaGraphKernelNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphKernelNodeSetAttribute, handle_cudaGraphKernelNodeSetAttribute, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphMemAllocNodeGetParams, handle_cudaGraphMemAllocNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphMemFreeNodeGetParams, handle_cudaGraphMemFreeNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphMemsetNodeGetParams, handle_cudaGraphMemsetNodeGetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphMemsetNodeSetParams, handle_cudaGraphMemsetNodeSetParams, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphNodeFindInClone, handle_cudaGraphNodeFindInClone, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphNodeGetEnabled, handle_cudaGraphNodeGetEnabled, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphNodeGetType, handle_cudaGraphNodeGetType, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphNodeSetEnabled, handle_cudaGraphNodeSetEnabled, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphReleaseUserObject, handle_cudaGraphReleaseUserObject, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphRetainUserObject, handle_cudaGraphRetainUserObject, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphUpload, handle_cudaGraphUpload, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsMapResources, handle_cudaGraphicsMapResources, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsResourceGetMappedMipmappedArray, handle_cudaGraphicsResourceGetMappedMipmappedArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsResourceGetMappedPointer, handle_cudaGraphicsResourceGetMappedPointer, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsResourceSetMapFlags, handle_cudaGraphicsResourceSetMapFlags, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsSubResourceGetMappedArray, handle_cudaGraphicsSubResourceGetMappedArray, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsUnmapResources, handle_cudaGraphicsUnmapResources, rpc_backend::cudart) \
  HANDLER(RPC_cudaGraphicsUnregisterResource, handle_cudaGraphicsUnregisterResource, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterManagedVar, handle___cudaRegisterManagedVar, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterTexture, handle___cudaRegisterTexture, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterSurface, handle___cudaRegisterSurface, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterHostVar, handle___cudaRegisterHostVar, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterUnifiedTable, handle___cudaRegisterUnifiedTable, rpc_backend::cudart) \
  HANDLER(RPC___cudaPushCallConfiguration, handle___cudaPushCallConfiguration, rpc_backend::cudart) \
  HANDLER(RPC___cudaPopCallConfiguration, handle___cudaPopCallConfiguration, rpc_backend::cudart) \
  HANDLER(RPC___cudaRegisterFatBinaryEnd, handle___cudaRegisterFatBinaryEnd, rpc_backend::cudart) \
  HANDLER(RPC___cudaInitModule, handle___cudaInitModule, rpc_backend::cudart)
#define LUPINE_CUBLAS_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cublasGetLoggerCallback, handle_cublasGetLoggerCallback, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetMatrix, handle_cublasGetMatrix, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetMatrixAsync, handle_cublasGetMatrixAsync, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetStatusName, handle_cublasGetStatusName, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetStatusString, handle_cublasGetStatusString, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetVector, handle_cublasGetVector, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetVectorAsync, handle_cublasGetVectorAsync, rpc_backend::cublas) \
  HANDLER(RPC_cublasLoggerConfigure, handle_cublasLoggerConfigure, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetLoggerCallback, handle_cublasSetLoggerCallback, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetMatrix, handle_cublasSetMatrix, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetMatrixAsync, handle_cublasSetMatrixAsync, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetVector, handle_cublasSetVector, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetVectorAsync, handle_cublasSetVectorAsync, rpc_backend::cublas) \
  HANDLER(RPC_cublasAsumEx, handle_cublasAsumEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasAxpyEx, handle_cublasAxpyEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCaxpy_v2, handle_cublasCaxpy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCcopy_v2, handle_cublasCcopy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCdgmm, handle_cublasCdgmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasCdotc_v2, handle_cublasCdotc_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCdotu_v2, handle_cublasCdotu_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgbmv_v2, handle_cublasCgbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgeam, handle_cublasCgeam, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgelsBatched, handle_cublasCgelsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemm3m, handle_cublasCgemm3m, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemm3mBatched, handle_cublasCgemm3mBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemm3mEx, handle_cublasCgemm3mEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemm3mStridedBatched, handle_cublasCgemm3mStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemmBatched, handle_cublasCgemmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemmEx, handle_cublasCgemmEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemmStridedBatched, handle_cublasCgemmStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemm_v2, handle_cublasCgemm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemvBatched, handle_cublasCgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemvStridedBatched, handle_cublasCgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgemv_v2, handle_cublasCgemv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgeqrfBatched, handle_cublasCgeqrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgerc_v2, handle_cublasCgerc_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgeru_v2, handle_cublasCgeru_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgetrfBatched, handle_cublasCgetrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgetriBatched, handle_cublasCgetriBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCgetrsBatched, handle_cublasCgetrsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasChbmv_v2, handle_cublasChbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasChemm_v2, handle_cublasChemm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasChemv_v2, handle_cublasChemv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCher2_v2, handle_cublasCher2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCher2k_v2, handle_cublasCher2k_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCher_v2, handle_cublasCher_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCherk3mEx, handle_cublasCherk3mEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCherkEx, handle_cublasCherkEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCherk_v2, handle_cublasCherk_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCherkx, handle_cublasCherkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasChpmv_v2, handle_cublasChpmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasChpr2_v2, handle_cublasChpr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasChpr_v2, handle_cublasChpr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCmatinvBatched, handle_cublasCmatinvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCopyEx, handle_cublasCopyEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCreate_v2, handle_cublasCreate_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCrot_v2, handle_cublasCrot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCrotg_v2, handle_cublasCrotg_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCscal_v2, handle_cublasCscal_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsrot_v2, handle_cublasCsrot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsscal_v2, handle_cublasCsscal_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCswap_v2, handle_cublasCswap_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsymm_v2, handle_cublasCsymm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsymv_v2, handle_cublasCsymv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyr2_v2, handle_cublasCsyr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyr2k_v2, handle_cublasCsyr2k_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyr_v2, handle_cublasCsyr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyrk3mEx, handle_cublasCsyrk3mEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyrkEx, handle_cublasCsyrkEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyrk_v2, handle_cublasCsyrk_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCsyrkx, handle_cublasCsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtbmv_v2, handle_cublasCtbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtbsv_v2, handle_cublasCtbsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtpmv_v2, handle_cublasCtpmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtpsv_v2, handle_cublasCtpsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtpttr, handle_cublasCtpttr, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtrmm_v2, handle_cublasCtrmm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtrmv_v2, handle_cublasCtrmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtrsmBatched, handle_cublasCtrsmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtrsm_v2, handle_cublasCtrsm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtrsv_v2, handle_cublasCtrsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasCtrttp, handle_cublasCtrttp, rpc_backend::cublas) \
  HANDLER(RPC_cublasDasum_v2, handle_cublasDasum_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDaxpy_v2, handle_cublasDaxpy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDcopy_v2, handle_cublasDcopy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDdgmm, handle_cublasDdgmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasDdot_v2, handle_cublasDdot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDestroy_v2, handle_cublasDestroy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgbmv_v2, handle_cublasDgbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgeam, handle_cublasDgeam, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgelsBatched, handle_cublasDgelsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgemmBatched, handle_cublasDgemmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgemmStridedBatched, handle_cublasDgemmStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgemm_v2, handle_cublasDgemm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgemvBatched, handle_cublasDgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgemvStridedBatched, handle_cublasDgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgemv_v2, handle_cublasDgemv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgeqrfBatched, handle_cublasDgeqrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDger_v2, handle_cublasDger_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgetrfBatched, handle_cublasDgetrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgetriBatched, handle_cublasDgetriBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDgetrsBatched, handle_cublasDgetrsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDmatinvBatched, handle_cublasDmatinvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDnrm2_v2, handle_cublasDnrm2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDotEx, handle_cublasDotEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasDotcEx, handle_cublasDotcEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasDrot_v2, handle_cublasDrot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDrotg_v2, handle_cublasDrotg_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDrotm_v2, handle_cublasDrotm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDrotmg_v2, handle_cublasDrotmg_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsbmv_v2, handle_cublasDsbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDscal_v2, handle_cublasDscal_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDspmv_v2, handle_cublasDspmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDspr2_v2, handle_cublasDspr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDspr_v2, handle_cublasDspr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDswap_v2, handle_cublasDswap_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsymm_v2, handle_cublasDsymm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsymv_v2, handle_cublasDsymv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsyr2_v2, handle_cublasDsyr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsyr2k_v2, handle_cublasDsyr2k_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsyr_v2, handle_cublasDsyr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsyrk_v2, handle_cublasDsyrk_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDsyrkx, handle_cublasDsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtbmv_v2, handle_cublasDtbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtbsv_v2, handle_cublasDtbsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtpmv_v2, handle_cublasDtpmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtpsv_v2, handle_cublasDtpsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtpttr, handle_cublasDtpttr, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtrmm_v2, handle_cublasDtrmm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtrmv_v2, handle_cublasDtrmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtrsmBatched, handle_cublasDtrsmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtrsm_v2, handle_cublasDtrsm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtrsv_v2, handle_cublasDtrsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDtrttp, handle_cublasDtrttp, rpc_backend::cublas) \
  HANDLER(RPC_cublasDzasum_v2, handle_cublasDzasum_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasDznrm2_v2, handle_cublasDznrm2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasGemmBatchedEx, handle_cublasGemmBatchedEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasGemmEx, handle_cublasGemmEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasGemmStridedBatchedEx, handle_cublasGemmStridedBatchedEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetAtomicsMode, handle_cublasGetAtomicsMode, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetCudartVersion, handle_cublasGetCudartVersion, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetMathMode, handle_cublasGetMathMode, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetPointerMode_v2, handle_cublasGetPointerMode_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetProperty, handle_cublasGetProperty, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetSmCountTarget, handle_cublasGetSmCountTarget, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetStream_v2, handle_cublasGetStream_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasGetVersion_v2, handle_cublasGetVersion_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasHSHgemvBatched, handle_cublasHSHgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasHSHgemvStridedBatched, handle_cublasHSHgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasHSSgemvBatched, handle_cublasHSSgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasHSSgemvStridedBatched, handle_cublasHSSgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasHgemm, handle_cublasHgemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasHgemmBatched, handle_cublasHgemmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasHgemmStridedBatched, handle_cublasHgemmStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasIamaxEx, handle_cublasIamaxEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasIaminEx, handle_cublasIaminEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasIcamax_v2, handle_cublasIcamax_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIcamin_v2, handle_cublasIcamin_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIdamax_v2, handle_cublasIdamax_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIdamin_v2, handle_cublasIdamin_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIsamax_v2, handle_cublasIsamax_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIsamin_v2, handle_cublasIsamin_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIzamax_v2, handle_cublasIzamax_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasIzamin_v2, handle_cublasIzamin_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasNrm2Ex, handle_cublasNrm2Ex, rpc_backend::cublas) \
  HANDLER(RPC_cublasRotEx, handle_cublasRotEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasRotgEx, handle_cublasRotgEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasRotmEx, handle_cublasRotmEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasRotmgEx, handle_cublasRotmgEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasSasum_v2, handle_cublasSasum_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSaxpy_v2, handle_cublasSaxpy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasScalEx, handle_cublasScalEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasScasum_v2, handle_cublasScasum_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasScnrm2_v2, handle_cublasScnrm2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasScopy_v2, handle_cublasScopy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSdgmm, handle_cublasSdgmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasSdot_v2, handle_cublasSdot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetAtomicsMode, handle_cublasSetAtomicsMode, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetMathMode, handle_cublasSetMathMode, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetPointerMode_v2, handle_cublasSetPointerMode_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetSmCountTarget, handle_cublasSetSmCountTarget, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetStream_v2, handle_cublasSetStream_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSetWorkspace_v2, handle_cublasSetWorkspace_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgbmv_v2, handle_cublasSgbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgeam, handle_cublasSgeam, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgelsBatched, handle_cublasSgelsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemmBatched, handle_cublasSgemmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemmEx, handle_cublasSgemmEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemmStridedBatched, handle_cublasSgemmStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemm_v2, handle_cublasSgemm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemvBatched, handle_cublasSgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemvStridedBatched, handle_cublasSgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgemv_v2, handle_cublasSgemv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgeqrfBatched, handle_cublasSgeqrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSger_v2, handle_cublasSger_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgetrfBatched, handle_cublasSgetrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgetriBatched, handle_cublasSgetriBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSgetrsBatched, handle_cublasSgetrsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSmatinvBatched, handle_cublasSmatinvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasSnrm2_v2, handle_cublasSnrm2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSrot_v2, handle_cublasSrot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSrotg_v2, handle_cublasSrotg_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSrotm_v2, handle_cublasSrotm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSrotmg_v2, handle_cublasSrotmg_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsbmv_v2, handle_cublasSsbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSscal_v2, handle_cublasSscal_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSspmv_v2, handle_cublasSspmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSspr2_v2, handle_cublasSspr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSspr_v2, handle_cublasSspr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSswap_v2, handle_cublasSswap_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsymm_v2, handle_cublasSsymm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsymv_v2, handle_cublasSsymv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsyr2_v2, handle_cublasSsyr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsyr2k_v2, handle_cublasSsyr2k_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsyr_v2, handle_cublasSsyr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsyrk_v2, handle_cublasSsyrk_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasSsyrkx, handle_cublasSsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasStbmv_v2, handle_cublasStbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStbsv_v2, handle_cublasStbsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStpmv_v2, handle_cublasStpmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStpsv_v2, handle_cublasStpsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStpttr, handle_cublasStpttr, rpc_backend::cublas) \
  HANDLER(RPC_cublasStrmm_v2, handle_cublasStrmm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStrmv_v2, handle_cublasStrmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStrsmBatched, handle_cublasStrsmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasStrsm_v2, handle_cublasStrsm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStrsv_v2, handle_cublasStrsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasStrttp, handle_cublasStrttp, rpc_backend::cublas) \
  HANDLER(RPC_cublasSwapEx, handle_cublasSwapEx, rpc_backend::cublas) \
  HANDLER(RPC_cublasTSSgemvBatched, handle_cublasTSSgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasTSSgemvStridedBatched, handle_cublasTSSgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasTSTgemvBatched, handle_cublasTSTgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasTSTgemvStridedBatched, handle_cublasTSTgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasUint8gemmBias, handle_cublasUint8gemmBias, rpc_backend::cublas) \
  HANDLER(RPC_cublasZaxpy_v2, handle_cublasZaxpy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZcopy_v2, handle_cublasZcopy_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZdgmm, handle_cublasZdgmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasZdotc_v2, handle_cublasZdotc_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZdotu_v2, handle_cublasZdotu_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZdrot_v2, handle_cublasZdrot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZdscal_v2, handle_cublasZdscal_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgbmv_v2, handle_cublasZgbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgeam, handle_cublasZgeam, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgelsBatched, handle_cublasZgelsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemm3m, handle_cublasZgemm3m, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemmBatched, handle_cublasZgemmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemmStridedBatched, handle_cublasZgemmStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemm_v2, handle_cublasZgemm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemvBatched, handle_cublasZgemvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemvStridedBatched, handle_cublasZgemvStridedBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgemv_v2, handle_cublasZgemv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgeqrfBatched, handle_cublasZgeqrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgerc_v2, handle_cublasZgerc_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgeru_v2, handle_cublasZgeru_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgetrfBatched, handle_cublasZgetrfBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgetriBatched, handle_cublasZgetriBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZgetrsBatched, handle_cublasZgetrsBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZhbmv_v2, handle_cublasZhbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZhemm_v2, handle_cublasZhemm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZhemv_v2, handle_cublasZhemv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZher2_v2, handle_cublasZher2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZher2k_v2, handle_cublasZher2k_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZher_v2, handle_cublasZher_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZherk_v2, handle_cublasZherk_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZherkx, handle_cublasZherkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasZhpmv_v2, handle_cublasZhpmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZhpr2_v2, handle_cublasZhpr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZhpr_v2, handle_cublasZhpr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZmatinvBatched, handle_cublasZmatinvBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZrot_v2, handle_cublasZrot_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZrotg_v2, handle_cublasZrotg_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZscal_v2, handle_cublasZscal_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZswap_v2, handle_cublasZswap_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsymm_v2, handle_cublasZsymm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsymv_v2, handle_cublasZsymv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsyr2_v2, handle_cublasZsyr2_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsyr2k_v2, handle_cublasZsyr2k_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsyr_v2, handle_cublasZsyr_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsyrk_v2, handle_cublasZsyrk_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZsyrkx, handle_cublasZsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtbmv_v2, handle_cublasZtbmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtbsv_v2, handle_cublasZtbsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtpmv_v2, handle_cublasZtpmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtpsv_v2, handle_cublasZtpsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtpttr, handle_cublasZtpttr, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtrmm_v2, handle_cublasZtrmm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtrmv_v2, handle_cublasZtrmv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtrsmBatched, handle_cublasZtrsmBatched, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtrsm_v2, handle_cublasZtrsm_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtrsv_v2, handle_cublasZtrsv_v2, rpc_backend::cublas) \
  HANDLER(RPC_cublasZtrttp, handle_cublasZtrttp, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCreate, handle_cublasXtCreate, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDestroy, handle_cublasXtDestroy, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtGetNumBoards, handle_cublasXtGetNumBoards, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtMaxBoards, handle_cublasXtMaxBoards, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDeviceSelect, handle_cublasXtDeviceSelect, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSetBlockDim, handle_cublasXtSetBlockDim, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtGetBlockDim, handle_cublasXtGetBlockDim, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtGetPinningMemMode, handle_cublasXtGetPinningMemMode, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSetPinningMemMode, handle_cublasXtSetPinningMemMode, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSetCpuRatio, handle_cublasXtSetCpuRatio, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSgemm, handle_cublasXtSgemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDgemm, handle_cublasXtDgemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCgemm, handle_cublasXtCgemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZgemm, handle_cublasXtZgemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSsyrk, handle_cublasXtSsyrk, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDsyrk, handle_cublasXtDsyrk, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCsyrk, handle_cublasXtCsyrk, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZsyrk, handle_cublasXtZsyrk, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCherk, handle_cublasXtCherk, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZherk, handle_cublasXtZherk, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSsyr2k, handle_cublasXtSsyr2k, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDsyr2k, handle_cublasXtDsyr2k, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCsyr2k, handle_cublasXtCsyr2k, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZsyr2k, handle_cublasXtZsyr2k, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCherkx, handle_cublasXtCherkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZherkx, handle_cublasXtZherkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtStrsm, handle_cublasXtStrsm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDtrsm, handle_cublasXtDtrsm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCtrsm, handle_cublasXtCtrsm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZtrsm, handle_cublasXtZtrsm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSsymm, handle_cublasXtSsymm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDsymm, handle_cublasXtDsymm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCsymm, handle_cublasXtCsymm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZsymm, handle_cublasXtZsymm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtChemm, handle_cublasXtChemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZhemm, handle_cublasXtZhemm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSsyrkx, handle_cublasXtSsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDsyrkx, handle_cublasXtDsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCsyrkx, handle_cublasXtCsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZsyrkx, handle_cublasXtZsyrkx, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCher2k, handle_cublasXtCher2k, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZher2k, handle_cublasXtZher2k, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtSspmm, handle_cublasXtSspmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDspmm, handle_cublasXtDspmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCspmm, handle_cublasXtCspmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZspmm, handle_cublasXtZspmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtStrmm, handle_cublasXtStrmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtDtrmm, handle_cublasXtDtrmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtCtrmm, handle_cublasXtCtrmm, rpc_backend::cublas) \
  HANDLER(RPC_cublasXtZtrmm, handle_cublasXtZtrmm, rpc_backend::cublas)
#define LUPINE_CUBLASLT_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cublasLtGetStatusName, handle_cublasLtGetStatusName, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtGetStatusString, handle_cublasLtGetStatusString, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtLoggerSetCallback, handle_cublasLtLoggerSetCallback, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtLoggerSetFile, handle_cublasLtLoggerSetFile, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtCreate, handle_cublasLtCreate, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtDestroy, handle_cublasLtDestroy, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtGetCudartVersion, handle_cublasLtGetCudartVersion, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtGetProperty, handle_cublasLtGetProperty, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtGetVersion, handle_cublasLtGetVersion, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtHeuristicsCacheGetCapacity, handle_cublasLtHeuristicsCacheGetCapacity, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtHeuristicsCacheSetCapacity, handle_cublasLtHeuristicsCacheSetCapacity, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtLoggerForceDisable, handle_cublasLtLoggerForceDisable, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtLoggerSetLevel, handle_cublasLtLoggerSetLevel, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtLoggerSetMask, handle_cublasLtLoggerSetMask, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmul, handle_cublasLtMatmul, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoCapGetAttribute, handle_cublasLtMatmulAlgoCapGetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoCheck, handle_cublasLtMatmulAlgoCheck, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoConfigGetAttribute, handle_cublasLtMatmulAlgoConfigGetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoConfigSetAttribute, handle_cublasLtMatmulAlgoConfigSetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoGetHeuristic, handle_cublasLtMatmulAlgoGetHeuristic, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoGetIds, handle_cublasLtMatmulAlgoGetIds, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulAlgoInit, handle_cublasLtMatmulAlgoInit, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulDescGetAttribute, handle_cublasLtMatmulDescGetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulDescInit_internal, handle_cublasLtMatmulDescInit_internal, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulDescSetAttribute, handle_cublasLtMatmulDescSetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulPreferenceGetAttribute, handle_cublasLtMatmulPreferenceGetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulPreferenceInit_internal, handle_cublasLtMatmulPreferenceInit_internal, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatmulPreferenceSetAttribute, handle_cublasLtMatmulPreferenceSetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixLayoutGetAttribute, handle_cublasLtMatrixLayoutGetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixLayoutInit_internal, handle_cublasLtMatrixLayoutInit_internal, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixLayoutSetAttribute, handle_cublasLtMatrixLayoutSetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixTransform, handle_cublasLtMatrixTransform, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixTransformDescGetAttribute, handle_cublasLtMatrixTransformDescGetAttribute, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixTransformDescInit_internal, handle_cublasLtMatrixTransformDescInit_internal, rpc_backend::cublas) \
  HANDLER(RPC_cublasLtMatrixTransformDescSetAttribute, handle_cublasLtMatrixTransformDescSetAttribute, rpc_backend::cublas)
#define LUPINE_CUFFT_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cufftDestroy, handle_cufftDestroy, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtFree, handle_cufftXtFree, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtMalloc, handle_cufftXtMalloc, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtMemcpy, handle_cufftXtMemcpy, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtSetCallback, handle_cufftXtSetCallback, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtSetWorkArea, handle_cufftXtSetWorkArea, rpc_backend::cufft) \
  HANDLER(RPC_cufftCreate, handle_cufftCreate, rpc_backend::cufft) \
  HANDLER(RPC_cufftEstimate1d, handle_cufftEstimate1d, rpc_backend::cufft) \
  HANDLER(RPC_cufftEstimate2d, handle_cufftEstimate2d, rpc_backend::cufft) \
  HANDLER(RPC_cufftEstimate3d, handle_cufftEstimate3d, rpc_backend::cufft) \
  HANDLER(RPC_cufftEstimateMany, handle_cufftEstimateMany, rpc_backend::cufft) \
  HANDLER(RPC_cufftExecC2C, handle_cufftExecC2C, rpc_backend::cufft) \
  HANDLER(RPC_cufftExecC2R, handle_cufftExecC2R, rpc_backend::cufft) \
  HANDLER(RPC_cufftExecD2Z, handle_cufftExecD2Z, rpc_backend::cufft) \
  HANDLER(RPC_cufftExecR2C, handle_cufftExecR2C, rpc_backend::cufft) \
  HANDLER(RPC_cufftExecZ2D, handle_cufftExecZ2D, rpc_backend::cufft) \
  HANDLER(RPC_cufftExecZ2Z, handle_cufftExecZ2Z, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetProperty, handle_cufftGetProperty, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetSize, handle_cufftGetSize, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetSize1d, handle_cufftGetSize1d, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetSize2d, handle_cufftGetSize2d, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetSize3d, handle_cufftGetSize3d, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetSizeMany, handle_cufftGetSizeMany, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetSizeMany64, handle_cufftGetSizeMany64, rpc_backend::cufft) \
  HANDLER(RPC_cufftGetVersion, handle_cufftGetVersion, rpc_backend::cufft) \
  HANDLER(RPC_cufftMakePlan1d, handle_cufftMakePlan1d, rpc_backend::cufft) \
  HANDLER(RPC_cufftMakePlan2d, handle_cufftMakePlan2d, rpc_backend::cufft) \
  HANDLER(RPC_cufftMakePlan3d, handle_cufftMakePlan3d, rpc_backend::cufft) \
  HANDLER(RPC_cufftMakePlanMany, handle_cufftMakePlanMany, rpc_backend::cufft) \
  HANDLER(RPC_cufftMakePlanMany64, handle_cufftMakePlanMany64, rpc_backend::cufft) \
  HANDLER(RPC_cufftPlan1d, handle_cufftPlan1d, rpc_backend::cufft) \
  HANDLER(RPC_cufftPlan2d, handle_cufftPlan2d, rpc_backend::cufft) \
  HANDLER(RPC_cufftPlan3d, handle_cufftPlan3d, rpc_backend::cufft) \
  HANDLER(RPC_cufftPlanMany, handle_cufftPlanMany, rpc_backend::cufft) \
  HANDLER(RPC_cufftSetAutoAllocation, handle_cufftSetAutoAllocation, rpc_backend::cufft) \
  HANDLER(RPC_cufftSetStream, handle_cufftSetStream, rpc_backend::cufft) \
  HANDLER(RPC_cufftSetWorkArea, handle_cufftSetWorkArea, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtClearCallback, handle_cufftXtClearCallback, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExec, handle_cufftXtExec, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptor, handle_cufftXtExecDescriptor, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptorC2C, handle_cufftXtExecDescriptorC2C, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptorC2R, handle_cufftXtExecDescriptorC2R, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptorD2Z, handle_cufftXtExecDescriptorD2Z, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptorR2C, handle_cufftXtExecDescriptorR2C, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptorZ2D, handle_cufftXtExecDescriptorZ2D, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtExecDescriptorZ2Z, handle_cufftXtExecDescriptorZ2Z, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtGetSizeMany, handle_cufftXtGetSizeMany, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtMakePlanMany, handle_cufftXtMakePlanMany, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtQueryPlan, handle_cufftXtQueryPlan, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtSetCallbackSharedSize, handle_cufftXtSetCallbackSharedSize, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtSetGPUs, handle_cufftXtSetGPUs, rpc_backend::cufft) \
  HANDLER(RPC_cufftXtSetWorkAreaPolicy, handle_cufftXtSetWorkAreaPolicy, rpc_backend::cufft)
#define LUPINE_CUDNN_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cudnnGetErrorString, handle_cudnnGetErrorString, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendSetAttribute, handle_cudnnBackendSetAttribute, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendGetAttribute, handle_cudnnBackendGetAttribute, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnMultiHeadAttnForward, handle_cudnnMultiHeadAttnForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnMultiHeadAttnBackwardData, handle_cudnnMultiHeadAttnBackwardData, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCTCLoss, handle_cudnnCTCLoss, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCTCLossWorkspaceSize, handle_cudnnGetCTCLossWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetVersion, handle_cudnnGetVersion, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetMaxDeviceVersion, handle_cudnnGetMaxDeviceVersion, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCudartVersion, handle_cudnnGetCudartVersion, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetLastErrorString, handle_cudnnGetLastErrorString, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnQueryRuntimeError, handle_cudnnQueryRuntimeError, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetProperty, handle_cudnnGetProperty, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreate, handle_cudnnCreate, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroy, handle_cudnnDestroy, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetStream, handle_cudnnSetStream, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetStream, handle_cudnnGetStream, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGraphVersionCheck, handle_cudnnGraphVersionCheck, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendCreateDescriptor, handle_cudnnBackendCreateDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendDestroyDescriptor, handle_cudnnBackendDestroyDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendInitialize, handle_cudnnBackendInitialize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendFinalize, handle_cudnnBackendFinalize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBackendExecute, handle_cudnnBackendExecute, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateTensorDescriptor, handle_cudnnCreateTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetTensor4dDescriptor, handle_cudnnSetTensor4dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetTensor4dDescriptorEx, handle_cudnnSetTensor4dDescriptorEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetTensor4dDescriptor, handle_cudnnGetTensor4dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetTensorNdDescriptor, handle_cudnnSetTensorNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetTensorNdDescriptorEx, handle_cudnnSetTensorNdDescriptorEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetTensorNdDescriptor, handle_cudnnGetTensorNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetTensorSizeInBytes, handle_cudnnGetTensorSizeInBytes, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyTensorDescriptor, handle_cudnnDestroyTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnInitTransformDest, handle_cudnnInitTransformDest, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateTensorTransformDescriptor, handle_cudnnCreateTensorTransformDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetTensorTransformDescriptor, handle_cudnnSetTensorTransformDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetTensorTransformDescriptor, handle_cudnnGetTensorTransformDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyTensorTransformDescriptor, handle_cudnnDestroyTensorTransformDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnTransformTensor, handle_cudnnTransformTensor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnTransformTensorEx, handle_cudnnTransformTensorEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnAddTensor, handle_cudnnAddTensor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateOpTensorDescriptor, handle_cudnnCreateOpTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetOpTensorDescriptor, handle_cudnnSetOpTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetOpTensorDescriptor, handle_cudnnGetOpTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyOpTensorDescriptor, handle_cudnnDestroyOpTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnOpTensor, handle_cudnnOpTensor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateReduceTensorDescriptor, handle_cudnnCreateReduceTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetReduceTensorDescriptor, handle_cudnnSetReduceTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetReduceTensorDescriptor, handle_cudnnGetReduceTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyReduceTensorDescriptor, handle_cudnnDestroyReduceTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetReductionIndicesSize, handle_cudnnGetReductionIndicesSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetReductionWorkspaceSize, handle_cudnnGetReductionWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnReduceTensor, handle_cudnnReduceTensor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetTensor, handle_cudnnSetTensor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnScaleTensor, handle_cudnnScaleTensor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateFilterDescriptor, handle_cudnnCreateFilterDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetFilter4dDescriptor, handle_cudnnSetFilter4dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetFilter4dDescriptor, handle_cudnnGetFilter4dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetFilterNdDescriptor, handle_cudnnSetFilterNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetFilterNdDescriptor, handle_cudnnGetFilterNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetFilterSizeInBytes, handle_cudnnGetFilterSizeInBytes, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnTransformFilter, handle_cudnnTransformFilter, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyFilterDescriptor, handle_cudnnDestroyFilterDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSoftmaxForward, handle_cudnnSoftmaxForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreatePoolingDescriptor, handle_cudnnCreatePoolingDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetPooling2dDescriptor, handle_cudnnSetPooling2dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetPooling2dDescriptor, handle_cudnnGetPooling2dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetPoolingNdDescriptor, handle_cudnnSetPoolingNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetPoolingNdDescriptor, handle_cudnnGetPoolingNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetPoolingNdForwardOutputDim, handle_cudnnGetPoolingNdForwardOutputDim, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetPooling2dForwardOutputDim, handle_cudnnGetPooling2dForwardOutputDim, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyPoolingDescriptor, handle_cudnnDestroyPoolingDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnPoolingForward, handle_cudnnPoolingForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateActivationDescriptor, handle_cudnnCreateActivationDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetActivationDescriptor, handle_cudnnSetActivationDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetActivationDescriptor, handle_cudnnGetActivationDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetActivationDescriptorSwishBeta, handle_cudnnSetActivationDescriptorSwishBeta, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetActivationDescriptorSwishBeta, handle_cudnnGetActivationDescriptorSwishBeta, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyActivationDescriptor, handle_cudnnDestroyActivationDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnActivationForward, handle_cudnnActivationForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateLRNDescriptor, handle_cudnnCreateLRNDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetLRNDescriptor, handle_cudnnSetLRNDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetLRNDescriptor, handle_cudnnGetLRNDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyLRNDescriptor, handle_cudnnDestroyLRNDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnLRNCrossChannelForward, handle_cudnnLRNCrossChannelForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDivisiveNormalizationForward, handle_cudnnDivisiveNormalizationForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDeriveBNTensorDescriptor, handle_cudnnDeriveBNTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBatchNormalizationForwardInference, handle_cudnnBatchNormalizationForwardInference, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDeriveNormTensorDescriptor, handle_cudnnDeriveNormTensorDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnNormalizationForwardInference, handle_cudnnNormalizationForwardInference, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateSpatialTransformerDescriptor, handle_cudnnCreateSpatialTransformerDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetSpatialTransformerNdDescriptor, handle_cudnnSetSpatialTransformerNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroySpatialTransformerDescriptor, handle_cudnnDestroySpatialTransformerDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSpatialTfGridGeneratorForward, handle_cudnnSpatialTfGridGeneratorForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSpatialTfSamplerForward, handle_cudnnSpatialTfSamplerForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateDropoutDescriptor, handle_cudnnCreateDropoutDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyDropoutDescriptor, handle_cudnnDestroyDropoutDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDropoutGetStatesSize, handle_cudnnDropoutGetStatesSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDropoutGetReserveSpaceSize, handle_cudnnDropoutGetReserveSpaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetDropoutDescriptor, handle_cudnnSetDropoutDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRestoreDropoutDescriptor, handle_cudnnRestoreDropoutDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetDropoutDescriptor, handle_cudnnGetDropoutDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDropoutForward, handle_cudnnDropoutForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnOpsVersionCheck, handle_cudnnOpsVersionCheck, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSoftmaxBackward, handle_cudnnSoftmaxBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnPoolingBackward, handle_cudnnPoolingBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnActivationBackward, handle_cudnnActivationBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnLRNCrossChannelBackward, handle_cudnnLRNCrossChannelBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDivisiveNormalizationBackward, handle_cudnnDivisiveNormalizationBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, handle_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetBatchNormalizationBackwardExWorkspaceSize, handle_cudnnGetBatchNormalizationBackwardExWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetBatchNormalizationTrainingExReserveSpaceSize, handle_cudnnGetBatchNormalizationTrainingExReserveSpaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBatchNormalizationForwardTraining, handle_cudnnBatchNormalizationForwardTraining, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBatchNormalizationForwardTrainingEx, handle_cudnnBatchNormalizationForwardTrainingEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBatchNormalizationBackward, handle_cudnnBatchNormalizationBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBatchNormalizationBackwardEx, handle_cudnnBatchNormalizationBackwardEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetNormalizationForwardTrainingWorkspaceSize, handle_cudnnGetNormalizationForwardTrainingWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetNormalizationBackwardWorkspaceSize, handle_cudnnGetNormalizationBackwardWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetNormalizationTrainingReserveSpaceSize, handle_cudnnGetNormalizationTrainingReserveSpaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnNormalizationForwardTraining, handle_cudnnNormalizationForwardTraining, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnNormalizationBackward, handle_cudnnNormalizationBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSpatialTfGridGeneratorBackward, handle_cudnnSpatialTfGridGeneratorBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSpatialTfSamplerBackward, handle_cudnnSpatialTfSamplerBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDropoutBackward, handle_cudnnDropoutBackward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateConvolutionDescriptor, handle_cudnnCreateConvolutionDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyConvolutionDescriptor, handle_cudnnDestroyConvolutionDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetConvolutionMathType, handle_cudnnSetConvolutionMathType, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionMathType, handle_cudnnGetConvolutionMathType, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetConvolutionGroupCount, handle_cudnnSetConvolutionGroupCount, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionGroupCount, handle_cudnnGetConvolutionGroupCount, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetConvolutionReorderType, handle_cudnnSetConvolutionReorderType, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionReorderType, handle_cudnnGetConvolutionReorderType, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetConvolution2dDescriptor, handle_cudnnSetConvolution2dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolution2dDescriptor, handle_cudnnGetConvolution2dDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetConvolutionNdDescriptor, handle_cudnnSetConvolutionNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionNdDescriptor, handle_cudnnGetConvolutionNdDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolution2dForwardOutputDim, handle_cudnnGetConvolution2dForwardOutputDim, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionNdForwardOutputDim, handle_cudnnGetConvolutionNdForwardOutputDim, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionForwardAlgorithmMaxCount, handle_cudnnGetConvolutionForwardAlgorithmMaxCount, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionForwardAlgorithm_v7, handle_cudnnGetConvolutionForwardAlgorithm_v7, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFindConvolutionForwardAlgorithm, handle_cudnnFindConvolutionForwardAlgorithm, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFindConvolutionForwardAlgorithmEx, handle_cudnnFindConvolutionForwardAlgorithmEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnIm2Col, handle_cudnnIm2Col, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnReorderFilterAndBias, handle_cudnnReorderFilterAndBias, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionForwardWorkspaceSize, handle_cudnnGetConvolutionForwardWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnConvolutionForward, handle_cudnnConvolutionForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnConvolutionBiasActivationForward, handle_cudnnConvolutionBiasActivationForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionBackwardDataAlgorithmMaxCount, handle_cudnnGetConvolutionBackwardDataAlgorithmMaxCount, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFindConvolutionBackwardDataAlgorithm, handle_cudnnFindConvolutionBackwardDataAlgorithm, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFindConvolutionBackwardDataAlgorithmEx, handle_cudnnFindConvolutionBackwardDataAlgorithmEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionBackwardDataAlgorithm_v7, handle_cudnnGetConvolutionBackwardDataAlgorithm_v7, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionBackwardDataWorkspaceSize, handle_cudnnGetConvolutionBackwardDataWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnConvolutionBackwardData, handle_cudnnConvolutionBackwardData, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetFoldedConvBackwardDataDescriptors, handle_cudnnGetFoldedConvBackwardDataDescriptors, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCnnVersionCheck, handle_cudnnCnnVersionCheck, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, handle_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFindConvolutionBackwardFilterAlgorithm, handle_cudnnFindConvolutionBackwardFilterAlgorithm, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFindConvolutionBackwardFilterAlgorithmEx, handle_cudnnFindConvolutionBackwardFilterAlgorithmEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionBackwardFilterAlgorithm_v7, handle_cudnnGetConvolutionBackwardFilterAlgorithm_v7, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetConvolutionBackwardFilterWorkspaceSize, handle_cudnnGetConvolutionBackwardFilterWorkspaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnConvolutionBackwardFilter, handle_cudnnConvolutionBackwardFilter, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnConvolutionBackwardBias, handle_cudnnConvolutionBackwardBias, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateFusedOpsConstParamPack, handle_cudnnCreateFusedOpsConstParamPack, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyFusedOpsConstParamPack, handle_cudnnDestroyFusedOpsConstParamPack, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateFusedOpsVariantParamPack, handle_cudnnCreateFusedOpsVariantParamPack, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyFusedOpsVariantParamPack, handle_cudnnDestroyFusedOpsVariantParamPack, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateFusedOpsPlan, handle_cudnnCreateFusedOpsPlan, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyFusedOpsPlan, handle_cudnnDestroyFusedOpsPlan, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnMakeFusedOpsPlan, handle_cudnnMakeFusedOpsPlan, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnFusedOpsExecute, handle_cudnnFusedOpsExecute, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateRNNDescriptor, handle_cudnnCreateRNNDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyRNNDescriptor, handle_cudnnDestroyRNNDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetRNNDescriptor_v8, handle_cudnnSetRNNDescriptor_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetRNNDescriptor_v8, handle_cudnnGetRNNDescriptor_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNSetClip_v8, handle_cudnnRNNSetClip_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNSetClip_v9, handle_cudnnRNNSetClip_v9, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNGetClip_v8, handle_cudnnRNNGetClip_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNGetClip_v9, handle_cudnnRNNGetClip_v9, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnBuildRNNDynamic, handle_cudnnBuildRNNDynamic, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetRNNTempSpaceSizes, handle_cudnnGetRNNTempSpaceSizes, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetRNNWeightSpaceSize, handle_cudnnGetRNNWeightSpaceSize, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetRNNWeightParams, handle_cudnnGetRNNWeightParams, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateRNNDataDescriptor, handle_cudnnCreateRNNDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyRNNDataDescriptor, handle_cudnnDestroyRNNDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetRNNDataDescriptor, handle_cudnnSetRNNDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetRNNDataDescriptor, handle_cudnnGetRNNDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNForward, handle_cudnnRNNForward, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateSeqDataDescriptor, handle_cudnnCreateSeqDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroySeqDataDescriptor, handle_cudnnDestroySeqDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetSeqDataDescriptor, handle_cudnnSetSeqDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetSeqDataDescriptor, handle_cudnnGetSeqDataDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateAttnDescriptor, handle_cudnnCreateAttnDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyAttnDescriptor, handle_cudnnDestroyAttnDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetAttnDescriptor, handle_cudnnSetAttnDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetAttnDescriptor, handle_cudnnGetAttnDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetMultiHeadAttnBuffers, handle_cudnnGetMultiHeadAttnBuffers, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetMultiHeadAttnWeights, handle_cudnnGetMultiHeadAttnWeights, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnAdvVersionCheck, handle_cudnnAdvVersionCheck, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNBackwardData_v8, handle_cudnnRNNBackwardData_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnRNNBackwardWeights_v8, handle_cudnnRNNBackwardWeights_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnMultiHeadAttnBackwardWeights, handle_cudnnMultiHeadAttnBackwardWeights, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCreateCTCLossDescriptor, handle_cudnnCreateCTCLossDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetCTCLossDescriptor, handle_cudnnSetCTCLossDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetCTCLossDescriptorEx, handle_cudnnSetCTCLossDescriptorEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetCTCLossDescriptor_v8, handle_cudnnSetCTCLossDescriptor_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnSetCTCLossDescriptor_v9, handle_cudnnSetCTCLossDescriptor_v9, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCTCLossDescriptor, handle_cudnnGetCTCLossDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCTCLossDescriptorEx, handle_cudnnGetCTCLossDescriptorEx, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCTCLossDescriptor_v8, handle_cudnnGetCTCLossDescriptor_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCTCLossDescriptor_v9, handle_cudnnGetCTCLossDescriptor_v9, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnDestroyCTCLossDescriptor, handle_cudnnDestroyCTCLossDescriptor, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnCTCLoss_v8, handle_cudnnCTCLoss_v8, rpc_backend::cudnn) \
  HANDLER(RPC_cudnnGetCTCLossWorkspaceSize_v8, handle_cudnnGetCTCLossWorkspaceSize_v8, rpc_backend::cudnn)
#define LUPINE_CURAND_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_curandGetDirectionVectors32, handle_curandGetDirectionVectors32, rpc_backend::curand) \
  HANDLER(RPC_curandGetScrambleConstants32, handle_curandGetScrambleConstants32, rpc_backend::curand) \
  HANDLER(RPC_curandGetDirectionVectors64, handle_curandGetDirectionVectors64, rpc_backend::curand) \
  HANDLER(RPC_curandGetScrambleConstants64, handle_curandGetScrambleConstants64, rpc_backend::curand) \
  HANDLER(RPC_curandCreateGenerator, handle_curandCreateGenerator, rpc_backend::curand) \
  HANDLER(RPC_curandCreateGeneratorHost, handle_curandCreateGeneratorHost, rpc_backend::curand) \
  HANDLER(RPC_curandDestroyGenerator, handle_curandDestroyGenerator, rpc_backend::curand) \
  HANDLER(RPC_curandGetVersion, handle_curandGetVersion, rpc_backend::curand) \
  HANDLER(RPC_curandGetProperty, handle_curandGetProperty, rpc_backend::curand) \
  HANDLER(RPC_curandSetStream, handle_curandSetStream, rpc_backend::curand) \
  HANDLER(RPC_curandSetPseudoRandomGeneratorSeed, handle_curandSetPseudoRandomGeneratorSeed, rpc_backend::curand) \
  HANDLER(RPC_curandSetGeneratorOffset, handle_curandSetGeneratorOffset, rpc_backend::curand) \
  HANDLER(RPC_curandSetGeneratorOrdering, handle_curandSetGeneratorOrdering, rpc_backend::curand) \
  HANDLER(RPC_curandSetQuasiRandomGeneratorDimensions, handle_curandSetQuasiRandomGeneratorDimensions, rpc_backend::curand) \
  HANDLER(RPC_curandGenerate, handle_curandGenerate, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateLongLong, handle_curandGenerateLongLong, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateUniform, handle_curandGenerateUniform, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateUniformDouble, handle_curandGenerateUniformDouble, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateNormal, handle_curandGenerateNormal, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateNormalDouble, handle_curandGenerateNormalDouble, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateLogNormal, handle_curandGenerateLogNormal, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateLogNormalDouble, handle_curandGenerateLogNormalDouble, rpc_backend::curand) \
  HANDLER(RPC_curandCreatePoissonDistribution, handle_curandCreatePoissonDistribution, rpc_backend::curand) \
  HANDLER(RPC_curandDestroyDistribution, handle_curandDestroyDistribution, rpc_backend::curand) \
  HANDLER(RPC_curandGeneratePoisson, handle_curandGeneratePoisson, rpc_backend::curand) \
  HANDLER(RPC_curandGeneratePoissonMethod, handle_curandGeneratePoissonMethod, rpc_backend::curand) \
  HANDLER(RPC_curandGenerateSeeds, handle_curandGenerateSeeds, rpc_backend::curand)
#define LUPINE_CUSPARSE_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cusparseGetErrorName, handle_cusparseGetErrorName, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetErrorString, handle_cusparseGetErrorString, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreate, handle_cusparseCreate, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroy, handle_cusparseDestroy, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetVersion, handle_cusparseGetVersion, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetProperty, handle_cusparseGetProperty, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSetStream, handle_cusparseSetStream, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetStream, handle_cusparseGetStream, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetPointerMode, handle_cusparseGetPointerMode, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSetPointerMode, handle_cusparseSetPointerMode, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseLoggerOpenFile, handle_cusparseLoggerOpenFile, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseLoggerSetLevel, handle_cusparseLoggerSetLevel, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseLoggerSetMask, handle_cusparseLoggerSetMask, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseLoggerForceDisable, handle_cusparseLoggerForceDisable, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateMatDescr, handle_cusparseCreateMatDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyMatDescr, handle_cusparseDestroyMatDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSetMatType, handle_cusparseSetMatType, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetMatType, handle_cusparseGetMatType, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSetMatFillMode, handle_cusparseSetMatFillMode, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetMatFillMode, handle_cusparseGetMatFillMode, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSetMatDiagType, handle_cusparseSetMatDiagType, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetMatDiagType, handle_cusparseGetMatDiagType, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSetMatIndexBase, handle_cusparseSetMatIndexBase, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGetMatIndexBase, handle_cusparseGetMatIndexBase, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateCsric02Info, handle_cusparseCreateCsric02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyCsric02Info, handle_cusparseDestroyCsric02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateBsric02Info, handle_cusparseCreateBsric02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyBsric02Info, handle_cusparseDestroyBsric02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateCsrilu02Info, handle_cusparseCreateCsrilu02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyCsrilu02Info, handle_cusparseDestroyCsrilu02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateBsrilu02Info, handle_cusparseCreateBsrilu02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyBsrilu02Info, handle_cusparseDestroyBsrilu02Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateBsrsv2Info, handle_cusparseCreateBsrsv2Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyBsrsv2Info, handle_cusparseDestroyBsrsv2Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateBsrsm2Info, handle_cusparseCreateBsrsm2Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyBsrsm2Info, handle_cusparseDestroyBsrsm2Info, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateCsru2csrInfo, handle_cusparseCreateCsru2csrInfo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyCsru2csrInfo, handle_cusparseDestroyCsru2csrInfo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateColorInfo, handle_cusparseCreateColorInfo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyColorInfo, handle_cusparseDestroyColorInfo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreatePruneInfo, handle_cusparseCreatePruneInfo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyPruneInfo, handle_cusparseDestroyPruneInfo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgemvi, handle_cusparseSgemvi, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgemvi_bufferSize, handle_cusparseSgemvi_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgemvi, handle_cusparseDgemvi, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgemvi_bufferSize, handle_cusparseDgemvi_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgemvi, handle_cusparseCgemvi, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgemvi_bufferSize, handle_cusparseCgemvi_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgemvi, handle_cusparseZgemvi, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgemvi_bufferSize, handle_cusparseZgemvi_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrmv, handle_cusparseSbsrmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrmv, handle_cusparseDbsrmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrmv, handle_cusparseCbsrmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrmv, handle_cusparseZbsrmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrxmv, handle_cusparseSbsrxmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrxmv, handle_cusparseDbsrxmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrxmv, handle_cusparseCbsrxmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrxmv, handle_cusparseZbsrxmv, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXbsrsv2_zeroPivot, handle_cusparseXbsrsv2_zeroPivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrsv2_bufferSize, handle_cusparseSbsrsv2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrsv2_bufferSize, handle_cusparseDbsrsv2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrsv2_bufferSize, handle_cusparseCbsrsv2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrsv2_bufferSize, handle_cusparseZbsrsv2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrsv2_analysis, handle_cusparseSbsrsv2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrsv2_analysis, handle_cusparseDbsrsv2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrsv2_analysis, handle_cusparseCbsrsv2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrsv2_analysis, handle_cusparseZbsrsv2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrsv2_solve, handle_cusparseSbsrsv2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrsv2_solve, handle_cusparseDbsrsv2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrsv2_solve, handle_cusparseCbsrsv2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrsv2_solve, handle_cusparseZbsrsv2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrmm, handle_cusparseSbsrmm, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrmm, handle_cusparseDbsrmm, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrmm, handle_cusparseCbsrmm, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrmm, handle_cusparseZbsrmm, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXbsrsm2_zeroPivot, handle_cusparseXbsrsm2_zeroPivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrsm2_bufferSize, handle_cusparseSbsrsm2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrsm2_bufferSize, handle_cusparseDbsrsm2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrsm2_bufferSize, handle_cusparseCbsrsm2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrsm2_bufferSize, handle_cusparseZbsrsm2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrsm2_analysis, handle_cusparseSbsrsm2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrsm2_analysis, handle_cusparseDbsrsm2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrsm2_analysis, handle_cusparseCbsrsm2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrsm2_analysis, handle_cusparseZbsrsm2_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrsm2_solve, handle_cusparseSbsrsm2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrsm2_solve, handle_cusparseDbsrsm2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrsm2_solve, handle_cusparseCbsrsm2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrsm2_solve, handle_cusparseZbsrsm2_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrilu02_numericBoost, handle_cusparseScsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrilu02_numericBoost, handle_cusparseDcsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrilu02_numericBoost, handle_cusparseCcsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrilu02_numericBoost, handle_cusparseZcsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsrilu02_zeroPivot, handle_cusparseXcsrilu02_zeroPivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrilu02_bufferSize, handle_cusparseScsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrilu02_bufferSize, handle_cusparseDcsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrilu02_bufferSize, handle_cusparseCcsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrilu02_bufferSize, handle_cusparseZcsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrilu02_analysis, handle_cusparseScsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrilu02_analysis, handle_cusparseDcsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrilu02_analysis, handle_cusparseCcsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrilu02_analysis, handle_cusparseZcsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrilu02, handle_cusparseScsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrilu02, handle_cusparseDcsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrilu02, handle_cusparseCcsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrilu02, handle_cusparseZcsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrilu02_numericBoost, handle_cusparseSbsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrilu02_numericBoost, handle_cusparseDbsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrilu02_numericBoost, handle_cusparseCbsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrilu02_numericBoost, handle_cusparseZbsrilu02_numericBoost, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXbsrilu02_zeroPivot, handle_cusparseXbsrilu02_zeroPivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrilu02_bufferSize, handle_cusparseSbsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrilu02_bufferSize, handle_cusparseDbsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrilu02_bufferSize, handle_cusparseCbsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrilu02_bufferSize, handle_cusparseZbsrilu02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrilu02_analysis, handle_cusparseSbsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrilu02_analysis, handle_cusparseDbsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrilu02_analysis, handle_cusparseCbsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrilu02_analysis, handle_cusparseZbsrilu02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsrilu02, handle_cusparseSbsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsrilu02, handle_cusparseDbsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsrilu02, handle_cusparseCbsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsrilu02, handle_cusparseZbsrilu02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsric02_zeroPivot, handle_cusparseXcsric02_zeroPivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsric02_bufferSize, handle_cusparseScsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsric02_bufferSize, handle_cusparseDcsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsric02_bufferSize, handle_cusparseCcsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsric02_bufferSize, handle_cusparseZcsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsric02_analysis, handle_cusparseScsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsric02_analysis, handle_cusparseDcsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsric02_analysis, handle_cusparseCcsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsric02_analysis, handle_cusparseZcsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsric02, handle_cusparseScsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsric02, handle_cusparseDcsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsric02, handle_cusparseCcsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsric02, handle_cusparseZcsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXbsric02_zeroPivot, handle_cusparseXbsric02_zeroPivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsric02_bufferSize, handle_cusparseSbsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsric02_bufferSize, handle_cusparseDbsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsric02_bufferSize, handle_cusparseCbsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsric02_bufferSize, handle_cusparseZbsric02_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsric02_analysis, handle_cusparseSbsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsric02_analysis, handle_cusparseDbsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsric02_analysis, handle_cusparseCbsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsric02_analysis, handle_cusparseZbsric02_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsric02, handle_cusparseSbsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsric02, handle_cusparseDbsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsric02, handle_cusparseCbsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsric02, handle_cusparseZbsric02, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsv2_bufferSizeExt, handle_cusparseSgtsv2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsv2_bufferSizeExt, handle_cusparseDgtsv2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsv2_bufferSizeExt, handle_cusparseCgtsv2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsv2_bufferSizeExt, handle_cusparseZgtsv2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsv2, handle_cusparseSgtsv2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsv2, handle_cusparseDgtsv2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsv2, handle_cusparseCgtsv2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsv2, handle_cusparseZgtsv2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsv2_nopivot_bufferSizeExt, handle_cusparseSgtsv2_nopivot_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsv2_nopivot_bufferSizeExt, handle_cusparseDgtsv2_nopivot_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsv2_nopivot_bufferSizeExt, handle_cusparseCgtsv2_nopivot_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsv2_nopivot_bufferSizeExt, handle_cusparseZgtsv2_nopivot_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsv2_nopivot, handle_cusparseSgtsv2_nopivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsv2_nopivot, handle_cusparseDgtsv2_nopivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsv2_nopivot, handle_cusparseCgtsv2_nopivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsv2_nopivot, handle_cusparseZgtsv2_nopivot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsv2StridedBatch_bufferSizeExt, handle_cusparseSgtsv2StridedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsv2StridedBatch_bufferSizeExt, handle_cusparseDgtsv2StridedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsv2StridedBatch_bufferSizeExt, handle_cusparseCgtsv2StridedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsv2StridedBatch_bufferSizeExt, handle_cusparseZgtsv2StridedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsv2StridedBatch, handle_cusparseSgtsv2StridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsv2StridedBatch, handle_cusparseDgtsv2StridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsv2StridedBatch, handle_cusparseCgtsv2StridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsv2StridedBatch, handle_cusparseZgtsv2StridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsvInterleavedBatch_bufferSizeExt, handle_cusparseSgtsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsvInterleavedBatch_bufferSizeExt, handle_cusparseDgtsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsvInterleavedBatch_bufferSizeExt, handle_cusparseCgtsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsvInterleavedBatch_bufferSizeExt, handle_cusparseZgtsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgtsvInterleavedBatch, handle_cusparseSgtsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgtsvInterleavedBatch, handle_cusparseDgtsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgtsvInterleavedBatch, handle_cusparseCgtsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgtsvInterleavedBatch, handle_cusparseZgtsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgpsvInterleavedBatch_bufferSizeExt, handle_cusparseSgpsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgpsvInterleavedBatch_bufferSizeExt, handle_cusparseDgpsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgpsvInterleavedBatch_bufferSizeExt, handle_cusparseCgpsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgpsvInterleavedBatch_bufferSizeExt, handle_cusparseZgpsvInterleavedBatch_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgpsvInterleavedBatch, handle_cusparseSgpsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgpsvInterleavedBatch, handle_cusparseDgpsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgpsvInterleavedBatch, handle_cusparseCgpsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgpsvInterleavedBatch, handle_cusparseZgpsvInterleavedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrgeam2_bufferSizeExt, handle_cusparseScsrgeam2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrgeam2_bufferSizeExt, handle_cusparseDcsrgeam2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrgeam2_bufferSizeExt, handle_cusparseCcsrgeam2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrgeam2_bufferSizeExt, handle_cusparseZcsrgeam2_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsrgeam2Nnz, handle_cusparseXcsrgeam2Nnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrgeam2, handle_cusparseScsrgeam2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrgeam2, handle_cusparseDcsrgeam2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrgeam2, handle_cusparseCcsrgeam2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrgeam2, handle_cusparseZcsrgeam2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsrcolor, handle_cusparseScsrcolor, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsrcolor, handle_cusparseDcsrcolor, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsrcolor, handle_cusparseCcsrcolor, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsrcolor, handle_cusparseZcsrcolor, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSnnz, handle_cusparseSnnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnnz, handle_cusparseDnnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCnnz, handle_cusparseCnnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZnnz, handle_cusparseZnnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSnnz_compress, handle_cusparseSnnz_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnnz_compress, handle_cusparseDnnz_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCnnz_compress, handle_cusparseCnnz_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZnnz_compress, handle_cusparseZnnz_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsr2csr_compress, handle_cusparseScsr2csr_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsr2csr_compress, handle_cusparseDcsr2csr_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsr2csr_compress, handle_cusparseCcsr2csr_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsr2csr_compress, handle_cusparseZcsr2csr_compress, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcoo2csr, handle_cusparseXcoo2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsr2coo, handle_cusparseXcsr2coo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsr2bsrNnz, handle_cusparseXcsr2bsrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsr2bsr, handle_cusparseScsr2bsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsr2bsr, handle_cusparseDcsr2bsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsr2bsr, handle_cusparseCcsr2bsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsr2bsr, handle_cusparseZcsr2bsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSbsr2csr, handle_cusparseSbsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDbsr2csr, handle_cusparseDbsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCbsr2csr, handle_cusparseCbsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZbsr2csr, handle_cusparseZbsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgebsr2gebsc_bufferSize, handle_cusparseSgebsr2gebsc_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgebsr2gebsc_bufferSize, handle_cusparseDgebsr2gebsc_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgebsr2gebsc_bufferSize, handle_cusparseCgebsr2gebsc_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgebsr2gebsc_bufferSize, handle_cusparseZgebsr2gebsc_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgebsr2gebsc, handle_cusparseSgebsr2gebsc, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgebsr2gebsc, handle_cusparseDgebsr2gebsc, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgebsr2gebsc, handle_cusparseCgebsr2gebsc, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgebsr2gebsc, handle_cusparseZgebsr2gebsc, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgebsr2csr, handle_cusparseSgebsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgebsr2csr, handle_cusparseDgebsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgebsr2csr, handle_cusparseCgebsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgebsr2csr, handle_cusparseZgebsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsr2gebsr_bufferSize, handle_cusparseScsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsr2gebsr_bufferSize, handle_cusparseDcsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsr2gebsr_bufferSize, handle_cusparseCcsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsr2gebsr_bufferSize, handle_cusparseZcsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsr2gebsrNnz, handle_cusparseXcsr2gebsrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsr2gebsr, handle_cusparseScsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsr2gebsr, handle_cusparseDcsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsr2gebsr, handle_cusparseCcsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsr2gebsr, handle_cusparseZcsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgebsr2gebsr_bufferSize, handle_cusparseSgebsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgebsr2gebsr_bufferSize, handle_cusparseDgebsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgebsr2gebsr_bufferSize, handle_cusparseCgebsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgebsr2gebsr_bufferSize, handle_cusparseZgebsr2gebsr_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXgebsr2gebsrNnz, handle_cusparseXgebsr2gebsrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSgebsr2gebsr, handle_cusparseSgebsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDgebsr2gebsr, handle_cusparseDgebsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCgebsr2gebsr, handle_cusparseCgebsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZgebsr2gebsr, handle_cusparseZgebsr2gebsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateIdentityPermutation, handle_cusparseCreateIdentityPermutation, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcoosort_bufferSizeExt, handle_cusparseXcoosort_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcoosortByRow, handle_cusparseXcoosortByRow, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcoosortByColumn, handle_cusparseXcoosortByColumn, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsrsort_bufferSizeExt, handle_cusparseXcsrsort_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcsrsort, handle_cusparseXcsrsort, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcscsort_bufferSizeExt, handle_cusparseXcscsort_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseXcscsort, handle_cusparseXcscsort, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsru2csr_bufferSizeExt, handle_cusparseScsru2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsru2csr_bufferSizeExt, handle_cusparseDcsru2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsru2csr_bufferSizeExt, handle_cusparseCcsru2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsru2csr_bufferSizeExt, handle_cusparseZcsru2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsru2csr, handle_cusparseScsru2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsru2csr, handle_cusparseDcsru2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsru2csr, handle_cusparseCcsru2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsru2csr, handle_cusparseZcsru2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScsr2csru, handle_cusparseScsr2csru, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDcsr2csru, handle_cusparseDcsr2csru, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCcsr2csru, handle_cusparseCcsr2csru, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseZcsr2csru, handle_cusparseZcsr2csru, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneDense2csr_bufferSizeExt, handle_cusparseHpruneDense2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneDense2csr_bufferSizeExt, handle_cusparseSpruneDense2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneDense2csr_bufferSizeExt, handle_cusparseDpruneDense2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneDense2csrNnz, handle_cusparseHpruneDense2csrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneDense2csrNnz, handle_cusparseSpruneDense2csrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneDense2csrNnz, handle_cusparseDpruneDense2csrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneDense2csr, handle_cusparseHpruneDense2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneDense2csr, handle_cusparseSpruneDense2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneDense2csr, handle_cusparseDpruneDense2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneCsr2csr_bufferSizeExt, handle_cusparseHpruneCsr2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneCsr2csr_bufferSizeExt, handle_cusparseSpruneCsr2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneCsr2csr_bufferSizeExt, handle_cusparseDpruneCsr2csr_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneCsr2csrNnz, handle_cusparseHpruneCsr2csrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneCsr2csrNnz, handle_cusparseSpruneCsr2csrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneCsr2csrNnz, handle_cusparseDpruneCsr2csrNnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneCsr2csr, handle_cusparseHpruneCsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneCsr2csr, handle_cusparseSpruneCsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneCsr2csr, handle_cusparseDpruneCsr2csr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneDense2csrByPercentage_bufferSizeExt, handle_cusparseHpruneDense2csrByPercentage_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneDense2csrByPercentage_bufferSizeExt, handle_cusparseSpruneDense2csrByPercentage_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneDense2csrByPercentage_bufferSizeExt, handle_cusparseDpruneDense2csrByPercentage_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneDense2csrNnzByPercentage, handle_cusparseHpruneDense2csrNnzByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneDense2csrNnzByPercentage, handle_cusparseSpruneDense2csrNnzByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneDense2csrNnzByPercentage, handle_cusparseDpruneDense2csrNnzByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneDense2csrByPercentage, handle_cusparseHpruneDense2csrByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneDense2csrByPercentage, handle_cusparseSpruneDense2csrByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneDense2csrByPercentage, handle_cusparseDpruneDense2csrByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneCsr2csrByPercentage_bufferSizeExt, handle_cusparseHpruneCsr2csrByPercentage_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneCsr2csrByPercentage_bufferSizeExt, handle_cusparseSpruneCsr2csrByPercentage_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneCsr2csrByPercentage_bufferSizeExt, handle_cusparseDpruneCsr2csrByPercentage_bufferSizeExt, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneCsr2csrNnzByPercentage, handle_cusparseHpruneCsr2csrNnzByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneCsr2csrNnzByPercentage, handle_cusparseSpruneCsr2csrNnzByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneCsr2csrNnzByPercentage, handle_cusparseDpruneCsr2csrNnzByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseHpruneCsr2csrByPercentage, handle_cusparseHpruneCsr2csrByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpruneCsr2csrByPercentage, handle_cusparseSpruneCsr2csrByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDpruneCsr2csrByPercentage, handle_cusparseDpruneCsr2csrByPercentage, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCsr2cscEx2, handle_cusparseCsr2cscEx2, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCsr2cscEx2_bufferSize, handle_cusparseCsr2cscEx2_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateSpVec, handle_cusparseCreateSpVec, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroySpVec, handle_cusparseDestroySpVec, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpVecGet, handle_cusparseSpVecGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpVecGetIndexBase, handle_cusparseSpVecGetIndexBase, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpVecGetValues, handle_cusparseSpVecGetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpVecSetValues, handle_cusparseSpVecSetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateDnVec, handle_cusparseCreateDnVec, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyDnVec, handle_cusparseDestroyDnVec, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnVecGet, handle_cusparseDnVecGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnVecGetValues, handle_cusparseDnVecGetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnVecSetValues, handle_cusparseDnVecSetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroySpMat, handle_cusparseDestroySpMat, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatGetFormat, handle_cusparseSpMatGetFormat, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatGetIndexBase, handle_cusparseSpMatGetIndexBase, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatGetValues, handle_cusparseSpMatGetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatSetValues, handle_cusparseSpMatSetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatGetSize, handle_cusparseSpMatGetSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatGetStridedBatch, handle_cusparseSpMatGetStridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCooSetStridedBatch, handle_cusparseCooSetStridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCsrSetStridedBatch, handle_cusparseCsrSetStridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatGetAttribute, handle_cusparseSpMatGetAttribute, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMatSetAttribute, handle_cusparseSpMatSetAttribute, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateCsr, handle_cusparseCreateCsr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateCsc, handle_cusparseCreateCsc, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCsrGet, handle_cusparseCsrGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCscGet, handle_cusparseCscGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCsrSetPointers, handle_cusparseCsrSetPointers, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCscSetPointers, handle_cusparseCscSetPointers, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateCoo, handle_cusparseCreateCoo, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCooGet, handle_cusparseCooGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCooSetPointers, handle_cusparseCooSetPointers, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateBlockedEll, handle_cusparseCreateBlockedEll, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseBlockedEllGet, handle_cusparseBlockedEllGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseCreateDnMat, handle_cusparseCreateDnMat, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDestroyDnMat, handle_cusparseDestroyDnMat, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnMatGet, handle_cusparseDnMatGet, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnMatGetValues, handle_cusparseDnMatGetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnMatSetValues, handle_cusparseDnMatSetValues, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnMatSetStridedBatch, handle_cusparseDnMatSetStridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDnMatGetStridedBatch, handle_cusparseDnMatGetStridedBatch, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseAxpby, handle_cusparseAxpby, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseGather, handle_cusparseGather, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseScatter, handle_cusparseScatter, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseRot, handle_cusparseRot, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpVV_bufferSize, handle_cusparseSpVV_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpVV, handle_cusparseSpVV, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSparseToDense_bufferSize, handle_cusparseSparseToDense_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSparseToDense, handle_cusparseSparseToDense, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDenseToSparse_bufferSize, handle_cusparseDenseToSparse_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDenseToSparse_analysis, handle_cusparseDenseToSparse_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseDenseToSparse_convert, handle_cusparseDenseToSparse_convert, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMV, handle_cusparseSpMV, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMV_bufferSize, handle_cusparseSpMV_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSV_createDescr, handle_cusparseSpSV_createDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSV_destroyDescr, handle_cusparseSpSV_destroyDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSV_bufferSize, handle_cusparseSpSV_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSV_analysis, handle_cusparseSpSV_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSV_solve, handle_cusparseSpSV_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSM_createDescr, handle_cusparseSpSM_createDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSM_destroyDescr, handle_cusparseSpSM_destroyDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSM_bufferSize, handle_cusparseSpSM_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSM_analysis, handle_cusparseSpSM_analysis, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpSM_solve, handle_cusparseSpSM_solve, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMM_bufferSize, handle_cusparseSpMM_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMM_preprocess, handle_cusparseSpMM_preprocess, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMM, handle_cusparseSpMM, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMM_createDescr, handle_cusparseSpGEMM_createDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMM_destroyDescr, handle_cusparseSpGEMM_destroyDescr, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMM_workEstimation, handle_cusparseSpGEMM_workEstimation, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMM_compute, handle_cusparseSpGEMM_compute, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMM_copy, handle_cusparseSpGEMM_copy, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMMreuse_workEstimation, handle_cusparseSpGEMMreuse_workEstimation, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMMreuse_nnz, handle_cusparseSpGEMMreuse_nnz, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMMreuse_copy, handle_cusparseSpGEMMreuse_copy, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpGEMMreuse_compute, handle_cusparseSpGEMMreuse_compute, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSDDMM_bufferSize, handle_cusparseSDDMM_bufferSize, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSDDMM_preprocess, handle_cusparseSDDMM_preprocess, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSDDMM, handle_cusparseSDDMM, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMMOp_createPlan, handle_cusparseSpMMOp_createPlan, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMMOp, handle_cusparseSpMMOp, rpc_backend::cusparse) \
  HANDLER(RPC_cusparseSpMMOp_destroyPlan, handle_cusparseSpMMOp_destroyPlan, rpc_backend::cusparse)
#define LUPINE_CUSOLVER_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cusolverDnIRSInfosGetResidualHistory, handle_cusolverDnIRSInfosGetResidualHistory, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfExtractBundledFactorsHost, handle_cusolverRfExtractBundledFactorsHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfExtractSplitFactorsHost, handle_cusolverRfExtractSplitFactorsHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfBatchSetupHost, handle_cusolverRfBatchSetupHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverGetProperty, handle_cusolverGetProperty, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverGetVersion, handle_cusolverGetVersion, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCreate, handle_cusolverDnCreate, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDestroy, handle_cusolverDnDestroy, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSetStream, handle_cusolverDnSetStream, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnGetStream, handle_cusolverDnGetStream, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsCreate, handle_cusolverDnIRSParamsCreate, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsDestroy, handle_cusolverDnIRSParamsDestroy, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetRefinementSolver, handle_cusolverDnIRSParamsSetRefinementSolver, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetSolverMainPrecision, handle_cusolverDnIRSParamsSetSolverMainPrecision, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetSolverLowestPrecision, handle_cusolverDnIRSParamsSetSolverLowestPrecision, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetSolverPrecisions, handle_cusolverDnIRSParamsSetSolverPrecisions, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetTol, handle_cusolverDnIRSParamsSetTol, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetTolInner, handle_cusolverDnIRSParamsSetTolInner, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetMaxIters, handle_cusolverDnIRSParamsSetMaxIters, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsSetMaxItersInner, handle_cusolverDnIRSParamsSetMaxItersInner, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsGetMaxIters, handle_cusolverDnIRSParamsGetMaxIters, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsEnableFallback, handle_cusolverDnIRSParamsEnableFallback, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSParamsDisableFallback, handle_cusolverDnIRSParamsDisableFallback, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSInfosDestroy, handle_cusolverDnIRSInfosDestroy, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSInfosCreate, handle_cusolverDnIRSInfosCreate, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSInfosGetNiters, handle_cusolverDnIRSInfosGetNiters, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSInfosGetOuterNiters, handle_cusolverDnIRSInfosGetOuterNiters, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSInfosRequestResidual, handle_cusolverDnIRSInfosRequestResidual, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSInfosGetMaxIters, handle_cusolverDnIRSInfosGetMaxIters, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZZgesv, handle_cusolverDnZZgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZCgesv, handle_cusolverDnZCgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZKgesv, handle_cusolverDnZKgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZEgesv, handle_cusolverDnZEgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZYgesv, handle_cusolverDnZYgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCCgesv, handle_cusolverDnCCgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCEgesv, handle_cusolverDnCEgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCKgesv, handle_cusolverDnCKgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCYgesv, handle_cusolverDnCYgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDDgesv, handle_cusolverDnDDgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDSgesv, handle_cusolverDnDSgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDHgesv, handle_cusolverDnDHgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDBgesv, handle_cusolverDnDBgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDXgesv, handle_cusolverDnDXgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSSgesv, handle_cusolverDnSSgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSHgesv, handle_cusolverDnSHgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSBgesv, handle_cusolverDnSBgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSXgesv, handle_cusolverDnSXgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZZgesv_bufferSize, handle_cusolverDnZZgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZCgesv_bufferSize, handle_cusolverDnZCgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZKgesv_bufferSize, handle_cusolverDnZKgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZEgesv_bufferSize, handle_cusolverDnZEgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZYgesv_bufferSize, handle_cusolverDnZYgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCCgesv_bufferSize, handle_cusolverDnCCgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCKgesv_bufferSize, handle_cusolverDnCKgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCEgesv_bufferSize, handle_cusolverDnCEgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCYgesv_bufferSize, handle_cusolverDnCYgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDDgesv_bufferSize, handle_cusolverDnDDgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDSgesv_bufferSize, handle_cusolverDnDSgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDHgesv_bufferSize, handle_cusolverDnDHgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDBgesv_bufferSize, handle_cusolverDnDBgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDXgesv_bufferSize, handle_cusolverDnDXgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSSgesv_bufferSize, handle_cusolverDnSSgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSHgesv_bufferSize, handle_cusolverDnSHgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSBgesv_bufferSize, handle_cusolverDnSBgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSXgesv_bufferSize, handle_cusolverDnSXgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZZgels, handle_cusolverDnZZgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZCgels, handle_cusolverDnZCgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZKgels, handle_cusolverDnZKgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZEgels, handle_cusolverDnZEgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZYgels, handle_cusolverDnZYgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCCgels, handle_cusolverDnCCgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCKgels, handle_cusolverDnCKgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCEgels, handle_cusolverDnCEgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCYgels, handle_cusolverDnCYgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDDgels, handle_cusolverDnDDgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDSgels, handle_cusolverDnDSgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDHgels, handle_cusolverDnDHgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDBgels, handle_cusolverDnDBgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDXgels, handle_cusolverDnDXgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSSgels, handle_cusolverDnSSgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSHgels, handle_cusolverDnSHgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSBgels, handle_cusolverDnSBgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSXgels, handle_cusolverDnSXgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZZgels_bufferSize, handle_cusolverDnZZgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZCgels_bufferSize, handle_cusolverDnZCgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZKgels_bufferSize, handle_cusolverDnZKgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZEgels_bufferSize, handle_cusolverDnZEgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZYgels_bufferSize, handle_cusolverDnZYgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCCgels_bufferSize, handle_cusolverDnCCgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCKgels_bufferSize, handle_cusolverDnCKgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCEgels_bufferSize, handle_cusolverDnCEgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCYgels_bufferSize, handle_cusolverDnCYgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDDgels_bufferSize, handle_cusolverDnDDgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDSgels_bufferSize, handle_cusolverDnDSgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDHgels_bufferSize, handle_cusolverDnDHgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDBgels_bufferSize, handle_cusolverDnDBgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDXgels_bufferSize, handle_cusolverDnDXgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSSgels_bufferSize, handle_cusolverDnSSgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSHgels_bufferSize, handle_cusolverDnSHgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSBgels_bufferSize, handle_cusolverDnSBgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSXgels_bufferSize, handle_cusolverDnSXgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSXgesv, handle_cusolverDnIRSXgesv, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSXgesv_bufferSize, handle_cusolverDnIRSXgesv_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSXgels, handle_cusolverDnIRSXgels, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnIRSXgels_bufferSize, handle_cusolverDnIRSXgels_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotrf_bufferSize, handle_cusolverDnSpotrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotrf_bufferSize, handle_cusolverDnDpotrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotrf_bufferSize, handle_cusolverDnCpotrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotrf_bufferSize, handle_cusolverDnZpotrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotrf, handle_cusolverDnSpotrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotrf, handle_cusolverDnDpotrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotrf, handle_cusolverDnCpotrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotrf, handle_cusolverDnZpotrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotrs, handle_cusolverDnSpotrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotrs, handle_cusolverDnDpotrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotrs, handle_cusolverDnCpotrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotrs, handle_cusolverDnZpotrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotrfBatched, handle_cusolverDnSpotrfBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotrfBatched, handle_cusolverDnDpotrfBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotrfBatched, handle_cusolverDnCpotrfBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotrfBatched, handle_cusolverDnZpotrfBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotrsBatched, handle_cusolverDnSpotrsBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotrsBatched, handle_cusolverDnDpotrsBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotrsBatched, handle_cusolverDnCpotrsBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotrsBatched, handle_cusolverDnZpotrsBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotri_bufferSize, handle_cusolverDnSpotri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotri_bufferSize, handle_cusolverDnDpotri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotri_bufferSize, handle_cusolverDnCpotri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotri_bufferSize, handle_cusolverDnZpotri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSpotri, handle_cusolverDnSpotri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDpotri, handle_cusolverDnDpotri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCpotri, handle_cusolverDnCpotri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZpotri, handle_cusolverDnZpotri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXtrtri_bufferSize, handle_cusolverDnXtrtri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXtrtri, handle_cusolverDnXtrtri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSlauum_bufferSize, handle_cusolverDnSlauum_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDlauum_bufferSize, handle_cusolverDnDlauum_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnClauum_bufferSize, handle_cusolverDnClauum_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZlauum_bufferSize, handle_cusolverDnZlauum_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSlauum, handle_cusolverDnSlauum, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDlauum, handle_cusolverDnDlauum, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnClauum, handle_cusolverDnClauum, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZlauum, handle_cusolverDnZlauum, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgetrf_bufferSize, handle_cusolverDnSgetrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgetrf_bufferSize, handle_cusolverDnDgetrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgetrf_bufferSize, handle_cusolverDnCgetrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgetrf_bufferSize, handle_cusolverDnZgetrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgetrf, handle_cusolverDnSgetrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgetrf, handle_cusolverDnDgetrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgetrf, handle_cusolverDnCgetrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgetrf, handle_cusolverDnZgetrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSlaswp, handle_cusolverDnSlaswp, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDlaswp, handle_cusolverDnDlaswp, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnClaswp, handle_cusolverDnClaswp, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZlaswp, handle_cusolverDnZlaswp, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgetrs, handle_cusolverDnSgetrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgetrs, handle_cusolverDnDgetrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgetrs, handle_cusolverDnCgetrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgetrs, handle_cusolverDnZgetrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgeqrf_bufferSize, handle_cusolverDnSgeqrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgeqrf_bufferSize, handle_cusolverDnDgeqrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgeqrf_bufferSize, handle_cusolverDnCgeqrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgeqrf_bufferSize, handle_cusolverDnZgeqrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgeqrf, handle_cusolverDnSgeqrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgeqrf, handle_cusolverDnDgeqrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgeqrf, handle_cusolverDnCgeqrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgeqrf, handle_cusolverDnZgeqrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSorgqr_bufferSize, handle_cusolverDnSorgqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDorgqr_bufferSize, handle_cusolverDnDorgqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCungqr_bufferSize, handle_cusolverDnCungqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZungqr_bufferSize, handle_cusolverDnZungqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSorgqr, handle_cusolverDnSorgqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDorgqr, handle_cusolverDnDorgqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCungqr, handle_cusolverDnCungqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZungqr, handle_cusolverDnZungqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSormqr_bufferSize, handle_cusolverDnSormqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDormqr_bufferSize, handle_cusolverDnDormqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCunmqr_bufferSize, handle_cusolverDnCunmqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZunmqr_bufferSize, handle_cusolverDnZunmqr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSormqr, handle_cusolverDnSormqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDormqr, handle_cusolverDnDormqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCunmqr, handle_cusolverDnCunmqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZunmqr, handle_cusolverDnZunmqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsytrf_bufferSize, handle_cusolverDnSsytrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsytrf_bufferSize, handle_cusolverDnDsytrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCsytrf_bufferSize, handle_cusolverDnCsytrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZsytrf_bufferSize, handle_cusolverDnZsytrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsytrf, handle_cusolverDnSsytrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsytrf, handle_cusolverDnDsytrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCsytrf, handle_cusolverDnCsytrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZsytrf, handle_cusolverDnZsytrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsytrs_bufferSize, handle_cusolverDnXsytrs_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsytrs, handle_cusolverDnXsytrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsytri_bufferSize, handle_cusolverDnSsytri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsytri_bufferSize, handle_cusolverDnDsytri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCsytri_bufferSize, handle_cusolverDnCsytri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZsytri_bufferSize, handle_cusolverDnZsytri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsytri, handle_cusolverDnSsytri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsytri, handle_cusolverDnDsytri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCsytri, handle_cusolverDnCsytri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZsytri, handle_cusolverDnZsytri, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgebrd_bufferSize, handle_cusolverDnSgebrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgebrd_bufferSize, handle_cusolverDnDgebrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgebrd_bufferSize, handle_cusolverDnCgebrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgebrd_bufferSize, handle_cusolverDnZgebrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgebrd, handle_cusolverDnSgebrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgebrd, handle_cusolverDnDgebrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgebrd, handle_cusolverDnCgebrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgebrd, handle_cusolverDnZgebrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSorgbr_bufferSize, handle_cusolverDnSorgbr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDorgbr_bufferSize, handle_cusolverDnDorgbr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCungbr_bufferSize, handle_cusolverDnCungbr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZungbr_bufferSize, handle_cusolverDnZungbr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSorgbr, handle_cusolverDnSorgbr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDorgbr, handle_cusolverDnDorgbr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCungbr, handle_cusolverDnCungbr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZungbr, handle_cusolverDnZungbr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsytrd_bufferSize, handle_cusolverDnSsytrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsytrd_bufferSize, handle_cusolverDnDsytrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChetrd_bufferSize, handle_cusolverDnChetrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhetrd_bufferSize, handle_cusolverDnZhetrd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsytrd, handle_cusolverDnSsytrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsytrd, handle_cusolverDnDsytrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChetrd, handle_cusolverDnChetrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhetrd, handle_cusolverDnZhetrd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSorgtr_bufferSize, handle_cusolverDnSorgtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDorgtr_bufferSize, handle_cusolverDnDorgtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCungtr_bufferSize, handle_cusolverDnCungtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZungtr_bufferSize, handle_cusolverDnZungtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSorgtr, handle_cusolverDnSorgtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDorgtr, handle_cusolverDnDorgtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCungtr, handle_cusolverDnCungtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZungtr, handle_cusolverDnZungtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSormtr_bufferSize, handle_cusolverDnSormtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDormtr_bufferSize, handle_cusolverDnDormtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCunmtr_bufferSize, handle_cusolverDnCunmtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZunmtr_bufferSize, handle_cusolverDnZunmtr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSormtr, handle_cusolverDnSormtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDormtr, handle_cusolverDnDormtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCunmtr, handle_cusolverDnCunmtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZunmtr, handle_cusolverDnZunmtr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvd_bufferSize, handle_cusolverDnSgesvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvd_bufferSize, handle_cusolverDnDgesvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvd_bufferSize, handle_cusolverDnCgesvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvd_bufferSize, handle_cusolverDnZgesvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvd, handle_cusolverDnSgesvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvd, handle_cusolverDnDgesvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvd, handle_cusolverDnCgesvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvd, handle_cusolverDnZgesvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevd_bufferSize, handle_cusolverDnSsyevd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevd_bufferSize, handle_cusolverDnDsyevd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevd_bufferSize, handle_cusolverDnCheevd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevd_bufferSize, handle_cusolverDnZheevd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevd, handle_cusolverDnSsyevd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevd, handle_cusolverDnDsyevd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevd, handle_cusolverDnCheevd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevd, handle_cusolverDnZheevd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevdx_bufferSize, handle_cusolverDnSsyevdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevdx_bufferSize, handle_cusolverDnDsyevdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevdx_bufferSize, handle_cusolverDnCheevdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevdx_bufferSize, handle_cusolverDnZheevdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevdx, handle_cusolverDnSsyevdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevdx, handle_cusolverDnDsyevdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevdx, handle_cusolverDnCheevdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevdx, handle_cusolverDnZheevdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsygvdx_bufferSize, handle_cusolverDnSsygvdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsygvdx_bufferSize, handle_cusolverDnDsygvdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChegvdx_bufferSize, handle_cusolverDnChegvdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhegvdx_bufferSize, handle_cusolverDnZhegvdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsygvdx, handle_cusolverDnSsygvdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsygvdx, handle_cusolverDnDsygvdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChegvdx, handle_cusolverDnChegvdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhegvdx, handle_cusolverDnZhegvdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsygvd_bufferSize, handle_cusolverDnSsygvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsygvd_bufferSize, handle_cusolverDnDsygvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChegvd_bufferSize, handle_cusolverDnChegvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhegvd_bufferSize, handle_cusolverDnZhegvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsygvd, handle_cusolverDnSsygvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsygvd, handle_cusolverDnDsygvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChegvd, handle_cusolverDnChegvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhegvd, handle_cusolverDnZhegvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCreateSyevjInfo, handle_cusolverDnCreateSyevjInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDestroySyevjInfo, handle_cusolverDnDestroySyevjInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevjSetTolerance, handle_cusolverDnXsyevjSetTolerance, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevjSetMaxSweeps, handle_cusolverDnXsyevjSetMaxSweeps, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevjSetSortEig, handle_cusolverDnXsyevjSetSortEig, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevjGetResidual, handle_cusolverDnXsyevjGetResidual, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevjGetSweeps, handle_cusolverDnXsyevjGetSweeps, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevjBatched_bufferSize, handle_cusolverDnSsyevjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevjBatched_bufferSize, handle_cusolverDnDsyevjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevjBatched_bufferSize, handle_cusolverDnCheevjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevjBatched_bufferSize, handle_cusolverDnZheevjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevjBatched, handle_cusolverDnSsyevjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevjBatched, handle_cusolverDnDsyevjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevjBatched, handle_cusolverDnCheevjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevjBatched, handle_cusolverDnZheevjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevj_bufferSize, handle_cusolverDnSsyevj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevj_bufferSize, handle_cusolverDnDsyevj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevj_bufferSize, handle_cusolverDnCheevj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevj_bufferSize, handle_cusolverDnZheevj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsyevj, handle_cusolverDnSsyevj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsyevj, handle_cusolverDnDsyevj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCheevj, handle_cusolverDnCheevj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZheevj, handle_cusolverDnZheevj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsygvj_bufferSize, handle_cusolverDnSsygvj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsygvj_bufferSize, handle_cusolverDnDsygvj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChegvj_bufferSize, handle_cusolverDnChegvj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhegvj_bufferSize, handle_cusolverDnZhegvj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSsygvj, handle_cusolverDnSsygvj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDsygvj, handle_cusolverDnDsygvj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnChegvj, handle_cusolverDnChegvj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZhegvj, handle_cusolverDnZhegvj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCreateGesvdjInfo, handle_cusolverDnCreateGesvdjInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDestroyGesvdjInfo, handle_cusolverDnDestroyGesvdjInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdjSetTolerance, handle_cusolverDnXgesvdjSetTolerance, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdjSetMaxSweeps, handle_cusolverDnXgesvdjSetMaxSweeps, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdjSetSortEig, handle_cusolverDnXgesvdjSetSortEig, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdjGetResidual, handle_cusolverDnXgesvdjGetResidual, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdjGetSweeps, handle_cusolverDnXgesvdjGetSweeps, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvdjBatched_bufferSize, handle_cusolverDnSgesvdjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvdjBatched_bufferSize, handle_cusolverDnDgesvdjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvdjBatched_bufferSize, handle_cusolverDnCgesvdjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvdjBatched_bufferSize, handle_cusolverDnZgesvdjBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvdjBatched, handle_cusolverDnSgesvdjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvdjBatched, handle_cusolverDnDgesvdjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvdjBatched, handle_cusolverDnCgesvdjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvdjBatched, handle_cusolverDnZgesvdjBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvdj_bufferSize, handle_cusolverDnSgesvdj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvdj_bufferSize, handle_cusolverDnDgesvdj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvdj_bufferSize, handle_cusolverDnCgesvdj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvdj_bufferSize, handle_cusolverDnZgesvdj_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvdj, handle_cusolverDnSgesvdj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvdj, handle_cusolverDnDgesvdj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvdj, handle_cusolverDnCgesvdj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvdj, handle_cusolverDnZgesvdj, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvdaStridedBatched_bufferSize, handle_cusolverDnSgesvdaStridedBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvdaStridedBatched_bufferSize, handle_cusolverDnDgesvdaStridedBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvdaStridedBatched_bufferSize, handle_cusolverDnCgesvdaStridedBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvdaStridedBatched_bufferSize, handle_cusolverDnZgesvdaStridedBatched_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSgesvdaStridedBatched, handle_cusolverDnSgesvdaStridedBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDgesvdaStridedBatched, handle_cusolverDnDgesvdaStridedBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCgesvdaStridedBatched, handle_cusolverDnCgesvdaStridedBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnZgesvdaStridedBatched, handle_cusolverDnZgesvdaStridedBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnCreateParams, handle_cusolverDnCreateParams, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnDestroyParams, handle_cusolverDnDestroyParams, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnSetAdvOptions, handle_cusolverDnSetAdvOptions, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXpotrf_bufferSize, handle_cusolverDnXpotrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXpotrf, handle_cusolverDnXpotrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXpotrs, handle_cusolverDnXpotrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgeqrf_bufferSize, handle_cusolverDnXgeqrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgeqrf, handle_cusolverDnXgeqrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgetrf_bufferSize, handle_cusolverDnXgetrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgetrf, handle_cusolverDnXgetrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgetrs, handle_cusolverDnXgetrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevd_bufferSize, handle_cusolverDnXsyevd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevd, handle_cusolverDnXsyevd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevdx_bufferSize, handle_cusolverDnXsyevdx_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXsyevdx, handle_cusolverDnXsyevdx, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvd_bufferSize, handle_cusolverDnXgesvd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvd, handle_cusolverDnXgesvd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdp_bufferSize, handle_cusolverDnXgesvdp_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdp, handle_cusolverDnXgesvdp, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdr_bufferSize, handle_cusolverDnXgesvdr_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverDnXgesvdr, handle_cusolverDnXgesvdr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCreate, handle_cusolverSpCreate, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDestroy, handle_cusolverSpDestroy, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpSetStream, handle_cusolverSpSetStream, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpGetStream, handle_cusolverSpGetStream, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrissymHost, handle_cusolverSpXcsrissymHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrlsvluHost, handle_cusolverSpScsrlsvluHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrlsvluHost, handle_cusolverSpDcsrlsvluHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrlsvluHost, handle_cusolverSpCcsrlsvluHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrlsvluHost, handle_cusolverSpZcsrlsvluHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrlsvqr, handle_cusolverSpScsrlsvqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrlsvqr, handle_cusolverSpDcsrlsvqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrlsvqr, handle_cusolverSpCcsrlsvqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrlsvqr, handle_cusolverSpZcsrlsvqr, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrlsvqrHost, handle_cusolverSpScsrlsvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrlsvqrHost, handle_cusolverSpDcsrlsvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrlsvqrHost, handle_cusolverSpCcsrlsvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrlsvqrHost, handle_cusolverSpZcsrlsvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrlsvcholHost, handle_cusolverSpScsrlsvcholHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrlsvcholHost, handle_cusolverSpDcsrlsvcholHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrlsvcholHost, handle_cusolverSpCcsrlsvcholHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrlsvcholHost, handle_cusolverSpZcsrlsvcholHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrlsvchol, handle_cusolverSpScsrlsvchol, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrlsvchol, handle_cusolverSpDcsrlsvchol, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrlsvchol, handle_cusolverSpCcsrlsvchol, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrlsvchol, handle_cusolverSpZcsrlsvchol, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrlsqvqrHost, handle_cusolverSpScsrlsqvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrlsqvqrHost, handle_cusolverSpDcsrlsqvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrlsqvqrHost, handle_cusolverSpCcsrlsqvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrlsqvqrHost, handle_cusolverSpZcsrlsqvqrHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsreigvsiHost, handle_cusolverSpScsreigvsiHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsreigvsiHost, handle_cusolverSpDcsreigvsiHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsreigvsiHost, handle_cusolverSpCcsreigvsiHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsreigvsiHost, handle_cusolverSpZcsreigvsiHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsreigvsi, handle_cusolverSpScsreigvsi, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsreigvsi, handle_cusolverSpDcsreigvsi, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsreigvsi, handle_cusolverSpCcsreigvsi, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsreigvsi, handle_cusolverSpZcsreigvsi, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsreigsHost, handle_cusolverSpScsreigsHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsreigsHost, handle_cusolverSpDcsreigsHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsreigsHost, handle_cusolverSpCcsreigsHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsreigsHost, handle_cusolverSpZcsreigsHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrsymrcmHost, handle_cusolverSpXcsrsymrcmHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrsymmdqHost, handle_cusolverSpXcsrsymmdqHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrsymamdHost, handle_cusolverSpXcsrsymamdHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrmetisndHost, handle_cusolverSpXcsrmetisndHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrzfdHost, handle_cusolverSpScsrzfdHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrzfdHost, handle_cusolverSpDcsrzfdHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrzfdHost, handle_cusolverSpCcsrzfdHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrzfdHost, handle_cusolverSpZcsrzfdHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrperm_bufferSizeHost, handle_cusolverSpXcsrperm_bufferSizeHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrpermHost, handle_cusolverSpXcsrpermHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCreateCsrqrInfo, handle_cusolverSpCreateCsrqrInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDestroyCsrqrInfo, handle_cusolverSpDestroyCsrqrInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrqrAnalysisBatched, handle_cusolverSpXcsrqrAnalysisBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrBufferInfoBatched, handle_cusolverSpScsrqrBufferInfoBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrBufferInfoBatched, handle_cusolverSpDcsrqrBufferInfoBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrBufferInfoBatched, handle_cusolverSpCcsrqrBufferInfoBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrBufferInfoBatched, handle_cusolverSpZcsrqrBufferInfoBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrsvBatched, handle_cusolverSpScsrqrsvBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrsvBatched, handle_cusolverSpDcsrqrsvBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrsvBatched, handle_cusolverSpCcsrqrsvBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrsvBatched, handle_cusolverSpZcsrqrsvBatched, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCreateCsrluInfoHost, handle_cusolverSpCreateCsrluInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDestroyCsrluInfoHost, handle_cusolverSpDestroyCsrluInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrluAnalysisHost, handle_cusolverSpXcsrluAnalysisHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrluBufferInfoHost, handle_cusolverSpScsrluBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrluBufferInfoHost, handle_cusolverSpDcsrluBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrluBufferInfoHost, handle_cusolverSpCcsrluBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrluBufferInfoHost, handle_cusolverSpZcsrluBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrluFactorHost, handle_cusolverSpScsrluFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrluFactorHost, handle_cusolverSpDcsrluFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrluFactorHost, handle_cusolverSpCcsrluFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrluFactorHost, handle_cusolverSpZcsrluFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrluZeroPivotHost, handle_cusolverSpScsrluZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrluZeroPivotHost, handle_cusolverSpDcsrluZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrluZeroPivotHost, handle_cusolverSpCcsrluZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrluZeroPivotHost, handle_cusolverSpZcsrluZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrluSolveHost, handle_cusolverSpScsrluSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrluSolveHost, handle_cusolverSpDcsrluSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrluSolveHost, handle_cusolverSpCcsrluSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrluSolveHost, handle_cusolverSpZcsrluSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrluNnzHost, handle_cusolverSpXcsrluNnzHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrluExtractHost, handle_cusolverSpScsrluExtractHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrluExtractHost, handle_cusolverSpDcsrluExtractHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrluExtractHost, handle_cusolverSpCcsrluExtractHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrluExtractHost, handle_cusolverSpZcsrluExtractHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCreateCsrqrInfoHost, handle_cusolverSpCreateCsrqrInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDestroyCsrqrInfoHost, handle_cusolverSpDestroyCsrqrInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrqrAnalysisHost, handle_cusolverSpXcsrqrAnalysisHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrBufferInfoHost, handle_cusolverSpScsrqrBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrBufferInfoHost, handle_cusolverSpDcsrqrBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrBufferInfoHost, handle_cusolverSpCcsrqrBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrBufferInfoHost, handle_cusolverSpZcsrqrBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrSetupHost, handle_cusolverSpScsrqrSetupHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrSetupHost, handle_cusolverSpDcsrqrSetupHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrSetupHost, handle_cusolverSpCcsrqrSetupHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrSetupHost, handle_cusolverSpZcsrqrSetupHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrFactorHost, handle_cusolverSpScsrqrFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrFactorHost, handle_cusolverSpDcsrqrFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrFactorHost, handle_cusolverSpCcsrqrFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrFactorHost, handle_cusolverSpZcsrqrFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrZeroPivotHost, handle_cusolverSpScsrqrZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrZeroPivotHost, handle_cusolverSpDcsrqrZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrZeroPivotHost, handle_cusolverSpCcsrqrZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrZeroPivotHost, handle_cusolverSpZcsrqrZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrSolveHost, handle_cusolverSpScsrqrSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrSolveHost, handle_cusolverSpDcsrqrSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrSolveHost, handle_cusolverSpCcsrqrSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrSolveHost, handle_cusolverSpZcsrqrSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrqrAnalysis, handle_cusolverSpXcsrqrAnalysis, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrBufferInfo, handle_cusolverSpScsrqrBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrBufferInfo, handle_cusolverSpDcsrqrBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrBufferInfo, handle_cusolverSpCcsrqrBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrBufferInfo, handle_cusolverSpZcsrqrBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrSetup, handle_cusolverSpScsrqrSetup, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrSetup, handle_cusolverSpDcsrqrSetup, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrSetup, handle_cusolverSpCcsrqrSetup, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrSetup, handle_cusolverSpZcsrqrSetup, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrFactor, handle_cusolverSpScsrqrFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrFactor, handle_cusolverSpDcsrqrFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrFactor, handle_cusolverSpCcsrqrFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrFactor, handle_cusolverSpZcsrqrFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrZeroPivot, handle_cusolverSpScsrqrZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrZeroPivot, handle_cusolverSpDcsrqrZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrZeroPivot, handle_cusolverSpCcsrqrZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrZeroPivot, handle_cusolverSpZcsrqrZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrqrSolve, handle_cusolverSpScsrqrSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrqrSolve, handle_cusolverSpDcsrqrSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrqrSolve, handle_cusolverSpCcsrqrSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrqrSolve, handle_cusolverSpZcsrqrSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCreateCsrcholInfoHost, handle_cusolverSpCreateCsrcholInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDestroyCsrcholInfoHost, handle_cusolverSpDestroyCsrcholInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrcholAnalysisHost, handle_cusolverSpXcsrcholAnalysisHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholBufferInfoHost, handle_cusolverSpScsrcholBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholBufferInfoHost, handle_cusolverSpDcsrcholBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholBufferInfoHost, handle_cusolverSpCcsrcholBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholBufferInfoHost, handle_cusolverSpZcsrcholBufferInfoHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholFactorHost, handle_cusolverSpScsrcholFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholFactorHost, handle_cusolverSpDcsrcholFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholFactorHost, handle_cusolverSpCcsrcholFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholFactorHost, handle_cusolverSpZcsrcholFactorHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholZeroPivotHost, handle_cusolverSpScsrcholZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholZeroPivotHost, handle_cusolverSpDcsrcholZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholZeroPivotHost, handle_cusolverSpCcsrcholZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholZeroPivotHost, handle_cusolverSpZcsrcholZeroPivotHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholSolveHost, handle_cusolverSpScsrcholSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholSolveHost, handle_cusolverSpDcsrcholSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholSolveHost, handle_cusolverSpCcsrcholSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholSolveHost, handle_cusolverSpZcsrcholSolveHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCreateCsrcholInfo, handle_cusolverSpCreateCsrcholInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDestroyCsrcholInfo, handle_cusolverSpDestroyCsrcholInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpXcsrcholAnalysis, handle_cusolverSpXcsrcholAnalysis, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholBufferInfo, handle_cusolverSpScsrcholBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholBufferInfo, handle_cusolverSpDcsrcholBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholBufferInfo, handle_cusolverSpCcsrcholBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholBufferInfo, handle_cusolverSpZcsrcholBufferInfo, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholFactor, handle_cusolverSpScsrcholFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholFactor, handle_cusolverSpDcsrcholFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholFactor, handle_cusolverSpCcsrcholFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholFactor, handle_cusolverSpZcsrcholFactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholZeroPivot, handle_cusolverSpScsrcholZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholZeroPivot, handle_cusolverSpDcsrcholZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholZeroPivot, handle_cusolverSpCcsrcholZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholZeroPivot, handle_cusolverSpZcsrcholZeroPivot, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholSolve, handle_cusolverSpScsrcholSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholSolve, handle_cusolverSpDcsrcholSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholSolve, handle_cusolverSpCcsrcholSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholSolve, handle_cusolverSpZcsrcholSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpScsrcholDiag, handle_cusolverSpScsrcholDiag, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpDcsrcholDiag, handle_cusolverSpDcsrcholDiag, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpCcsrcholDiag, handle_cusolverSpCcsrcholDiag, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverSpZcsrcholDiag, handle_cusolverSpZcsrcholDiag, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfCreate, handle_cusolverRfCreate, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfDestroy, handle_cusolverRfDestroy, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfGetMatrixFormat, handle_cusolverRfGetMatrixFormat, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSetMatrixFormat, handle_cusolverRfSetMatrixFormat, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSetNumericProperties, handle_cusolverRfSetNumericProperties, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfGetNumericProperties, handle_cusolverRfGetNumericProperties, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfGetNumericBoostReport, handle_cusolverRfGetNumericBoostReport, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSetAlgs, handle_cusolverRfSetAlgs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfGetAlgs, handle_cusolverRfGetAlgs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfGetResetValuesFastMode, handle_cusolverRfGetResetValuesFastMode, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSetResetValuesFastMode, handle_cusolverRfSetResetValuesFastMode, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSetupHost, handle_cusolverRfSetupHost, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSetupDevice, handle_cusolverRfSetupDevice, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfResetValues, handle_cusolverRfResetValues, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfAnalyze, handle_cusolverRfAnalyze, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfRefactor, handle_cusolverRfRefactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfAccessBundledFactorsDevice, handle_cusolverRfAccessBundledFactorsDevice, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfSolve, handle_cusolverRfSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfBatchResetValues, handle_cusolverRfBatchResetValues, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfBatchAnalyze, handle_cusolverRfBatchAnalyze, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfBatchRefactor, handle_cusolverRfBatchRefactor, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfBatchSolve, handle_cusolverRfBatchSolve, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverRfBatchZeroPivot, handle_cusolverRfBatchZeroPivot, rpc_backend::cusolver)
#define LUPINE_CUSOLVERMG_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_cusolverMgCreate, handle_cusolverMgCreate, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgDestroy, handle_cusolverMgDestroy, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgDeviceSelect, handle_cusolverMgDeviceSelect, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgCreateDeviceGrid, handle_cusolverMgCreateDeviceGrid, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgDestroyGrid, handle_cusolverMgDestroyGrid, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgCreateMatrixDesc, handle_cusolverMgCreateMatrixDesc, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgDestroyMatrixDesc, handle_cusolverMgDestroyMatrixDesc, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgSyevd_bufferSize, handle_cusolverMgSyevd_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgSyevd, handle_cusolverMgSyevd, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgGetrf_bufferSize, handle_cusolverMgGetrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgGetrf, handle_cusolverMgGetrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgGetrs_bufferSize, handle_cusolverMgGetrs_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgGetrs, handle_cusolverMgGetrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgPotrf_bufferSize, handle_cusolverMgPotrf_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgPotrf, handle_cusolverMgPotrf, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgPotrs_bufferSize, handle_cusolverMgPotrs_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgPotrs, handle_cusolverMgPotrs, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgPotri_bufferSize, handle_cusolverMgPotri_bufferSize, rpc_backend::cusolver) \
  HANDLER(RPC_cusolverMgPotri, handle_cusolverMgPotri, rpc_backend::cusolver)
#define LUPINE_NVRTC_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_nvrtcGetErrorString, handle_nvrtcGetErrorString, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetSupportedArchs, handle_nvrtcGetSupportedArchs, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcCreateProgram, handle_nvrtcCreateProgram, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcCompileProgram, handle_nvrtcCompileProgram, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetPTX, handle_nvrtcGetPTX, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetCUBIN, handle_nvrtcGetCUBIN, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetProgramLog, handle_nvrtcGetProgramLog, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetLoweredName, handle_nvrtcGetLoweredName, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcVersion, handle_nvrtcVersion, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetNumSupportedArchs, handle_nvrtcGetNumSupportedArchs, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcDestroyProgram, handle_nvrtcDestroyProgram, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetPTXSize, handle_nvrtcGetPTXSize, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetCUBINSize, handle_nvrtcGetCUBINSize, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcGetProgramLogSize, handle_nvrtcGetProgramLogSize, rpc_backend::nvrtc) \
  HANDLER(RPC_nvrtcAddNameExpression, handle_nvrtcAddNameExpression, rpc_backend::nvrtc)
#define LUPINE_NCCL_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_ncclGetErrorString, handle_ncclGetErrorString, rpc_backend::nccl) \
  HANDLER(RPC_ncclGetLastError, handle_ncclGetLastError, rpc_backend::nccl) \
  HANDLER(RPC_ncclGetVersion, handle_ncclGetVersion, rpc_backend::nccl) \
  HANDLER(RPC_ncclGetUniqueId, handle_ncclGetUniqueId, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommInitRankConfig, handle_ncclCommInitRankConfig, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommInitRank, handle_ncclCommInitRank, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommInitAll, handle_ncclCommInitAll, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommFinalize, handle_ncclCommFinalize, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommDestroy, handle_ncclCommDestroy, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommAbort, handle_ncclCommAbort, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommGetAsyncError, handle_ncclCommGetAsyncError, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommCount, handle_ncclCommCount, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommCuDevice, handle_ncclCommCuDevice, rpc_backend::nccl) \
  HANDLER(RPC_ncclCommUserRank, handle_ncclCommUserRank, rpc_backend::nccl) \
  HANDLER(RPC_ncclRedOpCreatePreMulSum, handle_ncclRedOpCreatePreMulSum, rpc_backend::nccl) \
  HANDLER(RPC_ncclRedOpDestroy, handle_ncclRedOpDestroy, rpc_backend::nccl) \
  HANDLER(RPC_ncclReduce, handle_ncclReduce, rpc_backend::nccl) \
  HANDLER(RPC_ncclBcast, handle_ncclBcast, rpc_backend::nccl) \
  HANDLER(RPC_ncclBroadcast, handle_ncclBroadcast, rpc_backend::nccl) \
  HANDLER(RPC_ncclAllReduce, handle_ncclAllReduce, rpc_backend::nccl) \
  HANDLER(RPC_ncclReduceScatter, handle_ncclReduceScatter, rpc_backend::nccl) \
  HANDLER(RPC_ncclAllGather, handle_ncclAllGather, rpc_backend::nccl) \
  HANDLER(RPC_ncclSend, handle_ncclSend, rpc_backend::nccl) \
  HANDLER(RPC_ncclRecv, handle_ncclRecv, rpc_backend::nccl) \
  HANDLER(RPC_ncclGroupStart, handle_ncclGroupStart, rpc_backend::nccl) \
  HANDLER(RPC_ncclGroupEnd, handle_ncclGroupEnd, rpc_backend::nccl)
#define LUPINE_NVJITLINK_RPC_HANDLERS(HANDLER) \
  HANDLER(RPC_nvJitLinkCreate, handle_nvJitLinkCreate, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkAddData, handle_nvJitLinkAddData, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkAddFile, handle_nvJitLinkAddFile, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetLinkedCubin, handle_nvJitLinkGetLinkedCubin, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetLinkedPtx, handle_nvJitLinkGetLinkedPtx, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetErrorLog, handle_nvJitLinkGetErrorLog, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetInfoLog, handle_nvJitLinkGetInfoLog, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkDestroy, handle_nvJitLinkDestroy, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkComplete, handle_nvJitLinkComplete, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetLinkedCubinSize, handle_nvJitLinkGetLinkedCubinSize, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetLinkedPtxSize, handle_nvJitLinkGetLinkedPtxSize, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetErrorLogSize, handle_nvJitLinkGetErrorLogSize, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkGetInfoLogSize, handle_nvJitLinkGetInfoLogSize, rpc_backend::nvjitlink) \
  HANDLER(RPC_nvJitLinkVersion, handle_nvJitLinkVersion, rpc_backend::nvjitlink)
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
  HANDLER(RPC_hipSetDevice, handle_hipSetDevice, rpc_backend::hip) \
  HANDLER(RPC_hipGetDevice, handle_hipGetDevice, rpc_backend::hip) \
  HANDLER(RPC_hipGetLastError, handle_hipGetLastError, rpc_backend::hip) \
  HANDLER(RPC_hipExtGetLastError, handle_hipExtGetLastError, rpc_backend::hip) \
  HANDLER(RPC_hipPeekAtLastError, handle_hipPeekAtLastError, rpc_backend::hip) \
  HANDLER(RPC_hipGetDevicePropertiesR0600, handle_hipGetDevicePropertiesR0600, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetName, handle_hipDeviceGetName, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceTotalMem, handle_hipDeviceTotalMem, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetAttribute, handle_hipDeviceGetAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipDriverGetVersion, handle_hipDriverGetVersion, rpc_backend::hip) \
  HANDLER(RPC_hipRuntimeGetVersion, handle_hipRuntimeGetVersion, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceComputeCapability, handle_hipDeviceComputeCapability, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetUuid, handle_hipDeviceGetUuid, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetP2PAttribute, handle_hipDeviceGetP2PAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetPCIBusId, handle_hipDeviceGetPCIBusId, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetByPCIBusId, handle_hipDeviceGetByPCIBusId, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceSynchronize, handle_hipDeviceSynchronize, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceReset, handle_hipDeviceReset, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetDefaultMemPool, handle_hipDeviceGetDefaultMemPool, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceSetMemPool, handle_hipDeviceSetMemPool, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetMemPool, handle_hipDeviceGetMemPool, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetTexture1DLinearMaxWidth, handle_hipDeviceGetTexture1DLinearMaxWidth, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceSetCacheConfig, handle_hipDeviceSetCacheConfig, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetCacheConfig, handle_hipDeviceGetCacheConfig, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetLimit, handle_hipDeviceGetLimit, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceSetLimit, handle_hipDeviceSetLimit, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetSharedMemConfig, handle_hipDeviceGetSharedMemConfig, rpc_backend::hip) \
  HANDLER(RPC_hipGetDeviceFlags, handle_hipGetDeviceFlags, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceSetSharedMemConfig, handle_hipDeviceSetSharedMemConfig, rpc_backend::hip) \
  HANDLER(RPC_hipSetDeviceFlags, handle_hipSetDeviceFlags, rpc_backend::hip) \
  HANDLER(RPC_hipChooseDeviceR0600, handle_hipChooseDeviceR0600, rpc_backend::hip) \
  HANDLER(RPC_hipExtGetLinkTypeAndHopCount, handle_hipExtGetLinkTypeAndHopCount, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGetStreamPriorityRange, handle_hipDeviceGetStreamPriorityRange, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceGraphMemTrim, handle_hipDeviceGraphMemTrim, rpc_backend::hip) \
  HANDLER(RPC_hipProfilerStart, handle_hipProfilerStart, rpc_backend::hip) \
  HANDLER(RPC_hipProfilerStop, handle_hipProfilerStop, rpc_backend::hip) \
  HANDLER(RPC_hipIpcGetMemHandle, handle_hipIpcGetMemHandle, rpc_backend::hip) \
  HANDLER(RPC_hipIpcOpenMemHandle, handle_hipIpcOpenMemHandle, rpc_backend::hip) \
  HANDLER(RPC_hipIpcCloseMemHandle, handle_hipIpcCloseMemHandle, rpc_backend::hip) \
  HANDLER(RPC_hipIpcGetEventHandle, handle_hipIpcGetEventHandle, rpc_backend::hip) \
  HANDLER(RPC_hipIpcOpenEventHandle, handle_hipIpcOpenEventHandle, rpc_backend::hip) \
  HANDLER(RPC_hipStreamCreate, handle_hipStreamCreate, rpc_backend::hip) \
  HANDLER(RPC_hipStreamCreateWithFlags, handle_hipStreamCreateWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipStreamCreateWithPriority, handle_hipStreamCreateWithPriority, rpc_backend::hip) \
  HANDLER(RPC_hipStreamDestroy, handle_hipStreamDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipStreamQuery, handle_hipStreamQuery, rpc_backend::hip) \
  HANDLER(RPC_hipStreamSynchronize, handle_hipStreamSynchronize, rpc_backend::hip) \
  HANDLER(RPC_hipStreamWaitEvent, handle_hipStreamWaitEvent, rpc_backend::hip) \
  HANDLER(RPC_hipStreamGetFlags, handle_hipStreamGetFlags, rpc_backend::hip) \
  HANDLER(RPC_hipStreamGetId, handle_hipStreamGetId, rpc_backend::hip) \
  HANDLER(RPC_hipStreamGetPriority, handle_hipStreamGetPriority, rpc_backend::hip) \
  HANDLER(RPC_hipStreamGetDevice, handle_hipStreamGetDevice, rpc_backend::hip) \
  HANDLER(RPC_hipExtStreamCreateWithCUMask, handle_hipExtStreamCreateWithCUMask, rpc_backend::hip) \
  HANDLER(RPC_hipExtStreamGetCUMask, handle_hipExtStreamGetCUMask, rpc_backend::hip) \
  HANDLER(RPC_hipStreamSetAttribute, handle_hipStreamSetAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipStreamGetAttribute, handle_hipStreamGetAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipStreamCopyAttributes, handle_hipStreamCopyAttributes, rpc_backend::hip) \
  HANDLER(RPC_hipStreamWaitValue32, handle_hipStreamWaitValue32, rpc_backend::hip) \
  HANDLER(RPC_hipStreamWaitValue64, handle_hipStreamWaitValue64, rpc_backend::hip) \
  HANDLER(RPC_hipStreamWriteValue32, handle_hipStreamWriteValue32, rpc_backend::hip) \
  HANDLER(RPC_hipStreamWriteValue64, handle_hipStreamWriteValue64, rpc_backend::hip) \
  HANDLER(RPC_hipStreamBatchMemOp, handle_hipStreamBatchMemOp, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddBatchMemOpNode, handle_hipGraphAddBatchMemOpNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphBatchMemOpNodeSetParams, handle_hipGraphBatchMemOpNodeSetParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecBatchMemOpNodeSetParams, handle_hipGraphExecBatchMemOpNodeSetParams, rpc_backend::hip) \
  HANDLER(RPC_hipEventCreateWithFlags, handle_hipEventCreateWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipEventCreate, handle_hipEventCreate, rpc_backend::hip) \
  HANDLER(RPC_hipEventRecordWithFlags, handle_hipEventRecordWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipEventRecord, handle_hipEventRecord, rpc_backend::hip) \
  HANDLER(RPC_hipEventDestroy, handle_hipEventDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipEventSynchronize, handle_hipEventSynchronize, rpc_backend::hip) \
  HANDLER(RPC_hipEventElapsedTime, handle_hipEventElapsedTime, rpc_backend::hip) \
  HANDLER(RPC_hipEventQuery, handle_hipEventQuery, rpc_backend::hip) \
  HANDLER(RPC_hipPointerGetAttributes, handle_hipPointerGetAttributes, rpc_backend::hip) \
  HANDLER(RPC_hipMalloc, handle_hipMalloc, rpc_backend::hip) \
  HANDLER(RPC_hipExtMallocWithFlags, handle_hipExtMallocWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipMallocManaged, handle_hipMallocManaged, rpc_backend::hip) \
  HANDLER(RPC_hipMemPrefetchAsync, handle_hipMemPrefetchAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemPrefetchAsync_v2, handle_hipMemPrefetchAsync_v2, rpc_backend::hip) \
  HANDLER(RPC_hipMemAdvise, handle_hipMemAdvise, rpc_backend::hip) \
  HANDLER(RPC_hipMemAdvise_v2, handle_hipMemAdvise_v2, rpc_backend::hip) \
  HANDLER(RPC_hipMemRangeGetAttribute, handle_hipMemRangeGetAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipStreamAttachMemAsync, handle_hipStreamAttachMemAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMallocAsync, handle_hipMallocAsync, rpc_backend::hip) \
  HANDLER(RPC_hipFreeAsync, handle_hipFreeAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolTrimTo, handle_hipMemPoolTrimTo, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolSetAccess, handle_hipMemPoolSetAccess, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolGetAccess, handle_hipMemPoolGetAccess, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolCreate, handle_hipMemPoolCreate, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolDestroy, handle_hipMemPoolDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipMallocFromPoolAsync, handle_hipMallocFromPoolAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolExportPointer, handle_hipMemPoolExportPointer, rpc_backend::hip) \
  HANDLER(RPC_hipMemPoolImportPointer, handle_hipMemPoolImportPointer, rpc_backend::hip) \
  HANDLER(RPC_hipMallocPitch, handle_hipMallocPitch, rpc_backend::hip) \
  HANDLER(RPC_hipMemAllocPitch, handle_hipMemAllocPitch, rpc_backend::hip) \
  HANDLER(RPC_hipFree, handle_hipFree, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyHtoD, handle_hipMemcpyHtoD, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyDtoH, handle_hipMemcpyDtoH, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyDtoD, handle_hipMemcpyDtoD, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyAtoD, handle_hipMemcpyAtoD, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyDtoA, handle_hipMemcpyDtoA, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyAtoA, handle_hipMemcpyAtoA, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyDtoDAsync, handle_hipMemcpyDtoDAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyAtoH, handle_hipMemcpyAtoH, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyHtoA, handle_hipMemcpyHtoA, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpy2DArrayToArray, handle_hipMemcpy2DArrayToArray, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyPeer, handle_hipMemcpyPeer, rpc_backend::hip) \
  HANDLER(RPC_hipMemcpyPeerAsync, handle_hipMemcpyPeerAsync, rpc_backend::hip) \
  HANDLER(RPC_hipModuleGetGlobal, handle_hipModuleGetGlobal, rpc_backend::hip) \
  HANDLER(RPC_hipMemset, handle_hipMemset, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD8, handle_hipMemsetD8, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD8Async, handle_hipMemsetD8Async, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD16, handle_hipMemsetD16, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD16Async, handle_hipMemsetD16Async, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD32, handle_hipMemsetD32, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetAsync, handle_hipMemsetAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD32Async, handle_hipMemsetD32Async, rpc_backend::hip) \
  HANDLER(RPC_hipMemset2D, handle_hipMemset2D, rpc_backend::hip) \
  HANDLER(RPC_hipMemset2DAsync, handle_hipMemset2DAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemset3D, handle_hipMemset3D, rpc_backend::hip) \
  HANDLER(RPC_hipMemset3DAsync, handle_hipMemset3DAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD2D8, handle_hipMemsetD2D8, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD2D8Async, handle_hipMemsetD2D8Async, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD2D16, handle_hipMemsetD2D16, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD2D16Async, handle_hipMemsetD2D16Async, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD2D32, handle_hipMemsetD2D32, rpc_backend::hip) \
  HANDLER(RPC_hipMemsetD2D32Async, handle_hipMemsetD2D32Async, rpc_backend::hip) \
  HANDLER(RPC_hipMemGetInfo, handle_hipMemGetInfo, rpc_backend::hip) \
  HANDLER(RPC_hipMemPtrGetInfo, handle_hipMemPtrGetInfo, rpc_backend::hip) \
  HANDLER(RPC_hipMemGetAddressRange, handle_hipMemGetAddressRange, rpc_backend::hip) \
  HANDLER(RPC_hipMallocArray, handle_hipMallocArray, rpc_backend::hip) \
  HANDLER(RPC_hipArrayCreate, handle_hipArrayCreate, rpc_backend::hip) \
  HANDLER(RPC_hipArrayDestroy, handle_hipArrayDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipArray3DCreate, handle_hipArray3DCreate, rpc_backend::hip) \
  HANDLER(RPC_hipMalloc3D, handle_hipMalloc3D, rpc_backend::hip) \
  HANDLER(RPC_hipFreeArray, handle_hipFreeArray, rpc_backend::hip) \
  HANDLER(RPC_hipMalloc3DArray, handle_hipMalloc3DArray, rpc_backend::hip) \
  HANDLER(RPC_hipArrayGetInfo, handle_hipArrayGetInfo, rpc_backend::hip) \
  HANDLER(RPC_hipArrayGetDescriptor, handle_hipArrayGetDescriptor, rpc_backend::hip) \
  HANDLER(RPC_hipArray3DGetDescriptor, handle_hipArray3DGetDescriptor, rpc_backend::hip) \
  HANDLER(RPC_hipMallocMipmappedArray, handle_hipMallocMipmappedArray, rpc_backend::hip) \
  HANDLER(RPC_hipFreeMipmappedArray, handle_hipFreeMipmappedArray, rpc_backend::hip) \
  HANDLER(RPC_hipGetMipmappedArrayLevel, handle_hipGetMipmappedArrayLevel, rpc_backend::hip) \
  HANDLER(RPC_hipMipmappedArrayCreate, handle_hipMipmappedArrayCreate, rpc_backend::hip) \
  HANDLER(RPC_hipMipmappedArrayDestroy, handle_hipMipmappedArrayDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipMipmappedArrayGetLevel, handle_hipMipmappedArrayGetLevel, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceCanAccessPeer, handle_hipDeviceCanAccessPeer, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceEnablePeerAccess, handle_hipDeviceEnablePeerAccess, rpc_backend::hip) \
  HANDLER(RPC_hipDeviceDisablePeerAccess, handle_hipDeviceDisablePeerAccess, rpc_backend::hip) \
  HANDLER(RPC_hipCtxCreate, handle_hipCtxCreate, rpc_backend::hip) \
  HANDLER(RPC_hipCtxDestroy, handle_hipCtxDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipCtxPopCurrent, handle_hipCtxPopCurrent, rpc_backend::hip) \
  HANDLER(RPC_hipCtxPushCurrent, handle_hipCtxPushCurrent, rpc_backend::hip) \
  HANDLER(RPC_hipCtxSetCurrent, handle_hipCtxSetCurrent, rpc_backend::hip) \
  HANDLER(RPC_hipCtxGetCurrent, handle_hipCtxGetCurrent, rpc_backend::hip) \
  HANDLER(RPC_hipCtxGetDevice, handle_hipCtxGetDevice, rpc_backend::hip) \
  HANDLER(RPC_hipCtxGetApiVersion, handle_hipCtxGetApiVersion, rpc_backend::hip) \
  HANDLER(RPC_hipCtxGetCacheConfig, handle_hipCtxGetCacheConfig, rpc_backend::hip) \
  HANDLER(RPC_hipCtxSetCacheConfig, handle_hipCtxSetCacheConfig, rpc_backend::hip) \
  HANDLER(RPC_hipCtxSetSharedMemConfig, handle_hipCtxSetSharedMemConfig, rpc_backend::hip) \
  HANDLER(RPC_hipCtxGetSharedMemConfig, handle_hipCtxGetSharedMemConfig, rpc_backend::hip) \
  HANDLER(RPC_hipCtxSynchronize, handle_hipCtxSynchronize, rpc_backend::hip) \
  HANDLER(RPC_hipCtxGetFlags, handle_hipCtxGetFlags, rpc_backend::hip) \
  HANDLER(RPC_hipCtxEnablePeerAccess, handle_hipCtxEnablePeerAccess, rpc_backend::hip) \
  HANDLER(RPC_hipCtxDisablePeerAccess, handle_hipCtxDisablePeerAccess, rpc_backend::hip) \
  HANDLER(RPC_hipDevicePrimaryCtxGetState, handle_hipDevicePrimaryCtxGetState, rpc_backend::hip) \
  HANDLER(RPC_hipDevicePrimaryCtxRelease, handle_hipDevicePrimaryCtxRelease, rpc_backend::hip) \
  HANDLER(RPC_hipDevicePrimaryCtxRetain, handle_hipDevicePrimaryCtxRetain, rpc_backend::hip) \
  HANDLER(RPC_hipDevicePrimaryCtxReset, handle_hipDevicePrimaryCtxReset, rpc_backend::hip) \
  HANDLER(RPC_hipDevicePrimaryCtxSetFlags, handle_hipDevicePrimaryCtxSetFlags, rpc_backend::hip) \
  HANDLER(RPC_hipModuleUnload, handle_hipModuleUnload, rpc_backend::hip) \
  HANDLER(RPC_hipModuleGetFunction, handle_hipModuleGetFunction, rpc_backend::hip) \
  HANDLER(RPC_hipModuleGetFunctionCount, handle_hipModuleGetFunctionCount, rpc_backend::hip) \
  HANDLER(RPC_hipLibraryUnload, handle_hipLibraryUnload, rpc_backend::hip) \
  HANDLER(RPC_hipLibraryGetKernel, handle_hipLibraryGetKernel, rpc_backend::hip) \
  HANDLER(RPC_hipLibraryGetKernelCount, handle_hipLibraryGetKernelCount, rpc_backend::hip) \
  HANDLER(RPC_hipLibraryEnumerateKernels, handle_hipLibraryEnumerateKernels, rpc_backend::hip) \
  HANDLER(RPC_hipKernelGetLibrary, handle_hipKernelGetLibrary, rpc_backend::hip) \
  HANDLER(RPC_hipFuncGetAttribute, handle_hipFuncGetAttribute, rpc_backend::hip) \
  HANDLER(RPC_hipModuleOccupancyMaxPotentialBlockSize, handle_hipModuleOccupancyMaxPotentialBlockSize, rpc_backend::hip) \
  HANDLER(RPC_hipModuleOccupancyMaxPotentialBlockSizeWithFlags, handle_hipModuleOccupancyMaxPotentialBlockSizeWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, handle_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, rpc_backend::hip) \
  HANDLER(RPC_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, handle_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipCreateTextureObject, handle_hipCreateTextureObject, rpc_backend::hip) \
  HANDLER(RPC_hipDestroyTextureObject, handle_hipDestroyTextureObject, rpc_backend::hip) \
  HANDLER(RPC_hipGetChannelDesc, handle_hipGetChannelDesc, rpc_backend::hip) \
  HANDLER(RPC_hipGetTextureObjectResourceDesc, handle_hipGetTextureObjectResourceDesc, rpc_backend::hip) \
  HANDLER(RPC_hipGetTextureObjectResourceViewDesc, handle_hipGetTextureObjectResourceViewDesc, rpc_backend::hip) \
  HANDLER(RPC_hipGetTextureObjectTextureDesc, handle_hipGetTextureObjectTextureDesc, rpc_backend::hip) \
  HANDLER(RPC_hipTexObjectCreate, handle_hipTexObjectCreate, rpc_backend::hip) \
  HANDLER(RPC_hipTexObjectDestroy, handle_hipTexObjectDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipTexObjectGetResourceDesc, handle_hipTexObjectGetResourceDesc, rpc_backend::hip) \
  HANDLER(RPC_hipTexObjectGetResourceViewDesc, handle_hipTexObjectGetResourceViewDesc, rpc_backend::hip) \
  HANDLER(RPC_hipTexObjectGetTextureDesc, handle_hipTexObjectGetTextureDesc, rpc_backend::hip) \
  HANDLER(RPC_hipCreateSurfaceObject, handle_hipCreateSurfaceObject, rpc_backend::hip) \
  HANDLER(RPC_hipDestroySurfaceObject, handle_hipDestroySurfaceObject, rpc_backend::hip) \
  HANDLER(RPC_hipStreamBeginCapture, handle_hipStreamBeginCapture, rpc_backend::hip) \
  HANDLER(RPC_hipStreamBeginCaptureToGraph, handle_hipStreamBeginCaptureToGraph, rpc_backend::hip) \
  HANDLER(RPC_hipStreamEndCapture, handle_hipStreamEndCapture, rpc_backend::hip) \
  HANDLER(RPC_hipStreamGetCaptureInfo, handle_hipStreamGetCaptureInfo, rpc_backend::hip) \
  HANDLER(RPC_hipStreamIsCapturing, handle_hipStreamIsCapturing, rpc_backend::hip) \
  HANDLER(RPC_hipStreamUpdateCaptureDependencies, handle_hipStreamUpdateCaptureDependencies, rpc_backend::hip) \
  HANDLER(RPC_hipThreadExchangeStreamCaptureMode, handle_hipThreadExchangeStreamCaptureMode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphCreate, handle_hipGraphCreate, rpc_backend::hip) \
  HANDLER(RPC_hipGraphDestroy, handle_hipGraphDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddDependencies, handle_hipGraphAddDependencies, rpc_backend::hip) \
  HANDLER(RPC_hipGraphRemoveDependencies, handle_hipGraphRemoveDependencies, rpc_backend::hip) \
  HANDLER(RPC_hipGraphGetEdges, handle_hipGraphGetEdges, rpc_backend::hip) \
  HANDLER(RPC_hipGraphGetNodes, handle_hipGraphGetNodes, rpc_backend::hip) \
  HANDLER(RPC_hipGraphGetRootNodes, handle_hipGraphGetRootNodes, rpc_backend::hip) \
  HANDLER(RPC_hipGraphNodeGetDependencies, handle_hipGraphNodeGetDependencies, rpc_backend::hip) \
  HANDLER(RPC_hipGraphNodeGetDependentNodes, handle_hipGraphNodeGetDependentNodes, rpc_backend::hip) \
  HANDLER(RPC_hipGraphNodeGetType, handle_hipGraphNodeGetType, rpc_backend::hip) \
  HANDLER(RPC_hipGraphDestroyNode, handle_hipGraphDestroyNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphClone, handle_hipGraphClone, rpc_backend::hip) \
  HANDLER(RPC_hipGraphNodeFindInClone, handle_hipGraphNodeFindInClone, rpc_backend::hip) \
  HANDLER(RPC_hipGraphInstantiate, handle_hipGraphInstantiate, rpc_backend::hip) \
  HANDLER(RPC_hipGraphInstantiateWithFlags, handle_hipGraphInstantiateWithFlags, rpc_backend::hip) \
  HANDLER(RPC_hipGraphInstantiateWithParams, handle_hipGraphInstantiateWithParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphLaunch, handle_hipGraphLaunch, rpc_backend::hip) \
  HANDLER(RPC_hipGraphUpload, handle_hipGraphUpload, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecGetFlags, handle_hipGraphExecGetFlags, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecDestroy, handle_hipGraphExecDestroy, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecUpdate, handle_hipGraphExecUpdate, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddMemsetNode, handle_hipGraphAddMemsetNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphMemsetNodeGetParams, handle_hipGraphMemsetNodeGetParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphMemsetNodeSetParams, handle_hipGraphMemsetNodeSetParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecMemsetNodeSetParams, handle_hipGraphExecMemsetNodeSetParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddChildGraphNode, handle_hipGraphAddChildGraphNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphChildGraphNodeGetGraph, handle_hipGraphChildGraphNodeGetGraph, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecChildGraphNodeSetParams, handle_hipGraphExecChildGraphNodeSetParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddEmptyNode, handle_hipGraphAddEmptyNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddEventRecordNode, handle_hipGraphAddEventRecordNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphEventRecordNodeGetEvent, handle_hipGraphEventRecordNodeGetEvent, rpc_backend::hip) \
  HANDLER(RPC_hipGraphEventRecordNodeSetEvent, handle_hipGraphEventRecordNodeSetEvent, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecEventRecordNodeSetEvent, handle_hipGraphExecEventRecordNodeSetEvent, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddEventWaitNode, handle_hipGraphAddEventWaitNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphEventWaitNodeGetEvent, handle_hipGraphEventWaitNodeGetEvent, rpc_backend::hip) \
  HANDLER(RPC_hipGraphEventWaitNodeSetEvent, handle_hipGraphEventWaitNodeSetEvent, rpc_backend::hip) \
  HANDLER(RPC_hipGraphExecEventWaitNodeSetEvent, handle_hipGraphExecEventWaitNodeSetEvent, rpc_backend::hip) \
  HANDLER(RPC_hipGraphAddMemFreeNode, handle_hipGraphAddMemFreeNode, rpc_backend::hip) \
  HANDLER(RPC_hipGraphMemFreeNodeGetParams, handle_hipGraphMemFreeNodeGetParams, rpc_backend::hip) \
  HANDLER(RPC_hipGraphKernelNodeCopyAttributes, handle_hipGraphKernelNodeCopyAttributes, rpc_backend::hip) \
  HANDLER(RPC_hipGraphNodeSetEnabled, handle_hipGraphNodeSetEnabled, rpc_backend::hip) \
  HANDLER(RPC_hipGraphNodeGetEnabled, handle_hipGraphNodeGetEnabled, rpc_backend::hip) \
  HANDLER(RPC_hipDrvGraphAddMemsetNode, handle_hipDrvGraphAddMemsetNode, rpc_backend::hip) \
  HANDLER(RPC_hipDrvGraphAddMemFreeNode, handle_hipDrvGraphAddMemFreeNode, rpc_backend::hip) \
  HANDLER(RPC_hipDrvGraphExecMemsetNodeSetParams, handle_hipDrvGraphExecMemsetNodeSetParams, rpc_backend::hip) \
  HANDLER(RPC_hipMemAddressFree, handle_hipMemAddressFree, rpc_backend::hip) \
  HANDLER(RPC_hipMemAddressReserve, handle_hipMemAddressReserve, rpc_backend::hip) \
  HANDLER(RPC_hipMemCreate, handle_hipMemCreate, rpc_backend::hip) \
  HANDLER(RPC_hipMemGetAccess, handle_hipMemGetAccess, rpc_backend::hip) \
  HANDLER(RPC_hipMemGetAllocationGranularity, handle_hipMemGetAllocationGranularity, rpc_backend::hip) \
  HANDLER(RPC_hipMemGetAllocationPropertiesFromHandle, handle_hipMemGetAllocationPropertiesFromHandle, rpc_backend::hip) \
  HANDLER(RPC_hipMemMap, handle_hipMemMap, rpc_backend::hip) \
  HANDLER(RPC_hipMemMapArrayAsync, handle_hipMemMapArrayAsync, rpc_backend::hip) \
  HANDLER(RPC_hipMemRelease, handle_hipMemRelease, rpc_backend::hip) \
  HANDLER(RPC_hipMemRetainAllocationHandle, handle_hipMemRetainAllocationHandle, rpc_backend::hip) \
  HANDLER(RPC_hipMemSetAccess, handle_hipMemSetAccess, rpc_backend::hip) \
  HANDLER(RPC_hipMemUnmap, handle_hipMemUnmap, rpc_backend::hip)
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
#if CUDA_VERSION >= 12090
LUPINE_DECLARE_HANDLER(RPC_cuLogsRegisterCallback,
                       handle_cuLogsRegisterCallback, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12090
LUPINE_DECLARE_HANDLER(RPC_cuLogsUnregisterCallback,
                       handle_cuLogsUnregisterCallback, rpc_backend::cuda)
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
#if CUDA_VERSION >= 12090
LUPINE_DECLARE_HANDLER(RPC_cuLogsCurrent, handle_cuLogsCurrent,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12090
LUPINE_DECLARE_HANDLER(RPC_cuLogsDumpToFile, handle_cuLogsDumpToFile,
                       rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12090
LUPINE_DECLARE_HANDLER(RPC_cuLogsDumpToMemory, handle_cuLogsDumpToMemory,
                       rpc_backend::cuda)
#endif
#endif
#ifdef LUPINE_BUILD_CUDART_BACKEND
LUPINE_CUDART_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaFuncGetName, handle_cudaFuncGetName,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLaunchHostFunc_v2, handle_cudaLaunchHostFunc_v2,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamGetCaptureInfo_v2,
                       handle_cudaStreamGetCaptureInfo_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamGetCaptureInfo_v3,
                       handle_cudaStreamGetCaptureInfo_v3, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphAddNode, handle_cudaGraphAddNode,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphAddNode_v2, handle_cudaGraphAddNode_v2,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphInstantiateWithParams,
                       handle_cudaGraphInstantiateWithParams,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC___cudaLaunchKernel, handle___cudaLaunchKernel,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDevResourceGenerateDesc,
                       handle_cudaDevResourceGenerateDesc, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDevSmResourceSplit,
                       handle_cudaDevSmResourceSplit, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDevSmResourceSplitByCount,
                       handle_cudaDevSmResourceSplitByCount,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDeviceGetDevResource,
                       handle_cudaDeviceGetDevResource, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDeviceGetExecutionCtx,
                       handle_cudaDeviceGetExecutionCtx, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDeviceGetHostAtomicCapabilities,
                       handle_cudaDeviceGetHostAtomicCapabilities,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaDeviceGetP2PAtomicCapabilities,
                       handle_cudaDeviceGetP2PAtomicCapabilities,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaDeviceRegisterAsyncNotification,
                       handle_cudaDeviceRegisterAsyncNotification,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaDeviceUnregisterAsyncNotification,
                       handle_cudaDeviceUnregisterAsyncNotification,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxDestroy,
                       handle_cudaExecutionCtxDestroy, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxGetDevResource,
                       handle_cudaExecutionCtxGetDevResource,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxGetDevice,
                       handle_cudaExecutionCtxGetDevice, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxGetId, handle_cudaExecutionCtxGetId,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxRecordEvent,
                       handle_cudaExecutionCtxRecordEvent, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxStreamCreate,
                       handle_cudaExecutionCtxStreamCreate, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxSynchronize,
                       handle_cudaExecutionCtxSynchronize, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaExecutionCtxWaitEvent,
                       handle_cudaExecutionCtxWaitEvent, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaFuncGetParamCount, handle_cudaFuncGetParamCount,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaFuncGetParamInfo, handle_cudaFuncGetParamInfo,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGetDriverEntryPointByVersion,
                       handle_cudaGetDriverEntryPointByVersion,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGetKernel, handle_cudaGetKernel,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGreenCtxCreate, handle_cudaGreenCtxCreate,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaInitDevice, handle_cudaInitDevice,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaKernelSetAttributeForDevice,
                       handle_cudaKernelSetAttributeForDevice,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryEnumerateKernels,
                       handle_cudaLibraryEnumerateKernels, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryGetGlobal, handle_cudaLibraryGetGlobal,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryGetKernel, handle_cudaLibraryGetKernel,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryGetKernelCount,
                       handle_cudaLibraryGetKernelCount, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryGetManaged, handle_cudaLibraryGetManaged,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryGetUnifiedFunction,
                       handle_cudaLibraryGetUnifiedFunction,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryLoadData, handle_cudaLibraryLoadData,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryLoadFromFile,
                       handle_cudaLibraryLoadFromFile, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLibraryUnload, handle_cudaLibraryUnload,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLogsCurrent, handle_cudaLogsCurrent,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLogsDumpToFile, handle_cudaLogsDumpToFile,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLogsDumpToMemory, handle_cudaLogsDumpToMemory,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLogsRegisterCallback,
                       handle_cudaLogsRegisterCallback, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaLogsUnregisterCallback,
                       handle_cudaLogsUnregisterCallback, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12020 && CUDART_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemAdvise_v2, handle_cudaMemAdvise_v2,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemDiscardAndPrefetchBatchAsync,
                       handle_cudaMemDiscardAndPrefetchBatchAsync,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemDiscardBatchAsync,
                       handle_cudaMemDiscardBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemGetDefaultMemPool,
                       handle_cudaMemGetDefaultMemPool, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemGetMemPool, handle_cudaMemGetMemPool,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12020 && CUDART_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemPrefetchAsync_v2,
                       handle_cudaMemPrefetchAsync_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemPrefetchBatchAsync,
                       handle_cudaMemPrefetchBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemSetMemPool, handle_cudaMemSetMemPool,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemcpy3DBatchAsync,
                       handle_cudaMemcpy3DBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemcpy3DWithAttributesAsync,
                       handle_cudaMemcpy3DWithAttributesAsync,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemcpyBatchAsync, handle_cudaMemcpyBatchAsync,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaMemcpyWithAttributesAsync,
                       handle_cudaMemcpyWithAttributesAsync,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamBeginCaptureToGraph,
                       handle_cudaStreamBeginCaptureToGraph,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamBeginRecaptureToGraph,
                       handle_cudaStreamBeginRecaptureToGraph,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamGetDevResource,
                       handle_cudaStreamGetDevResource, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamGetDevice, handle_cudaStreamGetDevice,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamGetId, handle_cudaStreamGetId,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_cudaStreamUpdateCaptureDependencies_v2,
                       handle_cudaStreamUpdateCaptureDependencies_v2,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphAddDependencies,
                       handle_cudaGraphAddDependencies, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphConditionalHandleCreate,
                       handle_cudaGraphConditionalHandleCreate,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphConditionalHandleCreate_v2,
                       handle_cudaGraphConditionalHandleCreate_v2,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphExecGetFlags, handle_cudaGraphExecGetFlags,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphExecGetId, handle_cudaGraphExecGetId,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphExecNodeSetParams,
                       handle_cudaGraphExecNodeSetParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphGetEdges, handle_cudaGraphGetEdges,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphGetId, handle_cudaGraphGetId,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphKernelNodeCopyAttributes,
                       handle_cudaGraphKernelNodeCopyAttributes,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeGetContainingGraph,
                       handle_cudaGraphNodeGetContainingGraph,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeGetDependencies,
                       handle_cudaGraphNodeGetDependencies, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeGetDependentNodes,
                       handle_cudaGraphNodeGetDependentNodes,
                       rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeGetLocalId,
                       handle_cudaGraphNodeGetLocalId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeGetParams,
                       handle_cudaGraphNodeGetParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeGetToolsId,
                       handle_cudaGraphNodeGetToolsId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphNodeSetParams,
                       handle_cudaGraphNodeSetParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC_cudaGraphRemoveDependencies,
                       handle_cudaGraphRemoveDependencies, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
LUPINE_DECLARE_HANDLER(RPC___cudaGetKernel, handle___cudaGetKernel,
                       rpc_backend::cudart)
#endif
#endif
#ifdef LUPINE_BUILD_CUBLAS_BACKEND
LUPINE_CUBLAS_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGetMatrixAsync_64,
                       handle_cublasGetMatrixAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGetMatrix_64, handle_cublasGetMatrix_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGetVectorAsync_64,
                       handle_cublasGetVectorAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGetVector_64, handle_cublasGetVector_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSetMatrixAsync_64,
                       handle_cublasSetMatrixAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSetMatrix_64, handle_cublasSetMatrix_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSetVectorAsync_64,
                       handle_cublasSetVectorAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSetVector_64, handle_cublasSetVector_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasAsumEx_64, handle_cublasAsumEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasAxpyEx_64, handle_cublasAxpyEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCaxpy_v2_64, handle_cublasCaxpy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCcopy_v2_64, handle_cublasCcopy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCdgmm_64, handle_cublasCdgmm_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCdotc_v2_64, handle_cublasCdotc_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCdotu_v2_64, handle_cublasCdotu_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgbmv_v2_64, handle_cublasCgbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgeam_64, handle_cublasCgeam_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemm3mBatched_64,
                       handle_cublasCgemm3mBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemm3mEx_64, handle_cublasCgemm3mEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemm3mStridedBatched_64,
                       handle_cublasCgemm3mStridedBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemm3m_64, handle_cublasCgemm3m_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemmBatched_64, handle_cublasCgemmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemmEx_64, handle_cublasCgemmEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemmStridedBatched_64,
                       handle_cublasCgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemm_v2_64, handle_cublasCgemm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemvBatched_64, handle_cublasCgemvBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemvStridedBatched_64,
                       handle_cublasCgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgemv_v2_64, handle_cublasCgemv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgerc_v2_64, handle_cublasCgerc_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCgeru_v2_64, handle_cublasCgeru_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasChbmv_v2_64, handle_cublasChbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasChemm_v2_64, handle_cublasChemm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasChemv_v2_64, handle_cublasChemv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCher2_v2_64, handle_cublasCher2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCher2k_v2_64, handle_cublasCher2k_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCher_v2_64, handle_cublasCher_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCherk3mEx_64, handle_cublasCherk3mEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCherkEx_64, handle_cublasCherkEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCherk_v2_64, handle_cublasCherk_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCherkx_64, handle_cublasCherkx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasChpmv_v2_64, handle_cublasChpmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasChpr2_v2_64, handle_cublasChpr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasChpr_v2_64, handle_cublasChpr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCopyEx_64, handle_cublasCopyEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCrot_v2_64, handle_cublasCrot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCscal_v2_64, handle_cublasCscal_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsrot_v2_64, handle_cublasCsrot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsscal_v2_64, handle_cublasCsscal_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCswap_v2_64, handle_cublasCswap_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsymm_v2_64, handle_cublasCsymm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsymv_v2_64, handle_cublasCsymv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyr2_v2_64, handle_cublasCsyr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyr2k_v2_64, handle_cublasCsyr2k_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyr_v2_64, handle_cublasCsyr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyrk3mEx_64, handle_cublasCsyrk3mEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyrkEx_64, handle_cublasCsyrkEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyrk_v2_64, handle_cublasCsyrk_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCsyrkx_64, handle_cublasCsyrkx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtbmv_v2_64, handle_cublasCtbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtbsv_v2_64, handle_cublasCtbsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtpmv_v2_64, handle_cublasCtpmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtpsv_v2_64, handle_cublasCtpsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtrmm_v2_64, handle_cublasCtrmm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtrmv_v2_64, handle_cublasCtrmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtrsmBatched_64, handle_cublasCtrsmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtrsm_v2_64, handle_cublasCtrsm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasCtrsv_v2_64, handle_cublasCtrsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDasum_v2_64, handle_cublasDasum_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDaxpy_v2_64, handle_cublasDaxpy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDcopy_v2_64, handle_cublasDcopy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDdgmm_64, handle_cublasDdgmm_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDdot_v2_64, handle_cublasDdot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgbmv_v2_64, handle_cublasDgbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgeam_64, handle_cublasDgeam_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgemmBatched_64, handle_cublasDgemmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
LUPINE_DECLARE_HANDLER(RPC_cublasDgemmGroupedBatched,
                       handle_cublasDgemmGroupedBatched, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
LUPINE_DECLARE_HANDLER(RPC_cublasDgemmGroupedBatched_64,
                       handle_cublasDgemmGroupedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgemmStridedBatched_64,
                       handle_cublasDgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgemm_v2_64, handle_cublasDgemm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgemvBatched_64, handle_cublasDgemvBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgemvStridedBatched_64,
                       handle_cublasDgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDgemv_v2_64, handle_cublasDgemv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDger_v2_64, handle_cublasDger_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDnrm2_v2_64, handle_cublasDnrm2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDotEx_64, handle_cublasDotEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDotcEx_64, handle_cublasDotcEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDrot_v2_64, handle_cublasDrot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDrotm_v2_64, handle_cublasDrotm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsbmv_v2_64, handle_cublasDsbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDscal_v2_64, handle_cublasDscal_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDspmv_v2_64, handle_cublasDspmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDspr2_v2_64, handle_cublasDspr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDspr_v2_64, handle_cublasDspr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDswap_v2_64, handle_cublasDswap_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsymm_v2_64, handle_cublasDsymm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsymv_v2_64, handle_cublasDsymv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsyr2_v2_64, handle_cublasDsyr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsyr2k_v2_64, handle_cublasDsyr2k_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsyr_v2_64, handle_cublasDsyr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsyrk_v2_64, handle_cublasDsyrk_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDsyrkx_64, handle_cublasDsyrkx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtbmv_v2_64, handle_cublasDtbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtbsv_v2_64, handle_cublasDtbsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtpmv_v2_64, handle_cublasDtpmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtpsv_v2_64, handle_cublasDtpsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtrmm_v2_64, handle_cublasDtrmm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtrmv_v2_64, handle_cublasDtrmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtrsmBatched_64, handle_cublasDtrsmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtrsm_v2_64, handle_cublasDtrsm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDtrsv_v2_64, handle_cublasDtrsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDzasum_v2_64, handle_cublasDzasum_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasDznrm2_v2_64, handle_cublasDznrm2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGemmBatchedEx_64,
                       handle_cublasGemmBatchedEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGemmEx_64, handle_cublasGemmEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
LUPINE_DECLARE_HANDLER(RPC_cublasGemmGroupedBatchedEx,
                       handle_cublasGemmGroupedBatchedEx, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
LUPINE_DECLARE_HANDLER(RPC_cublasGemmGroupedBatchedEx_64,
                       handle_cublasGemmGroupedBatchedEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasGemmStridedBatchedEx_64,
                       handle_cublasGemmStridedBatchedEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasGetEmulationSpecialValuesSupport,
                       handle_cublasGetEmulationSpecialValuesSupport,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
LUPINE_DECLARE_HANDLER(RPC_cublasGetEmulationStrategy,
                       handle_cublasGetEmulationStrategy, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(
    RPC_cublasGetFixedPointEmulationMantissaBitCountPointer,
    handle_cublasGetFixedPointEmulationMantissaBitCountPointer,
    rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasGetFixedPointEmulationMantissaBitOffset,
                       handle_cublasGetFixedPointEmulationMantissaBitOffset,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasGetFixedPointEmulationMantissaControl,
                       handle_cublasGetFixedPointEmulationMantissaControl,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasGetFixedPointEmulationMaxMantissaBitCount,
                       handle_cublasGetFixedPointEmulationMaxMantissaBitCount,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHSHgemvBatched_64,
                       handle_cublasHSHgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHSHgemvStridedBatched_64,
                       handle_cublasHSHgemvStridedBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHSSgemvBatched_64,
                       handle_cublasHSSgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHSSgemvStridedBatched_64,
                       handle_cublasHSSgemvStridedBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHgemmBatched_64, handle_cublasHgemmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHgemmStridedBatched_64,
                       handle_cublasHgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasHgemm_64, handle_cublasHgemm_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIamaxEx_64, handle_cublasIamaxEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIaminEx_64, handle_cublasIaminEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIcamax_v2_64, handle_cublasIcamax_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIcamin_v2_64, handle_cublasIcamin_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIdamax_v2_64, handle_cublasIdamax_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIdamin_v2_64, handle_cublasIdamin_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIsamax_v2_64, handle_cublasIsamax_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIsamin_v2_64, handle_cublasIsamin_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIzamax_v2_64, handle_cublasIzamax_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasIzamin_v2_64, handle_cublasIzamin_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasNrm2Ex_64, handle_cublasNrm2Ex_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasRotEx_64, handle_cublasRotEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasRotmEx_64, handle_cublasRotmEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSasum_v2_64, handle_cublasSasum_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSaxpy_v2_64, handle_cublasSaxpy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasScalEx_64, handle_cublasScalEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasScasum_v2_64, handle_cublasScasum_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasScnrm2_v2_64, handle_cublasScnrm2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasScopy_v2_64, handle_cublasScopy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSdgmm_64, handle_cublasSdgmm_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSdot_v2_64, handle_cublasSdot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasSetEmulationSpecialValuesSupport,
                       handle_cublasSetEmulationSpecialValuesSupport,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
LUPINE_DECLARE_HANDLER(RPC_cublasSetEmulationStrategy,
                       handle_cublasSetEmulationStrategy, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
LUPINE_DECLARE_HANDLER(RPC_cublasSetEnvironmentMode,
                       handle_cublasSetEnvironmentMode, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(
    RPC_cublasSetFixedPointEmulationMantissaBitCountPointer,
    handle_cublasSetFixedPointEmulationMantissaBitCountPointer,
    rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasSetFixedPointEmulationMantissaBitOffset,
                       handle_cublasSetFixedPointEmulationMantissaBitOffset,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasSetFixedPointEmulationMantissaControl,
                       handle_cublasSetFixedPointEmulationMantissaControl,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasSetFixedPointEmulationMaxMantissaBitCount,
                       handle_cublasSetFixedPointEmulationMaxMantissaBitCount,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgbmv_v2_64, handle_cublasSgbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgeam_64, handle_cublasSgeam_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemmBatched_64, handle_cublasSgemmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemmEx_64, handle_cublasSgemmEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
LUPINE_DECLARE_HANDLER(RPC_cublasSgemmGroupedBatched,
                       handle_cublasSgemmGroupedBatched, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
LUPINE_DECLARE_HANDLER(RPC_cublasSgemmGroupedBatched_64,
                       handle_cublasSgemmGroupedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemmStridedBatched_64,
                       handle_cublasSgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemm_v2_64, handle_cublasSgemm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemvBatched_64, handle_cublasSgemvBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemvStridedBatched_64,
                       handle_cublasSgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSgemv_v2_64, handle_cublasSgemv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSger_v2_64, handle_cublasSger_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSnrm2_v2_64, handle_cublasSnrm2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSrot_v2_64, handle_cublasSrot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSrotm_v2_64, handle_cublasSrotm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsbmv_v2_64, handle_cublasSsbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSscal_v2_64, handle_cublasSscal_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSspmv_v2_64, handle_cublasSspmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSspr2_v2_64, handle_cublasSspr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSspr_v2_64, handle_cublasSspr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSswap_v2_64, handle_cublasSswap_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsymm_v2_64, handle_cublasSsymm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsymv_v2_64, handle_cublasSsymv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsyr2_v2_64, handle_cublasSsyr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsyr2k_v2_64, handle_cublasSsyr2k_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsyr_v2_64, handle_cublasSsyr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsyrk_v2_64, handle_cublasSsyrk_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSsyrkx_64, handle_cublasSsyrkx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStbmv_v2_64, handle_cublasStbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStbsv_v2_64, handle_cublasStbsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStpmv_v2_64, handle_cublasStpmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStpsv_v2_64, handle_cublasStpsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStrmm_v2_64, handle_cublasStrmm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStrmv_v2_64, handle_cublasStrmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStrsmBatched_64, handle_cublasStrsmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStrsm_v2_64, handle_cublasStrsm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasStrsv_v2_64, handle_cublasStrsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasSwapEx_64, handle_cublasSwapEx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasTSSgemvBatched_64,
                       handle_cublasTSSgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasTSSgemvStridedBatched_64,
                       handle_cublasTSSgemvStridedBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasTSTgemvBatched_64,
                       handle_cublasTSTgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasTSTgemvStridedBatched_64,
                       handle_cublasTSTgemvStridedBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZaxpy_v2_64, handle_cublasZaxpy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZcopy_v2_64, handle_cublasZcopy_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZdgmm_64, handle_cublasZdgmm_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZdotc_v2_64, handle_cublasZdotc_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZdotu_v2_64, handle_cublasZdotu_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZdrot_v2_64, handle_cublasZdrot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZdscal_v2_64, handle_cublasZdscal_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgbmv_v2_64, handle_cublasZgbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgeam_64, handle_cublasZgeam_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemm3m_64, handle_cublasZgemm3m_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemmBatched_64, handle_cublasZgemmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemmStridedBatched_64,
                       handle_cublasZgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemm_v2_64, handle_cublasZgemm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemvBatched_64, handle_cublasZgemvBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemvStridedBatched_64,
                       handle_cublasZgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgemv_v2_64, handle_cublasZgemv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgerc_v2_64, handle_cublasZgerc_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZgeru_v2_64, handle_cublasZgeru_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZhbmv_v2_64, handle_cublasZhbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZhemm_v2_64, handle_cublasZhemm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZhemv_v2_64, handle_cublasZhemv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZher2_v2_64, handle_cublasZher2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZher2k_v2_64, handle_cublasZher2k_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZher_v2_64, handle_cublasZher_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZherk_v2_64, handle_cublasZherk_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZherkx_64, handle_cublasZherkx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZhpmv_v2_64, handle_cublasZhpmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZhpr2_v2_64, handle_cublasZhpr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZhpr_v2_64, handle_cublasZhpr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZrot_v2_64, handle_cublasZrot_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZscal_v2_64, handle_cublasZscal_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZswap_v2_64, handle_cublasZswap_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsymm_v2_64, handle_cublasZsymm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsymv_v2_64, handle_cublasZsymv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsyr2_v2_64, handle_cublasZsyr2_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsyr2k_v2_64, handle_cublasZsyr2k_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsyr_v2_64, handle_cublasZsyr_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsyrk_v2_64, handle_cublasZsyrk_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZsyrkx_64, handle_cublasZsyrkx_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtbmv_v2_64, handle_cublasZtbmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtbsv_v2_64, handle_cublasZtbsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtpmv_v2_64, handle_cublasZtpmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtpsv_v2_64, handle_cublasZtpsv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtrmm_v2_64, handle_cublasZtrmm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtrmv_v2_64, handle_cublasZtrmv_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtrsmBatched_64, handle_cublasZtrsmBatched_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtrsm_v2_64, handle_cublasZtrsm_v2_64,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
LUPINE_DECLARE_HANDLER(RPC_cublasZtrsv_v2_64, handle_cublasZtrsv_v2_64,
                       rpc_backend::cublas)
#endif
LUPINE_CUBLASLT_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_lupineCublasLtEmulationDescCopy,
                       handle_lupineCublasLtEmulationDescCopy,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
LUPINE_DECLARE_HANDLER(RPC_cublasLtDisableCpuInstructionsSetMask,
                       handle_cublasLtDisableCpuInstructionsSetMask,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasLtEmulationDescGetAttribute,
                       handle_cublasLtEmulationDescGetAttribute,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasLtEmulationDescInit_internal,
                       handle_cublasLtEmulationDescInit_internal,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
LUPINE_DECLARE_HANDLER(RPC_cublasLtEmulationDescSetAttribute,
                       handle_cublasLtEmulationDescSetAttribute,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130300
LUPINE_DECLARE_HANDLER(RPC_cublasLtMatmulAlgoCheckForStream,
                       handle_cublasLtMatmulAlgoCheckForStream,
                       rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130300
LUPINE_DECLARE_HANDLER(RPC_cublasLtMatmulAlgoGetHeuristicForStream,
                       handle_cublasLtMatmulAlgoGetHeuristicForStream,
                       rpc_backend::cublas)
#endif
#endif
#ifdef LUPINE_BUILD_CUFFT_BACKEND
LUPINE_CUFFT_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUFFT_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cufftXtSetJITCallback, handle_cufftXtSetJITCallback,
                       rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11300 && CUFFT_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC___cufftXtSetJITCallback_12_7,
                       handle___cufftXtSetJITCallback_12_7, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11200
LUPINE_DECLARE_HANDLER(RPC_cufftGetPlanPropertyInt64,
                       handle_cufftGetPlanPropertyInt64, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11200
LUPINE_DECLARE_HANDLER(RPC_cufftResetPlanProperty,
                       handle_cufftResetPlanProperty, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11200
LUPINE_DECLARE_HANDLER(RPC_cufftSetPlanPropertyInt64,
                       handle_cufftSetPlanPropertyInt64, rpc_backend::cufft)
#endif
#endif
#ifdef LUPINE_BUILD_CUDNN_BACKEND
LUPINE_CUDNN_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUDNN_VERSION >= 92300
LUPINE_DECLARE_HANDLER(RPC_cudnnGetExecutionPlanWorkspaceSize,
                       handle_cudnnGetExecutionPlanWorkspaceSize,
                       rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 90500
LUPINE_DECLARE_HANDLER(RPC_cudnnBackendPopulateCudaGraph,
                       handle_cudnnBackendPopulateCudaGraph, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 90500
LUPINE_DECLARE_HANDLER(RPC_cudnnBackendUpdateCudaGraph,
                       handle_cudnnBackendUpdateCudaGraph, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92200
LUPINE_DECLARE_HANDLER(RPC_cudnnSubquadraticOpsVersionCheck,
                       handle_cudnnSubquadraticOpsVersionCheck,
                       rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92200
LUPINE_DECLARE_HANDLER(RPC_cudnnCausalConv1dForward,
                       handle_cudnnCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92200
LUPINE_DECLARE_HANDLER(RPC_cudnnCausalConv1dBackward,
                       handle_cudnnCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
LUPINE_DECLARE_HANDLER(RPC_cudnnCausalConv1dNwhForward,
                       handle_cudnnCausalConv1dNwhForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
LUPINE_DECLARE_HANDLER(RPC_cudnnCausalConv1dNwhBackward,
                       handle_cudnnCausalConv1dNwhBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
LUPINE_DECLARE_HANDLER(RPC_cudnnB2BCausalConv1dForward,
                       handle_cudnnB2BCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
LUPINE_DECLARE_HANDLER(RPC_cudnnB2BCausalConv1dBackward,
                       handle_cudnnB2BCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnFFTCausalConv1dForward,
                       handle_cudnnFFTCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnFFTCausalConv1dBackward,
                       handle_cudnnFFTCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnLongFFTCausalConv1dGetBufferSizes,
                       handle_cudnnLongFFTCausalConv1dGetBufferSizes,
                       rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnLongFFTCausalConv1dForward,
                       handle_cudnnLongFFTCausalConv1dForward,
                       rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnLongFFTCausalConv1dBackward,
                       handle_cudnnLongFFTCausalConv1dBackward,
                       rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnGnnAggSimpleForward,
                       handle_cudnnGnnAggSimpleForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
LUPINE_DECLARE_HANDLER(RPC_cudnnGnnAggSimpleBackward,
                       handle_cudnnGnnAggSimpleBackward, rpc_backend::cudnn)
#endif
#endif
#ifdef LUPINE_BUILD_CURAND_BACKEND
LUPINE_CURAND_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)

#endif
#ifdef LUPINE_BUILD_CUSPARSE_BACKEND
LUPINE_CUSPARSE_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstSpVec,
                       handle_cusparseCreateConstSpVec, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstSpVecGet, handle_cusparseConstSpVecGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstSpVecGetValues,
                       handle_cusparseConstSpVecGetValues,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstDnVec,
                       handle_cusparseCreateConstDnVec, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstDnVecGet, handle_cusparseConstDnVecGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstDnVecGetValues,
                       handle_cusparseConstDnVecGetValues,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstSpMatGetValues,
                       handle_cusparseConstSpMatGetValues,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusparseBsrSetStridedBatch,
                       handle_cusparseBsrSetStridedBatch, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstCsr,
                       handle_cusparseCreateConstCsr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstCsc,
                       handle_cusparseCreateConstCsc, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstCsrGet, handle_cusparseConstCsrGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstCscGet, handle_cusparseConstCscGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateBsr, handle_cusparseCreateBsr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstBsr,
                       handle_cusparseCreateConstBsr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstCoo,
                       handle_cusparseCreateConstCoo, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstCooGet, handle_cusparseConstCooGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstBlockedEll,
                       handle_cusparseCreateConstBlockedEll,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstBlockedEllGet,
                       handle_cusparseConstBlockedEllGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateSlicedEll,
                       handle_cusparseCreateSlicedEll, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12102
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstSlicedEll,
                       handle_cusparseCreateConstSlicedEll,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateConstDnMat,
                       handle_cusparseCreateConstDnMat, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstDnMatGet, handle_cusparseConstDnMatGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstDnMatGetValues,
                       handle_cusparseConstDnMatGetValues,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12300
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMV_preprocess,
                       handle_cusparseSpMV_preprocess, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp_bufferSize,
                       handle_cusparseSpMVOp_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp_createDescr,
                       handle_cusparseSpMVOp_createDescr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp_destroyDescr,
                       handle_cusparseSpMVOp_destroyDescr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp_createPlan,
                       handle_cusparseSpMVOp_createPlan, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp_destroyPlan,
                       handle_cusparseSpMVOp_destroyPlan, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp_setGlobalUserData,
                       handle_cusparseSpMVOp_setGlobalUserData,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMVOp, handle_cusparseSpMVOp,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusparseSpSV_updateMatrix,
                       handle_cusparseSpSV_updateMatrix, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12300
LUPINE_DECLARE_HANDLER(RPC_cusparseSpSM_updateMatrix,
                       handle_cusparseSpSM_updateMatrix, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEMM_getNumProducts,
                       handle_cusparseSpGEMM_getNumProducts,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEMM_estimateMemory,
                       handle_cusparseSpGEMM_estimateMemory,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEAM_createDescr,
                       handle_cusparseSpGEAM_createDescr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEAM_destroyDescr,
                       handle_cusparseSpGEAM_destroyDescr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEAM_bufferSize,
                       handle_cusparseSpGEAM_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEAM_nnz, handle_cusparseSpGEAM_nnz,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
LUPINE_DECLARE_HANDLER(RPC_cusparseSpGEAM, handle_cusparseSpGEAM,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCaxpyi, handle_cusparseCaxpyi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsc2dense, handle_cusparseCcsc2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsr2dense, handle_cusparseCcsr2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrgemm2, handle_cusparseCcsrgemm2,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrgemm2_bufferSizeExt,
                       handle_cusparseCcsrgemm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrsm2_analysis,
                       handle_cusparseCcsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrsm2_bufferSizeExt,
                       handle_cusparseCcsrsm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrsm2_solve, handle_cusparseCcsrsm2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrsv2_analysis,
                       handle_cusparseCcsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrsv2_bufferSize,
                       handle_cusparseCcsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCcsrsv2_solve, handle_cusparseCcsrsv2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCdense2csc, handle_cusparseCdense2csc,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCdense2csr, handle_cusparseCdense2csr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCgemmi, handle_cusparseCgemmi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCgthr, handle_cusparseCgthr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCgthrz, handle_cusparseCgthrz,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstrainedGeMM,
                       handle_cusparseConstrainedGeMM, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseConstrainedGeMM_bufferSize,
                       handle_cusparseConstrainedGeMM_bufferSize,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCooAoSGet, handle_cusparseCooAoSGet,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateCooAoS, handle_cusparseCreateCooAoS,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateCsrgemm2Info,
                       handle_cusparseCreateCsrgemm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateCsrsm2Info,
                       handle_cusparseCreateCsrsm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCreateCsrsv2Info,
                       handle_cusparseCreateCsrsv2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCsctr, handle_cusparseCsctr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCsrmvEx, handle_cusparseCsrmvEx,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseCsrmvEx_bufferSize,
                       handle_cusparseCsrmvEx_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDaxpyi, handle_cusparseDaxpyi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsc2dense, handle_cusparseDcsc2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsr2dense, handle_cusparseDcsr2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrgemm2, handle_cusparseDcsrgemm2,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrgemm2_bufferSizeExt,
                       handle_cusparseDcsrgemm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrsm2_analysis,
                       handle_cusparseDcsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrsm2_bufferSizeExt,
                       handle_cusparseDcsrsm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrsm2_solve, handle_cusparseDcsrsm2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrsv2_analysis,
                       handle_cusparseDcsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrsv2_bufferSize,
                       handle_cusparseDcsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDcsrsv2_solve, handle_cusparseDcsrsv2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDdense2csc, handle_cusparseDdense2csc,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDdense2csr, handle_cusparseDdense2csr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDestroyCsrgemm2Info,
                       handle_cusparseDestroyCsrgemm2Info,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDestroyCsrsm2Info,
                       handle_cusparseDestroyCsrsm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDestroyCsrsv2Info,
                       handle_cusparseDestroyCsrsv2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDgemmi, handle_cusparseDgemmi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDgthr, handle_cusparseDgthr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDgthrz, handle_cusparseDgthrz,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDroti, handle_cusparseDroti,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseDsctr, handle_cusparseDsctr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSaxpyi, handle_cusparseSaxpyi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsc2dense, handle_cusparseScsc2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsr2dense, handle_cusparseScsr2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrgemm2, handle_cusparseScsrgemm2,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrgemm2_bufferSizeExt,
                       handle_cusparseScsrgemm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrsm2_analysis,
                       handle_cusparseScsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrsm2_bufferSizeExt,
                       handle_cusparseScsrsm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrsm2_solve, handle_cusparseScsrsm2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrsv2_analysis,
                       handle_cusparseScsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrsv2_bufferSize,
                       handle_cusparseScsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseScsrsv2_solve, handle_cusparseScsrsv2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSdense2csc, handle_cusparseSdense2csc,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSdense2csr, handle_cusparseSdense2csr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSgemmi, handle_cusparseSgemmi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSgthr, handle_cusparseSgthr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSgthrz, handle_cusparseSgthrz,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSpMatSetStridedBatch,
                       handle_cusparseSpMatSetStridedBatch,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSroti, handle_cusparseSroti,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseSsctr, handle_cusparseSsctr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseXcsrgemm2Nnz, handle_cusparseXcsrgemm2Nnz,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseXcsrsm2_zeroPivot,
                       handle_cusparseXcsrsm2_zeroPivot, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseXcsrsv2_zeroPivot,
                       handle_cusparseXcsrsv2_zeroPivot, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZaxpyi, handle_cusparseZaxpyi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsc2dense, handle_cusparseZcsc2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsr2dense, handle_cusparseZcsr2dense,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrgemm2, handle_cusparseZcsrgemm2,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrgemm2_bufferSizeExt,
                       handle_cusparseZcsrgemm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrsm2_analysis,
                       handle_cusparseZcsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrsm2_bufferSizeExt,
                       handle_cusparseZcsrsm2_bufferSizeExt,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrsm2_solve, handle_cusparseZcsrsm2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrsv2_analysis,
                       handle_cusparseZcsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrsv2_bufferSize,
                       handle_cusparseZcsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZcsrsv2_solve, handle_cusparseZcsrsv2_solve,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZdense2csc, handle_cusparseZdense2csc,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZdense2csr, handle_cusparseZdense2csr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZgemmi, handle_cusparseZgemmi,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZgthr, handle_cusparseZgthr,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZgthrz, handle_cusparseZgthrz,
                       rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
LUPINE_DECLARE_HANDLER(RPC_cusparseZsctr, handle_cusparseZsctr,
                       rpc_backend::cusparse)
#endif
#endif
#ifdef LUPINE_BUILD_CUSOLVER_BACKEND
LUPINE_CUSOLVER_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUSOLVER_VERSION >= 11500
LUPINE_DECLARE_HANDLER(RPC_cusolverDnSetDeterministicMode,
                       handle_cusolverDnSetDeterministicMode,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11500
LUPINE_DECLARE_HANDLER(RPC_cusolverDnGetDeterministicMode,
                       handle_cusolverDnGetDeterministicMode,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
LUPINE_DECLARE_HANDLER(RPC_cusolverDnSetMathMode, handle_cusolverDnSetMathMode,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
LUPINE_DECLARE_HANDLER(RPC_cusolverDnGetMathMode, handle_cusolverDnGetMathMode,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
LUPINE_DECLARE_HANDLER(RPC_cusolverDnSetEmulationStrategy,
                       handle_cusolverDnSetEmulationStrategy,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
LUPINE_DECLARE_HANDLER(RPC_cusolverDnGetEmulationStrategy,
                       handle_cusolverDnGetEmulationStrategy,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnSetFixedPointEmulationMantissaControl,
                       handle_cusolverDnSetFixedPointEmulationMantissaControl,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnGetFixedPointEmulationMantissaControl,
                       handle_cusolverDnGetFixedPointEmulationMantissaControl,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(
    RPC_cusolverDnSetFixedPointEmulationMaxMantissaBitCount,
    handle_cusolverDnSetFixedPointEmulationMaxMantissaBitCount,
    rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(
    RPC_cusolverDnGetFixedPointEmulationMaxMantissaBitCount,
    handle_cusolverDnGetFixedPointEmulationMaxMantissaBitCount,
    rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnSetFixedPointEmulationMantissaBitOffset,
                       handle_cusolverDnSetFixedPointEmulationMantissaBitOffset,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnGetFixedPointEmulationMantissaBitOffset,
                       handle_cusolverDnGetFixedPointEmulationMantissaBitOffset,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnSetEmulationSpecialValuesSupport,
                       handle_cusolverDnSetEmulationSpecialValuesSupport,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnGetEmulationSpecialValuesSupport,
                       handle_cusolverDnGetEmulationSpecialValuesSupport,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXsygvd_bufferSize,
                       handle_cusolverDnXsygvd_bufferSize,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXsygvd, handle_cusolverDnXsygvd,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXsygvdx_bufferSize,
                       handle_cusolverDnXsygvdx_bufferSize,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXsygvdx, handle_cusolverDnXsygvdx,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXstedc_bufferSize,
                       handle_cusolverDnXstedc_bufferSize,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXstedc, handle_cusolverDnXstedc,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXsyevBatched_bufferSize,
                       handle_cusolverDnXsyevBatched_bufferSize,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXsyevBatched,
                       handle_cusolverDnXsyevBatched, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXgeev_bufferSize,
                       handle_cusolverDnXgeev_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXgeev, handle_cusolverDnXgeev,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11600
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXlarft_bufferSize,
                       handle_cusolverDnXlarft_bufferSize,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11600
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXlarft, handle_cusolverDnXlarft,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXpolar_bufferSize,
                       handle_cusolverDnXpolar_bufferSize,
                       rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
LUPINE_DECLARE_HANDLER(RPC_cusolverDnXpolar, handle_cusolverDnXpolar,
                       rpc_backend::cusolver)
#endif
LUPINE_CUSOLVERMG_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)

#endif
#ifdef LUPINE_BUILD_NVRTC_BACKEND
LUPINE_NVRTC_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if CUDA_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetLTOIR, handle_nvrtcGetLTOIR,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetOptiXIR, handle_nvrtcGetOptiXIR,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetNVVM, handle_nvrtcGetNVVM,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13020
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetTileIR, handle_nvrtcGetTileIR,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13030
LUPINE_DECLARE_HANDLER(RPC_nvrtcInstallBundledHeaders,
                       handle_nvrtcInstallBundledHeaders, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13030
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetBundledHeadersInfo,
                       handle_nvrtcGetBundledHeadersInfo, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13030
LUPINE_DECLARE_HANDLER(RPC_nvrtcRemoveBundledHeaders,
                       handle_nvrtcRemoveBundledHeaders, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetLTOIRSize, handle_nvrtcGetLTOIRSize,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12000
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetOptiXIRSize, handle_nvrtcGetOptiXIRSize,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION < 13000
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetNVVMSize, handle_nvrtcGetNVVMSize,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetPCHHeapSize, handle_nvrtcGetPCHHeapSize,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
LUPINE_DECLARE_HANDLER(RPC_nvrtcSetPCHHeapSize, handle_nvrtcSetPCHHeapSize,
                       rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetPCHCreateStatus,
                       handle_nvrtcGetPCHCreateStatus, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetPCHHeapSizeRequired,
                       handle_nvrtcGetPCHHeapSizeRequired, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13020
LUPINE_DECLARE_HANDLER(RPC_nvrtcGetTileIRSize, handle_nvrtcGetTileIRSize,
                       rpc_backend::nvrtc)
#endif
#endif
#ifdef LUPINE_BUILD_NCCL_BACKEND
LUPINE_NCCL_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetStr, handle_ncclParamGetStr,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetParameter, handle_ncclParamGetParameter,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetAllParameterKeys,
                       handle_ncclParamGetAllParameterKeys, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
LUPINE_DECLARE_HANDLER(RPC_ncclMemAlloc, handle_ncclMemAlloc, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
LUPINE_DECLARE_HANDLER(RPC_ncclMemFree, handle_ncclMemFree, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22807
LUPINE_DECLARE_HANDLER(RPC_ncclCommRevoke, handle_ncclCommRevoke,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21801
LUPINE_DECLARE_HANDLER(RPC_ncclCommSplit, handle_ncclCommSplit,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22703
LUPINE_DECLARE_HANDLER(RPC_ncclCommShrink, handle_ncclCommShrink,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
LUPINE_DECLARE_HANDLER(RPC_ncclCommGetUniqueId, handle_ncclCommGetUniqueId,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
LUPINE_DECLARE_HANDLER(RPC_ncclCommGrow, handle_ncclCommGrow, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22304
LUPINE_DECLARE_HANDLER(RPC_ncclCommInitRankScalable,
                       handle_ncclCommInitRankScalable, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22403
LUPINE_DECLARE_HANDLER(RPC_ncclResetDebugInit, handle_ncclResetDebugInit,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
LUPINE_DECLARE_HANDLER(RPC_ncclCommRegister, handle_ncclCommRegister,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
LUPINE_DECLARE_HANDLER(RPC_ncclCommDeregister, handle_ncclCommDeregister,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22907
LUPINE_DECLARE_HANDLER(RPC_ncclCommSuspend, handle_ncclCommSuspend,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22907
LUPINE_DECLARE_HANDLER(RPC_ncclCommResume, handle_ncclCommResume,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22907
LUPINE_DECLARE_HANDLER(RPC_ncclCommMemStats, handle_ncclCommMemStats,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22703
LUPINE_DECLARE_HANDLER(RPC_ncclCommWindowRegister,
                       handle_ncclCommWindowRegister, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22703
LUPINE_DECLARE_HANDLER(RPC_ncclCommWindowDeregister,
                       handle_ncclCommWindowDeregister, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
LUPINE_DECLARE_HANDLER(RPC_ncclWinGetUserPtr, handle_ncclWinGetUserPtr,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22803
LUPINE_DECLARE_HANDLER(RPC_ncclAlltoAll, handle_ncclAlltoAll, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22803
LUPINE_DECLARE_HANDLER(RPC_ncclGather, handle_ncclGather, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22803
LUPINE_DECLARE_HANDLER(RPC_ncclScatter, handle_ncclScatter, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclAllReduceConfig, handle_ncclAllReduceConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclBroadcastConfig, handle_ncclBroadcastConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclReduceConfig, handle_ncclReduceConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclAllGatherConfig, handle_ncclAllGatherConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclReduceScatterConfig,
                       handle_ncclReduceScatterConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclAlltoAllConfig, handle_ncclAlltoAllConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclGatherConfig, handle_ncclGatherConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
LUPINE_DECLARE_HANDLER(RPC_ncclScatterConfig, handle_ncclScatterConfig,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
LUPINE_DECLARE_HANDLER(RPC_ncclPutSignal, handle_ncclPutSignal,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
LUPINE_DECLARE_HANDLER(RPC_ncclSignal, handle_ncclSignal, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
LUPINE_DECLARE_HANDLER(RPC_ncclWaitSignal, handle_ncclWaitSignal,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22203
LUPINE_DECLARE_HANDLER(RPC_ncclGroupSimulateEnd, handle_ncclGroupSimulateEnd,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamBind, handle_ncclParamBind,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetI8, handle_ncclParamGetI8,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetI16, handle_ncclParamGetI16,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetI32, handle_ncclParamGetI32,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetI64, handle_ncclParamGetI64,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetU8, handle_ncclParamGetU8,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetU16, handle_ncclParamGetU16,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetU32, handle_ncclParamGetU32,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGetU64, handle_ncclParamGetU64,
                       rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamGet, handle_ncclParamGet, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
LUPINE_DECLARE_HANDLER(RPC_ncclParamDumpAll, handle_ncclParamDumpAll,
                       rpc_backend::nccl)
#endif
#endif
#ifdef LUPINE_BUILD_NVJITLINK_BACKEND
LUPINE_NVJITLINK_RPC_HANDLERS(LUPINE_DECLARE_HANDLER)
#if LUPINE_NVJITLINK_HAS_LTOIR
LUPINE_DECLARE_HANDLER(RPC_nvJitLinkGetLinkedLTOIR,
                       handle_nvJitLinkGetLinkedLTOIR, rpc_backend::nvjitlink)
#endif
#if LUPINE_NVJITLINK_HAS_LTOIR
LUPINE_DECLARE_HANDLER(RPC_nvJitLinkGetLinkedLTOIRSize,
                       handle_nvJitLinkGetLinkedLTOIRSize,
                       rpc_backend::nvjitlink)
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
#if CUDA_VERSION >= 12090
      LUPINE_REGISTER_HANDLER(RPC_cuLogsRegisterCallback, handle_cuLogsRegisterCallback, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12090
      LUPINE_REGISTER_HANDLER(RPC_cuLogsUnregisterCallback, handle_cuLogsUnregisterCallback, rpc_backend::cuda)
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
#if CUDA_VERSION >= 12090
      LUPINE_REGISTER_HANDLER(RPC_cuLogsCurrent, handle_cuLogsCurrent, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12090
      LUPINE_REGISTER_HANDLER(RPC_cuLogsDumpToFile, handle_cuLogsDumpToFile, rpc_backend::cuda)
#endif
#if CUDA_VERSION >= 12090
      LUPINE_REGISTER_HANDLER(RPC_cuLogsDumpToMemory, handle_cuLogsDumpToMemory, rpc_backend::cuda)
#endif
#endif
#ifdef LUPINE_BUILD_CUDART_BACKEND
      LUPINE_CUDART_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaFuncGetName, handle_cudaFuncGetName, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLaunchHostFunc_v2, handle_cudaLaunchHostFunc_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamGetCaptureInfo_v2, handle_cudaStreamGetCaptureInfo_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamGetCaptureInfo_v3, handle_cudaStreamGetCaptureInfo_v3, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphAddNode, handle_cudaGraphAddNode, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphAddNode_v2, handle_cudaGraphAddNode_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphInstantiateWithParams, handle_cudaGraphInstantiateWithParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC___cudaLaunchKernel, handle___cudaLaunchKernel, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDevResourceGenerateDesc, handle_cudaDevResourceGenerateDesc, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDevSmResourceSplit, handle_cudaDevSmResourceSplit, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDevSmResourceSplitByCount, handle_cudaDevSmResourceSplitByCount, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDeviceGetDevResource, handle_cudaDeviceGetDevResource, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDeviceGetExecutionCtx, handle_cudaDeviceGetExecutionCtx, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDeviceGetHostAtomicCapabilities, handle_cudaDeviceGetHostAtomicCapabilities, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaDeviceGetP2PAtomicCapabilities, handle_cudaDeviceGetP2PAtomicCapabilities, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaDeviceRegisterAsyncNotification, handle_cudaDeviceRegisterAsyncNotification, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaDeviceUnregisterAsyncNotification, handle_cudaDeviceUnregisterAsyncNotification, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxDestroy, handle_cudaExecutionCtxDestroy, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxGetDevResource, handle_cudaExecutionCtxGetDevResource, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxGetDevice, handle_cudaExecutionCtxGetDevice, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxGetId, handle_cudaExecutionCtxGetId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxRecordEvent, handle_cudaExecutionCtxRecordEvent, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxStreamCreate, handle_cudaExecutionCtxStreamCreate, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxSynchronize, handle_cudaExecutionCtxSynchronize, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaExecutionCtxWaitEvent, handle_cudaExecutionCtxWaitEvent, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaFuncGetParamCount, handle_cudaFuncGetParamCount, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaFuncGetParamInfo, handle_cudaFuncGetParamInfo, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGetDriverEntryPointByVersion, handle_cudaGetDriverEntryPointByVersion, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGetKernel, handle_cudaGetKernel, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGreenCtxCreate, handle_cudaGreenCtxCreate, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaInitDevice, handle_cudaInitDevice, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaKernelSetAttributeForDevice, handle_cudaKernelSetAttributeForDevice, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryEnumerateKernels, handle_cudaLibraryEnumerateKernels, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryGetGlobal, handle_cudaLibraryGetGlobal, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryGetKernel, handle_cudaLibraryGetKernel, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryGetKernelCount, handle_cudaLibraryGetKernelCount, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryGetManaged, handle_cudaLibraryGetManaged, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryGetUnifiedFunction, handle_cudaLibraryGetUnifiedFunction, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryLoadData, handle_cudaLibraryLoadData, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryLoadFromFile, handle_cudaLibraryLoadFromFile, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLibraryUnload, handle_cudaLibraryUnload, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLogsCurrent, handle_cudaLogsCurrent, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLogsDumpToFile, handle_cudaLogsDumpToFile, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLogsDumpToMemory, handle_cudaLogsDumpToMemory, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLogsRegisterCallback, handle_cudaLogsRegisterCallback, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaLogsUnregisterCallback, handle_cudaLogsUnregisterCallback, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12020 && CUDART_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemAdvise_v2, handle_cudaMemAdvise_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemDiscardAndPrefetchBatchAsync, handle_cudaMemDiscardAndPrefetchBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemDiscardBatchAsync, handle_cudaMemDiscardBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemGetDefaultMemPool, handle_cudaMemGetDefaultMemPool, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemGetMemPool, handle_cudaMemGetMemPool, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12020 && CUDART_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemPrefetchAsync_v2, handle_cudaMemPrefetchAsync_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemPrefetchBatchAsync, handle_cudaMemPrefetchBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemSetMemPool, handle_cudaMemSetMemPool, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemcpy3DBatchAsync, handle_cudaMemcpy3DBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemcpy3DWithAttributesAsync, handle_cudaMemcpy3DWithAttributesAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemcpyBatchAsync, handle_cudaMemcpyBatchAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaMemcpyWithAttributesAsync, handle_cudaMemcpyWithAttributesAsync, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamBeginCaptureToGraph, handle_cudaStreamBeginCaptureToGraph, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamBeginRecaptureToGraph, handle_cudaStreamBeginRecaptureToGraph, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamGetDevResource, handle_cudaStreamGetDevResource, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamGetDevice, handle_cudaStreamGetDevice, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamGetId, handle_cudaStreamGetId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaStreamUpdateCaptureDependencies_v2, handle_cudaStreamUpdateCaptureDependencies_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphAddDependencies, handle_cudaGraphAddDependencies, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphConditionalHandleCreate, handle_cudaGraphConditionalHandleCreate, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphConditionalHandleCreate_v2, handle_cudaGraphConditionalHandleCreate_v2, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphExecGetFlags, handle_cudaGraphExecGetFlags, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphExecGetId, handle_cudaGraphExecGetId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphExecNodeSetParams, handle_cudaGraphExecNodeSetParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphGetEdges, handle_cudaGraphGetEdges, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphGetId, handle_cudaGraphGetId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphKernelNodeCopyAttributes, handle_cudaGraphKernelNodeCopyAttributes, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeGetContainingGraph, handle_cudaGraphNodeGetContainingGraph, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeGetDependencies, handle_cudaGraphNodeGetDependencies, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeGetDependentNodes, handle_cudaGraphNodeGetDependentNodes, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeGetLocalId, handle_cudaGraphNodeGetLocalId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeGetParams, handle_cudaGraphNodeGetParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeGetToolsId, handle_cudaGraphNodeGetToolsId, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphNodeSetParams, handle_cudaGraphNodeSetParams, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC_cudaGraphRemoveDependencies, handle_cudaGraphRemoveDependencies, rpc_backend::cudart)
#endif
#if CUDART_VERSION >= 13000
      LUPINE_REGISTER_HANDLER(RPC___cudaGetKernel, handle___cudaGetKernel, rpc_backend::cudart)
#endif
#endif
#ifdef LUPINE_BUILD_CUBLAS_BACKEND
      LUPINE_CUBLAS_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGetMatrixAsync_64, handle_cublasGetMatrixAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGetMatrix_64, handle_cublasGetMatrix_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGetVectorAsync_64, handle_cublasGetVectorAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGetVector_64, handle_cublasGetVector_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSetMatrixAsync_64, handle_cublasSetMatrixAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSetMatrix_64, handle_cublasSetMatrix_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSetVectorAsync_64, handle_cublasSetVectorAsync_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSetVector_64, handle_cublasSetVector_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasAsumEx_64, handle_cublasAsumEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasAxpyEx_64, handle_cublasAxpyEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCaxpy_v2_64, handle_cublasCaxpy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCcopy_v2_64, handle_cublasCcopy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCdgmm_64, handle_cublasCdgmm_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCdotc_v2_64, handle_cublasCdotc_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCdotu_v2_64, handle_cublasCdotu_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgbmv_v2_64, handle_cublasCgbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgeam_64, handle_cublasCgeam_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemm3mBatched_64, handle_cublasCgemm3mBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemm3mEx_64, handle_cublasCgemm3mEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemm3mStridedBatched_64, handle_cublasCgemm3mStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemm3m_64, handle_cublasCgemm3m_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemmBatched_64, handle_cublasCgemmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemmEx_64, handle_cublasCgemmEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemmStridedBatched_64, handle_cublasCgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemm_v2_64, handle_cublasCgemm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemvBatched_64, handle_cublasCgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemvStridedBatched_64, handle_cublasCgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgemv_v2_64, handle_cublasCgemv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgerc_v2_64, handle_cublasCgerc_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCgeru_v2_64, handle_cublasCgeru_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasChbmv_v2_64, handle_cublasChbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasChemm_v2_64, handle_cublasChemm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasChemv_v2_64, handle_cublasChemv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCher2_v2_64, handle_cublasCher2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCher2k_v2_64, handle_cublasCher2k_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCher_v2_64, handle_cublasCher_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCherk3mEx_64, handle_cublasCherk3mEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCherkEx_64, handle_cublasCherkEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCherk_v2_64, handle_cublasCherk_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCherkx_64, handle_cublasCherkx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasChpmv_v2_64, handle_cublasChpmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasChpr2_v2_64, handle_cublasChpr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasChpr_v2_64, handle_cublasChpr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCopyEx_64, handle_cublasCopyEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCrot_v2_64, handle_cublasCrot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCscal_v2_64, handle_cublasCscal_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsrot_v2_64, handle_cublasCsrot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsscal_v2_64, handle_cublasCsscal_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCswap_v2_64, handle_cublasCswap_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsymm_v2_64, handle_cublasCsymm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsymv_v2_64, handle_cublasCsymv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyr2_v2_64, handle_cublasCsyr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyr2k_v2_64, handle_cublasCsyr2k_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyr_v2_64, handle_cublasCsyr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyrk3mEx_64, handle_cublasCsyrk3mEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyrkEx_64, handle_cublasCsyrkEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyrk_v2_64, handle_cublasCsyrk_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCsyrkx_64, handle_cublasCsyrkx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtbmv_v2_64, handle_cublasCtbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtbsv_v2_64, handle_cublasCtbsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtpmv_v2_64, handle_cublasCtpmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtpsv_v2_64, handle_cublasCtpsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtrmm_v2_64, handle_cublasCtrmm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtrmv_v2_64, handle_cublasCtrmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtrsmBatched_64, handle_cublasCtrsmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtrsm_v2_64, handle_cublasCtrsm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasCtrsv_v2_64, handle_cublasCtrsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDasum_v2_64, handle_cublasDasum_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDaxpy_v2_64, handle_cublasDaxpy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDcopy_v2_64, handle_cublasDcopy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDdgmm_64, handle_cublasDdgmm_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDdot_v2_64, handle_cublasDdot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgbmv_v2_64, handle_cublasDgbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgeam_64, handle_cublasDgeam_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemmBatched_64, handle_cublasDgemmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemmGroupedBatched, handle_cublasDgemmGroupedBatched, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemmGroupedBatched_64, handle_cublasDgemmGroupedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemmStridedBatched_64, handle_cublasDgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemm_v2_64, handle_cublasDgemm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemvBatched_64, handle_cublasDgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemvStridedBatched_64, handle_cublasDgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDgemv_v2_64, handle_cublasDgemv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDger_v2_64, handle_cublasDger_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDnrm2_v2_64, handle_cublasDnrm2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDotEx_64, handle_cublasDotEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDotcEx_64, handle_cublasDotcEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDrot_v2_64, handle_cublasDrot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDrotm_v2_64, handle_cublasDrotm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsbmv_v2_64, handle_cublasDsbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDscal_v2_64, handle_cublasDscal_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDspmv_v2_64, handle_cublasDspmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDspr2_v2_64, handle_cublasDspr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDspr_v2_64, handle_cublasDspr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDswap_v2_64, handle_cublasDswap_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsymm_v2_64, handle_cublasDsymm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsymv_v2_64, handle_cublasDsymv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsyr2_v2_64, handle_cublasDsyr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsyr2k_v2_64, handle_cublasDsyr2k_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsyr_v2_64, handle_cublasDsyr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsyrk_v2_64, handle_cublasDsyrk_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDsyrkx_64, handle_cublasDsyrkx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtbmv_v2_64, handle_cublasDtbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtbsv_v2_64, handle_cublasDtbsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtpmv_v2_64, handle_cublasDtpmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtpsv_v2_64, handle_cublasDtpsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtrmm_v2_64, handle_cublasDtrmm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtrmv_v2_64, handle_cublasDtrmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtrsmBatched_64, handle_cublasDtrsmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtrsm_v2_64, handle_cublasDtrsm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDtrsv_v2_64, handle_cublasDtrsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDzasum_v2_64, handle_cublasDzasum_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasDznrm2_v2_64, handle_cublasDznrm2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGemmBatchedEx_64, handle_cublasGemmBatchedEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGemmEx_64, handle_cublasGemmEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
      LUPINE_REGISTER_HANDLER(RPC_cublasGemmGroupedBatchedEx, handle_cublasGemmGroupedBatchedEx, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
      LUPINE_REGISTER_HANDLER(RPC_cublasGemmGroupedBatchedEx_64, handle_cublasGemmGroupedBatchedEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasGemmStridedBatchedEx_64, handle_cublasGemmStridedBatchedEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasGetEmulationSpecialValuesSupport, handle_cublasGetEmulationSpecialValuesSupport, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
      LUPINE_REGISTER_HANDLER(RPC_cublasGetEmulationStrategy, handle_cublasGetEmulationStrategy, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasGetFixedPointEmulationMantissaBitCountPointer, handle_cublasGetFixedPointEmulationMantissaBitCountPointer, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasGetFixedPointEmulationMantissaBitOffset, handle_cublasGetFixedPointEmulationMantissaBitOffset, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasGetFixedPointEmulationMantissaControl, handle_cublasGetFixedPointEmulationMantissaControl, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasGetFixedPointEmulationMaxMantissaBitCount, handle_cublasGetFixedPointEmulationMaxMantissaBitCount, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHSHgemvBatched_64, handle_cublasHSHgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHSHgemvStridedBatched_64, handle_cublasHSHgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHSSgemvBatched_64, handle_cublasHSSgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHSSgemvStridedBatched_64, handle_cublasHSSgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHgemmBatched_64, handle_cublasHgemmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHgemmStridedBatched_64, handle_cublasHgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasHgemm_64, handle_cublasHgemm_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIamaxEx_64, handle_cublasIamaxEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIaminEx_64, handle_cublasIaminEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIcamax_v2_64, handle_cublasIcamax_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIcamin_v2_64, handle_cublasIcamin_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIdamax_v2_64, handle_cublasIdamax_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIdamin_v2_64, handle_cublasIdamin_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIsamax_v2_64, handle_cublasIsamax_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIsamin_v2_64, handle_cublasIsamin_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIzamax_v2_64, handle_cublasIzamax_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasIzamin_v2_64, handle_cublasIzamin_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasNrm2Ex_64, handle_cublasNrm2Ex_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasRotEx_64, handle_cublasRotEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasRotmEx_64, handle_cublasRotmEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSasum_v2_64, handle_cublasSasum_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSaxpy_v2_64, handle_cublasSaxpy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasScalEx_64, handle_cublasScalEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasScasum_v2_64, handle_cublasScasum_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasScnrm2_v2_64, handle_cublasScnrm2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasScopy_v2_64, handle_cublasScopy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSdgmm_64, handle_cublasSdgmm_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSdot_v2_64, handle_cublasSdot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasSetEmulationSpecialValuesSupport, handle_cublasSetEmulationSpecialValuesSupport, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
      LUPINE_REGISTER_HANDLER(RPC_cublasSetEmulationStrategy, handle_cublasSetEmulationStrategy, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
      LUPINE_REGISTER_HANDLER(RPC_cublasSetEnvironmentMode, handle_cublasSetEnvironmentMode, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasSetFixedPointEmulationMantissaBitCountPointer, handle_cublasSetFixedPointEmulationMantissaBitCountPointer, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasSetFixedPointEmulationMantissaBitOffset, handle_cublasSetFixedPointEmulationMantissaBitOffset, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasSetFixedPointEmulationMantissaControl, handle_cublasSetFixedPointEmulationMantissaControl, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasSetFixedPointEmulationMaxMantissaBitCount, handle_cublasSetFixedPointEmulationMaxMantissaBitCount, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgbmv_v2_64, handle_cublasSgbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgeam_64, handle_cublasSgeam_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemmBatched_64, handle_cublasSgemmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemmEx_64, handle_cublasSgemmEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemmGroupedBatched, handle_cublasSgemmGroupedBatched, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120500
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemmGroupedBatched_64, handle_cublasSgemmGroupedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemmStridedBatched_64, handle_cublasSgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemm_v2_64, handle_cublasSgemm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemvBatched_64, handle_cublasSgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemvStridedBatched_64, handle_cublasSgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSgemv_v2_64, handle_cublasSgemv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSger_v2_64, handle_cublasSger_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSnrm2_v2_64, handle_cublasSnrm2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSrot_v2_64, handle_cublasSrot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSrotm_v2_64, handle_cublasSrotm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsbmv_v2_64, handle_cublasSsbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSscal_v2_64, handle_cublasSscal_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSspmv_v2_64, handle_cublasSspmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSspr2_v2_64, handle_cublasSspr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSspr_v2_64, handle_cublasSspr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSswap_v2_64, handle_cublasSswap_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsymm_v2_64, handle_cublasSsymm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsymv_v2_64, handle_cublasSsymv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsyr2_v2_64, handle_cublasSsyr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsyr2k_v2_64, handle_cublasSsyr2k_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsyr_v2_64, handle_cublasSsyr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsyrk_v2_64, handle_cublasSsyrk_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSsyrkx_64, handle_cublasSsyrkx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStbmv_v2_64, handle_cublasStbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStbsv_v2_64, handle_cublasStbsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStpmv_v2_64, handle_cublasStpmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStpsv_v2_64, handle_cublasStpsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStrmm_v2_64, handle_cublasStrmm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStrmv_v2_64, handle_cublasStrmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStrsmBatched_64, handle_cublasStrsmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStrsm_v2_64, handle_cublasStrsm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasStrsv_v2_64, handle_cublasStrsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasSwapEx_64, handle_cublasSwapEx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasTSSgemvBatched_64, handle_cublasTSSgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasTSSgemvStridedBatched_64, handle_cublasTSSgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasTSTgemvBatched_64, handle_cublasTSTgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasTSTgemvStridedBatched_64, handle_cublasTSTgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZaxpy_v2_64, handle_cublasZaxpy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZcopy_v2_64, handle_cublasZcopy_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZdgmm_64, handle_cublasZdgmm_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZdotc_v2_64, handle_cublasZdotc_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZdotu_v2_64, handle_cublasZdotu_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZdrot_v2_64, handle_cublasZdrot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZdscal_v2_64, handle_cublasZdscal_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgbmv_v2_64, handle_cublasZgbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgeam_64, handle_cublasZgeam_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemm3m_64, handle_cublasZgemm3m_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemmBatched_64, handle_cublasZgemmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemmStridedBatched_64, handle_cublasZgemmStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemm_v2_64, handle_cublasZgemm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemvBatched_64, handle_cublasZgemvBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemvStridedBatched_64, handle_cublasZgemvStridedBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgemv_v2_64, handle_cublasZgemv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgerc_v2_64, handle_cublasZgerc_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZgeru_v2_64, handle_cublasZgeru_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZhbmv_v2_64, handle_cublasZhbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZhemm_v2_64, handle_cublasZhemm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZhemv_v2_64, handle_cublasZhemv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZher2_v2_64, handle_cublasZher2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZher2k_v2_64, handle_cublasZher2k_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZher_v2_64, handle_cublasZher_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZherk_v2_64, handle_cublasZherk_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZherkx_64, handle_cublasZherkx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZhpmv_v2_64, handle_cublasZhpmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZhpr2_v2_64, handle_cublasZhpr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZhpr_v2_64, handle_cublasZhpr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZrot_v2_64, handle_cublasZrot_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZscal_v2_64, handle_cublasZscal_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZswap_v2_64, handle_cublasZswap_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsymm_v2_64, handle_cublasZsymm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsymv_v2_64, handle_cublasZsymv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsyr2_v2_64, handle_cublasZsyr2_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsyr2k_v2_64, handle_cublasZsyr2k_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsyr_v2_64, handle_cublasZsyr_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsyrk_v2_64, handle_cublasZsyrk_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZsyrkx_64, handle_cublasZsyrkx_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtbmv_v2_64, handle_cublasZtbmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtbsv_v2_64, handle_cublasZtbsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtpmv_v2_64, handle_cublasZtpmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtpsv_v2_64, handle_cublasZtpsv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtrmm_v2_64, handle_cublasZtrmm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtrmv_v2_64, handle_cublasZtrmv_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtrsmBatched_64, handle_cublasZtrsmBatched_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtrsm_v2_64, handle_cublasZtrsm_v2_64, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120000
      LUPINE_REGISTER_HANDLER(RPC_cublasZtrsv_v2_64, handle_cublasZtrsv_v2_64, rpc_backend::cublas)
#endif
      LUPINE_CUBLASLT_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_lupineCublasLtEmulationDescCopy, handle_lupineCublasLtEmulationDescCopy, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 120900
      LUPINE_REGISTER_HANDLER(RPC_cublasLtDisableCpuInstructionsSetMask, handle_cublasLtDisableCpuInstructionsSetMask, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasLtEmulationDescGetAttribute, handle_cublasLtEmulationDescGetAttribute, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasLtEmulationDescInit_internal, handle_cublasLtEmulationDescInit_internal, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130100
      LUPINE_REGISTER_HANDLER(RPC_cublasLtEmulationDescSetAttribute, handle_cublasLtEmulationDescSetAttribute, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130300
      LUPINE_REGISTER_HANDLER(RPC_cublasLtMatmulAlgoCheckForStream, handle_cublasLtMatmulAlgoCheckForStream, rpc_backend::cublas)
#endif
#if CUBLAS_VERSION >= 130300
      LUPINE_REGISTER_HANDLER(RPC_cublasLtMatmulAlgoGetHeuristicForStream, handle_cublasLtMatmulAlgoGetHeuristicForStream, rpc_backend::cublas)
#endif
#endif
#ifdef LUPINE_BUILD_CUFFT_BACKEND
      LUPINE_CUFFT_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUFFT_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cufftXtSetJITCallback, handle_cufftXtSetJITCallback, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11300 && CUFFT_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC___cufftXtSetJITCallback_12_7, handle___cufftXtSetJITCallback_12_7, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11200
      LUPINE_REGISTER_HANDLER(RPC_cufftGetPlanPropertyInt64, handle_cufftGetPlanPropertyInt64, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11200
      LUPINE_REGISTER_HANDLER(RPC_cufftResetPlanProperty, handle_cufftResetPlanProperty, rpc_backend::cufft)
#endif
#if CUFFT_VERSION >= 11200
      LUPINE_REGISTER_HANDLER(RPC_cufftSetPlanPropertyInt64, handle_cufftSetPlanPropertyInt64, rpc_backend::cufft)
#endif
#endif
#ifdef LUPINE_BUILD_CUDNN_BACKEND
      LUPINE_CUDNN_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUDNN_VERSION >= 92300
      LUPINE_REGISTER_HANDLER(RPC_cudnnGetExecutionPlanWorkspaceSize, handle_cudnnGetExecutionPlanWorkspaceSize, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 90500
      LUPINE_REGISTER_HANDLER(RPC_cudnnBackendPopulateCudaGraph, handle_cudnnBackendPopulateCudaGraph, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 90500
      LUPINE_REGISTER_HANDLER(RPC_cudnnBackendUpdateCudaGraph, handle_cudnnBackendUpdateCudaGraph, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92200
      LUPINE_REGISTER_HANDLER(RPC_cudnnSubquadraticOpsVersionCheck, handle_cudnnSubquadraticOpsVersionCheck, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92200
      LUPINE_REGISTER_HANDLER(RPC_cudnnCausalConv1dForward, handle_cudnnCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92200
      LUPINE_REGISTER_HANDLER(RPC_cudnnCausalConv1dBackward, handle_cudnnCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
      LUPINE_REGISTER_HANDLER(RPC_cudnnCausalConv1dNwhForward, handle_cudnnCausalConv1dNwhForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
      LUPINE_REGISTER_HANDLER(RPC_cudnnCausalConv1dNwhBackward, handle_cudnnCausalConv1dNwhBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
      LUPINE_REGISTER_HANDLER(RPC_cudnnB2BCausalConv1dForward, handle_cudnnB2BCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92400
      LUPINE_REGISTER_HANDLER(RPC_cudnnB2BCausalConv1dBackward, handle_cudnnB2BCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnFFTCausalConv1dForward, handle_cudnnFFTCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnFFTCausalConv1dBackward, handle_cudnnFFTCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnLongFFTCausalConv1dGetBufferSizes, handle_cudnnLongFFTCausalConv1dGetBufferSizes, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnLongFFTCausalConv1dForward, handle_cudnnLongFFTCausalConv1dForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnLongFFTCausalConv1dBackward, handle_cudnnLongFFTCausalConv1dBackward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnGnnAggSimpleForward, handle_cudnnGnnAggSimpleForward, rpc_backend::cudnn)
#endif
#if CUDNN_VERSION >= 92600
      LUPINE_REGISTER_HANDLER(RPC_cudnnGnnAggSimpleBackward, handle_cudnnGnnAggSimpleBackward, rpc_backend::cudnn)
#endif
#endif
#ifdef LUPINE_BUILD_CURAND_BACKEND
      LUPINE_CURAND_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)

#endif
#ifdef LUPINE_BUILD_CUSPARSE_BACKEND
      LUPINE_CUSPARSE_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstSpVec, handle_cusparseCreateConstSpVec, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstSpVecGet, handle_cusparseConstSpVecGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstSpVecGetValues, handle_cusparseConstSpVecGetValues, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstDnVec, handle_cusparseCreateConstDnVec, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstDnVecGet, handle_cusparseConstDnVecGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstDnVecGetValues, handle_cusparseConstDnVecGetValues, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstSpMatGetValues, handle_cusparseConstSpMatGetValues, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusparseBsrSetStridedBatch, handle_cusparseBsrSetStridedBatch, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstCsr, handle_cusparseCreateConstCsr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstCsc, handle_cusparseCreateConstCsc, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstCsrGet, handle_cusparseConstCsrGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstCscGet, handle_cusparseConstCscGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateBsr, handle_cusparseCreateBsr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstBsr, handle_cusparseCreateConstBsr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstCoo, handle_cusparseCreateConstCoo, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstCooGet, handle_cusparseConstCooGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstBlockedEll, handle_cusparseCreateConstBlockedEll, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstBlockedEllGet, handle_cusparseConstBlockedEllGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateSlicedEll, handle_cusparseCreateSlicedEll, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12102
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstSlicedEll, handle_cusparseCreateConstSlicedEll, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateConstDnMat, handle_cusparseCreateConstDnMat, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstDnMatGet, handle_cusparseConstDnMatGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstDnMatGetValues, handle_cusparseConstDnMatGetValues, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12300
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMV_preprocess, handle_cusparseSpMV_preprocess, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp_bufferSize, handle_cusparseSpMVOp_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp_createDescr, handle_cusparseSpMVOp_createDescr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp_destroyDescr, handle_cusparseSpMVOp_destroyDescr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp_createPlan, handle_cusparseSpMVOp_createPlan, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp_destroyPlan, handle_cusparseSpMVOp_destroyPlan, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp_setGlobalUserData, handle_cusparseSpMVOp_setGlobalUserData, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12702
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMVOp, handle_cusparseSpMVOp, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpSV_updateMatrix, handle_cusparseSpSV_updateMatrix, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12300
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpSM_updateMatrix, handle_cusparseSpSM_updateMatrix, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEMM_getNumProducts, handle_cusparseSpGEMM_getNumProducts, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEMM_estimateMemory, handle_cusparseSpGEMM_estimateMemory, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEAM_createDescr, handle_cusparseSpGEAM_createDescr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEAM_destroyDescr, handle_cusparseSpGEAM_destroyDescr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEAM_bufferSize, handle_cusparseSpGEAM_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEAM_nnz, handle_cusparseSpGEAM_nnz, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION >= 12801
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpGEAM, handle_cusparseSpGEAM, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCaxpyi, handle_cusparseCaxpyi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsc2dense, handle_cusparseCcsc2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsr2dense, handle_cusparseCcsr2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrgemm2, handle_cusparseCcsrgemm2, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrgemm2_bufferSizeExt, handle_cusparseCcsrgemm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrsm2_analysis, handle_cusparseCcsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrsm2_bufferSizeExt, handle_cusparseCcsrsm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrsm2_solve, handle_cusparseCcsrsm2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrsv2_analysis, handle_cusparseCcsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrsv2_bufferSize, handle_cusparseCcsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCcsrsv2_solve, handle_cusparseCcsrsv2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCdense2csc, handle_cusparseCdense2csc, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCdense2csr, handle_cusparseCdense2csr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCgemmi, handle_cusparseCgemmi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCgthr, handle_cusparseCgthr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCgthrz, handle_cusparseCgthrz, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstrainedGeMM, handle_cusparseConstrainedGeMM, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseConstrainedGeMM_bufferSize, handle_cusparseConstrainedGeMM_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCooAoSGet, handle_cusparseCooAoSGet, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateCooAoS, handle_cusparseCreateCooAoS, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateCsrgemm2Info, handle_cusparseCreateCsrgemm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateCsrsm2Info, handle_cusparseCreateCsrsm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCreateCsrsv2Info, handle_cusparseCreateCsrsv2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCsctr, handle_cusparseCsctr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCsrmvEx, handle_cusparseCsrmvEx, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseCsrmvEx_bufferSize, handle_cusparseCsrmvEx_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDaxpyi, handle_cusparseDaxpyi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsc2dense, handle_cusparseDcsc2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsr2dense, handle_cusparseDcsr2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrgemm2, handle_cusparseDcsrgemm2, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrgemm2_bufferSizeExt, handle_cusparseDcsrgemm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrsm2_analysis, handle_cusparseDcsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrsm2_bufferSizeExt, handle_cusparseDcsrsm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrsm2_solve, handle_cusparseDcsrsm2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrsv2_analysis, handle_cusparseDcsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrsv2_bufferSize, handle_cusparseDcsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDcsrsv2_solve, handle_cusparseDcsrsv2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDdense2csc, handle_cusparseDdense2csc, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDdense2csr, handle_cusparseDdense2csr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDestroyCsrgemm2Info, handle_cusparseDestroyCsrgemm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDestroyCsrsm2Info, handle_cusparseDestroyCsrsm2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDestroyCsrsv2Info, handle_cusparseDestroyCsrsv2Info, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDgemmi, handle_cusparseDgemmi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDgthr, handle_cusparseDgthr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDgthrz, handle_cusparseDgthrz, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDroti, handle_cusparseDroti, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseDsctr, handle_cusparseDsctr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSaxpyi, handle_cusparseSaxpyi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsc2dense, handle_cusparseScsc2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsr2dense, handle_cusparseScsr2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrgemm2, handle_cusparseScsrgemm2, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrgemm2_bufferSizeExt, handle_cusparseScsrgemm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrsm2_analysis, handle_cusparseScsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrsm2_bufferSizeExt, handle_cusparseScsrsm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrsm2_solve, handle_cusparseScsrsm2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrsv2_analysis, handle_cusparseScsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrsv2_bufferSize, handle_cusparseScsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseScsrsv2_solve, handle_cusparseScsrsv2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSdense2csc, handle_cusparseSdense2csc, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSdense2csr, handle_cusparseSdense2csr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSgemmi, handle_cusparseSgemmi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSgthr, handle_cusparseSgthr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSgthrz, handle_cusparseSgthrz, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSpMatSetStridedBatch, handle_cusparseSpMatSetStridedBatch, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSroti, handle_cusparseSroti, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseSsctr, handle_cusparseSsctr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseXcsrgemm2Nnz, handle_cusparseXcsrgemm2Nnz, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseXcsrsm2_zeroPivot, handle_cusparseXcsrsm2_zeroPivot, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseXcsrsv2_zeroPivot, handle_cusparseXcsrsv2_zeroPivot, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZaxpyi, handle_cusparseZaxpyi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsc2dense, handle_cusparseZcsc2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsr2dense, handle_cusparseZcsr2dense, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrgemm2, handle_cusparseZcsrgemm2, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrgemm2_bufferSizeExt, handle_cusparseZcsrgemm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrsm2_analysis, handle_cusparseZcsrsm2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrsm2_bufferSizeExt, handle_cusparseZcsrsm2_bufferSizeExt, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrsm2_solve, handle_cusparseZcsrsm2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrsv2_analysis, handle_cusparseZcsrsv2_analysis, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrsv2_bufferSize, handle_cusparseZcsrsv2_bufferSize, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZcsrsv2_solve, handle_cusparseZcsrsv2_solve, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZdense2csc, handle_cusparseZdense2csc, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZdense2csr, handle_cusparseZdense2csr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZgemmi, handle_cusparseZgemmi, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZgthr, handle_cusparseZgthr, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZgthrz, handle_cusparseZgthrz, rpc_backend::cusparse)
#endif
#if CUSPARSE_VERSION < 12000
      LUPINE_REGISTER_HANDLER(RPC_cusparseZsctr, handle_cusparseZsctr, rpc_backend::cusparse)
#endif
#endif
#ifdef LUPINE_BUILD_CUSOLVER_BACKEND
      LUPINE_CUSOLVER_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUSOLVER_VERSION >= 11500
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetDeterministicMode, handle_cusolverDnSetDeterministicMode, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11500
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetDeterministicMode, handle_cusolverDnGetDeterministicMode, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetMathMode, handle_cusolverDnSetMathMode, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetMathMode, handle_cusolverDnGetMathMode, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetEmulationStrategy, handle_cusolverDnSetEmulationStrategy, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12003
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetEmulationStrategy, handle_cusolverDnGetEmulationStrategy, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetFixedPointEmulationMantissaControl, handle_cusolverDnSetFixedPointEmulationMantissaControl, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetFixedPointEmulationMantissaControl, handle_cusolverDnGetFixedPointEmulationMantissaControl, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetFixedPointEmulationMaxMantissaBitCount, handle_cusolverDnSetFixedPointEmulationMaxMantissaBitCount, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetFixedPointEmulationMaxMantissaBitCount, handle_cusolverDnGetFixedPointEmulationMaxMantissaBitCount, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetFixedPointEmulationMantissaBitOffset, handle_cusolverDnSetFixedPointEmulationMantissaBitOffset, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetFixedPointEmulationMantissaBitOffset, handle_cusolverDnGetFixedPointEmulationMantissaBitOffset, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnSetEmulationSpecialValuesSupport, handle_cusolverDnSetEmulationSpecialValuesSupport, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnGetEmulationSpecialValuesSupport, handle_cusolverDnGetEmulationSpecialValuesSupport, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXsygvd_bufferSize, handle_cusolverDnXsygvd_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXsygvd, handle_cusolverDnXsygvd, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXsygvdx_bufferSize, handle_cusolverDnXsygvdx_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12100
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXsygvdx, handle_cusolverDnXsygvdx, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXstedc_bufferSize, handle_cusolverDnXstedc_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXstedc, handle_cusolverDnXstedc, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXsyevBatched_bufferSize, handle_cusolverDnXsyevBatched_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXsyevBatched, handle_cusolverDnXsyevBatched, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXgeev_bufferSize, handle_cusolverDnXgeev_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11701
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXgeev, handle_cusolverDnXgeev, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11600
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXlarft_bufferSize, handle_cusolverDnXlarft_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 11600
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXlarft, handle_cusolverDnXlarft, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXpolar_bufferSize, handle_cusolverDnXpolar_bufferSize, rpc_backend::cusolver)
#endif
#if CUSOLVER_VERSION >= 12200
      LUPINE_REGISTER_HANDLER(RPC_cusolverDnXpolar, handle_cusolverDnXpolar, rpc_backend::cusolver)
#endif
      LUPINE_CUSOLVERMG_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)

#endif
#ifdef LUPINE_BUILD_NVRTC_BACKEND
      LUPINE_NVRTC_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if CUDA_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetLTOIR, handle_nvrtcGetLTOIR, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetOptiXIR, handle_nvrtcGetOptiXIR, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetNVVM, handle_nvrtcGetNVVM, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13020
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetTileIR, handle_nvrtcGetTileIR, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13030
      LUPINE_REGISTER_HANDLER(RPC_nvrtcInstallBundledHeaders, handle_nvrtcInstallBundledHeaders, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13030
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetBundledHeadersInfo, handle_nvrtcGetBundledHeadersInfo, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13030
      LUPINE_REGISTER_HANDLER(RPC_nvrtcRemoveBundledHeaders, handle_nvrtcRemoveBundledHeaders, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetLTOIRSize, handle_nvrtcGetLTOIRSize, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12000
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetOptiXIRSize, handle_nvrtcGetOptiXIRSize, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION < 13000
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetNVVMSize, handle_nvrtcGetNVVMSize, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetPCHHeapSize, handle_nvrtcGetPCHHeapSize, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
      LUPINE_REGISTER_HANDLER(RPC_nvrtcSetPCHHeapSize, handle_nvrtcSetPCHHeapSize, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetPCHCreateStatus, handle_nvrtcGetPCHCreateStatus, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 12080
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetPCHHeapSizeRequired, handle_nvrtcGetPCHHeapSizeRequired, rpc_backend::nvrtc)
#endif
#if CUDA_VERSION >= 13020
      LUPINE_REGISTER_HANDLER(RPC_nvrtcGetTileIRSize, handle_nvrtcGetTileIRSize, rpc_backend::nvrtc)
#endif
#endif
#ifdef LUPINE_BUILD_NCCL_BACKEND
      LUPINE_NCCL_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetStr, handle_ncclParamGetStr, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetParameter, handle_ncclParamGetParameter, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetAllParameterKeys, handle_ncclParamGetAllParameterKeys, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
      LUPINE_REGISTER_HANDLER(RPC_ncclMemAlloc, handle_ncclMemAlloc, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
      LUPINE_REGISTER_HANDLER(RPC_ncclMemFree, handle_ncclMemFree, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22807
      LUPINE_REGISTER_HANDLER(RPC_ncclCommRevoke, handle_ncclCommRevoke, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21801
      LUPINE_REGISTER_HANDLER(RPC_ncclCommSplit, handle_ncclCommSplit, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22703
      LUPINE_REGISTER_HANDLER(RPC_ncclCommShrink, handle_ncclCommShrink, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
      LUPINE_REGISTER_HANDLER(RPC_ncclCommGetUniqueId, handle_ncclCommGetUniqueId, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
      LUPINE_REGISTER_HANDLER(RPC_ncclCommGrow, handle_ncclCommGrow, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22304
      LUPINE_REGISTER_HANDLER(RPC_ncclCommInitRankScalable, handle_ncclCommInitRankScalable, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22403
      LUPINE_REGISTER_HANDLER(RPC_ncclResetDebugInit, handle_ncclResetDebugInit, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
      LUPINE_REGISTER_HANDLER(RPC_ncclCommRegister, handle_ncclCommRegister, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 21903
      LUPINE_REGISTER_HANDLER(RPC_ncclCommDeregister, handle_ncclCommDeregister, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22907
      LUPINE_REGISTER_HANDLER(RPC_ncclCommSuspend, handle_ncclCommSuspend, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22907
      LUPINE_REGISTER_HANDLER(RPC_ncclCommResume, handle_ncclCommResume, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22907
      LUPINE_REGISTER_HANDLER(RPC_ncclCommMemStats, handle_ncclCommMemStats, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22703
      LUPINE_REGISTER_HANDLER(RPC_ncclCommWindowRegister, handle_ncclCommWindowRegister, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22703
      LUPINE_REGISTER_HANDLER(RPC_ncclCommWindowDeregister, handle_ncclCommWindowDeregister, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
      LUPINE_REGISTER_HANDLER(RPC_ncclWinGetUserPtr, handle_ncclWinGetUserPtr, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22803
      LUPINE_REGISTER_HANDLER(RPC_ncclAlltoAll, handle_ncclAlltoAll, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22803
      LUPINE_REGISTER_HANDLER(RPC_ncclGather, handle_ncclGather, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22803
      LUPINE_REGISTER_HANDLER(RPC_ncclScatter, handle_ncclScatter, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclAllReduceConfig, handle_ncclAllReduceConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclBroadcastConfig, handle_ncclBroadcastConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclReduceConfig, handle_ncclReduceConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclAllGatherConfig, handle_ncclAllGatherConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclReduceScatterConfig, handle_ncclReduceScatterConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclAlltoAllConfig, handle_ncclAlltoAllConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclGatherConfig, handle_ncclGatherConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23102
      LUPINE_REGISTER_HANDLER(RPC_ncclScatterConfig, handle_ncclScatterConfig, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
      LUPINE_REGISTER_HANDLER(RPC_ncclPutSignal, handle_ncclPutSignal, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
      LUPINE_REGISTER_HANDLER(RPC_ncclSignal, handle_ncclSignal, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22902
      LUPINE_REGISTER_HANDLER(RPC_ncclWaitSignal, handle_ncclWaitSignal, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 22203
      LUPINE_REGISTER_HANDLER(RPC_ncclGroupSimulateEnd, handle_ncclGroupSimulateEnd, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamBind, handle_ncclParamBind, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetI8, handle_ncclParamGetI8, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetI16, handle_ncclParamGetI16, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetI32, handle_ncclParamGetI32, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetI64, handle_ncclParamGetI64, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetU8, handle_ncclParamGetU8, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetU16, handle_ncclParamGetU16, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetU32, handle_ncclParamGetU32, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGetU64, handle_ncclParamGetU64, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamGet, handle_ncclParamGet, rpc_backend::nccl)
#endif
#if NCCL_VERSION_CODE >= 23007
      LUPINE_REGISTER_HANDLER(RPC_ncclParamDumpAll, handle_ncclParamDumpAll, rpc_backend::nccl)
#endif
#endif
#ifdef LUPINE_BUILD_NVJITLINK_BACKEND
      LUPINE_NVJITLINK_RPC_HANDLERS(LUPINE_REGISTER_HANDLER)
#if LUPINE_NVJITLINK_HAS_LTOIR
      LUPINE_REGISTER_HANDLER(RPC_nvJitLinkGetLinkedLTOIR, handle_nvJitLinkGetLinkedLTOIR, rpc_backend::nvjitlink)
#endif
#if LUPINE_NVJITLINK_HAS_LTOIR
      LUPINE_REGISTER_HANDLER(RPC_nvJitLinkGetLinkedLTOIRSize, handle_nvJitLinkGetLinkedLTOIRSize, rpc_backend::nvjitlink)
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
#undef LUPINE_CUDART_RPC_HANDLERS
#undef LUPINE_CUBLAS_RPC_HANDLERS
#undef LUPINE_CUBLASLT_RPC_HANDLERS
#undef LUPINE_CUFFT_RPC_HANDLERS
#undef LUPINE_CUDNN_RPC_HANDLERS
#undef LUPINE_CURAND_RPC_HANDLERS
#undef LUPINE_CUSPARSE_RPC_HANDLERS
#undef LUPINE_CUSOLVER_RPC_HANDLERS
#undef LUPINE_CUSOLVERMG_RPC_HANDLERS
#undef LUPINE_NVRTC_RPC_HANDLERS
#undef LUPINE_NCCL_RPC_HANDLERS
#undef LUPINE_NVJITLINK_RPC_HANDLERS
#undef LUPINE_NVML_RPC_HANDLERS
#undef LUPINE_HIP_RPC_HANDLERS
