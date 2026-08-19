#include "rpc_server.h"

#include "copy_pipeline.h"
#include "cuda_server.h"
#include "gen_api.h"
#include "nvml_server.h"

// clang-format off
#define LUPINE_RPC_HANDLERS(GENERATED, MANUAL) \
  MANUAL(RPC_cuGetErrorName, handle_manual_cuGetErrorName, lupine_cuda_backend, "cuGetErrorName") \
  MANUAL(RPC_cuGetErrorString, handle_manual_cuGetErrorString, lupine_cuda_backend, "cuGetErrorString") \
  MANUAL(LUPINE_RPC_cuGetExportTableMetadata, handle_manual_cuGetExportTableMetadata, lupine_cuda_backend, "cuGetExportTable metadata") \
  MANUAL(LUPINE_RPC_cuPrivateGetModuleNode, handle_manual_cuPrivateGetModuleNode, lupine_cuda_backend, "private module node") \
  MANUAL(RPC_cuModuleLoad, handle_manual_cuModuleLoad, lupine_cuda_backend, "cuModuleLoad") \
  MANUAL(RPC_cuModuleLoadData, handle_manual_cuModuleLoadData, lupine_cuda_backend, "cuModuleLoadData") \
  MANUAL(LUPINE_RPC_lupineFunctionParamLayoutSnapshot, handle_manual_lupineFunctionParamLayoutSnapshot, lupine_cuda_backend, "lupineFunctionParamLayoutSnapshot") \
  MANUAL(LUPINE_RPC_lupineFunctionAttributeSnapshot, handle_manual_lupineFunctionAttributeSnapshot, lupine_cuda_backend, "lupineFunctionAttributeSnapshot") \
  MANUAL(RPC_cuLibraryLoadData, handle_manual_cuLibraryLoadData, lupine_cuda_backend, "cuLibraryLoadData") \
  MANUAL(LUPINE_RPC_lupineLibrarySnapshot, handle_manual_lupineLibrarySnapshot, lupine_cuda_backend, "lupineLibrarySnapshot") \
  MANUAL(LUPINE_RPC_lupineLibraryAttributeSnapshot, handle_manual_lupineLibraryAttributeSnapshot, lupine_cuda_backend, "lupineLibraryAttributeSnapshot") \
  MANUAL(RPC_cuCtxCreate_v2, handle_manual_cuCtxCreate_v2, lupine_cuda_backend, "cuCtxCreate_v2") \
  MANUAL(RPC_cuDevicePrimaryCtxRetain, handle_manual_cuDevicePrimaryCtxRetain, lupine_cuda_backend, "cuDevicePrimaryCtxRetain") \
  MANUAL(RPC_cuDevicePrimaryCtxRelease_v2, handle_manual_cuDevicePrimaryCtxRelease_v2, lupine_cuda_backend, "cuDevicePrimaryCtxRelease_v2") \
  MANUAL(RPC_cuDevicePrimaryCtxReset_v2, handle_manual_cuDevicePrimaryCtxReset_v2, lupine_cuda_backend, "cuDevicePrimaryCtxReset_v2") \
  MANUAL(RPC_cuCtxAttach, handle_manual_cuCtxAttach, lupine_cuda_backend, "cuCtxAttach") \
  MANUAL(RPC_cuCtxDestroy_v2, handle_manual_cuCtxDestroy_v2, lupine_cuda_backend, "cuCtxDestroy_v2") \
  MANUAL(RPC_cuCtxDetach, handle_manual_cuCtxDetach, lupine_cuda_backend, "cuCtxDetach") \
  MANUAL(RPC_cuMemPoolSetAttribute, handle_manual_cuMemPoolSetAttribute, lupine_cuda_backend, "cuMemPoolSetAttribute") \
  MANUAL(RPC_cuMemPoolGetAttribute, handle_manual_cuMemPoolGetAttribute, lupine_cuda_backend, "cuMemPoolGetAttribute") \
  MANUAL(RPC_cuMemExportToShareableHandle, handle_manual_cuMemExportToShareableHandle, lupine_cuda_backend, "cuMemExportToShareableHandle") \
  MANUAL(RPC_cuMemImportFromShareableHandle, handle_manual_cuMemImportFromShareableHandle, lupine_cuda_backend, "cuMemImportFromShareableHandle") \
  MANUAL(RPC_cuMemPoolExportToShareableHandle, handle_manual_cuMemPoolExportToShareableHandle, lupine_cuda_backend, "cuMemPoolExportToShareableHandle") \
  MANUAL(RPC_cuMemPoolImportFromShareableHandle, handle_manual_cuMemPoolImportFromShareableHandle, lupine_cuda_backend, "cuMemPoolImportFromShareableHandle") \
  MANUAL(RPC_cuPointerGetAttribute, handle_manual_cuPointerGetAttribute, lupine_cuda_backend, "cuPointerGetAttribute") \
  MANUAL(RPC_cuPointerSetAttribute, handle_manual_cuPointerSetAttribute, lupine_cuda_backend, "cuPointerSetAttribute") \
  MANUAL(RPC_cuPointerGetAttributes, handle_manual_cuPointerGetAttributes, lupine_cuda_backend, "cuPointerGetAttributes") \
  MANUAL(RPC_cuLinkCreate_v2, handle_manual_cuLinkCreate_v2, lupine_cuda_backend, "cuLinkCreate_v2") \
  MANUAL(RPC_cuLinkAddData_v2, handle_manual_cuLinkAddData_v2, lupine_cuda_backend, "cuLinkAddData_v2") \
  MANUAL(RPC_cuLinkAddFile_v2, handle_manual_cuLinkAddFile_v2, lupine_cuda_backend, "cuLinkAddFile_v2") \
  MANUAL(RPC_cuLinkComplete, handle_manual_cuLinkComplete, lupine_cuda_backend, "cuLinkComplete") \
  MANUAL(RPC_cuLinkDestroy, handle_manual_cuLinkDestroy, lupine_cuda_backend, "cuLinkDestroy") \
  MANUAL(RPC_cuMemcpy3D_v2, handle_manual_cuMemcpy3D_v2, lupine_cuda_backend, "cuMemcpy3D_v2") \
  MANUAL(RPC_cuMemcpy2D_v2, handle_manual_cuMemcpy2D_v2, lupine_cuda_backend, "cuMemcpy2D_v2") \
  MANUAL(RPC_cuMemcpy2DUnaligned_v2, handle_manual_cuMemcpy2DUnaligned_v2, lupine_cuda_backend, "cuMemcpy2DUnaligned_v2") \
  MANUAL(RPC_cuMemcpy2DAsync_v2, handle_manual_cuMemcpy2DAsync_v2, lupine_cuda_backend, "cuMemcpy2DAsync_v2") \
  MANUAL(RPC_cuMemcpyDtoH_v2, handle_manual_cuMemcpyDtoH_v2, lupine_cuda_backend, "cuMemcpyDtoH_v2") \
  MANUAL(RPC_cuMemcpyAtoH_v2, handle_manual_cuMemcpyAtoH_v2, lupine_cuda_backend, "cuMemcpyAtoH_v2") \
  MANUAL(RPC_cuMemHostAlloc, handle_manual_cuMemHostAlloc, lupine_cuda_backend, "cuMemHostAlloc") \
  MANUAL(RPC_cuMemHostGetFlags, handle_manual_cuMemHostGetFlags, lupine_cuda_backend, "cuMemHostGetFlags") \
  MANUAL(RPC_cuDeviceGetGraphMemAttribute, handle_manual_cuDeviceGetGraphMemAttribute, lupine_cuda_backend, "cuDeviceGetGraphMemAttribute") \
  MANUAL(RPC_cuDeviceSetGraphMemAttribute, handle_manual_cuDeviceSetGraphMemAttribute, lupine_cuda_backend, "cuDeviceSetGraphMemAttribute") \
  MANUAL(RPC_cuLibraryGetModule, handle_manual_cuLibraryGetModule, lupine_cuda_backend, "cuLibraryGetModule") \
  MANUAL(RPC_cuLibraryUnload, handle_manual_cuLibraryUnload, lupine_cuda_backend, "cuLibraryUnload") \
  MANUAL(RPC_cuModuleGetGlobal_v2, handle_manual_cuModuleGetGlobal_v2, lupine_cuda_backend, "cuModuleGetGlobal_v2") \
  MANUAL(RPC_cuOccupancyMaxPotentialBlockSize, [](conn_t *conn) { return handle_manual_cuOccupancyMaxPotentialBlockSize(conn, false); }, lupine_cuda_backend, "cuOccupancyMaxPotentialBlockSize") \
  MANUAL(RPC_cuOccupancyMaxPotentialBlockSizeWithFlags, [](conn_t *conn) { return handle_manual_cuOccupancyMaxPotentialBlockSize(conn, true); }, lupine_cuda_backend, "cuOccupancyMaxPotentialBlockSizeWithFlags") \
  MANUAL(RPC_cuLaunchKernel, handle_manual_cuLaunchKernel, lupine_cuda_backend, "cuLaunchKernel") \
  MANUAL(RPC_cuLaunchKernelEx, handle_manual_cuLaunchKernelEx, lupine_cuda_backend, "cuLaunchKernelEx") \
  MANUAL(RPC_cuLaunchCooperativeKernel, handle_manual_cuLaunchCooperativeKernel, lupine_cuda_backend, "cuLaunchCooperativeKernel") \
  MANUAL(RPC_cuGraphAddKernelNode_v2, handle_manual_cuGraphAddKernelNode, lupine_cuda_backend, "cuGraphAddKernelNode") \
  MANUAL(RPC_cuGraphKernelNodeGetParams_v2, handle_manual_cuGraphKernelNodeGetParams, lupine_cuda_backend, "cuGraphKernelNodeGetParams_v2") \
  MANUAL(RPC_cuGraphKernelNodeSetParams_v2, handle_manual_cuGraphKernelNodeSetParams, lupine_cuda_backend, "cuGraphKernelNodeSetParams_v2") \
  MANUAL(RPC_cuGraphAddMemcpyNode, handle_manual_cuGraphAddMemcpyNode, lupine_cuda_backend, "cuGraphAddMemcpyNode") \
  MANUAL(RPC_cuGraphAddMemsetNode, handle_manual_cuGraphAddMemsetNode, lupine_cuda_backend, "cuGraphAddMemsetNode") \
  MANUAL(RPC_cuGraphAddHostNode, handle_manual_cuGraphAddHostNode, lupine_cuda_backend, "cuGraphAddHostNode") \
  MANUAL(RPC_cuGraphExecKernelNodeSetParams_v2, handle_manual_cuGraphExecKernelNodeSetParams, lupine_cuda_backend, "cuGraphExecKernelNodeSetParams_v2") \
  MANUAL(LUPINE_RPC_cuGraphConditionalHandleCreate, handle_manual_cuGraphConditionalHandleCreate, lupine_cuda_backend, "cuGraphConditionalHandleCreate") \
  MANUAL(LUPINE_RPC_cuGraphAddNode_v2, handle_manual_cuGraphAddNode, lupine_cuda_backend, "cuGraphAddNode") \
  MANUAL(RPC_cuGraphLaunch, handle_manual_cuGraphLaunch, lupine_cuda_backend, "cuGraphLaunch") \
  MANUAL(RPC_cuGraphGetEdges_v2, handle_manual_cuGraphGetEdges, lupine_cuda_backend, "cuGraphGetEdges") \
  MANUAL(RPC_cuGraphNodeGetDependencies_v2, handle_manual_cuGraphNodeGetDependencies, lupine_cuda_backend, "cuGraphNodeGetDependencies") \
  MANUAL(RPC_cuGraphNodeGetDependentNodes_v2, handle_manual_cuGraphNodeGetDependentNodes, lupine_cuda_backend, "cuGraphNodeGetDependentNodes") \
  MANUAL(LUPINE_RPC_cuMemPrefetchAsync, handle_manual_cuMemPrefetchAsync, lupine_cuda_backend, "cuMemPrefetchAsync") \
  MANUAL(RPC_cuGraphHostNodeGetParams, handle_manual_cuGraphHostNodeGetParams, lupine_cuda_backend, "cuGraphHostNodeGetParams") \
  MANUAL(RPC_cuGraphHostNodeSetParams, handle_manual_cuGraphHostNodeSetParams, lupine_cuda_backend, "cuGraphHostNodeSetParams") \
  MANUAL(RPC_cuGraphExecHostNodeSetParams, handle_manual_cuGraphExecHostNodeSetParams, lupine_cuda_backend, "cuGraphExecHostNodeSetParams") \
  MANUAL(RPC_cuLaunchHostFunc, handle_manual_cuLaunchHostFunc, lupine_cuda_backend, "cuLaunchHostFunc") \
  MANUAL(RPC_cuStreamAddCallback, handle_manual_cuStreamAddCallback, lupine_cuda_backend, "cuStreamAddCallback") \
  MANUAL(RPC_cuEventRecord, [](conn_t *conn) { return handle_manual_cuEventRecord(conn, false); }, lupine_cuda_backend, "cuEventRecord") \
  MANUAL(RPC_cuEventRecordWithFlags, [](conn_t *conn) { return handle_manual_cuEventRecord(conn, true); }, lupine_cuda_backend, "cuEventRecordWithFlags") \
  MANUAL(RPC_cuEventQuery, handle_manual_cuEventQuery, lupine_cuda_backend, "cuEventQuery") \
  MANUAL(LUPINE_RPC_lupineEventQueryBatch, handle_manual_lupineEventQueryBatch, lupine_cuda_backend, "lupineEventQueryBatch") \
  MANUAL(RPC_cuStreamWaitEvent, handle_manual_cuStreamWaitEvent, lupine_cuda_backend, "cuStreamWaitEvent") \
  MANUAL(LUPINE_RPC_cuStreamBeginCaptureToGraph, handle_manual_cuStreamBeginCaptureToGraph, lupine_cuda_backend, "cuStreamBeginCaptureToGraph") \
  MANUAL(RPC_cuStreamUpdateCaptureDependencies_v2, handle_manual_cuStreamUpdateCaptureDependencies, lupine_cuda_backend, "cuStreamUpdateCaptureDependencies") \
  MANUAL(LUPINE_RPC_cuStreamGetCaptureInfo_v3, handle_manual_cuStreamGetCaptureInfo, lupine_cuda_backend, "cuStreamGetCaptureInfo") \
  MANUAL(RPC_cuStreamBeginCapture_v2, handle_manual_cuStreamBeginCapture, lupine_cuda_backend, "cuStreamBeginCapture") \
  MANUAL(RPC_cuStreamEndCapture, handle_manual_cuStreamEndCapture, lupine_cuda_backend, "cuStreamEndCapture") \
  MANUAL(RPC_cuGraphClone, handle_manual_cuGraphClone, lupine_cuda_backend, "cuGraphClone") \
  MANUAL(RPC_cuGraphInstantiateWithFlags, handle_manual_cuGraphInstantiateWithFlags, lupine_cuda_backend, "cuGraphInstantiateWithFlags") \
  MANUAL(RPC_cuGraphInstantiateWithParams, handle_manual_cuGraphInstantiateWithParams, lupine_cuda_backend, "cuGraphInstantiateWithParams") \
  MANUAL(RPC_cuGraphExecDestroy, handle_manual_cuGraphExecDestroy, lupine_cuda_backend, "cuGraphExecDestroy") \
  MANUAL(RPC_cuGraphDestroy, handle_manual_cuGraphDestroy, lupine_cuda_backend, "cuGraphDestroy") \
  MANUAL(RPC_cuMemcpyHtoD_v2, handle_manual_cuMemcpyHtoD_v2, lupine_cuda_backend, "cuMemcpyHtoD_v2") \
  MANUAL(RPC_cuMemcpyHtoDAsync_v2, handle_manual_cuMemcpyHtoDAsync_v2, lupine_cuda_backend, "cuMemcpyHtoDAsync_v2") \
  MANUAL(LUPINE_RPC_lupineManagedHostFlush, handle_manual_lupineManagedHostFlush, lupine_cuda_backend, "lupineManagedHostFlush") \
  MANUAL(LUPINE_RPC_lupineDeviceSnapshot, handle_manual_lupineDeviceSnapshot, lupine_cuda_backend, "lupineDeviceSnapshot") \
  MANUAL(RPC_cuMemcpyDtoHAsync_v2, handle_manual_cuMemcpyDtoHAsync_v2, lupine_cuda_backend, "cuMemcpyDtoHAsync_v2") \
  MANUAL(RPC_cuCtxSynchronize, handle_manual_cuCtxSynchronize, lupine_cuda_backend, "cuCtxSynchronize") \
  MANUAL(RPC_cuStreamSynchronize, handle_manual_cuStreamSynchronize, lupine_cuda_backend, "cuStreamSynchronize") \
  MANUAL(RPC_cuEventSynchronize, handle_manual_cuEventSynchronize, lupine_cuda_backend, "cuEventSynchronize") \
  GENERATED(RPC_cuInit, handle_cuInit, lupine_cuda_backend) \
  GENERATED(RPC_cuDriverGetVersion, handle_cuDriverGetVersion, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGet, handle_cuDeviceGet, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetCount, handle_cuDeviceGetCount, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetName, handle_cuDeviceGetName, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetUuid_v2, handle_cuDeviceGetUuid_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetLuid, handle_cuDeviceGetLuid, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceTotalMem_v2, handle_cuDeviceTotalMem_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetTexture1DLinearMaxWidth, handle_cuDeviceGetTexture1DLinearMaxWidth, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetAttribute, handle_cuDeviceGetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceSetMemPool, handle_cuDeviceSetMemPool, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetMemPool, handle_cuDeviceGetMemPool, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetDefaultMemPool, handle_cuDeviceGetDefaultMemPool, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetExecAffinitySupport, handle_cuDeviceGetExecAffinitySupport, lupine_cuda_backend) \
  GENERATED(RPC_cuFlushGPUDirectRDMAWrites, handle_cuFlushGPUDirectRDMAWrites, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetProperties, handle_cuDeviceGetProperties, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceComputeCapability, handle_cuDeviceComputeCapability, lupine_cuda_backend) \
  GENERATED(RPC_cuDevicePrimaryCtxSetFlags_v2, handle_cuDevicePrimaryCtxSetFlags_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuDevicePrimaryCtxGetState, handle_cuDevicePrimaryCtxGetState, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxPushCurrent_v2, handle_cuCtxPushCurrent_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxPopCurrent_v2, handle_cuCtxPopCurrent_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxSetCurrent, handle_cuCtxSetCurrent, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetCurrent, handle_cuCtxGetCurrent, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetDevice, handle_cuCtxGetDevice, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetFlags, handle_cuCtxGetFlags, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetId, handle_cuCtxGetId, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxSetLimit, handle_cuCtxSetLimit, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetLimit, handle_cuCtxGetLimit, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetCacheConfig, handle_cuCtxGetCacheConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxSetCacheConfig, handle_cuCtxSetCacheConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetApiVersion, handle_cuCtxGetApiVersion, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetStreamPriorityRange, handle_cuCtxGetStreamPriorityRange, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxResetPersistingL2Cache, handle_cuCtxResetPersistingL2Cache, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetExecAffinity, handle_cuCtxGetExecAffinity, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxGetSharedMemConfig, handle_cuCtxGetSharedMemConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxSetSharedMemConfig, handle_cuCtxSetSharedMemConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuModuleUnload, handle_cuModuleUnload, lupine_cuda_backend) \
  GENERATED(RPC_cuModuleGetLoadingMode, handle_cuModuleGetLoadingMode, lupine_cuda_backend) \
  GENERATED(RPC_cuModuleGetFunction, handle_cuModuleGetFunction, lupine_cuda_backend) \
  GENERATED(RPC_cuModuleGetTexRef, handle_cuModuleGetTexRef, lupine_cuda_backend) \
  GENERATED(RPC_cuModuleGetSurfRef, handle_cuModuleGetSurfRef, lupine_cuda_backend) \
  GENERATED(RPC_cuLibraryLoadFromFile, handle_cuLibraryLoadFromFile, lupine_cuda_backend) \
  GENERATED(RPC_cuLibraryGetKernel, handle_cuLibraryGetKernel, lupine_cuda_backend) \
  GENERATED(RPC_cuKernelGetFunction, handle_cuKernelGetFunction, lupine_cuda_backend) \
  GENERATED(RPC_cuKernelGetLibrary, handle_cuKernelGetLibrary, lupine_cuda_backend) \
  GENERATED(RPC_cuLibraryGetGlobal, handle_cuLibraryGetGlobal, lupine_cuda_backend) \
  GENERATED(RPC_cuLibraryGetManaged, handle_cuLibraryGetManaged, lupine_cuda_backend) \
  GENERATED(RPC_cuLibraryGetUnifiedFunction, handle_cuLibraryGetUnifiedFunction, lupine_cuda_backend) \
  GENERATED(RPC_cuKernelGetAttribute, handle_cuKernelGetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuKernelSetAttribute, handle_cuKernelSetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuKernelSetCacheConfig, handle_cuKernelSetCacheConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuKernelGetParamInfo, handle_cuKernelGetParamInfo, lupine_cuda_backend) \
  GENERATED(RPC_cuMemGetInfo_v2, handle_cuMemGetInfo_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAlloc_v2, handle_cuMemAlloc_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAllocPitch_v2, handle_cuMemAllocPitch_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemFree_v2, handle_cuMemFree_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemGetAddressRange_v2, handle_cuMemGetAddressRange_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAllocHost_v2, handle_cuMemAllocHost_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemFreeHost, handle_cuMemFreeHost, lupine_cuda_backend) \
  GENERATED(RPC_cuMemHostGetDevicePointer_v2, handle_cuMemHostGetDevicePointer_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAllocManaged, handle_cuMemAllocManaged, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetByPCIBusId, handle_cuDeviceGetByPCIBusId, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetPCIBusId, handle_cuDeviceGetPCIBusId, lupine_cuda_backend) \
  GENERATED(RPC_cuIpcGetEventHandle, handle_cuIpcGetEventHandle, lupine_cuda_backend) \
  GENERATED(RPC_cuIpcOpenEventHandle, handle_cuIpcOpenEventHandle, lupine_cuda_backend) \
  GENERATED(RPC_cuIpcGetMemHandle, handle_cuIpcGetMemHandle, lupine_cuda_backend) \
  GENERATED(RPC_cuIpcOpenMemHandle_v2, handle_cuIpcOpenMemHandle_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuIpcCloseMemHandle, handle_cuIpcCloseMemHandle, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpy, handle_cuMemcpy, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyPeer, handle_cuMemcpyPeer, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyDtoD_v2, handle_cuMemcpyDtoD_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyDtoA_v2, handle_cuMemcpyDtoA_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyAtoD_v2, handle_cuMemcpyAtoD_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyAtoA_v2, handle_cuMemcpyAtoA_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyPeerAsync, handle_cuMemcpyPeerAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuMemcpyDtoDAsync_v2, handle_cuMemcpyDtoDAsync_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD8_v2, handle_cuMemsetD8_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD16_v2, handle_cuMemsetD16_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD32_v2, handle_cuMemsetD32_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD2D8_v2, handle_cuMemsetD2D8_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD2D16_v2, handle_cuMemsetD2D16_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD2D32_v2, handle_cuMemsetD2D32_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD8Async, handle_cuMemsetD8Async, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD16Async, handle_cuMemsetD16Async, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD32Async, handle_cuMemsetD32Async, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD2D8Async, handle_cuMemsetD2D8Async, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD2D16Async, handle_cuMemsetD2D16Async, lupine_cuda_backend) \
  GENERATED(RPC_cuMemsetD2D32Async, handle_cuMemsetD2D32Async, lupine_cuda_backend) \
  GENERATED(RPC_cuArrayCreate_v2, handle_cuArrayCreate_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuArrayGetDescriptor_v2, handle_cuArrayGetDescriptor_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuArrayGetSparseProperties, handle_cuArrayGetSparseProperties, lupine_cuda_backend) \
  GENERATED(RPC_cuMipmappedArrayGetSparseProperties, handle_cuMipmappedArrayGetSparseProperties, lupine_cuda_backend) \
  GENERATED(RPC_cuArrayGetMemoryRequirements, handle_cuArrayGetMemoryRequirements, lupine_cuda_backend) \
  GENERATED(RPC_cuMipmappedArrayGetMemoryRequirements, handle_cuMipmappedArrayGetMemoryRequirements, lupine_cuda_backend) \
  GENERATED(RPC_cuArrayGetPlane, handle_cuArrayGetPlane, lupine_cuda_backend) \
  GENERATED(RPC_cuArrayDestroy, handle_cuArrayDestroy, lupine_cuda_backend) \
  GENERATED(RPC_cuArray3DCreate_v2, handle_cuArray3DCreate_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuArray3DGetDescriptor_v2, handle_cuArray3DGetDescriptor_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuMipmappedArrayCreate, handle_cuMipmappedArrayCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuMipmappedArrayGetLevel, handle_cuMipmappedArrayGetLevel, lupine_cuda_backend) \
  GENERATED(RPC_cuMipmappedArrayDestroy, handle_cuMipmappedArrayDestroy, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAddressReserve, handle_cuMemAddressReserve, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAddressFree, handle_cuMemAddressFree, lupine_cuda_backend) \
  GENERATED(RPC_cuMemCreate, handle_cuMemCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuMemRelease, handle_cuMemRelease, lupine_cuda_backend) \
  GENERATED(RPC_cuMemMap, handle_cuMemMap, lupine_cuda_backend) \
  GENERATED(RPC_cuMemMapArrayAsync, handle_cuMemMapArrayAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuMemUnmap, handle_cuMemUnmap, lupine_cuda_backend) \
  GENERATED(RPC_cuMemSetAccess, handle_cuMemSetAccess, lupine_cuda_backend) \
  GENERATED(RPC_cuMemGetAccess, handle_cuMemGetAccess, lupine_cuda_backend) \
  GENERATED(RPC_cuMemGetAllocationGranularity, handle_cuMemGetAllocationGranularity, lupine_cuda_backend) \
  GENERATED(RPC_cuMemGetAllocationPropertiesFromHandle, handle_cuMemGetAllocationPropertiesFromHandle, lupine_cuda_backend) \
  GENERATED(RPC_cuMemFreeAsync, handle_cuMemFreeAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAllocAsync, handle_cuMemAllocAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolTrimTo, handle_cuMemPoolTrimTo, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolSetAccess, handle_cuMemPoolSetAccess, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolGetAccess, handle_cuMemPoolGetAccess, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolCreate, handle_cuMemPoolCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolDestroy, handle_cuMemPoolDestroy, lupine_cuda_backend) \
  GENERATED(RPC_cuMemAllocFromPoolAsync, handle_cuMemAllocFromPoolAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolExportPointer, handle_cuMemPoolExportPointer, lupine_cuda_backend) \
  GENERATED(RPC_cuMemPoolImportPointer, handle_cuMemPoolImportPointer, lupine_cuda_backend) \
  GENERATED(RPC_cuMemRangeGetAttributes, handle_cuMemRangeGetAttributes, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamCreate, handle_cuStreamCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamCreateWithPriority, handle_cuStreamCreateWithPriority, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamGetPriority, handle_cuStreamGetPriority, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamGetFlags, handle_cuStreamGetFlags, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamGetId, handle_cuStreamGetId, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamGetCtx, handle_cuStreamGetCtx, lupine_cuda_backend) \
  GENERATED(RPC_cuThreadExchangeStreamCaptureMode, handle_cuThreadExchangeStreamCaptureMode, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamIsCapturing, handle_cuStreamIsCapturing, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamAttachMemAsync, handle_cuStreamAttachMemAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamQuery, handle_cuStreamQuery, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamDestroy_v2, handle_cuStreamDestroy_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamCopyAttributes, handle_cuStreamCopyAttributes, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamGetAttribute, handle_cuStreamGetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamSetAttribute, handle_cuStreamSetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuEventCreate, handle_cuEventCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuEventDestroy_v2, handle_cuEventDestroy_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuEventElapsedTime_v2, handle_cuEventElapsedTime_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuImportExternalMemory, handle_cuImportExternalMemory, lupine_cuda_backend) \
  GENERATED(RPC_cuExternalMemoryGetMappedBuffer, handle_cuExternalMemoryGetMappedBuffer, lupine_cuda_backend) \
  GENERATED(RPC_cuExternalMemoryGetMappedMipmappedArray, handle_cuExternalMemoryGetMappedMipmappedArray, lupine_cuda_backend) \
  GENERATED(RPC_cuDestroyExternalMemory, handle_cuDestroyExternalMemory, lupine_cuda_backend) \
  GENERATED(RPC_cuImportExternalSemaphore, handle_cuImportExternalSemaphore, lupine_cuda_backend) \
  GENERATED(RPC_cuSignalExternalSemaphoresAsync, handle_cuSignalExternalSemaphoresAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuWaitExternalSemaphoresAsync, handle_cuWaitExternalSemaphoresAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuDestroyExternalSemaphore, handle_cuDestroyExternalSemaphore, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamWaitValue32_v2, handle_cuStreamWaitValue32_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamWaitValue64_v2, handle_cuStreamWaitValue64_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamWriteValue32_v2, handle_cuStreamWriteValue32_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamWriteValue64_v2, handle_cuStreamWriteValue64_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuStreamBatchMemOp_v2, handle_cuStreamBatchMemOp_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncGetAttribute, handle_cuFuncGetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncSetAttribute, handle_cuFuncSetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncSetCacheConfig, handle_cuFuncSetCacheConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncGetModule, handle_cuFuncGetModule, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncGetParamInfo, handle_cuFuncGetParamInfo, lupine_cuda_backend) \
  GENERATED(RPC_cuLaunchCooperativeKernelMultiDevice, handle_cuLaunchCooperativeKernelMultiDevice, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncSetBlockShape, handle_cuFuncSetBlockShape, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncSetSharedSize, handle_cuFuncSetSharedSize, lupine_cuda_backend) \
  GENERATED(RPC_cuParamSetSize, handle_cuParamSetSize, lupine_cuda_backend) \
  GENERATED(RPC_cuParamSeti, handle_cuParamSeti, lupine_cuda_backend) \
  GENERATED(RPC_cuParamSetf, handle_cuParamSetf, lupine_cuda_backend) \
  GENERATED(RPC_cuLaunch, handle_cuLaunch, lupine_cuda_backend) \
  GENERATED(RPC_cuLaunchGrid, handle_cuLaunchGrid, lupine_cuda_backend) \
  GENERATED(RPC_cuLaunchGridAsync, handle_cuLaunchGridAsync, lupine_cuda_backend) \
  GENERATED(RPC_cuParamSetTexRef, handle_cuParamSetTexRef, lupine_cuda_backend) \
  GENERATED(RPC_cuFuncSetSharedMemConfig, handle_cuFuncSetSharedMemConfig, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphCreate, handle_cuGraphCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphMemcpyNodeGetParams, handle_cuGraphMemcpyNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphMemcpyNodeSetParams, handle_cuGraphMemcpyNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphMemsetNodeGetParams, handle_cuGraphMemsetNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphMemsetNodeSetParams, handle_cuGraphMemsetNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddChildGraphNode, handle_cuGraphAddChildGraphNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphChildGraphNodeGetGraph, handle_cuGraphChildGraphNodeGetGraph, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddEmptyNode, handle_cuGraphAddEmptyNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddEventRecordNode, handle_cuGraphAddEventRecordNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphEventRecordNodeGetEvent, handle_cuGraphEventRecordNodeGetEvent, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphEventRecordNodeSetEvent, handle_cuGraphEventRecordNodeSetEvent, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddEventWaitNode, handle_cuGraphAddEventWaitNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphEventWaitNodeGetEvent, handle_cuGraphEventWaitNodeGetEvent, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphEventWaitNodeSetEvent, handle_cuGraphEventWaitNodeSetEvent, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddExternalSemaphoresSignalNode, handle_cuGraphAddExternalSemaphoresSignalNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExternalSemaphoresSignalNodeGetParams, handle_cuGraphExternalSemaphoresSignalNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExternalSemaphoresSignalNodeSetParams, handle_cuGraphExternalSemaphoresSignalNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddExternalSemaphoresWaitNode, handle_cuGraphAddExternalSemaphoresWaitNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExternalSemaphoresWaitNodeGetParams, handle_cuGraphExternalSemaphoresWaitNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExternalSemaphoresWaitNodeSetParams, handle_cuGraphExternalSemaphoresWaitNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddBatchMemOpNode, handle_cuGraphAddBatchMemOpNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphBatchMemOpNodeGetParams, handle_cuGraphBatchMemOpNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphBatchMemOpNodeSetParams, handle_cuGraphBatchMemOpNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecBatchMemOpNodeSetParams, handle_cuGraphExecBatchMemOpNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddMemAllocNode, handle_cuGraphAddMemAllocNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphMemAllocNodeGetParams, handle_cuGraphMemAllocNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphAddMemFreeNode, handle_cuGraphAddMemFreeNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphMemFreeNodeGetParams, handle_cuGraphMemFreeNodeGetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGraphMemTrim, handle_cuDeviceGraphMemTrim, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphNodeFindInClone, handle_cuGraphNodeFindInClone, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphNodeGetType, handle_cuGraphNodeGetType, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphGetNodes, handle_cuGraphGetNodes, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphGetRootNodes, handle_cuGraphGetRootNodes, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphDestroyNode, handle_cuGraphDestroyNode, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecGetFlags, handle_cuGraphExecGetFlags, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecMemcpyNodeSetParams, handle_cuGraphExecMemcpyNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecMemsetNodeSetParams, handle_cuGraphExecMemsetNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecChildGraphNodeSetParams, handle_cuGraphExecChildGraphNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecEventRecordNodeSetEvent, handle_cuGraphExecEventRecordNodeSetEvent, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecEventWaitNodeSetEvent, handle_cuGraphExecEventWaitNodeSetEvent, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams, handle_cuGraphExecExternalSemaphoresSignalNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams, handle_cuGraphExecExternalSemaphoresWaitNodeSetParams, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphNodeSetEnabled, handle_cuGraphNodeSetEnabled, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphNodeGetEnabled, handle_cuGraphNodeGetEnabled, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphUpload, handle_cuGraphUpload, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphExecUpdate_v2, handle_cuGraphExecUpdate_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphKernelNodeCopyAttributes, handle_cuGraphKernelNodeCopyAttributes, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphKernelNodeGetAttribute, handle_cuGraphKernelNodeGetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphKernelNodeSetAttribute, handle_cuGraphKernelNodeSetAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphDebugDotPrint, handle_cuGraphDebugDotPrint, lupine_cuda_backend) \
  GENERATED(RPC_cuUserObjectRetain, handle_cuUserObjectRetain, lupine_cuda_backend) \
  GENERATED(RPC_cuUserObjectRelease, handle_cuUserObjectRelease, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphRetainUserObject, handle_cuGraphRetainUserObject, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphReleaseUserObject, handle_cuGraphReleaseUserObject, lupine_cuda_backend) \
  GENERATED(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor, handle_cuOccupancyMaxActiveBlocksPerMultiprocessor, lupine_cuda_backend) \
  GENERATED(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, lupine_cuda_backend) \
  GENERATED(RPC_cuOccupancyAvailableDynamicSMemPerBlock, handle_cuOccupancyAvailableDynamicSMemPerBlock, lupine_cuda_backend) \
  GENERATED(RPC_cuOccupancyMaxPotentialClusterSize, handle_cuOccupancyMaxPotentialClusterSize, lupine_cuda_backend) \
  GENERATED(RPC_cuOccupancyMaxActiveClusters, handle_cuOccupancyMaxActiveClusters, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetArray, handle_cuTexRefSetArray, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetMipmappedArray, handle_cuTexRefSetMipmappedArray, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetAddress_v2, handle_cuTexRefSetAddress_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetAddress2D_v3, handle_cuTexRefSetAddress2D_v3, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetFormat, handle_cuTexRefSetFormat, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetAddressMode, handle_cuTexRefSetAddressMode, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetFilterMode, handle_cuTexRefSetFilterMode, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetMipmapFilterMode, handle_cuTexRefSetMipmapFilterMode, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetMipmapLevelBias, handle_cuTexRefSetMipmapLevelBias, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetMipmapLevelClamp, handle_cuTexRefSetMipmapLevelClamp, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetMaxAnisotropy, handle_cuTexRefSetMaxAnisotropy, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetBorderColor, handle_cuTexRefSetBorderColor, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefSetFlags, handle_cuTexRefSetFlags, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetAddress_v2, handle_cuTexRefGetAddress_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetArray, handle_cuTexRefGetArray, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetMipmappedArray, handle_cuTexRefGetMipmappedArray, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetAddressMode, handle_cuTexRefGetAddressMode, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetFilterMode, handle_cuTexRefGetFilterMode, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetFormat, handle_cuTexRefGetFormat, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetMipmapFilterMode, handle_cuTexRefGetMipmapFilterMode, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetMipmapLevelBias, handle_cuTexRefGetMipmapLevelBias, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetMipmapLevelClamp, handle_cuTexRefGetMipmapLevelClamp, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetMaxAnisotropy, handle_cuTexRefGetMaxAnisotropy, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetBorderColor, handle_cuTexRefGetBorderColor, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefGetFlags, handle_cuTexRefGetFlags, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefCreate, handle_cuTexRefCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuTexRefDestroy, handle_cuTexRefDestroy, lupine_cuda_backend) \
  GENERATED(RPC_cuSurfRefSetArray, handle_cuSurfRefSetArray, lupine_cuda_backend) \
  GENERATED(RPC_cuSurfRefGetArray, handle_cuSurfRefGetArray, lupine_cuda_backend) \
  GENERATED(RPC_cuTexObjectCreate, handle_cuTexObjectCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuTexObjectDestroy, handle_cuTexObjectDestroy, lupine_cuda_backend) \
  GENERATED(RPC_cuTexObjectGetResourceDesc, handle_cuTexObjectGetResourceDesc, lupine_cuda_backend) \
  GENERATED(RPC_cuTexObjectGetTextureDesc, handle_cuTexObjectGetTextureDesc, lupine_cuda_backend) \
  GENERATED(RPC_cuTexObjectGetResourceViewDesc, handle_cuTexObjectGetResourceViewDesc, lupine_cuda_backend) \
  GENERATED(RPC_cuSurfObjectCreate, handle_cuSurfObjectCreate, lupine_cuda_backend) \
  GENERATED(RPC_cuSurfObjectDestroy, handle_cuSurfObjectDestroy, lupine_cuda_backend) \
  GENERATED(RPC_cuSurfObjectGetResourceDesc, handle_cuSurfObjectGetResourceDesc, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceCanAccessPeer, handle_cuDeviceCanAccessPeer, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxEnablePeerAccess, handle_cuCtxEnablePeerAccess, lupine_cuda_backend) \
  GENERATED(RPC_cuCtxDisablePeerAccess, handle_cuCtxDisablePeerAccess, lupine_cuda_backend) \
  GENERATED(RPC_cuDeviceGetP2PAttribute, handle_cuDeviceGetP2PAttribute, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsUnregisterResource, handle_cuGraphicsUnregisterResource, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsSubResourceGetMappedArray, handle_cuGraphicsSubResourceGetMappedArray, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsResourceGetMappedMipmappedArray, handle_cuGraphicsResourceGetMappedMipmappedArray, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsResourceGetMappedPointer_v2, handle_cuGraphicsResourceGetMappedPointer_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsResourceSetMapFlags_v2, handle_cuGraphicsResourceSetMapFlags_v2, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsMapResources, handle_cuGraphicsMapResources, lupine_cuda_backend) \
  GENERATED(RPC_cuGraphicsUnmapResources, handle_cuGraphicsUnmapResources, lupine_cuda_backend) \
  GENERATED(RPC_nvmlInit_v2, handle_nvmlInit_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlInitWithFlags, handle_nvmlInitWithFlags, lupine_nvml_backend) \
  GENERATED(RPC_nvmlShutdown, handle_nvmlShutdown, lupine_nvml_backend) \
  GENERATED(RPC_nvmlSystemGetDriverVersion, handle_nvmlSystemGetDriverVersion, lupine_nvml_backend) \
  GENERATED(RPC_nvmlSystemGetNVMLVersion, handle_nvmlSystemGetNVMLVersion, lupine_nvml_backend) \
  GENERATED(RPC_nvmlSystemGetCudaDriverVersion, handle_nvmlSystemGetCudaDriverVersion, lupine_nvml_backend) \
  GENERATED(RPC_nvmlSystemGetCudaDriverVersion_v2, handle_nvmlSystemGetCudaDriverVersion_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetCount_v2, handle_nvmlDeviceGetCount_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetHandleByIndex_v2, handle_nvmlDeviceGetHandleByIndex_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetHandleByUUID, handle_nvmlDeviceGetHandleByUUID, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetHandleByPciBusId_v2, handle_nvmlDeviceGetHandleByPciBusId_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetName, handle_nvmlDeviceGetName, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetUUID, handle_nvmlDeviceGetUUID, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetIndex, handle_nvmlDeviceGetIndex, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMinorNumber, handle_nvmlDeviceGetMinorNumber, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPciInfo_v3, handle_nvmlDeviceGetPciInfo_v3, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMemoryInfo, handle_nvmlDeviceGetMemoryInfo, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetUtilizationRates, handle_nvmlDeviceGetUtilizationRates, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetTemperature, handle_nvmlDeviceGetTemperature, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPowerUsage, handle_nvmlDeviceGetPowerUsage, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPowerManagementLimit, handle_nvmlDeviceGetPowerManagementLimit, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetClockInfo, handle_nvmlDeviceGetClockInfo, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMaxClockInfo, handle_nvmlDeviceGetMaxClockInfo, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPerformanceState, handle_nvmlDeviceGetPerformanceState, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetComputeMode, handle_nvmlDeviceGetComputeMode, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPersistenceMode, handle_nvmlDeviceGetPersistenceMode, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetFanSpeed, handle_nvmlDeviceGetFanSpeed, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetBrand, handle_nvmlDeviceGetBrand, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetVbiosVersion, handle_nvmlDeviceGetVbiosVersion, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetSerial, handle_nvmlDeviceGetSerial, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetBoardPartNumber, handle_nvmlDeviceGetBoardPartNumber, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetDisplayMode, handle_nvmlDeviceGetDisplayMode, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetDisplayActive, handle_nvmlDeviceGetDisplayActive, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetCurrPcieLinkGeneration, handle_nvmlDeviceGetCurrPcieLinkGeneration, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetCurrPcieLinkWidth, handle_nvmlDeviceGetCurrPcieLinkWidth, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMaxPcieLinkGeneration, handle_nvmlDeviceGetMaxPcieLinkGeneration, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMaxPcieLinkWidth, handle_nvmlDeviceGetMaxPcieLinkWidth, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPcieThroughput, handle_nvmlDeviceGetPcieThroughput, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetPcieReplayCounter, handle_nvmlDeviceGetPcieReplayCounter, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetComputeRunningProcesses, handle_nvmlDeviceGetComputeRunningProcesses, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetComputeRunningProcesses_v2, handle_nvmlDeviceGetComputeRunningProcesses_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetGraphicsRunningProcesses, handle_nvmlDeviceGetGraphicsRunningProcesses, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetGraphicsRunningProcesses_v2, handle_nvmlDeviceGetGraphicsRunningProcesses_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMPSComputeRunningProcesses, handle_nvmlDeviceGetMPSComputeRunningProcesses, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2, handle_nvmlDeviceGetMPSComputeRunningProcesses_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlEventSetCreate, handle_nvmlEventSetCreate, lupine_nvml_backend) \
  GENERATED(RPC_nvmlEventSetFree, handle_nvmlEventSetFree, lupine_nvml_backend) \
  GENERATED(RPC_nvmlEventSetWait_v2, handle_nvmlEventSetWait_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceRegisterEvents, handle_nvmlDeviceRegisterEvents, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMaxMigDeviceCount, handle_nvmlDeviceGetMaxMigDeviceCount, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetTotalEccErrors, handle_nvmlDeviceGetTotalEccErrors, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetDetailedEccErrors, handle_nvmlDeviceGetDetailedEccErrors, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMemoryErrorCounter, handle_nvmlDeviceGetMemoryErrorCounter, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetEccMode, handle_nvmlDeviceGetEccMode, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetTemperatureV, handle_nvmlDeviceGetTemperatureV, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetEnforcedPowerLimit, handle_nvmlDeviceGetEnforcedPowerLimit, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMemoryInfo_v2, handle_nvmlDeviceGetMemoryInfo_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetMigMode, handle_nvmlDeviceGetMigMode, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetVirtualizationMode, handle_nvmlDeviceGetVirtualizationMode, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceIsMigDeviceHandle, handle_nvmlDeviceIsMigDeviceHandle, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetNvLinkRemoteDeviceType, handle_nvmlDeviceGetNvLinkRemoteDeviceType, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2, handle_nvmlDeviceGetNvLinkRemotePciInfo_v2, lupine_nvml_backend) \
  GENERATED(RPC_nvmlDeviceGetCudaComputeCapability, handle_nvmlDeviceGetCudaComputeCapability, lupine_nvml_backend)
// clang-format on

#define LUPINE_DECLARE_GENERATED(operation, handler, backend)                  \
  int handler(conn_t *conn);
#define LUPINE_DECLARE_MANUAL(operation, handler, backend, name)
LUPINE_RPC_HANDLERS(LUPINE_DECLARE_GENERATED, LUPINE_DECLARE_MANUAL)
#undef LUPINE_DECLARE_GENERATED
#undef LUPINE_DECLARE_MANUAL

const rpc_handler_registry &lupine_rpc_handlers() {
#define LUPINE_REGISTER_GENERATED(operation, handler, backend)                 \
  {operation, {handler, &backend}},
#define LUPINE_REGISTER_MANUAL(operation, handler, backend, name)              \
  {operation, {handler, &backend, name, rpc_handler_error_style::manual}},
  static const rpc_handler_registry handlers = {
      LUPINE_RPC_HANDLERS(LUPINE_REGISTER_GENERATED, LUPINE_REGISTER_MANUAL)
#if CUDA_VERSION >= 12000
          LUPINE_REGISTER_MANUAL(RPC_cuTensorMapEncodeTiled,
                                 handle_manual_cuTensorMapEncodeTiled,
                                 lupine_cuda_backend, "cuTensorMapEncodeTiled")
#endif
  };
#undef LUPINE_REGISTER_GENERATED
#undef LUPINE_REGISTER_MANUAL
  return handlers;
}

#undef LUPINE_RPC_HANDLERS
