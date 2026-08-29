#ifndef LUPINE_CUDA_SERVER_H
#define LUPINE_CUDA_SERVER_H

#include <cuda.h>

#include "rpc.h"

int handle_cuGetErrorName(conn_t *conn);
int handle_cuGetErrorString(conn_t *conn);
int handle_cuGetExportTableMetadata(conn_t *conn);
int handle_cuPrivateGetModuleNode(conn_t *conn);
int handle_cuModuleLoad(conn_t *conn);
int handle_cuModuleLoadData(conn_t *conn);
int handle_lupineFunctionParamLayoutSnapshot(conn_t *conn);
int handle_lupineFunctionAttributeSnapshot(conn_t *conn);
int handle_cuLibraryLoadData(conn_t *conn);
int handle_lupineLibrarySnapshot(conn_t *conn);
int handle_lupineLibraryAttributeSnapshot(conn_t *conn);
int handle_cuMemPoolSetAttribute(conn_t *conn);
int handle_cuMemPoolGetAttribute(conn_t *conn);
int handle_cuMemExportToShareableHandle(conn_t *conn);
int handle_cuMemImportFromShareableHandle(conn_t *conn);
int handle_cuMemPoolExportToShareableHandle(conn_t *conn);
int handle_cuMemPoolImportFromShareableHandle(conn_t *conn);
int handle_cuMemRangeGetAttributes(conn_t *conn);
int handle_cuPointerGetAttribute(conn_t *conn);
int handle_cuPointerSetAttribute(conn_t *conn);
int handle_cuPointerGetAttributes(conn_t *conn);
int handle_cuLinkCreate_v2(conn_t *conn);
int handle_cuLinkAddData_v2(conn_t *conn);
int handle_cuLinkAddFile_v2(conn_t *conn);
int handle_cuLinkComplete(conn_t *conn);
int handle_cuLinkDestroy(conn_t *conn);
int handle_cuMemcpy3D_v2(conn_t *conn);
int handle_cuMemcpy3DAsync_v2(conn_t *conn);
int handle_cuMemcpy3DPeer(conn_t *conn);
int handle_cuMemcpy3DPeerAsync(conn_t *conn);
int handle_cuMemcpy2D_v2(conn_t *conn);
int handle_cuMemcpy2DUnaligned_v2(conn_t *conn);
int handle_cuMemcpy2DAsync_v2(conn_t *conn);
int handle_cuMemcpyAtoH_v2(conn_t *conn);
int handle_cuDeviceGetGraphMemAttribute(conn_t *conn);
int handle_cuDeviceSetGraphMemAttribute(conn_t *conn);
int handle_cuLibraryGetModule(conn_t *conn);
int handle_cuLibraryUnload(conn_t *conn);
int handle_cuModuleGetGlobal_v2(conn_t *conn);
int handle_cuLaunchKernel(conn_t *conn);
int handle_cuLaunchKernelEx(conn_t *conn);
int handle_cuLaunchCooperativeKernel(conn_t *conn);
int handle_cuGraphAddKernelNode(conn_t *conn);
int handle_cuGraphKernelNodeGetParams(conn_t *conn);
int handle_cuGraphKernelNodeSetParams(conn_t *conn);
int handle_cuGraphAddMemcpyNode(conn_t *conn);
int handle_cuGraphAddMemsetNode(conn_t *conn);
int handle_cuGraphAddHostNode(conn_t *conn);
int handle_cuGraphExecKernelNodeSetParams(conn_t *conn);
int handle_cuGraphConditionalHandleCreate(conn_t *conn);
int handle_cuGraphAddNode(conn_t *conn);
int handle_cuGraphHostNodeGetParams(conn_t *conn);
int handle_cuGraphHostNodeSetParams(conn_t *conn);
int handle_cuGraphExecHostNodeSetParams(conn_t *conn);
int handle_cuLaunchHostFunc(conn_t *conn);
int handle_cuStreamAddCallback(conn_t *conn);
int handle_cuEventRecord(conn_t *conn);
int handle_cuEventRecordWithFlags(conn_t *conn);
int handle_cuEventQuery(conn_t *conn);
int handle_lupineEventQueryBatch(conn_t *conn);
int handle_cuStreamWaitEvent(conn_t *conn);
int handle_cuStreamBeginCaptureToGraph(conn_t *conn);
int handle_cuStreamUpdateCaptureDependencies(conn_t *conn);
int handle_cuStreamGetCaptureInfo(conn_t *conn);
int handle_cuStreamBeginCapture(conn_t *conn);
int handle_cuStreamEndCapture(conn_t *conn);
int handle_cuGraphClone(conn_t *conn);
int handle_cuGraphInstantiate_v2(conn_t *conn);
int handle_cuGraphInstantiateWithFlags(conn_t *conn);
int handle_cuGraphInstantiateWithParams(conn_t *conn);
int handle_cuGraphExecDestroy(conn_t *conn);
int handle_cuGraphDestroy(conn_t *conn);
int handle_cuMemcpyHtoDAsync_v2(conn_t *conn);
int handle_lupineDeviceSnapshot(conn_t *conn);
int handle_lupineManagedHostFlush(conn_t *conn);
int handle_cuMemcpyDtoHAsync_v2(conn_t *conn);
int handle_cuMemHostAlloc(conn_t *conn);
int handle_cuMemHostGetFlags(conn_t *conn);
int handle_cuMemFreeHost(conn_t *conn);
int handle_cuMemAllocManaged(conn_t *conn);
int handle_cuMemFree_v2(conn_t *conn);
void lupine_server_cleanup_identity_allocations(conn_t *conn);
int handle_cuCtxSynchronize(conn_t *conn);
int handle_cuStreamSynchronize(conn_t *conn);
int handle_cuGraphLaunch(conn_t *conn);
int handle_cuEventSynchronize(conn_t *conn);
int handle_cuOccupancyMaxPotentialBlockSize(conn_t *conn);
int handle_cuOccupancyMaxPotentialBlockSizeWithFlags(conn_t *conn);
#if CUDA_VERSION >= 12000
int handle_cuTensorMapEncodeTiled(conn_t *conn);
#endif

bool lupine_server_initialize_connection(conn_t *conn);
void lupine_server_cleanup_connection(conn_t *conn);

int handle_cuDevicePrimaryCtxRetain(conn_t *conn);
int handle_cuDevicePrimaryCtxRelease_v2(conn_t *conn);
int handle_cuDevicePrimaryCtxReset_v2(conn_t *conn);
int handle_cuCtxAttach(conn_t *conn);
int handle_cuCtxCreate_v2(conn_t *conn);
int handle_cuCtxDestroy_v2(conn_t *conn);
int handle_cuCtxDetach(conn_t *conn);
int handle_cuMemcpyHtoD_v2(conn_t *conn);
int handle_cuMemcpyDtoH_v2(conn_t *conn);

#endif
