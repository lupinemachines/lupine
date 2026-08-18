#ifndef LUPINE_MANUAL_SERVER_H
#define LUPINE_MANUAL_SERVER_H

#include <cuda.h>

#include "rpc.h"

int handle_manual_cuGetErrorName(conn_t *conn);
int handle_manual_cuGetErrorString(conn_t *conn);
int handle_manual_cuGetExportTableMetadata(conn_t *conn);
int handle_manual_cuPrivateGetModuleNode(conn_t *conn);
int handle_manual_cuModuleLoad(conn_t *conn);
int handle_manual_cuModuleLoadData(conn_t *conn);
int handle_manual_lupineFunctionParamLayoutSnapshot(conn_t *conn);
int handle_manual_lupineFunctionAttributeSnapshot(conn_t *conn);
int handle_manual_cuLibraryLoadData(conn_t *conn);
int handle_manual_lupineLibrarySnapshot(conn_t *conn);
int handle_manual_lupineLibraryAttributeSnapshot(conn_t *conn);
int handle_manual_cuMemPoolSetAttribute(conn_t *conn);
int handle_manual_cuMemPoolGetAttribute(conn_t *conn);
int handle_manual_cuMemExportToShareableHandle(conn_t *conn);
int handle_manual_cuMemImportFromShareableHandle(conn_t *conn);
int handle_manual_cuMemPoolExportToShareableHandle(conn_t *conn);
int handle_manual_cuMemPoolImportFromShareableHandle(conn_t *conn);
int handle_manual_cuPointerGetAttribute(conn_t *conn);
int handle_manual_cuPointerSetAttribute(conn_t *conn);
int handle_manual_cuPointerGetAttributes(conn_t *conn);
int handle_manual_cuLinkCreate_v2(conn_t *conn);
int handle_manual_cuLinkAddData_v2(conn_t *conn);
int handle_manual_cuLinkAddFile_v2(conn_t *conn);
int handle_manual_cuLinkComplete(conn_t *conn);
int handle_manual_cuLinkDestroy(conn_t *conn);
int handle_manual_cuMemcpy3D_v2(conn_t *conn);
int handle_manual_cuMemcpy2D_v2(conn_t *conn);
int handle_manual_cuMemcpy2DUnaligned_v2(conn_t *conn);
int handle_manual_cuMemcpy2DAsync_v2(conn_t *conn);
int handle_manual_cuMemcpyAtoH_v2(conn_t *conn);
int handle_manual_cuGraphAddMemAllocNode(conn_t *conn);
int handle_manual_cuGraphAddMemFreeNode(conn_t *conn);
int handle_manual_cuDeviceGetGraphMemAttribute(conn_t *conn);
int handle_manual_cuDeviceSetGraphMemAttribute(conn_t *conn);
int handle_manual_cuLibraryGetModule(conn_t *conn);
int handle_manual_cuLibraryUnload(conn_t *conn);
int handle_manual_cuModuleGetGlobal_v2(conn_t *conn);
int handle_manual_cuLaunchKernel(conn_t *conn);
int handle_manual_cuLaunchKernelEx(conn_t *conn);
int handle_manual_cuLaunchCooperativeKernel(conn_t *conn);
int handle_manual_cuGraphAddKernelNode(conn_t *conn);
int handle_manual_cuGraphAddMemcpyNode(conn_t *conn);
int handle_manual_cuGraphAddMemsetNode(conn_t *conn);
int handle_manual_cuGraphAddHostNode(conn_t *conn);
int handle_manual_cuGraphConditionalHandleCreate(conn_t *conn);
int handle_manual_cuGraphAddNode(conn_t *conn);
int handle_manual_cuGraphGetEdges(conn_t *conn);
int handle_manual_cuGraphNodeGetDependencies(conn_t *conn);
int handle_manual_cuGraphNodeGetDependentNodes(conn_t *conn);
int handle_manual_cuMemPrefetchAsync(conn_t *conn);
int handle_manual_cuGraphHostNodeGetParams(conn_t *conn);
int handle_manual_cuGraphHostNodeSetParams(conn_t *conn);
int handle_manual_cuGraphExecHostNodeSetParams(conn_t *conn);
int handle_manual_cuLaunchHostFunc(conn_t *conn);
int handle_manual_cuStreamAddCallback(conn_t *conn);
int handle_manual_cuEventRecord(conn_t *conn, bool with_flags);
int handle_manual_cuEventQuery(conn_t *conn);
int handle_manual_lupineEventQueryBatch(conn_t *conn);
int handle_manual_cuStreamWaitEvent(conn_t *conn);
int handle_manual_cuStreamBeginCaptureToGraph(conn_t *conn);
int handle_manual_cuStreamUpdateCaptureDependencies(conn_t *conn);
int handle_manual_cuStreamGetCaptureInfo(conn_t *conn);
int handle_manual_cuStreamBeginCapture(conn_t *conn);
int handle_manual_cuStreamEndCapture(conn_t *conn);
int handle_manual_cuGraphClone(conn_t *conn);
int handle_manual_cuGraphInstantiateWithFlags(conn_t *conn);
int handle_manual_cuGraphInstantiateWithParams(conn_t *conn);
int handle_manual_cuGraphExecDestroy(conn_t *conn);
int handle_manual_cuGraphDestroy(conn_t *conn);
int handle_manual_cuMemcpyHtoDAsync_v2(conn_t *conn);
int handle_manual_lupineDeviceSnapshot(conn_t *conn);
int handle_manual_lupineManagedHostFlush(conn_t *conn);
int handle_manual_cuMemcpyDtoHAsync_v2(conn_t *conn);
int handle_manual_cuMemHostAlloc(conn_t *conn);
int handle_manual_cuMemHostGetFlags(conn_t *conn);
int handle_manual_cuCtxSynchronize(conn_t *conn);
int handle_manual_cuStreamSynchronize(conn_t *conn);
int handle_manual_cuGraphLaunch(conn_t *conn);
int handle_manual_cuEventSynchronize(conn_t *conn);
int handle_manual_cuOccupancyMaxPotentialBlockSize(conn_t *conn,
                                                   bool with_flags);

#endif
