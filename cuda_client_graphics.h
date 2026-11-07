#ifndef LUPINE_CUDA_CLIENT_GRAPHICS_H
#define LUPINE_CUDA_CLIENT_GRAPHICS_H

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

// Graphics interop on a remote route. A registered OpenGL object lives in the
// client's GL context, which the server cannot reach, so the client keeps a
// CUDA copy on the server: mapping uploads the GL contents into it and
// unmapping writes them back.
CUresult
lupine_staged_cuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource,
                                         GLuint buffer, unsigned int Flags);
CUresult
lupine_staged_cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource,
                                        GLuint image, GLenum target,
                                        unsigned int Flags);
CUresult lupine_staged_cuGLGetDevices_v2(unsigned int *pCudaDeviceCount,
                                         CUdevice *pCudaDevices,
                                         unsigned int cudaDeviceCount,
                                         CUGLDeviceList deviceList);
CUresult
lupine_staged_cuGraphicsUnregisterResource(CUgraphicsResource resource);
CUresult lupine_staged_cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel);
CUresult lupine_staged_cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource);
CUresult lupine_staged_cuGraphicsResourceGetMappedPointer_v2(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource);
CUresult
lupine_staged_cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                               unsigned int flags);
CUresult lupine_staged_cuGraphicsMapResources(unsigned int count,
                                              CUgraphicsResource *resources,
                                              CUstream hStream);
CUresult lupine_staged_cuGraphicsUnmapResources(unsigned int count,
                                                CUgraphicsResource *resources,
                                                CUstream hStream);

#endif
