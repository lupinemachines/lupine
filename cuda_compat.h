#ifndef LUPINE_CUDA_COMPAT_H
#define LUPINE_CUDA_COMPAT_H

#include <cuda.h>

#ifndef LUPINE_CUDA_COMPAT_TYPES_ONLY
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#if CUDA_VERSION < 12000
typedef struct CUlibrary_st *CUlibrary;
typedef struct CUkernel_st *CUkernel;
typedef int CUlibraryOption;
#define CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE ((CUlibraryOption)0)
#define CU_LIBRARY_BINARY_IS_PRESERVED ((CUlibraryOption)1)
typedef struct CUDA_GRAPH_INSTANTIATE_PARAMS_st {
  unsigned long long flags;
  CUstream hUploadStream;
  CUgraphNode *errorNode_out;
  CUgraphExecUpdateResult *result_out;
  char *logBuffer;
  unsigned long long bufferSize;
} CUDA_GRAPH_INSTANTIATE_PARAMS;

typedef struct CUgraphExecUpdateResultInfo_st {
  CUgraphExecUpdateResult result;
  CUgraphNode errorNode;
} CUgraphExecUpdateResultInfo;

#ifdef LUPINE_CUDA_COMPAT_TYPES_ONLY
#ifdef __cplusplus
extern "C" {
#endif
CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec, CUgraphNode,
                                           const CUDA_KERNEL_NODE_PARAMS *);
#ifdef __cplusplus
}
#endif
#endif

#ifndef LUPINE_CUDA_COMPAT_TYPES_ONLY
static inline CUresult cuCtxGetId(CUcontext, unsigned long long *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuStreamGetId(CUstream, unsigned long long *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryLoadFromFile(CUlibrary *, const char *,
                                             CUjit_option *, void **,
                                             unsigned int, CUlibraryOption *,
                                             void **, unsigned int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryLoadData(CUlibrary *, const void *,
                                         CUjit_option *, void **, unsigned int,
                                         CUlibraryOption *, void **,
                                         unsigned int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryUnload(CUlibrary) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryGetKernel(CUkernel *, CUlibrary, const char *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryGetModule(CUmodule *, CUlibrary) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuKernelGetFunction(CUfunction *, CUkernel) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryGetGlobal(CUdeviceptr *, size_t *, CUlibrary,
                                          const char *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryGetManaged(CUdeviceptr *, size_t *, CUlibrary,
                                           const char *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuLibraryGetUnifiedFunction(void **, CUlibrary,
                                                   const char *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuKernelGetAttribute(int *, CUfunction_attribute,
                                            CUkernel, CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuKernelSetAttribute(CUfunction_attribute, int, CUkernel,
                                            CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuKernelSetCacheConfig(CUkernel, CUfunc_cache,
                                              CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex,
                                            size_t *paramOffset,
                                            size_t *paramSize) {
  using cuKernelGetParamInfo_t =
      CUresult(CUDAAPI *)(CUkernel, size_t, size_t *, size_t *);
#ifdef _WIN32
  static HMODULE lib = LoadLibraryA("nvcuda.dll");
  static auto fn = reinterpret_cast<cuKernelGetParamInfo_t>(
      lib != nullptr ? GetProcAddress(lib, "cuKernelGetParamInfo") : nullptr);
#else
  static auto fn = reinterpret_cast<cuKernelGetParamInfo_t>(
      dlsym(RTLD_DEFAULT, "cuKernelGetParamInfo"));
#endif
  return fn != nullptr ? fn(kernel, paramIndex, paramOffset, paramSize)
                       : CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex,
                                          size_t *paramOffset,
                                          size_t *paramSize) {
  using cuFuncGetParamInfo_t =
      CUresult(CUDAAPI *)(CUfunction, size_t, size_t *, size_t *);
#ifdef _WIN32
  static HMODULE lib = LoadLibraryA("nvcuda.dll");
  static auto fn = reinterpret_cast<cuFuncGetParamInfo_t>(
      lib != nullptr ? GetProcAddress(lib, "cuFuncGetParamInfo") : nullptr);
#else
  static auto fn = reinterpret_cast<cuFuncGetParamInfo_t>(
      dlsym(RTLD_DEFAULT, "cuFuncGetParamInfo"));
#endif
  return fn != nullptr ? fn(func, paramIndex, paramOffset, paramSize)
                       : CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult
cuGraphKernelNodeGetParams_v2(CUgraphNode hNode,
                              CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphKernelNodeGetParams(hNode, nodeParams);
}

static inline CUresult
cuGraphKernelNodeSetParams_v2(CUgraphNode hNode,
                              const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphKernelNodeSetParams(hNode, nodeParams);
}

static inline CUresult
cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode,
                                  const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams);
}

static inline CUresult
cuGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph,
                             CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams) {
  unsigned long long flags =
      instantiateParams == nullptr ? 0 : instantiateParams->flags;
  return cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
}

static inline CUresult cuGraphExecGetFlags(CUgraphExec, cuuint64_t *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuGraphExecUpdate_v2(CUgraphExec, CUgraph,
                                            CUgraphExecUpdateResultInfo *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif
#endif

#if CUDA_VERSION < 12000
typedef int CUdriverProcAddressQueryResult;
#define CU_GET_PROC_ADDRESS_SUCCESS 0
#define CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND 1
#endif

#if CUDA_VERSION < 11080
typedef struct CUlaunchAttribute_st {
  int id;
  int pad[7];
} CUlaunchAttribute;

typedef struct CUlaunchConfig_st {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute *attrs;
  unsigned int numAttrs;
} CUlaunchConfig;

#ifndef LUPINE_CUDA_COMPAT_TYPES_ONLY
static inline CUresult
cuOccupancyMaxPotentialClusterSize(int *, CUfunction, const CUlaunchConfig *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuOccupancyMaxActiveClusters(int *, CUfunction,
                                                    const CUlaunchConfig *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif
#endif

#if CUDA_VERSION < 12000
typedef struct CUgraphEdgeData_st {
  unsigned char from_port;
  unsigned char to_port;
  unsigned char type;
  unsigned char reserved[5];
} CUgraphEdgeData;

typedef unsigned long long CUgraphConditionalHandle;

typedef struct CUgraphNodeParams_st {
  CUgraphNodeType type;
  CUDA_KERNEL_NODE_PARAMS kernel;
  struct {
    CUgraphConditionalHandle handle;
    unsigned int type;
    unsigned int size;
    CUgraph *phGraph_out;
  } conditional;
} CUgraphNodeParams;

#ifndef CU_GRAPH_NODE_TYPE_CONDITIONAL
#define CU_GRAPH_NODE_TYPE_CONDITIONAL ((CUgraphNodeType)8)
#endif

#ifndef LUPINE_CUDA_COMPAT_TYPES_ONLY
static inline CUresult
cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                        const CUgraphNode *dependencies, size_t numDependencies,
                        const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphAddKernelNode(phGraphNode, hGraph, dependencies,
                              numDependencies, nodeParams);
}

static inline CUresult
cuGraphConditionalHandleCreate(CUgraphConditionalHandle *, CUgraph, CUcontext,
                               unsigned int, unsigned int) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuGraphAddNode_v2(CUgraphNode *, CUgraph,
                                         const CUgraphNode *,
                                         const CUgraphEdgeData *, size_t,
                                         CUgraphNodeParams *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuGraphAddNode(CUgraphNode *, CUgraph,
                                      const CUgraphNode *, size_t,
                                      CUgraphNodeParams *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuStreamBeginCaptureToGraph(CUstream, CUgraph,
                                                   const CUgraphNode *,
                                                   const CUgraphEdgeData *,
                                                   size_t,
                                                   CUstreamCaptureMode) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

static inline CUresult cuStreamGetCaptureInfo_v3(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  if (edgeData_out != nullptr) {
    *edgeData_out = nullptr;
  }
  return cuStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out,
                                   dependencies_out, numDependencies_out);
}

static inline CUresult cuStreamUpdateCaptureDependencies_v2(
    CUstream stream, CUgraphNode *dependencies, const CUgraphEdgeData *,
    size_t numDependencies, unsigned int flags) {
  return cuStreamUpdateCaptureDependencies(stream, dependencies,
                                           numDependencies, flags);
}
#endif
#endif

#endif
