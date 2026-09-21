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

// Resolves in the real driver the server links, whatever headers built it.
static inline void *lupine_driver_symbol(const char *name) {
#ifdef _WIN32
  static HMODULE lib = LoadLibraryA("nvcuda.dll");
  return lib != nullptr ? reinterpret_cast<void *>(GetProcAddress(lib, name))
                        : nullptr;
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

#define LUPINE_FORWARD_DRIVER(name, ...)                                       \
  static auto fn =                                                             \
      reinterpret_cast<decltype(&name)>(lupine_driver_symbol(#name));          \
  return fn != nullptr ? fn(__VA_ARGS__) : CUDA_ERROR_NOT_SUPPORTED
#endif

// Attribute-snapshot wire entry carrying the function's device (the server's
// current device at snapshot time); sits below every real CUfunction_attribute.
#define LUPINE_FUNC_ATTRIBUTE_DEVICE (-1)

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
CUresult cuLibraryLoadData(CUlibrary *, const void *, CUjit_option *, void **,
                           unsigned int, CUlibraryOption *, void **,
                           unsigned int);
CUresult cuGraphAddKernelNode_v2(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, const CUDA_KERNEL_NODE_PARAMS *);
CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode,
                                       const CUDA_KERNEL_NODE_PARAMS *);
CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec, CUgraphNode,
                                           const CUDA_KERNEL_NODE_PARAMS *);
CUresult cuKernelGetParamInfo(CUkernel, size_t, size_t *, size_t *);
CUresult cuFuncGetParamInfo(CUfunction, size_t, size_t *, size_t *);
CUresult cuKernelGetAttribute(int *, CUfunction_attribute, CUkernel, CUdevice);
CUresult cuKernelGetFunction(CUfunction *, CUkernel);
CUresult cuLibraryGetKernel(CUkernel *, CUlibrary, const char *);
CUresult cuLibraryGetModule(CUmodule *, CUlibrary);
#ifdef __cplusplus
}
#endif
#endif

#ifndef LUPINE_CUDA_COMPAT_TYPES_ONLY
// A runtime newer than these headers registers every kernel through the
// library API, so the server forwards it rather than failing it.
static inline CUresult cuCtxGetId(CUcontext ctx, unsigned long long *id) {
  LUPINE_FORWARD_DRIVER(cuCtxGetId, ctx, id);
}

static inline CUresult cuStreamGetId(CUstream stream, unsigned long long *id) {
  LUPINE_FORWARD_DRIVER(cuStreamGetId, stream, id);
}

static inline CUresult cuLibraryLoadFromFile(
    CUlibrary *library, const char *fileName, CUjit_option *jitOptions,
    void **jitOptionsValues, unsigned int numJitOptions,
    CUlibraryOption *libraryOptions, void **libraryOptionValues,
    unsigned int numLibraryOptions) {
  LUPINE_FORWARD_DRIVER(cuLibraryLoadFromFile, library, fileName, jitOptions,
                        jitOptionsValues, numJitOptions, libraryOptions,
                        libraryOptionValues, numLibraryOptions);
}

static inline CUresult
cuLibraryLoadData(CUlibrary *library, const void *code,
                  CUjit_option *jitOptions, void **jitOptionsValues,
                  unsigned int numJitOptions, CUlibraryOption *libraryOptions,
                  void **libraryOptionValues, unsigned int numLibraryOptions) {
  LUPINE_FORWARD_DRIVER(cuLibraryLoadData, library, code, jitOptions,
                        jitOptionsValues, numJitOptions, libraryOptions,
                        libraryOptionValues, numLibraryOptions);
}

static inline CUresult cuLibraryUnload(CUlibrary library) {
  LUPINE_FORWARD_DRIVER(cuLibraryUnload, library);
}

static inline CUresult cuLibraryGetKernel(CUkernel *kernel, CUlibrary library,
                                          const char *name) {
  LUPINE_FORWARD_DRIVER(cuLibraryGetKernel, kernel, library, name);
}

static inline CUresult cuLibraryGetModule(CUmodule *module, CUlibrary library) {
  LUPINE_FORWARD_DRIVER(cuLibraryGetModule, module, library);
}

static inline CUresult cuKernelGetFunction(CUfunction *function,
                                           CUkernel kernel) {
  LUPINE_FORWARD_DRIVER(cuKernelGetFunction, function, kernel);
}

static inline CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                          CUlibrary library, const char *name) {
  LUPINE_FORWARD_DRIVER(cuLibraryGetGlobal, dptr, bytes, library, name);
}

static inline CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                                           CUlibrary library,
                                           const char *name) {
  LUPINE_FORWARD_DRIVER(cuLibraryGetManaged, dptr, bytes, library, name);
}

static inline CUresult cuLibraryGetUnifiedFunction(void **fptr,
                                                   CUlibrary library,
                                                   const char *symbol) {
  LUPINE_FORWARD_DRIVER(cuLibraryGetUnifiedFunction, fptr, library, symbol);
}

static inline CUresult cuKernelGetAttribute(int *pi,
                                            CUfunction_attribute attrib,
                                            CUkernel kernel, CUdevice dev) {
  LUPINE_FORWARD_DRIVER(cuKernelGetAttribute, pi, attrib, kernel, dev);
}

static inline CUresult cuKernelSetAttribute(CUfunction_attribute attrib,
                                            int val, CUkernel kernel,
                                            CUdevice dev) {
  LUPINE_FORWARD_DRIVER(cuKernelSetAttribute, attrib, val, kernel, dev);
}

static inline CUresult
cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) {
  LUPINE_FORWARD_DRIVER(cuKernelSetCacheConfig, kernel, config, dev);
}

static inline CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex,
                                            size_t *paramOffset,
                                            size_t *paramSize) {
  LUPINE_FORWARD_DRIVER(cuKernelGetParamInfo, kernel, paramIndex, paramOffset,
                        paramSize);
}

static inline CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex,
                                          size_t *paramOffset,
                                          size_t *paramSize) {
  LUPINE_FORWARD_DRIVER(cuFuncGetParamInfo, func, paramIndex, paramOffset,
                        paramSize);
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

// cuKernelGetLibrary was added in CUDA 12.5.
#if CUDA_VERSION < 12050
#ifdef LUPINE_CUDA_COMPAT_TYPES_ONLY
#ifdef __cplusplus
extern "C" {
#endif
CUresult cuKernelGetLibrary(CUlibrary *, CUkernel);
#ifdef __cplusplus
}
#endif
#endif
#ifndef LUPINE_CUDA_COMPAT_TYPES_ONLY
static inline CUresult cuKernelGetLibrary(CUlibrary *library, CUkernel kernel) {
  LUPINE_FORWARD_DRIVER(cuKernelGetLibrary, library, kernel);
}
#endif
#endif

// Applications can bundle a runtime newer than the headers lupine was built
// with, and that runtime stops launching kernels if these lookups fail. The
// server's driver has them regardless, so an older build still forwards them.
#if CUDA_VERSION < 12080
#ifdef LUPINE_CUDA_COMPAT_TYPES_ONLY
#ifdef __cplusplus
extern "C" {
#endif
#if CUDA_VERSION < 12030
CUresult cuKernelGetName(const char **, CUkernel);
CUresult cuFuncGetName(const char **, CUfunction);
#endif
CUresult cuStreamGetDevice(CUstream, CUdevice *);
#ifdef __cplusplus
}
#endif
#else
#if CUDA_VERSION < 12030
static inline CUresult cuKernelGetName(const char **name, CUkernel hfunc) {
  LUPINE_FORWARD_DRIVER(cuKernelGetName, name, hfunc);
}

static inline CUresult cuFuncGetName(const char **name, CUfunction hfunc) {
  LUPINE_FORWARD_DRIVER(cuFuncGetName, name, hfunc);
}
#endif

static inline CUresult cuStreamGetDevice(CUstream hStream, CUdevice *device) {
  LUPINE_FORWARD_DRIVER(cuStreamGetDevice, hStream, device);
}
#endif
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
  unsigned char pad;
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

// Graph edge-data query ABIs were added in CUDA 12.3. Keep the generated v2
// RPC surface available to older client builds and adapt server calls to the
// legacy queries when edge data is not requested.
#if CUDA_VERSION < 12030
#ifdef LUPINE_CUDA_COMPAT_TYPES_ONLY
#ifdef __cplusplus
extern "C" {
#endif
CUresult cuGraphGetEdges_v2(CUgraph, CUgraphNode *, CUgraphNode *,
                            CUgraphEdgeData *, size_t *);
CUresult cuGraphNodeGetDependencies_v2(CUgraphNode, CUgraphNode *,
                                       CUgraphEdgeData *, size_t *);
CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode, CUgraphNode *,
                                         CUgraphEdgeData *, size_t *);
#ifdef __cplusplus
}
#endif
#else
static inline CUresult cuGraphGetEdges_v2(CUgraph graph, CUgraphNode *from,
                                          CUgraphNode *to,
                                          CUgraphEdgeData *edgeData,
                                          size_t *numEdges) {
  if (edgeData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuGraphGetEdges(graph, from, to, numEdges);
}

static inline CUresult cuGraphNodeGetDependencies_v2(CUgraphNode node,
                                                     CUgraphNode *dependencies,
                                                     CUgraphEdgeData *edgeData,
                                                     size_t *numDependencies) {
  if (edgeData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuGraphNodeGetDependencies(node, dependencies, numDependencies);
}

static inline CUresult
cuGraphNodeGetDependentNodes_v2(CUgraphNode node, CUgraphNode *dependentNodes,
                                CUgraphEdgeData *edgeData,
                                size_t *numDependentNodes) {
  if (edgeData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuGraphNodeGetDependentNodes(node, dependentNodes, numDependentNodes);
}
#endif
#endif

#endif
