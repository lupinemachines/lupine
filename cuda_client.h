#ifndef LUPINE_CUDA_CLIENT_H
#define LUPINE_CUDA_CLIENT_H

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include "cache.h"

bool lupine_cuda_is_initialized();

struct lupine_kernel_attribute_key {
  int route_id = -2;
  CUkernel kernel = nullptr;
  int attribute = 0;
  int device = -1;

  bool operator==(const lupine_kernel_attribute_key &other) const {
    return route_id == other.route_id && kernel == other.kernel &&
           attribute == other.attribute && device == other.device;
  }
};

struct lupine_function_attribute_key {
  int route_id = -2;
  CUfunction function = nullptr;
  int attribute = 0;

  bool operator==(const lupine_function_attribute_key &other) const {
    return route_id == other.route_id && function == other.function &&
           attribute == other.attribute;
  }
};

// A successful set request need not equal the value the driver reports.
enum class lupine_attribute_source { query, set };

bool lupine_attribute_get(
    const lupine_function_attribute_key &key, int &value,
    lupine_attribute_source source = lupine_attribute_source::query);
bool lupine_attribute_get(const lupine_kernel_attribute_key &key, int &value);
void lupine_attribute_set(
    const lupine_function_attribute_key &key, int value,
    lupine_attribute_source source = lupine_attribute_source::query);
void lupine_attribute_set(
    const lupine_kernel_attribute_key &key, int value,
    lupine_attribute_source source = lupine_attribute_source::query);

struct lupine_route;

bool lupine_try_cached_module_function(CUmodule module, const char *name,
                                       lupine_route route,
                                       CUfunction *function);
void lupine_finish_module_get_function(CUfunction function, CUmodule module,
                                       const char *name, lupine_route route);
bool lupine_try_cached_library_kernel(CUlibrary library, const char *name,
                                      CUkernel *kernel);
CUresult lupine_prepare_kernel_get_function(CUfunction *pFunc, CUkernel *kernel,
                                            lupine_route *route,
                                            CUcontext *current_context);
CUresult lupine_finish_kernel_get_function(CUresult return_value,
                                           lupine_route route,
                                           CUcontext current_context,
                                           CUkernel kernel, CUfunction function,
                                           CUfunction *pFunc);
bool lupine_try_cached_kernel_library(CUkernel kernel, CUlibrary *pLib);
bool lupine_try_cached_occupancy(lupine_route route, CUfunction func,
                                 int blockSize, size_t dynamicSMemSize,
                                 unsigned int flags, bool with_flags,
                                 int *numBlocks);
void lupine_store_cached_occupancy(lupine_route route, CUfunction func,
                                   int blockSize, size_t dynamicSMemSize,
                                   unsigned int flags, bool with_flags,
                                   int numBlocks);
bool lupine_try_cached_stream_is_capturing(
    CUstream hStream, CUstreamCaptureStatus *captureStatus);

#endif
