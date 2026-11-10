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

#endif
