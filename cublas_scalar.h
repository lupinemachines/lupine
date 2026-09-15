#ifndef LUPINE_CUBLAS_SCALAR_H
#define LUPINE_CUBLAS_SCALAR_H

#include <cuComplex.h>
#include <library_types.h>

#include <cstddef>

// The width of a scalar an Ex call names by data type, shared by the cuBLAS
// and cuBLASLt shims. A type the library does not accept for a scalar is sent
// at the widest width so that the bytes, never a client address, travel.
inline size_t data_type_width(cudaDataType type) {
  switch (type) {
  case CUDA_R_8I:
  case CUDA_R_8U:
    return 1;
  case CUDA_R_16F:
  case CUDA_R_16BF:
  case CUDA_R_16I:
  case CUDA_R_16U:
  case CUDA_C_8I:
  case CUDA_C_8U:
    return 2;
  case CUDA_R_32F:
  case CUDA_R_32I:
  case CUDA_R_32U:
  case CUDA_C_16F:
  case CUDA_C_16BF:
  case CUDA_C_16I:
  case CUDA_C_16U:
    return 4;
  case CUDA_R_64F:
  case CUDA_R_64I:
  case CUDA_R_64U:
  case CUDA_C_32F:
  case CUDA_C_32I:
  case CUDA_C_32U:
    return 8;
  default:
    return sizeof(cuDoubleComplex);
  }
}

#endif
