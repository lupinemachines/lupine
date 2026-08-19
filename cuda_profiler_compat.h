#ifndef LUPINE_CUDA_PROFILER_COMPAT_H
#define LUPINE_CUDA_PROFILER_COMPAT_H

#if __has_include(<cudaProfiler.h>)
#include <cudaProfiler.h>
#else
// The redistributable CUDA runtime headers contain cuda.h but omit the
// deprecated profiler header. CUDA 12+ still preserves this ABI type while
// cuProfilerInitialize itself always reports unsupported on a real driver.
typedef enum CUoutput_mode_enum {
  CU_OUT_KEY_VALUE_PAIR = 0x00,
  CU_OUT_CSV = 0x01,
} CUoutput_mode;
#endif

#endif
