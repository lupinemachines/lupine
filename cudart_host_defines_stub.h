/*
 * Host-side stand-in for nvcc's crt/host_defines.h.
 *
 * The redistributable CUDA header wheels ship cuda_runtime_api.h without the
 * compiler-internal crt/ directory it includes. Host code only needs the
 * qualifiers to expand to nothing and the calling-convention macros to exist,
 * so the root build places this file as crt/host_defines.h when the CUDA
 * include directory has no crt/ of its own.
 */

#ifndef LUPINE_CUDART_HOST_DEFINES_H
#define LUPINE_CUDART_HOST_DEFINES_H

#define __host__
#define __device__
#define __device_builtin__
#define __device_builtin_runtime__
#define __align__(n) alignas(n)
#define __builtin_align__(n) alignas(n)
#define __global__
#define __forceinline__ inline
#define __no_return__ [[noreturn]]
#define __annotate__(a)
#define __location__(a)
#define __launch_bounds__(...)
#if defined(_MSC_VER)
#define __restrict__ __restrict
#define __inline__ __inline
#else
#define __inline__ inline
#endif
#define __thread__ __thread
#define __import__
#define __export__
#define __cdecl

#if defined(_MSC_VER) && defined(_M_IX86)
#define CUDARTAPI __stdcall
#else
#define CUDARTAPI
#endif
#define CUDARTAPI_CDECL

#define __dv(x) = x
#define __cudart_builtin__
#define __cudart_builtin_device__

#endif
