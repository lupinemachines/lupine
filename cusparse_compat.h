#ifndef LUPINE_CUSPARSE_COMPAT_H
#define LUPINE_CUSPARSE_COMPAT_H

// The experimental block holds the SpMVOp calls, which the shim forwards too.
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#include <cusparse.h>

// cuSPARSE 12.0 retyped the descriptors a call only reads as const. The
// generated code is written against the newer names; on 11.x each names the
// descriptor type the same calls took there.
#if CUSPARSE_VER_MAJOR < 12
typedef cusparseSpVecDescr_t cusparseConstSpVecDescr_t;
typedef cusparseDnVecDescr_t cusparseConstDnVecDescr_t;
typedef cusparseSpMatDescr_t cusparseConstSpMatDescr_t;
typedef cusparseDnMatDescr_t cusparseConstDnMatDescr_t;
#endif

#endif
