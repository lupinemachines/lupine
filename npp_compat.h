#ifndef LUPINE_NPP_COMPAT_H
#define LUPINE_NPP_COMPAT_H

#include <cstddef>

#include <npp.h>

// The parameters NPP widened from int to size_t, named for the release that
// widened them, so one declaration matches every release's header.
#if NPP_VERSION >= 12205
typedef size_t lupine_npp_length_t;
#else
typedef int lupine_npp_length_t;
#endif
#if NPP_VERSION >= 12303
typedef size_t lupine_npp_watershed_size_t;
#else
typedef int lupine_npp_watershed_size_t;
#endif

#endif
