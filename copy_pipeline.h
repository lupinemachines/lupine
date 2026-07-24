#ifndef LUPINE_COPY_PIPELINE_H
#define LUPINE_COPY_PIPELINE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <vector>

#include "rpc.h"

template <typename Allocation> struct lupine_dirty_range {
  Allocation *allocation = nullptr;
  uintptr_t start = 0;
  uintptr_t end = 0;
};

template <typename Allocation, typename AllocationBase>
std::vector<lupine_dirty_range<Allocation>>
lupine_sort_and_coalesce_dirty_ranges(
    std::vector<lupine_dirty_range<Allocation>> ranges,
    AllocationBase allocation_base) {
  std::sort(ranges.begin(), ranges.end(), [&](const auto &a, const auto &b) {
    uintptr_t a_base = allocation_base(a.allocation);
    uintptr_t b_base = allocation_base(b.allocation);
    if (a_base != b_base) {
      return a_base < b_base;
    }
    if (a.start != b.start) {
      return a.start < b.start;
    }
    return a.end < b.end;
  });

  std::vector<lupine_dirty_range<Allocation>> merged;
  merged.reserve(ranges.size());
  for (const auto &range : ranges) {
    if (merged.empty() || merged.back().allocation != range.allocation ||
        merged.back().end < range.start) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }
  return merged;
}

#ifdef LUPINE_RPC_SERVER
bool lupine_server_initialize_connection(conn_t *conn);
void lupine_server_cleanup_connection(conn_t *conn);

int lupine_server_copy_htod_async(conn_t *conn, int framed,
                                  CUdeviceptr destination, size_t bytes,
                                  CUstream stream, CUresult &result);

int handle_manual_cuDevicePrimaryCtxRetain(conn_t *conn);
int handle_manual_cuDevicePrimaryCtxRelease_v2(conn_t *conn);
int handle_manual_cuDevicePrimaryCtxReset_v2(conn_t *conn);
int handle_manual_cuCtxAttach(conn_t *conn);
int handle_manual_cuCtxCreate_v2(conn_t *conn);
int handle_manual_cuCtxDestroy_v2(conn_t *conn);
int handle_manual_cuCtxDetach(conn_t *conn);
int handle_manual_cuMemcpyHtoD_v2(conn_t *conn);
int handle_manual_cuMemcpyDtoH_v2(conn_t *conn);
#endif

#endif
