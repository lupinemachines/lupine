#ifndef LUPINE_INTERNAL_CUDA_QUERY_H
#define LUPINE_INTERNAL_CUDA_QUERY_H

// Metadata warming uses expected failing CUDA queries to find parameter-list
// boundaries and unsupported attributes. These are Lupine's queries, rather
// than application calls, and must not be surfaced to application log hooks.
inline thread_local unsigned lupine_internal_cuda_query_depth = 0;

class lupine_internal_cuda_query {
public:
  lupine_internal_cuda_query() { ++lupine_internal_cuda_query_depth; }
  ~lupine_internal_cuda_query() { --lupine_internal_cuda_query_depth; }
  lupine_internal_cuda_query(const lupine_internal_cuda_query &) = delete;
  lupine_internal_cuda_query &
  operator=(const lupine_internal_cuda_query &) = delete;
};

inline bool lupine_is_internal_cuda_query() {
  return lupine_internal_cuda_query_depth != 0;
}

#endif
