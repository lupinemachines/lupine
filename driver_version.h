#pragma once

#include <algorithm>
#include <cstddef>

// A process exposes one CUDA driver version even when its virtual device table
// spans several local and remote drivers. Report the newest API level that the
// client shim and every available route can all support.
inline int lupine_highest_common_driver_version(int client_version,
                                                const int *route_versions,
                                                std::size_t route_count) {
  if (client_version <= 0 || route_versions == nullptr || route_count == 0) {
    return 0;
  }
  int negotiated = client_version;
  for (std::size_t index = 0; index < route_count; ++index) {
    if (route_versions[index] <= 0) {
      return 0;
    }
    negotiated = std::min(negotiated, route_versions[index]);
  }
  return negotiated;
}
