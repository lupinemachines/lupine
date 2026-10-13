#include "nvml_server.h"

#include <cuda.h>
#include <nvml.h>

#include <cstdlib>

#include "codegen/gen_rpc_ids.h"
#include "nvml_runtime.h"

// CUDA <= 12.6 ships NVML API 12, which does not define the versioned
// temperature struct. The host driver exports the symbol on newer drivers; this
// local definition preserves the ABI when building against older CUDA images.
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) ||                        \
    (defined(NVML_API_VERSION) && NVML_API_VERSION >= 13)
using lupine_nvmlTemperature_t = nvmlTemperature_t;
#else
typedef struct {
  unsigned int version;
  nvmlTemperatureSensors_t sensorType;
  int temperature;
} lupine_nvmlTemperature_t;
#endif

namespace {

nvmlReturn_t function_not_found() { return NVML_ERROR_FUNCTION_NOT_FOUND; }

template <typename Fn> Fn nvml_symbol(const char *name) {
  return lupine_nvml_symbol<Fn>(name);
}

} // namespace

#include "codegen/gen_nvml_server.inc"
