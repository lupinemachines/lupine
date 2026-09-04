#ifndef LUPINE_NVML_RUNTIME_H
#define LUPINE_NVML_RUNTIME_H

#include <cstdint>

uintptr_t lupine_nvml_symbol_address(const char *name);

template <typename Fn> Fn lupine_nvml_symbol(const char *name) {
  return reinterpret_cast<Fn>(lupine_nvml_symbol_address(name));
}

#endif
