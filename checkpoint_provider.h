#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LUPINE_CHECKPOINT_PROVIDER_ABI_VERSION 1u
#define LUPINE_CHECKPOINT_PROVIDER_SYMBOL "lupinecr_get_lupine_provider_v1"

typedef struct lupine_checkpoint_provider_v1 {
  size_t struct_size;
  uint32_t abi_version;

  // Called in a freshly forked connection process before its first CUDA call.
  // Providers can begin observing RM/UVM activity here.
  int (*start)(void);

  // Writes a complete checkpoint for the current connection. The provider
  // owns the file layout beneath directory.
  int (*checkpoint)(const char *directory, uint64_t connection_id);

  // Stops observation and releases provider-owned process state.
  void (*stop)(void);
} lupine_checkpoint_provider_v1;

typedef const lupine_checkpoint_provider_v1 *(
    *lupine_checkpoint_provider_get_v1_fn)(void);

#ifdef __cplusplus
}
#endif
