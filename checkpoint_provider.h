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

  // Restores the named connection before its first CUDA RPC is dispatched.
  // A missing checkpoint is success; malformed or unrestorable state fails the
  // connection rather than allowing it to continue with empty GPU memory.
  int (*restore)(const char *connection_id);

  // Writes a complete checkpoint for the current connection. The identifier
  // is null when the client did not supply one. The provider owns storage,
  // file layout, and any fallback policy for unnamed connections.
  int (*checkpoint)(const char *connection_id);

  // Stops observation and releases provider-owned process state.
  void (*stop)(void);

  // Optional tail extension. A provider that already traces this process can
  // capture the successful native RM allocation made by operation(argument).
  // Return ENOSYS without calling operation when capture is unavailable.
  // Other failures must never cause the operation to be retried.
  int (*capture_rm_alloc)(int (*operation)(void *), void *argument, int *rm_fd,
                          uint32_t *h_client, uint32_t *h_memory);
} lupine_checkpoint_provider_v1;

typedef const lupine_checkpoint_provider_v1 *(
    *lupine_checkpoint_provider_get_v1_fn)(void);

#ifdef __cplusplus
}
#endif
