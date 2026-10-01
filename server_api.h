#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __linux__
// Run one connection in the caller's isolated worker process. Owns the connected
// stream socket and returns 0 after clean disconnect, nonzero on failure.
// The caller manages listening, process isolation, and checkpoint/restore.
__attribute__((visibility("default")))
int lupine_server_serve_connection_v1(int connected_fd);
#endif

#ifdef __cplusplus
}
#endif
