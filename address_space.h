#ifndef LUPINE_ADDRESS_SPACE_H
#define LUPINE_ADDRESS_SPACE_H

#include <cstddef>
#include <cstdint>

struct conn_t;

// Each connection owns a disjoint 1 TiB identity-mapped arena. The client
// also reserves a writable alias at a fixed displacement from that arena.
// Keeping the application/server address identical preserves pointer values
// embedded in mirrored memory; only transport writes use the alias.
static constexpr uintptr_t LUPINE_VA_FIRST_BASE = UINT64_C(0x700000000000);
static constexpr size_t LUPINE_VA_ARENA_SIZE = UINT64_C(0x010000000000);
static constexpr unsigned int LUPINE_VA_ARENA_COUNT = 8;
static constexpr intptr_t LUPINE_VA_WRITE_OFFSET = -INT64_C(0x200000000000);

// Returns 0 with an arena reserved, 1 when identity arenas are unsupported on
// this platform, and -1 when no candidate arena is available. min_slot lets a
// reconnect skip a range the peer rejected.
int lupine_va_reserve_client(conn_t *conn, unsigned int min_slot,
                             unsigned int *slot);

// Accepts the exact arena proposed by a client. The range is validated against
// the candidate band before any mapping is attempted.
int lupine_va_reserve_server(conn_t *conn, uintptr_t base, size_t size);

bool lupine_va_contains(const conn_t *conn, uintptr_t address, size_t size);
void lupine_va_destroy(conn_t *conn);

#endif
