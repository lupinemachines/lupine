#ifndef LUPINE_HANDLES_H
#define LUPINE_HANDLES_H

#include <cstdint>

// Client-allocated stand-ins for handles the server has not returned yet. A
// wrapper mints one, hands it straight back to the application, and the real
// value is fulfilled later from the response; every wrapper that sends the
// handle onward resolves it back to the server's value first. The server never
// sees a synthetic value.
//
// Reserved-range invariant: a synthetic handle always carries
// LUPINE_HANDLE_TAG in its top 16 bits. That pattern is non-canonical on
// x86-64 (bits 63:47 are not all equal) and lies outside the AArch64 user and
// kernel ranges, so no pointer any server can hand back can land inside the
// reserved range. lupine_handle_is_synthetic therefore never misclassifies a
// real handle, and mint never collides with one.
static constexpr uint64_t LUPINE_HANDLE_TAG = 0xC0DEull;
static constexpr unsigned LUPINE_HANDLE_TAG_SHIFT = 48;
static constexpr uint64_t LUPINE_HANDLE_TAG_MASK = 0xFFFFull
                                                   << LUPINE_HANDLE_TAG_SHIFT;

enum lupine_handle_family : uint32_t {
  LUPINE_HANDLE_EVENT = 0,
  LUPINE_HANDLE_FAMILY_COUNT,
};

// Allocates an unfulfilled synthetic handle in the given family.
extern "C" uintptr_t lupine_handle_mint(lupine_handle_family family);

// Publishes the server's handle for a synthetic. A real of 0 marks the
// creation as failed; resolve then yields 0 and callers report
// CUDA_ERROR_INVALID_HANDLE at the point of use.
extern "C" void lupine_handle_fulfill(lupine_handle_family family,
                                      uintptr_t synthetic, uintptr_t real);

// Returns the server's handle, blocking until it is fulfilled. Values outside
// the reserved range pass through unchanged; forgotten or never-minted
// synthetics resolve to 0.
extern "C" uintptr_t lupine_handle_resolve(lupine_handle_family family,
                                           uintptr_t synthetic);

// Non-blocking resolve. Returns false when the synthetic exists but has not
// been fulfilled yet.
extern "C" bool lupine_handle_try_resolve(lupine_handle_family family,
                                          uintptr_t synthetic, uintptr_t *real);

// Drops a synthetic. Anyone still blocked in resolve wakes and gets 0.
extern "C" void lupine_handle_forget(lupine_handle_family family,
                                     uintptr_t synthetic);

extern "C" bool lupine_handle_is_synthetic(uintptr_t value);

#endif
