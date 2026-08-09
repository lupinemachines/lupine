#include "handles.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace {

bool test_mint_produces_distinct_synthetics() {
  uintptr_t first = lupine_handle_mint(LUPINE_HANDLE_EVENT);
  uintptr_t second = lupine_handle_mint(LUPINE_HANDLE_EVENT);
  if (first == 0 || second == 0 || first == second) {
    std::cerr << "FAIL: mint did not produce distinct handles\n";
    return false;
  }
  if (!lupine_handle_is_synthetic(first) ||
      !lupine_handle_is_synthetic(second)) {
    std::cerr << "FAIL: minted handles are not tagged synthetic\n";
    return false;
  }
  lupine_handle_forget(LUPINE_HANDLE_EVENT, first);
  lupine_handle_forget(LUPINE_HANDLE_EVENT, second);
  return true;
}

bool test_is_synthetic_rejects_real_pointers() {
  void *heap = malloc(64);
  uintptr_t candidates[] = {
      0,
      1,
      reinterpret_cast<uintptr_t>(heap),
      reinterpret_cast<uintptr_t>(&candidates),
      reinterpret_cast<uintptr_t>(
          reinterpret_cast<void *>(&test_is_synthetic_rejects_real_pointers)),
      0x00007FFFFFFFFFFFull,
      0xFFFFFFFFFFFFFFFFull,
      0xFFFF800000000000ull,
  };
  bool ok = true;
  for (uintptr_t candidate : candidates) {
    if (lupine_handle_is_synthetic(candidate)) {
      std::cerr << "FAIL: real-looking value " << std::hex << candidate
                << " classified synthetic\n";
      ok = false;
    }
  }
  free(heap);
  return ok;
}

bool test_try_resolve_before_and_after_fulfill() {
  uintptr_t synthetic = lupine_handle_mint(LUPINE_HANDLE_EVENT);
  uintptr_t real = 0;
  if (lupine_handle_try_resolve(LUPINE_HANDLE_EVENT, synthetic, &real)) {
    std::cerr << "FAIL: try_resolve succeeded before fulfill\n";
    return false;
  }
  lupine_handle_fulfill(LUPINE_HANDLE_EVENT, synthetic, 0xABCD1234);
  if (!lupine_handle_try_resolve(LUPINE_HANDLE_EVENT, synthetic, &real) ||
      real != 0xABCD1234) {
    std::cerr << "FAIL: try_resolve did not return the fulfilled handle\n";
    return false;
  }
  if (lupine_handle_resolve(LUPINE_HANDLE_EVENT, synthetic) != 0xABCD1234) {
    std::cerr << "FAIL: resolve did not return the fulfilled handle\n";
    return false;
  }
  lupine_handle_forget(LUPINE_HANDLE_EVENT, synthetic);
  return true;
}

bool test_resolve_passes_through_real_values() {
  uintptr_t real = 0;
  if (lupine_handle_resolve(LUPINE_HANDLE_EVENT, 0x1234) != 0x1234 ||
      !lupine_handle_try_resolve(LUPINE_HANDLE_EVENT, 0x1234, &real) ||
      real != 0x1234) {
    std::cerr << "FAIL: real values did not pass through\n";
    return false;
  }
  return true;
}

bool test_resolve_blocks_until_fulfilled() {
  constexpr int kHandles = 16;
  std::vector<uintptr_t> synthetics;
  for (int i = 0; i < kHandles; ++i) {
    synthetics.push_back(lupine_handle_mint(LUPINE_HANDLE_EVENT));
  }

  std::vector<uintptr_t> resolved(kHandles, 0);
  std::vector<std::thread> readers;
  for (int i = 0; i < kHandles; ++i) {
    readers.emplace_back([&, i] {
      resolved[i] = lupine_handle_resolve(LUPINE_HANDLE_EVENT, synthetics[i]);
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (int i = 0; i < kHandles; ++i) {
    lupine_handle_fulfill(LUPINE_HANDLE_EVENT, synthetics[i],
                          0x1000 + static_cast<uintptr_t>(i));
  }
  for (auto &reader : readers) {
    reader.join();
  }

  for (int i = 0; i < kHandles; ++i) {
    if (resolved[i] != 0x1000 + static_cast<uintptr_t>(i)) {
      std::cerr << "FAIL: resolve returned the wrong handle\n";
      return false;
    }
    lupine_handle_forget(LUPINE_HANDLE_EVENT, synthetics[i]);
  }
  return true;
}

bool test_creation_failure_resolves_to_zero() {
  uintptr_t synthetic = lupine_handle_mint(LUPINE_HANDLE_EVENT);
  std::thread fulfiller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lupine_handle_fulfill(LUPINE_HANDLE_EVENT, synthetic, 0);
  });
  uintptr_t real = lupine_handle_resolve(LUPINE_HANDLE_EVENT, synthetic);
  fulfiller.join();
  lupine_handle_forget(LUPINE_HANDLE_EVENT, synthetic);
  if (real != 0) {
    std::cerr << "FAIL: failed creation did not resolve to zero\n";
    return false;
  }
  return true;
}

bool test_forget_releases_waiters() {
  uintptr_t synthetic = lupine_handle_mint(LUPINE_HANDLE_EVENT);
  std::thread forgetter([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lupine_handle_forget(LUPINE_HANDLE_EVENT, synthetic);
  });
  uintptr_t real = lupine_handle_resolve(LUPINE_HANDLE_EVENT, synthetic);
  forgetter.join();
  if (real != 0) {
    std::cerr << "FAIL: forget did not release the waiter with zero\n";
    return false;
  }
  uintptr_t ignored = 1;
  if (!lupine_handle_try_resolve(LUPINE_HANDLE_EVENT, synthetic, &ignored) ||
      ignored != 0) {
    std::cerr << "FAIL: forgotten handle did not resolve to zero\n";
    return false;
  }
  return true;
}

bool test_named_mint_is_stable_per_owner_and_name() {
  uintptr_t library = lupine_handle_mint(LUPINE_HANDLE_LIBRARY);
  uintptr_t other = lupine_handle_mint(LUPINE_HANDLE_LIBRARY);
  uintptr_t first =
      lupine_handle_mint_named(LUPINE_HANDLE_KERNEL, library, "kernel_a");
  uintptr_t again =
      lupine_handle_mint_named(LUPINE_HANDLE_KERNEL, library, "kernel_a");
  uintptr_t sibling =
      lupine_handle_mint_named(LUPINE_HANDLE_KERNEL, library, "kernel_b");
  uintptr_t elsewhere =
      lupine_handle_mint_named(LUPINE_HANDLE_KERNEL, other, "kernel_a");
  if (first == 0 || first != again) {
    std::cerr << "FAIL: repeated named mint did not return the same handle\n";
    return false;
  }
  if (first == sibling || first == elsewhere || sibling == elsewhere) {
    std::cerr << "FAIL: distinct names or owners shared a handle\n";
    return false;
  }
  if (lupine_handle_family_of(first) != LUPINE_HANDLE_KERNEL ||
      lupine_handle_family_of(library) != LUPINE_HANDLE_LIBRARY ||
      lupine_handle_family_of(0x1234) != LUPINE_HANDLE_FAMILY_COUNT) {
    std::cerr << "FAIL: family_of misread a handle\n";
    return false;
  }

  lupine_handle_fulfill(LUPINE_HANDLE_KERNEL, first, 0x5150);
  if (lupine_handle_resolve(LUPINE_HANDLE_KERNEL, again) != 0x5150) {
    std::cerr << "FAIL: named handle did not resolve to its real value\n";
    return false;
  }

  lupine_handle_forget_owned(LUPINE_HANDLE_KERNEL, library);
  uintptr_t reminted =
      lupine_handle_mint_named(LUPINE_HANDLE_KERNEL, library, "kernel_a");
  if (reminted == first || reminted == 0) {
    std::cerr << "FAIL: forget_owned did not drop the name index\n";
    return false;
  }
  uintptr_t survivor = 0;
  if (lupine_handle_try_resolve(LUPINE_HANDLE_KERNEL, elsewhere, &survivor)) {
    std::cerr << "FAIL: forget_owned reached another owner's handles\n";
    return false;
  }
  lupine_handle_forget_owned(LUPINE_HANDLE_KERNEL, library);
  lupine_handle_forget_owned(LUPINE_HANDLE_KERNEL, other);
  lupine_handle_forget(LUPINE_HANDLE_LIBRARY, library);
  lupine_handle_forget(LUPINE_HANDLE_LIBRARY, other);
  return true;
}

int applied_payloads = 0;

void count_applied_payload(uintptr_t, uintptr_t, void *payload, void *context) {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ++applied_payloads;
  *static_cast<int *>(context) = *static_cast<int *>(payload);
  delete static_cast<int *>(payload);
}

bool test_payload_applies_once_before_any_resolver_returns() {
  constexpr int kReaders = 8;
  uintptr_t synthetic = lupine_handle_mint(LUPINE_HANDLE_LIBRARY);
  applied_payloads = 0;
  int applied_value = 0;

  std::vector<uintptr_t> resolved(kReaders, 0);
  std::vector<int> seen(kReaders, -1);
  std::vector<std::thread> readers;
  for (int i = 0; i < kReaders; ++i) {
    readers.emplace_back([&, i] {
      resolved[i] =
          lupine_handle_resolve_applied(LUPINE_HANDLE_LIBRARY, synthetic,
                                        count_applied_payload, &applied_value);
      seen[i] = applied_payloads;
    });
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  lupine_handle_fulfill_payload(LUPINE_HANDLE_LIBRARY, synthetic, 0xFACE,
                                new int(77));
  for (auto &reader : readers) {
    reader.join();
  }
  lupine_handle_forget(LUPINE_HANDLE_LIBRARY, synthetic);

  if (applied_payloads != 1 || applied_value != 77) {
    std::cerr << "FAIL: payload was not applied exactly once\n";
    return false;
  }
  for (int i = 0; i < kReaders; ++i) {
    if (resolved[i] != 0xFACE || seen[i] != 1) {
      std::cerr << "FAIL: resolver returned before the payload was applied\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  if (!test_mint_produces_distinct_synthetics() ||
      !test_is_synthetic_rejects_real_pointers() ||
      !test_try_resolve_before_and_after_fulfill() ||
      !test_resolve_passes_through_real_values() ||
      !test_resolve_blocks_until_fulfilled() ||
      !test_creation_failure_resolves_to_zero() ||
      !test_forget_releases_waiters() ||
      !test_named_mint_is_stable_per_owner_and_name() ||
      !test_payload_applies_once_before_any_resolver_returns()) {
    return 1;
  }
  std::cout << "handle table tests passed\n";
  return 0;
}
