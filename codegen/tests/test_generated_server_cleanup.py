import re
import subprocess
import tempfile
import unittest
from pathlib import Path


CODEGEN_DIR = Path(__file__).resolve().parents[1]
GENERATED_SERVER = CODEGEN_DIR / "gen_server.cpp"


def extract_function(source: str, name: str) -> str:
    start = source.index(f"int {name}(")
    opening_brace = source.index("{", start)
    depth = 0

    for offset in range(opening_brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]

    raise AssertionError(f"unterminated generated function: {name}")


class GeneratedServerCleanupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated_source = GENERATED_SERVER.read_text()
        cls.library_load_handler = extract_function(
            cls.generated_source, "handle_cuLibraryLoadFromFile"
        )

    def test_every_generated_malloc_is_scope_owned(self) -> None:
        allocations = re.findall(
            r"(\w+)_storage\.reset\(std::malloc\(", self.generated_source
        )
        self.assertGreater(len(allocations), 30)
        self.assertNotIn("free((void *)", self.generated_source)

        for name in allocations:
            self.assertRegex(
                self.generated_source,
                rf"{name}\s*=\s*static_cast<[^;]+>\({name}_storage\.get\(\)\);",
            )

    def test_success_and_early_errors_release_all_buffers(self) -> None:
        harness = f"""
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>

struct conn_t {{}};
using CUresult = int;
using CUlibrary = void *;
using CUjit_option = int;
using CUlibraryOption = int;

namespace {{
struct FreeDeleter {{
  void operator()(void *ptr) const noexcept {{ std::free(ptr); }}
}};
using MallocPtr = std::unique_ptr<void, FreeDeleter>;
}} // namespace

extern "C" void *__real_malloc(std::size_t size);
extern "C" void __real_free(void *ptr);

namespace {{
constexpr std::size_t kMaxTrackedAllocations = 64;
void *tracked_allocations[kMaxTrackedAllocations] = {{}};
std::size_t tracked_count = 0;
int allocation_calls = 0;
int fail_allocation_at = 0;
bool invalid_free = false;
bool tracking = false;
int rpc_calls = 0;
int read_calls = 0;
int fail_rpc_at = 0;

bool should_fail_rpc() {{
  ++rpc_calls;
  return fail_rpc_at != 0 && rpc_calls == fail_rpc_at;
}}

bool buffers_were_released() {{
  tracking = false;
  const bool clean = tracked_count == 0 && !invalid_free;
  for (std::size_t i = 0; i < tracked_count; ++i)
    __real_free(tracked_allocations[i]);
  tracked_count = 0;
  return clean;
}}
}} // namespace

extern "C" void *__wrap_malloc(std::size_t size) {{
  if (tracking) {{
    ++allocation_calls;
    if (fail_allocation_at != 0 && allocation_calls == fail_allocation_at)
      return nullptr;
  }}

  void *ptr = __real_malloc(size);
  if (tracking && ptr != nullptr) {{
    if (tracked_count == kMaxTrackedAllocations)
      std::abort();
    tracked_allocations[tracked_count++] = ptr;
  }}
  return ptr;
}}

extern "C" void __wrap_free(void *ptr) {{
  if (!tracking || ptr == nullptr) {{
    __real_free(ptr);
    return;
  }}

  auto *entry = std::find(tracked_allocations,
                          tracked_allocations + tracked_count, ptr);
  if (entry == tracked_allocations + tracked_count) {{
    invalid_free = true;
    return;
  }}

  *entry = tracked_allocations[--tracked_count];
  __real_free(ptr);
}}

int rpc_read(conn_t *, void *destination, std::size_t size) {{
  ++read_calls;
  if (should_fail_rpc())
    return -1;
  if (destination == nullptr)
    return -1;

  std::memset(destination, 0, size);
  if (read_calls == 1) {{
    const std::size_t file_name_length = 9;
    std::memcpy(destination, &file_name_length, sizeof(file_name_length));
  }} else if (read_calls == 2) {{
    const char file_name[] = "test.ptx";
    std::memcpy(destination, file_name, std::min(size, sizeof(file_name)));
  }} else if (read_calls == 3 || read_calls == 6) {{
    const unsigned int option_count = 2;
    std::memcpy(destination, &option_count, sizeof(option_count));
  }}
  return 0;
}}

int rpc_read_end(conn_t *) {{ return should_fail_rpc() ? -1 : 1; }}
int rpc_write_start_response(conn_t *, int) {{
  return should_fail_rpc() ? -1 : 0;
}}
int rpc_write(conn_t *, const void *, std::size_t) {{
  return should_fail_rpc() ? -1 : 0;
}}
int rpc_write_end(conn_t *) {{ return should_fail_rpc() ? -1 : 0; }}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *, CUjit_option *,
                               void **, unsigned int, CUlibraryOption *,
                               void **, unsigned int) {{
  *library = reinterpret_cast<void *>(1);
  return 0;
}}

{self.library_load_handler}

bool run_case(int rpc_failure, int allocation_failure, int expected_result) {{
  rpc_calls = 0;
  read_calls = 0;
  fail_rpc_at = rpc_failure;
  allocation_calls = 0;
  fail_allocation_at = allocation_failure;
  invalid_free = false;
  tracked_count = 0;
  tracking = true;

  conn_t conn;
  const int result = handle_cuLibraryLoadFromFile(&conn);
  return result == expected_result && buffers_were_released();
}}

int main() {{
  if (!run_case(0, 0, 0))
    return 1;
  if (allocation_calls != 5)
    return 4;

  for (int rpc_failure = 1; rpc_failure <= 13; ++rpc_failure) {{
    if (!run_case(rpc_failure, 0, -1))
      return 2;
  }}

  for (int allocation_failure = 1; allocation_failure <= 5;
       ++allocation_failure) {{
    if (!run_case(0, allocation_failure, -1))
      return 3;
  }}
  return 0;
}}
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "generated_cleanup_test.cpp"
            binary_path = Path(temp_dir) / "generated_cleanup_test"
            source_path.write_text(harness)
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O0",
                    str(source_path),
                    "-Wl,--wrap=malloc",
                    "-Wl,--wrap=free",
                    "-o",
                    str(binary_path),
                ],
                check=True,
            )
            subprocess.run([str(binary_path)], check=True)


if __name__ == "__main__":
    unittest.main()
