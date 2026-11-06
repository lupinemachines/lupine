// Preloaded into test clients only. Check the driver in the workload process,
// including runtime samples that load libcuda dynamically, before main runs.
#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

static bool same_file(const char *left, const char *right) {
  struct stat a {};
  struct stat b {};
  return left && right && stat(left, &a) == 0 && stat(right, &b) == 0 &&
         a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

__attribute__((constructor)) static void verify_driver() {
  const char *executable = std::getenv("LUPINE_TEST_EXECUTABLE");
  char actual[PATH_MAX]{};
  ssize_t size = readlink("/proc/self/exe", actual, sizeof(actual) - 1);
  // env/timeout/SSH helpers inherit LD_PRELOAD but cannot attest for the test.
  if (size < 0 || !same_file(actual, executable)) {
    return;
  }
  const char *expected = std::getenv("LUPINE_TEST_DRIVER");
  void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  const char *symbols[] = {"cuInit", "cuGetProcAddress"};
  for (const char *symbol : symbols) {
    Dl_info info{};
    void *address = driver ? dlsym(driver, symbol) : nullptr;
    if (!address || !dladdr(address, &info) ||
        !same_file(info.dli_fname, expected)) {
      std::fprintf(stderr,
                   "DRIVER GUARD FAILED: %s resolved to %s; expected %s\n",
                   symbol, info.dli_fname ? info.dli_fname : "<missing>",
                   expected ? expected : "<unset>");
      _exit(126);
    }
  }
  std::fprintf(stderr, "LUPINE_TEST_DRIVER_OK %s %s\n", executable, expected);
  std::fflush(stderr);
}
