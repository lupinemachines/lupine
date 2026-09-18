// The context read a runtime call leaves pending costs a round trip. It is
// only needed while the client does not already know a non-null context for
// the lane, so steady-state traffic must stop paying for it - and each of the
// three runtime rebinders must start paying again.
//
// Each phase runs in a child process with LUPINE_RPC_STATS pointed at a file;
// the parent reads back how many cuCtxGetCurrent requests that phase sent.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

// codegen/gen_rpc_ids.h: RPC ids are stable CRC32 hashes of their op names.
static const int kCuCtxGetCurrentOp = 1159499964;

static const char *kPhaseVariable = "LUPINE_CTX_RPC_PHASE";
static const char *kStatsVariable = "LUPINE_RPC_STATS";

static int failures = 0;

static void require_fatal(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

// ---------------------------------------------------------------------------
// Phases, run in the child
// ---------------------------------------------------------------------------

static const int kRounds = 100;

static void runtime_traffic() {
  void *buffer = nullptr;
  require_fatal(cudaMalloc(&buffer, 256) == cudaSuccess, "cudaMalloc failed");
  require_fatal(cudaMemset(buffer, 0, 256) == cudaSuccess, "cudaMemset failed");
  require_fatal(cudaFree(buffer) == cudaSuccess, "cudaFree failed");
}

static CUcontext read_context() {
  CUcontext context = nullptr;
  require_fatal(cuCtxGetCurrent(&context) == CUDA_SUCCESS,
                "cuCtxGetCurrent failed");
  return context;
}

static int device_count() {
  int count = 0;
  require_fatal(cudaGetDeviceCount(&count) == cudaSuccess && count > 0,
                "cudaGetDeviceCount failed");
  return count;
}

// Nothing rebinds the lane, so after the first read the client already knows
// the context and every later read is free.
static int phase_steady() {
  require_fatal(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  runtime_traffic();
  CUcontext first = read_context();
  require_fatal(first != nullptr, "no context bound");
  for (int i = 0; i < kRounds; ++i) {
    runtime_traffic();
    require_fatal(read_context() == first, "the context moved on its own");
  }
  return 0;
}

static int phase_set_device() {
  const int count = device_count();
  if (count < 2) {
    std::printf("note: one device only; cudaSetDevice cannot rebind\n");
    return 0;
  }
  for (int i = 0; i < kRounds; ++i) {
    require_fatal(cudaSetDevice(i % count) == cudaSuccess,
                  "cudaSetDevice failed");
    runtime_traffic();
    require_fatal(read_context() != nullptr, "no context after cudaSetDevice");
  }
  return 0;
}

static int phase_set_valid_devices() {
  const int count = device_count();
  for (int i = 0; i < kRounds; ++i) {
    int preference[2] = {i % count, (i + 1) % count};
    if (cudaSetValidDevices(preference, count > 1 ? 2 : 1) != cudaSuccess) {
      std::printf("note: cudaSetValidDevices rejected here\n");
      return 0;
    }
    runtime_traffic();
    require_fatal(read_context() != nullptr,
                  "no context after cudaSetValidDevices");
  }
  return 0;
}

static int phase_device_reset() {
  require_fatal(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  for (int i = 0; i < kRounds; ++i) {
    runtime_traffic();
    require_fatal(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset failed");
    runtime_traffic();
    require_fatal(read_context() != nullptr, "no context after a reset");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// The parent
// ---------------------------------------------------------------------------

static long requests_for_op(const char *path, int op) {
  FILE *file = std::fopen(path, "r");
  if (file == nullptr) {
    std::fprintf(stderr, "FAIL: no RPC statistics at %s: %s\n", path,
                 std::strerror(errno));
    ++failures;
    return -1;
  }
  long total = -1;
  int recorded_op = 0;
  unsigned long long count = 0;
  unsigned long long wait_ns = 0;
  while (std::fscanf(file, "%d\t%llu\t%llu\n", &recorded_op, &count,
                     &wait_ns) == 3) {
    if (recorded_op == op) {
      total = static_cast<long>(count);
    }
  }
  std::fclose(file);
  return total < 0 ? 0 : total;
}

static long run_phase(const char *self, const char *phase) {
  std::string stats = std::string("/tmp/lupine-ctx-rpc-count-") + phase + "-" +
                      std::to_string(getpid()) + ".tsv";
  std::remove(stats.c_str());
  pid_t child = fork();
  require_fatal(child >= 0, "fork failed");
  if (child == 0) {
    setenv(kPhaseVariable, phase, 1);
    setenv(kStatsVariable, stats.c_str(), 1);
    execl(self, self, static_cast<char *>(nullptr));
    std::perror("execl");
    _exit(127);
  }
  int status = 0;
  require_fatal(waitpid(child, &status, 0) == child, "waitpid failed");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "FAIL: phase %s exited with status %d\n", phase,
                 status);
    ++failures;
    return -1;
  }
  long requests = requests_for_op(stats.c_str(), kCuCtxGetCurrentOp);
  std::remove(stats.c_str());
  std::printf("phase %s: cuCtxGetCurrent requests=%ld\n", phase, requests);
  return requests;
}

// A rebinder must not be skipped over: each round has to go and ask again.
static void expect_rebinds(const char *phase, long requests) {
  if (requests < 0) {
    return;
  }
  if (requests == 0) {
    std::printf("note: %s sent no requests; the phase reported itself skipped\n",
                phase);
    return;
  }
  if (requests < kRounds) {
    std::fprintf(stderr,
                 "FAIL: %s sent %ld cuCtxGetCurrent requests for %d rebinds; "
                 "the read was skipped across a rebinding\n",
                 phase, requests, kRounds);
    ++failures;
  }
}

int main() {
  const char *phase = getenv(kPhaseVariable);
  if (phase != nullptr) {
    if (std::strcmp(phase, "steady") == 0) {
      return phase_steady();
    }
    if (std::strcmp(phase, "set-device") == 0) {
      return phase_set_device();
    }
    if (std::strcmp(phase, "set-valid-devices") == 0) {
      return phase_set_valid_devices();
    }
    if (std::strcmp(phase, "device-reset") == 0) {
      return phase_device_reset();
    }
    std::fprintf(stderr, "unknown phase %s\n", phase);
    return 1;
  }

  char self[4096];
  ssize_t length = readlink("/proc/self/exe", self, sizeof(self) - 1);
  require_fatal(length > 0, "cannot find this executable");
  self[length] = '\0';

  // The first read on a lane is unavoidable: the context handle is the
  // server's, and nothing the client did predicts it. A handful more covers
  // the connection setup around it.
  const long steady_budget = 16;
  long steady = run_phase(self, "steady");
  if (steady > steady_budget) {
    std::fprintf(stderr,
                 "FAIL: steady traffic sent %ld cuCtxGetCurrent requests over "
                 "%d rounds (budget %ld)\n",
                 steady, kRounds, steady_budget);
    ++failures;
  }

  expect_rebinds("set-device", run_phase(self, "set-device"));
  expect_rebinds("set-valid-devices", run_phase(self, "set-valid-devices"));
  expect_rebinds("device-reset", run_phase(self, "device-reset"));

  std::printf(failures == 0 ? "PASSED\n" : "FAILED\n");
  return failures == 0 ? 0 : 1;
}
