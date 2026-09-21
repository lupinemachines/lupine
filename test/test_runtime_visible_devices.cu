#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

extern "C" CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static std::string uuid_string(const char *bytes) {
  std::string uuid = "GPU-";
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      uuid += '-';
    }
    char hex[3];
    std::snprintf(hex, sizeof(hex), "%02x",
                  static_cast<unsigned char>(bytes[i]));
    uuid += hex;
  }
  return uuid;
}

static void check_binding(int expected) {
  // A driver context invalidates the runtime's cached device answer. Reading
  // it and selecting that device again is the sequence used by autograd's
  // worker threads, and must stay in the client's filtered ordinal space.
  int actual = -1;
  require(cudaGetDevice(&actual) == cudaSuccess && actual == expected,
          "cudaGetDevice returned a server ordinal");
  require(cudaSetDevice(actual) == cudaSuccess,
          "cudaSetDevice rejected the current visible device");
  CUdevice driver_device = -1;
  require(cuCtxGetDevice(&driver_device) == CUDA_SUCCESS &&
              driver_device == expected,
          "cudaSetDevice bound a different device from the driver");
}

static void check_visible_devices(int expected_count, char **expected_uuids) {
  // Runtime discovery must initialize the driver without an explicit cuInit.
  int runtime_count = -1;
  require(cudaGetDeviceCount(&runtime_count) == cudaSuccess &&
              runtime_count == expected_count,
          "runtime device count does not match the filter");
  int driver_count = -1;
  require(cuDeviceGetCount(&driver_count) == CUDA_SUCCESS &&
              driver_count == expected_count,
          "driver device count does not match the filter");
  if (expected_count > 0) {
    check_binding(0);
    std::thread default_worker([] { check_binding(0); });
    default_worker.join();
  }
  require(cudaSetDevice(expected_count) == cudaErrorInvalidDevice,
          "cudaSetDevice accepted an ordinal outside the visible table");

  for (int ordinal = 0; ordinal < expected_count; ++ordinal) {
    cudaDeviceProp properties{};
    require(cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess,
            "cudaGetDeviceProperties failed");
    require(uuid_string(properties.uuid.bytes) == expected_uuids[ordinal],
            "runtime device properties name the wrong GPU");

    require(cudaSetDevice(ordinal) == cudaSuccess, "cudaSetDevice failed");
    check_binding(ordinal);

    CUdevice device = -1;
    require(cuDeviceGet(&device, ordinal) == CUDA_SUCCESS,
            "cuDeviceGet failed");
    CUcontext context = nullptr;
    require(cuCtxCreate_v2(&context, 0, device) == CUDA_SUCCESS,
            "cuCtxCreate failed");
    check_binding(ordinal);
    std::thread worker([context, ordinal] {
      require(cuCtxSetCurrent(context) == CUDA_SUCCESS,
              "worker cuCtxSetCurrent failed");
      check_binding(ordinal);
      CUcontext current = nullptr;
      require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr,
              "worker current context is missing");
      void *buffer = nullptr;
      require(cudaMalloc(&buffer, 256) == cudaSuccess, "cudaMalloc failed");
      CUcontext allocation_context = nullptr;
      require(cuPointerGetAttribute(
                  &allocation_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                  reinterpret_cast<CUdeviceptr>(buffer)) == CUDA_SUCCESS &&
                  allocation_context == current,
              "runtime allocation landed on the wrong device");
      require(cudaFree(buffer) == cudaSuccess, "cudaFree failed");
      require(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS,
              "worker context detach failed");
    });
    worker.join();
    require(cuCtxDestroy(context) == CUDA_SUCCESS, "cuCtxDestroy failed");
  }
}

static void run_filtered(const char *exe, const std::string &filter,
                         const std::vector<std::string> &expected) {
  std::vector<char *> args{const_cast<char *>(exe),
                           const_cast<char *>("--filtered")};
  for (const auto &uuid : expected) {
    args.push_back(const_cast<char *>(uuid.c_str()));
  }
  args.push_back(nullptr);
  std::printf("checking CUDA_VISIBLE_DEVICES=%s\n", filter.c_str());
  std::fflush(stdout);
  const pid_t child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    // Exec gives each filter a fresh device table and fresh RPC connections.
    if (setenv("CUDA_VISIBLE_DEVICES", filter.c_str(), 1) == 0) {
      execv(exe, args.data());
    }
    _exit(127);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child, "waitpid failed");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "filtered runtime device test failed");
}

int main(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "--filtered") {
    check_visible_devices(argc - 2, argv + 2);
    return 0;
  }

  require(cuInit(0) == CUDA_SUCCESS, "cuInit failed");
  int count = 0;
  require(cuDeviceGetCount(&count) == CUDA_SUCCESS, "cuDeviceGetCount failed");
  if (count < 2) {
    std::printf("SKIP: visible device remapping needs two server GPUs\n");
    return 0;
  }
  std::vector<std::string> uuids;
  for (int ordinal = 0; ordinal < 2; ++ordinal) {
    CUdevice device = -1;
    CUuuid uuid{};
    require(cuDeviceGet(&device, ordinal) == CUDA_SUCCESS,
            "cuDeviceGet failed");
    require(cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS,
            "cuDeviceGetUuid failed");
    uuids.push_back(uuid_string(uuid.bytes));
  }
  run_filtered(argv[0], uuids[1], {uuids[1]});
  run_filtered(argv[0], uuids[1] + "," + uuids[0], {uuids[1], uuids[0]});
  run_filtered(argv[0], "1", {uuids[1]});
  run_filtered(argv[0], "", {});
  std::printf("runtime visible devices test passed\n");
  return 0;
}
