#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(CUresult result, const char *call) {
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(call) + " returned CUDA error " +
                             std::to_string(result));
  }
}
#define CUDA(call) check((call), #call)

static void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

static std::string uuid(CUdevice device) {
  CUuuid id{};
  CUDA(cuDeviceGetUuid(&id, device));
  std::ostringstream out;
  out << "GPU-" << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      out << '-';
    }
    out << std::setw(2)
        << static_cast<unsigned int>(static_cast<unsigned char>(id.bytes[i]));
  }
  return out.str();
}

static void inventory(const std::vector<CUdevice> &devices) {
  int driver = 0;
  CUDA(cuDriverGetVersion(&driver));
  std::cout << "{\"driver_version\":" << driver << ",\"devices\":[";
  for (size_t i = 0; i < devices.size(); ++i) {
    char name[256]{};
    CUDA(cuDeviceGetName(name, sizeof(name), devices[i]));
    std::cout << (i ? "," : "") << "{\"uuid\":" << std::quoted(uuid(devices[i]))
              << ",\"name\":" << std::quoted(name) << '}';
  }
  std::cout << "],\"peer_access\":[";
  for (size_t i = 0; i < devices.size(); ++i) {
    std::cout << (i ? ",[" : "[");
    for (size_t j = 0; j < devices.size(); ++j) {
      int access = 0;
      if (i != j) {
        CUDA(cuDeviceCanAccessPeer(&access, devices[i], devices[j]));
      }
      std::cout << (j ? "," : "") << access;
    }
    std::cout << ']';
  }
  std::cout << "]}\n";
}

static std::vector<unsigned int> pattern(int device) {
  std::vector<unsigned int> data(16384);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<unsigned int>(i * 2654435761U) ^ (device + 1);
  }
  return data;
}

static void memory(int ordinal) {
  auto input = pattern(ordinal);
  std::vector<unsigned int> output(input.size());
  size_t bytes = input.size() * sizeof(input[0]);
  CUdeviceptr allocation = 0;
  CUDA(cuMemAlloc(&allocation, bytes));
  CUDA(cuMemcpyHtoD(allocation, input.data(), bytes));
  CUDA(cuMemcpyDtoH(output.data(), allocation, bytes));
  require(input == output, "allocation round-trip mismatch");
  CUDA(cuMemFree(allocation));
}

static void streams(int ordinal) {
  auto input = pattern(ordinal);
  size_t bytes = input.size() * sizeof(input[0]);
  unsigned int *pinned = nullptr;
  CUDA(cuMemAllocHost(reinterpret_cast<void **>(&pinned), 2 * bytes));
  std::copy(input.begin(), input.end(), pinned);
  std::fill(pinned + input.size(), pinned + 2 * input.size(), 0);
  CUdeviceptr allocation = 0;
  CUstream producer = nullptr, consumer = nullptr;
  CUevent ready = nullptr;
  CUDA(cuMemAlloc(&allocation, bytes));
  CUDA(cuStreamCreate(&producer, CU_STREAM_NON_BLOCKING));
  CUDA(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING));
  CUDA(cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING));
  CUDA(cuMemcpyHtoDAsync(allocation, pinned, bytes, producer));
  CUDA(cuEventRecord(ready, producer));
  CUDA(cuStreamWaitEvent(consumer, ready, 0));
  CUDA(cuMemcpyDtoHAsync(pinned + input.size(), allocation, bytes, consumer));
  CUDA(cuStreamSynchronize(consumer));
  require(std::equal(input.begin(), input.end(), pinned + input.size()),
          "stream copy mismatch");
  CUDA(cuEventDestroy(ready));
  CUDA(cuStreamDestroy(consumer));
  CUDA(cuStreamDestroy(producer));
  CUDA(cuMemFree(allocation));
  CUDA(cuMemFreeHost(pinned));
}

static void kernel() {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUdeviceptr allocation = 0;
  CUDA(cuModuleLoad(&module, "topology_kernel.ptx"));
  CUDA(cuModuleGetFunction(&function, module, "topology_kernel"));
  CUDA(cuMemAlloc(&allocation, 256 * sizeof(unsigned int)));
  void *args[] = {&allocation};
  CUDA(cuLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
  CUDA(cuCtxSynchronize());
  std::vector<unsigned int> output(256);
  CUDA(cuMemcpyDtoH(output.data(), allocation,
                    output.size() * sizeof(output[0])));
  for (size_t i = 0; i < output.size(); ++i) {
    require(output[i] == i * 17 + 3,
            "kernel output mismatch at " + std::to_string(i));
  }
  CUDA(cuMemFree(allocation));
  CUDA(cuModuleUnload(module));
}

static void peer_copy(const std::vector<CUcontext> &contexts, int src, int dst,
                      bool async) {
  auto input = pattern(src);
  std::vector<unsigned int> output(input.size());
  size_t bytes = input.size() * sizeof(input[0]);
  CUdeviceptr source = 0, destination = 0;
  CUDA(cuCtxSetCurrent(contexts[src]));
  CUDA(cuMemAlloc(&source, bytes));
  CUDA(cuMemcpyHtoD(source, input.data(), bytes));
  CUDA(cuCtxSetCurrent(contexts[dst]));
  CUDA(cuMemAlloc(&destination, bytes));
  CUDA(cuMemsetD8(destination, 0, bytes));
  if (async) {
    CUstream stream = nullptr;
    CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUDA(cuMemcpyPeerAsync(destination, contexts[dst], source, contexts[src],
                           bytes, stream));
    CUDA(cuStreamSynchronize(stream));
    CUDA(cuStreamDestroy(stream));
  } else {
    CUDA(
        cuMemcpyPeer(destination, contexts[dst], source, contexts[src], bytes));
  }
  CUDA(cuMemcpyDtoH(output.data(), destination, bytes));
  require(input == output, "peer copy mismatch " + std::to_string(src) +
                               " -> " + std::to_string(dst));
  CUDA(cuMemFree(destination));
  CUDA(cuCtxSetCurrent(contexts[src]));
  CUDA(cuMemFree(source));
}

int main(int argc, char **argv) {
  try {
    require(argc >= 2,
            "usage: topology_test MODE [UUIDS [SRC DST EXPECTED_PEER]]");
    std::string mode = argv[1];
    CUDA(cuInit(0));
    int count = 0;
    CUDA(cuDeviceGetCount(&count));
    std::vector<CUdevice> devices(count);
    for (int i = 0; i < count; ++i) {
      CUDA(cuDeviceGet(&devices[i], i));
    }
    if (mode == "inventory") {
      inventory(devices);
      return 0;
    }
    require(argc >= 3, "expected UUID sequence is required");
    std::string actual;
    for (CUdevice device : devices) {
      actual += (actual.empty() ? "" : ",") + uuid(device);
    }
    require(actual == argv[2],
            "device order mismatch: got " + actual + ", want " + argv[2]);
    if (mode == "enumeration") {
      return 0;
    }
    std::vector<CUcontext> contexts(count);
    for (int i = 0; i < count; ++i) {
      CUDA(cuDevicePrimaryCtxRetain(&contexts[i], devices[i]));
    }
    if (mode.rfind("peer-", 0) == 0) {
      require(argc >= 5, "peer test needs source and destination ordinals");
      int src = std::stoi(argv[3]), dst = std::stoi(argv[4]);
      require(src >= 0 && src < count && dst >= 0 && dst < count && src != dst,
              "invalid peer device pair");
      if (mode == "peer-access") {
        require(argc == 6, "peer-access needs expected capability");
        int access = -1;
        CUDA(cuDeviceCanAccessPeer(&access, devices[src], devices[dst]));
        require(access == std::stoi(argv[5]), "peer capability mismatch");
        CUDA(cuCtxSetCurrent(contexts[src]));
        CUresult result = cuCtxEnablePeerAccess(contexts[dst], 0);
        if (access) {
          check(result, "cuCtxEnablePeerAccess");
          CUDA(cuCtxDisablePeerAccess(contexts[dst]));
        } else {
          require(result == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
                  "cuCtxEnablePeerAccess returned " + std::to_string(result) +
                      ", expected PEER_ACCESS_UNSUPPORTED");
        }
      } else {
        require(mode == "peer-copy" || mode == "peer-copy-async",
                "unknown peer test");
        peer_copy(contexts, src, dst, mode == "peer-copy-async");
      }
    } else {
      // Reverse order after creating contexts to exercise context switching.
      for (int i = count - 1; i >= 0; --i) {
        std::cout << "Running " << mode << " on device " << i << " ("
                  << uuid(devices[i]) << ")" << std::endl;
        CUDA(cuCtxSetCurrent(contexts[i]));
        CUdevice current = -1;
        CUDA(cuCtxGetDevice(&current));
        require(current == devices[i],
                "current context belongs to the wrong device");
        if (mode == "memory") {
          memory(i);
        } else if (mode == "streams") {
          streams(i);
        } else if (mode == "kernel") {
          kernel();
        } else {
          throw std::runtime_error("unknown test " + mode);
        }
      }
    }
    for (CUdevice device : devices) {
      CUDA(cuDevicePrimaryCtxRelease(device));
    }
    std::cout << "PASS " << mode << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
