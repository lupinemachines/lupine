// Exercises kernel parameter layout metadata identity and lifetime rules.
// Auto-discovered by test/run_custom_tests.sh via the test_*.cu glob.
#include <cuda.h>

#include <stdint.h>
#include <stdio.h>

#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

static const char kOneValuePtx[] = ".version 6.4\n"
                                   ".target sm_52\n"
                                   ".address_size 64\n"
                                   "\n"
                                   ".visible .entry same_name(\n"
                                   "    .param .u64 output,\n"
                                   "    .param .u64 value)\n"
                                   "{\n"
                                   "    .reg .b64 %rd<3>;\n"
                                   "    ld.param.u64 %rd0, [output];\n"
                                   "    ld.param.u64 %rd1, [value];\n"
                                   "    st.global.u64 [%rd0], %rd1;\n"
                                   "    ret;\n"
                                   "}\n";

static const char kTwoValuePtx[] = ".version 6.4\n"
                                   ".target sm_52\n"
                                   ".address_size 64\n"
                                   "\n"
                                   ".visible .entry same_name(\n"
                                   "    .param .u64 output,\n"
                                   "    .param .u64 first,\n"
                                   "    .param .u64 second)\n"
                                   "{\n"
                                   "    .reg .b64 %rd<5>;\n"
                                   "    ld.param.u64 %rd0, [output];\n"
                                   "    ld.param.u64 %rd1, [first];\n"
                                   "    ld.param.u64 %rd2, [second];\n"
                                   "    add.u64 %rd3, %rd1, %rd2;\n"
                                   "    st.global.u64 [%rd0], %rd3;\n"
                                   "    ret;\n"
                                   "}\n";

static const char *error_name(CUresult result) {
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  return name == nullptr ? "unknown" : name;
}

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  fprintf(stderr, "%s failed: %s (%d)\n", operation, error_name(result),
          static_cast<int>(result));
  return false;
}

static bool load_function(const char *ptx, CUmodule *module,
                          CUfunction *function) {
  return check(cuModuleLoadData(module, ptx), "cuModuleLoadData") &&
         check(cuModuleGetFunction(function, *module, "same_name"),
               "cuModuleGetFunction");
}

static bool read_output(CUdeviceptr output, uint64_t expected) {
  if (!check(cuCtxSynchronize(), "cuCtxSynchronize")) {
    return false;
  }
  uint64_t actual = 0;
  if (!check(cuMemcpyDtoH(&actual, output, sizeof(actual)), "cuMemcpyDtoH")) {
    return false;
  }
  if (actual != expected) {
    fprintf(stderr, "kernel output mismatch: expected %llu, got %llu\n",
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(actual));
    return false;
  }
  return true;
}

static bool launch_one(CUfunction function, CUdeviceptr output,
                       uint64_t value) {
  uint64_t zero = 0;
  if (!check(cuMemcpyHtoD(output, &zero, sizeof(zero)), "cuMemcpyHtoD")) {
    return false;
  }
  void *params[] = {&output, &value};
  return check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, params,
                              nullptr),
               "cuLaunchKernel(one)") &&
         read_output(output, value);
}

static bool launch_two(CUfunction function, CUdeviceptr output, uint64_t first,
                       uint64_t second) {
  uint64_t zero = 0;
  if (!check(cuMemcpyHtoD(output, &zero, sizeof(zero)), "cuMemcpyHtoD")) {
    return false;
  }
  void *params[] = {&output, &first, &second};
  return check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, params,
                              nullptr),
               "cuLaunchKernel(two)") &&
         read_output(output, first + second);
}

int main() {
  if (!check(cuInit(0), "cuInit")) {
    return 1;
  }
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (!check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return 1;
  }

  CUdeviceptr output = 0;
  if (!check(cuMemAlloc(&output, sizeof(uint64_t)), "cuMemAlloc")) {
    return 1;
  }

  // Same symbol name, different live modules and different layouts. Metadata
  // identity must be the function handle, not the symbol name.
  CUmodule one_module = nullptr;
  CUmodule two_module = nullptr;
  CUfunction one_function = nullptr;
  CUfunction two_function = nullptr;
  if (!load_function(kOneValuePtx, &one_module, &one_function) ||
      !load_function(kTwoValuePtx, &two_module, &two_function) ||
      !launch_one(one_function, output, 17) ||
      !launch_two(two_function, output, 19, 23) ||
      !check(cuModuleUnload(one_module), "cuModuleUnload(one)") ||
      !check(cuModuleUnload(two_module), "cuModuleUnload(two)")) {
    return 1;
  }

  // Alternate layouts across unload/reload cycles. CUDA commonly reuses the
  // opaque function value, so stale entries keyed only by that value are
  // exposed; correctness must not depend on whether reuse happens on a given
  // driver version. Launch from a persistent worker so replacing metadata on
  // this thread must be visible to launches from another thread.
  std::mutex worker_mutex;
  std::condition_variable worker_condition;
  CUfunction worker_function = nullptr;
  uint64_t worker_iteration = 0;
  bool worker_use_two = false;
  bool worker_ready = false;
  bool worker_context_ok = false;
  bool worker_pending = false;
  bool worker_done = false;
  bool worker_result = false;
  bool worker_stop = false;
  std::thread worker([&] {
    bool context_ok = check(cuCtxSetCurrent(context), "worker cuCtxSetCurrent");
    {
      std::lock_guard<std::mutex> lock(worker_mutex);
      worker_context_ok = context_ok;
      worker_ready = true;
    }
    worker_condition.notify_all();
    if (!context_ok) {
      return;
    }

    for (;;) {
      std::unique_lock<std::mutex> lock(worker_mutex);
      worker_condition.wait(lock,
                            [&] { return worker_pending || worker_stop; });
      if (worker_stop) {
        return;
      }
      CUfunction function = worker_function;
      uint64_t iteration = worker_iteration;
      bool use_two = worker_use_two;
      worker_pending = false;
      lock.unlock();

      bool launched = use_two ? launch_two(function, output, iteration, 100)
                              : launch_one(function, output, iteration);
      lock.lock();
      worker_result = launched;
      worker_done = true;
      lock.unlock();
      worker_condition.notify_all();
    }
  });
  {
    std::unique_lock<std::mutex> lock(worker_mutex);
    worker_condition.wait(lock, [&] { return worker_ready; });
  }
  if (!worker_context_ok) {
    worker.join();
    return 1;
  }

  auto launch_on_worker = [&](CUfunction function, bool use_two,
                              uint64_t iteration) {
    std::unique_lock<std::mutex> lock(worker_mutex);
    worker_function = function;
    worker_use_two = use_two;
    worker_iteration = iteration;
    worker_done = false;
    worker_pending = true;
    worker_condition.notify_all();
    worker_condition.wait(lock, [&] { return worker_done; });
    return worker_result;
  };

  std::map<uintptr_t, bool> previous_layouts;
  unsigned int cross_layout_reuses = 0;
  bool loop_ok = true;
  for (unsigned int iteration = 0; iteration < 32; ++iteration) {
    const bool use_two = (iteration & 1U) != 0;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!load_function(use_two ? kTwoValuePtx : kOneValuePtx, &module,
                       &function)) {
      loop_ok = false;
      break;
    }
    const uintptr_t function_value = reinterpret_cast<uintptr_t>(function);
    auto previous = previous_layouts.find(function_value);
    if (previous != previous_layouts.end() && previous->second != use_two) {
      ++cross_layout_reuses;
    }
    previous_layouts[function_value] = use_two;

    const bool launched = launch_on_worker(function, use_two, iteration);
    const bool unloaded = check(cuModuleUnload(module), "cuModuleUnload(loop)");
    if (!launched || !unloaded) {
      loop_ok = false;
      break;
    }
  }
  {
    std::lock_guard<std::mutex> lock(worker_mutex);
    worker_stop = true;
  }
  worker_condition.notify_all();
  worker.join();
  if (!loop_ok) {
    return 1;
  }

  if (!check(cuMemFree(output), "cuMemFree") ||
      !check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")) {
    return 1;
  }
  printf("PASS: layout metadata module identity/lifetime "
         "(cross-layout handle reuses: %u)\n",
         cross_layout_reuses);
  return 0;
}
