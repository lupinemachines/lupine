// Exercises remote cuLaunchKernelEx attribute forwarding.
// Auto-discovered by test/run_custom_tests.sh via the test_*.cu glob.
#include <cuda.h>

#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 11080
int main() {
  printf("SKIP: cuLaunchKernelEx requires CUDA 11.8 or newer\n");
  return 0;
}
#else

static const char kSetValuePtx[] =
    ".version 6.4\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry set_value(.param .u64 output, .param .u32 value)\n"
    "{\n"
    "  .reg .b64 %rd<2>;\n"
    "  .reg .b32 %r<2>;\n"
    "  ld.param.u64 %rd1, [output];\n"
    "  ld.param.u32 %r1, [value];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

static const char kClusterSetValuePtx[] =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    ".visible .entry set_value(.param .u64 output, .param .u32 value)\n"
    "{\n"
    "  .reg .b64 %rd<2>;\n"
    "  .reg .b32 %r<2>;\n"
    "  ld.param.u64 %rd1, [output];\n"
    "  ld.param.u32 %r1, [value];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

static const char kProgrammaticDependencyPtx[] =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    ".visible .entry producer(.param .u64 output, .param .u32 value)\n"
    "{\n"
    "  .reg .b64 %rd<2>;\n"
    "  .reg .b32 %r<2>;\n"
    "  ld.param.u64 %rd1, [output];\n"
    "  ld.param.u32 %r1, [value];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  griddepcontrol.launch_dependents;\n"
    "  ret;\n"
    "}\n"
    ".visible .entry dependent(.param .u64 output, .param .u32 value)\n"
    "{\n"
    "  .reg .b64 %rd<2>;\n"
    "  .reg .b32 %r<2>;\n"
    "  griddepcontrol.wait;\n"
    "  ld.param.u64 %rd1, [output];\n"
    "  ld.param.u32 %r1, [value];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

static std::string many_parameter_ptx() {
  std::string ptx = ".version 6.4\n"
                    ".target sm_52\n"
                    ".address_size 64\n"
                    ".visible .entry many_params(\n"
                    "  .param .u64 output";
  for (int i = 1; i <= 64; ++i) {
    ptx += ",\n  .param .u32 value" + std::to_string(i);
  }
  ptx += "\n)\n"
         "{\n"
         "  .reg .b64 %rd<2>;\n"
         "  .reg .b32 %r<2>;\n"
         "  ld.param.u64 %rd1, [output];\n"
         "  ld.param.u32 %r1, [value64];\n"
         "  st.global.u32 [%rd1], %r1;\n"
         "  ret;\n"
         "}\n";
  return ptx;
}

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

static CUlaunchConfig config_for(CUlaunchAttribute *attribute) {
  CUlaunchConfig config = {};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.attrs = attribute;
  config.numAttrs = attribute == nullptr ? 0 : 1;
  return config;
}

static bool output_is(CUdeviceptr output, uint32_t expected) {
  uint32_t actual = UINT32_MAX;
  return check(cuMemcpyDtoH(&actual, output, sizeof(actual)), "cuMemcpyDtoH") &&
         actual == expected;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUdeviceptr output = 0;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuModuleLoadData(&module, kSetValuePtx), "cuModuleLoadData") ||
      !check(cuModuleGetFunction(&function, module, "set_value"),
             "cuModuleGetFunction") ||
      !check(cuMemAlloc(&output, sizeof(uint32_t)), "cuMemAlloc")) {
    return 1;
  }

  uint32_t zero = 0;
  uint32_t value = 42;
  void *params[] = {&output, &value};

  CUlaunchConfig legacy_config = config_for(nullptr);
  if (!check(cuLaunchKernelEx(&legacy_config, function, params, nullptr),
             "cuLaunchKernelEx(no attributes)") ||
      !check(cuCtxSynchronize(), "sync no-attribute launch") ||
      !output_is(output, value)) {
    return 1;
  }

  CUmodule many_module = nullptr;
  CUfunction many_function = nullptr;
  std::string many_ptx = many_parameter_ptx();
  uint32_t many_values[64] = {};
  std::vector<void *> many_params(65);
  many_params[0] = &output;
  for (uint32_t i = 0; i < 64; ++i) {
    many_values[i] = i + 1;
    many_params[i + 1] = &many_values[i];
  }
  if (!check(cuModuleLoadData(&many_module, many_ptx.c_str()),
             "cuModuleLoadData(65 params)") ||
      !check(cuModuleGetFunction(&many_function, many_module, "many_params"),
             "cuModuleGetFunction(65 params)") ||
      !check(cuLaunchKernelEx(&legacy_config, many_function, many_params.data(),
                              nullptr),
             "cuLaunchKernelEx(65 params)") ||
      !check(cuCtxSynchronize(), "sync 65-param launch") ||
      !output_is(output, many_values[63])) {
    return 1;
  }
  cuModuleUnload(many_module);

  // A zero cluster dimension is invalid. The old wrapper silently discarded
  // it, launched set_value through cuLaunchKernel, and returned success.
  CUlaunchAttribute invalid_cluster = {};
  invalid_cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  invalid_cluster.value.clusterDim.x = 0;
  invalid_cluster.value.clusterDim.y = 1;
  invalid_cluster.value.clusterDim.z = 1;
  CUlaunchConfig config = config_for(&invalid_cluster);
  if (!check(cuMemcpyHtoD(output, &zero, sizeof(zero)), "reset output")) {
    return 1;
  }
  CUresult result = cuLaunchKernelEx(&config, function, params, nullptr);
  printf("invalid cluster result: %s (%d)\n", error_name(result),
         static_cast<int>(result));
  if (result == CUDA_SUCCESS || !check(cuCtxSynchronize(), "sync invalid") ||
      !output_is(output, 0)) {
    fprintf(stderr,
            "invalid cluster attribute was not rejected before launch: %s "
            "(%d)\n",
            error_name(result), static_cast<int>(result));
    return 1;
  }

  // Native attributes are forwarded without an ID whitelist. The server's
  // CUDA driver owns validation of IDs it does not recognize.
  CUlaunchAttribute unknown = {};
  unknown.id = static_cast<CUlaunchAttributeID>(15);
  config = config_for(&unknown);
  result = cuLaunchKernelEx(&config, function, params, nullptr);
  printf("unknown attribute result: %s (%d)\n", error_name(result),
         static_cast<int>(result));
  if (result == CUDA_SUCCESS || !output_is(output, 0)) {
    fprintf(stderr, "CUDA accepted unknown launch attribute ID 15\n");
    return 1;
  }

  // Preserve a null attribute pointer so the server-side CUDA driver owns
  // validation of the complete launch config.
  config = config_for(nullptr);
  config.numAttrs = 1;
  result = cuLaunchKernelEx(&config, function, params, nullptr);
  if (result == CUDA_SUCCESS || !output_is(output, 0)) {
    fprintf(stderr, "CUDA accepted a null launch attribute array\n");
    return 1;
  }

  int cluster_launch = 0;
  if (!check(cuDeviceGetAttribute(&cluster_launch,
                                  CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, device),
             "cuDeviceGetAttribute(cluster launch)")) {
    return 1;
  }
  printf("cluster launch support: %d\n", cluster_launch);
  if (cluster_launch != 0) {
    CUmodule cluster_module = nullptr;
    CUfunction cluster_function = nullptr;
    CUlaunchAttribute cluster = {};
    cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster.value.clusterDim.x = 2;
    cluster.value.clusterDim.y = 1;
    cluster.value.clusterDim.z = 1;
    config = config_for(&cluster);
    config.gridDimX = 2;
    if (!check(cuModuleLoadData(&cluster_module, kClusterSetValuePtx),
               "cuModuleLoadData(cluster)") ||
        !check(
            cuModuleGetFunction(&cluster_function, cluster_module, "set_value"),
            "cuModuleGetFunction(cluster)") ||
        !check(cuLaunchKernelEx(&config, cluster_function, params, nullptr),
               "cuLaunchKernelEx(cluster dimension)") ||
        !check(cuCtxSynchronize(), "sync cluster launch") ||
        !output_is(output, value)) {
      return 1;
    }
    cuModuleUnload(cluster_module);
  } else {
    printf("SKIP: cluster launch is not supported by this device\n");
  }

  // Programmatic dependent launch is a Hopper feature. Exercise the supported
  // attribute on capable hardware; older devices still run the deterministic
  // rejection checks above.
  if (cluster_launch != 0) {
    CUmodule dependency_module = nullptr;
    CUfunction producer = nullptr;
    CUfunction dependent = nullptr;
    CUlaunchAttribute programmatic = {};
    programmatic.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    programmatic.value.programmaticStreamSerializationAllowed = 1;
    config = config_for(&programmatic);
    uint32_t producer_value = 41;
    uint32_t dependent_value = 43;
    void *producer_params[] = {&output, &producer_value};
    void *dependent_params[] = {&output, &dependent_value};
    if (!check(cuModuleLoadData(&dependency_module, kProgrammaticDependencyPtx),
               "cuModuleLoadData(programmatic dependency)") ||
        !check(cuModuleGetFunction(&producer, dependency_module, "producer"),
               "cuModuleGetFunction(producer)") ||
        !check(cuModuleGetFunction(&dependent, dependency_module, "dependent"),
               "cuModuleGetFunction(dependent)") ||
        !check(cuLaunchKernel(producer, 1, 1, 1, 1, 1, 1, 0, nullptr,
                              producer_params, nullptr),
               "cuLaunchKernel(producer)") ||
        !check(cuLaunchKernelEx(&config, dependent, dependent_params, nullptr),
               "cuLaunchKernelEx(programmatic dependency)") ||
        !check(cuCtxSynchronize(), "sync programmatic dependency") ||
        !output_is(output, dependent_value)) {
      return 1;
    }
    cuModuleUnload(dependency_module);
  } else {
    printf("SKIP: programmatic dependent launch is not supported by this "
           "device\n");
  }

  cuMemFree(output);
  cuModuleUnload(module);
  printf("PASS: cuLaunchKernelEx forwards launch attributes\n");
  return 0;
}
#endif
