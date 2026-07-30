#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

static const char *error_name(CUresult result) {
  const char *name = nullptr;
  return cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr
             ? name
             : "unknown";
}

static void check(CUresult result, const char *expr, int line) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 error_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static constexpr char kPrintfPtx[] = R"ptx(
.version 6.0
.target sm_52
.address_size 64

.extern .func (.param .b32 func_retval0) vprintf
(
    .param .b64 vprintf_param_0,
    .param .b64 vprintf_param_1
);

.global .align 1 .b8 message[25] = {
    108, 117, 112, 105, 110, 101, 45, 100, 101, 118, 105, 99,
    101, 45, 112, 114, 105, 110, 116, 102, 45, 37, 100, 10
};

.visible .entry print_token()
{
    .local .align 8 .b8 arguments[8];
    .reg .b32 result;
    .reg .b64 local_arguments;
    .reg .b64 generic_arguments;
    .reg .b64 global_address;
    .reg .b64 generic_message;

    mov.u64 local_arguments, arguments;
    cvta.local.u64 generic_arguments, local_arguments;
    mov.u32 result, 346;
    st.local.u32 [local_arguments], result;
    mov.u64 global_address, message;
    cvta.global.u64 generic_message, global_address;

    {
        .param .b64 message_param;
        .param .b64 arguments_param;
        .param .b32 return_param;
        st.param.b64 [message_param], generic_message;
        st.param.b64 [arguments_param], generic_arguments;
        call.uni (return_param), vprintf, (message_param, arguments_param);
    }
    ret;
}
)ptx";

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CHECK(cuDeviceGet(&device, 0));
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif
  CHECK(cuModuleLoadData(&module, kPrintfPtx));
  CHECK(cuModuleGetFunction(&function, module, "print_token"));

  FILE *captured = tmpfile();
  if (captured == nullptr) {
    std::perror("tmpfile");
    return EXIT_FAILURE;
  }
  int saved_stdout = dup(STDOUT_FILENO);
  if (saved_stdout < 0 || dup2(fileno(captured), STDOUT_FILENO) < 0) {
    std::perror("redirect stdout");
    return EXIT_FAILURE;
  }

  CHECK(
      cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  CHECK(cuCtxSynchronize());
  std::fflush(stdout);

  if (dup2(saved_stdout, STDOUT_FILENO) < 0) {
    std::perror("restore stdout");
    return EXIT_FAILURE;
  }
  close(saved_stdout);
  if (std::fseek(captured, 0, SEEK_SET) != 0) {
    std::perror("fseek");
    return EXIT_FAILURE;
  }

  std::string output;
  char buffer[256];
  while (size_t bytes = std::fread(buffer, 1, sizeof(buffer), captured)) {
    output.append(buffer, bytes);
  }
  std::fclose(captured);

  CHECK(cuModuleUnload(module));
  CHECK(cuCtxDestroy(context));

  if (output.find("lupine-device-printf-346") == std::string::npos) {
    std::fprintf(stderr,
                 "missing forwarded device printf; captured %zu bytes\n",
                 output.size());
    return EXIT_FAILURE;
  }

  std::printf("PASS: PTX vprintf capability enables stdout forwarding\n");
  return EXIT_SUCCESS;
}
