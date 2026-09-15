// Exercises the cuFFT shim end to end against the remote device: plan
// creation in every style, the optional layout arrays, work area management,
// streams, the Xt entry points and multi-GPU descriptors on one device named
// twice and, when two GPUs are present, on both, each transform checked against
// a CPU DFT.
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <nvrtc.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUFFT(call)                                                      \
  do {                                                                         \
    cufftResult status = (call);                                               \
    if (status != CUFFT_SUCCESS) {                                             \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static std::vector<cufftComplex> random_signal(size_t n, unsigned seed) {
  std::vector<cufftComplex> signal(n);
  unsigned state = seed;
  for (auto &value : signal) {
    state = state * 1664525u + 1013904223u;
    value.x = static_cast<float>((state >> 8) % 1000) / 500.0f - 1.0f;
    state = state * 1664525u + 1013904223u;
    value.y = static_cast<float>((state >> 8) % 1000) / 500.0f - 1.0f;
  }
  return signal;
}

// Forward DFT of `batch` signals of `n` points, packed back to back.
static std::vector<cufftComplex>
reference_dft(const std::vector<cufftComplex> &in, int n, int batch) {
  std::vector<cufftComplex> out(in.size());
  for (int b = 0; b < batch; ++b) {
    for (int k = 0; k < n; ++k) {
      double re = 0.0, im = 0.0;
      for (int j = 0; j < n; ++j) {
        const double angle = -2.0 * M_PI * k * j / n;
        const cufftComplex x = in[static_cast<size_t>(b) * n + j];
        re += x.x * cos(angle) - x.y * sin(angle);
        im += x.x * sin(angle) + x.y * cos(angle);
      }
      out[static_cast<size_t>(b) * n + k] = {static_cast<float>(re),
                                             static_cast<float>(im)};
    }
  }
  return out;
}

static int compare(const cufftComplex *got,
                   const std::vector<cufftComplex> &want, size_t count,
                   const char *what) {
  for (size_t i = 0; i < count; ++i) {
    const double scale = 1.0 + std::fabs(want[i].x) + std::fabs(want[i].y);
    if (std::fabs(got[i].x - want[i].x) > 1e-3 * scale ||
        std::fabs(got[i].y - want[i].y) > 1e-3 * scale) {
      fprintf(stderr, "%s[%zu] = (%g, %g), expected (%g, %g)\n", what, i,
              got[i].x, got[i].y, want[i].x, want[i].y);
      return 1;
    }
  }
  return 0;
}

static int test_plan1d_c2c() {
  const int n = 64, batch = 2;
  const std::vector<cufftComplex> signal = random_signal(n * batch, 1);
  const std::vector<cufftComplex> spectrum = reference_dft(signal, n, batch);

  cufftHandle plan = 0;
  CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
  size_t work_size = 0;
  CHECK_CUFFT(cufftGetSize(plan, &work_size));
  size_t estimate = 0;
  CHECK_CUFFT(cufftEstimate1d(n, CUFFT_C2C, batch, &estimate));

  cufftComplex *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, signal.size() * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(device, signal.data(),
                        signal.size() * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice));
  CHECK_CUFFT(cufftExecC2C(plan, device, device, CUFFT_FORWARD));
  std::vector<cufftComplex> result(signal.size());
  CHECK_CUDA(cudaMemcpy(result.data(), device,
                        result.size() * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));
  if (compare(result.data(), spectrum, result.size(), "plan1d forward")) {
    return 1;
  }

  // The inverse transform is unnormalized: divide by n to get the signal back.
  CHECK_CUFFT(cufftExecC2C(plan, device, device, CUFFT_INVERSE));
  CHECK_CUDA(cudaMemcpy(result.data(), device,
                        result.size() * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));
  for (auto &value : result) {
    value.x /= n;
    value.y /= n;
  }
  if (compare(result.data(), signal, result.size(), "plan1d inverse")) {
    return 1;
  }
  CHECK_CUDA(cudaFree(device));
  CHECK_CUFFT(cufftDestroy(plan));
  return 0;
}

// cufftPlanMany with null embeds takes the optional-array path; the 64-bit
// plan carries explicit ones.
static int test_plan_many_r2c_c2r() {
  const int n = 32, batch = 3;
  std::vector<cufftReal> signal(static_cast<size_t>(n) * batch);
  for (size_t i = 0; i < signal.size(); ++i) {
    signal[i] = static_cast<float>((i * 7) % 11) / 5.0f - 1.0f;
  }
  std::vector<cufftComplex> complex_signal(signal.size());
  for (size_t i = 0; i < signal.size(); ++i) {
    complex_signal[i] = {signal[i], 0.0f};
  }
  const std::vector<cufftComplex> spectrum =
      reference_dft(complex_signal, n, batch);
  const int half = n / 2 + 1;

  cufftHandle forward = 0;
  int sizes[1] = {n};
  CHECK_CUFFT(cufftPlanMany(&forward, 1, sizes, nullptr, 1, n, nullptr, 1, half,
                            CUFFT_R2C, batch));

  cufftReal *device_real = nullptr;
  cufftComplex *device_complex = nullptr;
  CHECK_CUDA(cudaMalloc(&device_real, signal.size() * sizeof(cufftReal)));
  CHECK_CUDA(cudaMalloc(&device_complex, static_cast<size_t>(half) * batch *
                                             sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(device_real, signal.data(),
                        signal.size() * sizeof(cufftReal),
                        cudaMemcpyHostToDevice));
  CHECK_CUFFT(cufftExecR2C(forward, device_real, device_complex));
  std::vector<cufftComplex> result(static_cast<size_t>(half) * batch);
  CHECK_CUDA(cudaMemcpy(result.data(), device_complex,
                        result.size() * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));
  for (int b = 0; b < batch; ++b) {
    std::vector<cufftComplex> want(spectrum.begin() + b * n,
                                   spectrum.begin() + b * n + half);
    if (compare(result.data() + b * half, want, half, "plan many r2c")) {
      return 1;
    }
  }

  cufftHandle inverse = 0;
  CHECK_CUFFT(cufftCreate(&inverse));
  long long int sizes64[1] = {n};
  long long int inembed[1] = {half};
  long long int onembed[1] = {n};
  size_t work_size = 0;
  CHECK_CUFFT(cufftMakePlanMany64(inverse, 1, sizes64, inembed, 1, half,
                                  onembed, 1, n, CUFFT_C2R, batch, &work_size));
  CHECK_CUFFT(cufftExecC2R(inverse, device_complex, device_real));
  std::vector<cufftReal> round_trip(signal.size());
  CHECK_CUDA(cudaMemcpy(round_trip.data(), device_real,
                        round_trip.size() * sizeof(cufftReal),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < signal.size(); ++i) {
    if (std::fabs(round_trip[i] / n - signal[i]) > 1e-3) {
      fprintf(stderr, "plan many c2r[%zu] = %g, expected %g\n", i,
              round_trip[i] / n, signal[i]);
      return 1;
    }
  }

  size_t estimate = 0;
  CHECK_CUFFT(cufftEstimateMany(1, sizes, nullptr, 1, n, nullptr, 1, half,
                                CUFFT_R2C, batch, &estimate));
  CHECK_CUFFT(cufftGetSizeMany64(inverse, 1, sizes64, inembed, 1, half, onembed,
                                 1, n, CUFFT_C2R, batch, &estimate));

  CHECK_CUDA(cudaFree(device_real));
  CHECK_CUDA(cudaFree(device_complex));
  CHECK_CUFFT(cufftDestroy(forward));
  CHECK_CUFFT(cufftDestroy(inverse));
  return 0;
}

// A caller-managed work area on a plan bound to a stream, in double precision.
static int test_work_area_and_stream_z2z() {
  const int n = 48;
  const std::vector<cufftComplex> single = random_signal(n, 3);
  std::vector<cufftDoubleComplex> signal(n);
  for (int i = 0; i < n; ++i) {
    signal[i] = {single[i].x, single[i].y};
  }
  const std::vector<cufftComplex> spectrum = reference_dft(single, n, 1);

  cufftHandle plan = 0;
  CHECK_CUFFT(cufftCreate(&plan));
  CHECK_CUFFT(cufftSetAutoAllocation(plan, 0));
  size_t work_size = 0;
  CHECK_CUFFT(cufftMakePlan1d(plan, n, CUFFT_Z2Z, 1, &work_size));
  void *work_area = nullptr;
  CHECK_CUDA(cudaMalloc(&work_area, work_size != 0 ? work_size : 1));
  CHECK_CUFFT(cufftSetWorkArea(plan, work_area));

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUFFT(cufftSetStream(plan, stream));

  cufftDoubleComplex *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, signal.size() * sizeof(cufftDoubleComplex)));
  CHECK_CUDA(cudaMemcpyAsync(device, signal.data(),
                             signal.size() * sizeof(cufftDoubleComplex),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUFFT(cufftExecZ2Z(plan, device, device, CUFFT_FORWARD));
  std::vector<cufftDoubleComplex> result(n);
  CHECK_CUDA(cudaMemcpyAsync(result.data(), device,
                             result.size() * sizeof(cufftDoubleComplex),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<cufftComplex> narrowed(n);
  for (int i = 0; i < n; ++i) {
    narrowed[i] = {static_cast<float>(result[i].x),
                   static_cast<float>(result[i].y)};
  }
  if (compare(narrowed.data(), spectrum, n, "z2z with work area")) {
    return 1;
  }

  CHECK_CUDA(cudaFree(device));
  CHECK_CUDA(cudaFree(work_area));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUFFT(cufftDestroy(plan));
  return 0;
}

// The Xt plan API with data types, a 2D transform checked row by row and
// column by column through the batched 1D reference.
static int test_xt_plan_2d() {
  const int rows = 8, cols = 8;
  const std::vector<cufftComplex> signal = random_signal(rows * cols, 5);
  std::vector<cufftComplex> spectrum = reference_dft(signal, cols, rows);
  std::vector<cufftComplex> transposed(rows * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      transposed[static_cast<size_t>(c) * rows + r] =
          spectrum[static_cast<size_t>(r) * cols + c];
    }
  }
  transposed = reference_dft(transposed, rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      spectrum[static_cast<size_t>(r) * cols + c] =
          transposed[static_cast<size_t>(c) * rows + r];
    }
  }

  cufftHandle plan = 0;
  CHECK_CUFFT(cufftCreate(&plan));
  long long int sizes[2] = {rows, cols};
  size_t work_size = 0;
  CHECK_CUFFT(cufftXtMakePlanMany(plan, 2, sizes, nullptr, 1, rows * cols,
                                  CUDA_C_32F, nullptr, 1, rows * cols,
                                  CUDA_C_32F, 1, &work_size, CUDA_C_32F));
  CHECK_CUFFT(cufftXtGetSizeMany(plan, 2, sizes, nullptr, 1, rows * cols,
                                 CUDA_C_32F, nullptr, 1, rows * cols,
                                 CUDA_C_32F, 1, &work_size, CUDA_C_32F));

  cufftComplex *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, signal.size() * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(device, signal.data(),
                        signal.size() * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice));
  CHECK_CUFFT(cufftXtExec(plan, device, device, CUFFT_FORWARD));
  std::vector<cufftComplex> result(signal.size());
  CHECK_CUDA(cudaMemcpy(result.data(), device,
                        result.size() * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));
  if (compare(result.data(), spectrum, result.size(), "xt 2d forward")) {
    return 1;
  }
  CHECK_CUDA(cudaFree(device));
  CHECK_CUFFT(cufftDestroy(plan));
  return 0;
}

// A multi-GPU descriptor: the plan reports one work size per GPU, the client's
// copy of the descriptor reports the layout, and the host copies round-trip the
// signal through the transform. Two distinct devices are skipped when the
// server has one or the pair cannot share a plan; a device named twice, as
// single-GPU callers do, always runs.
static int test_xt_multi_gpu(int first, int second) {
  const bool distinct = first != second;
  if (distinct) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
      printf("multi-GPU descriptor: skipped, one device\n");
      return 0;
    }
  }
  const int n = 256;
  const std::vector<cufftComplex> signal = random_signal(n, 9);
  const std::vector<cufftComplex> spectrum = reference_dft(signal, n, 1);

  cufftHandle plan = 0;
  CHECK_CUFFT(cufftCreate(&plan));
  int gpus[2] = {first, second};
  cufftResult set = cufftXtSetGPUs(plan, 2, gpus);
  if (distinct && set != CUFFT_SUCCESS) {
    printf("multi-GPU descriptor: skipped, devices cannot share a plan\n");
    CHECK_CUFFT(cufftDestroy(plan));
    return 0;
  }
  CHECK_CUFFT(set);
  size_t work_sizes[2] = {SIZE_MAX, SIZE_MAX};
  cufftResult made = cufftMakePlan1d(plan, n, CUFFT_C2C, 1, work_sizes);
  if (distinct && made != CUFFT_SUCCESS) {
    printf("multi-GPU descriptor: skipped, plan not supported (%d)\n",
           static_cast<int>(made));
    CHECK_CUFFT(cufftDestroy(plan));
    return 0;
  }
  CHECK_CUFFT(made);
  EXPECT(work_sizes[0] != SIZE_MAX && work_sizes[1] != SIZE_MAX);
  size_t queried[2] = {SIZE_MAX, SIZE_MAX};
  CHECK_CUFFT(cufftGetSize(plan, queried));
  EXPECT(queried[0] == work_sizes[0] && queried[1] == work_sizes[1]);

  cudaLibXtDesc *descriptor = nullptr;
  CHECK_CUFFT(cufftXtMalloc(plan, &descriptor, CUFFT_XT_FORMAT_INPLACE));
  EXPECT(descriptor->descriptor->nGPUs == 2);
  EXPECT(descriptor->descriptor->size[0] + descriptor->descriptor->size[1] ==
         n * sizeof(cufftComplex));
  EXPECT(descriptor->descriptor->data[0] != nullptr);

  std::vector<cufftComplex> host = signal;
  CHECK_CUFFT(
      cufftXtMemcpy(plan, descriptor, host.data(), CUFFT_COPY_HOST_TO_DEVICE));
  CHECK_CUFFT(
      cufftXtExecDescriptorC2C(plan, descriptor, descriptor, CUFFT_FORWARD));
  CHECK_CUFFT(
      cufftXtMemcpy(plan, host.data(), descriptor, CUFFT_COPY_DEVICE_TO_HOST));
  if (compare(host.data(), spectrum, host.size(), "multi-GPU forward")) {
    return 1;
  }
  CHECK_CUFFT(cufftXtFree(descriptor));
  CHECK_CUFFT(cufftDestroy(plan));
  printf("multi-GPU descriptor on devices %d and %d: passed\n", first, second);
  return 0;
}

#if CUFFT_VERSION >= 11300
// An LTO load callback compiled at run time. cuFFT reads the symbol name and
// fatbin passed to cufftXtSetJITCallback when the plan is made, not when the
// callback is set, so the plan below only works if both are still readable
// then.
static const char *const scale_callback_source = R"(
struct scale_complex { float x; float y; };
scale_complex scale_callback(void *input, unsigned long long idx, void *info,
                             void *sharedmem) {
  const float scale = *static_cast<const float *>(info);
  const scale_complex *in = static_cast<const scale_complex *>(input);
  return scale_complex{in[idx].x * scale, in[idx].y * scale};
}
)";

static int test_jit_callback() {
  nvrtcProgram program = nullptr;
  EXPECT(nvrtcCreateProgram(&program, scale_callback_source, "scale.cu", 0,
                            nullptr, nullptr) == NVRTC_SUCCESS);
  const char *options[] = {"--std=c++11", "--relocatable-device-code=true",
                           "-default-device", "-dlto", "-arch=compute_75"};
  if (nvrtcCompileProgram(program, 5, options) != NVRTC_SUCCESS) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::vector<char> log(log_size + 1);
    nvrtcGetProgramLog(program, log.data());
    fprintf(stderr, "callback compilation failed: %s\n", log.data());
    return 1;
  }
  size_t lto_size = 0;
  EXPECT(nvrtcGetLTOIRSize(program, &lto_size) == NVRTC_SUCCESS);
  std::vector<char> lto(lto_size);
  EXPECT(nvrtcGetLTOIR(program, lto.data()) == NVRTC_SUCCESS);
  nvrtcDestroyProgram(&program);

  const int n = 128, batch = 3;
  const float scale = 2.0f;
  const std::vector<cufftComplex> signal = random_signal(n * batch, 7);
  const std::vector<cufftComplex> spectrum = reference_dft(signal, n, batch);

  float *device_scale = nullptr;
  CHECK_CUDA(cudaMalloc(&device_scale, sizeof(scale)));
  CHECK_CUDA(cudaMemcpy(device_scale, &scale, sizeof(scale),
                        cudaMemcpyHostToDevice));
  cufftHandle plan = 0;
  CHECK_CUFFT(cufftCreate(&plan));
  void *caller_info = device_scale;
  CHECK_CUFFT(cufftXtSetJITCallback(plan, "scale_callback", lto.data(),
                                    lto.size(), CUFFT_CB_LD_COMPLEX,
                                    &caller_info));
  size_t work_size = 0;
  CHECK_CUFFT(cufftMakePlan1d(plan, n, CUFFT_C2C, batch, &work_size));

  cufftComplex *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, spectrum.size() * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMemcpy(device, spectrum.data(),
                        spectrum.size() * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice));
  CHECK_CUFFT(cufftExecC2C(plan, device, device, CUFFT_INVERSE));
  std::vector<cufftComplex> result(signal.size());
  CHECK_CUDA(cudaMemcpy(result.data(), device,
                        result.size() * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));
  // The callback scales the input the unnormalized inverse transform loads.
  for (auto &value : result) {
    value.x /= n * scale;
    value.y /= n * scale;
  }
  if (compare(result.data(), signal, result.size(), "JIT callback inverse")) {
    return 1;
  }
  CHECK_CUDA(cudaFree(device));
  CHECK_CUDA(cudaFree(device_scale));
  CHECK_CUFFT(cufftDestroy(plan));
  printf("JIT callback: passed\n");
  return 0;
}
#endif

int main() {
  int version = 0;
  CHECK_CUFFT(cufftGetVersion(&version));
  EXPECT(version > 0);
  int major = 0;
  CHECK_CUFFT(cufftGetProperty(MAJOR_VERSION, &major));
  EXPECT(major == version / 1000);

  // The policy's optional limit takes the nullable pointer path. The library
  // accepts MINIMAL without a limit once the plan is made, and rejects USER
  // with one as CUFFT_INVALID_PLAN; either way the value reached it.
  cufftHandle policy_plan = 0;
  CHECK_CUFFT(cufftCreate(&policy_plan));
  size_t policy_work_size = 0;
  CHECK_CUFFT(
      cufftMakePlan1d(policy_plan, 64, CUFFT_C2C, 1, &policy_work_size));
  CHECK_CUFFT(
      cufftXtSetWorkAreaPolicy(policy_plan, CUFFT_WORKAREA_MINIMAL, nullptr));
  size_t limit = 1 << 20;
  cufftResult policy =
      cufftXtSetWorkAreaPolicy(policy_plan, CUFFT_WORKAREA_USER, &limit);
  EXPECT(policy == CUFFT_SUCCESS || policy == CUFFT_INVALID_PLAN);
  CHECK_CUFFT(cufftDestroy(policy_plan));

  if (test_plan1d_c2c() || test_plan_many_r2c_c2r() ||
      test_work_area_and_stream_z2z() || test_xt_plan_2d() ||
      test_xt_multi_gpu(0, 0) || test_xt_multi_gpu(0, 1)) {
    return 1;
  }
#if CUFFT_VERSION >= 11300
  if (test_jit_callback()) {
    return 1;
  }
#endif
  printf("cufft shim: all checks passed (cuFFT %d)\n", version);
  return 0;
}
