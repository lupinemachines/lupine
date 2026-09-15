// Exercises the NPP shims end to end on every device: the core library's
// version, device name and properties and the global stream and its context,
// then one call from each image and signal library checked against a CPU
// reference: nppiMalloc and nppsMalloc memory, nppiAdd, nppiResize and
// nppiWarpAffine, channel swaps, color twists and gray conversion,
// nppiFilterBox, nppiMean_StdDev and nppiMinMaxIndx with their host-sized
// scratch buffers and device results, even and ranged histograms with host
// levels, and nppsSum.
#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if NPP_VERSION >= 13000
// NPP 13 exports the core device and stream calls without declaring them.
extern "C" {
int nppGetGpuNumSMs(void);
int nppGetMaxThreadsPerBlock(void);
int nppGetMaxThreadsPerSM(void);
int nppGetGpuDeviceProperties(int *pMaxThreadsPerSM, int *pMaxThreadsPerBlock,
                              int *pNumberOfSMs);
const char *nppGetGpuName(void);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
NppStatus nppSetStream(cudaStream_t hStream);
cudaStream_t nppGetStream(void);
}
#endif

#if NPP_VERSION >= 12205
typedef size_t buffer_size_t;
#else
typedef int buffer_size_t;
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_NPP(call)                                                        \
  do {                                                                         \
    NppStatus status = (call);                                                 \
    if (status != NPP_SUCCESS) {                                               \
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

static const int kWidth = 37;
static const int kHeight = 23;

struct Image {
  Npp8u *data = nullptr;
  int step = 0;
  int channels = 1;
  ~Image() { nppiFree(data); }
};

static int allocate(Image &image, int width, int height, int channels) {
  image.channels = channels;
  image.data = channels == 1 ? nppiMalloc_8u_C1(width, height, &image.step)
                             : nppiMalloc_8u_C3(width, height, &image.step);
  EXPECT(image.data != nullptr);
  EXPECT(image.step >= width * channels);
  return 0;
}

static int upload(const Image &image, const std::vector<Npp8u> &host, int width,
                  int height) {
  const int row = width * image.channels;
  for (int y = 0; y < height; ++y) {
    CHECK_CUDA(cudaMemcpy(image.data + y * image.step, host.data() + y * row,
                          row, cudaMemcpyHostToDevice));
  }
  return 0;
}

static int download(const Image &image, std::vector<Npp8u> &host, int width,
                    int height) {
  const int row = width * image.channels;
  host.assign(static_cast<size_t>(row) * height, 0);
  for (int y = 0; y < height; ++y) {
    CHECK_CUDA(cudaMemcpy(host.data() + y * row, image.data + y * image.step,
                          row, cudaMemcpyDeviceToHost));
  }
  return 0;
}

static std::vector<Npp8u> pattern(int width, int height, int channels,
                                  int seed) {
  std::vector<Npp8u> host(static_cast<size_t>(width) * height * channels);
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<Npp8u>((i * 37 + seed * 11 + (i / 7) * 5) % 251);
  }
  return host;
}

static int test_core(int device) {
  const NppLibraryVersion *version = nppGetLibVersion();
  EXPECT(version != nullptr);
  EXPECT(version->major == NPP_VER_MAJOR);
  const char *name = nppGetGpuName();
  EXPECT(name != nullptr && name[0] != '\0');
  EXPECT(nppGetGpuNumSMs() > 0);
  EXPECT(nppGetMaxThreadsPerBlock() > 0);
  int threads_per_sm = 0, threads_per_block = 0, sms = 0;
  EXPECT(nppGetGpuDeviceProperties(&threads_per_sm, &threads_per_block, &sms) ==
         0);
  EXPECT(threads_per_block == nppGetMaxThreadsPerBlock());
  printf("device %d: %s, NPP %d.%d.%d\n", device, name, version->major,
         version->minor, version->build);

  // NPP fills the context, and names the GPU, from the device current at its
  // first call and keeps that for the process.
  NppStreamContext context{};
  CHECK_NPP(nppGetStreamContext(&context));
  EXPECT(context.nCudaDeviceId >= 0 && context.nCudaDeviceId <= device);

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_NPP(nppSetStream(stream));
  EXPECT(nppGetStream() == stream);
  NppStreamContext stream_context{};
  CHECK_NPP(nppGetStreamContext(&stream_context));
  EXPECT(stream_context.hStream == stream);
  EXPECT(stream_context.nCudaDeviceId >= 0 &&
         stream_context.nCudaDeviceId <= device);

  // A call on the stream's context runs on the stream's server.
  std::vector<Npp8u> a = pattern(kWidth, kHeight, 1, 1);
  Image src, dst;
  if (allocate(src, kWidth, kHeight, 1) || allocate(dst, kWidth, kHeight, 1) ||
      upload(src, a, kWidth, kHeight)) {
    return 1;
  }
  CHECK_NPP(nppiAddC_8u_C1RSfs_Ctx(src.data, src.step, 3, dst.data, dst.step,
                                   NppiSize{kWidth, kHeight}, 0,
                                   stream_context));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<Npp8u> out;
  if (download(dst, out, kWidth, kHeight)) {
    return 1;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT(out[i] == std::min(255, a[i] + 3));
  }
  CHECK_NPP(nppSetStream(nullptr));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}

static int test_arithmetic_and_geometry(const NppStreamContext &context) {
  const NppiSize size{kWidth, kHeight};
  std::vector<Npp8u> a = pattern(kWidth, kHeight, 1, 2);
  std::vector<Npp8u> b = pattern(kWidth, kHeight, 1, 5);
  Image src1, src2, dst;
  if (allocate(src1, kWidth, kHeight, 1) ||
      allocate(src2, kWidth, kHeight, 1) || allocate(dst, kWidth, kHeight, 1) ||
      upload(src1, a, kWidth, kHeight) || upload(src2, b, kWidth, kHeight)) {
    return 1;
  }
  CHECK_NPP(nppiAdd_8u_C1RSfs_Ctx(src1.data, src1.step, src2.data, src2.step,
                                  dst.data, dst.step, size, 0, context));
  std::vector<Npp8u> out;
  if (download(dst, out, kWidth, kHeight)) {
    return 1;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT(out[i] == std::min(255, a[i] + b[i]));
  }

  // A nearest-neighbor 2x enlargement repeats each pixel in a 2x2 block.
  Image big;
  if (allocate(big, 2 * kWidth, 2 * kHeight, 1)) {
    return 1;
  }
  CHECK_NPP(nppiResize_8u_C1R_Ctx(
      src1.data, src1.step, size, NppiRect{0, 0, kWidth, kHeight}, big.data,
      big.step, NppiSize{2 * kWidth, 2 * kHeight},
      NppiRect{0, 0, 2 * kWidth, 2 * kHeight}, NPPI_INTER_NN, context));
  if (download(big, out, 2 * kWidth, 2 * kHeight)) {
    return 1;
  }
  for (int y = 0; y < 2 * kHeight; ++y) {
    for (int x = 0; x < 2 * kWidth; ++x) {
      EXPECT(out[y * 2 * kWidth + x] == a[(y / 2) * kWidth + x / 2]);
    }
  }

  // A whole-pixel translation moves the image; uncovered pixels keep theirs.
  const double coeffs[2][3] = {{1, 0, 3}, {0, 1, 2}};
  std::vector<Npp8u> background(a.size(), 7);
  if (upload(dst, background, kWidth, kHeight)) {
    return 1;
  }
  CHECK_NPP(nppiWarpAffine_8u_C1R_Ctx(src1.data, size, src1.step,
                                      NppiRect{0, 0, kWidth, kHeight}, dst.data,
                                      dst.step, NppiRect{0, 0, kWidth, kHeight},
                                      coeffs, NPPI_INTER_NN, context));
  if (download(dst, out, kWidth, kHeight)) {
    return 1;
  }
  for (int y = 2; y < kHeight; ++y) {
    for (int x = 3; x < kWidth; ++x) {
      EXPECT(out[y * kWidth + x] == a[(y - 2) * kWidth + (x - 3)]);
    }
  }
  EXPECT(out[0] == 7);
  return 0;
}

static int test_color(const NppStreamContext &context) {
  const NppiSize size{kWidth, kHeight};
  std::vector<Npp8u> rgb = pattern(kWidth, kHeight, 3, 3);
  Image src, dst, gray;
  if (allocate(src, kWidth, kHeight, 3) || allocate(dst, kWidth, kHeight, 3) ||
      allocate(gray, kWidth, kHeight, 1) || upload(src, rgb, kWidth, kHeight)) {
    return 1;
  }
  const int order[3] = {2, 0, 1};
  CHECK_NPP(nppiSwapChannels_8u_C3R_Ctx(src.data, src.step, dst.data, dst.step,
                                        size, order, context));
  std::vector<Npp8u> out;
  if (download(dst, out, kWidth, kHeight)) {
    return 1;
  }
  for (size_t i = 0; i < out.size(); i += 3) {
    EXPECT(out[i] == rgb[i + 2] && out[i + 1] == rgb[i] &&
           out[i + 2] == rgb[i + 1]);
  }

  // The twist reverses the channels and adds one to green.
  const Npp32f twist[3][4] = {{0, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 0}};
  CHECK_NPP(nppiColorTwist32f_8u_C3R_Ctx(src.data, src.step, dst.data, dst.step,
                                         size, twist, context));
  if (download(dst, out, kWidth, kHeight)) {
    return 1;
  }
  for (size_t i = 0; i < out.size(); i += 3) {
    EXPECT(out[i] == rgb[i + 2] &&
           out[i + 1] == std::min(255, rgb[i + 1] + 1) && out[i + 2] == rgb[i]);
  }

  CHECK_NPP(nppiRGBToGray_8u_C3C1R_Ctx(src.data, src.step, gray.data, gray.step,
                                       size, context));
  if (download(gray, out, kWidth, kHeight)) {
    return 1;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    const double y =
        0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2];
    EXPECT(std::fabs(out[i] - y) <= 1.0);
  }
  return 0;
}

static int test_filter_and_statistics(const NppStreamContext &context) {
  const NppiSize size{kWidth, kHeight};
  std::vector<Npp8u> a = pattern(kWidth, kHeight, 1, 4);
  a[5 * kWidth + 9] = 0;
  a[17 * kWidth + 30] = 255;
  for (Npp8u &value : a) {
    if (&value != &a[5 * kWidth + 9] && &value != &a[17 * kWidth + 30]) {
      value = static_cast<Npp8u>(std::min(250, std::max(1, int(value))));
    }
  }
  Image src, dst;
  if (allocate(src, kWidth, kHeight, 1) || allocate(dst, kWidth, kHeight, 1) ||
      upload(src, a, kWidth, kHeight)) {
    return 1;
  }
  CHECK_NPP(nppiFilterBoxBorder_8u_C1R_Ctx(
      src.data, src.step, size, NppiPoint{0, 0}, dst.data, dst.step, size,
      NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, context));
  std::vector<Npp8u> out;
  if (download(dst, out, kWidth, kHeight)) {
    return 1;
  }
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      int sum = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int sy = std::min(kHeight - 1, std::max(0, y + dy));
          const int sx = std::min(kWidth - 1, std::max(0, x + dx));
          sum += a[sy * kWidth + sx];
        }
      }
      EXPECT(std::fabs(out[y * kWidth + x] - sum / 9.0) <= 1.0);
    }
  }

  double mean = 0, squares = 0;
  for (Npp8u value : a) {
    mean += value;
    squares += double(value) * value;
  }
  const double n = double(a.size());
  mean /= n;
  const double population = std::sqrt(squares / n - mean * mean);
  const double sample = population * std::sqrt(n / (n - 1));

  buffer_size_t scratch_size = 0;
  CHECK_NPP(
      nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(size, &scratch_size, context));
  void *scratch = nullptr;
  CHECK_CUDA(cudaMalloc(&scratch, std::max<size_t>(scratch_size, 1)));
  void *results = nullptr;
  CHECK_CUDA(cudaMalloc(&results, 64));
  Npp64f *device_mean = static_cast<Npp64f *>(results);
  Npp64f *device_stddev = device_mean + 1;
  CHECK_NPP(nppiMean_StdDev_8u_C1R_Ctx(src.data, src.step, size,
                                       static_cast<Npp8u *>(scratch),
                                       device_mean, device_stddev, context));
  Npp64f host[2];
  CHECK_CUDA(cudaMemcpy(host, results, sizeof(host), cudaMemcpyDeviceToHost));
  EXPECT(std::fabs(host[0] - mean) < 1e-6 * mean);
  EXPECT(std::fabs(host[1] - population) < 1e-6 * population ||
         std::fabs(host[1] - sample) < 1e-6 * sample);
  CHECK_CUDA(cudaFree(scratch));

  CHECK_NPP(
      nppiMinMaxIndxGetBufferHostSize_8u_C1R_Ctx(size, &scratch_size, context));
  CHECK_CUDA(cudaMalloc(&scratch, std::max<size_t>(scratch_size, 1)));
  Npp8u *device_min = static_cast<Npp8u *>(results);
  Npp8u *device_max = device_min + 1;
  NppiPoint *device_min_index =
      reinterpret_cast<NppiPoint *>(static_cast<Npp8u *>(results) + 8);
  NppiPoint *device_max_index = device_min_index + 1;
  CHECK_NPP(nppiMinMaxIndx_8u_C1R_Ctx(
      src.data, src.step, size, device_min, device_max, device_min_index,
      device_max_index, static_cast<Npp8u *>(scratch), context));
  Npp8u extrema[24];
  CHECK_CUDA(
      cudaMemcpy(extrema, results, sizeof(extrema), cudaMemcpyDeviceToHost));
  NppiPoint min_index, max_index;
  std::memcpy(&min_index, extrema + 8, sizeof(min_index));
  std::memcpy(&max_index, extrema + 16, sizeof(max_index));
  EXPECT(extrema[0] == 0 && extrema[1] == 255);
  EXPECT(min_index.x == 9 && min_index.y == 5);
  EXPECT(max_index.x == 30 && max_index.y == 17);
  CHECK_CUDA(cudaFree(scratch));
  CHECK_CUDA(cudaFree(results));
  return 0;
}

static int test_histograms(const NppStreamContext &context) {
  const NppiSize size{kWidth, kHeight};
  std::vector<Npp8u> rgb = pattern(kWidth, kHeight, 3, 6);
  Image src;
  if (allocate(src, kWidth, kHeight, 3) || upload(src, rgb, kWidth, kHeight)) {
    return 1;
  }

  // Each channel counts into its own bins: host level counts and bounds,
  // device histograms.
  int levels[3] = {5, 9, 17};
  Npp32s lower[3] = {0, 16, 0};
  Npp32s upper[3] = {256, 240, 256};
  buffer_size_t scratch_size = 0;
#if NPP_VERSION >= 12101
  CHECK_NPP(nppiHistogramEvenGetBufferSize_8u_C3R_Ctx(size, levels,
                                                      &scratch_size, context));
#else
  // Before NPP 12.1.1 the histogram buffer-size calls take no context.
  CHECK_NPP(nppiHistogramEvenGetBufferSize_8u_C3R(size, levels, &scratch_size));
#endif
  void *scratch = nullptr;
  CHECK_CUDA(cudaMalloc(&scratch, std::max<size_t>(scratch_size, 1)));
  void *hist_memory = nullptr;
  CHECK_CUDA(cudaMalloc(&hist_memory, 3 * 16 * sizeof(Npp32s)));
  Npp32s *hist[3] = {static_cast<Npp32s *>(hist_memory),
                     static_cast<Npp32s *>(hist_memory) + 16,
                     static_cast<Npp32s *>(hist_memory) + 32};
  CHECK_NPP(nppiHistogramEven_8u_C3R_Ctx(
      src.data, src.step, size, hist, levels, lower, upper,
      static_cast<Npp8u *>(scratch), context));
  std::vector<Npp32s> counts(48);
  CHECK_CUDA(cudaMemcpy(counts.data(), hist_memory, counts.size() * 4,
                        cudaMemcpyDeviceToHost));
  for (int c = 0; c < 3; ++c) {
    std::vector<Npp32s> level_values(levels[c]);
    CHECK_NPP(nppiEvenLevelsHost_32s(level_values.data(), levels[c], lower[c],
                                     upper[c]));
    EXPECT(level_values.front() == lower[c]);
    EXPECT(level_values.back() == upper[c]);
    for (int bin = 0; bin + 1 < levels[c]; ++bin) {
      Npp32s expected = 0;
      for (size_t i = c; i < rgb.size(); i += 3) {
        expected +=
            rgb[i] >= level_values[bin] && rgb[i] < level_values[bin + 1];
      }
      EXPECT(counts[16 * c + bin] == expected);
    }
  }
  CHECK_CUDA(cudaFree(scratch));

  Image gray;
  if (allocate(gray, kWidth, kHeight, 1)) {
    return 1;
  }
  std::vector<Npp8u> g = pattern(kWidth, kHeight, 1, 8);
  if (upload(gray, g, kWidth, kHeight)) {
    return 1;
  }
  const Npp32s range_levels[6] = {0, 10, 50, 51, 200, 256};
  void *device_levels = nullptr;
  CHECK_CUDA(cudaMalloc(&device_levels, sizeof(range_levels)));
  CHECK_CUDA(cudaMemcpy(device_levels, range_levels, sizeof(range_levels),
                        cudaMemcpyHostToDevice));
#if NPP_VERSION >= 12101
  CHECK_NPP(nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(size, 6, &scratch_size,
                                                       context));
#else
  CHECK_NPP(nppiHistogramRangeGetBufferSize_8u_C1R(size, 6, &scratch_size));
#endif
  CHECK_CUDA(cudaMalloc(&scratch, std::max<size_t>(scratch_size, 1)));
  CHECK_NPP(nppiHistogramRange_8u_C1R_Ctx(
      gray.data, gray.step, size, static_cast<Npp32s *>(hist_memory),
      static_cast<Npp32s *>(device_levels), 6, static_cast<Npp8u *>(scratch),
      context));
  CHECK_CUDA(cudaMemcpy(counts.data(), hist_memory, 5 * sizeof(Npp32s),
                        cudaMemcpyDeviceToHost));
  for (int bin = 0; bin < 5; ++bin) {
    Npp32s expected = 0;
    for (Npp8u value : g) {
      expected += value >= range_levels[bin] && value < range_levels[bin + 1];
    }
    EXPECT(counts[bin] == expected);
  }
  CHECK_CUDA(cudaFree(scratch));
  CHECK_CUDA(cudaFree(device_levels));
  CHECK_CUDA(cudaFree(hist_memory));
  return 0;
}

static int test_signal(const NppStreamContext &context) {
  const size_t length = 1000;
  std::vector<Npp32f> host(length);
  double expected = 0;
  for (size_t i = 0; i < length; ++i) {
    host[i] = static_cast<Npp32f>(std::sin(0.01 * i) + 0.25);
    expected += host[i];
  }
  Npp32f *signal = nppsMalloc_32f(length);
  EXPECT(signal != nullptr);
  CHECK_CUDA(cudaMemcpy(signal, host.data(), length * sizeof(Npp32f),
                        cudaMemcpyHostToDevice));
  buffer_size_t scratch_size = 0;
  CHECK_NPP(nppsSumGetBufferSize_32f_Ctx(length, &scratch_size, context));
  Npp8u *scratch = nppsMalloc_8u(std::max<size_t>(scratch_size, 1));
  EXPECT(scratch != nullptr);
  Npp32f *sum = nppsMalloc_32f(1);
  EXPECT(sum != nullptr);
  CHECK_NPP(nppsSum_32f_Ctx(signal, length, sum, scratch, context));
  Npp32f result = 0;
  CHECK_CUDA(cudaMemcpy(&result, sum, sizeof(result), cudaMemcpyDeviceToHost));
  EXPECT(std::fabs(result - expected) < 1e-3 * std::fabs(expected));
  nppsFree(sum);
  nppsFree(scratch);
  nppsFree(signal);
  return 0;
}

int main() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  EXPECT(devices > 0);
  for (int device = 0; device < devices; ++device) {
    CHECK_CUDA(cudaSetDevice(device));
    if (test_core(device)) {
      fprintf(stderr, "core failed on device %d\n", device);
      return 1;
    }
    NppStreamContext context{};
    CHECK_NPP(nppGetStreamContext(&context));
    if (test_arithmetic_and_geometry(context) || test_color(context) ||
        test_filter_and_statistics(context) || test_histograms(context) ||
        test_signal(context)) {
      fprintf(stderr, "failed on device %d\n", device);
      return 1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  printf("NPP shim test passed on %d device(s)\n", devices);
  return 0;
}
