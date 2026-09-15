// Exercises the cuRAND shim end to end against the remote device: every
// generator type, seeding, offsets and ordering, each distribution into device
// memory checked statistically, host generators against device ones, a stream,
// the Sobol tables, error statuses and, when two GPUs are present, a generator
// on each.
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
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

#define CHECK_CURAND(call)                                                     \
  do {                                                                         \
    curandStatus_t status = (call);                                            \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
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

static const size_t kSamples = 1 << 20;

template <typename T> static int to_host(T *device, std::vector<T> &host) {
  CHECK_CUDA(cudaMemcpy(host.data(), device, host.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return 0;
}

template <typename T>
static void moments(const std::vector<T> &values, double *mean,
                    double *variance) {
  double sum = 0.0, square = 0.0;
  for (const T &value : values) {
    sum += value;
    square += static_cast<double>(value) * value;
  }
  *mean = sum / values.size();
  *variance = square / values.size() - *mean * *mean;
}

static int near(const char *what, double got, double want, double tolerance) {
  if (std::fabs(got - want) > tolerance) {
    fprintf(stderr, "%s = %g, expected %g +- %g\n", what, got, want, tolerance);
    return 1;
  }
  return 0;
}

static int test_generator_types() {
  const curandRngType_t types[] = {
      CURAND_RNG_PSEUDO_XORWOW,          CURAND_RNG_PSEUDO_MRG32K3A,
      CURAND_RNG_PSEUDO_MTGP32,          CURAND_RNG_PSEUDO_MT19937,
      CURAND_RNG_PSEUDO_PHILOX4_32_10,   CURAND_RNG_QUASI_SOBOL32,
      CURAND_RNG_QUASI_SCRAMBLED_SOBOL32};
  std::vector<float> host(4096);
  float *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(float)));
  for (curandRngType_t type : types) {
    curandGenerator_t generator = nullptr;
    CHECK_CURAND(curandCreateGenerator(&generator, type));
    CHECK_CURAND(curandGenerateUniform(generator, device, host.size()));
    if (to_host(device, host)) {
      return 1;
    }
    double mean = 0.0, variance = 0.0;
    moments(host, &mean, &variance);
    if (near("uniform mean", mean, 0.5, 0.05)) {
      fprintf(stderr, "rng type %d\n", static_cast<int>(type));
      return 1;
    }
    CHECK_CURAND(curandDestroyGenerator(generator));
  }

  // 64-bit output needs a 64-bit quasirandom generator; asking a 32-bit one
  // for it is the library's type error.
  curandGenerator_t sobol64 = nullptr;
  CHECK_CURAND(
      curandCreateGenerator(&sobol64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64));
  CHECK_CURAND(curandSetQuasiRandomGeneratorDimensions(sobol64, 2));
  std::vector<unsigned long long> bits(1024);
  unsigned long long *device_bits = nullptr;
  CHECK_CUDA(cudaMalloc(&device_bits, bits.size() * sizeof(bits[0])));
  CHECK_CURAND(curandGenerateLongLong(sobol64, device_bits, bits.size()));
  if (to_host(device_bits, bits)) {
    return 1;
  }
  size_t high = 0;
  for (unsigned long long value : bits) {
    high += value >> 63;
  }
  EXPECT(high > bits.size() / 4 && high < bits.size() * 3 / 4);
  CHECK_CURAND(curandDestroyGenerator(sobol64));

  curandGenerator_t xorwow = nullptr;
  CHECK_CURAND(curandCreateGenerator(&xorwow, CURAND_RNG_PSEUDO_XORWOW));
  EXPECT(curandGenerateLongLong(xorwow, device_bits, bits.size()) ==
         CURAND_STATUS_TYPE_ERROR);
  EXPECT(curandSetQuasiRandomGeneratorDimensions(xorwow, 2) ==
         CURAND_STATUS_TYPE_ERROR);
  CHECK_CURAND(curandDestroyGenerator(xorwow));
  CHECK_CUDA(cudaFree(device_bits));
  CHECK_CUDA(cudaFree(device));
  printf("generator types: passed\n");
  return 0;
}

static int generate_uniform_bits(curandRngType_t type, unsigned long long seed,
                                 unsigned long long offset,
                                 curandOrdering_t order,
                                 std::vector<unsigned int> &out) {
  curandGenerator_t generator = nullptr;
  CHECK_CURAND(curandCreateGenerator(&generator, type));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, seed));
  CHECK_CURAND(curandSetGeneratorOffset(generator, offset));
  CHECK_CURAND(curandSetGeneratorOrdering(generator, order));
  CHECK_CURAND(curandGenerateSeeds(generator));
  unsigned int *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, out.size() * sizeof(unsigned int)));
  CHECK_CURAND(curandGenerate(generator, device, out.size()));
  if (to_host(device, out)) {
    return 1;
  }
  CHECK_CUDA(cudaFree(device));
  CHECK_CURAND(curandDestroyGenerator(generator));
  return 0;
}

static int test_seed_offset_ordering() {
  const size_t n = 1000;
  std::vector<unsigned int> a(n), b(n), c(n), d(n), e(n);
  if (generate_uniform_bits(CURAND_RNG_PSEUDO_XORWOW, 42, 0,
                            CURAND_ORDERING_PSEUDO_DEFAULT, a) ||
      generate_uniform_bits(CURAND_RNG_PSEUDO_XORWOW, 42, 0,
                            CURAND_ORDERING_PSEUDO_DEFAULT, b) ||
      generate_uniform_bits(CURAND_RNG_PSEUDO_XORWOW, 43, 0,
                            CURAND_ORDERING_PSEUDO_DEFAULT, c) ||
      generate_uniform_bits(CURAND_RNG_PSEUDO_XORWOW, 42, 1000,
                            CURAND_ORDERING_PSEUDO_DEFAULT, d) ||
      generate_uniform_bits(CURAND_RNG_PSEUDO_XORWOW, 42, 0,
                            CURAND_ORDERING_PSEUDO_SEEDED, e)) {
    return 1;
  }
  EXPECT(a == b);
  EXPECT(a != c);
  EXPECT(a != d);
  EXPECT(a != e);

  curandGenerator_t generator = nullptr;
  CHECK_CURAND(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
  EXPECT(curandSetGeneratorOrdering(generator, CURAND_ORDERING_QUASI_DEFAULT) ==
         CURAND_STATUS_OUT_OF_RANGE);
  CHECK_CURAND(curandDestroyGenerator(generator));
  printf("seed, offset and ordering: passed\n");
  return 0;
}

static int test_distributions() {
  curandGenerator_t generator = nullptr;
  CHECK_CURAND(
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, 7));

  std::vector<float> floats(kSamples);
  std::vector<double> doubles(kSamples);
  std::vector<unsigned int> counts(kSamples);
  float *device_floats = nullptr;
  double *device_doubles = nullptr;
  unsigned int *device_counts = nullptr;
  CHECK_CUDA(cudaMalloc(&device_floats, kSamples * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_doubles, kSamples * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&device_counts, kSamples * sizeof(unsigned int)));
  double mean = 0.0, variance = 0.0;

  CHECK_CURAND(
      curandGenerateUniformDouble(generator, device_doubles, kSamples));
  if (to_host(device_doubles, doubles)) {
    return 1;
  }
  moments(doubles, &mean, &variance);
  if (near("uniform double mean", mean, 0.5, 0.005) ||
      near("uniform double variance", variance, 1.0 / 12, 0.002)) {
    return 1;
  }

  CHECK_CURAND(
      curandGenerateNormal(generator, device_floats, kSamples, 3.0f, 2.0f));
  if (to_host(device_floats, floats)) {
    return 1;
  }
  moments(floats, &mean, &variance);
  if (near("normal mean", mean, 3.0, 0.02) ||
      near("normal variance", variance, 4.0, 0.05)) {
    return 1;
  }

  CHECK_CURAND(curandGenerateNormalDouble(generator, device_doubles, kSamples,
                                          -1.0, 0.5));
  if (to_host(device_doubles, doubles)) {
    return 1;
  }
  moments(doubles, &mean, &variance);
  if (near("normal double mean", mean, -1.0, 0.005) ||
      near("normal double variance", variance, 0.25, 0.005)) {
    return 1;
  }

  CHECK_CURAND(
      curandGenerateLogNormal(generator, device_floats, kSamples, 0.0f, 0.25f));
  if (to_host(device_floats, floats)) {
    return 1;
  }
  moments(floats, &mean, &variance);
  if (near("lognormal mean", mean, std::exp(0.25 * 0.25 / 2), 0.005)) {
    return 1;
  }

  CHECK_CURAND(curandGenerateLogNormalDouble(generator, device_doubles,
                                             kSamples, 0.5, 0.1));
  if (to_host(device_doubles, doubles)) {
    return 1;
  }
  moments(doubles, &mean, &variance);
  if (near("lognormal double mean", mean, std::exp(0.5 + 0.1 * 0.1 / 2),
           0.005)) {
    return 1;
  }

  CHECK_CURAND(curandGeneratePoisson(generator, device_counts, kSamples, 4.0));
  if (to_host(device_counts, counts)) {
    return 1;
  }
  moments(counts, &mean, &variance);
  if (near("poisson mean", mean, 4.0, 0.02) ||
      near("poisson variance", variance, 4.0, 0.1)) {
    return 1;
  }

  // The libraries tested (10.4.1, 10.4.3) answer INTERNAL_ERROR for every
  // method, natively too.
  curandStatus_t method = curandGeneratePoissonMethod(
      generator, device_counts, kSamples, 200.0, CURAND_POISSON);
  EXPECT(method == CURAND_STATUS_SUCCESS ||
         method == CURAND_STATUS_INTERNAL_ERROR);
  if (method == CURAND_STATUS_SUCCESS) {
    if (to_host(device_counts, counts)) {
      return 1;
    }
    moments(counts, &mean, &variance);
    if (near("poisson method mean", mean, 200.0, 0.2)) {
      return 1;
    }
  }

  curandDiscreteDistribution_t distribution = nullptr;
  CHECK_CURAND(curandCreatePoissonDistribution(10.0, &distribution));
  EXPECT(distribution != nullptr);
  CHECK_CURAND(curandDestroyDistribution(distribution));

  CHECK_CUDA(cudaFree(device_counts));
  CHECK_CUDA(cudaFree(device_doubles));
  CHECK_CUDA(cudaFree(device_floats));
  CHECK_CURAND(curandDestroyGenerator(generator));
  printf("distributions: passed\n");
  return 0;
}

// A host generator and a device generator with the same settings produce the
// same sequence, so the results that come back in the response are checked
// against the device's, both below and above the handler's inline slot.
static int test_host_generator() {
  for (size_t n : {size_t{8}, size_t{100000}}) {
    curandGenerator_t host_generator = nullptr;
    curandGenerator_t device_generator = nullptr;
    CHECK_CURAND(
        curandCreateGeneratorHost(&host_generator, CURAND_RNG_PSEUDO_MRG32K3A));
    CHECK_CURAND(
        curandCreateGenerator(&device_generator, CURAND_RNG_PSEUDO_MRG32K3A));
    for (curandGenerator_t generator : {host_generator, device_generator}) {
      CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, 1234));
      CHECK_CURAND(curandSetGeneratorOffset(generator, 17));
    }

    std::vector<float> host(n), device(n);
    float *device_floats = nullptr;
    CHECK_CUDA(cudaMalloc(&device_floats, n * sizeof(float)));
    CHECK_CURAND(
        curandGenerateNormal(host_generator, host.data(), n, 1.0f, 2.0f));
    CHECK_CURAND(
        curandGenerateNormal(device_generator, device_floats, n, 1.0f, 2.0f));
    if (to_host(device_floats, device)) {
      return 1;
    }
    for (size_t i = 0; i < n; ++i) {
      if (std::fabs(host[i] - device[i]) >
          1e-4f * (1.0f + std::fabs(host[i]))) {
        fprintf(stderr, "host normal[%zu] = %g, device %g\n", i, host[i],
                device[i]);
        return 1;
      }
    }

    std::vector<unsigned int> host_counts(n), device_counts(n);
    unsigned int *device_bits = nullptr;
    CHECK_CUDA(cudaMalloc(&device_bits, n * sizeof(unsigned int)));
    CHECK_CURAND(
        curandGeneratePoisson(host_generator, host_counts.data(), n, 3.0));
    CHECK_CURAND(curandGeneratePoisson(device_generator, device_bits, n, 3.0));
    if (to_host(device_bits, device_counts)) {
      return 1;
    }
    double host_mean = 0.0, device_mean = 0.0, variance = 0.0;
    moments(host_counts, &host_mean, &variance);
    moments(device_counts, &device_mean, &variance);
    if (n > 1000 && (near("host poisson mean", host_mean, 3.0, 0.05) ||
                     near("device poisson mean", device_mean, 3.0, 0.05))) {
      return 1;
    }

    std::vector<double> host_doubles(n);
    CHECK_CURAND(
        curandGenerateUniformDouble(host_generator, host_doubles.data(), n));
    for (double value : host_doubles) {
      EXPECT(value > 0.0 && value <= 1.0);
    }

    CHECK_CUDA(cudaFree(device_bits));
    CHECK_CUDA(cudaFree(device_floats));
    CHECK_CURAND(curandDestroyGenerator(device_generator));
    CHECK_CURAND(curandDestroyGenerator(host_generator));
  }

  curandGenerator_t sobol = nullptr;
  CHECK_CURAND(curandCreateGeneratorHost(&sobol, CURAND_RNG_QUASI_SOBOL64));
  CHECK_CURAND(curandSetQuasiRandomGeneratorDimensions(sobol, 1));
  std::vector<unsigned long long> points(4);
  CHECK_CURAND(curandGenerateLongLong(sobol, points.data(), points.size()));
  // The first dimension of a Sobol sequence is the van der Corput sequence.
  EXPECT(points[0] == 0);
  EXPECT(points[1] == 1ull << 63);
  CHECK_CURAND(curandDestroyGenerator(sobol));
  printf("host generator: passed\n");
  return 0;
}

static int test_stream() {
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  curandGenerator_t generator = nullptr;
  CHECK_CURAND(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
  CHECK_CURAND(curandSetStream(generator, stream));
  std::vector<float> host(kSamples);
  float *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, kSamples * sizeof(float)));
  CHECK_CURAND(curandGenerateUniform(generator, device, kSamples));
  CHECK_CUDA(cudaMemcpyAsync(host.data(), device, kSamples * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  double mean = 0.0, variance = 0.0;
  moments(host, &mean, &variance);
  if (near("stream uniform mean", mean, 0.5, 0.005)) {
    return 1;
  }
  CHECK_CUDA(cudaFree(device));
  CHECK_CURAND(curandDestroyGenerator(generator));
  CHECK_CUDA(cudaStreamDestroy(stream));
  printf("stream: passed\n");
  return 0;
}

static int test_sobol_tables() {
  // The first dimension's direction numbers are the powers of two.
  curandDirectionVectors32_t *vectors32 = nullptr;
  CHECK_CURAND(curandGetDirectionVectors32(
      &vectors32, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
  for (int bit = 0; bit < 32; ++bit) {
    EXPECT(vectors32[0][bit] == 1u << (31 - bit));
  }
  // The table covers the documented 20,000 dimensions.
  EXPECT(vectors32[19999][0] == 1u << 31);
  curandDirectionVectors32_t *again = nullptr;
  CHECK_CURAND(
      curandGetDirectionVectors32(&again, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
  EXPECT(again == vectors32);
  curandDirectionVectors32_t *scrambled32 = nullptr;
  CHECK_CURAND(curandGetDirectionVectors32(
      &scrambled32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
  EXPECT(memcmp(scrambled32, vectors32, sizeof(*vectors32)) != 0);

  curandDirectionVectors64_t *vectors64 = nullptr;
  CHECK_CURAND(curandGetDirectionVectors64(
      &vectors64, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
  for (int bit = 0; bit < 64; ++bit) {
    EXPECT(vectors64[0][bit] == 1ull << (63 - bit));
  }
  EXPECT(vectors64[19999][0] == 1ull << 63);

  curandDirectionVectors32_t *wrong = nullptr;
  EXPECT(curandGetDirectionVectors32(&wrong,
                                     CURAND_DIRECTION_VECTORS_64_JOEKUO6) ==
         CURAND_STATUS_OUT_OF_RANGE);
  EXPECT(wrong == nullptr);

  unsigned int *constants32 = nullptr;
  CHECK_CURAND(curandGetScrambleConstants32(&constants32));
  unsigned long long *constants64 = nullptr;
  CHECK_CURAND(curandGetScrambleConstants64(&constants64));
  size_t nonzero = 0;
  for (int i = 0; i < 20000; ++i) {
    nonzero += constants32[i] != 0;
    nonzero += constants64[i] != 0;
  }
  EXPECT(nonzero > 39000);
  printf("sobol tables: passed\n");
  return 0;
}

// Every device gets its own generator, all alive at once, and the same seed
// gives the same sequence on each.
static int test_multi_device() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  if (devices < 2) {
    printf("multi-device: skipped, %d device\n", devices);
    return 0;
  }
  const size_t n = 1 << 16;
  std::vector<curandGenerator_t> generators(devices);
  std::vector<unsigned int *> buffers(devices);
  for (int device = 0; device < devices; ++device) {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CURAND(
        curandCreateGenerator(&generators[device], CURAND_RNG_PSEUDO_XORWOW));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generators[device], 99));
    CHECK_CUDA(cudaMalloc(&buffers[device], n * sizeof(unsigned int)));
  }
  std::vector<std::vector<unsigned int>> results(devices,
                                                 std::vector<unsigned int>(n));
  for (int device = devices - 1; device >= 0; --device) {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CURAND(curandGenerate(generators[device], buffers[device], n));
    if (to_host(buffers[device], results[device])) {
      return 1;
    }
  }
  for (int device = 0; device < devices; ++device) {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaFree(buffers[device]));
    CHECK_CURAND(curandDestroyGenerator(generators[device]));
    EXPECT(results[device] == results[0]);
  }
  CHECK_CUDA(cudaSetDevice(0));
  printf("multi-device: passed on %d devices\n", devices);
  return 0;
}

int main() {
  int version = 0;
  CHECK_CURAND(curandGetVersion(&version));
  EXPECT(version > 0);
  int major = 0;
  CHECK_CURAND(curandGetProperty(MAJOR_VERSION, &major));
  EXPECT(major == version / 1000);

  if (test_generator_types() || test_seed_offset_ordering() ||
      test_distributions() || test_host_generator() || test_stream() ||
      test_sobol_tables() || test_multi_device()) {
    return 1;
  }
  printf("curand shim: all checks passed (cuRAND %d)\n", version);
  return 0;
}
