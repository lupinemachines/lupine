// Exercises the cuDNN shim end to end against the remote device: version and
// error queries, descriptor round trips (tensor, filter, convolution, pooling,
// activation, dropout, transform, RNN data), legacy forward convolution with
// float and double scaling, activation, softmax, pooling, fills, a derived
// descriptor, the CTC loss and attention window host arrays, a backend-API
// convolution graph and, when two GPUs are present, a convolution on the
// second device. Results are checked against CPU references.
#include <cuda_runtime.h>
#include <cudnn.h>

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

#define CHECK_CUDNN(call)                                                      \
  do {                                                                         \
    cudnnStatus_t status = (call);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "%s failed: %s\n", #call, cudnnGetErrorString(status));  \
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

template <typename T>
static int upload(const std::vector<T> &host, void **device) {
  CHECK_CUDA(cudaMalloc(device, host.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return 0;
}

template <typename T>
static int download(void *device, std::vector<T> &host) {
  CHECK_CUDA(cudaMemcpy(host.data(), device, host.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return 0;
}

template <typename T>
static int compare(const std::vector<T> &got, const std::vector<T> &want,
                   const char *what) {
  for (size_t i = 0; i < want.size(); ++i) {
    if (std::fabs(static_cast<double>(got[i] - want[i])) >
        1e-4 * (1.0 + std::fabs(static_cast<double>(want[i])))) {
      fprintf(stderr, "%s[%zu] = %g, expected %g\n", what, i,
              static_cast<double>(got[i]), static_cast<double>(want[i]));
      return 1;
    }
  }
  return 0;
}

// A 5x5 single-channel image and 3x3 filter, cross-correlated with padding 1.
static const int kImage = 5;
static const int kFilter = 3;

template <typename T> static std::vector<T> test_image() {
  std::vector<T> image(kImage * kImage);
  for (int i = 0; i < kImage * kImage; ++i) {
    image[i] = static_cast<T>((i * 7) % 11) - 5;
  }
  return image;
}

template <typename T> static std::vector<T> test_filter() {
  std::vector<T> filter(kFilter * kFilter);
  for (int i = 0; i < kFilter * kFilter; ++i) {
    filter[i] = static_cast<T>(i % 4) - 1;
  }
  return filter;
}

template <typename T>
static std::vector<T> reference_conv(const std::vector<T> &image,
                                     const std::vector<T> &filter, T alpha) {
  std::vector<T> out(kImage * kImage, 0);
  for (int y = 0; y < kImage; ++y) {
    for (int x = 0; x < kImage; ++x) {
      T sum = 0;
      for (int fy = 0; fy < kFilter; ++fy) {
        for (int fx = 0; fx < kFilter; ++fx) {
          const int iy = y + fy - 1;
          const int ix = x + fx - 1;
          if (iy >= 0 && iy < kImage && ix >= 0 && ix < kImage) {
            sum += image[iy * kImage + ix] * filter[fy * kFilter + fx];
          }
        }
      }
      out[y * kImage + x] = alpha * sum;
    }
  }
  return out;
}

// Legacy forward convolution. Doubles take 8-byte scaling factors, floats 4;
// beta blends into the prefilled output.
template <typename T>
static int legacy_conv(cudnnHandle_t handle, cudnnDataType_t type, T alpha,
                       T beta, const char *what) {
  const std::vector<T> image = test_image<T>();
  const std::vector<T> filter = test_filter<T>();
  cudnnTensorDescriptor_t x_desc = nullptr, y_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, type, 1, 1,
                                         kImage, kImage));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, type, CUDNN_TENSOR_NCHW, 1, 1,
                                         kFilter, kFilter));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, type));
  int n = 0, c = 0, h = 0, w = 0;
  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc,
                                                    &n, &c, &h, &w));
  EXPECT(n == 1 && c == 1 && h == kImage && w == kImage);
  CHECK_CUDNN(
      cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, type, n, c, h, w));

  int max_algos = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &max_algos));
  EXPECT(max_algos > 0);
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(max_algos);
  int returned = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, x_desc, w_desc, conv_desc, y_desc, max_algos, &returned,
      perf.data()));
  EXPECT(returned > 0 && returned <= max_algos);
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t workspace_size = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

  std::vector<T> prefill(kImage * kImage, static_cast<T>(2));
  void *x = nullptr, *wt = nullptr, *y = nullptr, *workspace = nullptr;
  if (upload(image, &x) || upload(filter, &wt) || upload(prefill, &y)) {
    return 1;
  }
  if (workspace_size != 0) {
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }
  CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, x_desc, x, w_desc, wt,
                                      conv_desc, algo, workspace,
                                      workspace_size, &beta, y_desc, y));
  std::vector<T> got(kImage * kImage);
  if (download(y, got)) {
    return 1;
  }
  std::vector<T> want = reference_conv(image, filter, alpha);
  for (T &value : want) {
    value += beta * static_cast<T>(2);
  }
  if (compare(got, want, what)) {
    return 1;
  }
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(wt));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
  printf("%s: passed\n", what);
  return 0;
}

static int test_descriptors(cudnnHandle_t handle) {
  cudnnTensorDescriptor_t tensor = nullptr;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor));
  const int dims[] = {2, 3, 4, 5};
  const int strides[] = {60, 20, 5, 1};
  CHECK_CUDNN(
      cudnnSetTensorNdDescriptor(tensor, CUDNN_DATA_FLOAT, 4, dims, strides));
  cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  int nb_dims = 0;
  int got_dims[CUDNN_DIM_MAX] = {};
  int got_strides[CUDNN_DIM_MAX] = {};
  CHECK_CUDNN(cudnnGetTensorNdDescriptor(tensor, CUDNN_DIM_MAX, &type, &nb_dims,
                                         got_dims, got_strides));
  EXPECT(type == CUDNN_DATA_FLOAT && nb_dims == 4);
  EXPECT(memcmp(got_dims, dims, sizeof(dims)) == 0);
  EXPECT(memcmp(got_strides, strides, sizeof(strides)) == 0);
  size_t bytes = 0;
  CHECK_CUDNN(cudnnGetTensorSizeInBytes(tensor, &bytes));
  EXPECT(bytes == 2 * 3 * 4 * 5 * sizeof(float));
  int n = 0, c = 0, h = 0, w = 0, ns = 0, cs = 0, hs = 0, ws = 0;
  CHECK_CUDNN(cudnnGetTensor4dDescriptor(tensor, &type, &n, &c, &h, &w, &ns,
                                         &cs, &hs, &ws));
  EXPECT(n == 2 && c == 3 && h == 4 && w == 5 && ns == 60 && ws == 1);
  EXPECT(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT, -1, 1, 1, 1) ==
         CUDNN_STATUS_BAD_PARAM);
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor));

  cudnnFilterDescriptor_t filter = nullptr;
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter));
  const int filter_dims[] = {8, 3, 3, 3};
  CHECK_CUDNN(cudnnSetFilterNdDescriptor(filter, CUDNN_DATA_HALF,
                                         CUDNN_TENSOR_NHWC, 4, filter_dims));
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  int got_filter[CUDNN_DIM_MAX] = {};
  CHECK_CUDNN(cudnnGetFilterNdDescriptor(filter, CUDNN_DIM_MAX, &type, &format,
                                         &nb_dims, got_filter));
  EXPECT(type == CUDNN_DATA_HALF && format == CUDNN_TENSOR_NHWC);
  EXPECT(nb_dims == 4 && memcmp(got_filter, filter_dims, sizeof(filter_dims)) == 0);
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter));

  cudnnConvolutionDescriptor_t conv = nullptr;
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv));
  const int pads[] = {1, 2};
  const int conv_strides[] = {2, 1};
  const int dilations[] = {1, 3};
  CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv, 2, pads, conv_strides,
                                              dilations, CUDNN_CONVOLUTION,
                                              CUDNN_DATA_FLOAT));
  CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv, 4));
  CHECK_CUDNN(cudnnSetConvolutionMathType(conv, CUDNN_TENSOR_OP_MATH));
  int length = 0;
  int got_pads[2] = {}, got_strides_conv[2] = {}, got_dilations[2] = {};
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(conv, 2, &length, got_pads,
                                              got_strides_conv, got_dilations,
                                              &mode, &type));
  EXPECT(length == 2 && got_pads[1] == 2 && got_strides_conv[0] == 2 &&
         got_dilations[1] == 3 && mode == CUDNN_CONVOLUTION);
  int groups = 0;
  cudnnMathType_t math = CUDNN_DEFAULT_MATH;
  CHECK_CUDNN(cudnnGetConvolutionGroupCount(conv, &groups));
  CHECK_CUDNN(cudnnGetConvolutionMathType(conv, &math));
  EXPECT(groups == 4 && math == CUDNN_TENSOR_OP_MATH);
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv));

  cudnnActivationDescriptor_t activation = nullptr;
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation));
  CHECK_CUDNN(cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_CLIPPED_RELU,
                                           CUDNN_PROPAGATE_NAN, 6.5));
  cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_RELU;
  cudnnNanPropagation_t nan = CUDNN_NOT_PROPAGATE_NAN;
  double coef = 0;
  CHECK_CUDNN(cudnnGetActivationDescriptor(activation, &activation_mode, &nan,
                                           &coef));
  EXPECT(activation_mode == CUDNN_ACTIVATION_CLIPPED_RELU &&
         nan == CUDNN_PROPAGATE_NAN && coef == 6.5);
  CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation));

  // Null pad and fold arrays take the optional-array path both ways.
  cudnnTensorTransformDescriptor_t transform = nullptr;
  CHECK_CUDNN(cudnnCreateTensorTransformDescriptor(&transform));
  const int32_t pad_after[] = {0, 0, 1, 2};
  CHECK_CUDNN(cudnnSetTensorTransformDescriptor(transform, 4, CUDNN_TENSOR_NCHW,
                                                nullptr, pad_after, nullptr,
                                                CUDNN_TRANSFORM_FOLD));
  cudnnTensorFormat_t dest_format = CUDNN_TENSOR_NHWC;
  int32_t got_pad_after[4] = {-1, -1, -1, -1};
  cudnnFoldingDirection_t direction = CUDNN_TRANSFORM_UNFOLD;
  CHECK_CUDNN(cudnnGetTensorTransformDescriptor(transform, 4, &dest_format,
                                                nullptr, got_pad_after, nullptr,
                                                &direction));
  EXPECT(dest_format == CUDNN_TENSOR_NCHW && direction == CUDNN_TRANSFORM_FOLD);
  EXPECT(memcmp(got_pad_after, pad_after, sizeof(pad_after)) == 0);
  CHECK_CUDNN(cudnnDestroyTensorTransformDescriptor(transform));

  // The padding fill is a host scalar in the data type, sent and received.
  cudnnRNNDataDescriptor_t rnn_data = nullptr;
  CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&rnn_data));
  const int seq_lengths[] = {4, 2, 3};
  double fill = 0.25;
  CHECK_CUDNN(cudnnSetRNNDataDescriptor(
      rnn_data, CUDNN_DATA_DOUBLE, CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
      4, 3, 16, seq_lengths, &fill));
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  int max_seq = 0, batch = 0, vector = 0;
  int got_lengths[3] = {};
  double got_fill = 0;
  CHECK_CUDNN(cudnnGetRNNDataDescriptor(rnn_data, &type, &layout, &max_seq,
                                        &batch, &vector, 3, got_lengths,
                                        &got_fill));
  EXPECT(type == CUDNN_DATA_DOUBLE && max_seq == 4 && batch == 3 &&
         vector == 16 && got_fill == 0.25);
  EXPECT(memcmp(got_lengths, seq_lengths, sizeof(seq_lengths)) == 0);
  CHECK_CUDNN(cudnnDestroyRNNDataDescriptor(rnn_data));

  // Dropout states are device memory whose address comes back unchanged.
  size_t states_size = 0;
  CHECK_CUDNN(cudnnDropoutGetStatesSize(handle, &states_size));
  void *states = nullptr;
  CHECK_CUDA(cudaMalloc(&states, states_size));
  cudnnDropoutDescriptor_t dropout = nullptr;
  CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout));
  CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout, handle, 0.375f, states,
                                        states_size, 1234));
  float got_dropout = 0;
  void *got_states = nullptr;
  unsigned long long seed = 0;
  CHECK_CUDNN(cudnnGetDropoutDescriptor(dropout, handle, &got_dropout,
                                        &got_states, &seed));
  EXPECT(got_dropout == 0.375f && got_states == states && seed == 1234);
  CHECK_CUDNN(cudnnDestroyDropoutDescriptor(dropout));
  CHECK_CUDA(cudaFree(states));
  printf("descriptors: passed\n");
  return 0;
}

// Pooling, activation and softmax on one float tensor, plus fills.
static int test_pointwise_ops(cudnnHandle_t handle) {
  const std::vector<float> input = {1, -2, 3, -4, 5, -6, 7, -8,
                                    9, 0,  2, 4,  -1, 3, 8, 6};
  cudnnTensorDescriptor_t x_desc = nullptr, pooled_desc = nullptr;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&pooled_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, 1, 1, 4, 4));
  void *x = nullptr, *y = nullptr;
  std::vector<float> zeros(16, 0);
  if (upload(input, &x) || upload(zeros, &y)) {
    return 1;
  }
  const float one = 1, zero = 0;

  cudnnPoolingDescriptor_t pool = nullptr;
  CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool));
  CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX,
                                          CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0,
                                          2, 2));
  int n = 0, c = 0, h = 0, w = 0;
  CHECK_CUDNN(
      cudnnGetPooling2dForwardOutputDim(pool, x_desc, &n, &c, &h, &w));
  EXPECT(n == 1 && c == 1 && h == 2 && w == 2);
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(pooled_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, n, c, h, w));
  CHECK_CUDNN(cudnnPoolingForward(handle, pool, &one, x_desc, x, &zero,
                                  pooled_desc, y));
  std::vector<float> pooled(4);
  if (download(y, pooled) || compare(pooled, {5, 7, 9, 8}, "max pool")) {
    return 1;
  }
  CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pool));

  cudnnActivationDescriptor_t relu = nullptr;
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu));
  CHECK_CUDNN(cudnnSetActivationDescriptor(relu, CUDNN_ACTIVATION_RELU,
                                           CUDNN_NOT_PROPAGATE_NAN, 0));
  const float half = 0.5f;
  CHECK_CUDNN(cudnnActivationForward(handle, relu, &half, x_desc, x, &zero,
                                     x_desc, y));
  std::vector<float> relu_out(16), relu_want(16);
  for (int i = 0; i < 16; ++i) {
    relu_want[i] = input[i] > 0 ? 0.5f * input[i] : 0;
  }
  if (download(y, relu_out) || compare(relu_out, relu_want, "relu")) {
    return 1;
  }
  CHECK_CUDNN(cudnnDestroyActivationDescriptor(relu));

  CHECK_CUDNN(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE, &one, x_desc, x,
                                  &zero, x_desc, y));
  std::vector<float> softmax(16), softmax_want(16);
  double total = 0;
  for (float value : input) {
    total += std::exp(static_cast<double>(value));
  }
  for (int i = 0; i < 16; ++i) {
    softmax_want[i] = static_cast<float>(std::exp(input[i]) / total);
  }
  if (download(y, softmax) || compare(softmax, softmax_want, "softmax")) {
    return 1;
  }

  // The fill value is one element of the tensor's type, the scale a float.
  const float value = 3.5f, scale = -2.0f;
  CHECK_CUDNN(cudnnSetTensor(handle, x_desc, y, &value));
  CHECK_CUDNN(cudnnScaleTensor(handle, x_desc, y, &scale));
  std::vector<float> filled(16);
  if (download(y, filled) ||
      compare(filled, std::vector<float>(16, -7.0f), "fill and scale")) {
    return 1;
  }

  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(pooled_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
  printf("pooling, activation, softmax, fills: passed\n");
  return 0;
}

// A descriptor the library derives has a type the client never saw set; its
// double scaling factors must still travel as doubles.
static int test_derived_descriptor(cudnnHandle_t handle) {
  cudnnTensorDescriptor_t x_desc = nullptr, derived = nullptr;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&derived));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_DOUBLE, 2, 3, 4, 4));
  CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(derived, x_desc,
                                            CUDNN_BATCHNORM_SPATIAL));
  const std::vector<double> a = {1.5, -2.0, 4.0};
  const std::vector<double> c = {10.0, 20.0, 30.0};
  void *a_dev = nullptr, *c_dev = nullptr;
  if (upload(a, &a_dev) || upload(c, &c_dev)) {
    return 1;
  }
  const double alpha = 3.0, beta = 0.5;
  CHECK_CUDNN(
      cudnnAddTensor(handle, &alpha, derived, a_dev, &beta, derived, c_dev));
  std::vector<double> got(3);
  if (download(c_dev, got) ||
      compare(got, {9.5, 4.0, 27.0}, "derived descriptor add")) {
    return 1;
  }
  CHECK_CUDA(cudaFree(a_dev));
  CHECK_CUDA(cudaFree(c_dev));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(derived));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
  printf("derived descriptor: passed\n");
  return 0;
}

// CTC loss over T=4 steps, one batch member, three classes with blank 0 and
// label [1]: the host label arrays are sized by the probabilities' batch.
static int test_ctc_loss(cudnnHandle_t handle) {
  const int steps = 4, classes = 3;
  const std::vector<float> logits = {0.5f, 1.0f, -0.5f, 0.2f, 0.3f, 0.1f,
                                     1.2f, -0.4f, 0.0f, 0.3f, 0.9f, -1.0f};
  cudnnTensorDescriptor_t probs_desc = nullptr, grads_desc = nullptr;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&probs_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&grads_desc));
  const int dims[] = {steps, 1, classes};
  const int strides[] = {classes, classes, 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(probs_desc, CUDNN_DATA_FLOAT, 3, dims,
                                         strides));
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(grads_desc, CUDNN_DATA_FLOAT, 3, dims,
                                         strides));
  cudnnCTCLossDescriptor_t ctc = nullptr;
  CHECK_CUDNN(cudnnCreateCTCLossDescriptor(&ctc));
  CHECK_CUDNN(cudnnSetCTCLossDescriptorEx(ctc, CUDNN_DATA_FLOAT,
                                          CUDNN_LOSS_NORMALIZATION_SOFTMAX,
                                          CUDNN_PROPAGATE_NAN));
  const int labels[] = {1};
  const int label_lengths[] = {1};
  const int input_lengths[] = {steps};
  size_t workspace_size = 0;
  CHECK_CUDNN(cudnnGetCTCLossWorkspaceSize(
      handle, probs_desc, grads_desc, labels, label_lengths, input_lengths,
      CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc, &workspace_size));
  void *probs = nullptr, *grads = nullptr, *costs = nullptr,
       *workspace = nullptr;
  if (upload(logits, &probs)) {
    return 1;
  }
  CHECK_CUDA(cudaMalloc(&grads, logits.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&costs, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  CHECK_CUDNN(cudnnCTCLoss(handle, probs_desc, probs, labels, label_lengths,
                           input_lengths, costs, grads_desc, grads,
                           CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc, workspace,
                           workspace_size));
  std::vector<float> cost(1);
  if (download(costs, cost)) {
    return 1;
  }

  // Forward algorithm over the extended label [blank, 1, blank].
  std::vector<double> p(steps * classes);
  for (int t = 0; t < steps; ++t) {
    double sum = 0;
    for (int k = 0; k < classes; ++k) {
      sum += std::exp(static_cast<double>(logits[t * classes + k]));
    }
    for (int k = 0; k < classes; ++k) {
      p[t * classes + k] = std::exp(logits[t * classes + k]) / sum;
    }
  }
  const int extended[] = {0, 1, 0};
  double alpha[3] = {p[0], p[1], 0};
  for (int t = 1; t < steps; ++t) {
    double next[3];
    for (int s = 0; s < 3; ++s) {
      double sum = alpha[s] + (s > 0 ? alpha[s - 1] : 0);
      next[s] = sum * p[t * classes + extended[s]];
    }
    memcpy(alpha, next, sizeof(alpha));
  }
  const float want = static_cast<float>(-std::log(alpha[1] + alpha[2]));
  if (compare(cost, {want}, "ctc cost")) {
    return 1;
  }
  CHECK_CUDA(cudaFree(probs));
  CHECK_CUDA(cudaFree(grads));
  CHECK_CUDA(cudaFree(costs));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDNN(cudnnDestroyCTCLossDescriptor(ctc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(grads_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(probs_desc));
  printf("ctc loss: passed\n");
  return 0;
}

static int make_seq_data(int time, int batch, int vector,
                         const std::vector<int> &lengths,
                         cudnnSeqDataDescriptor_t *desc) {
  int dims[CUDNN_SEQDATA_DIM_COUNT] = {};
  dims[CUDNN_SEQDATA_TIME_DIM] = time;
  dims[CUDNN_SEQDATA_BATCH_DIM] = batch;
  dims[CUDNN_SEQDATA_BEAM_DIM] = 1;
  dims[CUDNN_SEQDATA_VECT_DIM] = vector;
  const cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT] = {
      CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_TIME_DIM,
      CUDNN_SEQDATA_VECT_DIM};
  CHECK_CUDNN(cudnnCreateSeqDataDescriptor(desc));
  CHECK_CUDNN(cudnnSetSeqDataDescriptor(*desc, CUDNN_DATA_FLOAT,
                                        CUDNN_SEQDATA_DIM_COUNT, dims, axes,
                                        lengths.size(), lengths.data(),
                                        nullptr));
  return 0;
}

// Single-head attention without projections: each query step attends to every
// key step inside its host window. The window arrays span the query steps, or
// only up to the step an inference call processes.
static int test_multi_head_attention(cudnnHandle_t handle) {
  const int steps = 3, size = 4;
  const double scaler = 0.5;
  cudnnAttnDescriptor_t attn = nullptr;
  CHECK_CUDNN(cudnnCreateAttnDescriptor(&attn));
  CHECK_CUDNN(cudnnSetAttnDescriptor(
      attn, CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, 1, scaler, CUDNN_DATA_FLOAT,
      CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, nullptr, nullptr, size, size, size,
      0, 0, 0, 0, steps, steps, 1, 1));
  size_t weight_size = 0, workspace_size = 0;
  CHECK_CUDNN(cudnnGetMultiHeadAttnBuffers(handle, attn, &weight_size,
                                           &workspace_size, nullptr));
  const std::vector<int> lengths = {steps};
  cudnnSeqDataDescriptor_t q_desc = nullptr, k_desc = nullptr,
                           v_desc = nullptr, o_desc = nullptr;
  if (make_seq_data(steps, 1, size, lengths, &q_desc) ||
      make_seq_data(steps, 1, size, lengths, &k_desc) ||
      make_seq_data(steps, 1, size, lengths, &v_desc) ||
      make_seq_data(steps, 1, size, lengths, &o_desc)) {
    return 1;
  }
  std::vector<float> q(steps * size), k(steps * size), v(steps * size);
  for (int i = 0; i < steps * size; ++i) {
    q[i] = static_cast<float>((i * 5) % 7) * 0.25f - 0.5f;
    k[i] = static_cast<float>((i * 3) % 5) * 0.3f - 0.6f;
    v[i] = static_cast<float>(i % 6) - 2.0f;
  }
  void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr,
       *qo_lengths = nullptr, *kv_lengths = nullptr, *workspace = nullptr;
  std::vector<float> zeros(steps * size, 0);
  if (upload(q, &q_dev) || upload(k, &k_dev) || upload(v, &v_dev) ||
      upload(zeros, &o_dev) || upload(lengths, &qo_lengths) ||
      upload(lengths, &kv_lengths)) {
    return 1;
  }
  if (workspace_size != 0) {
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }
  const int lo[] = {0, 0, 0};
  const int hi[] = {steps, steps, steps};
  CHECK_CUDNN(cudnnMultiHeadAttnForward(
      handle, attn, -1, lo, hi, static_cast<const int *>(qo_lengths),
      static_cast<const int *>(kv_lengths), q_desc, q_dev, nullptr, k_desc,
      k_dev, v_desc, v_dev, o_desc, o_dev, weight_size, nullptr,
      workspace_size, workspace, 0, nullptr));
  std::vector<float> want(steps * size, 0);
  for (int t = 0; t < steps; ++t) {
    std::vector<double> scores(steps);
    double total = 0;
    for (int j = 0; j < steps; ++j) {
      double dot = 0;
      for (int d = 0; d < size; ++d) {
        dot += q[t * size + d] * k[j * size + d];
      }
      scores[j] = std::exp(scaler * dot);
      total += scores[j];
    }
    for (int j = 0; j < steps; ++j) {
      for (int d = 0; d < size; ++d) {
        want[t * size + d] +=
            static_cast<float>(scores[j] / total) * v[j * size + d];
      }
    }
  }
  std::vector<float> got(steps * size);
  if (download(o_dev, got) || compare(got, want, "attention")) {
    return 1;
  }

  // Step 1 alone, with windows for steps 0 and 1 only.
  const int lo_step[] = {0, 0};
  const int hi_step[] = {steps, steps};
  if (upload(zeros, &o_dev)) {
    return 1;
  }
  CHECK_CUDNN(cudnnMultiHeadAttnForward(
      handle, attn, 1, lo_step, hi_step, static_cast<const int *>(qo_lengths),
      static_cast<const int *>(kv_lengths), q_desc, q_dev, nullptr, k_desc,
      k_dev, v_desc, v_dev, o_desc, o_dev, weight_size, nullptr,
      workspace_size, workspace, 0, nullptr));
  if (download(o_dev, got)) {
    return 1;
  }
  for (int d = 0; d < size; ++d) {
    EXPECT(std::fabs(got[size + d] - want[size + d]) < 1e-4);
  }

  for (void *buffer :
       {q_dev, k_dev, v_dev, o_dev, qo_lengths, kv_lengths, workspace}) {
    CHECK_CUDA(cudaFree(buffer));
  }
  for (cudnnSeqDataDescriptor_t desc : {q_desc, k_desc, v_desc, o_desc}) {
    CHECK_CUDNN(cudnnDestroySeqDataDescriptor(desc));
  }
  CHECK_CUDNN(cudnnDestroyAttnDescriptor(attn));
  printf("multi-head attention: passed\n");
  return 0;
}

static int make_tensor(int64_t id, const int64_t dims[4], const int64_t strides[4],
                       cudnnBackendDescriptor_t *tensor) {
  const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  const int64_t alignment = 4;
  CHECK_CUDNN(
      cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, tensor));
  CHECK_CUDNN(cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_DATA_TYPE,
                                       CUDNN_TYPE_DATA_TYPE, 1, &type));
  CHECK_CUDNN(cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64, 4, dims));
  CHECK_CUDNN(cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_STRIDES,
                                       CUDNN_TYPE_INT64, 4, strides));
  CHECK_CUDNN(cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                       CUDNN_TYPE_INT64, 1, &id));
  CHECK_CUDNN(cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                       CUDNN_TYPE_INT64, 1, &alignment));
  CHECK_CUDNN(cudnnBackendFinalize(*tensor));
  return 0;
}

// The same convolution as the legacy test, built as a backend operation
// graph: every attribute type the graph needs crosses the wire, heuristics
// fill descriptors the caller created, and the plan runs on a variant pack.
static int test_backend_graph(cudnnHandle_t handle) {
  const int64_t x_dims[] = {1, 1, kImage, kImage};
  const int64_t x_strides[] = {kImage * kImage, kImage * kImage, kImage, 1};
  const int64_t w_dims[] = {1, 1, kFilter, kFilter};
  const int64_t w_strides[] = {kFilter * kFilter, kFilter * kFilter, kFilter, 1};
  cudnnBackendDescriptor_t x = nullptr, w = nullptr, y = nullptr;
  if (make_tensor('x', x_dims, x_strides, &x) ||
      make_tensor('w', w_dims, w_strides, &w) ||
      make_tensor('y', x_dims, x_strides, &y)) {
    return 1;
  }

  int64_t got_dims[8] = {};
  int64_t dims_count = 0;
  CHECK_CUDNN(cudnnBackendGetAttribute(x, CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64, 8, &dims_count,
                                       got_dims));
  EXPECT(dims_count == 4 && memcmp(got_dims, x_dims, sizeof(x_dims)) == 0);
  cudnnDataType_t got_type = CUDNN_DATA_DOUBLE;
  CHECK_CUDNN(cudnnBackendGetAttribute(x, CUDNN_ATTR_TENSOR_DATA_TYPE,
                                       CUDNN_TYPE_DATA_TYPE, 1, &dims_count,
                                       &got_type));
  EXPECT(got_type == CUDNN_DATA_FLOAT);

  const cudnnDataType_t comp_type = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const int64_t spatial = 2;
  const int64_t ones[] = {1, 1};
  cudnnBackendDescriptor_t conv = nullptr;
  CHECK_CUDNN(
      cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE, 1, &comp_type));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                       CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                       CUDNN_TYPE_INT64, 1, &spatial));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                       CUDNN_TYPE_INT64, 2, ones));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                       CUDNN_TYPE_INT64, 2, ones));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                       CUDNN_TYPE_INT64, 2, ones));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                       CUDNN_TYPE_INT64, 2, ones));
  CHECK_CUDNN(cudnnBackendFinalize(conv));

  const float alpha = 2.0f, beta = 0.0f;
  cudnnBackendDescriptor_t op = nullptr;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op));
  CHECK_CUDNN(cudnnBackendSetAttribute(
      op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv));
  CHECK_CUDNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x));
  CHECK_CUDNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w));
  CHECK_CUDNN(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y));
  CHECK_CUDNN(cudnnBackendSetAttribute(
      op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1,
      &alpha));
  CHECK_CUDNN(cudnnBackendSetAttribute(
      op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1,
      &beta));
  CHECK_CUDNN(cudnnBackendFinalize(op));

  cudnnBackendDescriptor_t graph = nullptr;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph));
  CHECK_CUDNN(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  CHECK_CUDNN(cudnnBackendSetAttribute(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                       CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendFinalize(graph));

  const cudnnBackendHeurMode_t heur_mode = CUDNN_HEUR_MODE_A;
  cudnnBackendDescriptor_t heur = nullptr;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));
  CHECK_CUDNN(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph));
  CHECK_CUDNN(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE,
                                       CUDNN_TYPE_HEUR_MODE, 1, &heur_mode));
  CHECK_CUDNN(cudnnBackendFinalize(heur));
  int64_t result_count = 0;
  CHECK_CUDNN(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 0,
                                       &result_count, nullptr));
  EXPECT(result_count > 0);
  std::vector<cudnnBackendDescriptor_t> configs(result_count);
  for (auto &config : configs) {
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &config));
  }
  int64_t filled = 0;
  CHECK_CUDNN(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       result_count, &filled, configs.data()));
  EXPECT(filled > 0);

  const std::vector<float> image = test_image<float>();
  const std::vector<float> filter = test_filter<float>();
  void *x_dev = nullptr, *w_dev = nullptr, *y_dev = nullptr;
  std::vector<float> zeros(kImage * kImage, 0);
  if (upload(image, &x_dev) || upload(filter, &w_dev) ||
      upload(zeros, &y_dev)) {
    return 1;
  }
  cudnnBackendDescriptor_t plan = nullptr;
  int64_t workspace_size = 0;
  for (int64_t i = 0; i < filled && plan == nullptr; ++i) {
    cudnnBackendDescriptor_t candidate = nullptr;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &candidate));
    CHECK_CUDNN(cudnnBackendSetAttribute(candidate, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                         CUDNN_TYPE_HANDLE, 1, &handle));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[i]));
    if (cudnnBackendFinalize(candidate) == CUDNN_STATUS_SUCCESS) {
      plan = candidate;
    } else {
      CHECK_CUDNN(cudnnBackendDestroyDescriptor(candidate));
    }
  }
  EXPECT(plan != nullptr);
  int64_t count = 0;
  CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                       CUDNN_TYPE_INT64, 1, &count,
                                       &workspace_size));
  void *workspace = nullptr;
  if (workspace_size != 0) {
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }
  const int64_t ids[] = {'x', 'w', 'y'};
  void *pointers[] = {x_dev, w_dev, y_dev};
  cudnnBackendDescriptor_t pack = nullptr;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack));
  CHECK_CUDNN(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       CUDNN_TYPE_INT64, 3, ids));
  CHECK_CUDNN(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       CUDNN_TYPE_VOID_PTR, 3, pointers));
  CHECK_CUDNN(cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                       CUDNN_TYPE_VOID_PTR, 1, &workspace));
  CHECK_CUDNN(cudnnBackendFinalize(pack));
#if CUDNN_VERSION >= 92300
  size_t plan_workspace = 0;
  CHECK_CUDNN(
      cudnnGetExecutionPlanWorkspaceSize(handle, plan, pack, &plan_workspace));
  EXPECT(plan_workspace <= static_cast<size_t>(workspace_size));
#endif
  CHECK_CUDNN(cudnnBackendExecute(handle, plan, pack));
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> got(kImage * kImage);
  if (download(y_dev, got) ||
      compare(got, reference_conv(image, filter, 2.0f), "backend conv")) {
    return 1;
  }

  CHECK_CUDA(cudaFree(x_dev));
  CHECK_CUDA(cudaFree(w_dev));
  CHECK_CUDA(cudaFree(y_dev));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(pack));
  CHECK_CUDNN(cudnnBackendDestroyDescriptor(plan));
  for (auto &config : configs) {
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(config));
  }
  for (cudnnBackendDescriptor_t descriptor : {heur, graph, op, conv, y, w, x}) {
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(descriptor));
  }
  printf("backend graph (%lld engine configs): passed\n",
         static_cast<long long>(filled));
  return 0;
}

static int test_device(int device) {
  CHECK_CUDA(cudaSetDevice(device));
  cudnnHandle_t handle = nullptr;
  CHECK_CUDNN(cudnnCreate(&handle));
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDNN(cudnnSetStream(handle, stream));
  cudaStream_t got_stream = nullptr;
  CHECK_CUDNN(cudnnGetStream(handle, &got_stream));
  EXPECT(got_stream == stream);
  char name[32];
  snprintf(name, sizeof(name), "device %d conv", device);
  if (legacy_conv<float>(handle, CUDNN_DATA_FLOAT, 1.0f, 0.0f, name)) {
    return 1;
  }
  CHECK_CUDNN(cudnnDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}

int main() {
  const size_t version = cudnnGetVersion();
  EXPECT(version >= 90000);
  int major = 0;
  CHECK_CUDNN(cudnnGetProperty(MAJOR_VERSION, &major));
  EXPECT(major == static_cast<int>(version / 10000));
  EXPECT(cudnnGetCudartVersion() > 0);
  EXPECT(strcmp(cudnnGetErrorString(CUDNN_STATUS_BAD_PARAM),
                "CUDNN_STATUS_BAD_PARAM") == 0);
  CHECK_CUDNN(cudnnGraphVersionCheck());
  CHECK_CUDNN(cudnnOpsVersionCheck());
  CHECK_CUDNN(cudnnCnnVersionCheck());
  CHECK_CUDNN(cudnnAdvVersionCheck());
  // A callback would run in the client, so the shim does not take one.
  const cudnnStatus_t callback = cudnnSetCallback(0, nullptr, nullptr);
  EXPECT(callback == CUDNN_STATUS_NOT_SUPPORTED ||
         callback == CUDNN_STATUS_SUCCESS);

  cudnnHandle_t handle = nullptr;
  CHECK_CUDNN(cudnnCreate(&handle));
  if (legacy_conv<float>(handle, CUDNN_DATA_FLOAT, 1.0f, 0.5f,
                         "float conv") ||
      legacy_conv<double>(handle, CUDNN_DATA_DOUBLE, 2.0, 0.25,
                          "double conv") ||
      test_descriptors(handle) || test_pointwise_ops(handle) ||
      test_derived_descriptor(handle) || test_ctc_loss(handle) ||
      test_multi_head_attention(handle) ||
      test_backend_graph(handle)) {
    return 1;
  }
  char message[256] = {};
  cudnnGetLastErrorString(message, sizeof(message));
  CHECK_CUDNN(cudnnDestroy(handle));

  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  for (int device = 1; device < devices; ++device) {
    if (test_device(device)) {
      return 1;
    }
  }
  if (devices < 2) {
    printf("second device: skipped, one device\n");
  }
  printf("cudnn shim: all checks passed (cuDNN %zu)\n", version);
  return 0;
}
