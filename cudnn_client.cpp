// cuDNN (libcudnn) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcudnn, sent on the
// CUDA driver shim's connections: the server child that owns the driver and
// runtime state runs the library too, and one lane per client thread keeps
// the APIs ordered. Nearly all of the surface is generated. The calls in this
// file carry something the generated marshalling cannot: a static string, a
// backend attribute whose element width depends on its type, or a host array
// whose length a descriptor holds. A callback into the client is a generated
// stub that returns CUDNN_STATUS_NOT_SUPPORTED.

#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

namespace {

cudnnStatus_t rpc_error() { return CUDNN_STATUS_NOT_INITIALIZED; }

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  return lupine_rpc_write_start_request(conn, op);
}
int rpc_write(conn_t *conn, const void *data, size_t size) {
  return lupine_rpc_write(conn, data, size);
}
int rpc_wait_for_response(conn_t *conn) {
  return lupine_rpc_wait_for_response(conn);
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  return lupine_rpc_read(conn, data, size);
}
int rpc_read_end(conn_t *conn) { return lupine_rpc_read_end(conn); }

// rpc_write queues the address and sends at rpc_wait_for_response, so every
// value must be storage the caller keeps alive until the response is read.
int write_values(conn_t *) { return 0; }
template <typename T, typename... Rest>
int write_values(conn_t *conn, const T &value, const Rest &...rest) {
  return rpc_write(conn, &value, sizeof(T)) < 0 ? -1
                                                : write_values(conn, rest...);
}

// ---------------------------------------------------------------------------
// Handles and descriptors
// ---------------------------------------------------------------------------

// A handle or descriptor is created on the runtime's current device and
// routes every later call to that connection. A data descriptor also keeps
// the type and shape it was last set to: a scaling factor's width follows the
// type, and the CTC loss sizes its host arrays by the batch. An attention
// descriptor keeps the query length its window arrays span.
struct descriptor_state {
  conn_t *conn = nullptr;
  int type = -1;
  int nb_dims = 0;
  int dims[CUDNN_DIM_MAX] = {};
  int windows = 0;
};

std::mutex descriptors_mutex;
std::unordered_map<const void *, descriptor_state> descriptors;

// A call without a handle goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(const void *handle) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(handle);
  return it == descriptors.end() ? nullptr : it->second.conn;
}

conn_t *connection_for_stream(cudaStream_t stream) {
  return stream == nullptr ? connection() : lupine_rpc_conn_for_stream(stream);
}

// A destroyed handle's address may come back from a later create, which
// records it afresh.
void note_handle_owner(conn_t *conn, const void *handle) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors[handle] = descriptor_state{conn};
}

void forget_handle(const void *handle) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors.erase(handle);
}

void note_data_type(const void *descriptor, cudnnDataType_t type, int nb_dims,
                    const int *dims) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(descriptor);
  if (it == descriptors.end()) {
    return;
  }
  it->second.type = type;
  it->second.nb_dims = std::max(0, std::min(nb_dims, CUDNN_DIM_MAX));
  std::copy(dims, dims + it->second.nb_dims, it->second.dims);
}

// The library rewrote the descriptor, so its type is asked for again.
void forget_data_type(const void *descriptor) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(descriptor);
  if (it != descriptors.end()) {
    it->second.type = -1;
  }
}

void note_attn_windows(const void *descriptor, int windows) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(descriptor);
  if (it != descriptors.end()) {
    it->second.windows = windows;
  }
}

bool recorded_state(const void *descriptor, descriptor_state *state) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(descriptor);
  if (it == descriptors.end()) {
    return false;
  }
  *state = it->second;
  return true;
}

descriptor_state tensor_state(cudnnTensorDescriptor_t descriptor) {
  descriptor_state state;
  if (recorded_state(descriptor, &state) && state.type >= 0) {
    return state;
  }
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  int nb_dims = 0;
  int dims[CUDNN_DIM_MAX] = {};
  int strides[CUDNN_DIM_MAX] = {};
  if (cudnnGetTensorNdDescriptor(descriptor, CUDNN_DIM_MAX, &type, &nb_dims,
                                 dims, strides) == CUDNN_STATUS_SUCCESS) {
    note_data_type(descriptor, type, nb_dims, dims);
  }
  state.type = type;
  state.nb_dims = std::max(0, std::min(nb_dims, CUDNN_DIM_MAX));
  std::copy(dims, dims + state.nb_dims, state.dims);
  return state;
}

int filter_type(cudnnFilterDescriptor_t descriptor) {
  descriptor_state state;
  if (recorded_state(descriptor, &state) && state.type >= 0) {
    return state.type;
  }
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  int nb_dims = 0;
  int dims[CUDNN_DIM_MAX] = {};
  if (cudnnGetFilterNdDescriptor(descriptor, CUDNN_DIM_MAX, &type, &format,
                                 &nb_dims, dims) == CUDNN_STATUS_SUCCESS) {
    note_data_type(descriptor, type, 0, nullptr);
  }
  return type;
}

// ---------------------------------------------------------------------------
// Host scalars
// ---------------------------------------------------------------------------

// Scaling factors and fill values always live on the host.
bool scalar_on_host(const void *, const char *) { return true; }

size_t data_type_width(int type) {
  switch (type) {
  case CUDNN_DATA_DOUBLE:
  case CUDNN_DATA_INT64:
    return 8;
  case CUDNN_DATA_FLOAT:
  case CUDNN_DATA_INT32:
    return 4;
  case CUDNN_DATA_HALF:
  case CUDNN_DATA_BFLOAT16:
    return 2;
#if CUDNN_VERSION >= 91100
  case CUDNN_DATA_UINT32:
    return 4;
#endif
#if CUDNN_VERSION >= 91400
  case CUDNN_DATA_COMPLEX_FP32:
    return 8;
  case CUDNN_DATA_COMPLEX_FP64:
    return 16;
#endif
  default:
    return 1;
  }
}

// The library reads a double scaling factor for double data and a float for
// every other type.
size_t scaling_width_for(int type) {
  return type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
}

size_t scaling_width(cudnnTensorDescriptor_t descriptor) {
  return scaling_width_for(tensor_state(descriptor).type);
}

size_t scaling_width(cudnnFilterDescriptor_t descriptor) {
  return scaling_width_for(filter_type(descriptor));
}

size_t element_width(cudnnTensorDescriptor_t descriptor) {
  return data_type_width(tensor_state(descriptor).type);
}

// The padding fill of an RNN or sequence data descriptor has the type it was
// set with; nothing but its setter gives it one.
size_t recorded_width(const void *descriptor) {
  descriptor_state state;
  return data_type_width(recorded_state(descriptor, &state) && state.type >= 0
                             ? state.type
                             : CUDNN_DATA_FLOAT);
}

} // namespace

#include "codegen/gen_cudnn_client.inc"

// ---------------------------------------------------------------------------
// Static strings
// ---------------------------------------------------------------------------

// The server's library owns the strings, so each is fetched once and kept for
// the life of the process.
extern "C" const char *cudnnGetErrorString(cudnnStatus_t status) {
  static std::mutex mutex;
  static std::unordered_map<int, std::string> texts;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = texts.find(status);
    if (it != texts.end()) {
      return it->second.c_str();
    }
  }
  constexpr uint32_t kMaxLength = 4096;
  conn_t *conn = connection();
  uint32_t length = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudnnGetErrorString) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxLength) {
    return "CUDNN_UNKNOWN_STATUS";
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return "CUDNN_UNKNOWN_STATUS";
  }
  std::lock_guard<std::mutex> lock(mutex);
  return texts.emplace(status, std::move(text)).first->second.c_str();
}

// ---------------------------------------------------------------------------
// Backend attributes
// ---------------------------------------------------------------------------

namespace {

// Every attribute type not named is an enum, a float or an int32.
uint64_t attribute_width(cudnnBackendAttributeType_t type) {
  switch (type) {
  case CUDNN_TYPE_HANDLE:
    return sizeof(cudnnHandle_t);
  case CUDNN_TYPE_BOOLEAN:
    return sizeof(bool);
  case CUDNN_TYPE_INT64:
    return sizeof(int64_t);
  case CUDNN_TYPE_DOUBLE:
    return sizeof(double);
  case CUDNN_TYPE_VOID_PTR:
    return sizeof(void *);
  case CUDNN_TYPE_BACKEND_DESCRIPTOR:
    return sizeof(cudnnBackendDescriptor_t);
  case CUDNN_TYPE_CHAR:
    return sizeof(char);
  case CUDNN_TYPE_FRACTION:
    return sizeof(cudnnFraction_t);
  default:
    return sizeof(int32_t);
  }
}

} // namespace

// Elements travel as the bytes the type makes them: handles, descriptors and
// device pointers as the server's values, everything else as data.
extern "C" cudnnStatus_t
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t elementCount, const void *arrayOfElements) {
  if (elementCount < 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const uint8_t present = arrayOfElements != nullptr;
  const uint64_t bytes = present ? static_cast<uint64_t>(elementCount) *
                                       attribute_width(attributeType)
                                 : 0;
  conn_t *conn = connection_for_handle(descriptor);
  cudnnStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudnnBackendSetAttribute) < 0 ||
      write_values(conn, descriptor, attributeName, attributeType, elementCount,
                   present, bytes) < 0 ||
      (bytes != 0 && rpc_write(conn, arrayOfElements, bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

// A descriptor-typed get copies into descriptors the caller created, so those
// handles are sent along; the answer carries at most the elements requested.
extern "C" cudnnStatus_t
cudnnBackendGetAttribute(const cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount, int64_t *elementCount,
                         void *arrayOfElements) {
  if (requestedElementCount < 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const uint64_t width = attribute_width(attributeType);
  const uint8_t count_present = elementCount != nullptr;
  const uint8_t array_present = arrayOfElements != nullptr;
  const uint64_t sent =
      array_present && attributeType == CUDNN_TYPE_BACKEND_DESCRIPTOR
          ? static_cast<uint64_t>(requestedElementCount) * width
          : 0;
  conn_t *conn = connection_for_handle(descriptor);
  cudnnStatus_t status = rpc_error();
  uint64_t returned = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudnnBackendGetAttribute) < 0 ||
      write_values(conn, descriptor, attributeName, attributeType,
                   requestedElementCount, width, count_present, array_present,
                   sent) < 0 ||
      (sent != 0 && rpc_write(conn, arrayOfElements, sent) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      (count_present &&
       rpc_read(conn, elementCount, sizeof(*elementCount)) < 0) ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (returned != 0 && rpc_read(conn, arrayOfElements, returned) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

// ---------------------------------------------------------------------------
// Host arrays sized by a descriptor
// ---------------------------------------------------------------------------

namespace {

// The label lengths and input lengths have one entry per batch member, the
// probabilities' second dimension, and the labels their sum.
struct ctc_arrays {
  uint32_t batch = 0;
  uint32_t labels = 0;
  uint8_t labels_present = 0;
  uint8_t label_lengths_present = 0;
  uint8_t input_lengths_present = 0;
};

ctc_arrays ctc_layout(cudnnTensorDescriptor_t probsDesc, const int *labels,
                      const int *labelLengths, const int *inputLengths) {
  ctc_arrays layout;
  const descriptor_state probs = tensor_state(probsDesc);
  layout.batch = probs.nb_dims > 1 ? static_cast<uint32_t>(probs.dims[1]) : 0;
  for (uint32_t i = 0; labelLengths != nullptr && i < layout.batch; ++i) {
    layout.labels += static_cast<uint32_t>(labelLengths[i]);
  }
  layout.labels_present = labels != nullptr;
  layout.label_lengths_present = labelLengths != nullptr;
  layout.input_lengths_present = inputLengths != nullptr;
  return layout;
}

int write_ctc_arrays(conn_t *conn, const ctc_arrays &layout, const int *labels,
                     const int *labelLengths, const int *inputLengths) {
  const size_t batch_bytes = layout.batch * sizeof(int);
  const size_t label_bytes = layout.labels * sizeof(int);
  return write_values(conn, layout.batch, layout.labels, layout.labels_present,
                      layout.label_lengths_present,
                      layout.input_lengths_present) < 0 ||
                 (layout.labels_present && label_bytes != 0 &&
                  rpc_write(conn, labels, label_bytes) < 0) ||
                 (layout.label_lengths_present && batch_bytes != 0 &&
                  rpc_write(conn, labelLengths, batch_bytes) < 0) ||
                 (layout.input_lengths_present && batch_bytes != 0 &&
                  rpc_write(conn, inputLengths, batch_bytes) < 0)
             ? -1
             : 0;
}

int attn_windows(cudnnAttnDescriptor_t descriptor) {
  descriptor_state state;
  return recorded_state(descriptor, &state) ? state.windows : 0;
}

} // namespace

extern "C" cudnnStatus_t cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
  if (sizeInBytes == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const ctc_arrays layout =
      ctc_layout(probsDesc, labels, labelLengths, inputLengths);
  conn_t *conn = connection_for_handle(handle);
  cudnnStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudnnGetCTCLossWorkspaceSize) < 0 ||
      write_values(conn, handle, probsDesc, gradientsDesc, algo, ctcLossDesc) <
          0 ||
      write_ctc_arrays(conn, layout, labels, labelLengths, inputLengths) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, sizeInBytes, sizeof(*sizeInBytes)) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" cudnnStatus_t
cudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
             const void *probs, const int hostLabels[],
             const int hostLabelLengths[], const int hostInputLengths[],
             void *costs, const cudnnTensorDescriptor_t gradientsDesc,
             void *gradients, cudnnCTCLossAlgo_t algo,
             cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace,
             size_t workSpaceSizeInBytes) {
  const ctc_arrays layout =
      ctc_layout(probsDesc, hostLabels, hostLabelLengths, hostInputLengths);
  conn_t *conn = connection_for_handle(handle);
  cudnnStatus_t status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cudnnCTCLoss) < 0 ||
      write_values(conn, handle, probsDesc, probs, costs, gradientsDesc,
                   gradients, algo, ctcLossDesc, workspace,
                   workSpaceSizeInBytes) < 0 ||
      write_ctc_arrays(conn, layout, hostLabels, hostLabelLengths,
                       hostInputLengths) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

// The window arrays have an entry per query time step: all of them, or up to
// the one step an inference call processes.
extern "C" cudnnStatus_t cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx,
    const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsQO[],
    const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  const uint32_t windows = currIdx < 0 ? attn_windows(attnDesc) : currIdx + 1;
  const size_t window_bytes = windows * sizeof(int);
  conn_t *conn = connection_for_handle(handle);
  cudnnStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudnnMultiHeadAttnForward) < 0 ||
      write_values(conn, handle, attnDesc, currIdx, devSeqLengthsQO,
                   devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys,
                   vDesc, values, oDesc, out, weightSizeInBytes, weights,
                   workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                   reserveSpace, windows) < 0 ||
      (window_bytes != 0 && (rpc_write(conn, loWinIdx, window_bytes) < 0 ||
                             rpc_write(conn, hiWinIdx, window_bytes) < 0)) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc, void *dkeys,
    const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  const uint32_t windows = attn_windows(attnDesc);
  const size_t window_bytes = windows * sizeof(int);
  conn_t *conn = connection_for_handle(handle);
  cudnnStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudnnMultiHeadAttnBackwardData) < 0 ||
      write_values(conn, handle, attnDesc, devSeqLengthsDQDO, devSeqLengthsDKDV,
                   doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys,
                   dvDesc, dvalues, values, weightSizeInBytes, weights,
                   workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                   reserveSpace, windows) < 0 ||
      (window_bytes != 0 && (rpc_write(conn, loWinIdx, window_bytes) < 0 ||
                             rpc_write(conn, hiWinIdx, window_bytes) < 0)) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}
