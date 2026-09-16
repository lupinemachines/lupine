#include <cudnn.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

cudnnStatus_t function_not_found() { return CUDNN_STATUS_NOT_SUPPORTED; }

// The library this server was compiled against; the handlers marshal its
// enums and scaling factors, so another major on the machine is not a match.
// libcudnn loads its operation sublibraries itself.
void *cudnn_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUDNN_MAJOR);
#ifdef _WIN32
    const std::string name = "cudnn64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcudnn.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cudnn_symbol(const char *name) {
  void *lib = cudnn_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

int read_values(conn_t *) { return 0; }
template <typename T, typename... Rest>
int read_values(conn_t *conn, T &value, Rest &...rest) {
  return rpc_read(conn, &value, sizeof(T)) < 0 ? -1
                                               : read_values(conn, rest...);
}

int write_status(conn_t *conn, int request_id, cudnnStatus_t status) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// The host arrays of the CTC loss: labels, then one length per batch member
// for the labels and for the inputs, each absent when the caller passed null.
struct ctc_arrays {
  std::vector<int> labels;
  std::vector<int> label_lengths;
  std::vector<int> input_lengths;
  uint8_t labels_present = 0;
  uint8_t label_lengths_present = 0;
  uint8_t input_lengths_present = 0;

  const int *labels_or_null() const {
    return labels_present ? labels.data() : nullptr;
  }
  const int *label_lengths_or_null() const {
    return label_lengths_present ? label_lengths.data() : nullptr;
  }
  const int *input_lengths_or_null() const {
    return input_lengths_present ? input_lengths.data() : nullptr;
  }
};

int read_ints(conn_t *conn, std::vector<int> &values, uint32_t count,
              uint8_t present) {
  values.resize(present ? count : 0);
  return values.empty() ? 0
                        : rpc_read(conn, values.data(), count * sizeof(int));
}

int read_ctc_arrays(conn_t *conn, ctc_arrays &arrays) {
  uint32_t batch = 0;
  uint32_t labels = 0;
  if (read_values(conn, batch, labels, arrays.labels_present,
                  arrays.label_lengths_present,
                  arrays.input_lengths_present) < 0 ||
      read_ints(conn, arrays.labels, labels, arrays.labels_present) < 0 ||
      read_ints(conn, arrays.label_lengths, batch,
                arrays.label_lengths_present) < 0 ||
      read_ints(conn, arrays.input_lengths, batch,
                arrays.input_lengths_present) < 0) {
    return -1;
  }
  return 0;
}

int read_windows(conn_t *conn, std::vector<int> &lo, std::vector<int> &hi) {
  uint32_t windows = 0;
  if (rpc_read(conn, &windows, sizeof(windows)) < 0) {
    return -1;
  }
  lo.resize(windows);
  hi.resize(windows);
  if (windows != 0 && (rpc_read(conn, lo.data(), windows * sizeof(int)) < 0 ||
                       rpc_read(conn, hi.data(), windows * sizeof(int)) < 0)) {
    return -1;
  }
  return 0;
}

} // namespace

// cudnnGetErrorString returns a static string, so the response carries the
// bytes length-prefixed instead of a status.
int handle_cudnnGetErrorString(conn_t *conn) {
  cudnnStatus_t status;
  if (rpc_read(conn, &status, sizeof(status)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(cudnnStatus_t);
  fn_t fn = cudnn_symbol<fn_t>("cudnnGetErrorString");
  const char *text = fn == nullptr ? nullptr : fn(status);
  uint32_t length = text == nullptr ? 0 : static_cast<uint32_t>(strlen(text));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, text, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudnnBackendSetAttribute(conn_t *conn) {
  cudnnBackendDescriptor_t descriptor;
  cudnnBackendAttributeName_t name;
  cudnnBackendAttributeType_t type;
  int64_t count;
  uint8_t present;
  uint64_t bytes;
  if (read_values(conn, descriptor, name, type, count, present, bytes) < 0) {
    return -1;
  }
  std::vector<unsigned char> elements(bytes);
  if (bytes != 0 && rpc_read(conn, elements.data(), bytes) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t =
      cudnnStatus_t (*)(cudnnBackendDescriptor_t, cudnnBackendAttributeName_t,
                        cudnnBackendAttributeType_t, int64_t, const void *);
  fn_t fn = cudnn_symbol<fn_t>("cudnnBackendSetAttribute");
  return write_status(conn, request_id,
                      fn == nullptr ? function_not_found()
                                    : fn(descriptor, name, type, count,
                                         present ? elements.data() : nullptr));
}

// The element storage starts as the descriptors the caller sent, or zeros; the
// answer is clamped to the elements requested.
int handle_cudnnBackendGetAttribute(conn_t *conn) {
  cudnnBackendDescriptor_t descriptor;
  cudnnBackendAttributeName_t name;
  cudnnBackendAttributeType_t type;
  int64_t requested;
  uint64_t width;
  uint8_t count_present;
  uint8_t array_present;
  uint64_t sent;
  if (read_values(conn, descriptor, name, type, requested, width, count_present,
                  array_present, sent) < 0) {
    return -1;
  }
  std::vector<unsigned char> elements(
      array_present ? static_cast<uint64_t>(requested) * width : 0);
  if (sent != 0 && rpc_read(conn, elements.data(), sent) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cudnnStatus_t (*)(
      const cudnnBackendDescriptor_t, cudnnBackendAttributeName_t,
      cudnnBackendAttributeType_t, int64_t, int64_t *, void *);
  fn_t fn = cudnn_symbol<fn_t>("cudnnBackendGetAttribute");
  int64_t count = 0;
  cudnnStatus_t status = fn == nullptr
                             ? function_not_found()
                             : fn(descriptor, name, type, requested,
                                  count_present ? &count : nullptr,
                                  array_present ? elements.data() : nullptr);
  uint64_t returned = 0;
  if (array_present) {
    const int64_t elements_returned =
        count_present ? std::max<int64_t>(0, std::min(count, requested))
                      : requested;
    returned = static_cast<uint64_t>(elements_returned) * width;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      (count_present && rpc_write(conn, &count, sizeof(count)) < 0) ||
      rpc_write(conn, &returned, sizeof(returned)) < 0 ||
      (returned != 0 && rpc_write(conn, elements.data(), returned) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudnnGetCTCLossWorkspaceSize(conn_t *conn) {
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t probs_desc;
  cudnnTensorDescriptor_t gradients_desc;
  cudnnCTCLossAlgo_t algo;
  cudnnCTCLossDescriptor_t ctc_desc;
  ctc_arrays arrays;
  if (read_values(conn, handle, probs_desc, gradients_desc, algo, ctc_desc) <
          0 ||
      read_ctc_arrays(conn, arrays) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t,
                                 const cudnnTensorDescriptor_t, const int *,
                                 const int *, const int *, cudnnCTCLossAlgo_t,
                                 cudnnCTCLossDescriptor_t, size_t *);
  fn_t fn = cudnn_symbol<fn_t>("cudnnGetCTCLossWorkspaceSize");
  size_t size = 0;
  cudnnStatus_t status =
      fn == nullptr
          ? function_not_found()
          : fn(handle, probs_desc, gradients_desc, arrays.labels_or_null(),
               arrays.label_lengths_or_null(), arrays.input_lengths_or_null(),
               algo, ctc_desc, &size);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &size, sizeof(size)) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudnnCTCLoss(conn_t *conn) {
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t probs_desc;
  const void *probs;
  void *costs;
  cudnnTensorDescriptor_t gradients_desc;
  void *gradients;
  cudnnCTCLossAlgo_t algo;
  cudnnCTCLossDescriptor_t ctc_desc;
  void *workspace;
  size_t workspace_size;
  ctc_arrays arrays;
  if (read_values(conn, handle, probs_desc, probs, costs, gradients_desc,
                  gradients, algo, ctc_desc, workspace, workspace_size) < 0 ||
      read_ctc_arrays(conn, arrays) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cudnnStatus_t (*)(
      cudnnHandle_t, const cudnnTensorDescriptor_t, const void *, const int *,
      const int *, const int *, void *, const cudnnTensorDescriptor_t, void *,
      cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, void *, size_t);
  fn_t fn = cudnn_symbol<fn_t>("cudnnCTCLoss");
  return write_status(
      conn, request_id,
      fn == nullptr ? function_not_found()
                    : fn(handle, probs_desc, probs, arrays.labels_or_null(),
                         arrays.label_lengths_or_null(),
                         arrays.input_lengths_or_null(), costs, gradients_desc,
                         gradients, algo, ctc_desc, workspace, workspace_size));
}

int handle_cudnnMultiHeadAttnForward(conn_t *conn) {
  cudnnHandle_t handle;
  cudnnAttnDescriptor_t attn_desc;
  int curr_idx;
  const int *dev_seq_lengths_qo;
  const int *dev_seq_lengths_kv;
  cudnnSeqDataDescriptor_t q_desc;
  const void *queries;
  const void *residuals;
  cudnnSeqDataDescriptor_t k_desc;
  const void *keys;
  cudnnSeqDataDescriptor_t v_desc;
  const void *values;
  cudnnSeqDataDescriptor_t o_desc;
  void *out;
  size_t weight_size;
  const void *weights;
  size_t workspace_size;
  void *workspace;
  size_t reserve_size;
  void *reserve;
  std::vector<int> lo;
  std::vector<int> hi;
  if (read_values(conn, handle, attn_desc, curr_idx, dev_seq_lengths_qo,
                  dev_seq_lengths_kv, q_desc, queries, residuals, k_desc, keys,
                  v_desc, values, o_desc, out, weight_size, weights,
                  workspace_size, workspace, reserve_size, reserve) < 0 ||
      read_windows(conn, lo, hi) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cudnnStatus_t (*)(
      cudnnHandle_t, const cudnnAttnDescriptor_t, int, const int *, const int *,
      const int *, const int *, const cudnnSeqDataDescriptor_t, const void *,
      const void *, const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, void *, size_t, const void *, size_t,
      void *, size_t, void *);
  fn_t fn = cudnn_symbol<fn_t>("cudnnMultiHeadAttnForward");
  return write_status(
      conn, request_id,
      fn == nullptr ? function_not_found()
                    : fn(handle, attn_desc, curr_idx, lo.data(), hi.data(),
                         dev_seq_lengths_qo, dev_seq_lengths_kv, q_desc,
                         queries, residuals, k_desc, keys, v_desc, values,
                         o_desc, out, weight_size, weights, workspace_size,
                         workspace, reserve_size, reserve));
}

int handle_cudnnMultiHeadAttnBackwardData(conn_t *conn) {
  cudnnHandle_t handle;
  cudnnAttnDescriptor_t attn_desc;
  const int *dev_seq_lengths_dqdo;
  const int *dev_seq_lengths_dkdv;
  cudnnSeqDataDescriptor_t do_desc;
  const void *dout;
  cudnnSeqDataDescriptor_t dq_desc;
  void *dqueries;
  const void *queries;
  cudnnSeqDataDescriptor_t dk_desc;
  void *dkeys;
  const void *keys;
  cudnnSeqDataDescriptor_t dv_desc;
  void *dvalues;
  const void *values;
  size_t weight_size;
  const void *weights;
  size_t workspace_size;
  void *workspace;
  size_t reserve_size;
  void *reserve;
  std::vector<int> lo;
  std::vector<int> hi;
  if (read_values(conn, handle, attn_desc, dev_seq_lengths_dqdo,
                  dev_seq_lengths_dkdv, do_desc, dout, dq_desc, dqueries,
                  queries, dk_desc, dkeys, keys, dv_desc, dvalues, values,
                  weight_size, weights, workspace_size, workspace, reserve_size,
                  reserve) < 0 ||
      read_windows(conn, lo, hi) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cudnnStatus_t (*)(
      cudnnHandle_t, const cudnnAttnDescriptor_t, const int *, const int *,
      const int *, const int *, const cudnnSeqDataDescriptor_t, const void *,
      const cudnnSeqDataDescriptor_t, void *, const void *,
      const cudnnSeqDataDescriptor_t, void *, const void *,
      const cudnnSeqDataDescriptor_t, void *, const void *, size_t,
      const void *, size_t, void *, size_t, void *);
  fn_t fn = cudnn_symbol<fn_t>("cudnnMultiHeadAttnBackwardData");
  return write_status(
      conn, request_id,
      fn == nullptr
          ? function_not_found()
          : fn(handle, attn_desc, lo.data(), hi.data(), dev_seq_lengths_dqdo,
               dev_seq_lengths_dkdv, do_desc, dout, dq_desc, dqueries, queries,
               dk_desc, dkeys, keys, dv_desc, dvalues, values, weight_size,
               weights, workspace_size, workspace, reserve_size, reserve));
}

#include "codegen/gen_cudnn_server.inc"
