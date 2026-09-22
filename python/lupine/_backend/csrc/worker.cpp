// Worker side of the torch backend: a table of real CUDA storages keyed by
// the host's handles, and boxed dispatch of the forwarded operators.
#include "backend.h"
#include "wire.h"

#include "dispatch.h"
#include "gen_rpc_ids.h"
#include "lupine_log.h"
#include "rpc.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace py = pybind11;

namespace lupine_torch::worker {

namespace {

constexpr size_t kMaxLanes = 64;

std::mutex g_table_mutex;
std::unordered_map<uint64_t, c10::Storage> g_table;
std::atomic<uint64_t> g_next_handle{UINT64_C(1) << 62};
std::mutex g_error_mutex;
std::string g_error;
std::mutex g_ops_mutex;
std::unordered_map<std::string, c10::OperatorHandle> g_ops;
std::atomic<int64_t> g_stream_id{-1};
std::atomic<int16_t> g_stream_device{0};

void record_error(const std::string &message) {
  LUPINE_LOG_ERROR("lupine torch worker: " << message);
  std::lock_guard<std::mutex> lock(g_error_mutex);
  if (g_error.empty()) {
    g_error = message;
  }
}

std::string take_error() {
  std::lock_guard<std::mutex> lock(g_error_mutex);
  std::string error;
  error.swap(g_error);
  return error;
}

at::Tensor make_tensor(c10::Storage storage, c10::ScalarType dtype,
                       c10::IntArrayRef sizes, c10::IntArrayRef strides,
                       int64_t offset) {
  at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CUDA),
      caffe2::TypeMeta::fromScalarType(dtype));
  t.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides, offset);
  return t;
}

// An unknown handle is a storage the host allocated and never sent an op
// for; it is created here on first use. A known one that is smaller than the
// host believes grew there and grows here in place, keeping its contents.
c10::Storage storage_for(const tensor_desc &d) {
  std::lock_guard<std::mutex> lock(g_table_mutex);
  auto found = g_table.find(d.handle);
  if (found == g_table.end()) {
    at::Tensor bytes = at::empty({static_cast<int64_t>(d.nbytes)},
                                 at::TensorOptions().dtype(at::kByte).device(
                                     c10::DeviceType::CUDA, d.device));
    found = g_table.emplace(d.handle, bytes.storage()).first;
  } else if (d.nbytes > found->second.nbytes()) {
    at::Tensor bytes =
        make_tensor(found->second, at::kByte,
                    {static_cast<int64_t>(found->second.nbytes())}, {1}, 0);
    bytes.resize_({static_cast<int64_t>(d.nbytes)});
  }
  return found->second;
}

at::Tensor materialize(const tensor_desc &d) {
  return make_tensor(storage_for(d), static_cast<c10::ScalarType>(d.dtype),
                     d.sizes, d.strides, d.offset);
}

void bind(uint64_t handle, const at::Tensor &t) {
  if (handle == 0 || !t.defined()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_table_mutex);
  g_table[handle] = t.storage();
}

uint64_t handle_for(const at::Tensor &t) {
  std::lock_guard<std::mutex> lock(g_table_mutex);
  c10::StorageImpl *impl = t.storage().unsafeGetStorageImpl();
  for (const auto &entry : g_table) {
    if (entry.second.unsafeGetStorageImpl() == impl) {
      return entry.first;
    }
  }
  uint64_t handle = g_next_handle.fetch_add(1);
  g_table.emplace(handle, t.storage());
  return handle;
}

tensor_desc describe(const at::Tensor &t) {
  tensor_desc d;
  d.handle = handle_for(t);
  d.nbytes = t.storage().nbytes();
  d.device = t.device().index();
  d.dtype = static_cast<int8_t>(t.scalar_type());
  d.offset = t.storage_offset();
  d.sizes.assign(t.sizes().begin(), t.sizes().end());
  d.strides.assign(t.strides().begin(), t.strides().end());
  return d;
}

const wire_context &wire() {
  static const wire_context ctx = {
      describe, [](const at::Tensor &t) { return t.device().is_cuda(); },
      [](const c10::Device &d) { return d.is_cuda(); }};
  return ctx;
}

const read_context &reading() {
  static const read_context ctx = {materialize, c10::DeviceType::CUDA};
  return ctx;
}

const c10::OperatorHandle &lookup(const std::string &name) {
  std::lock_guard<std::mutex> lock(g_ops_mutex);
  auto found = g_ops.find(name);
  if (found == g_ops.end()) {
    size_t dot = name.find('.');
    std::string base = name.substr(0, dot);
    std::string overload = dot == std::string::npos ? "" : name.substr(dot + 1);
    found = g_ops
                .emplace(name, c10::Dispatcher::singleton().findSchemaOrThrow(
                                   base.c_str(), overload.c_str()))
                .first;
  }
  return found->second;
}

// Every handler runs on the stream Python selected: the capture stream while
// a CUDA graph is being recorded, when even the allocation of a storage a
// descriptor names for the first time must land in the capture; the default
// stream otherwise.
class stream_scope {
public:
  stream_scope() : impl_(c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)) {
    int64_t stream_id = g_stream_id.load();
    if (stream_id >= 0) {
      previous_ = impl_->exchangeStream(c10::Stream::unpack3(
          stream_id, static_cast<c10::DeviceIndex>(g_stream_device.load()),
          c10::DeviceType::CUDA));
    }
  }
  ~stream_scope() {
    if (previous_.has_value()) {
      impl_->exchangeStream(*previous_);
    }
  }

private:
  const c10::impl::DeviceGuardImplInterface *impl_;
  std::optional<c10::Stream> previous_;
};

bool trace_enabled() {
  static const bool enabled = getenv("LUPINE_TORCH_TRACE") != nullptr;
  return enabled;
}

// Ops run on the device of their first tensor argument.
void run(const std::string &name, torch::jit::Stack &stack) {
  const c10::OperatorHandle &op = lookup(name);
  if (trace_enabled()) {
    std::cerr << "lupine-torch worker " << name;
    for (const c10::IValue &v : stack) {
      if (v.isTensor() && v.toTensor().defined()) {
        std::cerr << " " << v.toTensor().sizes();
      }
    }
    std::cerr << std::endl;
  }
  c10::impl::ExcludeDispatchKeyGuard no_autograd(
      c10::autograd_dispatch_keyset_with_ADInplaceOrView);
  c10::DeviceIndex device = 0;
  for (const c10::IValue &v : stack) {
    if (v.isTensor() && v.toTensor().defined() &&
        v.toTensor().device().is_cuda()) {
      device = v.toTensor().device().index();
      break;
    }
    if (v.isDevice() && v.toDevice().is_cuda() && v.toDevice().has_index()) {
      device = v.toDevice().index();
      break;
    }
  }
  c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA, device));
  op.callBoxed(&stack);
}

void synchronize_all() {
  const c10::impl::DeviceGuardImplInterface *impl =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);
  for (c10::DeviceIndex i = 0; i < impl->deviceCount(); ++i) {
    impl->synchronizeDevice(i);
  }
}

struct incoming {
  uint64_t sequence = 0;
  std::vector<uint8_t> body;
  std::vector<uint64_t> frees;
  int request_id = -1;
};

[[nodiscard]] int read_incoming(conn_t *conn, incoming *in) {
  uint32_t length = 0;
  if (rpc_read(conn, &in->sequence, sizeof(in->sequence)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  in->body.resize(length);
  if (length != 0 && rpc_read(conn, in->body.data(), length) < 0) {
    return -1;
  }
  return 0;
}

// The releases travel ahead of the body and apply inside the sequence gate,
// so a handle freed after an op that used it is still bound when that op runs.
reader open_body(incoming &in) {
  reader r(in.body.data(), in.body.size());
  uint32_t count = r.get<uint32_t>();
  in.frees.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    in.frees[i] = r.get<uint64_t>();
  }
  return r;
}

void apply_frees(const incoming &in) {
  if (in.frees.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_table_mutex);
  for (uint64_t handle : in.frees) {
    g_table.erase(handle);
  }
}

int respond_error(conn_t *conn, int request_id, const std::string &message) {
  uint8_t status = 1;
  uint32_t size = static_cast<uint32_t>(message.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &size, sizeof(size)) < 0 ||
      rpc_write(conn, message.data(), size) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int respond_blob(conn_t *conn, int request_id, const void *data, size_t size) {
  uint8_t status = 0;
  uint32_t length = static_cast<uint32_t>(size);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, data, size) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

std::string what(const std::exception &e) {
  if (const auto *error = dynamic_cast<const c10::Error *>(&e)) {
    return error->what_without_backtrace();
  }
  return e.what();
}

// Binds each returned tensor to the handle the host chose for it. The host
// wrote one handle per tensor and a count ahead of each tensor list.
void bind_results(reader &r, const torch::jit::Stack &results) {
  uint32_t count = r.get<uint32_t>();
  std::vector<uint64_t> handles(count);
  for (uint32_t i = 0; i < count; ++i) {
    handles[i] = r.get<uint64_t>();
  }
  size_t next = 0;
  for (const c10::IValue &v : results) {
    if (v.isTensor()) {
      TORCH_CHECK(next < handles.size(), "lupine worker: result handle count");
      bind(handles[next++], v.toTensor());
    } else if (v.isTensorList()) {
      TORCH_CHECK(next < handles.size(), "lupine worker: result handle count");
      uint64_t expected = handles[next++];
      auto items = v.toTensorVector();
      TORCH_CHECK(expected == items.size(),
                  "lupine worker: result list length ", items.size(),
                  " differs from the host's ", expected);
      for (const at::Tensor &t : items) {
        bind(handles[next++], t);
      }
    }
  }
}

// Decodes the call inside the sequence gate: descriptors resolve against the
// table as it stands once every earlier op has bound its results.
torch::jit::Stack decode_call(reader &r, std::string *name) {
  *name = r.get_string();
  uint8_t nargs = r.get<uint8_t>();
  torch::jit::Stack stack;
  stack.reserve(nargs);
  for (uint8_t i = 0; i < nargs; ++i) {
    stack.push_back(read_ivalue(r, reading()));
  }
  return stack;
}

int handle_op(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0 ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  profile *prof = profile::instance();
  uint64_t started = prof != nullptr ? profile::now() : 0;
  try {
    stream_scope stream;
    reader r = open_body(in);
    apply_frees(in);
    std::string name;
    torch::jit::Stack stack = decode_call(r, &name);
    uint64_t decoded = prof != nullptr ? profile::now() : 0;
    run(name, stack);
    bind_results(r, stack);
    if (prof != nullptr) {
      uint64_t finished = profile::now();
      prof->record(name, decoded - started, finished - decoded);
    }
  } catch (const std::exception &e) {
    record_error(what(e));
  }
  rpc_async_sequence_end(conn);
  return 0;
}

int handle_op_sync(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0 ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  writer w;
  std::string error;
  try {
    stream_scope stream;
    reader r = open_body(in);
    apply_frees(in);
    std::string name;
    torch::jit::Stack stack = decode_call(r, &name);
    run(name, stack);
    w.put<uint32_t>(static_cast<uint32_t>(stack.size()));
    for (const c10::IValue &v : stack) {
      write_ivalue(w, v, wire());
    }
    error = take_error();
  } catch (const std::exception &e) {
    error = what(e);
  }
  rpc_async_sequence_end(conn);
  if (!error.empty()) {
    return respond_error(conn, in.request_id, error);
  }
  return respond_blob(conn, in.request_id, w.buffer.data(), w.buffer.size());
}

int handle_copy_to_host(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0 ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  at::Tensor cpu;
  std::string error;
  try {
    stream_scope stream;
    reader r = open_body(in);
    apply_frees(in);
    tensor_desc d = read_desc(r);
    auto dtype = static_cast<c10::ScalarType>(r.get<int8_t>());
    at::Tensor t = materialize(d);
    if (t.scalar_type() != dtype) {
      t = t.to(dtype);
    }
    cpu = t.contiguous().cpu();
    error = take_error();
  } catch (const std::exception &e) {
    error = what(e);
  }
  rpc_async_sequence_end(conn);
  if (!error.empty()) {
    return respond_error(conn, in.request_id, error);
  }
  uint8_t status = 0;
  size_t nbytes = cpu.numel() * cpu.element_size();
  if (rpc_write_start_response(conn, in.request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, cpu.const_data_ptr(), nbytes) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_copy_from_host(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0) {
    return -1;
  }
  // The payload follows the body; it is read before the request ends and
  // lands directly in a host tensor shaped like the destination.
  reader r = open_body(in);
  tensor_desc d = read_desc(r);
  uint64_t nbytes = r.get<uint64_t>();
  at::Tensor source = at::empty(
      d.sizes,
      at::TensorOptions().dtype(static_cast<c10::ScalarType>(d.dtype)));
  if ((nbytes != 0 && rpc_read(conn, source.mutable_data_ptr(), nbytes) < 0) ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  try {
    stream_scope stream;
    apply_frees(in);
    at::Tensor target = materialize(d);
    c10::DeviceGuard guard(target.device());
    target.copy_(source);
  } catch (const std::exception &e) {
    record_error(what(e));
  }
  rpc_async_sequence_end(conn);
  return 0;
}

py::object worker_module() { return py::module_::import("lupine._worker"); }

int handle_exec(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0 ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  try {
    reader r = open_body(in);
    apply_frees(in);
    std::string code = r.get_string();
    py::gil_scoped_acquire gil;
    worker_module().attr("_exec")(code);
  } catch (const std::exception &e) {
    record_error(what(e));
  }
  rpc_async_sequence_end(conn);
  return 0;
}

int handle_eval(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0 ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  std::string result;
  std::string error;
  try {
    reader r = open_body(in);
    apply_frees(in);
    std::string code = r.get_string();
    py::gil_scoped_acquire gil;
    result = worker_module().attr("_eval")(code).cast<std::string>();
    error = take_error();
  } catch (const std::exception &e) {
    error = what(e);
  }
  rpc_async_sequence_end(conn);
  if (!error.empty()) {
    return respond_error(conn, in.request_id, error);
  }
  return respond_blob(conn, in.request_id, result.data(), result.size());
}

int handle_sync(conn_t *conn) {
  incoming in;
  if (read_incoming(conn, &in) < 0 ||
      (in.request_id = rpc_read_end(conn)) < 0 ||
      rpc_async_sequence_begin(conn, in.sequence) < 0) {
    return -1;
  }
  std::string error;
  try {
    (void)open_body(in);
    apply_frees(in);
    synchronize_all();
    error = take_error();
  } catch (const std::exception &e) {
    error = what(e);
  }
  rpc_async_sequence_end(conn);
  if (!error.empty()) {
    return respond_error(conn, in.request_id, error);
  }
  uint8_t status = 0;
  if (rpc_write_start_response(conn, in.request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int dispatch_op(conn_t *conn, int op) {
  switch (op) {
  case LUPINE_RPC_lupineTorchOp:
    return handle_op(conn);
  case LUPINE_RPC_lupineTorchOpSync:
    return handle_op_sync(conn);
  case LUPINE_RPC_lupineTorchCopyToHost:
    return handle_copy_to_host(conn);
  case LUPINE_RPC_lupineTorchCopyFromHost:
    return handle_copy_from_host(conn);
  case LUPINE_RPC_lupineTorchExec:
    return handle_exec(conn);
  case LUPINE_RPC_lupineTorchEval:
    return handle_eval(conn);
  case LUPINE_RPC_lupineTorchSync:
    return handle_sync(conn);
  default:
    LUPINE_LOG_ERROR("lupine torch worker: unknown op " << op);
    return -1;
  }
}

struct lane {
  int32_t id = -1;
  std::atomic<bool> done{false};
  std::thread thread;
};

void serve_lanes(conn_t &conn) {
  std::unordered_map<int32_t, std::shared_ptr<lane>> lanes;
  while (!conn.closed) {
    for (auto it = lanes.begin(); it != lanes.end();) {
      if (!it->second->done.load(std::memory_order_acquire)) {
        ++it;
        continue;
      }
      if (it->second->thread.joinable()) {
        it->second->thread.join();
      }
      it = lanes.erase(it);
    }
    int32_t stream_id = rpc_http2_accept_stream(&conn);
    if (stream_id < 0 || lanes.size() >= kMaxLanes) {
      break;
    }
    auto entry = std::make_shared<lane>();
    entry->id = stream_id;
    entry->thread = std::thread([&conn, entry]() {
      if (rpc_bind_http2_stream(&conn, entry->id) == 0) {
        while (!conn.closed) {
          int op = rpc_dispatch(&conn, 0);
          if (op < 0) {
            break;
          }
          if (dispatch_op(&conn, op) < 0) {
            (void)rpc_read_end(&conn);
            break;
          }
        }
        if (!conn.closed) {
          (void)rpc_http2_end_stream(&conn, entry->id);
        }
        rpc_unbind_http2_stream(&conn);
      }
      entry->done.store(true, std::memory_order_release);
    });
    lanes.emplace(stream_id, entry);
  }
  rpc_shutdown_transport_socket(&conn);
  for (auto &entry : lanes) {
    if (entry.second->thread.joinable()) {
      entry.second->thread.join();
    }
  }
}

} // namespace

void serve(int fd) {
  py::gil_scoped_release release;
  const rpc_http2_server_metadata metadata = {};
  if (lupine_connection_dispatch(fd, &metadata, nullptr) != 0) {
    close(fd);
    return;
  }
  conn_t conn = {};
  if (rpc_conn_init(&conn, fd, 1) < 0) {
    close(fd);
    return;
  }
  if (rpc_http2_server_init_with_metadata(&conn, &metadata) == 0) {
    serve_lanes(conn);
  }
  rpc_conn_destroy(&conn);
}

void set_stream(int64_t stream_id, int16_t device_index) {
  g_stream_device.store(device_index);
  g_stream_id.store(stream_id);
}

} // namespace lupine_torch::worker
