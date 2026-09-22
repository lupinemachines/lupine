// Host side of the torch backend: tensors are host-side metadata over a remote
// storage handle; operators are forwarded boxed to the worker.
#include "backend.h"
#include "wire.h"

#include "gen_rpc_ids.h"
#include "lupine_log.h"
#include "rpc.h"
#include "transport.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace lupine_torch::host {

namespace {

c10::DeviceType g_device_type = c10::DeviceType::PrivateUse1;
bool g_dual = false;
conn_t *g_conn = nullptr;
int g_device_count = 0;
std::atomic<uint64_t> g_next_handle{1};
std::mutex g_free_mutex;
std::vector<uint64_t> g_pending_free;
thread_local c10::DeviceIndex t_current_device = 0;
// The device index new storages are placed on while an empty() call runs.
thread_local c10::DeviceIndex t_alloc_device = 0;

bool is_ours_type(c10::DeviceType type) {
  return type == c10::DeviceType::PrivateUse1 ||
         (g_dual && type == c10::DeviceType::CUDA);
}

bool is_ours(const at::Tensor &t) {
  return t.defined() && is_ours_type(t.device().type());
}

c10::DispatchKey key_for(c10::DeviceType type) {
  return type == c10::DeviceType::CUDA ? c10::DispatchKey::CUDA
                                       : c10::DispatchKey::PrivateUse1;
}

void *fake_pointer(uint64_t handle) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(handle) << 20);
}

uint64_t handle_of(const c10::Storage &storage) {
  return static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(storage.data_ptr().get_context()));
}

void release_handle(void *context) {
  std::lock_guard<std::mutex> lock(g_free_mutex);
  g_pending_free.push_back(
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context)));
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

void *dispatch_thread(void *argument) {
  auto *conn = static_cast<conn_t *>(argument);
  while (!conn->closed) {
    int op = rpc_dispatch(conn, 1);
    if (op < 0 || conn->closed) {
      break;
    }
    if (rpc_read_end(conn) < 0) {
      break;
    }
  }
  return nullptr;
}

void retire_lane(uint64_t lane_id) {
  lupine_client_transport_retire_lane(lane_id);
}

[[noreturn]] void connection_lost() {
  TORCH_CHECK(false, "lupine: the torch worker connection is closed");
}

// A request payload built outside the connection lock: the async ticket, the
// body length, then the body. The ticket is patched in once the request is
// started, and the pending storage releases ride at the front of every body so
// a free never costs a message of its own.
constexpr size_t kHeaderBytes = sizeof(uint64_t) + sizeof(uint32_t);

class request {
public:
  explicit request(int op) : op_(op) {
    w_.put<uint64_t>(0);
    w_.put<uint32_t>(0);
    std::vector<uint64_t> frees;
    {
      std::lock_guard<std::mutex> lock(g_free_mutex);
      frees.swap(g_pending_free);
    }
    w_.put<uint32_t>(static_cast<uint32_t>(frees.size()));
    for (uint64_t handle : frees) {
      w_.put<uint64_t>(handle);
    }
  }

  writer &body() { return w_; }

  // Queues the request; `extra` is appended after the built payload.
  void send(const void *extra = nullptr, size_t extra_size = 0) {
    start(extra, extra_size);
    if (rpc_write_end(g_conn) < 0) {
      connection_lost();
    }
  }

  // Sends and waits for the response; the caller reads it and calls finish().
  void call(const void *extra = nullptr, size_t extra_size = 0) {
    start(extra, extra_size);
    if (rpc_wait_for_response(g_conn) < 0) {
      connection_lost();
    }
  }

  void finish() {
    if (rpc_read_end(g_conn) < 0) {
      connection_lost();
    }
  }

private:
  void start(const void *extra, size_t extra_size) {
    uint64_t sequence = 0;
    if (g_conn == nullptr ||
        rpc_write_start_async_request(g_conn, op_, &sequence) < 0) {
      connection_lost();
    }
    w_.patch<uint64_t>(0, sequence);
    w_.patch<uint32_t>(sizeof(uint64_t),
                       static_cast<uint32_t>(w_.buffer.size() - kHeaderBytes));
    if (rpc_write(g_conn, w_.buffer.data(), w_.buffer.size()) < 0 ||
        (extra_size != 0 && rpc_write(g_conn, extra, extra_size) < 0)) {
      rpc_write_end(g_conn);
      connection_lost();
    }
  }

  int op_;
  writer w_;
};

// Reads a status byte and raises the worker's message on failure.
void read_status() {
  uint8_t status = 1;
  if (rpc_read(g_conn, &status, sizeof(status)) < 0) {
    connection_lost();
  }
  if (status != 0) {
    uint32_t size = 0;
    if (rpc_read(g_conn, &size, sizeof(size)) < 0) {
      connection_lost();
    }
    std::string message(size, '\0');
    if (size != 0 && rpc_read(g_conn, message.data(), size) < 0) {
      connection_lost();
    }
    rpc_read_end(g_conn);
    TORCH_CHECK(false, "lupine worker: ", message);
  }
}

std::vector<uint8_t> read_blob() {
  uint32_t size = 0;
  if (rpc_read(g_conn, &size, sizeof(size)) < 0) {
    connection_lost();
  }
  std::vector<uint8_t> bytes(size);
  if (size != 0 && rpc_read(g_conn, bytes.data(), size) < 0) {
    connection_lost();
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// Storage and tensors
// ---------------------------------------------------------------------------

struct lupine_allocator final : at::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    (void)nbytes;
    uint64_t handle = g_next_handle.fetch_add(1);
    return {fake_pointer(handle),
            reinterpret_cast<void *>(static_cast<uintptr_t>(handle)),
            &release_handle, c10::Device(g_device_type, t_alloc_device)};
  }
  at::DeleterFnPtr raw_deleter() const override { return &release_handle; }
  void copy_data(void *, const void *, std::size_t) const override {
    TORCH_CHECK(false, "lupine: host-side copies of device storage");
  }
};

lupine_allocator g_allocator;

c10::Storage new_storage(size_t nbytes, c10::DeviceIndex index) {
  t_alloc_device = index;
  return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes, g_allocator.allocate(nbytes),
      &g_allocator, /*resizable=*/true));
}

at::Tensor make_tensor(c10::Storage storage, c10::ScalarType dtype,
                       c10::IntArrayRef sizes, c10::IntArrayRef strides,
                       int64_t offset) {
  c10::DeviceType type = storage.device_type();
  at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(key_for(type)),
      caffe2::TypeMeta::fromScalarType(dtype));
  t.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides, offset);
  return t;
}

tensor_desc describe(const at::Tensor &t) {
  tensor_desc d;
  const c10::Storage &storage = t.storage();
  d.handle = handle_of(storage);
  d.nbytes = storage.nbytes();
  d.device = t.device().index();
  d.dtype = static_cast<int8_t>(t.scalar_type());
  d.offset = t.storage_offset();
  d.sizes.assign(t.sizes().begin(), t.sizes().end());
  d.strides.assign(t.strides().begin(), t.strides().end());
  return d;
}

// The memo key for _fused_sdp_choice: the CUDA choice depends on dtype,
// head count, head size and the last stride, not on batch or sequence
// length, so those are dropped and one answer serves a whole generation.
tensor_desc describe_shape_only(const at::Tensor &t) {
  tensor_desc d = describe(t);
  d.handle = 0;
  d.nbytes = 0;
  d.offset = 0;
  size_t ndim = d.sizes.size();
  for (size_t i = 0; i < ndim; ++i) {
    bool keep = i + 1 == ndim || (ndim == 4 && i == 1);
    if (!keep) {
      d.sizes[i] = 0;
    }
    d.strides[i] = i + 1 == ndim ? (d.strides[i] == 1 ? 1 : 0) : 0;
  }
  return d;
}

const wire_context &wire() {
  static const wire_context ctx = {describe, is_ours, [](const c10::Device &d) {
                                     return is_ours_type(d.type());
                                   }};
  return ctx;
}

const wire_context &wire_shape_only() {
  static const wire_context ctx = {
      describe_shape_only, is_ours,
      [](const c10::Device &d) { return is_ours_type(d.type()); }};
  return ctx;
}

tensor_desc describe_for_key(const at::Tensor &t) {
  tensor_desc d = describe(t);
  d.handle = 0;
  d.nbytes = 0;
  d.offset = 0;
  d.device = 0;
  return d;
}

const wire_context &wire_key() {
  static const wire_context ctx = {
      describe_for_key, is_ours,
      [](const c10::Device &d) { return is_ours_type(d.type()); }};
  return ctx;
}

size_t required_nbytes(const at::Tensor &t) {
  return at::detail::computeStorageNbytes(t.sizes(), t.strides(),
                                          t.element_size(), t.storage_offset());
}

// Storage growth is host-side bookkeeping: the worker grows its copy in place
// when a descriptor arrives with more bytes than it holds.
void ensure_storage(const at::Tensor &t) {
  size_t needed = required_nbytes(t);
  const c10::Storage &storage = t.storage();
  if (needed > storage.nbytes()) {
    storage.unsafeGetStorageImpl()->set_nbytes(needed);
  }
}

// Result tensors of a sync call: handles the worker already knows map to the
// storage that carries them here, new ones get a host storage. A result over
// a mutable argument's storage is that argument (an out= tensor the worker
// may have resized), so its metadata is updated in place.
class result_binder {
public:
  void note(const at::Tensor &t, bool mutable_argument) {
    if (is_ours(t)) {
      storages_[handle_of(t.storage())] = t.storage();
      if (mutable_argument) {
        mutated_.emplace_back(t);
      }
    }
  }
  at::Tensor materialize(const tensor_desc &d) {
    auto found = storages_.find(d.handle);
    c10::Storage storage;
    if (found != storages_.end()) {
      storage = found->second;
      if (d.nbytes > storage.nbytes()) {
        storage.unsafeGetStorageImpl()->set_nbytes(d.nbytes);
      }
      for (const at::Tensor &t : mutated_) {
        if (t.storage().unsafeGetStorageImpl() ==
                storage.unsafeGetStorageImpl() &&
            t.scalar_type() == static_cast<c10::ScalarType>(d.dtype)) {
          t.unsafeGetTensorImpl()->set_sizes_and_strides(d.sizes, d.strides,
                                                         d.offset);
          return t;
        }
      }
    } else {
      t_alloc_device = static_cast<c10::DeviceIndex>(d.device);
      storage = c10::Storage(c10::make_intrusive<c10::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(), d.nbytes,
          at::DataPtr(
              fake_pointer(d.handle),
              reinterpret_cast<void *>(static_cast<uintptr_t>(d.handle)),
              &release_handle, c10::Device(g_device_type, d.device)),
          &g_allocator, /*resizable=*/true));
      storages_[d.handle] = storage;
    }
    return make_tensor(storage, static_cast<c10::ScalarType>(d.dtype), d.sizes,
                       d.strides, d.offset);
  }

private:
  std::unordered_map<uint64_t, c10::Storage> storages_;
  std::vector<at::Tensor> mutated_;
};

// ---------------------------------------------------------------------------
// Operator forwarding
// ---------------------------------------------------------------------------

bool trace_enabled() {
  static const bool enabled = getenv("LUPINE_TORCH_TRACE") != nullptr;
  return enabled;
}

// What an op does to its outputs for one set of argument shapes, learned
// from the meta kernel once and replayed for every later call with the same
// shapes: torch's meta kernels are pure in the argument metadata, and the
// Python-implemented ones cost hundreds of microseconds a call.
struct plan_output {
  enum kind : uint8_t { undefined, input, fresh, none, integer, real, boolean };
  kind k = undefined;
  // input: the flattened index of the argument tensor returned as is;
  // fresh: the storage group the output lives in.
  int32_t index = 0;
  int8_t dtype = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset = 0;
  int64_t ival = 0;
  double dval = 0;
};

struct plan_ret {
  bool is_list = false;
  std::vector<plan_output> items;
};

struct plan {
  std::vector<plan_ret> rets;
  std::vector<size_t> group_nbytes;
};

struct op_state {
  std::string name;
  bool mutates = false;
  bool sync = false;
  bool memo = false;
  // In-place ops (foo_, no out= argument) cannot change their arguments'
  // metadata through a Scalar or float value, so the plan key leaves those
  // out and an optimizer's per-step coefficients hit the same plan.
  bool inplace = false;
  std::mutex mutex;
  std::unordered_map<std::string, std::vector<uint8_t>> memo_results;
  std::unordered_map<std::string, std::shared_ptr<const plan>> plans;
};

op_state &state_for(const c10::OperatorHandle &op) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::unique_ptr<op_state>> states;
  static const std::unordered_set<std::string> memoized = {
      "aten::_fused_sdp_choice"};
  const c10::FunctionSchema &schema = op.schema();
  std::string name = schema.name();
  if (!schema.overload_name().empty()) {
    name += "." + schema.overload_name();
  }
  std::lock_guard<std::mutex> lock(mutex);
  auto found = states.find(name);
  if (found != states.end()) {
    return *found->second;
  }
  auto state = std::make_unique<op_state>();
  state->name = name;
  state->memo = memoized.count(schema.name()) != 0;
  bool has_out = false;
  for (const c10::Argument &argument : schema.arguments()) {
    if (argument.alias_info() != nullptr && argument.alias_info()->isWrite()) {
      state->mutates = true;
    }
    if (argument.name() == "out") {
      has_out = true;
    }
  }
  state->inplace =
      !has_out && !schema.name().empty() && schema.name().back() == '_';
  op_state &ref = *state;
  states.emplace(name, std::move(state));
  return ref;
}

struct meta_mirror {
  // Host tensors seen in the arguments and their meta counterparts.
  std::vector<std::pair<at::Tensor, at::Tensor>> inputs;
  std::vector<std::pair<c10::StorageImpl *, c10::Storage>> storages;
  c10::DeviceIndex device = -1;
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;

  c10::Storage meta_storage_for(const c10::Storage &storage) {
    for (auto &entry : storages) {
      if (entry.first == storage.unsafeGetStorageImpl()) {
        return entry.second;
      }
    }
    c10::Storage meta(c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), storage.nbytes(),
        at::DataPtr(nullptr, c10::Device(c10::DeviceType::Meta)),
        c10::GetAllocator(c10::DeviceType::Meta), /*resizable=*/true));
    storages.emplace_back(storage.unsafeGetStorageImpl(), meta);
    return meta;
  }

  at::Tensor to_meta(const at::Tensor &t) {
    if (!is_ours(t)) {
      return t;
    }
    if (device < 0) {
      device = t.device().index();
      device_type = t.device().type();
    }
    at::Tensor m = at::detail::make_tensor<c10::TensorImpl>(
        meta_storage_for(t.storage()),
        c10::DispatchKeySet(c10::DispatchKey::Meta), t.dtype());
    m.unsafeGetTensorImpl()->set_sizes_and_strides(t.sizes(), t.strides(),
                                                   t.storage_offset());
    inputs.emplace_back(t, m);
    return m;
  }

  c10::IValue convert(const c10::IValue &v) {
    if (v.isTensor()) {
      return to_meta(v.toTensor());
    }
    if (v.isTensorList()) {
      c10::List<at::Tensor> list;
      for (const at::Tensor &t : v.toTensorVector()) {
        list.push_back(to_meta(t));
      }
      return list;
    }
    if (v.isList() && kind_of(v) == list_kind::optional_tensor) {
      c10::List<std::optional<at::Tensor>> list;
      for (const c10::IValue &item : v.toListRef()) {
        if (item.isNone()) {
          list.push_back(std::nullopt);
        } else {
          list.push_back(to_meta(item.toTensor()));
        }
      }
      return list;
    }
    if (v.isDevice() && is_ours_type(v.toDevice().type())) {
      if (device < 0) {
        device =
            v.toDevice().has_index() ? v.toDevice().index() : t_current_device;
        device_type = v.toDevice().type();
      }
      return c10::Device(c10::DeviceType::Meta);
    }
    return v;
  }
};

// An argument the meta kernel resized (an out= tensor sized by the call)
// travels as an empty view: the worker's kernel sizes it the same way, and a
// non-empty tensor of another shape would draw a deprecation warning there.
void write_call(writer &w, const op_state &st, torch::jit::Stack *stack,
                size_t args_begin, size_t nargs, const wire_context &base,
                const std::vector<c10::TensorImpl *> &shrink = {}) {
  wire_context ctx = base;
  if (!shrink.empty()) {
    ctx.describe = [&shrink, &base](const at::Tensor &t) {
      tensor_desc d = base.describe(t);
      if (std::find(shrink.begin(), shrink.end(), t.unsafeGetTensorImpl()) !=
          shrink.end()) {
        d.sizes.assign(1, 0);
        d.strides.assign(1, 1);
      }
      return d;
    };
  }
  w.put_string(st.name);
  w.put<uint8_t>(static_cast<uint8_t>(nargs));
  for (size_t i = 0; i < nargs; ++i) {
    write_ivalue(w, (*stack)[args_begin + i], ctx);
  }
}

c10::Storage fresh_storage(size_t nbytes, const meta_mirror &mirror) {
  c10::DeviceIndex device =
      mirror.device < 0 ? t_current_device : mirror.device;
  c10::DeviceType saved = g_device_type;
  g_device_type = mirror.device < 0 ? g_device_type : mirror.device_type;
  c10::Storage storage = new_storage(nbytes, device);
  g_device_type = saved;
  return storage;
}

void note_shape(plan_output &out, const at::Tensor &m) {
  out.dtype = static_cast<int8_t>(m.scalar_type());
  out.sizes.assign(m.sizes().begin(), m.sizes().end());
  out.strides.assign(m.strides().begin(), m.strides().end());
  out.offset = m.storage_offset();
}

// Maps the meta result of an op onto host tensors, assigning the storage
// handle the worker must bind each result to.
struct output_mapper {
  meta_mirror &mirror;
  std::vector<uint64_t> handles;
  std::vector<std::pair<c10::StorageImpl *, c10::Storage>> fresh;
  bool needs_rpc = false;
  plan learned;
  bool cacheable = true;
  std::vector<c10::TensorImpl *> resized;

  at::Tensor map(const at::Tensor &m, plan_output &out) {
    if (!m.defined()) {
      handles.push_back(0);
      out.k = plan_output::undefined;
      return at::Tensor();
    }
    c10::StorageImpl *meta_storage = m.storage().unsafeGetStorageImpl();
    for (size_t i = 0; i < mirror.inputs.size(); ++i) {
      auto &input = mirror.inputs[i];
      if (input.second.unsafeGetTensorImpl() == m.unsafeGetTensorImpl()) {
        at::Tensor host = input.first;
        c10::TensorImpl *impl = host.unsafeGetTensorImpl();
        if (!m.sizes().equals(host.sizes()) ||
            !m.strides().equals(host.strides()) ||
            m.storage_offset() != host.storage_offset()) {
          impl->set_sizes_and_strides(m.sizes(), m.strides(),
                                      m.storage_offset());
          ensure_storage(host);
          cacheable = false;
          resized.push_back(impl);
        }
        handles.push_back(handle_of(host.storage()));
        out.k = plan_output::input;
        out.index = static_cast<int32_t>(i);
        return host;
      }
    }
    for (auto &entry : mirror.storages) {
      if (entry.second.unsafeGetStorageImpl() == meta_storage) {
        for (auto &input : mirror.inputs) {
          if (input.first.storage().unsafeGetStorageImpl() == entry.first) {
            at::Tensor host =
                make_tensor(input.first.storage(), m.scalar_type(), m.sizes(),
                            m.strides(), m.storage_offset());
            ensure_storage(host);
            handles.push_back(handle_of(host.storage()));
            cacheable = false;
            return host;
          }
        }
      }
    }
    for (size_t g = 0; g < fresh.size(); ++g) {
      if (fresh[g].first == meta_storage) {
        handles.push_back(handle_of(fresh[g].second));
        out.k = plan_output::fresh;
        out.index = static_cast<int32_t>(g);
        note_shape(out, m);
        return make_tensor(fresh[g].second, m.scalar_type(), m.sizes(),
                           m.strides(), m.storage_offset());
      }
    }
    needs_rpc = true;
    size_t nbytes = at::detail::computeStorageNbytes(
        m.sizes(), m.strides(), m.element_size(), m.storage_offset());
    c10::Storage storage = fresh_storage(nbytes, mirror);
    fresh.emplace_back(meta_storage, storage);
    learned.group_nbytes.push_back(nbytes);
    handles.push_back(handle_of(storage));
    out.k = plan_output::fresh;
    out.index = static_cast<int32_t>(fresh.size() - 1);
    note_shape(out, m);
    return make_tensor(storage, m.scalar_type(), m.sizes(), m.strides(),
                       m.storage_offset());
  }
};

// Replays a learned plan: the same allocations, handles and results the
// meta kernel would have produced, without running it.
bool apply_plan(const plan &p, op_state &st, torch::jit::Stack *stack,
                size_t args_begin, size_t nargs) {
  meta_mirror mirror;
  std::vector<at::Tensor> inputs;
  for (size_t i = 0; i < nargs; ++i) {
    const c10::IValue &v = (*stack)[args_begin + i];
    if (v.isTensor()) {
      if (is_ours(v.toTensor())) {
        inputs.push_back(v.toTensor());
      }
    } else if (v.isTensorList()) {
      for (const at::Tensor &t : v.toTensorVector()) {
        if (is_ours(t)) {
          inputs.push_back(t);
        }
      }
    } else if (v.isList() && kind_of(v) == list_kind::optional_tensor) {
      for (const c10::IValue &item : v.toListRef()) {
        if (!item.isNone() && is_ours(item.toTensor())) {
          inputs.push_back(item.toTensor());
        }
      }
    } else if (v.isDevice() && is_ours_type(v.toDevice().type()) &&
               mirror.device < 0 && inputs.empty()) {
      mirror.device =
          v.toDevice().has_index() ? v.toDevice().index() : t_current_device;
      mirror.device_type = v.toDevice().type();
    }
  }
  if (!inputs.empty()) {
    mirror.device = inputs[0].device().index();
    mirror.device_type = inputs[0].device().type();
  }
  std::vector<c10::Storage> groups(p.group_nbytes.size());
  std::vector<uint64_t> handles;
  std::vector<c10::IValue> results;
  bool needs_rpc = st.mutates || !p.group_nbytes.empty();
  auto build = [&](const plan_output &out) -> at::Tensor {
    switch (out.k) {
    case plan_output::undefined:
      handles.push_back(0);
      return at::Tensor();
    case plan_output::input:
      handles.push_back(handle_of(inputs[out.index].storage()));
      return inputs[out.index];
    default: {
      c10::Storage &storage = groups[out.index];
      if (!storage) {
        storage = fresh_storage(p.group_nbytes[out.index], mirror);
      }
      handles.push_back(handle_of(storage));
      return make_tensor(storage, static_cast<c10::ScalarType>(out.dtype),
                         out.sizes, out.strides, out.offset);
    }
    }
  };
  for (const plan_ret &ret : p.rets) {
    if (ret.is_list) {
      c10::List<at::Tensor> list;
      handles.push_back(ret.items.size());
      for (const plan_output &out : ret.items) {
        list.push_back(build(out));
      }
      results.emplace_back(std::move(list));
      continue;
    }
    const plan_output &out = ret.items[0];
    switch (out.k) {
    case plan_output::none:
      results.emplace_back();
      break;
    case plan_output::integer:
      results.emplace_back(out.ival);
      break;
    case plan_output::real:
      results.emplace_back(out.dval);
      break;
    case plan_output::boolean:
      results.emplace_back(out.ival != 0);
      break;
    default:
      results.emplace_back(build(out));
    }
  }
  if (needs_rpc) {
    request req(LUPINE_RPC_lupineTorchOp);
    writer &w = req.body();
    write_call(w, st, stack, args_begin, nargs, wire());
    w.put<uint32_t>(static_cast<uint32_t>(handles.size()));
    for (uint64_t handle : handles) {
      w.put<uint64_t>(handle);
    }
    req.send();
  }
  stack->erase(stack->begin() + args_begin, stack->end());
  for (c10::IValue &r : results) {
    stack->push_back(std::move(r));
  }
  return true;
}

// A plan key is the op's argument metadata. CPU tensor arguments travel by
// value, so a large one makes the call uncacheable rather than the key huge.
bool plan_key(const op_state &st, torch::jit::Stack *stack, size_t args_begin,
              size_t nargs, std::string *key) {
  writer w;
  for (size_t i = 0; i < nargs; ++i) {
    const c10::IValue &v = (*stack)[args_begin + i];
    if (v.isTensor() && v.toTensor().defined() && !is_ours(v.toTensor()) &&
        v.toTensor().numel() > 64) {
      return false;
    }
    if (st.inplace &&
        (v.isDouble() || v.isScalar() ||
         (v.isList() && !v.isIntList() && !v.isBoolList() &&
          (v.isDoubleList() || kind_of(v) == list_kind::scalar)))) {
      w.put_tag(tag::none);
      continue;
    }
    write_ivalue(w, v, wire_key());
  }
  key->assign(reinterpret_cast<const char *>(w.buffer.data()), w.buffer.size());
  return true;
}

// Fire-and-forget path: the meta kernel decides the result metadata here, the
// worker binds the results to the handles chosen here.
bool try_async(const c10::OperatorHandle &op, op_state &st,
               torch::jit::Stack *stack, size_t args_begin, size_t nargs) {
  const c10::FunctionSchema &schema = op.schema();
  std::string key;
  bool keyed = plan_key(st, stack, args_begin, nargs, &key);
  if (keyed) {
    std::shared_ptr<const plan> found;
    {
      std::lock_guard<std::mutex> lock(st.mutex);
      auto it = st.plans.find(key);
      if (it != st.plans.end()) {
        found = it->second;
      }
    }
    if (found) {
      return apply_plan(*found, st, stack, args_begin, nargs);
    }
  }
  meta_mirror mirror;
  torch::jit::Stack meta_stack;
  meta_stack.reserve(nargs);
  for (size_t i = 0; i < nargs; ++i) {
    meta_stack.push_back(mirror.convert((*stack)[args_begin + i]));
  }
  profile *prof = profile::instance();
  uint64_t meta_started = prof != nullptr ? profile::now() : 0;
  try {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Meta),
                       &meta_stack);
  } catch (const c10::NotImplementedError &e) {
    if (trace_enabled()) {
      std::cerr << "lupine-torch no meta kernel for " << st.name << ": "
                << e.what_without_backtrace() << std::endl;
    }
    st.sync = true;
    return false;
  } catch (const std::exception &e) {
    if (trace_enabled()) {
      std::cerr << "lupine-torch meta failed for " << st.name << ": "
                << e.what() << std::endl;
    }
    return false;
  }

  if (prof != nullptr) {
    prof->record("meta:" + st.name, profile::now() - meta_started, 0);
  }
  size_t nret = schema.returns().size();
  TORCH_INTERNAL_ASSERT(meta_stack.size() == nret);
  output_mapper mapper{mirror};
  mapper.needs_rpc = st.mutates;
  std::vector<c10::IValue> results;
  results.reserve(nret);
  for (const c10::IValue &r : meta_stack) {
    plan_ret ret;
    if (r.isTensor()) {
      ret.items.emplace_back();
      results.emplace_back(mapper.map(r.toTensor(), ret.items.back()));
    } else if (r.isTensorList()) {
      c10::List<at::Tensor> list;
      auto items = r.toTensorVector();
      mapper.handles.push_back(static_cast<uint64_t>(items.size()));
      ret.is_list = true;
      for (const at::Tensor &t : items) {
        ret.items.emplace_back();
        list.push_back(mapper.map(t, ret.items.back()));
      }
      results.emplace_back(std::move(list));
    } else if (r.isNone() || r.isInt() || r.isSymInt() || r.isDouble() ||
               r.isBool()) {
      // Shape-derived scalars (a sequence length, a count) come out of the
      // meta kernel like the shapes do.
      results.push_back(r);
      ret.items.emplace_back();
      plan_output &out = ret.items.back();
      if (r.isNone()) {
        out.k = plan_output::none;
      } else if (r.isBool()) {
        out.k = plan_output::boolean;
        out.ival = r.toBool() ? 1 : 0;
      } else if (r.isDouble()) {
        out.k = plan_output::real;
        out.dval = r.toDouble();
      } else {
        out.k = plan_output::integer;
        out.ival = r.isSymInt() ? r.toSymInt().expect_int() : r.toInt();
      }
    } else {
      return false;
    }
    mapper.learned.rets.push_back(std::move(ret));
  }
  if (keyed && mapper.cacheable) {
    std::lock_guard<std::mutex> lock(st.mutex);
    st.plans.emplace(std::move(key),
                     std::make_shared<const plan>(std::move(mapper.learned)));
  }

  if (mapper.needs_rpc) {
    request req(LUPINE_RPC_lupineTorchOp);
    writer &w = req.body();
    write_call(w, st, stack, args_begin, nargs, wire(), mapper.resized);
    w.put<uint32_t>(static_cast<uint32_t>(mapper.handles.size()));
    for (uint64_t handle : mapper.handles) {
      w.put<uint64_t>(handle);
    }
    req.send();
  }
  stack->erase(stack->begin() + args_begin, stack->end());
  for (c10::IValue &r : results) {
    stack->push_back(std::move(r));
  }
  return true;
}

std::vector<c10::IValue> decode_results(const std::vector<uint8_t> &blob,
                                        result_binder &binder) {
  reader r(blob.data(), blob.size());
  read_context ctx;
  ctx.accelerator = g_device_type;
  ctx.materialize = [&binder](const tensor_desc &d) {
    return binder.materialize(d);
  };
  uint32_t count = r.get<uint32_t>();
  std::vector<c10::IValue> results;
  for (uint32_t i = 0; i < count; ++i) {
    results.push_back(read_ivalue(r, ctx));
  }
  return results;
}

void run_sync(const c10::OperatorHandle &op, op_state &st,
              torch::jit::Stack *stack, size_t args_begin, size_t nargs) {
  result_binder binder;
  const auto &arguments = op.schema().arguments();
  for (size_t i = 0; i < nargs; ++i) {
    const c10::IValue &v = (*stack)[args_begin + i];
    const c10::AliasInfo *alias = arguments[i].alias_info();
    bool mutable_argument = alias != nullptr && alias->isWrite();
    if (v.isTensor()) {
      binder.note(v.toTensor(), mutable_argument);
    } else if (v.isTensorList()) {
      for (const at::Tensor &t : v.toTensorVector()) {
        binder.note(t, mutable_argument);
      }
    }
  }
  std::string memo_key;
  if (st.memo) {
    writer probe;
    write_call(probe, st, stack, args_begin, nargs, wire_shape_only());
    const std::vector<uint8_t> &b = probe.buffer;
    memo_key.assign(reinterpret_cast<const char *>(b.data()), b.size());
    std::lock_guard<std::mutex> lock(st.mutex);
    auto found = st.memo_results.find(memo_key);
    if (found != st.memo_results.end()) {
      std::vector<c10::IValue> results = decode_results(found->second, binder);
      stack->erase(stack->begin() + args_begin, stack->end());
      for (c10::IValue &r : results) {
        stack->push_back(std::move(r));
      }
      return;
    }
  }
  request req(LUPINE_RPC_lupineTorchOpSync);
  write_call(req.body(), st, stack, args_begin, nargs, wire());
  req.call();
  read_status();
  std::vector<uint8_t> blob = read_blob();
  req.finish();
  if (st.memo) {
    std::lock_guard<std::mutex> lock(st.mutex);
    st.memo_results[memo_key] = blob;
  }
  std::vector<c10::IValue> results = decode_results(blob, binder);
  stack->erase(stack->begin() + args_begin, stack->end());
  for (c10::IValue &r : results) {
    stack->push_back(std::move(r));
  }
}

void fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  TORCH_CHECK(g_conn != nullptr, "lupine: the torch backend is not connected");
  op_state &st = state_for(op);
  size_t nargs = op.schema().arguments().size();
  size_t args_begin = stack->size() - nargs;
  profile *prof = profile::instance();
  uint64_t started = prof != nullptr ? profile::now() : 0;
  if (!st.sync && !st.memo && try_async(op, st, stack, args_begin, nargs)) {
    if (prof != nullptr) {
      prof->record(st.name, profile::now() - started, 0);
    }
    if (trace_enabled()) {
      std::cerr << "lupine-torch async " << st.name << std::endl;
    }
    return;
  }
  if (trace_enabled()) {
    std::cerr << "lupine-torch sync " << st.name << std::endl;
  }
  run_sync(op, st, stack, args_begin, nargs);
  if (prof != nullptr) {
    prof->record(st.name, 0, profile::now() - started);
  }
}

torch::jit::Stack forward(const char *name, const char *overload,
                          torch::jit::Stack &&stack) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, overload);
  fallback(op, &stack);
  return std::move(stack);
}

// The composite scaled_dot_product_attention asks this stub which fused
// kernel to use; without it every backend but CPU and CUDA gets the math
// decomposition. The worker's CUDA torch answers, memoized per shape.
int64_t sdp_choice(const at::Tensor &query, const at::Tensor &key,
                   const at::Tensor &value,
                   const std::optional<at::Tensor> &attn_mask, double dropout_p,
                   bool is_causal, std::optional<double> scale,
                   bool enable_gqa) {
  torch::jit::Stack stack = forward(
      "aten::_fused_sdp_choice", "",
      {query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa});
  return stack.back().toInt();
}

// ---------------------------------------------------------------------------
// Copies
// ---------------------------------------------------------------------------

void copy_to_device(const at::Tensor &self, const at::Tensor &src) {
  at::Tensor payload = src;
  if (src.scalar_type() != self.scalar_type() ||
      !src.sizes().equals(self.sizes()) || !src.is_contiguous()) {
    payload = at::empty(self.sizes(), src.options().dtype(self.scalar_type()));
    payload.copy_(src);
  }
  size_t nbytes = payload.numel() * payload.element_size();
  request req(LUPINE_RPC_lupineTorchCopyFromHost);
  write_desc(req.body(), describe(self));
  req.body().put<uint64_t>(nbytes);
  req.send(payload.const_data_ptr(), nbytes);
}

// The worker sends src contiguous in its own shape, already cast to the
// destination dtype; broadcasting into self happens here on the CPU.
void copy_to_host(const at::Tensor &self, const at::Tensor &src) {
  request req(LUPINE_RPC_lupineTorchCopyToHost);
  write_desc(req.body(), describe(src));
  req.body().put<int8_t>(static_cast<int8_t>(self.scalar_type()));
  req.call();
  read_status();
  bool direct = self.is_contiguous() && self.sizes().equals(src.sizes());
  at::Tensor target = direct ? self : at::empty(src.sizes(), self.options());
  size_t nbytes = target.numel() * target.element_size();
  if (nbytes != 0 && rpc_read(g_conn, target.mutable_data_ptr(), nbytes) < 0) {
    connection_lost();
  }
  req.finish();
  if (!direct) {
    self.copy_(target);
  }
}

at::Tensor &copy_(at::Tensor &self, const at::Tensor &src, bool non_blocking) {
  (void)non_blocking;
  if (self.numel() == 0) {
    return self;
  }
  bool self_ours = is_ours(self);
  bool src_ours = is_ours(src);
  if (self_ours && src_ours) {
    forward("aten::copy_", "", {self, src, false});
  } else if (self_ours && src.device().is_cpu()) {
    copy_to_device(self, src);
  } else if (self.device().is_cpu() && src_ours) {
    copy_to_host(self, src);
  } else {
    TORCH_CHECK(false, "lupine: unsupported copy from ", src.device(), " to ",
                self.device());
  }
  return self;
}

at::Tensor _copy_from(const at::Tensor &self, const at::Tensor &dst,
                      bool non_blocking) {
  at::Tensor target = dst;
  copy_(target, self, non_blocking);
  return dst;
}

at::Tensor _copy_from_and_resize(const at::Tensor &self,
                                 const at::Tensor &dst) {
  at::Tensor target = dst;
  target.resize_(self.sizes());
  copy_(target, self, false);
  return dst;
}

at::Scalar _local_scalar_dense(const at::Tensor &self) {
  at::Tensor cpu = at::empty({}, self.options().device(c10::DeviceType::CPU));
  copy_to_host(cpu, self);
  return at::native::_local_scalar_dense_cpu(cpu);
}

// ---------------------------------------------------------------------------
// Allocation and metadata kernels
// ---------------------------------------------------------------------------

c10::DeviceIndex resolve_index(const std::optional<c10::Device> &device) {
  c10::Device d = c10::device_or_default(device);
  TORCH_CHECK(is_ours_type(d.type()), "lupine: expected a ", g_device_type,
              " device, got ", d);
  return d.has_index() ? d.index() : t_current_device;
}

at::Tensor empty_memory_format(c10::IntArrayRef size,
                               std::optional<c10::ScalarType> dtype,
                               std::optional<c10::Layout> layout,
                               std::optional<c10::Device> device,
                               std::optional<bool> pin_memory,
                               std::optional<c10::MemoryFormat> memory_format) {
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided,
              "lupine: only strided tensors are supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory),
              "lupine: pinned memory is a host allocation");
  c10::Device d = c10::device_or_default(device);
  t_alloc_device = resolve_index(device);
  c10::DeviceType saved = g_device_type;
  g_device_type = d.type();
  at::Tensor t = at::detail::empty_generic(
      size, &g_allocator, c10::DispatchKeySet(key_for(d.type())),
      c10::dtype_or_default(dtype), memory_format);
  g_device_type = saved;
  return t;
}

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                         std::optional<c10::ScalarType> dtype,
                         std::optional<c10::Layout> layout,
                         std::optional<c10::Device> device,
                         std::optional<bool> pin_memory) {
  TORCH_CHECK(c10::layout_or_default(layout) == c10::Layout::Strided,
              "lupine: only strided tensors are supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory),
              "lupine: pinned memory is a host allocation");
  c10::Device d = c10::device_or_default(device);
  t_alloc_device = resolve_index(device);
  c10::DeviceType saved = g_device_type;
  g_device_type = d.type();
  at::Tensor t = at::detail::empty_strided_generic(
      size, stride, &g_allocator, c10::DispatchKeySet(key_for(d.type())),
      c10::dtype_or_default(dtype));
  g_device_type = saved;
  return t;
}

at::Tensor as_strided(const at::Tensor &self, c10::SymIntArrayRef size,
                      c10::SymIntArrayRef stride,
                      std::optional<c10::SymInt> storage_offset) {
  std::optional<int64_t> offset;
  if (storage_offset.has_value()) {
    offset = storage_offset->expect_int();
  }
  return at::native::as_strided_tensorimpl(self, C10_AS_INTARRAYREF_SLOW(size),
                                           C10_AS_INTARRAYREF_SLOW(stride),
                                           offset);
}

at::Tensor view(const at::Tensor &self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor _reshape_alias(const at::Tensor &self, c10::SymIntArrayRef size,
                          c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW(size),
                                    C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor unfold(const at::Tensor &self, int64_t dimension, int64_t size,
                  int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

at::Tensor view_as_real(const at::Tensor &self) {
  return at::native::view_as_real(self);
}

at::Tensor view_as_complex(const at::Tensor &self) {
  return at::native::view_as_complex(self);
}

const at::Tensor &resize_(const at::Tensor &self, c10::SymIntArrayRef size,
                          std::optional<c10::MemoryFormat> memory_format) {
  meta_mirror mirror;
  at::Tensor m = mirror.to_meta(self);
  m.resize_(C10_AS_INTARRAYREF_SLOW(size), memory_format);
  self.unsafeGetTensorImpl()->set_sizes_and_strides(m.sizes(), m.strides(),
                                                    m.storage_offset());
  ensure_storage(self);
  return self;
}

at::Tensor &set_storage(at::Tensor &self, c10::Storage storage,
                        int64_t storage_offset, c10::IntArrayRef size,
                        c10::IntArrayRef stride) {
  c10::TensorImpl *impl = self.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(std::move(storage));
  if (stride.empty()) {
    std::vector<int64_t> contiguous(size.size(), 1);
    for (int64_t i = static_cast<int64_t>(size.size()) - 2; i >= 0; --i) {
      contiguous[i] = contiguous[i + 1] * std::max<int64_t>(size[i + 1], 1);
    }
    impl->set_sizes_and_strides(size, contiguous, storage_offset);
  } else {
    impl->set_sizes_and_strides(size, stride, storage_offset);
  }
  ensure_storage(self);
  return self;
}

at::Tensor &set_source_Storage(at::Tensor &self, c10::Storage storage) {
  int64_t numel = static_cast<int64_t>(storage.nbytes() / self.element_size());
  return set_storage(self, std::move(storage), 0, {numel}, {});
}

at::Tensor &set_source_Tensor(at::Tensor &self, const at::Tensor &source) {
  return set_storage(self, source.storage(), source.storage_offset(),
                     source.sizes(), source.strides());
}

at::Tensor &set_(at::Tensor &self) {
  return set_storage(self, new_storage(0, self.device().index()), 0, {0}, {});
}

at::Tensor _pin_memory(const at::Tensor &self,
                       std::optional<c10::Device> device) {
  (void)device;
  return self.clone();
}

bool _has_compatible_shallow_copy_type(const at::Tensor &, const at::Tensor &) {
  return true;
}

// ---------------------------------------------------------------------------
// Device guard, hooks, generator
// ---------------------------------------------------------------------------

struct guard_impl final : c10::impl::DeviceGuardImplInterface {
  explicit guard_impl(c10::DeviceType type) : type_(type) {}
  c10::DeviceType type() const override { return type_; }
  c10::Device exchangeDevice(c10::Device d) const override {
    c10::Device previous(type_, t_current_device);
    t_current_device = d.index();
    return previous;
  }
  c10::Device getDevice() const override {
    return c10::Device(type_, t_current_device);
  }
  void setDevice(c10::Device d) const override { t_current_device = d.index(); }
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    t_current_device = d.index();
  }
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream getNewStream(c10::Device d, int) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream getStreamFromGlobalPool(c10::Device d, bool) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, s.device());
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(g_device_count);
  }
  void record(void **event, const c10::Stream &, const c10::DeviceIndex,
              const c10::EventFlag) const override {
    *event = reinterpret_cast<void *>(1);
  }
  void block(void *, const c10::Stream &) const override {}
  bool queryEvent(void *) const override { return true; }
  void destroyEvent(void *, const c10::DeviceIndex) const noexcept override {}
  void synchronizeEvent(void *) const override { synchronize(); }
  bool queryStream(const c10::Stream &) const override {
    synchronize();
    return true;
  }
  void synchronizeStream(const c10::Stream &) const override { synchronize(); }
  void synchronizeDevice(const c10::DeviceIndex) const override {
    synchronize();
  }
  double elapsedTime(void *, void *, const c10::DeviceIndex) const override {
    return 0.0;
  }

private:
  c10::DeviceType type_;
};

// Seeds forwarded through the device module keep the worker's CUDA generator
// in step; the host object only remembers the seed for initial_seed().
struct generator_impl final : c10::GeneratorImpl {
  explicit generator_impl(c10::DeviceIndex index)
      : c10::GeneratorImpl(c10::Device(g_device_type, index),
                           c10::DispatchKeySet(key_for(g_device_type))) {}
  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    exec("torch.cuda.manual_seed_all(" + std::to_string(seed) + ")");
  }
  void set_offset(uint64_t) override {}
  uint64_t get_offset() const override { return 0; }
  uint64_t current_seed() const override { return seed_; }
  uint64_t seed() override {
    seed_ = c10::detail::getNonDeterministicRandom(true);
    set_current_seed(seed_);
    return seed_;
  }
  void set_state(const c10::TensorImpl &) override {
    TORCH_CHECK(false, "lupine: set_state is not supported; use "
                       "torch.cuda.set_rng_state");
  }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
    TORCH_CHECK(false, "lupine: get_state is not supported; use "
                       "torch.cuda.get_rng_state");
  }
  generator_impl *clone_impl() const override {
    return new generator_impl(device().index());
  }

private:
  uint64_t seed_ = 0;
};

struct hooks_impl final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return g_device_count > 0; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  void init() const override {}
  c10::DeviceIndex deviceCount() const override {
    return static_cast<c10::DeviceIndex>(g_device_count);
  }
  void setCurrentDevice(c10::DeviceIndex d) const override {
    t_current_device = d;
  }
  c10::DeviceIndex getCurrentDevice() const override {
    return t_current_device;
  }
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex d) const override {
    return host::exchange_device(d);
  }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex d) const override {
    return d < 0 ? t_current_device : host::exchange_device(d);
  }
  bool isPinnedPtr(const void *) const override { return false; }
  at::Allocator *getPinnedMemoryAllocator() const override {
    return c10::GetAllocator(c10::DeviceType::CPU);
  }
  const at::Generator &
  getDefaultGenerator(c10::DeviceIndex index) const override {
    static std::vector<at::Generator> generators = [] {
      std::vector<at::Generator> made;
      for (int i = 0; i < std::max(g_device_count, 1); ++i) {
        made.push_back(at::make_generator<generator_impl>(
            static_cast<c10::DeviceIndex>(i)));
      }
      return made;
    }();
    return generators.at(index < 0 ? t_current_device : index);
  }
  at::Generator getNewGenerator(c10::DeviceIndex index) const override {
    return at::make_generator<generator_impl>(index);
  }
  void resizePrivateUse1Bytes(const c10::Storage &storage,
                              size_t nbytes) const override {
    storage.unsafeGetStorageImpl()->set_nbytes(nbytes);
  }
};

struct cuda_hooks_impl final : at::CUDAHooksInterface {
  explicit cuda_hooks_impl(at::CUDAHooksArgs) {}
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return g_device_count > 0; }
  bool hasCUDA() const override { return g_device_count > 0; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  c10::DeviceIndex deviceCount() const override {
    return static_cast<c10::DeviceIndex>(g_device_count);
  }
  void setCurrentDevice(c10::DeviceIndex d) const override {
    t_current_device = d;
  }
  c10::DeviceIndex getCurrentDevice() const override {
    return t_current_device;
  }
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex d) const override {
    return host::exchange_device(d);
  }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex d) const override {
    return d < 0 ? t_current_device : host::exchange_device(d);
  }
  bool isPinnedPtr(const void *) const override { return false; }
  at::Allocator *getPinnedMemoryAllocator() const override {
    return c10::GetAllocator(c10::DeviceType::CPU);
  }
  const at::Generator &
  getDefaultGenerator(c10::DeviceIndex index) const override {
    return at::detail::getPrivateUse1Hooks().getDefaultGenerator(index);
  }
  at::Generator getNewGenerator(c10::DeviceIndex index) const override {
    return at::make_generator<generator_impl>(index);
  }
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void register_kernels(c10::DispatchKey key) {
  static std::vector<std::unique_ptr<torch::Library>> libraries;
  auto aten = std::make_unique<torch::Library>(torch::Library::IMPL, "aten",
                                               key, __FILE__, __LINE__);
  aten->impl("empty.memory_format", TORCH_FN(empty_memory_format));
  aten->impl("empty_strided", TORCH_FN(empty_strided));
  aten->impl("as_strided", TORCH_FN(as_strided));
  aten->impl("view", TORCH_FN(view));
  aten->impl("_reshape_alias", TORCH_FN(_reshape_alias));
  aten->impl("unfold", TORCH_FN(unfold));
  aten->impl("view_as_real", TORCH_FN(view_as_real));
  aten->impl("view_as_complex", TORCH_FN(view_as_complex));
  aten->impl("resize_", TORCH_FN(resize_));
  aten->impl("set_.source_Storage_storage_offset", TORCH_FN(set_storage));
  aten->impl("set_.source_Storage", TORCH_FN(set_source_Storage));
  aten->impl("set_.source_Tensor", TORCH_FN(set_source_Tensor));
  aten->impl("set_", TORCH_FN(set_));
  aten->impl("copy_", TORCH_FN(copy_));
  aten->impl("_copy_from", TORCH_FN(_copy_from));
  aten->impl("_copy_from_and_resize", TORCH_FN(_copy_from_and_resize));
  aten->impl("_local_scalar_dense", TORCH_FN(_local_scalar_dense));
  aten->impl("_pin_memory", TORCH_FN(_pin_memory));
  aten->impl("_has_compatible_shallow_copy_type",
             TORCH_FN(_has_compatible_shallow_copy_type));
  // Ops with a CompositeExplicitAutograd decomposition never reach the
  // fallback; these have a fused CUDA kernel the worker should run instead.
  static const char *const whole[] = {"native_layer_norm", "native_group_norm",
                                      "_fused_rms_norm", "convolution",
                                      "convolution_backward"};
  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();
  for (const char *name : whole) {
    if (dispatcher.findOp({std::string("aten::") + name, ""}).has_value()) {
      aten->impl(name, torch::CppFunction::makeFromBoxedFunction<&fallback>());
    }
  }
  for (const c10::OperatorName &op : dispatcher.getAllOpNames()) {
    if (op.name.rfind("aten::_foreach_", 0) != 0) {
      continue;
    }
    std::string name = op.name.substr(6);
    if (!op.overload_name.empty()) {
      name += "." + op.overload_name;
    }
    aten->impl(name.c_str(),
               torch::CppFunction::makeFromBoxedFunction<&fallback>());
  }
  libraries.push_back(std::move(aten));
  auto all = std::make_unique<torch::Library>(torch::Library::IMPL, "_", key,
                                              __FILE__, __LINE__);
  all->fallback(torch::CppFunction::makeFromBoxedFunction<&fallback>());
  libraries.push_back(std::move(all));
  if (key == c10::DispatchKey::CUDA) {
    at::native::_fused_sdp_choice_stub.set_cuda_dispatch_ptr(&sdp_choice);
  } else {
    at::native::_fused_sdp_choice_stub.set_privateuse1_dispatch_ptr(
        &sdp_choice);
  }
}

} // namespace

std::string connect(const std::string &address, bool dual) {
  TORCH_CHECK(g_conn == nullptr,
              "lupine: the torch backend is already connected");
  static const rpc_lifecycle_hooks hooks = {nullptr, retire_lane, nullptr,
                                            nullptr};
  rpc_set_lifecycle_hooks(&hooks);
  lupine_client_transport_config config;
  config.dial_policy = lupine_client_dial_policy::bounded_retry;
  config.strict_endpoints = true;
  config.log_missing_server = true;
  config.dispatch = dispatch_thread;
  config.w_offset = 0;
  TORCH_CHECK(lupine_client_transport_open(config, address.c_str()) == 0,
              "lupine: could not connect to the torch worker at ", address);
  g_conn = lupine_client_transport_connection(0);
  TORCH_CHECK(g_conn != nullptr, "lupine: no worker connection");
  g_dual = dual;
  g_device_type = dual ? c10::DeviceType::CUDA : c10::DeviceType::PrivateUse1;

  std::string info =
      eval("__import__('lupine._worker', fromlist=['x']).info()");
  {
    std::string key = "\"device_count\": ";
    size_t at = info.find(key);
    TORCH_CHECK(at != std::string::npos, "lupine: bad worker info ", info);
    g_device_count = std::atoi(info.c_str() + at + key.size());
  }

  static guard_impl private_guard(c10::DeviceType::PrivateUse1);
  static guard_impl cuda_guard(c10::DeviceType::CUDA);
  c10::impl::DeviceGuardImplRegistrar(c10::DeviceType::PrivateUse1,
                                      &private_guard);
  at::RegisterPrivateUse1HooksInterface(new hooks_impl());
  register_kernels(c10::DispatchKey::PrivateUse1);
  if (dual) {
    TORCH_CHECK(at::detail::getCUDAHooks().isBuilt() && g_device_count > 0 &&
                    at::detail::getCUDAHooks().deviceCount() == g_device_count,
                "lupine: torch created its CUDA hooks before the backend "
                "loaded; import lupine and call lupine.connect() before "
                "importing torch");
    c10::impl::DeviceGuardImplRegistrar(c10::DeviceType::CUDA, &cuda_guard);
    register_kernels(c10::DispatchKey::CUDA);
  }
  return info;
}

// A torch with no CUDA build has no CUDA hooks; the backend's own are
// registered when the extension loads, ahead of torch's first use of them.
void register_cuda_hooks() {
  if (at::CUDAHooksRegistry()->Has("CUDAHooks")) {
    return;
  }
  at::CUDAHooksRegistry()->Register("CUDAHooks", [](at::CUDAHooksArgs args) {
    return std::unique_ptr<at::CUDAHooksInterface>(new cuda_hooks_impl(args));
  });
}

bool connected() { return g_conn != nullptr; }

void disconnect() {
  if (g_conn == nullptr) {
    return;
  }
  request req(LUPINE_RPC_lupineTorchSync);
  try {
    req.call();
    read_status();
    req.finish();
  } catch (const c10::Error &) {
  }
  g_conn = nullptr;
  lupine_client_transport_close();
}

int device_count() { return g_device_count; }

c10::DeviceIndex current_device() { return t_current_device; }

void set_device(c10::DeviceIndex index) {
  TORCH_CHECK(index >= 0 && index < g_device_count, "lupine: invalid device ",
              index);
  t_current_device = index;
}

c10::DeviceIndex exchange_device(c10::DeviceIndex index) {
  c10::DeviceIndex previous = t_current_device;
  if (index >= 0) {
    set_device(index);
  }
  return previous;
}

void synchronize() {
  TORCH_CHECK(g_conn != nullptr, "lupine: the torch backend is not connected");
  request req(LUPINE_RPC_lupineTorchSync);
  req.call();
  read_status();
  req.finish();
}

void exec(const std::string &code) {
  TORCH_CHECK(g_conn != nullptr, "lupine: the torch backend is not connected");
  request req(LUPINE_RPC_lupineTorchExec);
  req.body().put_string(code);
  req.send();
}

std::string eval(const std::string &code) {
  TORCH_CHECK(g_conn != nullptr, "lupine: the torch backend is not connected");
  request req(LUPINE_RPC_lupineTorchEval);
  req.body().put_string(code);
  req.call();
  read_status();
  std::vector<uint8_t> blob = read_blob();
  req.finish();
  return std::string(blob.begin(), blob.end());
}

} // namespace lupine_torch::host
