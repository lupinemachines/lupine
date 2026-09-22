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
// The device new storages are placed on while an empty() call runs.
thread_local c10::Device t_alloc_device(c10::DeviceType::PrivateUse1, 0);

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
            &release_handle, t_alloc_device};
  }
  at::DeleterFnPtr raw_deleter() const override { return &release_handle; }
  void copy_data(void *, const void *, std::size_t) const override {
    TORCH_CHECK(false, "lupine: host-side copies of device storage");
  }
};

lupine_allocator g_allocator;

c10::Storage storage_for_handle(uint64_t handle, size_t nbytes,
                                c10::Device device) {
  return c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes,
      at::DataPtr(fake_pointer(handle),
                  reinterpret_cast<void *>(static_cast<uintptr_t>(handle)),
                  &release_handle, device),
      &g_allocator, /*resizable=*/true));
}

c10::Storage new_storage(size_t nbytes, c10::Device device) {
  return storage_for_handle(g_next_handle.fetch_add(1), nbytes, device);
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

// The plan key for _fused_sdp_choice: the CUDA choice depends on dtype,
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

// Storages are numbered by first appearance: which arguments share one is
// part of the key, their identity is not.
tensor_desc describe_for_key(const at::Tensor &t,
                             std::unordered_map<uint64_t, uint64_t> &ids) {
  tensor_desc d = describe(t);
  d.handle = ids.emplace(d.handle, ids.size()).first->second;
  d.nbytes = 0;
  d.offset = 0;
  d.device = 0;
  return d;
}

// Storage growth is host-side bookkeeping: the worker grows its copy in place
// when a descriptor arrives with more bytes than it holds.
void ensure_storage(const at::Tensor &t) {
  size_t needed = at::detail::computeStorageNbytes(
      t.sizes(), t.strides(), t.element_size(), t.storage_offset());
  const c10::Storage &storage = t.storage();
  if (needed > storage.nbytes()) {
    storage.unsafeGetStorageImpl()->set_nbytes(needed);
  }
}

// ---------------------------------------------------------------------------
// Operator forwarding
// ---------------------------------------------------------------------------

bool trace_enabled() {
  static const bool enabled = getenv("LUPINE_TORCH_TRACE") != nullptr;
  return enabled;
}

// What an op does to its outputs for one set of argument metadata, learned
// from the worker's report on the first call and replayed for every later
// call with the same key: the result metadata of an op is a function of its
// argument metadata, except for the data-dependent ops that stay synchronous.
struct plan_output {
  // input: the argument tensor returned as is (an in-place result, or an out=
  // tensor the kernel resized); input_storage: a view over an argument's
  // storage; fresh: a new storage, shared by every output in its group.
  enum kind : uint8_t {
    undefined,
    input,
    input_storage,
    fresh,
    none,
    integer,
    real,
    boolean
  };
  kind k = undefined;
  int32_t index = 0;
  bool same_metadata = false;
  int8_t dtype = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  // fresh: the storage offset; input and input_storage: relative to the
  // argument's offset, so the plan holds for arguments at any offset.
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
  // An op that neither writes an argument nor produces a storage is pure
  // host metadata once learned.
  bool needs_rpc = false;
};

struct op_state {
  std::string name;
  bool mutates = false;
  // Output shapes that depend on data: every call is a round trip.
  bool sync = false;
  bool bool_index = false;
  bool memo = false;
  // In-place ops (foo_, no out= argument) cannot change their arguments'
  // metadata through a Scalar or float value, so the plan key leaves those
  // out and an optimizer's per-step coefficients hit the same plan.
  bool inplace = false;
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<const plan>> plans;
};

op_state &state_for(const c10::OperatorHandle &op) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::unique_ptr<op_state>> states;
  static const std::unordered_set<std::string> memoized = {
      "aten::_fused_sdp_choice"};
  static const std::unordered_set<std::string> data_dependent = {
      "aten::nonzero",
      "aten::masked_select",
      "aten::_unique",
      "aten::_unique2",
      "aten::unique_dim",
      "aten::unique_consecutive",
      "aten::unique_dim_consecutive",
      "aten::bincount",
      "aten::repeat_interleave",
      "aten::equal"};
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
  state->sync = data_dependent.count(schema.name()) != 0;
  state->bool_index = schema.name() == "aten::index";
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

// The call as it travels: the pickle of (name, arguments...) and the tensor
// table it refers to. Symbolic ints and the device are made concrete for the
// worker; an undefined generator is the default one.
struct encoded {
  std::vector<char> bytes;
  std::vector<at::Tensor> table;
};

c10::IValue sanitize(const c10::IValue &v) {
  if (v.isDevice() && is_ours_type(v.toDevice().type())) {
    return v.toDevice().has_index()
               ? c10::Device(c10::DeviceType::CUDA, v.toDevice().index())
               : c10::Device(c10::DeviceType::CUDA);
  }
  if (v.isGenerator()) {
    TORCH_CHECK(!v.toGenerator().defined(),
                "lupine: explicit torch.Generator objects are not supported "
                "on the lupine backend; use torch.manual_seed");
    return c10::IValue();
  }
  if (v.isSymInt()) {
    return v.toSymInt().expect_int();
  }
  if (v.isSymFloat()) {
    return v.toSymFloat().expect_float();
  }
  if (v.isSymBool()) {
    return v.toSymBool().expect_bool();
  }
  if (v.isSymIntList()) {
    return c10::List<int64_t>(v.toIntVector());
  }
  return v;
}

encoded encode(const op_state &st, torch::jit::Stack *stack, size_t args_begin,
               size_t nargs) {
  std::vector<c10::IValue> values;
  values.reserve(nargs + 1);
  values.emplace_back(st.name);
  for (size_t i = 0; i < nargs; ++i) {
    values.push_back(sanitize((*stack)[args_begin + i]));
  }
  encoded e;
  e.bytes = pickle_stack(std::move(values), &e.table);
  return e;
}

// The argument tensors on the device in table order; the worker numbers
// them the same way when it reports which one a result aliases.
std::vector<at::Tensor> device_inputs(const encoded &e) {
  std::vector<at::Tensor> inputs;
  for (const at::Tensor &t : e.table) {
    if (is_ours(t)) {
      inputs.push_back(t);
    }
  }
  return inputs;
}

// A boolean index selects a data-dependent number of rows.
bool has_bool_index(const c10::IValue &indices) {
  for (const c10::IValue &item : indices.toListRef()) {
    if (!item.isNone() && (item.toTensor().scalar_type() == at::kBool ||
                           item.toTensor().scalar_type() == at::kByte)) {
      return true;
    }
  }
  return false;
}

// New storages go where the arguments are, else on the device argument.
c10::Device result_device(const std::vector<at::Tensor> &inputs,
                          torch::jit::Stack *stack, size_t args_begin,
                          size_t nargs) {
  if (!inputs.empty()) {
    return inputs[0].device();
  }
  for (size_t i = 0; i < nargs; ++i) {
    const c10::IValue &v = (*stack)[args_begin + i];
    if (v.isDevice() && is_ours_type(v.toDevice().type())) {
      return c10::Device(v.toDevice().type(), v.toDevice().has_index()
                                                  ? v.toDevice().index()
                                                  : t_current_device);
    }
  }
  return c10::Device(g_device_type, t_current_device);
}

at::Tensor build_output(const plan_output &out,
                        const std::vector<at::Tensor> &inputs,
                        const c10::Storage &storage) {
  switch (out.k) {
  case plan_output::undefined:
    return at::Tensor();
  case plan_output::input: {
    const at::Tensor &t = inputs[out.index];
    if (!out.same_metadata) {
      t.unsafeGetTensorImpl()->set_sizes_and_strides(
          out.sizes, out.strides, t.storage_offset() + out.offset);
      ensure_storage(t);
    }
    return t;
  }
  case plan_output::input_storage: {
    const at::Tensor &in = inputs[out.index];
    at::Tensor t =
        make_tensor(in.storage(), static_cast<c10::ScalarType>(out.dtype),
                    out.sizes, out.strides, in.storage_offset() + out.offset);
    ensure_storage(t);
    return t;
  }
  default:
    return make_tensor(storage, static_cast<c10::ScalarType>(out.dtype),
                       out.sizes, out.strides, out.offset);
  }
}

void write_call(writer &w, const encoded &e) {
  write_pickled(w, e.bytes, e.table, is_ours, describe);
}

// The result tensors of a replayed op, with the handles the worker binds
// them to; the worker checks the shapes against its own results.
void write_expected(writer &w, const std::vector<c10::IValue> &results) {
  auto put = [&w](const at::Tensor &t) {
    write_desc(w, is_ours(t) ? describe(t) : tensor_desc());
  };
  for (const c10::IValue &r : results) {
    if (r.isTensor()) {
      put(r.toTensor());
    } else if (r.isTensorList()) {
      auto items = r.toTensorVector();
      w.put<uint32_t>(static_cast<uint32_t>(items.size()));
      for (const at::Tensor &t : items) {
        put(t);
      }
    }
  }
}

void replace_results(torch::jit::Stack *stack, size_t args_begin,
                     std::vector<c10::IValue> &results) {
  stack->erase(stack->begin() + args_begin, stack->end());
  for (c10::IValue &r : results) {
    stack->push_back(std::move(r));
  }
}

// Replays a learned plan: the same results the worker reported for these
// argument shapes, over fresh handles, and the op fire-and-forget.
void replay(const plan &p, const encoded &e, torch::jit::Stack *stack,
            size_t args_begin, size_t nargs,
            const std::vector<at::Tensor> &inputs) {
  std::vector<c10::Storage> groups(p.group_nbytes.size());
  std::vector<c10::IValue> results;
  auto build = [&](const plan_output &out) -> at::Tensor {
    c10::Storage storage;
    if (out.k == plan_output::fresh) {
      if (!groups[out.index]) {
        groups[out.index] =
            new_storage(p.group_nbytes[out.index],
                        result_device(inputs, stack, args_begin, nargs));
      }
      storage = groups[out.index];
    }
    return build_output(out, inputs, storage);
  };
  for (const plan_ret &ret : p.rets) {
    if (ret.is_list) {
      c10::List<at::Tensor> list;
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
  if (p.needs_rpc) {
    request req(LUPINE_RPC_lupineTorchOp);
    write_call(req.body(), e);
    write_expected(req.body(), results);
    req.send();
  }
  replace_results(stack, args_begin, results);
}

// A plan key is the pickled call with its tensors described by metadata
// alone. CPU tensor arguments travel by value, so a large one makes the call
// uncacheable rather than the key huge.
bool plan_key(const op_state &st, const encoded &e, std::string *key) {
  for (const at::Tensor &t : e.table) {
    if (t.defined() && !is_ours(t) && t.numel() > 64) {
      return false;
    }
  }
  writer w;
  if (!st.inplace) {
    w.put_bytes(e.bytes.data(), e.bytes.size());
  }
  std::unordered_map<uint64_t, uint64_t> ids;
  write_tensors(w, e.table, is_ours, [&](const at::Tensor &t) {
    return st.memo ? describe_shape_only(t) : describe_for_key(t, ids);
  });
  key->assign(reinterpret_cast<const char *>(w.buffer.data()), w.buffer.size());
  return true;
}

// Runs the op on the worker and builds its results from the report: each
// tensor is an argument, a view over an argument's storage, or a storage the
// worker created and handed a handle for. Returns the plan learned, or null
// when a result has no plan form.
std::shared_ptr<plan> run_sync(const op_state &st, const encoded &e,
                               torch::jit::Stack *stack, size_t args_begin,
                               size_t nargs,
                               const std::vector<at::Tensor> &inputs) {
  request req(LUPINE_RPC_lupineTorchOpSync);
  write_call(req.body(), e);
  req.call();
  read_status();
  std::vector<uint8_t> blob = read_blob();
  req.finish();

  reader r(blob.data(), blob.size());
  struct alias {
    uint8_t kind;
    int32_t index;
  };
  std::vector<alias> aliases(r.get<uint32_t>());
  for (alias &a : aliases) {
    a.kind = r.get<uint8_t>();
    a.index = r.get<int32_t>();
  }
  auto learned = std::make_shared<plan>();
  std::unordered_map<uint64_t, size_t> groups;
  std::vector<c10::Storage> group_storage;
  std::unordered_map<c10::TensorImpl *, plan_output> outputs;
  size_t next = 0;
  auto materialize = [&](const tensor_desc &d) {
    plan_output out;
    out.dtype = d.dtype;
    out.sizes = d.sizes;
    out.strides = d.strides;
    out.offset = d.offset;
    const alias &a = aliases.at(next++);
    c10::Storage storage;
    if (a.kind != 0) {
      const at::Tensor &in = inputs.at(a.index);
      out.k = a.kind == 1 ? plan_output::input : plan_output::input_storage;
      out.index = a.index;
      out.offset = d.offset - in.storage_offset();
      out.same_metadata = a.kind == 1 && in.sizes().equals(d.sizes) &&
                          in.strides().equals(d.strides) && out.offset == 0;
    } else {
      auto found = groups.find(d.handle);
      if (found == groups.end()) {
        found = groups.emplace(d.handle, group_storage.size()).first;
        group_storage.push_back(storage_for_handle(
            d.handle, d.nbytes,
            result_device(inputs, stack, args_begin, nargs)));
        learned->group_nbytes.push_back(d.nbytes);
      }
      out.k = plan_output::fresh;
      out.index = static_cast<int32_t>(found->second);
      storage = group_storage[found->second];
    }
    at::Tensor t = build_output(out, inputs, storage);
    outputs.emplace(t.unsafeGetTensorImpl(), std::move(out));
    return t;
  };
  std::vector<c10::IValue> results = read_stack(r, materialize);

  auto note = [&](const at::Tensor &t) {
    return t.defined() ? outputs.at(t.unsafeGetTensorImpl()) : plan_output();
  };
  bool cacheable = true;
  for (const c10::IValue &v : results) {
    plan_ret ret;
    if (v.isTensor()) {
      ret.items.push_back(note(v.toTensor()));
    } else if (v.isTensorList()) {
      ret.is_list = true;
      for (const at::Tensor &t : v.toTensorVector()) {
        ret.items.push_back(note(t));
      }
    } else {
      ret.items.emplace_back();
      plan_output &out = ret.items.back();
      if (v.isNone()) {
        out.k = plan_output::none;
      } else if (v.isBool()) {
        out.k = plan_output::boolean;
        out.ival = v.toBool() ? 1 : 0;
      } else if (v.isDouble()) {
        out.k = plan_output::real;
        out.dval = v.toDouble();
      } else if (v.isInt()) {
        out.k = plan_output::integer;
        out.ival = v.toInt();
      } else {
        cacheable = false;
      }
    }
    learned->rets.push_back(std::move(ret));
  }
  learned->needs_rpc = st.mutates || !learned->group_nbytes.empty();
  replace_results(stack, args_begin, results);
  return cacheable ? learned : nullptr;
}

void fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  TORCH_CHECK(g_conn != nullptr, "lupine: the torch backend is not connected");
  op_state &st = state_for(op);
  size_t nargs = op.schema().arguments().size();
  size_t args_begin = stack->size() - nargs;
  profile *prof = profile::instance();
  uint64_t started = prof != nullptr ? profile::now() : 0;
  encoded e = encode(st, stack, args_begin, nargs);
  std::vector<at::Tensor> inputs = device_inputs(e);
  std::string key;
  bool keyed = !st.sync &&
               !(st.bool_index && has_bool_index((*stack)[args_begin + 1])) &&
               plan_key(st, e, &key);
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
      replay(*found, e, stack, args_begin, nargs, inputs);
      if (prof != nullptr) {
        prof->record(st.name, profile::now() - started, 0);
      }
      if (trace_enabled()) {
        std::cerr << "lupine-torch replay " << st.name << std::endl;
      }
      return;
    }
  }
  if (trace_enabled()) {
    std::cerr << "lupine-torch sync " << st.name << std::endl;
  }
  std::shared_ptr<plan> learned =
      run_sync(st, e, stack, args_begin, nargs, inputs);
  if (keyed && learned) {
    std::lock_guard<std::mutex> lock(st.mutex);
    st.plans.emplace(std::move(key), std::move(learned));
  }
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
  t_alloc_device = c10::Device(d.type(), resolve_index(device));
  return at::detail::empty_generic(size, &g_allocator,
                                   c10::DispatchKeySet(key_for(d.type())),
                                   c10::dtype_or_default(dtype), memory_format);
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
  t_alloc_device = c10::Device(d.type(), resolve_index(device));
  return at::detail::empty_strided_generic(
      size, stride, &g_allocator, c10::DispatchKeySet(key_for(d.type())),
      c10::dtype_or_default(dtype));
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
  c10::TensorImpl *impl = self.unsafeGetTensorImpl();
  impl->set_sizes_contiguous(C10_AS_INTARRAYREF_SLOW(size));
  if (memory_format.has_value() &&
      *memory_format != c10::MemoryFormat::Contiguous) {
    impl->empty_tensor_restride(*memory_format);
  }
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
  return set_storage(self, new_storage(0, self.device()), 0, {0}, {});
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
