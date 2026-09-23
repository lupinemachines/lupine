// Tensors on the backend device are host-side metadata over a storage handle
// the worker owns. This file gives torch the device: storages that carry a
// handle instead of memory, the guard, hooks and generator, and the kernels
// that only touch metadata. Everything else reaches the Python fallback.
#include "backend.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/ops/_fused_sdp_choice.h>
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
#include <memory>
#include <vector>

namespace lupine_torch {

namespace {

c10::DeviceType g_device_type = c10::DeviceType::PrivateUse1;
bool g_dual = false;
int g_device_count = 0;
std::atomic<uint64_t> g_next_handle{1};
std::atomic<release_fn> g_release{nullptr};
std::atomic<sync_fn> g_sync{nullptr};
thread_local c10::DeviceIndex t_current_device = 0;
// The device new storages are placed on while an empty() call runs.
thread_local c10::Device t_alloc_device(c10::DeviceType::PrivateUse1, 0);

bool is_ours_type(c10::DeviceType type) {
  return type == c10::DeviceType::PrivateUse1 ||
         (g_dual && type == c10::DeviceType::CUDA);
}

c10::DispatchKey key_for(c10::DeviceType type) {
  return type == c10::DeviceType::CUDA ? c10::DispatchKey::CUDA
                                       : c10::DispatchKey::PrivateUse1;
}

void *fake_pointer(uint64_t handle) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(handle)
                                  << kHandleShift);
}

void synchronize() {
  sync_fn sync = g_sync.load();
  if (sync != nullptr) {
    sync();
  }
}

void release_handle(void *context) {
  release_fn release = g_release.load();
  if (release != nullptr) {
    release(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(context)));
  }
}

struct lupine_allocator final : at::Allocator {
  at::DataPtr allocate(size_t) override {
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
// Kernels
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
  t_alloc_device = self.device();
  c10::Storage storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), 0, &g_allocator,
      /*resizable=*/true));
  return set_storage(self, std::move(storage), 0, {0}, {});
}

at::Tensor _pin_memory(const at::Tensor &self, std::optional<c10::Device>) {
  return self.clone();
}

bool _has_compatible_shallow_copy_type(const at::Tensor &, const at::Tensor &) {
  return true;
}

// The three metadata edits Python needs that no aten op expresses: a storage
// under a handle the worker assigned, a view over a tensor's storage in any
// dtype, and an in-place retarget of sizes, strides and offset (an out= tensor
// the kernel resized); the views grow the storage the host believes in when
// they reach past it.
at::Tensor storage(int64_t handle, int64_t nbytes, c10::Device device) {
  c10::Storage storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes,
      at::DataPtr(fake_pointer(handle),
                  reinterpret_cast<void *>(static_cast<uintptr_t>(handle)),
                  &release_handle, device),
      &g_allocator, /*resizable=*/true));
  at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(key_for(device.type())),
      caffe2::TypeMeta::fromScalarType(at::kByte));
  t.unsafeGetTensorImpl()->set_sizes_contiguous({nbytes});
  return t;
}

at::Tensor view_over(const at::Tensor &base, c10::ScalarType dtype,
                     c10::IntArrayRef size, c10::IntArrayRef stride,
                     int64_t offset) {
  at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(
      c10::Storage(base.storage()), base.key_set(),
      caffe2::TypeMeta::fromScalarType(dtype));
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, offset);
  ensure_storage(t);
  return t;
}

at::Tensor &set_metadata_(at::Tensor &self, c10::IntArrayRef size,
                          c10::IntArrayRef stride, int64_t offset) {
  self.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, offset);
  ensure_storage(self);
  return self;
}

// The composite scaled_dot_product_attention asks this stub which fused
// kernel to use; without it every backend but CPU and CUDA gets the math
// decomposition. The op itself has no kernel here, so it reaches the Python
// fallback and the worker's CUDA torch answers.
int64_t sdp_choice(const at::Tensor &query, const at::Tensor &key,
                   const at::Tensor &value,
                   const std::optional<at::Tensor> &attn_mask, double dropout_p,
                   bool is_causal, std::optional<double> scale,
                   bool enable_gqa) {
  return at::_fused_sdp_choice(query, key, value, attn_mask, dropout_p,
                               is_causal, scale, enable_gqa);
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

// The device module forwards seeds to the worker's CUDA generator; the host
// object only remembers the seed for initial_seed().
struct generator_impl final : c10::GeneratorImpl {
  explicit generator_impl(c10::DeviceIndex index)
      : c10::GeneratorImpl(c10::Device(g_device_type, index),
                           c10::DispatchKeySet(key_for(g_device_type))) {}
  void set_current_seed(uint64_t seed) override { seed_ = seed; }
  void set_offset(uint64_t) override {}
  uint64_t get_offset() const override { return 0; }
  uint64_t current_seed() const override { return seed_; }
  uint64_t seed() override {
    seed_ = c10::detail::getNonDeterministicRandom(true);
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
    return static_cast<c10::DeviceIndex>(lupine_torch::exchange_device(d));
  }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex d) const override {
    return d < 0 ? t_current_device : exchangeDevice(d);
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
    return static_cast<c10::DeviceIndex>(lupine_torch::exchange_device(d));
  }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex d) const override {
    return d < 0 ? t_current_device : exchangeDevice(d);
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
  aten->impl("_pin_memory", TORCH_FN(_pin_memory));
  aten->impl("_has_compatible_shallow_copy_type",
             TORCH_FN(_has_compatible_shallow_copy_type));
  libraries.push_back(std::move(aten));
  if (key == c10::DispatchKey::CUDA) {
    at::native::_fused_sdp_choice_stub.set_cuda_dispatch_ptr(&sdp_choice);
  } else {
    at::native::_fused_sdp_choice_stub.set_privateuse1_dispatch_ptr(
        &sdp_choice);
  }
}

} // namespace

TORCH_LIBRARY(lupine, m) {
  m.def("storage(int handle, int nbytes, Device device) -> Tensor");
  m.def("view_over(Tensor base, ScalarType dtype, int[] size, int[] stride, "
        "int offset) -> Tensor");
  m.def("set_metadata_(Tensor(a!) self, int[] size, int[] stride, int offset) "
        "-> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(lupine, CompositeExplicitAutograd, m) {
  m.impl("storage", TORCH_FN(storage));
  m.impl("view_over", TORCH_FN(view_over));
  m.impl("set_metadata_", TORCH_FN(set_metadata_));
}

void register_backend(int device_count, bool dual, release_fn release,
                      sync_fn sync) {
  TORCH_CHECK(g_device_count == 0, "lupine: the torch backend is already "
                                   "registered in this process");
  TORCH_CHECK(device_count > 0, "lupine: the worker has no device");
  g_device_count = device_count;
  g_dual = dual;
  g_device_type = dual ? c10::DeviceType::CUDA : c10::DeviceType::PrivateUse1;
  g_release.store(release);
  g_sync.store(sync);
  static guard_impl private_guard(c10::DeviceType::PrivateUse1);
  static guard_impl cuda_guard(c10::DeviceType::CUDA);
  c10::impl::DeviceGuardImplRegistrar(c10::DeviceType::PrivateUse1,
                                      &private_guard);
  at::RegisterPrivateUse1HooksInterface(new hooks_impl());
  register_kernels(c10::DispatchKey::PrivateUse1);
  if (dual) {
    TORCH_CHECK(at::detail::getCUDAHooks().isBuilt() &&
                    at::detail::getCUDAHooks().deviceCount() == device_count,
                "lupine: torch created its CUDA hooks before the backend "
                "loaded; import lupine and call lupine.connect() before "
                "importing torch");
    c10::impl::DeviceGuardImplRegistrar(c10::DeviceType::CUDA, &cuda_guard);
    register_kernels(c10::DispatchKey::CUDA);
  }
}

void clear_callbacks() {
  g_release.store(nullptr);
  g_sync.store(nullptr);
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

int device_count() { return g_device_count; }

int current_device() { return t_current_device; }

void set_device(int index) {
  TORCH_CHECK(index >= 0 && index < g_device_count, "lupine: invalid device ",
              index);
  t_current_device = static_cast<c10::DeviceIndex>(index);
}

int exchange_device(int index) {
  int previous = t_current_device;
  if (index >= 0) {
    set_device(index);
  }
  return previous;
}

} // namespace lupine_torch
