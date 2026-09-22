// Wire encoding of boxed torch operator calls shared by the host backend and
// the worker. Every value is a tag byte followed by its payload; tensors travel
// as a storage handle plus view metadata, so either end rebuilds the view over
// the storage it owns for that handle.
#ifndef LUPINE_TORCH_WIRE_H
#define LUPINE_TORCH_WIRE_H

#include <ATen/ATen.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <c10/core/Scalar.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lupine_torch {

struct tensor_desc {
  uint64_t handle = 0;
  // The storage size the sender believes; a receiver whose storage is smaller
  // grows it in place before building the view.
  uint64_t nbytes = 0;
  int16_t device = 0;
  int8_t dtype = 0;
  int64_t offset = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

enum class tag : uint8_t {
  none = 0,
  boolean,
  integer,
  real,
  string,
  tensor,
  cpu_tensor,
  device,
  dtype,
  layout,
  memory_format,
  scalar,
  list,
  complex,
  generator,
  undefined_tensor,
};

enum class list_kind : uint8_t {
  generic = 0,
  integer,
  real,
  boolean,
  tensor,
  optional_tensor,
  scalar,
  string,
};

// Device kinds on the wire: the accelerator maps to the backend device on the
// host and to CUDA in the worker.
constexpr uint8_t kDeviceCpu = 0;
constexpr uint8_t kDeviceAccelerator = 1;

class writer {
public:
  std::vector<uint8_t> buffer;

  template <typename T> void put(const T &value) {
    static_assert(std::is_trivially_copyable<T>::value, "pod only");
    size_t at = buffer.size();
    buffer.resize(at + sizeof(T));
    std::memcpy(buffer.data() + at, &value, sizeof(T));
  }
  template <typename T> void patch(size_t at, const T &value) {
    std::memcpy(buffer.data() + at, &value, sizeof(T));
  }
  void put_bytes(const void *data, size_t size) {
    size_t at = buffer.size();
    buffer.resize(at + size);
    if (size != 0) {
      std::memcpy(buffer.data() + at, data, size);
    }
  }
  void put_string(const std::string &value) {
    put<uint32_t>(static_cast<uint32_t>(value.size()));
    put_bytes(value.data(), value.size());
  }
  void put_tag(tag value) { put<uint8_t>(static_cast<uint8_t>(value)); }
};

class reader {
public:
  reader(const uint8_t *data, size_t size) : data_(data), size_(size) {}

  template <typename T> T get() {
    T value;
    TORCH_CHECK(cursor_ + sizeof(T) <= size_, "lupine wire: truncated payload");
    std::memcpy(&value, data_ + cursor_, sizeof(T));
    cursor_ += sizeof(T);
    return value;
  }
  const uint8_t *get_bytes(size_t size) {
    TORCH_CHECK(cursor_ + size <= size_, "lupine wire: truncated payload");
    const uint8_t *at = data_ + cursor_;
    cursor_ += size;
    return at;
  }
  std::string get_string() {
    uint32_t size = get<uint32_t>();
    const uint8_t *at = get_bytes(size);
    return std::string(reinterpret_cast<const char *>(at), size);
  }
  tag get_tag() { return static_cast<tag>(get<uint8_t>()); }
  bool done() const { return cursor_ == size_; }

private:
  const uint8_t *data_;
  size_t size_;
  size_t cursor_ = 0;
};

inline void write_desc(writer &w, const tensor_desc &d) {
  w.put<uint64_t>(d.handle);
  w.put<uint64_t>(d.nbytes);
  w.put<int16_t>(d.device);
  w.put<int8_t>(d.dtype);
  w.put<int64_t>(d.offset);
  w.put<uint8_t>(static_cast<uint8_t>(d.sizes.size()));
  for (size_t i = 0; i < d.sizes.size(); ++i) {
    w.put<int64_t>(d.sizes[i]);
    w.put<int64_t>(d.strides[i]);
  }
}

inline tensor_desc read_desc(reader &r) {
  tensor_desc d;
  d.handle = r.get<uint64_t>();
  d.nbytes = r.get<uint64_t>();
  d.device = r.get<int16_t>();
  d.dtype = r.get<int8_t>();
  d.offset = r.get<int64_t>();
  uint8_t ndim = r.get<uint8_t>();
  d.sizes.resize(ndim);
  d.strides.resize(ndim);
  for (uint8_t i = 0; i < ndim; ++i) {
    d.sizes[i] = r.get<int64_t>();
    d.strides[i] = r.get<int64_t>();
  }
  return d;
}

// Describes a tensor that lives on the sender's accelerator device.
using desc_of = std::function<tensor_desc(const at::Tensor &)>;
// Rebuilds a tensor over the receiver's storage for a descriptor.
using tensor_of = std::function<at::Tensor(const tensor_desc &)>;

inline void write_scalar(writer &w, const c10::Scalar &s) {
  if (s.isBoolean()) {
    w.put<uint8_t>(2);
    w.put<uint8_t>(s.toBool() ? 1 : 0);
  } else if (s.isIntegral(false)) {
    w.put<uint8_t>(0);
    w.put<int64_t>(s.toLong());
  } else if (s.isComplex()) {
    w.put<uint8_t>(3);
    auto c = s.toComplexDouble();
    w.put<double>(c.real());
    w.put<double>(c.imag());
  } else {
    w.put<uint8_t>(1);
    w.put<double>(s.toDouble());
  }
}

inline c10::Scalar read_scalar(reader &r) {
  switch (r.get<uint8_t>()) {
  case 0:
    return c10::Scalar(r.get<int64_t>());
  case 1:
    return c10::Scalar(r.get<double>());
  case 2:
    return c10::Scalar(r.get<uint8_t>() != 0);
  default: {
    double re = r.get<double>();
    double im = r.get<double>();
    return c10::Scalar(c10::complex<double>(re, im));
  }
  }
}

inline void write_device(writer &w, const c10::Device &d, bool accelerator) {
  w.put<uint8_t>(accelerator ? kDeviceAccelerator : kDeviceCpu);
  w.put<int16_t>(d.has_index() ? d.index() : -1);
}

inline void write_cpu_tensor(writer &w, const at::Tensor &t) {
  at::Tensor c = t.contiguous();
  w.put<int8_t>(static_cast<int8_t>(c.scalar_type()));
  w.put<uint8_t>(static_cast<uint8_t>(c.dim()));
  for (int64_t s : c.sizes()) {
    w.put<int64_t>(s);
  }
  size_t nbytes = c.numel() * c.element_size();
  w.put<uint64_t>(nbytes);
  w.put_bytes(c.const_data_ptr(), nbytes);
}

inline at::Tensor read_cpu_tensor(reader &r) {
  auto dtype = static_cast<c10::ScalarType>(r.get<int8_t>());
  uint8_t ndim = r.get<uint8_t>();
  std::vector<int64_t> sizes(ndim);
  for (uint8_t i = 0; i < ndim; ++i) {
    sizes[i] = r.get<int64_t>();
  }
  uint64_t nbytes = r.get<uint64_t>();
  const uint8_t *bytes = r.get_bytes(nbytes);
  at::Tensor t = at::empty(sizes, at::TensorOptions().dtype(dtype));
  if (nbytes != 0) {
    std::memcpy(t.mutable_data_ptr(), bytes, nbytes);
  }
  return t;
}

struct wire_context {
  desc_of describe;
  // Whether a tensor belongs to the sender's accelerator device.
  std::function<bool(const at::Tensor &)> is_accelerator;
  std::function<bool(const c10::Device &)> is_accelerator_device;
};

inline void write_ivalue(writer &w, const c10::IValue &v,
                         const wire_context &ctx);

inline list_kind kind_of(const c10::IValue &v) {
  if (v.isIntList()) {
    return list_kind::integer;
  }
  if (v.isDoubleList()) {
    return list_kind::real;
  }
  if (v.isBoolList()) {
    return list_kind::boolean;
  }
  if (v.isTensorList()) {
    return list_kind::tensor;
  }
  auto type = v.toList().elementType();
  if (type->kind() == c10::TypeKind::OptionalType &&
      type->expectRef<c10::OptionalType>().getElementType()->kind() ==
          c10::TypeKind::TensorType) {
    return list_kind::optional_tensor;
  }
  if (type->kind() == c10::TypeKind::NumberType) {
    return list_kind::scalar;
  }
  if (type->kind() == c10::TypeKind::StringType) {
    return list_kind::string;
  }
  if (type->kind() == c10::TypeKind::TensorType) {
    return list_kind::tensor;
  }
  return list_kind::generic;
}

inline void write_ivalue(writer &w, const c10::IValue &v,
                         const wire_context &ctx) {
  if (v.isNone()) {
    w.put_tag(tag::none);
  } else if (v.isBool()) {
    w.put_tag(tag::boolean);
    w.put<uint8_t>(v.toBool() ? 1 : 0);
  } else if (v.isInt()) {
    w.put_tag(tag::integer);
    w.put<int64_t>(v.toInt());
  } else if (v.isSymInt()) {
    w.put_tag(tag::integer);
    w.put<int64_t>(v.toSymInt().expect_int());
  } else if (v.isDouble()) {
    w.put_tag(tag::real);
    w.put<double>(v.toDouble());
  } else if (v.isSymFloat()) {
    w.put_tag(tag::real);
    w.put<double>(v.toSymFloat().expect_float());
  } else if (v.isSymBool()) {
    w.put_tag(tag::boolean);
    w.put<uint8_t>(v.toSymBool().expect_bool() ? 1 : 0);
  } else if (v.isComplexDouble()) {
    w.put_tag(tag::complex);
    auto c = v.toComplexDouble();
    w.put<double>(c.real());
    w.put<double>(c.imag());
  } else if (v.isString()) {
    w.put_tag(tag::string);
    w.put_string(v.toStringRef());
  } else if (v.isTensor()) {
    const at::Tensor &t = v.toTensor();
    if (!t.defined()) {
      w.put_tag(tag::undefined_tensor);
    } else if (ctx.is_accelerator(t)) {
      w.put_tag(tag::tensor);
      write_desc(w, ctx.describe(t));
    } else {
      TORCH_CHECK(t.device().is_cpu(), "lupine: cannot send a tensor on ",
                  t.device(), " to the worker");
      w.put_tag(tag::cpu_tensor);
      write_cpu_tensor(w, t);
    }
  } else if (v.isDevice()) {
    w.put_tag(tag::device);
    write_device(w, v.toDevice(), ctx.is_accelerator_device(v.toDevice()));
  } else if (v.isScalar()) {
    w.put_tag(tag::scalar);
    write_scalar(w, v.toScalar());
  } else if (v.isGenerator()) {
    TORCH_CHECK(!v.toGenerator().defined(),
                "lupine: explicit torch.Generator objects are not supported "
                "on the lupine backend; use torch.manual_seed");
    w.put_tag(tag::none);
  } else if (v.isList()) {
    w.put_tag(tag::list);
    list_kind kind = kind_of(v);
    w.put<uint8_t>(static_cast<uint8_t>(kind));
    auto list = v.toListRef();
    w.put<uint32_t>(static_cast<uint32_t>(list.size()));
    for (const c10::IValue &item : list) {
      write_ivalue(w, item, ctx);
    }
  } else if (v.isTuple()) {
    w.put_tag(tag::list);
    w.put<uint8_t>(static_cast<uint8_t>(list_kind::generic));
    const auto &items = v.toTupleRef().elements();
    w.put<uint32_t>(static_cast<uint32_t>(items.size()));
    for (const c10::IValue &item : items) {
      write_ivalue(w, item, ctx);
    }
  } else if (v.isStorage()) {
    TORCH_CHECK(false, "lupine: storage arguments are handled on the host");
  } else {
    // ScalarType, Layout and MemoryFormat are ints in IValue; they take the
    // integer path above. Anything else has no wire form.
    TORCH_CHECK(false, "lupine: unsupported operator argument ", v.tagKind());
  }
}

struct read_context {
  tensor_of materialize;
  c10::DeviceType accelerator = c10::DeviceType::CUDA;
};

inline c10::IValue read_ivalue(reader &r, const read_context &ctx) {
  switch (r.get_tag()) {
  case tag::none:
    return c10::IValue();
  case tag::boolean:
    return c10::IValue(r.get<uint8_t>() != 0);
  case tag::integer:
    return c10::IValue(r.get<int64_t>());
  case tag::real:
    return c10::IValue(r.get<double>());
  case tag::complex: {
    double re = r.get<double>();
    double im = r.get<double>();
    return c10::IValue(c10::complex<double>(re, im));
  }
  case tag::string:
    return c10::IValue(r.get_string());
  case tag::tensor:
    return c10::IValue(ctx.materialize(read_desc(r)));
  case tag::undefined_tensor:
    return c10::IValue(at::Tensor());
  case tag::cpu_tensor:
    return c10::IValue(read_cpu_tensor(r));
  case tag::device: {
    uint8_t kind = r.get<uint8_t>();
    int16_t index = r.get<int16_t>();
    if (kind == kDeviceCpu) {
      return c10::IValue(c10::Device(c10::DeviceType::CPU));
    }
    return c10::IValue(
        c10::Device(ctx.accelerator, static_cast<int8_t>(index)));
  }
  case tag::dtype:
    return c10::IValue(static_cast<int64_t>(r.get<int8_t>()));
  case tag::layout:
  case tag::memory_format:
    return c10::IValue(static_cast<int64_t>(r.get<int8_t>()));
  case tag::scalar:
    return c10::IValue(read_scalar(r));
  case tag::generator:
    return c10::IValue();
  case tag::list: {
    auto kind = static_cast<list_kind>(r.get<uint8_t>());
    uint32_t count = r.get<uint32_t>();
    switch (kind) {
    case list_kind::integer: {
      c10::List<int64_t> list;
      list.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        list.push_back(read_ivalue(r, ctx).toInt());
      }
      return c10::IValue(std::move(list));
    }
    case list_kind::real: {
      c10::List<double> list;
      for (uint32_t i = 0; i < count; ++i) {
        list.push_back(read_ivalue(r, ctx).toDouble());
      }
      return c10::IValue(std::move(list));
    }
    case list_kind::boolean: {
      c10::List<bool> list;
      for (uint32_t i = 0; i < count; ++i) {
        list.push_back(read_ivalue(r, ctx).toBool());
      }
      return c10::IValue(std::move(list));
    }
    case list_kind::tensor: {
      c10::List<at::Tensor> list;
      for (uint32_t i = 0; i < count; ++i) {
        c10::IValue item = read_ivalue(r, ctx);
        list.push_back(item.isNone() ? at::Tensor() : item.toTensor());
      }
      return c10::IValue(std::move(list));
    }
    case list_kind::optional_tensor: {
      c10::List<std::optional<at::Tensor>> list;
      for (uint32_t i = 0; i < count; ++i) {
        c10::IValue item = read_ivalue(r, ctx);
        if (item.isNone()) {
          list.push_back(std::nullopt);
        } else {
          list.push_back(item.toTensor());
        }
      }
      return c10::IValue(std::move(list));
    }
    case list_kind::scalar: {
      c10::List<c10::Scalar> list;
      for (uint32_t i = 0; i < count; ++i) {
        list.push_back(read_ivalue(r, ctx).toScalar());
      }
      return c10::IValue(std::move(list));
    }
    case list_kind::string: {
      c10::List<std::string> list;
      for (uint32_t i = 0; i < count; ++i) {
        list.push_back(read_ivalue(r, ctx).toStringRef());
      }
      return c10::IValue(std::move(list));
    }
    default: {
      c10::List<c10::IValue> list(c10::AnyType::get());
      for (uint32_t i = 0; i < count; ++i) {
        list.push_back(read_ivalue(r, ctx));
      }
      return c10::IValue(std::move(list));
    }
    }
  }
  }
  TORCH_CHECK(false, "lupine wire: unknown tag");
}

// Per-op timing, enabled by LUPINE_TORCH_PROFILE=<path>: one line per op
// name with its count and the nanoseconds spent in each phase, written at
// exit.
class profile {
public:
  static profile *instance() {
    static profile *p = [] {
      const char *path = getenv("LUPINE_TORCH_PROFILE");
      return path == nullptr ? nullptr : new profile(path);
    }();
    return p;
  }
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void record(const std::string &name, uint64_t phase_a, uint64_t phase_b) {
    std::lock_guard<std::mutex> lock(mutex_);
    entry &e = entries_[name];
    ++e.count;
    e.a += phase_a;
    e.b += phase_b;
  }

private:
  struct entry {
    uint64_t count = 0;
    uint64_t a = 0;
    uint64_t b = 0;
  };
  explicit profile(const char *path) : path_(path) {
    std::atexit([] {
      profile *p = instance();
      FILE *f = fopen(p->path_.c_str(), "w");
      if (f == nullptr) {
        return;
      }
      for (const auto &item : p->entries_) {
        fprintf(f, "%s\t%llu\t%llu\t%llu\n", item.first.c_str(),
                (unsigned long long)item.second.count,
                (unsigned long long)item.second.a,
                (unsigned long long)item.second.b);
      }
      fclose(f);
    });
  }
  std::string path_;
  std::mutex mutex_;
  std::unordered_map<std::string, entry> entries_;
};

} // namespace lupine_torch

#endif
