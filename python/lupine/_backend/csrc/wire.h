// Wire encoding of boxed torch operator calls shared by the host backend and
// the worker: torch's own pickler serialises the argument and result stacks,
// with every tensor replaced by a reference into a table sent alongside. A
// table entry is a storage handle plus view metadata, so either end rebuilds
// the view over the storage it owns for that handle, or a CPU tensor by value.
#ifndef LUPINE_TORCH_WIRE_H
#define LUPINE_TORCH_WIRE_H

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/serialization/pickle.h>

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
using is_device = std::function<bool(const at::Tensor &)>;

// How a tensor table entry travels: a storage handle plus view metadata, a
// CPU tensor by value, or nothing.
enum class tensor_kind : uint8_t { undefined, handle, cpu };

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

// The tensor table the pickle refers to, in the pickler's order.
inline void write_tensors(writer &w, const std::vector<at::Tensor> &table,
                          const is_device &ours, const desc_of &describe) {
  w.put<uint32_t>(static_cast<uint32_t>(table.size()));
  for (const at::Tensor &t : table) {
    if (!t.defined()) {
      w.put<uint8_t>(static_cast<uint8_t>(tensor_kind::undefined));
    } else if (ours(t)) {
      w.put<uint8_t>(static_cast<uint8_t>(tensor_kind::handle));
      write_desc(w, describe(t));
    } else {
      TORCH_CHECK(t.device().is_cpu(), "lupine: cannot send a tensor on ",
                  t.device(), " to the worker");
      w.put<uint8_t>(static_cast<uint8_t>(tensor_kind::cpu));
      write_cpu_tensor(w, t);
    }
  }
}

inline std::vector<at::Tensor> read_tensors(reader &r,
                                            const tensor_of &materialize) {
  std::vector<at::Tensor> table(r.get<uint32_t>());
  for (at::Tensor &t : table) {
    switch (static_cast<tensor_kind>(r.get<uint8_t>())) {
    case tensor_kind::handle:
      t = materialize(read_desc(r));
      break;
    case tensor_kind::cpu:
      t = read_cpu_tensor(r);
      break;
    default:
      break;
    }
  }
  return table;
}

// A stack on the wire: torch's pickle of a tuple of its values, tensors
// replaced by table references, then the table.
inline std::vector<char> pickle_stack(std::vector<c10::IValue> values,
                                      std::vector<at::Tensor> *table) {
  return torch::jit::pickle(c10::ivalue::Tuple::create(std::move(values)),
                            table);
}

inline void write_pickled(writer &w, const std::vector<char> &bytes,
                          const std::vector<at::Tensor> &table,
                          const is_device &ours, const desc_of &describe) {
  w.put<uint32_t>(static_cast<uint32_t>(bytes.size()));
  w.put_bytes(bytes.data(), bytes.size());
  write_tensors(w, table, ours, describe);
}

inline std::vector<c10::IValue> read_stack(reader &r,
                                           const tensor_of &materialize) {
  uint32_t size = r.get<uint32_t>();
  const uint8_t *bytes = r.get_bytes(size);
  std::vector<at::Tensor> table = read_tensors(r, materialize);
  return torch::jit::unpickle(reinterpret_cast<const char *>(bytes), size,
                              nullptr, table)
      .toTupleRef()
      .elements()
      .vec();
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
