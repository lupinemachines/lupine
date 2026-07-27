/* For each table to support, we define a wrapper class which holds the table
 * and implements all of the benchmarked operations. Below we list all the
 * methods each wrapper must implement.
 *
 * constructor(size_t n) // n is the initial capacity
 * template <typename K, typename V>
 * bool read(const K& k, V& v) const
 * template <typename K, typename V>
 * bool insert(const K& k, const V& v)
 * template <typename K>
 * bool erase(const K& k)
 * template <typename K, typename V>
 * bool update(const K& k, const V& v)
 * template <typename K, typename V>
 * void upsert(const K& k, Updater fn, const V& v)
 */

#ifndef _UNIVERSAL_TABLE_WRAPPER_HH
#define _UNIVERSAL_TABLE_WRAPPER_HH

#include <atomic>
#include <memory>
#include <utility>

#ifndef KEY
#error Must define KEY symbol as valid key type
#endif

#ifndef VALUE
#error Must define VALUE symbol as valid value type
#endif

#ifdef TRACKING_ALLOCATOR
std::atomic<size_t> universal_benchmark_current_bytes_allocated =
    ATOMIC_VAR_INIT(0);

template <template <typename> class WrappedAlloc, typename T> class Allocator {
private:
  using traits = std::allocator_traits<WrappedAlloc<T>>;

public:
  using value_type = typename traits::value_type;
  using pointer = typename traits::pointer;
  using const_pointer = typename traits::const_pointer;
  using reference = value_type &;
  using const_reference = const value_type &;
  using void_pointer = typename traits::void_pointer;
  using const_void_pointer = typename traits::const_void_pointer;
  using size_type = typename traits::size_type;
  using difference_type = typename traits::difference_type;

  template <typename U> struct rebind {
    using other = Allocator<WrappedAlloc, U>;
  };

  Allocator() {}

  template <typename U>
  Allocator(const Allocator<WrappedAlloc, U> &alloc)
      : allocator_(alloc.allocator_) {}

  pointer address(reference x) const { return std::addressof(x); }

  const_pointer const_address(const_reference x) const {
    return std::addressof(x);
  }

  pointer allocate(size_type n, const_void_pointer hint = nullptr) {
    universal_benchmark_current_bytes_allocated.fetch_add(
        n * sizeof(value_type), std::memory_order_acq_rel);
    return traits::allocate(allocator_, n, hint);
  }

  void deallocate(pointer p, size_type n) {
    universal_benchmark_current_bytes_allocated.fetch_sub(
        n * sizeof(value_type), std::memory_order_acq_rel);
    traits::deallocate(allocator_, p, n);
  }

  size_type max_size() const { return traits::max_size(allocator_); }

  template <typename... Args> void construct(pointer p, Args &&... args) {
    traits::construct(allocator_, p, std::forward<Args>(args)...);
  }

  void destroy(pointer p) { traits::destroy(allocator_, p); }

  bool operator==(const Allocator &) { return true; }

  bool operator!=(const Allocator &) { return false; }

  typename traits::allocator_type allocator_;
};

class Sampler {
public:
  Sampler(size_t num_ops)
      : total_samples(num_ops / sampling_period), current_iter(0),
        samples(total_samples) {}

  void iter() {
    if ((current_iter & sampling_period_mask) == 0) {
      samples[current_iter >> sampling_period_power] =
          universal_benchmark_current_bytes_allocated.load(
              std::memory_order_acquire);
    }
    ++current_iter;
  }

  void store(std::vector<size_t> &v) const { v = samples; }

private:
  // We sample every 2^17 iterations (which is around 100K). Keeping the
  // sampling period a power of 2 makes it easy to check whether we need to
  // add a sample.
  static constexpr size_t sampling_period_power = 17;
  static constexpr size_t sampling_period = 1UL << 17;
  static constexpr size_t sampling_period_mask = sampling_period - 1;
  const size_t total_samples;
  size_t current_iter;
  std::vector<size_t> samples;
};

#else
template <template <typename> class WrappedAlloc, typename T>
using Allocator = WrappedAlloc<T>;

class Sampler {
public:
  Sampler(size_t) {}
  void iter() {}
  void store(std::vector<size_t> &v) const { v = samples; }

private:
  std::vector<size_t> samples;
};

#endif

#ifdef LIBCUCKOO
#define TABLE "LIBCUCKOO"
#define TABLE_TYPE "cuckoohash_map"
#include <libcuckoo/cuckoohash_map.hh>

class Table {
public:
  Table(size_t n) : tbl(n) {}

  template <typename K, typename V> bool read(const K &k, V &v) const {
    return tbl.find(k, v);
  }

  template <typename K, typename V> bool insert(const K &k, const V &v) {
    return tbl.insert(k, v);
  }

  template <typename K> bool erase(const K &k) { return tbl.erase(k); }

  template <typename K, typename V> bool update(const K &k, const V &v) {
    return tbl.update(k, v);
  }

  template <typename K, typename Updater, typename V>
  void upsert(const K &k, Updater fn, const V &v) {
    tbl.upsert(k, fn, v);
  }

private:
  libcuckoo::cuckoohash_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>,
                 Allocator<std::allocator, std::pair<const KEY, VALUE>>>
      tbl;
};

#else
#error Must define LIBCUCKOO
#endif

#endif // _UNIVERSAL_TABLE_WRAPPER_HH
