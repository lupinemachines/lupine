/** \file */

#ifndef _CUCKOOHASH_UTIL_HH
#define _CUCKOOHASH_UTIL_HH

#include "cuckoohash_config.hh" // for LIBCUCKOO_DEBUG
#include <exception>
#include <thread>
#include <utility>
#include <vector>

namespace libcuckoo {

#if LIBCUCKOO_DEBUG
//! When \ref LIBCUCKOO_DEBUG is 0, LIBCUCKOO_DBG will printing out status
//! messages in various situations
#define LIBCUCKOO_DBG(fmt, ...)                                                \
  fprintf(stderr,                                                              \
          "\x1b[32m"                                                           \
          "[libcuckoo:%s:%d:%lu] " fmt ""                                      \
          "\x1b[0m",                                                           \
          __FILE__, __LINE__,                                                  \
          std::hash<std::thread::id>()(std::this_thread::get_id()),            \
          __VA_ARGS__)
#else
//! When \ref LIBCUCKOO_DEBUG is 0, LIBCUCKOO_DBG does nothing
#define LIBCUCKOO_DBG(fmt, ...)                                                \
  do {                                                                         \
  } while (0)
#endif

/**
 * alignas() requires GCC >= 4.9, so we stick with the alignment attribute for
 * GCC.
 */
#ifdef __GNUC__
#define LIBCUCKOO_ALIGNAS(x) __attribute__((aligned(x)))
#else
#define LIBCUCKOO_ALIGNAS(x) alignas(x)
#endif

/**
 * At higher warning levels, MSVC produces an annoying warning that alignment
 * may cause wasted space: "structure was padded due to __declspec(align())".
 */
#ifdef _MSC_VER
#define LIBCUCKOO_SQUELCH_PADDING_WARNING __pragma(warning(suppress : 4324))
#else
#define LIBCUCKOO_SQUELCH_PADDING_WARNING
#endif

/**
 * At higher warning levels, MSVC may issue a deadcode warning which depends on
 * the template arguments given. For certain other template arguments, the code
 * is not really "dead".
 */
#ifdef _MSC_VER
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_BEGIN                               \
  do {                                                                         \
    __pragma(warning(push));                                                   \
    __pragma(warning(disable : 4702))                                          \
  } while (0)
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_END __pragma(warning(pop))
#else
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_BEGIN
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_END
#endif

/**
 * Thrown when an automatic expansion is triggered, but the load factor of the
 * table is below a minimum threshold, which can be set by the \ref
 * cuckoohash_map::minimum_load_factor method. This can happen if the hash
 * function does not properly distribute keys, or for certain adversarial
 * workloads.
 */
class load_factor_too_low : public std::exception {
public:
  /**
   * Constructor
   *
   * @param lf the load factor of the table when the exception was thrown
   */
  load_factor_too_low(const double lf) noexcept : load_factor_(lf) {}

  /**
   * @return a descriptive error message
   */
  virtual const char *what() const noexcept override {
    return "Automatic expansion triggered when load factor was below "
           "minimum threshold";
  }

  /**
   * @return the load factor of the table when the exception was thrown
   */
  double load_factor() const noexcept { return load_factor_; }

private:
  const double load_factor_;
};

/**
 * Thrown when an expansion is triggered, but the hashpower specified is greater
 * than the maximum, which can be set with the \ref
 * cuckoohash_map::maximum_hashpower method.
 */
class maximum_hashpower_exceeded : public std::exception {
public:
  /**
   * Constructor
   *
   * @param hp the hash power we were trying to expand to
   */
  maximum_hashpower_exceeded(const size_t hp) noexcept : hashpower_(hp) {}

  /**
   * @return a descriptive error message
   */
  virtual const char *what() const noexcept override {
    return "Expansion beyond maximum hashpower";
  }

  /**
   * @return the hashpower we were trying to expand to
   */
  size_t hashpower() const noexcept { return hashpower_; }

private:
  const size_t hashpower_;
};

/**
 * This enum indicates whether an insertion took place, or whether the
 * key-value pair was already in the table. See \ref cuckoohash_map::uprase_fn
 * for usage details.
 */
enum class UpsertContext {
  NEWLY_INSERTED,
  ALREADY_EXISTED,
};

namespace internal {

// Used to invoke the \ref uprase_fn functor with or without an \ref
// UpsertContext enum.  Note that if we cannot pass an upsert context and the
// desired context is <tt>UpsertContext:::NEWLY_INSERTED</tt>, then we do not
// invoke the functor at all.
//
// We implement this utility using C++11-style SFINAE, for maximum
// compatibility.
template <typename F, typename MappedType> class CanInvokeWithUpsertContext {
private:
  template <typename InnerF,
            typename = decltype(std::declval<InnerF>()(
                std::declval<MappedType &>(), std::declval<UpsertContext>()))>
  static std::true_type test(int);

  // Note: The argument type needs to be less-preferable than the first
  // overload so that it is picked only if the first overload cannot be
  // instantiated.
  template <typename InnerF> static std::false_type test(float);

public:
  using type = decltype(test<F>(0));
};

template <typename F, typename MappedType>
bool InvokeUpraseFn(F &f, MappedType &mapped, UpsertContext context,
                    std::true_type) {
  return f(mapped, context);
}

template <typename F, typename MappedType>
bool InvokeUpraseFn(F &f, MappedType &mapped, UpsertContext context,
                    std::false_type) {
  if (context == UpsertContext::ALREADY_EXISTED) {
    return f(mapped);
  } else {
    // Returning false indicates no deletion, making this a no-op.
    return false;
  }
}

// Upgrades an upsert functor to an uprase functor, which always returns false,
// so that we never erase the element.
template <typename F, typename MappedType, bool kCanInvokeWithUpsertContext>
struct UpsertToUpraseFn;

template <typename F, typename MappedType>
struct UpsertToUpraseFn<F, MappedType, true> {
  F &f;

  bool operator()(MappedType &mapped, UpsertContext context) const {
    f(mapped, context);
    return false;
  }
};

template <typename F, typename MappedType>
struct UpsertToUpraseFn<F, MappedType, false> {
  F &f;

  bool operator()(MappedType &mapped) {
    f(mapped);
    return false;
  }
};

} // namespace internal

} // namespace libcuckoo

#endif // _CUCKOOHASH_UTIL_HH
