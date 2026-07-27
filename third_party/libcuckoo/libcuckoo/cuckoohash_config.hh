/** \file */

#ifndef _CUCKOOHASH_CONFIG_HH
#define _CUCKOOHASH_CONFIG_HH

#include <cstddef>
#include <limits>

namespace libcuckoo {

//! The default maximum number of keys per bucket
constexpr size_t DEFAULT_SLOT_PER_BUCKET = 4;

//! The default number of elements in an empty hash table
constexpr size_t DEFAULT_SIZE = (1U << 16) * DEFAULT_SLOT_PER_BUCKET;

//! The default minimum load factor that the table allows for automatic
//! expansion. It must be a number between 0.0 and 1.0. The table will throw
//! load_factor_too_low if the load factor falls below this value
//! during an automatic expansion.
constexpr double DEFAULT_MINIMUM_LOAD_FACTOR = 0.05;

//! An alias for the value that sets no limit on the maximum hashpower. If this
//! value is set as the maximum hashpower limit, there will be no limit. This
//! is also the default initial value for the maximum hashpower in a table.
constexpr size_t NO_MAXIMUM_HASHPOWER = std::numeric_limits<size_t>::max();

//! set LIBCUCKOO_DEBUG to 1 to enable debug output
#define LIBCUCKOO_DEBUG 0

} // namespace libcuckoo

#endif // _CUCKOOHASH_CONFIG_HH
