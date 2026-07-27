#ifndef _TEST_UTIL_HH
#define _TEST_UTIL_HH

// Utilities for running stress tests and benchmarks
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>

#include <pcg/pcg_random.hpp>

std::mutex print_lock;
int main_return_value = EXIT_SUCCESS;
typedef std::lock_guard<std::mutex> mutex_guard;

// Prints a message if the two items aren't equal
template <class T, class U>
inline void do_expect_equal(T x, const char *xname, U y, const char *yname,
                            size_t line) {
  if (x != y) {
    mutex_guard m(print_lock);
    main_return_value = EXIT_FAILURE;
    std::cout << "ERROR:\t" << xname << "(" << x << ") does not equal " << yname
              << "(" << y << ") on line " << line << std::endl;
  }
}
#define EXPECT_EQ(x, y) do_expect_equal(x, #x, y, #y, __LINE__)

// Prints a message if the two items are equal
template <class T, class U>
inline void do_expect_not_equal(T x, const char *xname, U y, const char *yname,
                                size_t line) {
  if (x == y) {
    mutex_guard m(print_lock);
    main_return_value = EXIT_FAILURE;
    std::cout << "ERROR:\t" << xname << "(" << x << ") equals " << yname << "("
              << y << ") on line " << line << std::endl;
  }
}
#define EXPECT_NE(x, y) do_expect_not_equal(x, #x, y, #y, __LINE__)

// Prints a message if the item is false
inline void do_expect_true(bool x, const char *xname, size_t line) {
  if (!x) {
    mutex_guard m(print_lock);
    main_return_value = EXIT_FAILURE;
    std::cout << "ERROR:\t" << xname << "(" << x << ") is false on line "
              << line << std::endl;
  }
}
#define EXPECT_TRUE(x) do_expect_true(x, #x, __LINE__)

// Prints a message if the item is true
inline void do_expect_false(bool x, const char *xname, size_t line) {
  if (x) {
    mutex_guard m(print_lock);
    main_return_value = EXIT_FAILURE;
    std::cout << "ERROR:\t" << xname << "(" << x << ") is true on line " << line
              << std::endl;
  }
}
#define EXPECT_FALSE(x) do_expect_false(x, #x, __LINE__)

// Prints a message and exists if the two items aren't equal
template <class T, class U>
inline void do_assert_equal(T x, const char *xname, U y, const char *yname,
                            size_t line) {
  if (x != y) {
    mutex_guard m(print_lock);
    std::cout << "FATAL ERROR:\t" << xname << "(" << x << ") does not equal "
              << yname << "(" << y << ") on line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define ASSERT_EQ(x, y) do_assert_equal(x, #x, y, #y, __LINE__)

// Prints a message and exists if the item is false
inline void do_assert_true(bool x, const char *xname, size_t line) {
  if (!x) {
    mutex_guard m(print_lock);
    std::cout << "FATAL ERROR:\t" << xname << "(" << x << ") is false on line "
              << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define ASSERT_TRUE(x) do_assert_true(x, #x, __LINE__)

// Parses boolean flags and flags with positive integer arguments
void parse_flags(int argc, char **argv, const char *description,
                 const char *args[], size_t *arg_vars[], const char *arg_help[],
                 size_t arg_num, const char *flags[], bool *flag_vars[],
                 const char *flag_help[], size_t flag_num) {

  errno = 0;
  for (int i = 0; i < argc; i++) {
    for (size_t j = 0; j < arg_num; j++) {
      if (strcmp(argv[i], args[j]) == 0) {
        if (i == argc - 1) {
          std::cerr << "You must provide a positive integer argument"
                    << " after the " << args[j] << " argument" << std::endl;
          exit(EXIT_FAILURE);
        } else {
          size_t argval = strtoull(argv[i + 1], NULL, 10);
          if (errno != 0) {
            std::cerr << "The argument to " << args[j]
                      << " must be a valid size_t" << std::endl;
            exit(EXIT_FAILURE);
          } else {
            *(arg_vars[j]) = argval;
          }
        }
      }
    }
    for (size_t j = 0; j < flag_num; j++) {
      if (strcmp(argv[i], flags[j]) == 0) {
        *(flag_vars[j]) = true;
      }
    }
    if (strcmp(argv[i], "--help") == 0) {
      std::cerr << description << std::endl;
      std::cerr << "Arguments:" << std::endl;
      for (size_t j = 0; j < arg_num; j++) {
        std::cerr << args[j] << " (default " << *arg_vars[j] << "):\t"
                  << arg_help[j] << std::endl;
      }
      for (size_t j = 0; j < flag_num; j++) {
        std::cerr << flags[j] << " (default "
                  << (*flag_vars[j] ? "true" : "false") << "):\t"
                  << flag_help[j] << std::endl;
      }
      exit(0);
    }
  }
}

// generateKey is a function from a number to another given type, used to
// generate keys for insertion.
template <class T> T generateKey(size_t i) { return (T)i; }
// This specialization returns a stringified representation of the given
// integer, where the number is copied to the end of a long string of 'a's, in
// order to make comparisons and hashing take time.
template <> std::string generateKey<std::string>(size_t n) {
  const size_t min_length = 100;
  const std::string num(std::to_string(n));
  if (num.size() >= min_length) {
    return num;
  }
  std::string ret(min_length, 'a');
  const size_t startret = min_length - num.size();
  for (size_t i = 0; i < num.size(); i++) {
    ret[i + startret] = num[i];
  }
  return ret;
}

// An overloaded class that does the inserts for different table types. Inserts
// with a value of 0.
template <class Table> class insert_thread {
public:
  typedef typename std::vector<typename Table::key_type>::iterator it_t;
  static void func(Table &table, it_t begin, it_t end) {
    for (; begin != end; begin++) {
      ASSERT_TRUE(table.insert(*begin, 0));
    }
  }
};

// An overloaded class that does the reads for different table types. It
// repeatedly searches for the keys in the given range until the time is up. All
// the keys we're searching for should either be in the table or not in the
// table, so we assert that.
template <class Table> class read_thread {
public:
  typedef typename std::vector<typename Table::key_type>::iterator it_t;
  static void func(Table &table, it_t begin, it_t end,
                   std::atomic<size_t> &counter, bool in_table,
                   std::atomic<bool> &finished) {
    typename Table::mapped_type v;
    // We keep track of our own local counter for reads, to avoid
    // over-burdening the shared atomic counter
    size_t reads = 0;
    while (!finished.load(std::memory_order_acquire)) {
      for (auto it = begin; it != end; it++) {
        if (finished.load(std::memory_order_acquire)) {
          counter.fetch_add(reads);
          return;
        }
        ASSERT_EQ(in_table, table.find(*it, v));
        reads++;
      }
    }
  }
};

// An overloaded class that does a mixture of reads and inserts for different
// table types. It repeatedly searches for the keys in the given range until
// everything has been inserted.
template <class Table> class read_insert_thread {
public:
  typedef typename std::vector<typename Table::key_type>::iterator it_t;
  static void func(Table &table, it_t begin, it_t end,
                   std::atomic<size_t> &counter, const double insert_prob,
                   const size_t start_seed) {
    typename Table::mapped_type v;
    pcg64_fast gen(start_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto inserter_it = begin;
    auto reader_it = begin;
    size_t ops = 0;
    while (inserter_it != end) {
      if (dist(gen) < insert_prob) {
        // Do an insert
        ASSERT_TRUE(table.insert(*inserter_it, 0));
        ++inserter_it;
      } else {
        // Do a read
        ASSERT_EQ(table.find(*reader_it, v), (reader_it < inserter_it));
        ++reader_it;
        if (reader_it == end) {
          reader_it = begin;
        }
      }
      ++ops;
    }
    counter.fetch_add(ops);
  }
};

#endif // _TEST_UTIL_HH
