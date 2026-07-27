// Tests all operations and iterators concurrently. It doesn't check any
// operation for correctness, only making sure that everything completes without
// crashing.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdint.h>
#include <thread>
#include <utility>
#include <vector>

#include <libcuckoo/cuckoohash_config.hh>
#include <libcuckoo/cuckoohash_map.hh>
#include <pcg/pcg_random.hpp>
#include <test_util.hh>

typedef uint32_t KeyType;
typedef std::string KeyType2;
typedef uint32_t ValType;
typedef int32_t ValType2;

// The number of keys to size the table with, expressed as a power of
// 2. This can be set with the command line flag --power
size_t g_power = 24;
size_t g_numkeys; // Holds 2^power
// The number of threads spawned for each type of operation. This can
// be set with the command line flag --thread-num
size_t g_thread_num = 4;
// Whether to disable inserts or not. This can be set with the command
// line flag --disable-inserts
bool g_disable_inserts = false;
// Whether to disable deletes or not. This can be set with the command
// line flag --disable-deletes
bool g_disable_deletes = false;
// Whether to disable updates or not. This can be set with the command
// line flag --disable-updates
bool g_disable_updates = false;
// Whether to disable finds or not. This can be set with the command
// line flag --disable-finds
bool g_disable_finds = false;
// Whether to disable resizes operations or not. This can be set with
// the command line flag --disable-resizes
bool g_disable_resizes = false;
// Whether to disable iterator operations or not. This can be set with
// the command line flag --disable-iterators
bool g_disable_iterators = false;
// Whether to disable statistic operations or not. This can be set with
// the command line flag --disable-misc
bool g_disable_misc = false;
// Whether to disable clear operations or not. This can be set with
// the command line flag --disable-clears
bool g_disable_clears = false;
// How many seconds to run the test for. This can be set with the
// command line flag --time
size_t g_test_len = 10;
// The seed for the random number generator. If this isn't set to a
// nonzero value with the --seed flag, the current time is used
size_t g_seed = 0;
// Whether to use strings as the key
bool g_use_strings = false;

template <class KType> class AllEnvironment {
public:
  AllEnvironment() : table(g_numkeys), table2(g_numkeys), finished(false) {
    // Sets up the random number generator
    if (g_seed == 0) {
      g_seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    std::cout << "seed = " << g_seed << std::endl;
    gen_seed = g_seed;
    // Set minimum load factor and maximum hashpower to unbounded, so weird
    // operations don't throw exceptions.
    table.minimum_load_factor(0.0);
    table2.minimum_load_factor(0.0);
    table.maximum_hashpower(libcuckoo::NO_MAXIMUM_HASHPOWER);
    table2.maximum_hashpower(libcuckoo::NO_MAXIMUM_HASHPOWER);
  }

  libcuckoo::cuckoohash_map<KType, ValType> table;
  libcuckoo::cuckoohash_map<KType, ValType2> table2;
  size_t gen_seed;
  std::atomic<bool> finished;
};

template <class KType>
void stress_insert_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  std::uniform_int_distribution<size_t> ind_dist;
  std::uniform_int_distribution<ValType> val_dist;
  std::uniform_int_distribution<ValType2> val_dist2;
  pcg64_fast gen(thread_seed);
  while (!env->finished.load()) {
    // Insert a random key into the table
    KType k = generateKey<KType>(ind_dist(gen));
    env->table.insert(k, val_dist(gen));
    env->table2.insert(k, val_dist2(gen));
    env->table.insert_or_assign(k, val_dist(gen));
    env->table2.insert_or_assign(k, val_dist2(gen));
  }
}

template <class KType>
void delete_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  std::uniform_int_distribution<size_t> ind_dist;
  pcg64_fast gen(thread_seed);
  while (!env->finished.load()) {
    // Run deletes on a random key.
    const KType k = generateKey<KType>(ind_dist(gen));
    env->table.erase(k);
    env->table2.erase(k);
  }
}

template <class KType>
void update_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  std::uniform_int_distribution<size_t> ind_dist;
  std::uniform_int_distribution<ValType> val_dist;
  std::uniform_int_distribution<ValType2> val_dist2;
  std::uniform_int_distribution<size_t> third(0, 2);
  pcg64_fast gen(thread_seed);
  auto updatefn = [](ValType &v) { v += 3; };
  while (!env->finished.load()) {
    // Run updates, update_funcs, or upserts on a random key.
    const KType k = generateKey<KType>(ind_dist(gen));
    switch (third(gen)) {
    case 0:
      // update
      env->table.update(k, val_dist(gen));
      env->table2.update(k, val_dist2(gen));
      break;
    case 1:
      // update_fn
      env->table.update_fn(k, updatefn);
      env->table2.update_fn(k, [](ValType2 &v) { v += 10; });
      break;
    case 2:
      env->table.upsert(k, updatefn, val_dist(gen));
      env->table2.upsert(k, [](ValType2 &v) { v -= 50; }, val_dist2(gen));
    }
  }
}

template <class KType>
void find_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  std::uniform_int_distribution<size_t> ind_dist;
  pcg64_fast gen(thread_seed);
  ValType v;
  while (!env->finished.load()) {
    // Run finds on a random key.
    const KType k = generateKey<KType>(ind_dist(gen));
    env->table.find(k, v);
    try {
      env->table2.find(k);
    } catch (...) {
    }
  }
}

template <class KType>
void resize_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  pcg64_fast gen(thread_seed);
  // Resizes at a random time
  const size_t sleep_time = gen() % g_test_len;
  std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
  if (env->finished.load()) {
    return;
  }
  const size_t hashpower = env->table2.hashpower();
  if (gen() & 1) {
    env->table.rehash(hashpower + 1);
    env->table.rehash(hashpower / 2);
  } else {
    env->table2.reserve((1U << (hashpower + 1)) *
                        libcuckoo::DEFAULT_SLOT_PER_BUCKET);
    env->table2.reserve((1U << hashpower) * libcuckoo::DEFAULT_SLOT_PER_BUCKET);
  }
}

template <class KType>
void iterator_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  pcg64_fast gen(thread_seed);
  // Runs an iteration operation at a random time
  const size_t sleep_time = gen() % g_test_len;
  std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
  if (env->finished.load()) {
    return;
  }
  auto lt = env->table2.lock_table();
  for (auto &item : lt) {
    if (gen() & 1) {
      item.second++;
    }
  }
}

template <class KType> void misc_thread(AllEnvironment<KType> *env) {
  // Runs all the misc functions
  pcg64_fast gen(g_seed);
  while (!env->finished.load()) {
    env->table.slot_per_bucket();
    env->table.size();
    env->table.empty();
    env->table.bucket_count();
    env->table.load_factor();
    env->table.hash_function();
    env->table.key_eq();
  }
}

template <class KType>
void clear_thread(AllEnvironment<KType> *env, size_t thread_seed) {
  pcg64_fast gen(thread_seed);
  // Runs a clear operation at a random time
  const size_t sleep_time = gen() % g_test_len;
  std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
  if (env->finished.load()) {
    return;
  }
  env->table.clear();
}

// Spawns thread_num threads for each type of operation
template <class KType> void StressTest(AllEnvironment<KType> *env) {
  std::vector<std::thread> threads;
  for (size_t i = 0; i < g_thread_num; i++) {
    if (!g_disable_inserts) {
      threads.emplace_back(stress_insert_thread<KType>, env, env->gen_seed++);
    }
    if (!g_disable_deletes) {
      threads.emplace_back(delete_thread<KType>, env, env->gen_seed++);
    }
    if (!g_disable_updates) {
      threads.emplace_back(update_thread<KType>, env, env->gen_seed++);
    }
    if (!g_disable_finds) {
      threads.emplace_back(find_thread<KType>, env, env->gen_seed++);
    }
    if (!g_disable_resizes) {
      threads.emplace_back(resize_thread<KType>, env, env->gen_seed++);
    }
    if (!g_disable_iterators) {
      threads.emplace_back(iterator_thread<KType>, env, env->gen_seed++);
    }
    if (!g_disable_misc) {
      threads.emplace_back(misc_thread<KType>, env);
    }
    if (!g_disable_clears) {
      threads.emplace_back(clear_thread<KType>, env, env->gen_seed++);
    }
  }
  // Sleeps before ending the threads
  std::this_thread::sleep_for(std::chrono::seconds(g_test_len));
  env->finished.store(true);
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  std::cout << "----------Results----------" << std::endl;
  std::cout << "Final size:\t" << env->table.size() << std::endl;
  std::cout << "Final load factor:\t" << env->table.load_factor() << std::endl;
}

int main(int argc, char **argv) {
  const char *args[] = {"--power", "--thread-num", "--time", "--seed"};
  size_t *arg_vars[] = {&g_power, &g_thread_num, &g_test_len, &g_seed};
  const char *arg_help[] = {
      "The number of keys to size the table with, expressed as a power of 2",
      "The number of threads to spawn for each type of operation",
      "The number of seconds to run the test for",
      "The seed for the random number generator"};
  const char *flags[] = {
      "--disable-inserts", "--disable-deletes", "--disable-updates",
      "--disable-finds",   "--disable-resizes", "--disable-iterators",
      "--disable-misc",    "--disable-clears",  "--use-strings"};
  bool *flag_vars[] = {
      &g_disable_inserts, &g_disable_deletes, &g_disable_updates,
      &g_disable_finds,   &g_disable_resizes, &g_disable_iterators,
      &g_disable_misc,    &g_disable_clears,  &g_use_strings};
  const char *flag_help[] = {
      "If set, no inserts will be run",
      "If set, no deletes will be run",
      "If set, no updates will be run",
      "If set, no finds will be run",
      "If set, no resize operations will be run",
      "If set, no iterator operations will be run",
      "If set, no misc functions will be run",
      "If set, no clears will be run",
      "If set, the key type of the map will be std::string"};
  parse_flags(argc, argv, "Runs a stress test on inserts, deletes, and finds",
              args, arg_vars, arg_help, sizeof(args) / sizeof(const char *),
              flags, flag_vars, flag_help,
              sizeof(flags) / sizeof(const char *));
  g_numkeys = 1U << g_power;

  if (g_use_strings) {
    auto *env = new AllEnvironment<KeyType2>;
    StressTest(env);
    delete env;
  } else {
    auto *env = new AllEnvironment<KeyType>;
    StressTest(env);
    delete env;
  }
  return main_return_value;
}
