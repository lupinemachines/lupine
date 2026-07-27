// Tests concurrent inserts, deletes, updates, and finds. The test makes sure
// that multiple operations are not run on the same key, so that the accuracy of
// the operations can be verified.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <libcuckoo/cuckoohash_map.hh>
#include <pcg/pcg_random.hpp>
#include <test_util.hh>

typedef uint32_t KeyType;
typedef std::string KeyType2;
typedef uint32_t ValType;
typedef int32_t ValType2;

// The number of keys to size the table with, expressed as a power of
// 2. This can be set with the command line flag --power
size_t g_power = 25;
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
// How many seconds to run the test for. This can be set with the
// command line flag --time
size_t g_test_len = 10;
// The seed for the random number generator. If this isn't set to a
// nonzero value with the --seed flag, the current time is used
size_t g_seed = 0;
// Whether to use strings as the key
bool g_use_strings = false;

std::atomic<size_t> num_inserts = ATOMIC_VAR_INIT(0);
std::atomic<size_t> num_deletes = ATOMIC_VAR_INIT(0);
std::atomic<size_t> num_updates = ATOMIC_VAR_INIT(0);
std::atomic<size_t> num_finds = ATOMIC_VAR_INIT(0);

template <class KType> class AllEnvironment {
public:
  AllEnvironment()
      : table(g_numkeys), table2(g_numkeys), keys(g_numkeys), vals(g_numkeys),
        vals2(g_numkeys), in_table(new bool[g_numkeys]),
        in_use(new std::atomic_flag[g_numkeys]),
        val_dist(std::numeric_limits<ValType>::min(),
                 std::numeric_limits<ValType>::max()),
        val_dist2(std::numeric_limits<ValType2>::min(),
                  std::numeric_limits<ValType2>::max()),
        ind_dist(0, g_numkeys - 1), finished(false) {
    // Sets up the random number generator
    if (g_seed == 0) {
      g_seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    std::cout << "seed = " << g_seed << std::endl;
    gen_seed = g_seed;
    // Fills in all the vectors except vals, which will be filled
    // in by the insertion threads.
    for (size_t i = 0; i < g_numkeys; i++) {
      keys[i] = generateKey<KType>(i);
      in_table[i] = false;
      in_use[i].clear();
    }
  }

  libcuckoo::cuckoohash_map<KType, ValType> table;
  libcuckoo::cuckoohash_map<KType, ValType2> table2;
  std::vector<KType> keys;
  std::vector<ValType> vals;
  std::vector<ValType2> vals2;
  std::unique_ptr<bool[]> in_table;
  std::unique_ptr<std::atomic_flag[]> in_use;
  std::uniform_int_distribution<ValType> val_dist;
  std::uniform_int_distribution<ValType2> val_dist2;
  std::uniform_int_distribution<size_t> ind_dist;
  size_t gen_seed;
  // When set to true, it signals to the threads to stop running
  std::atomic<bool> finished;
};

template <class KType> void stress_insert_thread(AllEnvironment<KType> *env) {
  pcg64_fast gen(env->gen_seed);
  while (!env->finished.load()) {
    // Pick a random number between 0 and g_numkeys. If that slot is
    // not in use, lock the slot. Insert a random value into both
    // tables. The inserts should only be successful if the key
    // wasn't in the table. If the inserts succeeded, check that
    // the insertion were actually successful with another find
    // operation, and then store the values in their arrays and
    // set in_table to true and clear in_use
    size_t ind = env->ind_dist(gen);
    if (!env->in_use[ind].test_and_set()) {
      KType k = env->keys[ind];
      ValType v = env->val_dist(gen);
      ValType2 v2 = env->val_dist2(gen);
      bool res = env->table.insert(k, v);
      bool res2 = env->table2.insert(k, v2);
      EXPECT_NE(res, env->in_table[ind]);
      EXPECT_NE(res2, env->in_table[ind]);
      if (res) {
        EXPECT_EQ(v, env->table.find(k));
        EXPECT_EQ(v2, env->table2.find(k));
        env->vals[ind] = v;
        env->vals2[ind] = v2;
        env->in_table[ind] = true;
        num_inserts.fetch_add(2, std::memory_order_relaxed);
      }
      env->in_use[ind].clear();
    }
  }
}

template <class KType> void delete_thread(AllEnvironment<KType> *env) {
  pcg64_fast gen(env->gen_seed);
  while (!env->finished.load()) {
    // Run deletes on a random key, check that the deletes
    // succeeded only if the keys were in the table. If the
    // deletes succeeded, check that the keys are indeed not in
    // the tables anymore, and then set in_table to false
    size_t ind = env->ind_dist(gen);
    if (!env->in_use[ind].test_and_set()) {
      KType k = env->keys[ind];
      bool res = env->table.erase(k);
      bool res2 = env->table2.erase(k);
      EXPECT_EQ(res, env->in_table[ind]);
      EXPECT_EQ(res2, env->in_table[ind]);
      if (res) {
        ValType find_v = 0;
        ValType2 find_v2 = 0;
        EXPECT_FALSE(env->table.find(k, find_v));
        EXPECT_FALSE(env->table2.find(k, find_v2));
        env->in_table[ind] = false;
        num_deletes.fetch_add(2, std::memory_order_relaxed);
      }
      env->in_use[ind].clear();
    }
  }
}

template <class KType> void update_thread(AllEnvironment<KType> *env) {
  pcg64_fast gen(env->gen_seed);
  std::uniform_int_distribution<size_t> third(0, 2);
  auto updatefn = [](ValType &v) { v += 3; };
  auto updatefn2 = [](ValType2 &v) { v += 10; };
  while (!env->finished.load()) {
    // Run updates, update_fns, or upserts on a random key, check
    // that the operations succeeded only if the keys were in the
    // table (or that they succeeded regardless if it's an
    // upsert). If successful, check that the keys are indeed in
    // the table with the new value, and then set in_table to true
    size_t ind = env->ind_dist(gen);
    if (!env->in_use[ind].test_and_set()) {
      KType k = env->keys[ind];
      ValType v;
      ValType2 v2;
      bool res, res2;
      switch (third(gen)) {
      case 0:
        // update
        v = env->val_dist(gen);
        v2 = env->val_dist2(gen);
        res = env->table.update(k, v);
        res2 = env->table2.update(k, v2);
        EXPECT_EQ(res, env->in_table[ind]);
        EXPECT_EQ(res2, env->in_table[ind]);
        break;
      case 1:
        // update_fn
        v = env->vals[ind];
        v2 = env->vals2[ind];
        updatefn(v);
        updatefn2(v2);
        res = env->table.update_fn(k, updatefn);
        res2 = env->table2.update_fn(k, updatefn2);
        EXPECT_EQ(res, env->in_table[ind]);
        EXPECT_EQ(res2, env->in_table[ind]);
        break;
      case 2:
        // upsert
        if (env->in_table[ind]) {
          // Then it should run updatefn
          v = env->vals[ind];
          v2 = env->vals2[ind];
          updatefn(v);
          updatefn2(v2);
        } else {
          // Then it should run an insert
          v = env->val_dist(gen);
          v2 = env->val_dist2(gen);
        }
        // These upserts should always succeed, so set res and res2 to
        // true.
        env->table.upsert(k, updatefn, v);
        env->table2.upsert(k, updatefn2, v2);
        res = res2 = true;
        env->in_table[ind] = true;
        break;
      default:
        throw std::logic_error("Impossible");
      }
      if (res) {
        EXPECT_EQ(v, env->table.find(k));
        EXPECT_EQ(v2, env->table2.find(k));
        env->vals[ind] = v;
        env->vals2[ind] = v2;
        num_updates.fetch_add(2, std::memory_order_relaxed);
      }
      env->in_use[ind].clear();
    }
  }
}

template <class KType> void find_thread(AllEnvironment<KType> *env) {
  pcg64_fast gen(env->gen_seed);
  while (!env->finished.load()) {
    // Run finds on a random key and check that the presence of
    // the keys matches in_table
    size_t ind = env->ind_dist(gen);
    if (!env->in_use[ind].test_and_set()) {
      KType k = env->keys[ind];
      try {
        EXPECT_EQ(env->vals[ind], env->table.find(k));
        EXPECT_TRUE(env->in_table[ind]);
      } catch (const std::out_of_range &) {
        EXPECT_FALSE(env->in_table[ind]);
      }
      try {
        EXPECT_EQ(env->vals2[ind], env->table2.find(k));
        EXPECT_TRUE(env->in_table[ind]);
      } catch (const std::out_of_range &) {
        EXPECT_FALSE(env->in_table[ind]);
      }
      num_finds.fetch_add(2, std::memory_order_relaxed);
      env->in_use[ind].clear();
    }
  }
}

// Spawns g_thread_num insert, delete, update, and find threads
template <class KType> void StressTest(AllEnvironment<KType> *env) {
  std::vector<std::thread> threads;
  for (size_t i = 0; i < g_thread_num; i++) {
    if (!g_disable_inserts) {
      threads.emplace_back(stress_insert_thread<KType>, env);
    }
    if (!g_disable_deletes) {
      threads.emplace_back(delete_thread<KType>, env);
    }
    if (!g_disable_updates) {
      threads.emplace_back(update_thread<KType>, env);
    }
    if (!g_disable_finds) {
      threads.emplace_back(find_thread<KType>, env);
    }
  }
  // Sleeps before ending the threads
  std::this_thread::sleep_for(std::chrono::seconds(g_test_len));
  env->finished.store(true);
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i].join();
  }
  // Finds the number of slots that are filled
  size_t numfilled = 0;
  for (size_t i = 0; i < g_numkeys; i++) {
    if (env->in_table[i]) {
      numfilled++;
    }
  }
  EXPECT_EQ(numfilled, env->table.size());
  std::cout << "----------Results----------" << std::endl;
  std::cout << "Number of inserts:\t" << num_inserts.load() << std::endl;
  std::cout << "Number of deletes:\t" << num_deletes.load() << std::endl;
  std::cout << "Number of updates:\t" << num_updates.load() << std::endl;
  std::cout << "Number of finds:\t" << num_finds.load() << std::endl;
}

int main(int argc, char **argv) {
  const char *args[] = {"--power", "--thread-num", "--time", "--seed"};
  size_t *arg_vars[] = {&g_power, &g_thread_num, &g_test_len, &g_seed};
  const char *arg_help[] = {
      "The number of keys to size the table with, expressed as a power of 2",
      "The number of threads to spawn for each type of operation",
      "The number of seconds to run the test for",
      "The seed for the random number generator"};
  const char *flags[] = {"--disable-inserts", "--disable-deletes",
                         "--disable-updates", "--disable-finds",
                         "--use-strings"};
  bool *flag_vars[] = {&g_disable_inserts, &g_disable_deletes,
                       &g_disable_updates, &g_disable_finds, &g_use_strings};
  const char *flag_help[] = {
      "If set, no inserts will be run", "If set, no deletes will be run",
      "If set, no updates will be run", "If set, no finds will be run",
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
