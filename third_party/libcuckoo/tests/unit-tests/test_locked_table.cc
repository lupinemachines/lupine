#include "test_locked_table.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

void test_locked_table_typedefs() {
  using Tbl = IntIntTable;
  using Ltbl = Tbl::locked_table;
  const bool key_type = std::is_same<Tbl::key_type, Ltbl::key_type>::value;
  const bool mapped_type =
      std::is_same<Tbl::mapped_type, Ltbl::mapped_type>::value;
  const bool value_type =
      std::is_same<Tbl::value_type, Ltbl::value_type>::value;
  const bool size_type = std::is_same<Tbl::size_type, Ltbl::size_type>::value;
  const bool difference_type =
      std::is_same<Tbl::difference_type, Ltbl::difference_type>::value;
  const bool hasher = std::is_same<Tbl::hasher, Ltbl::hasher>::value;
  const bool key_equal = std::is_same<Tbl::key_equal, Ltbl::key_equal>::value;
  const bool allocator_type =
      std::is_same<Tbl::allocator_type, Ltbl::allocator_type>::value;
  const bool reference = std::is_same<Tbl::reference, Ltbl::reference>::value;
  const bool const_reference =
      std::is_same<Tbl::const_reference, Ltbl::const_reference>::value;
  const bool pointer = std::is_same<Tbl::pointer, Ltbl::pointer>::value;
  const bool const_pointer =
      std::is_same<Tbl::const_pointer, Ltbl::const_pointer>::value;
  TEST_CHECK(key_type);
  TEST_CHECK(mapped_type);
  TEST_CHECK(value_type);
  TEST_CHECK(size_type);
  TEST_CHECK(difference_type);
  TEST_CHECK(hasher);
  TEST_CHECK(key_equal);
  TEST_CHECK(allocator_type);
  TEST_CHECK(reference);
  TEST_CHECK(const_reference);
  TEST_CHECK(pointer);
  TEST_CHECK(const_pointer);
}

void test_locked_table_move() {
  // move constructor
  {
    IntIntTable tbl;
    auto lt = tbl.lock_table();
    auto lt2(std::move(lt));
    TEST_CHECK(!lt.is_active());
    TEST_CHECK(lt2.is_active());
  }

  // move assignment
  {
    IntIntTable tbl;
    auto lt = tbl.lock_table();
    auto lt2 = std::move(lt);
    TEST_CHECK(!lt.is_active());
    TEST_CHECK(lt2.is_active());
  }

  // iterators compare after table is moved
  {
    IntIntTable tbl;
    auto lt1 = tbl.lock_table();
    auto it1 = lt1.begin();
    auto it2 = lt1.begin();
    TEST_CHECK(it1 == it2);
    auto lt2(std::move(lt1));
    TEST_CHECK(it1 == it2);
  }
}

void test_locked_table_unlock() {
  IntIntTable tbl;
  tbl.insert(10, 10);
  auto lt = tbl.lock_table();
  lt.unlock();
  TEST_CHECK(!lt.is_active());
}

void test_locked_table_info() {
  IntIntTable tbl;
  tbl.insert(10, 10);
  auto lt = tbl.lock_table();
  TEST_CHECK(lt.is_active());

  // We should still be able to call table info operations on the
  // cuckoohash_map instance, because they shouldn't take locks.

  TEST_CHECK(lt.slot_per_bucket() == tbl.slot_per_bucket());
  TEST_CHECK(lt.get_allocator() == tbl.get_allocator());
  TEST_CHECK(lt.hashpower() == tbl.hashpower());
  TEST_CHECK(lt.bucket_count() == tbl.bucket_count());
  TEST_CHECK(lt.empty() == tbl.empty());
  TEST_CHECK(lt.size() == tbl.size());
  TEST_CHECK(lt.capacity() == tbl.capacity());
  TEST_CHECK(lt.load_factor() == tbl.load_factor());
  TEST_EXCEPTION(lt.minimum_load_factor(1.01), std::invalid_argument);
  lt.minimum_load_factor(lt.minimum_load_factor() * 2);
  lt.rehash(5);
  TEST_EXCEPTION(lt.maximum_hashpower(lt.hashpower() - 1),
                 std::invalid_argument);
  lt.maximum_hashpower(lt.hashpower() + 1);
  TEST_CHECK(lt.maximum_hashpower() == tbl.maximum_hashpower());
}

void test_locked_table_clear() {
  IntIntTable tbl;
  tbl.insert(10, 10);
  auto lt = tbl.lock_table();
  TEST_CHECK(lt.size() == 1);
  lt.clear();
  TEST_CHECK(lt.size() == 0);
  lt.clear();
  TEST_CHECK(lt.size() == 0);
}

void test_locked_table_insert_duplicate() {
  IntIntTable tbl;
  tbl.insert(10, 10);
  {
    auto lt = tbl.lock_table();
    auto result = lt.insert(10, 20);
    TEST_CHECK(result.first->first == 10);
    TEST_CHECK(result.first->second == 10);
    TEST_CHECK(!result.second);
    result.first->second = 50;
  }
  TEST_CHECK(tbl.find(10) == 50);
}

void test_locked_table_insert_new_key() {
  IntIntTable tbl;
  tbl.insert(10, 10);
  {
    auto lt = tbl.lock_table();
    auto result = lt.insert(20, 20);
    TEST_CHECK(result.first->first == 20);
    TEST_CHECK(result.first->second == 20);
    TEST_CHECK(result.second);
    result.first->second = 50;
  }
  TEST_CHECK(tbl.find(10) == 10);
  TEST_CHECK(tbl.find(20) == 50);
}

void test_locked_table_insert_lifetime() {
  // Successful insert
  {
    UniquePtrTable<int> tbl;
    auto lt = tbl.lock_table();
    std::unique_ptr<int> key(new int(20));
    std::unique_ptr<int> value(new int(20));
    auto result = lt.insert(std::move(key), std::move(value));
    TEST_CHECK(*result.first->first == 20);
    TEST_CHECK(*result.first->second == 20);
    TEST_CHECK(result.second);
    TEST_CHECK(!static_cast<bool>(key));
    TEST_CHECK(!static_cast<bool>(value));
  }

  // Unsuccessful insert
  {
    UniquePtrTable<int> tbl;
    tbl.insert(new int(20), new int(20));
    auto lt = tbl.lock_table();
    std::unique_ptr<int> key(new int(20));
    std::unique_ptr<int> value(new int(30));
    auto result = lt.insert(std::move(key), std::move(value));
    TEST_CHECK(*result.first->first == 20);
    TEST_CHECK(*result.first->second == 20);
    TEST_CHECK(!result.second);
    TEST_CHECK(static_cast<bool>(key));
    TEST_CHECK(static_cast<bool>(value));
  }
}

struct EraseFixture {
  IntIntTable tbl;
};

IntIntTable erase_table() {
  IntIntTable tbl;
  for (int i = 0; i < 5; ++i) {
    tbl.insert(i, i);
  }
  return tbl;
}

void test_locked_table_erase() {
  using lt_t = IntIntTable::locked_table;

  // simple erase
  {
    auto tbl = erase_table();
    auto lt = tbl.lock_table();
    lt_t::const_iterator const_it;
    const_it = lt.find(0);
    TEST_CHECK(const_it != lt.end());
    lt_t::const_iterator const_next = const_it;
    ++const_next;
    TEST_CHECK(static_cast<lt_t::const_iterator>(lt.erase(const_it)) ==
               const_next);
    TEST_CHECK(lt.size() == 4);

    lt_t::iterator it;
    it = lt.find(1);
    lt_t::iterator next = it;
    ++next;
    TEST_CHECK(lt.erase(static_cast<lt_t::const_iterator>(it)) == next);
    TEST_CHECK(lt.size() == 3);

    TEST_CHECK(lt.erase(2) == 1);
    TEST_CHECK(lt.size() == 2);
  }

  // erase doesn't ruin this iterator
  {
    auto tbl = erase_table();
    auto lt = tbl.lock_table();
    auto it = lt.begin();
    auto next = it;
    ++next;
    TEST_CHECK(lt.erase(it) == next);
    ++it;
    TEST_CHECK(it->first > 0);
    TEST_CHECK(it->first < 5);
    TEST_CHECK(it->second > 0);
    TEST_CHECK(it->second < 5);
  }

  // erase doesn't ruin other iterators
  {
    auto tbl = erase_table();
    auto lt = tbl.lock_table();
    auto it0 = lt.find(0);
    auto it1 = lt.find(1);
    auto it2 = lt.find(2);
    auto it3 = lt.find(3);
    auto it4 = lt.find(4);
    auto next = it2;
    ++next;
    TEST_CHECK(lt.erase(it2) == next);
    TEST_CHECK(it0->first == 0);
    TEST_CHECK(it0->second == 0);
    TEST_CHECK(it1->first == 1);
    TEST_CHECK(it1->second == 1);
    TEST_CHECK(it3->first == 3);
    TEST_CHECK(it3->second == 3);
    TEST_CHECK(it4->first == 4);
    TEST_CHECK(it4->second == 4);
  }
}

void test_locked_table_find() {
  IntIntTable tbl;
  using lt_t = IntIntTable::locked_table;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.insert(i, i).second);
  }
  bool found_begin_elem = false;
  bool found_last_elem = false;
  for (int i = 0; i < 10; ++i) {
    lt_t::iterator it = lt.find(i);
    lt_t::const_iterator const_it = lt.find(i);
    TEST_CHECK(it != lt.end());
    TEST_CHECK(it->first == i);
    TEST_CHECK(it->second == i);
    TEST_CHECK(const_it != lt.end());
    TEST_CHECK(const_it->first == i);
    TEST_CHECK(const_it->second == i);
    it->second++;
    if (it == lt.begin()) {
      found_begin_elem = true;
    }
    if (++it == lt.end()) {
      found_last_elem = true;
    }
  }
  TEST_CHECK(found_begin_elem);
  TEST_CHECK(found_last_elem);
  for (int i = 0; i < 10; ++i) {
    lt_t::iterator it = lt.find(i);
    TEST_CHECK(it->first == i);
    TEST_CHECK(it->second == i + 1);
  }
}

void test_locked_table_at() {
  IntIntTable tbl;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    int &val = lt.at(i);
    const int &const_val =
        const_cast<const IntIntTable::locked_table &>(lt).at(i);
    TEST_CHECK(val == i);
    TEST_CHECK(const_val == i);
    ++val;
  }
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.at(i) == i + 1);
  }
  TEST_EXCEPTION(lt.at(11), std::out_of_range);
}

void test_locked_table_operator_brackets() {
  IntIntTable tbl;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    int &val = lt[i];
    TEST_CHECK(val == i);
    ++val;
  }
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt[i] == i + 1);
  }
  TEST_CHECK(lt[11] == 0);
  TEST_CHECK(lt.at(11) == 0);
}

void test_locked_table_count() {
  IntIntTable tbl;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.count(i) == 1);
  }
  TEST_CHECK(lt.count(11) == 0);
}

void test_locked_table_equal_range() {
  IntIntTable tbl;
  using lt_t = IntIntTable::locked_table;
  auto lt = tbl.lock_table();
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(lt.insert(i, i).second);
  }
  for (int i = 0; i < 10; ++i) {
    std::pair<lt_t::iterator, lt_t::iterator> it_range = lt.equal_range(i);
    TEST_CHECK(it_range.first->first == i);
    TEST_CHECK(++it_range.first == it_range.second);
    std::pair<lt_t::const_iterator, lt_t::const_iterator> const_it_range =
        lt.equal_range(i);
    TEST_CHECK(const_it_range.first->first == i);
    TEST_CHECK(++const_it_range.first == const_it_range.second);
  }
  auto it_range = lt.equal_range(11);
  TEST_CHECK(it_range.first == lt.end());
  TEST_CHECK(it_range.second == lt.end());
}

void test_locked_table_rehash() {
  IntIntTable tbl(10);
  auto lt = tbl.lock_table();
  TEST_CHECK(lt.hashpower() == 2);
  lt.rehash(1);
  TEST_CHECK(lt.hashpower() == 1);
  lt.rehash(10);
  TEST_CHECK(lt.hashpower() == 10);
}

void test_locked_table_reserve() {
  IntIntTable tbl(10);
  auto lt = tbl.lock_table();
  TEST_CHECK(lt.hashpower() == 2);
  lt.reserve(1);
  TEST_CHECK(lt.hashpower() == 0);
  lt.reserve(4096);
  TEST_CHECK(lt.hashpower() == 10);
}

void test_locked_table_equality() {
  IntIntTable tbl1(40);
  auto lt1 = tbl1.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt1.insert(i, i);
  }

  IntIntTable tbl2(30);
  auto lt2 = tbl2.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt2.insert(i, i);
  }

  IntIntTable tbl3(30);
  auto lt3 = tbl3.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt3.insert(i, i + 1);
  }

  IntIntTable tbl4(40);
  auto lt4 = tbl4.lock_table();
  for (int i = 0; i < 10; ++i) {
    lt4.insert(i + 1, i);
  }

  TEST_CHECK(lt1 == lt2);
  TEST_CHECK(!(lt2 != lt1));

  TEST_CHECK(lt1 != lt3);
  TEST_CHECK(!(lt3 == lt1));
  TEST_CHECK(!(lt2 == lt3));
  TEST_CHECK(lt3 != lt2);

  TEST_CHECK(lt1 != lt4);
  TEST_CHECK(lt4 != lt1);
  TEST_CHECK(!(lt3 == lt4));
  TEST_CHECK(!(lt4 == lt3));
}

template <typename Table> void check_all_locks_taken(Table &tbl) {
  auto &locks = libcuckoo::UnitTestInternalAccess::get_current_locks(tbl);
  for (auto &lock : locks) {
    TEST_CHECK(!lock.try_lock());
  }
}

void test_locked_table_holds_locks_after_resize() {
  IntIntTable tbl(4);
  auto lt = tbl.lock_table();
  check_all_locks_taken(tbl);

  // After a cuckoo_fast_double, all locks are still taken
  for (int i = 0; i < 5; ++i) {
    lt.insert(i, i);
  }
  check_all_locks_taken(tbl);

  // After a cuckoo_simple_expand, all locks are still taken
  lt.rehash(10);
  check_all_locks_taken(tbl);
}

void test_locked_table_io() {
  IntIntTable tbl(0);
  auto lt = tbl.lock_table();
  for (int i = 0; i < 100; ++i) {
    lt.insert(i, i);
  }

  std::stringstream sstream;
  sstream << lt;

  IntIntTable tbl2;
  auto lt2 = tbl2.lock_table();
  sstream.seekg(0);
  sstream >> lt2;

  TEST_CHECK(100 == lt.size());
  for (int i = 0; i < 100; ++i) {
    TEST_CHECK(i == lt.at(i));
  }

  TEST_CHECK(100 == lt2.size());
  for (int i = 100; i < 1000; ++i) {
    lt2.insert(i, i);
  }
  for (int i = 0; i < 1000; ++i) {
    TEST_CHECK(i == lt2.at(i));
  }
}

void test_empty_locked_table_io() {
  IntIntTable tbl(0);
  auto lt = tbl.lock_table();
  lt.minimum_load_factor(0.5);
  lt.maximum_hashpower(10);

  std::stringstream sstream;
  sstream << lt;

  IntIntTable tbl2(0);
  auto lt2 = tbl2.lock_table();
  sstream.seekg(0);
  sstream >> lt2;

  TEST_CHECK(0 == lt.size());
  TEST_CHECK(0 == lt2.size());
  TEST_CHECK(0.5 == lt.minimum_load_factor());
  TEST_CHECK(10 == lt.maximum_hashpower());
}
