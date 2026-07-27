#include "test_iterator.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

void test_iterator_types() {
  using Ltbl = IntIntTable::locked_table;
  using It = Ltbl::iterator;
  using ConstIt = Ltbl::const_iterator;

  const bool it_difference_type =
      std::is_same<Ltbl::difference_type, It::difference_type>::value;
  const bool it_value_type =
      std::is_same<Ltbl::value_type, It::value_type>::value;
  const bool it_pointer = std::is_same<Ltbl::pointer, It::pointer>::value;
  const bool it_reference = std::is_same<Ltbl::reference, It::reference>::value;
  const bool it_iterator_category =
      std::is_same<std::bidirectional_iterator_tag,
                   It::iterator_category>::value;

  const bool const_it_difference_type =
      std::is_same<Ltbl::difference_type, ConstIt::difference_type>::value;
  const bool const_it_value_type =
      std::is_same<Ltbl::value_type, ConstIt::value_type>::value;
  const bool const_it_reference =
      std::is_same<Ltbl::const_reference, ConstIt::reference>::value;
  const bool const_it_pointer =
      std::is_same<Ltbl::const_pointer, ConstIt::pointer>::value;
  const bool const_it_iterator_category =
      std::is_same<std::bidirectional_iterator_tag,
                   ConstIt::iterator_category>::value;

  TEST_CHECK(it_difference_type);
  TEST_CHECK(it_value_type);
  TEST_CHECK(it_pointer);
  TEST_CHECK(it_reference);
  TEST_CHECK(it_iterator_category);

  TEST_CHECK(const_it_difference_type);
  TEST_CHECK(const_it_value_type);
  TEST_CHECK(const_it_pointer);
  TEST_CHECK(const_it_reference);
  TEST_CHECK(const_it_iterator_category);
}

void test_iterator_empty_table_iteration() {
  IntIntTable table;
  {
    auto lt = table.lock_table();
    TEST_CHECK(lt.begin() == lt.begin());
    TEST_CHECK(lt.begin() == lt.end());

    TEST_CHECK(lt.cbegin() == lt.begin());
    TEST_CHECK(lt.begin() == lt.end());

    TEST_CHECK(lt.cbegin() == lt.begin());
    TEST_CHECK(lt.cend() == lt.end());
  }
}

IntIntTable walkthrough_table() {
  IntIntTable table;
  for (int i = 0; i < 10; ++i) {
    table.insert(i, i);
  }
  return table;
}

void test_iterator_walkthrough() {
  // forward postfix walkthrough
  {
    auto table = walkthrough_table();

    auto lt = table.lock_table();
    auto it = lt.cbegin();
    for (size_t i = 0; i < table.size(); ++i) {
      TEST_CHECK((*it).first == (*it).second);
      TEST_CHECK(it->first == it->second);
      auto old_it = it;
      TEST_CHECK(old_it == it++);
    }
    TEST_CHECK(it == lt.end());
  }

  // forward prefix walkthrough
  {
    auto table = walkthrough_table();

    auto lt = table.lock_table();
    auto it = lt.cbegin();
    for (size_t i = 0; i < table.size(); ++i) {
      TEST_CHECK((*it).first == (*it).second);
      TEST_CHECK(it->first == it->second);
      ++it;
    }
    TEST_CHECK(it == lt.end());
  }

  // backwards postfix walkthrough
  {
    auto table = walkthrough_table();

    auto lt = table.lock_table();
    auto it = lt.cend();
    for (size_t i = 0; i < table.size(); ++i) {
      auto old_it = it;
      TEST_CHECK(old_it == it--);
      TEST_CHECK((*it).first == (*it).second);
      TEST_CHECK(it->first == it->second);
    }
    TEST_CHECK(it == lt.begin());
  }

  // backwards prefix walkthrough
  {
    auto table = walkthrough_table();

    auto lt = table.lock_table();
    auto it = lt.cend();
    for (size_t i = 0; i < table.size(); ++i) {
      --it;
      TEST_CHECK((*it).first == (*it).second);
      TEST_CHECK(it->first == it->second);
    }
    TEST_CHECK(it == lt.begin());
  }

  // walkthrough works after move
  {
    auto table = walkthrough_table();

    auto lt = table.lock_table();
    auto it = lt.cend();
    auto lt2 = std::move(lt);
    for (size_t i = 0; i < table.size(); ++i) {
      --it;
      TEST_CHECK((*it).first == (*it).second);
      TEST_CHECK(it->first == it->second);
    }
    TEST_CHECK(it == lt2.begin());
  }
}

void test_iterator_modification() {
  IntIntTable table;
  for (int i = 0; i < 10; ++i) {
    table.insert(i, i);
  }

  auto lt = table.lock_table();
  for (auto it = lt.begin(); it != lt.end(); ++it) {
    it->second = it->second + 1;
  }

  auto it = lt.cbegin();
  for (size_t i = 0; i < table.size(); ++i) {
    TEST_CHECK(it->first == it->second - 1);
    ++it;
  }
  TEST_CHECK(it == lt.end());
}

void test_iterator_lock_table_blocks_inserts() {
  IntIntTable table;
  auto lt = table.lock_table();
  std::thread thread([&table]() {
    for (int i = 0; i < 10; ++i) {
      table.insert(i, i);
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  TEST_CHECK(table.size() == 0);
  lt.unlock();
  thread.join();

  TEST_CHECK(table.size() == 10);
}

void test_iterator_cast_iterator_to_const_iterator() {
  IntIntTable table;
  for (int i = 0; i < 10; ++i) {
    table.insert(i, i);
  }
  auto lt = table.lock_table();
  for (IntIntTable::locked_table::iterator it = lt.begin(); it != lt.end();
       ++it) {
    TEST_CHECK(it->first == it->second);
    it->second++;
    IntIntTable::locked_table::const_iterator const_it = it;
    TEST_CHECK(it->first + 1 == it->second);
  }
}
