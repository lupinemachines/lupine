#include "test_noncopyable_types.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <string>
#include <utility>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

using Tbl = UniquePtrTable<int>;
using Uptr = std::unique_ptr<int>;

const size_t TBL_INIT = 1;
const size_t TBL_SIZE = TBL_INIT * Tbl::slot_per_bucket() * 2;

void check_key_eq(Tbl &tbl, int key, int expected_val) {
  TEST_CHECK(tbl.contains(Uptr(new int(key))));
  tbl.find_fn(Uptr(new int(key)), [expected_val](const Uptr &ptr) {
    TEST_CHECK(*ptr == expected_val);
  });
}

void test_noncopyable_types_insert_and_update() {
  Tbl tbl(TBL_INIT);
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    TEST_CHECK(tbl.insert(Uptr(new int(i)), Uptr(new int(i))));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i);
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.update(Uptr(new int(i)), Uptr(new int(i + 1)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i + 1);
  }
}

void test_noncopyable_types_insert_or_assign() {
  Tbl tbl(TBL_INIT);
  for (size_t i = 0; i < TBL_SIZE / 2; ++i) {
    TEST_CHECK(tbl.insert_or_assign(Uptr(new int(i)), Uptr(new int(i))));
  }
  for (size_t i = 0; i < TBL_SIZE / 2; ++i) {
    check_key_eq(tbl, i, i);
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.insert_or_assign(Uptr(new int(i)), Uptr(new int(10)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, 10);
  }
}

void test_noncopyable_types_upsert() {
  Tbl tbl(TBL_INIT);
  auto increment = [](Uptr &ptr) { *ptr += 1; };
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.upsert(Uptr(new int(i)), increment, Uptr(new int(i)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i);
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.upsert(Uptr(new int(i)), increment, Uptr(new int(i)));
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i + 1);
  }

  auto increment_if_already_existed_else_init =
      [](Uptr &ptr, const libcuckoo::UpsertContext context) {
        if (context == libcuckoo::UpsertContext::ALREADY_EXISTED) {
          *ptr += 1;
        } else {
          ptr = Uptr(new int(-1));
        }
      };
  for (size_t i = 0; i < TBL_SIZE * 2; ++i) {
    tbl.upsert(Uptr(new int(i)), increment_if_already_existed_else_init);
  }
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    check_key_eq(tbl, i, i + 2);
  }
  for (size_t i = TBL_SIZE; i < TBL_SIZE * 2; ++i) {
    check_key_eq(tbl, i, -1);
  }
}

void test_noncopyable_types_iteration() {
  Tbl tbl(TBL_INIT);
  for (size_t i = 0; i < TBL_SIZE; ++i) {
    tbl.insert(Uptr(new int(i)), Uptr(new int(i)));
  }
  {
    auto locked_tbl = tbl.lock_table();
    for (auto &kv : locked_tbl) {
      TEST_CHECK(*kv.first == *kv.second);
      *kv.second += 1;
    }
  }
  {
    auto locked_tbl = tbl.lock_table();
    for (auto &kv : locked_tbl) {
      TEST_CHECK(*kv.first == *kv.second - 1);
    }
  }
}

void test_noncopyable_types_nested_table() {
  typedef libcuckoo::cuckoohash_map<char, std::string> inner_tbl;
  typedef libcuckoo::cuckoohash_map<std::string, std::unique_ptr<inner_tbl>>
      nested_tbl;
  nested_tbl tbl;
  std::string keys[] = {"abc", "def"};
  for (std::string &k : keys) {
    tbl.insert(std::string(k), nested_tbl::mapped_type(new inner_tbl));
    tbl.update_fn(k, [&k](nested_tbl::mapped_type &t) {
      for (char c : k) {
        t->insert(c, std::string(k));
      }
    });
  }
  for (std::string &k : keys) {
    TEST_CHECK(tbl.contains(k));
    tbl.update_fn(k, [&k](nested_tbl::mapped_type &t) {
      for (char c : k) {
        TEST_CHECK(t->find(c) == k);
      }
    });
  }
}

void test_noncopyable_types_insert_lifetime() {
  // Successful insert
  {
    Tbl tbl;
    Uptr key(new int(20));
    Uptr value(new int(20));
    TEST_CHECK(tbl.insert(std::move(key), std::move(value)));
    TEST_CHECK(!static_cast<bool>(key));
    TEST_CHECK(!static_cast<bool>(value));
  }

  // Unsuccessful insert
  {
    Tbl tbl;
    tbl.insert(new int(20), new int(20));
    Uptr key(new int(20));
    Uptr value(new int(30));
    TEST_CHECK(!tbl.insert(std::move(key), std::move(value)));
    TEST_CHECK(static_cast<bool>(key));
    TEST_CHECK(static_cast<bool>(value));
  }
}

void test_noncopyable_types_erase_fn() {
  Tbl tbl;
  tbl.insert(new int(10), new int(10));
  auto decrement_and_erase = [](Uptr &p) {
    --(*p);
    return *p == 0;
  };
  Uptr k(new int(10));
  for (int i = 0; i < 9; ++i) {
    tbl.erase_fn(k, decrement_and_erase);
    TEST_CHECK(tbl.contains(k));
  }
  tbl.erase_fn(k, decrement_and_erase);
  TEST_CHECK(!tbl.contains(k));
}

void test_noncopyable_types_uprase_fn() {
  Tbl tbl;
  auto decrement_and_erase = [](Uptr &p) {
    --(*p);
    return *p == 0;
  };
  TEST_CHECK(
      tbl.uprase_fn(Uptr(new int(10)), decrement_and_erase, Uptr(new int(10))));
  Uptr k(new int(10)), v(new int(10));
  for (int i = 0; i < 10; ++i) {
    TEST_CHECK(!tbl.uprase_fn(std::move(k), decrement_and_erase, std::move(v)));
    TEST_CHECK((k && v));
    if (i < 9) {
      TEST_CHECK(tbl.contains(k));
    } else {
      TEST_CHECK(!tbl.contains(k));
    }
  }

  auto erase_if_newly_inserted_nullptr = [](Uptr &p,
                                            libcuckoo::UpsertContext context) {
    return (p.get() == nullptr &&
            context == libcuckoo::UpsertContext::NEWLY_INSERTED);
  };
  TEST_CHECK(tbl.uprase_fn(Uptr(new int(10)), erase_if_newly_inserted_nullptr));
  TEST_CHECK(!tbl.contains(k));
  TEST_CHECK(tbl.uprase_fn(Uptr(new int(10)), erase_if_newly_inserted_nullptr,
                           Uptr(new int(10))));
  TEST_CHECK(tbl.contains(k));
}
