#include "test_c_interface.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <cerrno>
#include <cstdio>

extern "C" {
#include "int_int_table.h"
}

int cuckoo_find_fn_value;
void cuckoo_find_fn(const int *value) { cuckoo_find_fn_value = *value; }

void cuckoo_increment_fn(int *value) { ++(*value); }

bool cuckoo_erase_fn(int *value) { return (*value) & 1; }

struct BasicFixture {
  int_int_table *tbl{};

  BasicFixture() { tbl = int_int_table_init(0); }

  ~BasicFixture() { int_int_table_free(tbl); }

  static BasicFixture empty() { return BasicFixture(); }

  static BasicFixture filled() {
    BasicFixture ret;
    for (int i = 0; i < 10; i++) {
      int_int_table_insert(ret.tbl, &i, &i);
    }
    return ret;
  }
};

void test_c_interface_basic() {
  // "empty table statistics"
  {
    auto fixture = BasicFixture::empty();
    auto *tbl = fixture.tbl;

    TEST_CHECK(int_int_table_hashpower(tbl) == 0);
    TEST_CHECK(int_int_table_bucket_count(tbl) == 1);
    TEST_CHECK(int_int_table_empty(tbl));
    TEST_CHECK(int_int_table_capacity(tbl) == 4);
    TEST_CHECK(int_int_table_load_factor(tbl) == 0);
  }

  // "find_fn"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      TEST_CHECK(cuckoo_find_fn_value == i);
    }
    for (int i = 10; i < 20; ++i) {
      TEST_CHECK(!int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
    }
  }

  // "update_fn"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(int_int_table_update_fn(tbl, &i, cuckoo_increment_fn));
    }
    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      TEST_CHECK(cuckoo_find_fn_value == i + 1);
    }
  }

  // "upsert"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(!int_int_table_upsert(tbl, &i, cuckoo_increment_fn, &i));
      TEST_CHECK(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      TEST_CHECK(cuckoo_find_fn_value == i + 1);
    }
    for (int i = 10; i < 20; ++i) {
      TEST_CHECK(int_int_table_upsert(tbl, &i, cuckoo_increment_fn, &i));
      TEST_CHECK(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      TEST_CHECK(cuckoo_find_fn_value == i);
    }
  }

  // "erase_fn"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 0; i < 10; ++i) {
      if (i & 1) {
        TEST_CHECK(int_int_table_erase_fn(tbl, &i, cuckoo_erase_fn));
        TEST_CHECK(!int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      } else {
        TEST_CHECK(int_int_table_erase_fn(tbl, &i, cuckoo_erase_fn));
        TEST_CHECK(int_int_table_find_fn(tbl, &i, cuckoo_find_fn));
      }
    }
  }

  // "find"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    int value;
    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(int_int_table_find(tbl, &i, &value));
      TEST_CHECK(value == i);
    }
    for (int i = 10; i < 20; ++i) {
      TEST_CHECK(!int_int_table_find(tbl, &i, &value));
    }
  }

  // "contains"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(int_int_table_contains(tbl, &i));
    }
    for (int i = 10; i < 20; ++i) {
      TEST_CHECK(!int_int_table_contains(tbl, &i));
    }
  }

  // "update"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    int value;
    for (int i = 0; i < 10; ++i) {
      int new_value = i + 1;
      TEST_CHECK(int_int_table_update(tbl, &i, &new_value));
      TEST_CHECK(int_int_table_find(tbl, &i, &value));
      TEST_CHECK(value == i + 1);
    }
    for (int i = 10; i < 20; ++i) {
      TEST_CHECK(!int_int_table_update(tbl, &i, &value));
    }
  }

  // "insert_or_assign"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(!int_int_table_insert_or_assign(tbl, &i, &i));
    }
    for (int i = 10; i < 20; ++i) {
      TEST_CHECK(int_int_table_insert_or_assign(tbl, &i, &i));
    }
    for (int i = 0; i < 20; ++i) {
      int value;
      TEST_CHECK(int_int_table_find(tbl, &i, &value));
      TEST_CHECK(value == i);
    }
  }

  // "erase"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    for (int i = 1; i < 10; i += 2) {
      TEST_CHECK(int_int_table_erase(tbl, &i));
    }
    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(int_int_table_contains(tbl, &i) != (i & 1));
    }
  }

  // "rehash"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    TEST_CHECK(int_int_table_rehash(tbl, 15));
    TEST_CHECK(int_int_table_hashpower(tbl) == 15);
  }

  // "reserve"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    TEST_CHECK(int_int_table_reserve(tbl, 30));
    TEST_CHECK(int_int_table_hashpower(tbl) == 3);
  }

  // "clear"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    int_int_table_clear(tbl);
    TEST_CHECK(int_int_table_empty(tbl));
  }

  // "read/write"
  {
    auto fixture = BasicFixture::filled();
    auto *tbl = fixture.tbl;

    FILE *fp = tmpfile();
    int_int_table_locked_table *ltbl = int_int_table_lock_table(tbl);
    TEST_CHECK(int_int_table_locked_table_write(ltbl, fp));
    rewind(fp);
    int_int_table *tbl2 = int_int_table_read(fp);
    TEST_CHECK(int_int_table_size(tbl2) == 10);
    for (int i = 0; i < 10; ++i) {
      int value;
      TEST_CHECK(int_int_table_find(tbl2, &i, &value));
      TEST_CHECK(i == value);
    }
    int_int_table_free(tbl2);
    int_int_table_locked_table_free(ltbl);
    fclose(fp);
  }
}

struct LockedTableFixture {
  int_int_table *tbl{};
  int_int_table_locked_table *ltbl{};

  LockedTableFixture() {
    tbl = int_int_table_init(0);
    ltbl = int_int_table_lock_table(tbl);
  }

  ~LockedTableFixture() {
    int_int_table_locked_table_free(ltbl);
    int_int_table_free(tbl);
  }

  static LockedTableFixture empty() { return LockedTableFixture(); }

  static LockedTableFixture filled() {
    LockedTableFixture ret;
    for (int i = 0; i < 10; i++) {
      int_int_table_locked_table_insert(ret.ltbl, &i, &i, NULL);
    }
    return ret;
  }
};

void test_c_interface_locked_table() {
  // "is_active/unlock"
  {
    auto fixture = LockedTableFixture::empty();
    auto *ltbl = fixture.ltbl;

    TEST_CHECK(int_int_table_locked_table_is_active(ltbl));
    int_int_table_locked_table_unlock(ltbl);
    TEST_CHECK(!int_int_table_locked_table_is_active(ltbl));
  }

  // "statistics"
  {
    auto fixture = LockedTableFixture::empty();
    auto *ltbl = fixture.ltbl;

    TEST_CHECK(int_int_table_locked_table_hashpower(ltbl) == 0);
    TEST_CHECK(int_int_table_locked_table_bucket_count(ltbl) == 1);
    TEST_CHECK(int_int_table_locked_table_empty(ltbl));
    TEST_CHECK(int_int_table_locked_table_size(ltbl) == 0);
    TEST_CHECK(int_int_table_locked_table_capacity(ltbl) == 4);
    TEST_CHECK(int_int_table_locked_table_load_factor(ltbl) == 0);
  }

  // "constant iteration"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int occurrences[10] = {};
    int_int_table_const_iterator *begin =
        int_int_table_locked_table_cbegin(ltbl);
    int_int_table_const_iterator *end = int_int_table_locked_table_cend(ltbl);
    for (; !int_int_table_const_iterator_equal(begin, end);
         int_int_table_const_iterator_increment(begin)) {
      ++occurrences[*int_int_table_const_iterator_key(begin)];
      TEST_CHECK(*int_int_table_const_iterator_key(begin) ==
                 *int_int_table_const_iterator_mapped(begin));
    }
    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(occurrences[i] == 1);
    }
    int_int_table_const_iterator_decrement(end);
    int_int_table_const_iterator_set(begin, end);
    TEST_CHECK(int_int_table_const_iterator_equal(begin, end));
    int_int_table_locked_table_set_cbegin(ltbl, begin);
    for (; !int_int_table_const_iterator_equal(end, begin);
         int_int_table_const_iterator_decrement(end)) {
      ++occurrences[*int_int_table_const_iterator_key(end)];
    }
    ++occurrences[*int_int_table_const_iterator_key(end)];
    for (int i = 0; i < 10; ++i) {
      TEST_CHECK(occurrences[i] == 2);
    }
    int_int_table_const_iterator_free(end);
    int_int_table_const_iterator_free(begin);
  }

  // "iteration"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_iterator *begin = int_int_table_locked_table_begin(ltbl);
    int_int_table_iterator *end = int_int_table_locked_table_end(ltbl);
    for (; !int_int_table_iterator_equal(begin, end);
         int_int_table_iterator_increment(begin)) {
      ++(*int_int_table_iterator_mapped(begin));
    }
    int_int_table_iterator_set(begin, end);
    TEST_CHECK(int_int_table_iterator_equal(begin, end));
    int_int_table_locked_table_set_begin(ltbl, begin);
    for (; !int_int_table_iterator_equal(begin, end);
         int_int_table_iterator_increment(begin)) {
      TEST_CHECK(*int_int_table_iterator_key(begin) + 1 ==
                 *int_int_table_iterator_mapped(begin));
    }
    int_int_table_iterator_free(end);
    int_int_table_iterator_free(begin);
  }

  // "clear"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_locked_table_clear(ltbl);
    TEST_CHECK(int_int_table_locked_table_size(ltbl) == 0);
  }

  // "insert with iterator"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_iterator *it = int_int_table_locked_table_begin(ltbl);
    int item = 11;
    TEST_CHECK(int_int_table_locked_table_insert(ltbl, &item, &item, it));
    TEST_CHECK(*int_int_table_iterator_key(it) == 11);
    TEST_CHECK(*int_int_table_iterator_mapped(it) == 11);
    item = 5;
    TEST_CHECK(!int_int_table_locked_table_insert(ltbl, &item, &item, it));
    TEST_CHECK(*int_int_table_iterator_key(it) == 5);
    ++(*int_int_table_iterator_mapped(it));
    TEST_CHECK(*int_int_table_iterator_mapped(it) == 6);
    int_int_table_iterator_free(it);
  }

  // "erase"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_iterator *it1 = int_int_table_locked_table_begin(ltbl);
    int_int_table_iterator *it2 = int_int_table_locked_table_begin(ltbl);
    int_int_table_iterator_increment(it2);

    int_int_table_locked_table_erase_it(ltbl, it1, it1);
    TEST_CHECK(int_int_table_iterator_equal(it1, it2));

    int_int_table_const_iterator *cbegin =
        int_int_table_locked_table_cbegin(ltbl);
    int_int_table_iterator_increment(it2);

    int_int_table_locked_table_erase_const_it(ltbl, cbegin, it1);
    TEST_CHECK(int_int_table_iterator_equal(it1, it2));

    int_int_table_const_iterator_free(cbegin);
    int_int_table_iterator_free(it2);
    int_int_table_iterator_free(it1);

    int successes = 0;
    for (int i = 0; i < 10; ++i) {
      successes += int_int_table_locked_table_erase(ltbl, &i);
    }
    TEST_CHECK(successes == 8);
    TEST_CHECK(int_int_table_locked_table_empty(ltbl));
  }

  // "find"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_iterator *it = int_int_table_locked_table_begin(ltbl);
    int_int_table_const_iterator *cit = int_int_table_locked_table_cbegin(ltbl);

    int item = 0;
    int_int_table_locked_table_find(ltbl, &item, it);
    TEST_CHECK(*int_int_table_iterator_key(it) == 0);
    TEST_CHECK(*int_int_table_iterator_mapped(it) == 0);
    item = 10;
    int_int_table_locked_table_find_const(ltbl, &item, cit);
    int_int_table_const_iterator *cend = int_int_table_locked_table_cend(ltbl);
    TEST_CHECK(int_int_table_const_iterator_equal(cit, cend));

    int_int_table_const_iterator_free(cend);
    int_int_table_const_iterator_free(cit);
    int_int_table_iterator_free(it);
  }

  // "rehash"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_locked_table_rehash(ltbl, 15);
    TEST_CHECK(int_int_table_locked_table_hashpower(ltbl) == 15);
  }

  // "reserve"
  {
    auto fixture = LockedTableFixture::filled();
    auto *ltbl = fixture.ltbl;

    int_int_table_locked_table_reserve(ltbl, 30);
    TEST_CHECK(int_int_table_locked_table_hashpower(ltbl) == 3);
  }
}
