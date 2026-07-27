#include "test_heterogeneous_compare.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <libcuckoo/cuckoohash_map.hh>

size_t int_constructions;
size_t copy_constructions;
size_t destructions;
size_t foo_comparisons;
size_t int_comparisons;
size_t foo_hashes;
size_t int_hashes;

class Foo {
public:
  int val;

  Foo(int v) {
    ++int_constructions;
    val = v;
  }

  Foo(const Foo &x) {
    ++copy_constructions;
    val = x.val;
  }

  ~Foo() { ++destructions; }
};

class foo_eq {
public:
  bool operator()(const Foo &left, const Foo &right) const {
    ++foo_comparisons;
    return left.val == right.val;
  }

  bool operator()(const Foo &left, const int right) const {
    ++int_comparisons;
    return left.val == right;
  }
};

class foo_hasher {
public:
  size_t operator()(const Foo &x) const {
    ++foo_hashes;
    return static_cast<size_t>(x.val);
  }

  size_t operator()(const int x) const {
    ++int_hashes;
    return static_cast<size_t>(x);
  }
};

typedef libcuckoo::cuckoohash_map<Foo, bool, foo_hasher, foo_eq> foo_map;

void init_section() {
  int_constructions = 0;
  copy_constructions = 0;
  destructions = 0;
  foo_comparisons = 0;
  int_comparisons = 0;
  foo_hashes = 0;
  int_hashes = 0;
}

void test_heterogeneous_compare() {
  // insert
  {
    init_section();
    {
      foo_map map;
      map.insert(0, true);
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 0);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 1);
  }

  // foo insert
  {
    init_section();
    {
      foo_map map;
      map.insert(Foo(0), true);
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 1);
    // One destruction of passed-in and moved argument, and one after the
    // table is destroyed.
    TEST_CHECK(destructions == 2);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 0);
    TEST_CHECK(foo_hashes == 1);
    TEST_CHECK(int_hashes == 0);
  }

  // insert_or_assign
  {
    init_section();
    {
      foo_map map;
      map.insert_or_assign(0, true);
      map.insert_or_assign(0, false);
      TEST_CHECK(!map.find(0));
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 2);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 3);
  }

  // foo insert_or_assign
  {
    init_section();
    {
      foo_map map;
      map.insert_or_assign(Foo(0), true);
      map.insert_or_assign(Foo(0), false);
      TEST_CHECK(!map.find(Foo(0)));
    }
    TEST_CHECK(int_constructions == 3);
    TEST_CHECK(copy_constructions == 1);
    // Three destructions of Foo arguments, and one in table destruction
    TEST_CHECK(destructions == 4);
    TEST_CHECK(foo_comparisons == 2);
    TEST_CHECK(int_comparisons == 0);
    TEST_CHECK(foo_hashes == 3);
    TEST_CHECK(int_hashes == 0);
  }

  // find
  {
    init_section();
    {
      foo_map map;
      map.insert(0, true);
      bool val;
      map.find(0, val);
      TEST_CHECK(val);
      TEST_CHECK(map.find(0, val) == true);
      TEST_CHECK(map.find(1, val) == false);
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 2);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 4);
  }

  // foo find
  {
    init_section();
    {
      foo_map map;
      map.insert(0, true);
      bool val;
      map.find(Foo(0), val);
      TEST_CHECK(val);
      TEST_CHECK(map.find(Foo(0), val) == true);
      TEST_CHECK(map.find(Foo(1), val) == false);
    }
    TEST_CHECK(int_constructions == 4);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 4);
    TEST_CHECK(foo_comparisons == 2);
    TEST_CHECK(int_comparisons == 0);
    TEST_CHECK(foo_hashes == 3);
    TEST_CHECK(int_hashes == 1);
  }

  // contains
  {
    init_section();
    {
      foo_map map(0);
      map.rehash(2);
      map.insert(0, true);
      TEST_CHECK(map.contains(0));
      // Shouldn't do comparison because of different partial key
      TEST_CHECK(!map.contains(4));
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 1);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 3);
  }

  // erase
  {
    init_section();
    {
      foo_map map;
      map.insert(0, true);
      TEST_CHECK(map.erase(0));
      TEST_CHECK(!map.contains(0));
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 1);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 3);
  }

  // update
  {
    init_section();
    {
      foo_map map;
      map.insert(0, true);
      TEST_CHECK(map.update(0, false));
      TEST_CHECK(!map.find(0));
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 2);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 3);
  }

  // update_fn
  {
    init_section();
    {
      foo_map map;
      map.insert(0, true);
      TEST_CHECK(map.update_fn(0, [](bool &val) { val = !val; }));
      TEST_CHECK(!map.find(0));
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 2);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 3);
  }

  // upsert
  {
    init_section();
    {
      foo_map map(0);
      map.rehash(2);
      auto neg = [](bool &val) { val = !val; };
      map.upsert(0, neg, true);
      map.upsert(0, neg, true);
      // Shouldn't do comparison because of different partial key
      map.upsert(4, neg, false);
      TEST_CHECK(!map.find(0));
      TEST_CHECK(!map.find(4));
    }
    TEST_CHECK(int_constructions == 2);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 2);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 3);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 5);
  }

  // uprase_fn
  {
    init_section();
    {
      foo_map map(0);
      map.rehash(2);
      auto fn = [](bool &val) {
        val = !val;
        return val;
      };
      TEST_CHECK(map.uprase_fn(0, fn, true));
      TEST_CHECK(!map.uprase_fn(0, fn, true));
      TEST_CHECK(map.contains(0));
      TEST_CHECK(!map.uprase_fn(0, fn, true));
      TEST_CHECK(!map.contains(0));
    }
    TEST_CHECK(int_constructions == 1);
    TEST_CHECK(copy_constructions == 0);
    TEST_CHECK(destructions == 1);
    TEST_CHECK(foo_comparisons == 0);
    TEST_CHECK(int_comparisons == 3);
    TEST_CHECK(foo_hashes == 0);
    TEST_CHECK(int_hashes == 5);
  }
}
