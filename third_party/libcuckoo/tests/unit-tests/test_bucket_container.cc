#include "test_bucket_container.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <libcuckoo/bucket_container.hh>

template <bool PROPAGATE_COPY_ASSIGNMENT = true,
          bool PROPAGATE_MOVE_ASSIGNMENT = true, bool PROPAGATE_SWAP = true>
struct allocator_wrapper {
  template <class T> class stateful_allocator {
  public:
    using value_type = T;
    using propagate_on_container_copy_assignment =
        std::integral_constant<bool, PROPAGATE_COPY_ASSIGNMENT>;
    using propagate_on_container_move_assignment =
        std::integral_constant<bool, PROPAGATE_MOVE_ASSIGNMENT>;
    using propagate_on_container_swap =
        std::integral_constant<bool, PROPAGATE_SWAP>;

    stateful_allocator() : id(0) {}
    stateful_allocator(const size_t &id_) : id(id_) {}
    stateful_allocator(const stateful_allocator &other) : id(other.id) {}
    template <class U>
    stateful_allocator(const stateful_allocator<U> &other) : id(other.id) {}

    stateful_allocator &operator=(const stateful_allocator &a) {
      id = a.id + 1;
      return *this;
    }

    stateful_allocator &operator=(stateful_allocator &&a) {
      id = a.id + 2;
      return *this;
    }

    T *allocate(size_t n) { return std::allocator<T>().allocate(n); }

    void deallocate(T *ptr, size_t n) {
      std::allocator<T>().deallocate(ptr, n);
    }

    stateful_allocator select_on_container_copy_construction() const {
      stateful_allocator copy(*this);
      ++copy.id;
      return copy;
    }

    bool operator==(const stateful_allocator &other) { return id == other.id; }

    bool operator!=(const stateful_allocator &other) { return id != other.id; }

    size_t id;
  };
};

template <class T, bool PCA = true, bool PMA = true>
using stateful_allocator =
    typename allocator_wrapper<PCA, PMA>::template stateful_allocator<T>;

const size_t SLOT_PER_BUCKET = 4;

template <class Alloc>
using TestingContainer =
    libcuckoo::bucket_container<std::shared_ptr<int>, int, Alloc, uint8_t,
                                SLOT_PER_BUCKET>;

using value_type = std::pair<const std::shared_ptr<int>, int>;

void test_bucket_container_default_constructor() {
  allocator_wrapper<>::stateful_allocator<value_type> a;
  TestingContainer<decltype(a)> tc(2, a);
  TEST_CHECK(tc.hashpower() == 2);
  TEST_CHECK(tc.size() == 4);
  TEST_CHECK(tc.get_allocator().id == 0);
  for (size_t i = 0; i < tc.size(); ++i) {
    for (size_t j = 0; j < SLOT_PER_BUCKET; ++j) {
      TEST_CHECK(!tc[i].occupied(j));
    }
  }
}

void test_bucket_container_simple_stateful_allocator() {
  allocator_wrapper<>::stateful_allocator<value_type> a(10);
  TestingContainer<decltype(a)> tc(2, a);
  TEST_CHECK(tc.hashpower() == 2);
  TEST_CHECK(tc.size() == 4);
  TEST_CHECK(tc.get_allocator().id == 10);
}

void test_bucket_container_copy_construction() {
  allocator_wrapper<>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(tc);

  TEST_CHECK(tc[0].occupied(0));
  TEST_CHECK(tc[0].partial(0) == 2);
  TEST_CHECK(*tc[0].key(0) == 10);
  TEST_CHECK(tc[0].mapped(0) == 5);
  TEST_CHECK(tc.get_allocator().id == 5);

  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(tc2.get_allocator().id == 6);
}

void test_bucket_container_move_construction() {
  allocator_wrapper<>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(std::move(tc));

  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(tc2.get_allocator().id == 5);
}

void test_bucket_container_copy_assignment_with_propagate() {
  allocator_wrapper<true>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(2, a);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc2 = tc;
  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].key(0).use_count() == 2);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(!tc2[1].occupied(0));

  TEST_CHECK(tc.get_allocator().id == 5);
  TEST_CHECK(tc2.get_allocator().id == 6);
}

void test_bucket_container_copy_assignment_no_propagate() {
  allocator_wrapper<false>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(2, a);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc2 = tc;
  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].key(0).use_count() == 2);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(!tc2[1].occupied(0));

  TEST_CHECK(tc.get_allocator().id == 5);
  TEST_CHECK(tc2.get_allocator().id == 5);
}

void test_bucket_container_move_assignment_with_propagate() {
  allocator_wrapper<>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(2, a);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc2 = std::move(tc);
  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(!tc2[1].occupied(0));
  TEST_CHECK(tc2.get_allocator().id == 7);
}

void test_bucket_container_move_assignment_no_propagate_equal() {
  allocator_wrapper<true, false>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(2, a);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc2 = std::move(tc);
  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].key(0).use_count() == 1);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(!tc2[1].occupied(0));
  TEST_CHECK(tc2.get_allocator().id == 5);
}

void test_bucket_container_move_assignment_no_propagate_unequal() {
  allocator_wrapper<true, false>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  allocator_wrapper<true, false>::stateful_allocator<value_type> a2(4);
  TestingContainer<decltype(a)> tc2(2, a2);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc2 = std::move(tc);
  TEST_CHECK(!tc2[1].occupied(0));
  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].key(0).use_count() == 1);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(!tc2[1].occupied(0));
  TEST_CHECK(tc2.get_allocator().id == 4);

  TEST_CHECK(tc[0].occupied(0));
  TEST_CHECK(tc[0].partial(0) == 2);
  TEST_CHECK(!tc[0].key(0));
}

void test_bucket_container_swap_no_propagate() {
  allocator_wrapper<true, true, false>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(2, a);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc.swap(tc2);

  TEST_CHECK(tc[1].occupied(0));
  TEST_CHECK(tc[1].partial(0) == 2);
  TEST_CHECK(*tc[1].key(0) == 10);
  TEST_CHECK(tc[1].key(0).use_count() == 1);
  TEST_CHECK(tc[1].mapped(0) == 5);
  TEST_CHECK(tc.get_allocator().id == 5);

  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].key(0).use_count() == 1);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(tc2.get_allocator().id == 5);
}

void test_bucket_container_swap_propagate() {
  allocator_wrapper<true, true, true>::stateful_allocator<value_type> a(5);
  TestingContainer<decltype(a)> tc(2, a);
  tc.setKV(0, 0, 2, std::make_shared<int>(10), 5);
  TestingContainer<decltype(a)> tc2(2, a);
  tc2.setKV(1, 0, 2, std::make_shared<int>(10), 5);

  tc.swap(tc2);

  TEST_CHECK(tc[1].occupied(0));
  TEST_CHECK(tc[1].partial(0) == 2);
  TEST_CHECK(*tc[1].key(0) == 10);
  TEST_CHECK(tc[1].key(0).use_count() == 1);
  TEST_CHECK(tc[1].mapped(0) == 5);
  TEST_CHECK(tc.get_allocator().id == 7);

  TEST_CHECK(tc2[0].occupied(0));
  TEST_CHECK(tc2[0].partial(0) == 2);
  TEST_CHECK(*tc2[0].key(0) == 10);
  TEST_CHECK(tc2[0].key(0).use_count() == 1);
  TEST_CHECK(tc2[0].mapped(0) == 5);
  TEST_CHECK(tc2.get_allocator().id == 7);
}

struct ExceptionInt {
  int x;
  static bool do_throw;

  ExceptionInt(int x_) : x(x_) { maybeThrow(); }

  ExceptionInt(const ExceptionInt &other) : x(other.x) { maybeThrow(); }

  ExceptionInt &operator=(const ExceptionInt &other) {
    x = other.x;
    maybeThrow();
    return *this;
  }

  ~ExceptionInt() { maybeThrow(); }

private:
  void maybeThrow() {
    if (do_throw) {
      throw std::runtime_error("thrown");
    }
  }
};

bool ExceptionInt::do_throw = false;

using ExceptionContainer =
    libcuckoo::bucket_container<ExceptionInt, int,
                                std::allocator<std::pair<ExceptionInt, int>>,
                                uint8_t, SLOT_PER_BUCKET>;

void test_bucket_container_setKV_with_throwing_type_maintains_strong_guarantee() {
  ExceptionContainer container(0, ExceptionContainer::allocator_type());
  container.setKV(0, 0, 0, ExceptionInt(10), 20);

  ExceptionInt::do_throw = true;
  TEST_EXCEPTION(container.setKV(0, 1, 0, 0, 0), std::runtime_error);
  ExceptionInt::do_throw = false;

  TEST_CHECK(container[0].occupied(0));
  TEST_CHECK(container[0].key(0).x == 10);
  TEST_CHECK(container[0].mapped(0) == 20);

  TEST_CHECK(!container[0].occupied(1));
}

void test_bucket_container_copy_assignment_with_throwing_type_is_destroyed_properly() {
  ExceptionContainer container(0, ExceptionContainer::allocator_type());
  container.setKV(0, 0, 0, ExceptionInt(10), 20);
  ExceptionContainer other(0, ExceptionContainer::allocator_type());

  ExceptionInt::do_throw = true;
  TEST_EXCEPTION(other = container, std::runtime_error);
  ExceptionInt::do_throw = false;
}

void test_bucket_container_copy_destroyed_buckets_container() {
  std::allocator<value_type> a;
  TestingContainer<decltype(a)> bc(2, a);
  TEST_CHECK(!bc.is_deallocated());
  bc.clear_and_deallocate();
  TEST_CHECK(bc.is_deallocated());
  auto bc2 = bc;
  TEST_CHECK(bc.is_deallocated());
  TEST_CHECK(bc2.is_deallocated());
  TEST_CHECK(bc.size() == bc2.size());
  TEST_CHECK(bc.get_allocator() == bc2.get_allocator());
}
