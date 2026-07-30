add_library(int_int_table STATIC int_int_table.cc)
target_link_libraries(int_int_table libcuckoo)

add_executable(unit_tests
    test_bucket_container.cc
    test_c_interface.cc
    test_constructor.cc
    test_hash_properties.cc
    test_heterogeneous_compare.cc
    test_iterator.cc
    test_locked_table.cc
    test_maximum_hashpower.cc
    test_minimum_load_factor.cc
    test_noncopyable_types.cc
    test_resize.cc
    test_user_exceptions.cc

    acutest.h
    test_runner.cc
    unit_test_util.cc
    unit_test_util.hh
)

target_link_libraries(unit_tests
    PRIVATE libcuckoo
    PRIVATE int_int_table
)

add_test(NAME unit_tests COMMAND unit_tests)
