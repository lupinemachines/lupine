add_executable(stress_checked stress_checked.cc)
target_link_libraries(stress_checked
    PRIVATE test_util
    PRIVATE pcg
    PRIVATE libcuckoo
)

add_executable(stress_unchecked stress_unchecked.cc)
target_link_libraries(stress_unchecked
    PRIVATE test_util
    PRIVATE pcg
    PRIVATE libcuckoo
)

add_test(NAME stress_checked COMMAND stress_checked)
add_test(NAME stress_unchecked COMMAND stress_unchecked)
