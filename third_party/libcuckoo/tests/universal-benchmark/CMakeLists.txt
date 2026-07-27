add_executable(universal_benchmark universal_benchmark.cc)

# put these in the cache so they show up in ccmake
set (UNIVERSAL_KEY uint64_t CACHE STRING "set the key type used by the universal benchmark")
set (UNIVERSAL_VALUE uint64_t CACHE STRING "set the value type used by the universal benchmark")
set (UNIVERSAL_TABLE LIBCUCKOO CACHE STRING "set the table type used by the universal benchmark")
option (UNIVERSAL_TRACKING_ALLOCATOR "if on, also sample memory usage in the universal benchmark")

target_link_libraries(universal_benchmark
    PRIVATE test_util
    PRIVATE libcuckoo
    PRIVATE pcg
)

target_compile_options(universal_benchmark
    PRIVATE -DKEY=${UNIVERSAL_KEY}
    PRIVATE -DVALUE=${UNIVERSAL_VALUE}
    PRIVATE -D${UNIVERSAL_TABLE}
)

if(UNIVERSAL_TRACKING_ALLOCATOR)
    target_compile_options(universal_benchmark PRIVATE -DTRACKING_ALLOCATOR)
endif()

add_test(NAME pure_read
         COMMAND universal_benchmark --reads 100 --prefill 75 --total-ops 500 --initial-capacity 23)
add_test(NAME pure_insert
         COMMAND universal_benchmark --inserts 100 --total-ops 75 --initial-capacity 23)
add_test(NAME pure_erase
         COMMAND universal_benchmark --erases 100 --prefill 75 --total-ops 75 --initial-capacity 23)
add_test(NAME pure_update
         COMMAND universal_benchmark --updates 100 --prefill 75 --total-ops 500 --initial-capacity 23)
add_test(NAME pure_upsert
         COMMAND universal_benchmark --upserts 100 --prefill 25 --total-ops 200 --initial-capacity 23)

add_test(NAME insert_expansion
         COMMAND universal_benchmark --inserts 100 --initial-capacity 4 --total-ops 13107200)
add_test(NAME read_insert_expansion
         COMMAND universal_benchmark --reads 80 --inserts 20 --initial-capacity 10 --total-ops 4096000)
