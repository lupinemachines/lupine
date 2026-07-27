add_executable(nested_table nested_table.cc)
target_link_libraries(nested_table libcuckoo)

add_executable(hellohash hellohash.cc)
target_link_libraries(hellohash libcuckoo)

add_executable(count_freq count_freq.cc)
target_link_libraries(count_freq libcuckoo)

add_library(int_str_table STATIC int_str_table.cc)
target_link_libraries(int_str_table libcuckoo)

add_library(blob_blob_table STATIC blob_blob_table.cc)
target_link_libraries(blob_blob_table libcuckoo)

add_executable(c_hash c_hash.c)
target_link_libraries(c_hash int_str_table blob_blob_table)

set_property(TARGET c_hash PROPERTY C_STANDARD 99)
