add_library(test_util INTERFACE)
target_include_directories(test_util INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# put these in the cache so they show up in ccmake
option (BUILD_TESTS "build all libcuckoo tests")
option (BUILD_STRESS_TESTS "build the stress tests")
option (BUILD_UNIT_TESTS "build the unit tests")
option (BUILD_UNIVERSAL_BENCHMARK "build the universal benchmark and associated tests")

# Add pcg if we're doing stress tests or universal benchmark
if (BUILD_TESTS OR
    BUILD_STRESS_TESTS OR
    BUILD_UNIVERSAL_BENCHMARK)
    add_subdirectory(pcg)
endif()

if (BUILD_TESTS OR BUILD_UNIT_TESTS)
    add_subdirectory(unit-tests)
endif()

if (BUILD_TESTS OR BUILD_STRESS_TESTS)
    add_subdirectory(stress-tests)
endif()

if (BUILD_TESTS OR BUILD_UNIVERSAL_BENCHMARK)
    add_subdirectory(universal-benchmark)
endif()
