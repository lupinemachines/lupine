cmake_minimum_required(VERSION 3.5.0)
project(libcuckoo
    VERSION 0.3.1
    LANGUAGES C CXX)

# put these in the cache so they show up in ccmake
option (BUILD_EXAMPLES "build example libcuckoo programs")

# Add the libcuckoo interface target
add_subdirectory(libcuckoo)

# Add C interface target
add_subdirectory(libcuckoo-c)

# Build examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build tests -- this only builds tests that were specified
enable_testing()
add_subdirectory(tests)
