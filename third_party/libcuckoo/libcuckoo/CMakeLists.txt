# for write_basic_package_version_file()
include(CMakePackageConfigHelpers)

# we require the use of threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# generate a version for cmake to use with find_package(libcuckoo)
set (libcuckoo_VERSION "${libcuckoo_VERSION_MAJOR}.${libcuckoo_VERSION_MINOR}")
set (libcuckoo_VERSION "${libcuckoo_VERSION}.${libcuckoo_VERSION_PATCH}")

# libcuckoo is an interface (all headers) library target
add_library(libcuckoo INTERFACE)
add_library(libcuckoo::libcuckoo ALIAS libcuckoo)

# tag libcuckoo target with a c++11 feature so that libcuckoo users
# will have c++11 turned on in their compile when they use this target.
# XXX: newer cmakes have a "cxx_std_11" feature that could be used
target_compile_features (libcuckoo INTERFACE cxx_constexpr)

# Include relative to the base directory
target_include_directories(libcuckoo INTERFACE
    $<BUILD_INTERFACE:${libcuckoo_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# switch on threading for all targets that link with libcuckoo
target_link_libraries(libcuckoo INTERFACE Threads::Threads)

# cmake packaging
set (libcuckoo_pkgloc "share/cmake/libcuckoo")

write_basic_package_version_file(
    "libcuckoo-config-version.cmake" VERSION ${libcuckoo_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(TARGETS libcuckoo EXPORT libcuckoo-targets)
install(EXPORT libcuckoo-targets
        NAMESPACE libcuckoo::
        DESTINATION ${libcuckoo_pkgloc}
        FILE "libcuckoo-targets.cmake")
install(FILES libcuckoo-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/libcuckoo-config-version.cmake
              DESTINATION ${libcuckoo_pkgloc})
install(
FILES
    cuckoohash_config.hh
    cuckoohash_map.hh
    cuckoohash_util.hh
    bucket_container.hh
DESTINATION
    ${CMAKE_INSTALL_PREFIX}/include/libcuckoo
)
