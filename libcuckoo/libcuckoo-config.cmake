#
# libcuckoo-config.cmake
#

include (CMakeFindDependencyMacro)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_dependency(Threads)

include ("${CMAKE_CURRENT_LIST_DIR}/libcuckoo-targets.cmake")
