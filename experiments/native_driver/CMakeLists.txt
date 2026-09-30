cmake_minimum_required(VERSION 3.18)
project(LupineNativeDriverExperiment LANGUAGES CXX)

# Standalone experiment: deliberately excluded from Lupine's default build.
# The preload library runs the native NVIDIA driver on a GPU-less CPU client;
# the backend executes raw device operations and never loads libcuda.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "The native-driver experiment requires Linux x86-64")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_library(native_driver_client SHARED client.cpp metadata.cpp)
target_link_libraries(native_driver_client PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
add_executable(native_driver_server server.cpp)
target_link_libraries(native_driver_server PRIVATE Threads::Threads)
# A static, non-PIE backend leaves the client's shared-library addresses free
# for temporary ioctl argument mappings. Collisions fail without replacement.
target_link_options(native_driver_server PRIVATE -static -no-pie)
add_executable(native_driver_probe probe.cpp)
target_link_libraries(native_driver_probe PRIVATE ${CMAKE_DL_LIBS})
add_library(native_driver_boundary SHARED ioctl_boundary.cpp)
target_link_libraries(native_driver_boundary PRIVATE ${CMAKE_DL_LIBS})

foreach(target native_driver_client native_driver_server native_driver_probe
               native_driver_boundary)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
endforeach()
