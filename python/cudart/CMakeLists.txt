# lupine_cudart — CUDA runtime API translation onto the LUPINE client shim.
#
# This is a Python-package-specific component: it exists so the `lupine`
# wheel can run CUDA binaries (including nvcc-compiled ones) on hosts with
# no NVIDIA runtime. It is intentionally not part of the root project.
#
# Usage (clients must already be built):
#   cmake -S python/cudart -B build-cudart \
#     -DLUPINE_CUDA_INCLUDE_DIR=<cuda/nvml headers> \
#     -DLUPINE_CLIENT_DIR=<dir with the built client shim>
#   cmake --build build-cudart --parallel
#
# Output names follow the platform's real runtime library so consumers link
# transparently: libcudart.so.13 (with the version node torch references),
# libcudart.dylib, cudart64_13.dll.

cmake_minimum_required(VERSION 3.18)
project(lupine_cudart LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT LUPINE_CUDA_INCLUDE_DIR)
  # Fall back to a full toolkit when present.
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_INCLUDE_DIRS)
    set(LUPINE_CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
  endif()
endif()
if(NOT LUPINE_CUDA_INCLUDE_DIR)
  message(FATAL_ERROR
    "Pass -DLUPINE_CUDA_INCLUDE_DIR=<dir with cuda.h> (the redistributable "
    "CUDA header wheels' include directory works).")
endif()

if(NOT LUPINE_CLIENT_DIR)
  message(FATAL_ERROR
    "Pass -DLUPINE_CLIENT_DIR=<dir with the built LUPINE client shim> "
    "(libcuda.so.1 / libcuda.dylib / nvcuda.dll).")
endif()

# Host-side expansion of the nvcc-internal crt/host_defines.h macros the
# redistributable headers reference; a full toolkit's real header shadows
# nothing because the stub is placed first only when no crt/ exists.
set(LUPINE_STUB_DIR ${CMAKE_CURRENT_BINARY_DIR}/stubs)
file(MAKE_DIRECTORY ${LUPINE_STUB_DIR}/crt)
file(COPY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/host_defines_stub.h
     ${LUPINE_STUB_DIR}/crt/host_defines.h ONLY_IF_DIFFERENT)

add_library(lupine_cudart SHARED lupine_cudart.cpp)
target_include_directories(lupine_cudart BEFORE PRIVATE
  ${LUPINE_STUB_DIR} ${LUPINE_CUDA_INCLUDE_DIR})
set_property(TARGET lupine_cudart PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET lupine_cudart PROPERTY BUILD_RPATH "$ORIGIN")

if(WIN32)
  # MSVC links the import library, not the DLL itself.
  set(LUPINE_CLIENT_LIB ${LUPINE_CLIENT_DIR}/nvcuda.lib)
  set_property(TARGET lupine_cudart PROPERTY OUTPUT_NAME cudart64_13)
  target_link_libraries(lupine_cudart PRIVATE ${LUPINE_CLIENT_LIB})
else()
  if(APPLE)
    set(LUPINE_CLIENT_LIB ${LUPINE_CLIENT_DIR}/libcuda.dylib)
    set_property(TARGET lupine_cudart PROPERTY OUTPUT_NAME cudart)
    set_property(TARGET lupine_cudart PROPERTY INSTALL_RPATH "@loader_path")
  else()
    set(LUPINE_CLIENT_LIB ${LUPINE_CLIENT_DIR}/libcuda.so.1)
    set_property(TARGET lupine_cudart PROPERTY OUTPUT_NAME cudart)
    set_property(TARGET lupine_cudart PROPERTY VERSION 13)
    set_property(TARGET lupine_cudart PROPERTY SOVERSION 13)
    target_link_options(lupine_cudart PRIVATE
      -static-libgcc
      -static-libstdc++
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cudart.exports")
  endif()
  target_link_libraries(lupine_cudart PRIVATE ${LUPINE_CLIENT_LIB} pthread)
endif()

if(NOT EXISTS ${LUPINE_CLIENT_LIB})
  message(FATAL_ERROR "Client shim not found at ${LUPINE_CLIENT_LIB}")
endif()

include(GNUInstallDirs)
install(TARGETS lupine_cudart LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
