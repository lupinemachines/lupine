cmake_minimum_required(VERSION 3.18)

project(LupineProject LANGUAGES C CXX)

if (NOT WIN32)
    enable_language(CUDA)
endif()

set(MIN_CUDA_VERSION 11.0)

find_package(CUDAToolkit REQUIRED)
find_path(NGHTTP2_INCLUDE_DIR nghttp2/nghttp2.h REQUIRED)
find_library(NGHTTP2_LIBRARY nghttp2 REQUIRED)

if (NOT CUDAToolkit_VERSION)
    message(FATAL_ERROR "CUDA Toolkit is required but not found.")
endif()

if (CUDAToolkit_VERSION VERSION_LESS MIN_CUDA_VERSION)
    message(FATAL_ERROR "CUDA Toolkit version must be >= ${MIN_CUDA_VERSION}. Found: ${CUDAToolkit_VERSION}")
endif()

message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT WIN32)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NGHTTP2_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4
)

# LZ4 is vendored (third_party/lz4, BSD-2-Clause) so builds do not depend on a
# system liblz4. It is built as a plain C static library to keep it out of the
# CUDA source handling below.
add_library(lupine_lz4 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lz4.c)
set_target_properties(lupine_lz4 PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Unoptimized LZ4 compresses slower than the wire, turning compression into a
# slowdown, so keep it optimized even when no CMAKE_BUILD_TYPE is set.
if(NOT MSVC AND NOT CMAKE_BUILD_TYPE)
  target_compile_options(lupine_lz4 PRIVATE -O2)
endif()

set(CLIENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/h2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compress.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/client_routing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_client.cpp
)

set(SERVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/h2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/manual_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nvml_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compress.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_server.cpp
)

set(NVML_CLIENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/h2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compress.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nvml_client.cpp
)

set(CLIENT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/client_routing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_client.h
)

set(SERVER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lupine_platform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/manual_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/nvml_server.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.h
)

set(CLIENT_OUTPUT lupine_driver)
set(NVML_CLIENT_OUTPUT lupine_nvml)
set(SERVER_OUTPUT lupine_driver_server)

if (NOT WIN32)
    add_library(${CLIENT_OUTPUT} SHARED ${CLIENT_SOURCES} ${CLIENT_HEADERS})
    target_include_directories(${CLIENT_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${CLIENT_OUTPUT} PRIVATE stdc++ ${NGHTTP2_LIBRARY} lupine_lz4)
    target_link_options(${CLIENT_OUTPUT} PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/client.exports"
    )
    set_target_properties(${CLIENT_OUTPUT} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME cuda
        VERSION 1
        SOVERSION 1
    )

    add_library(${NVML_CLIENT_OUTPUT} SHARED ${NVML_CLIENT_SOURCES})
    target_include_directories(${NVML_CLIENT_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${NVML_CLIENT_OUTPUT} PRIVATE stdc++ ${NGHTTP2_LIBRARY} lupine_lz4)
    target_link_options(${NVML_CLIENT_OUTPUT} PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/nvml.exports"
    )
    set_target_properties(${NVML_CLIENT_OUTPUT} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME nvidia-ml
        VERSION 1
        SOVERSION 1
    )
endif()

add_executable(${SERVER_OUTPUT} ${SERVER_SOURCES} ${SERVER_HEADERS})
target_include_directories(${SERVER_OUTPUT} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${SERVER_OUTPUT} PRIVATE CUDA::cuda_driver ${NGHTTP2_LIBRARY} lupine_lz4)

if (WIN32)
    target_link_libraries(${SERVER_OUTPUT} PRIVATE ws2_32)
else()
    target_link_libraries(${SERVER_OUTPUT} PRIVATE dl)

    add_executable(h2_test
        ${CMAKE_CURRENT_SOURCE_DIR}/h2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/compress.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/h2_test.cpp
    )
    target_link_libraries(h2_test PRIVATE stdc++ ${NGHTTP2_LIBRARY} lupine_lz4 pthread)
    enable_testing()
    add_test(NAME h2_test COMMAND h2_test)
    add_test(
        NAME nvml_ecc_exports
        COMMAND bash -c "nm -D \"$<TARGET_FILE:${NVML_CLIENT_OUTPUT}>\" | grep -q nvmlDeviceGetTotalEccErrors && nm -D \"$<TARGET_FILE:${NVML_CLIENT_OUTPUT}>\" | grep -q nvmlDeviceGetDetailedEccErrors && nm -D \"$<TARGET_FILE:${NVML_CLIENT_OUTPUT}>\" | grep -q nvmlDeviceGetMemoryErrorCounter"
    )
    set_source_files_properties(
        ${SERVER_SOURCES}
        PROPERTIES LANGUAGE CUDA
    )
endif()

if (NOT WIN32)
    target_compile_options(${SERVER_OUTPUT} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>: --compiler-options=-fPIC,-g,-Wno-deprecated-declarations>
    )

    target_compile_options(${CLIENT_OUTPUT} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -Wno-deprecated-declarations>
    )

    target_compile_options(${NVML_CLIENT_OUTPUT} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -Wno-deprecated-declarations>
    )

    message(STATUS "Building client output: ${CLIENT_OUTPUT}")
    message(STATUS "Building NVML client output: ${NVML_CLIENT_OUTPUT}")
endif()
message(STATUS "Building server output: ${SERVER_OUTPUT}")
