cmake_minimum_required(VERSION 3.18)

project(LupineProject LANGUAGES C CXX)

option(LUPINE_BUILD_CUDA "Build the CUDA client and server backend" ON)
option(LUPINE_BUILD_NVML "Build the NVML client and server backend" ON)
option(LUPINE_BUILD_HIP
       "Build the HIP client and server backend when HIP SDK headers are available"
       ON)
if(APPLE)
    set(LUPINE_BUILD_SERVER_DEFAULT OFF)
else()
    set(LUPINE_BUILD_SERVER_DEFAULT ON)
endif()
option(LUPINE_BUILD_SERVER "Build the RPC server"
       ${LUPINE_BUILD_SERVER_DEFAULT})
set(LUPINE_CUDA_DRIVER_LIBRARY "" CACHE FILEPATH
    "CUDA driver library or stub to use without a complete CUDA Toolkit")
set(LUPINE_CUDA_VERSION_OVERRIDE "" CACHE STRING
    "CUDA version reported by an SDK-free server build")
set(LUPINE_CUDA_SMEMCPY_LIBRARY "" CACHE FILEPATH
    "Precompiled CUDA smemcpy library for an SDK-free server build")

# Sanitizers are applied only to the CPU-only *_test executables. Set to a
# -fsanitize value such as "address,undefined" or "thread"; empty disables.
set(LUPINE_SANITIZE "" CACHE STRING
    "Sanitizer for CPU test targets (e.g. address,undefined or thread)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Client and server must come from the same build: the RPC opcode numbering and
# the struct layouts behind it are shared and unversioned. This identity travels
# with every connection so a mismatched pair is refused at connect instead of
# failing later as an opaque runtime error. Builds without git leave it empty,
# which reads as "cannot verify" rather than as a match. Committed builds compare
# exactly; a dirty tree is only flagged as dirty, not content-hashed, so two
# different uncommitted trees at the same commit still compare equal.
find_package(Git QUIET)
set(LUPINE_WIRE_IDENTITY "")
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev=12
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE LUPINE_WIRE_IDENTITY
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()

include(CTest)
find_package(Threads REQUIRED)
find_path(NGHTTP2_INCLUDE_DIR nghttp2/nghttp2.h REQUIRED)
find_library(NGHTTP2_LIBRARY nghttp2 REQUIRED)
find_package(OpenSSL QUIET)

set(MIN_CUDA_VERSION 11.0)
if(LUPINE_BUILD_CUDA OR LUPINE_BUILD_NVML)
    if(LUPINE_BUILD_SERVER AND LUPINE_BUILD_CUDA AND
       NOT LUPINE_CUDA_DRIVER_LIBRARY)
        find_package(CUDAToolkit REQUIRED)
        if(NOT CUDAToolkit_VERSION)
            message(FATAL_ERROR
                "CUDA Toolkit is required for the CUDA server backend.")
        endif()
        if(CUDAToolkit_VERSION VERSION_LESS MIN_CUDA_VERSION)
            message(FATAL_ERROR
                "CUDA Toolkit must be >= ${MIN_CUDA_VERSION}; found ${CUDAToolkit_VERSION}")
        endif()
        message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    else()
        find_path(LUPINE_CUDA_HEADER_DIR cuda.h
            HINTS
                ${CUDAToolkit_ROOT}/include
                $ENV{CUDA_HOME}/include
                $ENV{CUDA_PATH}/include
                /usr/local/cuda/include
            REQUIRED)
        list(APPEND CUDAToolkit_INCLUDE_DIRS ${LUPINE_CUDA_HEADER_DIR})
        if(LUPINE_BUILD_NVML)
            find_path(LUPINE_NVML_HEADER_DIR nvml.h
                HINTS
                    ${CUDAToolkit_ROOT}/include
                    $ENV{CUDA_HOME}/include
                    $ENV{CUDA_PATH}/include
                    /usr/local/cuda/include
                REQUIRED)
            list(APPEND CUDAToolkit_INCLUDE_DIRS ${LUPINE_NVML_HEADER_DIR})
        endif()
        file(STRINGS "${LUPINE_CUDA_HEADER_DIR}/cuda.h"
             LUPINE_CUDA_VERSION_LINE
             REGEX "^#define CUDA_VERSION +[0-9]+$")
        string(REGEX REPLACE ".*CUDA_VERSION +([0-9]+).*" "\\1"
               LUPINE_CUDA_HEADER_VERSION "${LUPINE_CUDA_VERSION_LINE}")
        message(STATUS
            "Using CUDA API headers version: ${LUPINE_CUDA_HEADER_VERSION}")

        if(LUPINE_BUILD_SERVER AND LUPINE_BUILD_CUDA)
            if(NOT EXISTS "${LUPINE_CUDA_DRIVER_LIBRARY}")
                message(FATAL_ERROR
                    "LUPINE_CUDA_DRIVER_LIBRARY does not exist: ${LUPINE_CUDA_DRIVER_LIBRARY}")
            endif()
            if(NOT LUPINE_CUDA_VERSION_OVERRIDE)
                message(FATAL_ERROR
                    "LUPINE_CUDA_VERSION_OVERRIDE is required with LUPINE_CUDA_DRIVER_LIBRARY")
            endif()
            set(CUDAToolkit_VERSION "${LUPINE_CUDA_VERSION_OVERRIDE}")
            if(CUDAToolkit_VERSION VERSION_LESS MIN_CUDA_VERSION)
                message(FATAL_ERROR
                    "CUDA Toolkit must be >= ${MIN_CUDA_VERSION}; found ${CUDAToolkit_VERSION}")
            endif()
            add_library(CUDA::cuda_driver UNKNOWN IMPORTED)
            set_target_properties(CUDA::cuda_driver PROPERTIES
                IMPORTED_LOCATION "${LUPINE_CUDA_DRIVER_LIBRARY}")
            message(STATUS
                "Using CUDA driver stub for SDK-free server build: ${LUPINE_CUDA_DRIVER_LIBRARY}")
        endif()
    endif()
endif()

if(LUPINE_BUILD_HIP)
    find_path(LUPINE_HIP_INCLUDE_DIR hip/hip_runtime_api.h
        HINTS
            $ENV{HIP_PATH}/include
            $ENV{ROCM_PATH}/include
            /opt/rocm/include)
    if(LUPINE_HIP_INCLUDE_DIR)
        message(STATUS "Found HIP SDK headers: ${LUPINE_HIP_INCLUDE_DIR}")
    else()
        message(STATUS "HIP SDK headers not found; disabling the HIP backend")
        set(LUPINE_BUILD_HIP OFF)
    endif()
endif()

if(LUPINE_BUILD_SERVER AND
   NOT (LUPINE_BUILD_CUDA OR LUPINE_BUILD_NVML OR LUPINE_BUILD_HIP))
    message(FATAL_ERROR
        "LUPINE_BUILD_SERVER requires at least one available backend")
endif()

# ops holds device code, so each backend needs a real offline compiler rather
# than the headers and driver stub the rest of the server builds against. The
# CUDA and HIP operations are independent, and either may be built alone.
if(LUPINE_BUILD_SERVER AND LUPINE_BUILD_CUDA)
    if(LUPINE_CUDA_SMEMCPY_LIBRARY)
        get_filename_component(lupine_cuda_smemcpy_library_dir
            "${LUPINE_CUDA_SMEMCPY_LIBRARY}" DIRECTORY)
        set(lupine_cudart_static_library
            "${lupine_cuda_smemcpy_library_dir}/libcudart_static.a")
        set(lupine_cudadevrt_library
            "${lupine_cuda_smemcpy_library_dir}/libcudadevrt.a")
        foreach(library
                LUPINE_CUDA_SMEMCPY_LIBRARY
                lupine_cudart_static_library
                lupine_cudadevrt_library)
            if(NOT EXISTS "${${library}}")
                message(FATAL_ERROR "${library} does not exist: ${${library}}")
            endif()
        endforeach()
        add_library(lupine_cuda_smemcpy STATIC IMPORTED)
        set_target_properties(lupine_cuda_smemcpy PROPERTIES
            IMPORTED_LOCATION "${LUPINE_CUDA_SMEMCPY_LIBRARY}")
        target_include_directories(lupine_cuda_smemcpy INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(lupine_cuda_smemcpy INTERFACE
            "${lupine_cudadevrt_library}"
            "${lupine_cudart_static_library}"
            Threads::Threads
            ${CMAKE_DL_LIBS})
        if(UNIX AND NOT APPLE)
            target_link_libraries(lupine_cuda_smemcpy INTERFACE rt)
        endif()
        message(STATUS
            "Using precompiled CUDA smemcpy: ${LUPINE_CUDA_SMEMCPY_LIBRARY}")
    elseif(NOT LUPINE_CUDA_DRIVER_LIBRARY)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR
            "LUPINE_CUDA_SMEMCPY_LIBRARY is required for an SDK-free CUDA server build")
    endif()
endif()

if(LUPINE_BUILD_SERVER AND LUPINE_BUILD_HIP AND
   NOT CMAKE_VERSION VERSION_LESS 3.21)
    # hipcc defaults to gfx906 alone, which loads but never runs on a newer
    # card, and clang's "native" needs /dev/kfd, so it fails on GPU-less
    # builders and under WSL. Ship a fat binary covering the supported CDNA and
    # RDNA targets instead. This must precede the compiler probe so the probe
    # and the build agree on the targets.
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES
            gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201)
    endif()
    # The HIP SDK check above only needs headers, which is all the client shim
    # and the server dispatch require. Device code additionally needs a HIP
    # compiler, so the operation is built only where one exists.
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
    endif()
endif()

get_property(lupine_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST lupine_enabled_languages OR
   "HIP" IN_LIST lupine_enabled_languages)
    add_subdirectory(ops)
endif()

function(lupine_internal_target target)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen
    )
endfunction()

function(lupine_apply_sanitizer target)
    if(LUPINE_SANITIZE)
        target_compile_options(${target} PRIVATE
            -fsanitize=${LUPINE_SANITIZE} -fno-omit-frame-pointer
            -fno-sanitize-recover=all -g)
        target_link_options(${target} PRIVATE -fsanitize=${LUPINE_SANITIZE})
    endif()
endfunction()

if(LUPINE_SANITIZE)
    set(LUPINE_TEST_TIMEOUT_SCALE 8)
else()
    set(LUPINE_TEST_TIMEOUT_SCALE 1)
endif()
function(lupine_scaled_timeout out base)
    math(EXPR scaled "${base} * ${LUPINE_TEST_TIMEOUT_SCALE}")
    set(${out} ${scaled} PARENT_SCOPE)
endfunction()

# Vendored LZ4 keeps the neutral core independent of host package versions.
add_library(lupine_lz4 STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lib/lz4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lib/lz4frame.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lib/lz4hc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lib/xxhash.c)
set_target_properties(lupine_lz4 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(lupine_lz4 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4/lib)
if(NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    target_compile_options(lupine_lz4 PRIVATE -O2)
endif()

# Accelerator-neutral wire protocol, HTTP/2, compression, and platform IPC.
add_library(lupine_rpc_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/h2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.cpp
)
lupine_internal_target(lupine_rpc_core)
target_include_directories(lupine_rpc_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${NGHTTP2_INCLUDE_DIR})
target_link_libraries(lupine_rpc_core PUBLIC
    Threads::Threads ${NGHTTP2_LIBRARY} lupine_lz4)
target_compile_definitions(lupine_rpc_core PRIVATE
    LUPINE_WIRE_IDENTITY="${LUPINE_WIRE_IDENTITY}")
if(OpenSSL_FOUND)
    target_compile_definitions(lupine_rpc_core PRIVATE LUPINE_TLS_OPENSSL)
    target_link_libraries(lupine_rpc_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()
if(WIN32)
    target_link_libraries(lupine_rpc_core PUBLIC ws2_32)
endif()

add_library(lupine_client_transport STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/transport.cpp)
lupine_internal_target(lupine_client_transport)
target_include_directories(lupine_client_transport PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lupine_client_transport PUBLIC lupine_rpc_core)
if(OpenSSL_FOUND)
    target_compile_definitions(lupine_client_transport PRIVATE
        LUPINE_TLS_OPENSSL)
    target_link_libraries(lupine_client_transport PUBLIC
        OpenSSL::SSL OpenSSL::Crypto)
endif()

add_library(lupine_ipc STATIC ${CMAKE_CURRENT_SOURCE_DIR}/ipc.cpp)
lupine_internal_target(lupine_ipc)
target_include_directories(lupine_ipc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lupine_ipc PUBLIC Threads::Threads)

add_library(lupine_server_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc_server.cpp)
lupine_internal_target(lupine_server_core)
target_include_directories(lupine_server_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lupine_server_core PUBLIC lupine_rpc_core)

if(LUPINE_BUILD_CUDA)
    add_library(lupine_cuda_checkpoint STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/checkpoint.cpp)
    set_target_properties(lupine_cuda_checkpoint PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(lupine_cuda_checkpoint PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lupine_cuda_checkpoint PUBLIC Threads::Threads)
endif()

if(LUPINE_BUILD_CUDA)
    add_library(lupine_cuda_client SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_client.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/events.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/routing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_client_memcpy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_cuda_client.cpp
    )
    target_include_directories(lupine_cuda_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(lupine_cuda_client PRIVATE LUPINE_RPC_CLIENT)
    target_link_libraries(lupine_cuda_client PRIVATE
        lupine_client_transport lupine_cuda_checkpoint lupine_ipc
        ${CMAKE_DL_LIBS})
    set_target_properties(lupine_cuda_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    if(WIN32)
        target_compile_definitions(lupine_cuda_client PRIVATE
            LUPINE_WINDOWS_POSIX_SHIMS
            _CRT_DECLARE_NONSTDC_NAMES=0)
        target_compile_options(lupine_cuda_client PRIVATE /wd4996)
        set_target_properties(lupine_cuda_client PROPERTIES
            OUTPUT_NAME nvcuda
            PREFIX ""
            WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_cuda_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_cuda_client PROPERTIES OUTPUT_NAME cuda)
        if(NOT APPLE)
            target_link_libraries(lupine_cuda_client PRIVATE stdc++)
            target_link_options(lupine_cuda_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cuda_client.exports")
            set_target_properties(lupine_cuda_client PROPERTIES
                VERSION 1
                SOVERSION 1)
        endif()
    endif()
endif()

if(LUPINE_BUILD_NVML)
    add_library(lupine_nvml_client SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/nvml_client.cpp)
    target_include_directories(lupine_nvml_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(lupine_nvml_client PRIVATE lupine_client_transport)
    set_target_properties(lupine_nvml_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    if(WIN32)
        target_compile_definitions(lupine_nvml_client PRIVATE
            LUPINE_WINDOWS_POSIX_SHIMS
            _CRT_DECLARE_NONSTDC_NAMES=0)
        target_compile_options(lupine_nvml_client PRIVATE /wd4996)
        set_target_properties(lupine_nvml_client PROPERTIES
            OUTPUT_NAME nvml
            PREFIX ""
            WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_nvml_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_nvml_client PROPERTIES
            OUTPUT_NAME nvidia-ml)
        if(NOT APPLE)
            target_link_libraries(lupine_nvml_client PRIVATE stdc++)
            target_link_options(lupine_nvml_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/nvml.exports")
            set_target_properties(lupine_nvml_client PROPERTIES
                VERSION 1
                SOVERSION 1)
        endif()
    endif()
    add_custom_target(lupine_nvml DEPENDS lupine_nvml_client)
endif()

if(NOT WIN32 AND LUPINE_BUILD_HIP)
    add_library(lupine_hip_client SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_client.cpp)
    target_include_directories(lupine_hip_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${LUPINE_HIP_INCLUDE_DIR})
    target_compile_definitions(lupine_hip_client PRIVATE
        __HIP_PLATFORM_AMD__)
    target_link_libraries(lupine_hip_client PRIVATE lupine_client_transport)
    target_link_options(lupine_hip_client PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/hip.exports")
    set_target_properties(lupine_hip_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME amdhip64
        VERSION 1
        SOVERSION 1)
    add_custom_target(lupine_hip DEPENDS lupine_hip_client)
endif()

if(LUPINE_BUILD_SERVER)
    if(LUPINE_BUILD_CUDA)
        add_library(lupine_cuda_server STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/server_checkpoint.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda_server.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda_server_memcpy.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_cuda_server.cpp
        )
        lupine_internal_target(lupine_cuda_server)
        target_include_directories(lupine_cuda_server PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS})
        target_compile_definitions(lupine_cuda_server PRIVATE
            LUPINE_RPC_SERVER)
        if(MSVC)
            target_compile_options(lupine_cuda_server PRIVATE /wd4996)
        else()
            target_compile_options(lupine_cuda_server PRIVATE
                -Wno-deprecated-declarations)
        endif()
        target_link_libraries(lupine_cuda_server PRIVATE
            lupine_server_core lupine_cuda_checkpoint lupine_ipc
            lupine_cuda_smemcpy CUDA::cuda_driver)
    endif()

    if(LUPINE_BUILD_NVML)
        add_library(lupine_nvml_server STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/nvml_server.cpp)
        lupine_internal_target(lupine_nvml_server)
        target_include_directories(lupine_nvml_server PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(lupine_nvml_server PRIVATE lupine_server_core)
    endif()

    if(LUPINE_BUILD_HIP)
        add_library(lupine_hip_server STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_server.cpp)
        lupine_internal_target(lupine_hip_server)
        target_include_directories(lupine_hip_server PRIVATE
            ${LUPINE_HIP_INCLUDE_DIR})
        target_compile_definitions(lupine_hip_server PRIVATE
            __HIP_PLATFORM_AMD__)
        target_link_libraries(lupine_hip_server PRIVATE lupine_server_core)
    endif()

    add_executable(lupine_driver_server
        ${CMAKE_CURRENT_SOURCE_DIR}/server.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/registry.cpp)
    target_include_directories(lupine_driver_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(lupine_driver_server PRIVATE
        lupine_server_core lupine_ipc ${CMAKE_DL_LIBS})
    if(LUPINE_BUILD_CUDA)
        target_compile_definitions(lupine_driver_server PRIVATE
            LUPINE_BUILD_CUDA_BACKEND
            LUPINE_RPC_SERVER
            LUPINE_BACKEND_VERSION="${CUDAToolkit_VERSION}")
        target_link_libraries(lupine_driver_server PRIVATE lupine_cuda_server)
    endif()
    if(LUPINE_BUILD_NVML)
        target_compile_definitions(lupine_driver_server PRIVATE
            LUPINE_BUILD_NVML_BACKEND)
        target_link_libraries(lupine_driver_server PRIVATE lupine_nvml_server)
    endif()
    if(LUPINE_BUILD_HIP)
        target_compile_definitions(lupine_driver_server PRIVATE
            LUPINE_BUILD_HIP_BACKEND)
        target_link_libraries(lupine_driver_server PRIVATE lupine_hip_server)
    endif()
endif()

# Tests that build wherever the project does. The Linux-only block below leans
# on socketpair(), MAP_FIXED_NOREPLACE and memfd_create(); anything free of
# those belongs here instead, so it covers every platform the client ships for.
if(BUILD_TESTING)
    add_executable(transport_test
        ${CMAKE_CURRENT_SOURCE_DIR}/transport_test.cpp)
    target_link_libraries(transport_test PRIVATE lupine_rpc_core)
    lupine_apply_sanitizer(transport_test)
    add_test(NAME transport_test COMMAND transport_test)


    add_executable(checkpoint_test
        ${CMAKE_CURRENT_SOURCE_DIR}/checkpoint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/checkpoint_test.cpp)
    target_include_directories(checkpoint_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(checkpoint_test PRIVATE Threads::Threads)
    lupine_apply_sanitizer(checkpoint_test)

    add_test(NAME checkpoint_test COMMAND checkpoint_test)
    lupine_scaled_timeout(ckpt_timeout 10)
    set_tests_properties(checkpoint_test PROPERTIES TIMEOUT ${ckpt_timeout})
endif()

# Unix: h2_test needs no socketpair or mmap any more, but it hangs partway
# through on Windows (issue #656, in test_rpc_write_queue_grows), so it stays
# off that platform until the transport difference behind that is understood.
if(BUILD_TESTING AND NOT WIN32)
    add_executable(h2_test ${CMAKE_CURRENT_SOURCE_DIR}/h2_test.cpp)
    target_include_directories(h2_test PRIVATE ${NGHTTP2_INCLUDE_DIR})
    if(CUDAToolkit_VERSION)
        target_compile_definitions(h2_test PRIVATE
            LUPINE_CUDA_VERSION="${CUDAToolkit_VERSION}")
    else()
        target_compile_definitions(h2_test PRIVATE
            LUPINE_CUDA_VERSION="core-only-test")
    endif()
    target_link_libraries(h2_test PRIVATE lupine_rpc_core)
    lupine_apply_sanitizer(h2_test)
    add_test(NAME h2_test COMMAND h2_test)
    lupine_scaled_timeout(h2_timeout 60)
    set_tests_properties(h2_test PROPERTIES TIMEOUT ${h2_timeout})

    add_executable(dispatch_test ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_test.cpp)
    target_link_libraries(dispatch_test PRIVATE lupine_rpc_core)
    lupine_apply_sanitizer(dispatch_test)
    add_test(NAME dispatch_test COMMAND dispatch_test)
    lupine_scaled_timeout(dispatch_timeout 30)
    set_tests_properties(dispatch_test PROPERTIES TIMEOUT ${dispatch_timeout})
endif()

# Linux-only, not merely Unix: these pair sockets with socketpair(), place
# mappings with MAP_FIXED_NOREPLACE and back them with memfd_create(), none of
# which macOS has. Portable tests belong in the block above.
if(BUILD_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")

    if(LUPINE_BUILD_CUDA)
        add_executable(events_test
            ${CMAKE_CURRENT_SOURCE_DIR}/events.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/events_test.cpp)
        target_include_directories(events_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(events_test PRIVATE Threads::Threads)
        lupine_apply_sanitizer(events_test)
    endif()

    add_executable(ipc_test ${CMAKE_CURRENT_SOURCE_DIR}/ipc_test.cpp)
    target_link_libraries(ipc_test PRIVATE lupine_ipc)
    lupine_apply_sanitizer(ipc_test)

    add_library(test_lupinecr_provider SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_checkpoint_provider.c)
    target_include_directories(test_lupinecr_provider PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_lupinecr_provider PROPERTIES
        PREFIX ""
        OUTPUT_NAME "test-lupinecr-provider")

    add_executable(server_checkpoint_test
        ${CMAKE_CURRENT_SOURCE_DIR}/checkpoint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/server_checkpoint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_server_checkpoint.cpp)
    target_include_directories(server_checkpoint_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(server_checkpoint_test PRIVATE
        ${CMAKE_DL_LIBS} Threads::Threads)
    lupine_apply_sanitizer(server_checkpoint_test)

    if(TARGET events_test)
        add_test(NAME events_test COMMAND events_test)
        lupine_scaled_timeout(events_timeout 10)
        set_tests_properties(events_test PROPERTIES TIMEOUT ${events_timeout})
    endif()
    add_test(NAME ipc_test COMMAND ipc_test)
    lupine_scaled_timeout(ipc_timeout 10)
    set_tests_properties(ipc_test PROPERTIES TIMEOUT ${ipc_timeout})
    add_test(NAME server_checkpoint_test
        COMMAND server_checkpoint_test
                $<TARGET_FILE:test_lupinecr_provider> present)
    lupine_scaled_timeout(sckpt_timeout 10)
    set_tests_properties(server_checkpoint_test PROPERTIES
        TIMEOUT ${sckpt_timeout})
    add_test(NAME server_checkpoint_missing_provider_test
        COMMAND server_checkpoint_test
                ${CMAKE_CURRENT_BINARY_DIR}/missing-liblupinecr.so missing)
    set_tests_properties(server_checkpoint_missing_provider_test PROPERTIES
        TIMEOUT ${sckpt_timeout})

    if(TARGET lupine_driver_server AND LUPINE_BUILD_CUDA)
        add_executable(server_sigterm_test
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_server_sigterm.cpp)
        target_link_libraries(server_sigterm_test PRIVATE Threads::Threads)
        add_test(NAME server_sigterm_test
            COMMAND server_sigterm_test $<TARGET_FILE:lupine_driver_server>)
        set_tests_properties(server_sigterm_test PROPERTIES
            TIMEOUT 10 SKIP_RETURN_CODE 77)
        if(TARGET lupine_cuda_server)
            add_test(NAME server_sigterm_provider_test
                COMMAND server_sigterm_test
                        $<TARGET_FILE:lupine_driver_server>
                        $<TARGET_FILE:test_lupinecr_provider>)
            set_tests_properties(server_sigterm_provider_test PROPERTIES
                TIMEOUT 10 SKIP_RETURN_CODE 77)
        endif()
    endif()

    if(TARGET lupine_nvml_client)
        add_test(NAME nvml_ecc_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_nvml_client>\" | grep -q nvmlDeviceGetTotalEccErrors && nm -D \"$<TARGET_FILE:lupine_nvml_client>\" | grep -q nvmlDeviceGetDetailedEccErrors && nm -D \"$<TARGET_FILE:lupine_nvml_client>\" | grep -q nvmlDeviceGetMemoryErrorCounter && nm -D \"$<TARGET_FILE:lupine_nvml_client>\" | grep -q nvmlDeviceGetCudaComputeCapability")
    endif()
endif()

message(STATUS "RPC core target: lupine_rpc_core")
message(STATUS "Client TLS (OpenSSL): ${OpenSSL_FOUND}")
if(TARGET lupine_cuda_client)
    message(STATUS "Building CUDA client: lupine_cuda_client (libcuda.so.1)")
endif()
if(TARGET lupine_nvml_client)
    message(STATUS
        "Building NVML client: lupine_nvml_client (libnvidia-ml.so.1)")
endif()
if(TARGET lupine_hip_client)
    message(STATUS "Building HIP client: lupine_hip_client (libamdhip64.so.1)")
endif()
if(TARGET lupine_driver_server)
    message(STATUS "Building server: lupine_driver_server")
endif()
