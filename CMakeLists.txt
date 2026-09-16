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
set(LUPINE_PRECOMPILED_OPS "" CACHE PATH
    "Optional directory containing precompiled device operations")
set(LUPINE_CLIENT_BUNDLE_INPUT "" CACHE PATH
    "Directory containing staged native clients to embed in the server")

# Sanitizers are applied only to the CPU-only *_test executables. Set to a
# -fsanitize value such as "address,undefined" or "thread"; empty disables.
set(LUPINE_SANITIZE "" CACHE STRING
    "Sanitizer for CPU test targets (e.g. address,undefined or thread)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
find_package(Threads REQUIRED)

set(LUPINE_MANYLINUX2014 OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   "$ENV{AUDITWHEEL_POLICY}" STREQUAL "manylinux2014")
    set(LUPINE_MANYLINUX2014 ON)
    include(ExternalProject)
    include(FetchContent)
    include(ProcessorCount)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    find_package(Perl REQUIRED)
    find_program(LUPINE_MAKE_EXECUTABLE NAMES gmake make REQUIRED)

    ProcessorCount(LUPINE_BUILD_JOBS)
    if(NOT LUPINE_BUILD_JOBS)
        set(LUPINE_BUILD_JOBS 2)
    endif()

    set(LUPINE_OPENSSL_PREFIX
        "${CMAKE_BINARY_DIR}/manylinux2014/openssl")
    file(MAKE_DIRECTORY "${LUPINE_OPENSSL_PREFIX}/include")

    function(lupine_add_manylinux_nghttp2)
        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_STATIC_LIBS ON)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        set(ENABLE_LIB_ONLY ON)
        FetchContent_Declare(lupine_manylinux_nghttp2
        URL
            https://github.com/nghttp2/nghttp2/releases/download/v1.70.0/nghttp2-1.70.0.tar.gz
        URL_HASH
            SHA256=aa317e2cf9dca6afa0aed68f8fad6ff303ec6982e25a78c75c0b65e2b9b3ded5)
        FetchContent_MakeAvailable(lupine_manylinux_nghttp2)
    endfunction()
    lupine_add_manylinux_nghttp2()
    set(NGHTTP2_INCLUDE_DIR "")
    set(NGHTTP2_LIBRARY nghttp2_static)

    ExternalProject_Add(lupine_manylinux_openssl_external
        URL
            https://github.com/openssl/openssl/releases/download/openssl-3.5.8/openssl-3.5.8.tar.gz
        URL_HASH
            SHA256=a8f84a39918ec6415ce765d9b429d313ba97b8143169c172e734b9514464f5b2
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND
            ${PERL_EXECUTABLE} <SOURCE_DIR>/Configure
            --prefix=${LUPINE_OPENSSL_PREFIX}
            --openssldir=/etc/ssl
            --libdir=lib
            no-module
            no-shared
            no-tests
            no-zlib
            -fPIC
        BUILD_COMMAND
            ${LUPINE_MAKE_EXECUTABLE} -j${LUPINE_BUILD_JOBS}
        INSTALL_COMMAND ${LUPINE_MAKE_EXECUTABLE} install_sw
        BUILD_BYPRODUCTS
            ${LUPINE_OPENSSL_PREFIX}/lib/libcrypto.a
            ${LUPINE_OPENSSL_PREFIX}/lib/libssl.a)

    add_library(OpenSSL::Crypto STATIC IMPORTED GLOBAL)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_LOCATION "${LUPINE_OPENSSL_PREFIX}/lib/libcrypto.a"
        INTERFACE_INCLUDE_DIRECTORIES "${LUPINE_OPENSSL_PREFIX}/include"
        INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
    add_dependencies(OpenSSL::Crypto lupine_manylinux_openssl_external)
    add_library(OpenSSL::SSL STATIC IMPORTED GLOBAL)
    set_target_properties(OpenSSL::SSL PROPERTIES
        IMPORTED_LOCATION "${LUPINE_OPENSSL_PREFIX}/lib/libssl.a"
        INTERFACE_INCLUDE_DIRECTORIES "${LUPINE_OPENSSL_PREFIX}/include"
        INTERFACE_LINK_LIBRARIES OpenSSL::Crypto)
    add_dependencies(OpenSSL::SSL lupine_manylinux_openssl_external)
    set(OpenSSL_FOUND TRUE)
else()
    find_path(NGHTTP2_INCLUDE_DIR nghttp2/nghttp2.h REQUIRED)
    find_library(NGHTTP2_LIBRARY nghttp2 REQUIRED)
    find_package(OpenSSL QUIET)
endif()

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

# A precompiled operations directory lets the SDK-neutral server build consume
# artifacts produced by each backend's SDK. Without one, build the operations
# locally with the available accelerator compilers.
if(LUPINE_BUILD_SERVER AND LUPINE_BUILD_CUDA)
    if(LUPINE_PRECOMPILED_OPS)
        if(NOT IS_DIRECTORY "${LUPINE_PRECOMPILED_OPS}")
            message(FATAL_ERROR
                "LUPINE_PRECOMPILED_OPS is not a directory: ${LUPINE_PRECOMPILED_OPS}")
        endif()
        message(STATUS
            "Using precompiled device operations: ${LUPINE_PRECOMPILED_OPS}")
    elseif(NOT LUPINE_CUDA_DRIVER_LIBRARY)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR
            "LUPINE_PRECOMPILED_OPS is required for an SDK-free CUDA server build")
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
if((LUPINE_BUILD_SERVER AND LUPINE_BUILD_CUDA) OR
   "CUDA" IN_LIST lupine_enabled_languages OR
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

function(lupine_manylinux2014_client target)
    if(LUPINE_MANYLINUX2014)
        target_link_options(${target} PRIVATE
            -static-libgcc
            -static-libstdc++
            "LINKER:--exclude-libs,libcrypto.a:libgcc.a:libgcc_eh.a:libnghttp2.a:libssl.a:libstdc++.a")
    endif()
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
    ${CMAKE_CURRENT_SOURCE_DIR}/client_bundle.cpp
)
lupine_internal_target(lupine_rpc_core)
target_include_directories(lupine_rpc_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${NGHTTP2_INCLUDE_DIR})
target_link_libraries(lupine_rpc_core PUBLIC
    Threads::Threads ${NGHTTP2_LIBRARY} lupine_lz4)
if(LUPINE_MANYLINUX2014)
    add_dependencies(lupine_rpc_core lupine_manylinux_openssl_external)
endif()
if(OpenSSL_FOUND)
    target_compile_definitions(lupine_rpc_core PRIVATE LUPINE_TLS_OPENSSL)
    target_link_libraries(lupine_rpc_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()
if(WIN32)
    target_link_libraries(lupine_rpc_core PUBLIC ws2_32)
endif()

add_library(lupine_client_transport STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/transport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monitor_client.cpp)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_client_rpc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/events.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/routing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/visible_devices.cpp
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
    lupine_manylinux2014_client(lupine_cuda_client)
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
            target_link_options(lupine_cuda_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cuda_client.exports")
            set_target_properties(lupine_cuda_client PROPERTIES
                VERSION 1
                SOVERSION 1)
        endif()
    endif()
endif()

if(LUPINE_BUILD_CUDA)
    if(LUPINE_CUDA_HEADER_VERSION)
        math(EXPR LUPINE_CUDART_MAJOR "${LUPINE_CUDA_HEADER_VERSION} / 1000")
    else()
        set(LUPINE_CUDART_MAJOR ${CUDAToolkit_VERSION_MAJOR})
    endif()
    set(LUPINE_CUDART_SOVERSION ${LUPINE_CUDART_MAJOR})
    set(LUPINE_CUDART_DLL_SUFFIX ${LUPINE_CUDART_MAJOR})
    if(LUPINE_CUDART_MAJOR EQUAL 11)
        set(LUPINE_CUDART_SOVERSION 11.0)
        set(LUPINE_CUDART_DLL_SUFFIX 110)
    endif()
    configure_file(cudart.exports.in cudart.exports @ONLY)

    add_library(lupine_cudart_headers INTERFACE)
    target_include_directories(lupine_cudart_headers INTERFACE
        ${CUDAToolkit_INCLUDE_DIRS})
    # Header-only CUDA wheels omit the compiler's host qualifier definitions.
    find_path(LUPINE_CUDA_CRT_INCLUDE_DIR crt/host_defines.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(NOT LUPINE_CUDA_CRT_INCLUDE_DIR)
        configure_file(python/cudart/host_defines_stub.h
            cudart-headers/crt/host_defines.h COPYONLY)
        target_include_directories(lupine_cudart_headers BEFORE INTERFACE
            ${CMAKE_CURRENT_BINARY_DIR}/cudart-headers)
    endif()

    add_library(lupine_cudart_client SHARED cudart_client.cpp)
    target_include_directories(lupine_cudart_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_link_libraries(lupine_cudart_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_ipc Threads::Threads)
    set_target_properties(lupine_cudart_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_cudart_client)
    if(WIN32)
        target_compile_options(lupine_cudart_client PRIVATE /wd4996)
        set_target_properties(lupine_cudart_client PROPERTIES
            OUTPUT_NAME cudart64_${LUPINE_CUDART_DLL_SUFFIX}
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_cudart_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_cudart_client PROPERTIES OUTPUT_NAME cudart)
        if(APPLE)
            set_target_properties(lupine_cudart_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_cudart_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/cudart.exports")
            set_target_properties(lupine_cudart_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_CUDART_SOVERSION}
                SOVERSION ${LUPINE_CUDART_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_cudart DEPENDS lupine_cudart_client)
endif()

if(LUPINE_BUILD_CUDA)
    # The cuBLAS shim needs the library's own headers, which the runtime header
    # wheels the native client builds use do not carry; without them it is left
    # out of the build.
    find_path(LUPINE_CUBLAS_INCLUDE_DIR cublas_api.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_CUBLAS_INCLUDE_DIR)
        file(STRINGS ${LUPINE_CUBLAS_INCLUDE_DIR}/cublas_api.h
             LUPINE_CUBLAS_MAJOR_LINE REGEX "^#define CUBLAS_VER_MAJOR ")
        string(REGEX REPLACE "^#define CUBLAS_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_CUBLAS_SOVERSION "${LUPINE_CUBLAS_MAJOR_LINE}")
        file(STRINGS ${LUPINE_CUBLAS_INCLUDE_DIR}/cublas_api.h
             LUPINE_CUBLAS_MINOR_LINE REGEX "^#define CUBLAS_VER_MINOR ")
        string(REGEX REPLACE "^#define CUBLAS_VER_MINOR ([0-9]+).*" "\\1"
               LUPINE_CUBLAS_MINOR "${LUPINE_CUBLAS_MINOR_LINE}")
        set(LUPINE_BUILD_CUBLAS ON)
    endif()
endif()

if(LUPINE_BUILD_CUBLAS)
    # Named like NVIDIA's (libcublas.so.13, cublas64_13.dll) with the matching
    # ELF version node, since consumers reference name@libcublas.so.13.
    # libcublasLt ships with libcublas and follows the same naming. Both run on
    # the runtime shim's current device, so they link both sibling shims.
    foreach(shim IN ITEMS cublas cublaslt)
        set(target lupine_${shim}_client)
        set(output cublas)
        if(shim STREQUAL cublaslt)
            set(output cublasLt)
        endif()
        configure_file(${shim}.exports.in ${shim}.exports @ONLY)
        add_library(${target} SHARED ${shim}_client.cpp)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
        target_link_libraries(${target} PRIVATE
            lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        lupine_manylinux2014_client(${target})
        if(WIN32)
            target_compile_options(${target} PRIVATE /wd4996)
            set_target_properties(${target} PROPERTIES
                OUTPUT_NAME ${output}64_${LUPINE_CUBLAS_SOVERSION}
                PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
        else()
            target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
            set_target_properties(${target} PROPERTIES OUTPUT_NAME ${output})
            if(APPLE)
                set_target_properties(${target} PROPERTIES
                    BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
            else()
                target_link_options(${target} PRIVATE
                    "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/${shim}.exports")
                set_target_properties(${target} PROPERTIES
                    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                    VERSION ${LUPINE_CUBLAS_SOVERSION}
                    SOVERSION ${LUPINE_CUBLAS_SOVERSION})
            endif()
        endif()
        add_custom_target(lupine_${shim} DEPENDS ${target})
    endforeach()
endif()

if(LUPINE_BUILD_CUDA)
    # The cuFFT shim needs the library's own headers, which the runtime header
    # wheels the native client builds use do not carry; without them it is left
    # out of the build.
    find_path(LUPINE_CUFFT_INCLUDE_DIR cufftXt.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_CUFFT_INCLUDE_DIR)
        file(STRINGS ${LUPINE_CUFFT_INCLUDE_DIR}/cufft.h
             LUPINE_CUFFT_MAJOR_LINE REGEX "^#define CUFFT_VER_MAJOR ")
        string(REGEX REPLACE "^#define CUFFT_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_CUFFT_SOVERSION "${LUPINE_CUFFT_MAJOR_LINE}")
        set(LUPINE_BUILD_CUFFT ON)
    endif()
endif()

if(LUPINE_BUILD_CUFFT)
    # Named like NVIDIA's (libcufft.so.12, cufft64_12.dll) with the matching
    # ELF version node; cuFFT's major is its own, not the toolkit's. It runs on
    # the runtime shim's current device, so it links both sibling shims.
    configure_file(cufft.exports.in cufft.exports @ONLY)
    add_library(lupine_cufft_client SHARED cufft_client.cpp)
    target_include_directories(lupine_cufft_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_link_libraries(lupine_cufft_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_cufft_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_cufft_client)
    if(WIN32)
        target_compile_options(lupine_cufft_client PRIVATE /wd4996)
        set_target_properties(lupine_cufft_client PROPERTIES
            OUTPUT_NAME cufft64_${LUPINE_CUFFT_SOVERSION}
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_cufft_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_cufft_client PROPERTIES OUTPUT_NAME cufft)
        if(APPLE)
            set_target_properties(lupine_cufft_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_cufft_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/cufft.exports")
            set_target_properties(lupine_cufft_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_CUFFT_SOVERSION}
                SOVERSION ${LUPINE_CUFFT_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_cufft DEPENDS lupine_cufft_client)
endif()

if(LUPINE_BUILD_CUDA)
    # cuDNN ships outside the toolkit, from its own packages or wheels. Point
    # LUPINE_CUDNN_INCLUDE_DIR at a cuDNN 9 include directory when it is not
    # beside the toolkit's headers; without one the shim is left out.
    find_path(LUPINE_CUDNN_INCLUDE_DIR cudnn_version.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV CUDNN_PATH
        PATH_SUFFIXES include)
    if(LUPINE_CUDNN_INCLUDE_DIR)
        file(STRINGS ${LUPINE_CUDNN_INCLUDE_DIR}/cudnn_version.h
             LUPINE_CUDNN_MAJOR_LINE REGEX "^#define CUDNN_MAJOR ")
        string(REGEX REPLACE "^#define CUDNN_MAJOR ([0-9]+).*" "\\1"
               LUPINE_CUDNN_SOVERSION "${LUPINE_CUDNN_MAJOR_LINE}")
        if(LUPINE_CUDNN_SOVERSION GREATER_EQUAL 9)
            set(LUPINE_BUILD_CUDNN ON)
        endif()
    endif()
endif()

if(LUPINE_BUILD_CUDNN)
    # Named like NVIDIA's (libcudnn.so.9, cudnn64_9.dll) with the matching ELF
    # version node. Consumers link only libcudnn, which exports the whole API;
    # the operation sublibraries it loads are the server's concern. It runs on
    # the runtime shim's current device, so it links both sibling shims.
    configure_file(cudnn.exports.in cudnn.exports @ONLY)
    add_library(lupine_cudnn_client SHARED cudnn_client.cpp)
    target_include_directories(lupine_cudnn_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${LUPINE_CUDNN_INCLUDE_DIR})
    target_link_libraries(lupine_cudnn_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_cudnn_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_cudnn_client)
    if(WIN32)
        target_compile_options(lupine_cudnn_client PRIVATE /wd4996)
        set_target_properties(lupine_cudnn_client PROPERTIES
            OUTPUT_NAME cudnn64_${LUPINE_CUDNN_SOVERSION}
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_cudnn_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_cudnn_client PROPERTIES OUTPUT_NAME cudnn)
        if(APPLE)
            set_target_properties(lupine_cudnn_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_cudnn_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/cudnn.exports")
            set_target_properties(lupine_cudnn_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_CUDNN_SOVERSION}
                SOVERSION ${LUPINE_CUDNN_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_cudnn DEPENDS lupine_cudnn_client)
endif()

if(LUPINE_BUILD_CUDA)
    # The cuRAND shim needs the library's own header, which the runtime header
    # wheels the native client builds use do not carry; without it it is left
    # out of the build.
    find_path(LUPINE_CURAND_INCLUDE_DIR curand.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_CURAND_INCLUDE_DIR)
        file(STRINGS ${LUPINE_CURAND_INCLUDE_DIR}/curand.h
             LUPINE_CURAND_MAJOR_LINE REGEX "^#define CURAND_VER_MAJOR ")
        string(REGEX REPLACE "^#define CURAND_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_CURAND_SOVERSION "${LUPINE_CURAND_MAJOR_LINE}")
        set(LUPINE_BUILD_CURAND ON)
    endif()
endif()

if(LUPINE_BUILD_CURAND)
    # Named like NVIDIA's (libcurand.so.10, curand64_10.dll) with the matching
    # ELF version node; cuRAND's major is its own, not the toolkit's. It runs
    # on the runtime shim's current device, so it links both sibling shims.
    configure_file(curand.exports.in curand.exports @ONLY)
    add_library(lupine_curand_client SHARED curand_client.cpp)
    target_include_directories(lupine_curand_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_link_libraries(lupine_curand_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_curand_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_curand_client)
    if(WIN32)
        target_compile_options(lupine_curand_client PRIVATE /wd4996)
        set_target_properties(lupine_curand_client PROPERTIES
            OUTPUT_NAME curand64_${LUPINE_CURAND_SOVERSION}
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_curand_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_curand_client PROPERTIES OUTPUT_NAME curand)
        if(APPLE)
            set_target_properties(lupine_curand_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_curand_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/curand.exports")
            set_target_properties(lupine_curand_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_CURAND_SOVERSION}
                SOVERSION ${LUPINE_CURAND_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_curand DEPENDS lupine_curand_client)
endif()

if(LUPINE_BUILD_CUDA)
    # The cuSPARSE shim needs the library's own header, which the runtime
    # header wheels the native client builds use do not carry; without it the
    # shim is left out of the build.
    find_path(LUPINE_CUSPARSE_INCLUDE_DIR cusparse.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_CUSPARSE_INCLUDE_DIR)
        file(STRINGS ${LUPINE_CUSPARSE_INCLUDE_DIR}/cusparse.h
             LUPINE_CUSPARSE_MAJOR_LINE REGEX "^#define CUSPARSE_VER_MAJOR ")
        string(REGEX REPLACE "^#define CUSPARSE_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_CUSPARSE_SOVERSION "${LUPINE_CUSPARSE_MAJOR_LINE}")
        set(LUPINE_BUILD_CUSPARSE ON)
    endif()
endif()

if(LUPINE_BUILD_CUSPARSE)
    # Named like NVIDIA's (libcusparse.so.12, cusparse64_12.dll) with the
    # matching ELF version node; cuSPARSE's major is its own, not the
    # toolkit's. It runs on the runtime shim's current device, so it links both
    # sibling shims.
    configure_file(cusparse.exports.in cusparse.exports @ONLY)
    add_library(lupine_cusparse_client SHARED cusparse_client.cpp)
    target_include_directories(lupine_cusparse_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_link_libraries(lupine_cusparse_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_cusparse_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_cusparse_client)
    if(WIN32)
        target_compile_options(lupine_cusparse_client PRIVATE /wd4996)
        set_target_properties(lupine_cusparse_client PROPERTIES
            OUTPUT_NAME cusparse64_${LUPINE_CUSPARSE_SOVERSION}
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_cusparse_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_cusparse_client PROPERTIES
            OUTPUT_NAME cusparse)
        if(APPLE)
            set_target_properties(lupine_cusparse_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_cusparse_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/cusparse.exports")
            set_target_properties(lupine_cusparse_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_CUSPARSE_SOVERSION}
                SOVERSION ${LUPINE_CUSPARSE_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_cusparse DEPENDS lupine_cusparse_client)
endif()

if(LUPINE_BUILD_CUDA)
    # The cuSOLVER shims need the library's own headers, which the runtime
    # header wheels the native client builds use do not carry; without them
    # the shims are left out of the build.
    find_path(LUPINE_CUSOLVER_INCLUDE_DIR cusolverMg.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_CUSOLVER_INCLUDE_DIR)
        file(STRINGS ${LUPINE_CUSOLVER_INCLUDE_DIR}/cusolver_common.h
             LUPINE_CUSOLVER_MAJOR_LINE REGEX "#[ ]*define CUSOLVER_VER_MAJOR ")
        string(REGEX REPLACE ".*define CUSOLVER_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_CUSOLVER_SOVERSION "${LUPINE_CUSOLVER_MAJOR_LINE}")
        set(LUPINE_BUILD_CUSOLVER ON)
    endif()
endif()

if(LUPINE_BUILD_CUSOLVER)
    # Named like NVIDIA's (libcusolver.so.12, cusolver64_12.dll and the Mg
    # pair) with the matching ELF version nodes; cuSOLVER's major is its own,
    # not the toolkit's. They run on the runtime shim's current device, so each
    # links both sibling shims.
    foreach(shim IN ITEMS cusolver cusolvermg)
        set(target lupine_${shim}_client)
        set(output cusolver)
        if(shim STREQUAL cusolvermg)
            set(output cusolverMg)
        endif()
        configure_file(${shim}.exports.in ${shim}.exports @ONLY)
        add_library(${target} SHARED ${shim}_client.cpp)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
        target_link_libraries(${target} PRIVATE
            lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        lupine_manylinux2014_client(${target})
        if(WIN32)
            target_compile_options(${target} PRIVATE /wd4996)
            set_target_properties(${target} PROPERTIES
                OUTPUT_NAME ${output}64_${LUPINE_CUSOLVER_SOVERSION}
                PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
        else()
            target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
            set_target_properties(${target} PROPERTIES OUTPUT_NAME ${output})
            if(APPLE)
                set_target_properties(${target} PROPERTIES
                    BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
            else()
                target_link_options(${target} PRIVATE
                    "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/${shim}.exports")
                set_target_properties(${target} PROPERTIES
                    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                    VERSION ${LUPINE_CUSOLVER_SOVERSION}
                    SOVERSION ${LUPINE_CUSOLVER_SOVERSION})
            endif()
        endif()
        add_custom_target(lupine_${shim} DEPENDS ${target})
    endforeach()
endif()

if(LUPINE_BUILD_CUDA)
    # The NVRTC shim needs the library's own header, which the runtime header
    # wheels the native client builds use do not carry; without it it is left
    # out of the build.
    find_path(LUPINE_NVRTC_INCLUDE_DIR nvrtc.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_NVRTC_INCLUDE_DIR)
        # NVRTC 11.x kept the 11.2 soname across every 11 release.
        set(LUPINE_NVRTC_SOVERSION ${LUPINE_CUDART_MAJOR})
        math(EXPR LUPINE_NVRTC_DLL_SUFFIX "${LUPINE_CUDART_MAJOR} * 10")
        if(LUPINE_CUDART_MAJOR EQUAL 11)
            set(LUPINE_NVRTC_SOVERSION 11.2)
            set(LUPINE_NVRTC_DLL_SUFFIX 112)
        endif()
        set(LUPINE_BUILD_NVRTC ON)
    endif()
endif()

if(LUPINE_BUILD_NVRTC)
    # Named like NVIDIA's (libnvrtc.so.13, nvrtc64_130_0.dll) with the matching
    # ELF version node. It runs on the runtime shim's current device, so it
    # links both sibling shims.
    configure_file(nvrtc.exports.in nvrtc.exports @ONLY)
    add_library(lupine_nvrtc_client SHARED nvrtc_client.cpp)
    target_include_directories(lupine_nvrtc_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_link_libraries(lupine_nvrtc_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_nvrtc_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_nvrtc_client)
    if(WIN32)
        target_compile_options(lupine_nvrtc_client PRIVATE /wd4996)
        set_target_properties(lupine_nvrtc_client PROPERTIES
            OUTPUT_NAME nvrtc64_${LUPINE_NVRTC_DLL_SUFFIX}_0
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_nvrtc_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_nvrtc_client PROPERTIES OUTPUT_NAME nvrtc)
        if(APPLE)
            set_target_properties(lupine_nvrtc_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_nvrtc_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/nvrtc.exports")
            set_target_properties(lupine_nvrtc_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_NVRTC_SOVERSION}
                SOVERSION ${LUPINE_NVRTC_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_nvrtc DEPENDS lupine_nvrtc_client)
endif()

if(LUPINE_BUILD_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # NCCL ships outside the toolkit, from its own packages or wheels. Point
    # LUPINE_NCCL_INCLUDE_DIR at an include directory holding nccl.h when it
    # is not beside the toolkit's headers; without one the shim is left out.
    find_path(LUPINE_NCCL_INCLUDE_DIR nccl.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV NCCL_HOME
        PATH_SUFFIXES include)
    if(LUPINE_NCCL_INCLUDE_DIR)
        file(STRINGS ${LUPINE_NCCL_INCLUDE_DIR}/nccl.h
             LUPINE_NCCL_MAJOR_LINE REGEX "^#define NCCL_MAJOR ")
        string(REGEX REPLACE "^#define NCCL_MAJOR ([0-9]+).*" "\\1"
               LUPINE_NCCL_SOVERSION "${LUPINE_NCCL_MAJOR_LINE}")
        file(STRINGS ${LUPINE_NCCL_INCLUDE_DIR}/nccl.h
             LUPINE_NCCL_CODE_LINE REGEX "^#define NCCL_VERSION_CODE ")
        string(REGEX REPLACE "^#define NCCL_VERSION_CODE ([0-9]+).*" "\\1"
               LUPINE_NCCL_VERSION_CODE "${LUPINE_NCCL_CODE_LINE}")
        # 2.14.3 brings the configuration struct and ncclInProgress.
        if(LUPINE_NCCL_VERSION_CODE GREATER_EQUAL 21403)
            set(LUPINE_BUILD_NCCL ON)
        endif()
    endif()
endif()

if(LUPINE_BUILD_NCCL)
    # Named like NVIDIA's (libnccl.so.2), whose symbols carry no version node.
    # It runs on the runtime shim's current device, so it links both sibling
    # shims. nccl.h declares ncclResetDebugInit only for Linux builds.
    configure_file(nccl.exports.in nccl.exports @ONLY)
    add_library(lupine_nccl_client SHARED nccl_client.cpp)
    target_include_directories(lupine_nccl_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${LUPINE_NCCL_INCLUDE_DIR})
    target_compile_definitions(lupine_nccl_client PRIVATE NCCL_OS_LINUX)
    target_link_libraries(lupine_nccl_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    target_compile_options(lupine_nccl_client PRIVATE
        -Wno-deprecated-declarations)
    target_link_options(lupine_nccl_client PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/nccl.exports")
    set_target_properties(lupine_nccl_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME nccl
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
        VERSION ${LUPINE_NCCL_SOVERSION}
        SOVERSION ${LUPINE_NCCL_SOVERSION})
    lupine_manylinux2014_client(lupine_nccl_client)
    add_custom_target(lupine_nccl DEPENDS lupine_nccl_client)
endif()

if(LUPINE_BUILD_CUDA)
    # The nvJitLink shim needs the library's own header, from 12.4 on: earlier
    # headers define the calls only as inline wrappers, which the shim cannot
    # replace. The header's release names the release-suffixed symbols the
    # shim exports, as NVIDIA's library does.
    find_path(LUPINE_NVJITLINK_INCLUDE_DIR nvJitLink.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_NVJITLINK_INCLUDE_DIR)
        file(READ ${LUPINE_NVJITLINK_INCLUDE_DIR}/nvJitLink.h
             LUPINE_NVJITLINK_HEADER)
        string(REGEX MATCH "__nvJitLinkCreate_([0-9]+)_([0-9]+)" _
               "${LUPINE_NVJITLINK_HEADER}")
        set(LUPINE_NVJITLINK_MAJOR ${CMAKE_MATCH_1})
        set(LUPINE_NVJITLINK_MINOR ${CMAKE_MATCH_2})
        string(FIND "${LUPINE_NVJITLINK_HEADER}" "NVJITLINK_NO_INLINE"
               LUPINE_NVJITLINK_NO_INLINE)
        string(FIND "${LUPINE_NVJITLINK_HEADER}" "nvJitLinkGetLinkedLTOIR"
               LUPINE_NVJITLINK_LTOIR)
        set(LUPINE_NVJITLINK_HAS_LTOIR 0)
        if(LUPINE_NVJITLINK_LTOIR GREATER -1)
            set(LUPINE_NVJITLINK_HAS_LTOIR 1)
        endif()
        if(LUPINE_NVJITLINK_MAJOR AND LUPINE_NVJITLINK_NO_INLINE GREATER -1)
            set(LUPINE_NVJITLINK_SOVERSION ${LUPINE_NVJITLINK_MAJOR})
            set(LUPINE_NVJITLINK_DEFINITIONS
                NVJITLINK_NO_INLINE
                LUPINE_NVJITLINK_MAJOR=${LUPINE_NVJITLINK_MAJOR}
                LUPINE_NVJITLINK_MINOR=${LUPINE_NVJITLINK_MINOR}
                LUPINE_NVJITLINK_HAS_LTOIR=${LUPINE_NVJITLINK_HAS_LTOIR})
            set(LUPINE_BUILD_NVJITLINK ON)
        endif()
    endif()
endif()

if(LUPINE_BUILD_NVJITLINK)
    # Named like NVIDIA's (libnvJitLink.so.13, nvJitLink_130_0.dll) with the
    # matching ELF version node. It runs on the runtime shim's current device,
    # so it links both sibling shims.
    configure_file(nvjitlink.exports.in nvjitlink.exports @ONLY)
    add_library(lupine_nvjitlink_client SHARED nvjitlink_client.cpp)
    target_include_directories(lupine_nvjitlink_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_compile_definitions(lupine_nvjitlink_client PRIVATE
        ${LUPINE_NVJITLINK_DEFINITIONS})
    target_link_libraries(lupine_nvjitlink_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_nvjitlink_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_nvjitlink_client)
    if(WIN32)
        target_compile_options(lupine_nvjitlink_client PRIVATE /wd4996)
        set_target_properties(lupine_nvjitlink_client PROPERTIES
            OUTPUT_NAME nvJitLink_${LUPINE_NVJITLINK_MAJOR}0_0
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_nvjitlink_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_nvjitlink_client PROPERTIES
            OUTPUT_NAME nvJitLink)
        if(APPLE)
            set_target_properties(lupine_nvjitlink_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_nvjitlink_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/nvjitlink.exports")
            set_target_properties(lupine_nvjitlink_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_NVJITLINK_SOVERSION}
                SOVERSION ${LUPINE_NVJITLINK_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_nvjitlink DEPENDS lupine_nvjitlink_client)
endif()

if(LUPINE_BUILD_CUDA)
    # The nvJPEG shim needs the library's own header, which the runtime header
    # wheels the native client builds use do not carry; without it it is left
    # out of the build.
    find_path(LUPINE_NVJPEG_INCLUDE_DIR nvjpeg.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_NVJPEG_INCLUDE_DIR)
        file(STRINGS ${LUPINE_NVJPEG_INCLUDE_DIR}/nvjpeg.h
             LUPINE_NVJPEG_MAJOR_LINE REGEX "#[ ]*define NVJPEG_VER_MAJOR ")
        string(REGEX REPLACE ".*define NVJPEG_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_NVJPEG_SOVERSION "${LUPINE_NVJPEG_MAJOR_LINE}")
        set(LUPINE_BUILD_NVJPEG ON)
    endif()
endif()

if(LUPINE_BUILD_NVJPEG)
    # Named like NVIDIA's (libnvjpeg.so.13, nvjpeg64_13.dll) with the matching
    # ELF version node. It runs on the runtime shim's current device, so it
    # links both sibling shims.
    configure_file(nvjpeg.exports.in nvjpeg.exports @ONLY)
    add_library(lupine_nvjpeg_client SHARED nvjpeg_client.cpp)
    target_include_directories(lupine_nvjpeg_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
    target_link_libraries(lupine_nvjpeg_client PRIVATE
        lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
    set_target_properties(lupine_nvjpeg_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    lupine_manylinux2014_client(lupine_nvjpeg_client)
    if(WIN32)
        target_compile_options(lupine_nvjpeg_client PRIVATE /wd4996)
        set_target_properties(lupine_nvjpeg_client PROPERTIES
            OUTPUT_NAME nvjpeg64_${LUPINE_NVJPEG_SOVERSION}
            PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    else()
        target_compile_options(lupine_nvjpeg_client PRIVATE
            -Wno-deprecated-declarations)
        set_target_properties(lupine_nvjpeg_client PROPERTIES
            OUTPUT_NAME nvjpeg)
        if(APPLE)
            set_target_properties(lupine_nvjpeg_client PROPERTIES
                BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
        else()
            target_link_options(lupine_nvjpeg_client PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/nvjpeg.exports")
            set_target_properties(lupine_nvjpeg_client PROPERTIES
                BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                VERSION ${LUPINE_NVJPEG_SOVERSION}
                SOVERSION ${LUPINE_NVJPEG_SOVERSION})
        endif()
    endif()
    add_custom_target(lupine_nvjpeg DEPENDS lupine_nvjpeg_client)
endif()

set(LUPINE_NPP_LIBRARIES
    nppc nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps)

if(LUPINE_BUILD_CUDA)
    # The NPP shims need the libraries' own headers, which the runtime header
    # wheels the native client builds use do not carry; without them the shims
    # are left out of the build.
    find_path(LUPINE_NPP_INCLUDE_DIR nppcore.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(LUPINE_NPP_INCLUDE_DIR)
        file(STRINGS ${LUPINE_NPP_INCLUDE_DIR}/npp.h
             LUPINE_NPP_MAJOR_LINE REGEX "#[ ]*define NPP_VER_MAJOR ")
        string(REGEX REPLACE ".*define NPP_VER_MAJOR ([0-9]+).*" "\\1"
               LUPINE_NPP_SOVERSION "${LUPINE_NPP_MAJOR_LINE}")
        set(LUPINE_BUILD_NPP ON)
    endif()
endif()

if(LUPINE_BUILD_NPP)
    # Named like NVIDIA's (libnppial.so.13, nppial64_13.dll) with the matching
    # ELF version nodes; NPP's major is its own, not the toolkit's. They run on
    # the runtime shim's current device, so each links both sibling shims, and
    # each image and signal library needs the core one as NVIDIA's do.
    foreach(LUPINE_NPP_LIBRARY IN LISTS LUPINE_NPP_LIBRARIES)
        set(target lupine_${LUPINE_NPP_LIBRARY}_client)
        configure_file(npp.exports.in ${LUPINE_NPP_LIBRARY}.exports @ONLY)
        if(LUPINE_NPP_LIBRARY STREQUAL nppc)
            add_library(${target} SHARED nppc_client.cpp)
        else()
            add_library(${target} SHARED npp_client.cpp)
            target_compile_definitions(${target} PRIVATE
                LUPINE_NPP_CLIENT_INC="codegen/gen_${LUPINE_NPP_LIBRARY}_client.inc")
            target_link_libraries(${target} PRIVATE lupine_nppc_client)
            if(NOT WIN32 AND NOT APPLE)
                # NVIDIA's image and signal libraries need libnppc even
                # where they call nothing in it.
                target_link_options(${target} PRIVATE "LINKER:--no-as-needed")
            endif()
        endif()
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
        target_link_libraries(${target} PRIVATE
            lupine_cudart_headers lupine_cuda_client lupine_cudart_client)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        lupine_manylinux2014_client(${target})
        if(WIN32)
            target_compile_options(${target} PRIVATE /wd4996)
            set_target_properties(${target} PROPERTIES
                OUTPUT_NAME ${LUPINE_NPP_LIBRARY}64_${LUPINE_NPP_SOVERSION}
                PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
        else()
            target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
            set_target_properties(${target} PROPERTIES
                OUTPUT_NAME ${LUPINE_NPP_LIBRARY})
            if(APPLE)
                set_target_properties(${target} PROPERTIES
                    BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
            else()
                target_link_options(${target} PRIVATE
                    "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/${LUPINE_NPP_LIBRARY}.exports")
                set_target_properties(${target} PROPERTIES
                    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
                    VERSION ${LUPINE_NPP_SOVERSION}
                    SOVERSION ${LUPINE_NPP_SOVERSION})
            endif()
        endif()
        add_custom_target(lupine_${LUPINE_NPP_LIBRARY} DEPENDS ${target})
    endforeach()
endif()

if(LUPINE_BUILD_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # cuFile is Linux-only and ships with the toolkit or in its own wheel.
    # Point LUPINE_CUFILE_INCLUDE_DIR at an include directory holding cufile.h
    # when it is not beside the toolkit's headers; without one the shim is left
    # out. cufile.h carries no version macro, so the entry points it declares
    # decide what the shim defines.
    find_path(LUPINE_CUFILE_INCLUDE_DIR cufile.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV CUFILE_HOME
        PATH_SUFFIXES include)
    if(LUPINE_CUFILE_INCLUDE_DIR)
        file(READ ${LUPINE_CUFILE_INCLUDE_DIR}/cufile.h LUPINE_CUFILE_HEADER)
        set(LUPINE_CUFILE_DEFINITIONS)
        set(lupine_cufile_features
            ASYNC cuFileReadAsync
            VERSION cuFileGetVersion
            PARAMETERS cuFileGetParameterSizeT
            PARAMETER_RANGE cuFileGetParameterMinMaxValue
            STATS cuFileGetStatsL1
            SLAB_ARRAYS cuFileSetParameterPosixPoolSlabArray
            P2P_FLAGS cuFileDriverGetP2PFlags
            BAR_SIZE cuFileGetBARSizeInKB
            TOPOLOGY cuFileExportPCIeTopology)
        while(lupine_cufile_features)
            list(POP_FRONT lupine_cufile_features
                 lupine_cufile_feature lupine_cufile_symbol)
            set(lupine_cufile_present 0)
            if(LUPINE_CUFILE_HEADER MATCHES "${lupine_cufile_symbol}")
                set(lupine_cufile_present 1)
            endif()
            list(APPEND LUPINE_CUFILE_DEFINITIONS
                 LUPINE_CUFILE_HAS_${lupine_cufile_feature}=${lupine_cufile_present})
        endwhile()
        # cuFileGetVersion has no header constant to report. The oldest release
        # that declares everything this header does is the answer that never
        # promises a caller an entry point the shim does not define.
        set(LUPINE_CUFILE_VERSION 1070)
        if(LUPINE_CUFILE_HEADER MATCHES "cuFileGetParameterSizeT")
            set(LUPINE_CUFILE_VERSION 1140)
        endif()
        if(LUPINE_CUFILE_HEADER MATCHES "cuFileGetStatsL1")
            set(LUPINE_CUFILE_VERSION 1180)
        endif()
        list(APPEND LUPINE_CUFILE_DEFINITIONS
             LUPINE_CUFILE_VERSION=${LUPINE_CUFILE_VERSION})
        set(LUPINE_BUILD_CUFILE ON)
    endif()
endif()

if(LUPINE_BUILD_CUFILE)
    # Named like NVIDIA's (libcufile.so.0, whose SONAME major is 0 for every
    # cuFile release), and its symbols carry no version node either. The file
    # is read on this side and the bytes move over the driver shim's
    # connections, so it links that shim and no server library.
    add_library(lupine_cufile_client SHARED cufile_client.cpp)
    target_include_directories(lupine_cufile_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${LUPINE_CUFILE_INCLUDE_DIR})
    target_compile_definitions(lupine_cufile_client PRIVATE
        ${LUPINE_CUFILE_DEFINITIONS})
    target_link_libraries(lupine_cufile_client PRIVATE
        lupine_cudart_headers lupine_cuda_client)
    target_link_options(lupine_cufile_client PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cufile.exports")
    set_target_properties(lupine_cufile_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME cufile
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
        VERSION 0
        SOVERSION 0)
    lupine_manylinux2014_client(lupine_cufile_client)
    add_custom_target(lupine_cufile DEPENDS lupine_cufile_client)
endif()

if(LUPINE_BUILD_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # CUPTI ships with the toolkit or in its own wheel. Point
    # LUPINE_CUPTI_INCLUDE_DIR at an include directory holding cupti_result.h
    # when it is not beside the toolkit's headers; without one the shim is left
    # out. Some toolkits keep these headers under extras/CUPTI instead.
    find_path(LUPINE_CUPTI_INCLUDE_DIR cupti_result.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
              ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI ENV CUPTI_HOME
        PATH_SUFFIXES include)
    if(LUPINE_CUPTI_INCLUDE_DIR)
        # An SDK-free build has cuda.h's version and no CUDAToolkit components,
        # the way the runtime shim's major is taken.
        if(LUPINE_CUDA_HEADER_VERSION)
            math(EXPR LUPINE_CUPTI_MAJOR "${LUPINE_CUDA_HEADER_VERSION} / 1000")
            math(EXPR LUPINE_CUPTI_MINOR
                 "(${LUPINE_CUDA_HEADER_VERSION} / 10) % 100")
        else()
            set(LUPINE_CUPTI_MAJOR ${CUDAToolkit_VERSION_MAJOR})
            set(LUPINE_CUPTI_MINOR ${CUDAToolkit_VERSION_MINOR})
        endif()
        # NVIDIA's SONAME is libcupti.so.<major> from CUDA 12 on, and
        # libcupti.so.<major>.<minor> before it (libcupti.so.11.8).
        set(LUPINE_CUPTI_SOVERSION ${LUPINE_CUPTI_MAJOR})
        if(LUPINE_CUPTI_MAJOR LESS 12)
            set(LUPINE_CUPTI_SOVERSION
                ${LUPINE_CUPTI_MAJOR}.${LUPINE_CUPTI_MINOR})
        endif()
        set(LUPINE_BUILD_CUPTI ON)
    endif()
endif()

if(LUPINE_BUILD_CUPTI)
    # The shim profiles nothing and forwards nothing, so it links no driver
    # shim and no server library: what it needs from the toolkit is the result
    # codes and the API version its headers carry.
    configure_file(cupti.exports.in cupti.exports @ONLY)
    add_library(lupine_cupti_client SHARED cupti_client.cpp)
    target_include_directories(lupine_cupti_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS}
        ${LUPINE_CUPTI_INCLUDE_DIR})
    target_link_options(lupine_cupti_client PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/cupti.exports")
    set_target_properties(lupine_cupti_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME cupti
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
        VERSION ${LUPINE_CUPTI_SOVERSION}
        SOVERSION ${LUPINE_CUPTI_SOVERSION})
    lupine_manylinux2014_client(lupine_cupti_client)
    add_custom_target(lupine_cupti DEPENDS lupine_cupti_client)
    # nvSHMEM is Linux-only and ships outside the toolkit, in its own package
    # or wheel, whose headers keep a directory of their own rather than sit
    # beside the toolkit's. Point LUPINE_NVSHMEM_INCLUDE_DIR or NVSHMEM_HOME at
    # them; without them the shim is left out. nvshmem_host.h arrives with the
    # release that splits the host library out, the one carrying the NVSHMEM
    # version node and the nvshmemid entry points the shim defines.
    find_path(LUPINE_NVSHMEM_INCLUDE_DIR nvshmem_host.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV NVSHMEM_HOME
        PATH_SUFFIXES include)
    # find_path leaves a cached value alone, so a directory named on the
    # command line still has to carry the header.
    if(EXISTS "${LUPINE_NVSHMEM_INCLUDE_DIR}/nvshmem_host.h")
        file(STRINGS ${LUPINE_NVSHMEM_INCLUDE_DIR}/non_abi/nvshmem_version.h
             LUPINE_NVSHMEM_MAJOR_LINE
             REGEX "^#define NVSHMEM_VENDOR_MAJOR_VERSION ")
        string(REGEX REPLACE "^#define NVSHMEM_VENDOR_MAJOR_VERSION ([0-9]+).*"
               "\\1" LUPINE_NVSHMEM_SOVERSION
               "${LUPINE_NVSHMEM_MAJOR_LINE}")
        if(LUPINE_NVSHMEM_SOVERSION GREATER_EQUAL 3)
            set(LUPINE_BUILD_NVSHMEM ON)
        endif()
    endif()
endif()

if(LUPINE_BUILD_NVSHMEM)
    # Named like NVIDIA's (libnvshmem_host.so.3), with its symbols in the
    # NVSHMEM version node the library's consumers ask for. Nothing is
    # forwarded, so it links no sibling shim and the server needs no nvSHMEM.
    configure_file(nvshmem.exports.in nvshmem.exports @ONLY)
    add_library(lupine_nvshmem_client SHARED nvshmem_client.cpp)
    target_include_directories(lupine_nvshmem_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${LUPINE_NVSHMEM_INCLUDE_DIR})
    target_link_libraries(lupine_nvshmem_client PRIVATE lupine_cudart_headers)
    target_link_options(lupine_nvshmem_client PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/nvshmem.exports")
    set_target_properties(lupine_nvshmem_client PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME nvshmem_host
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN"
        VERSION ${LUPINE_NVSHMEM_SOVERSION}
        SOVERSION ${LUPINE_NVSHMEM_SOVERSION})
    lupine_manylinux2014_client(lupine_nvshmem_client)
    add_custom_target(lupine_nvshmem DEPENDS lupine_nvshmem_client)
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
    lupine_manylinux2014_client(lupine_nvml_client)
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

# Every CUDA runtime and library client shim, so a consumer can load them all
# in place of NVIDIA's without naming each one.
get_property(lupine_runtime_client_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER lupine_runtime_client_targets INCLUDE REGEX "^lupine_(cu[a-z]+|nvrtc|nccl|nvjitlink|nvjpeg|npp[a-z]*|nvshmem)_client$")
list(REMOVE_ITEM lupine_runtime_client_targets lupine_cuda_client)
add_custom_target(lupine_runtime_clients)
if(lupine_runtime_client_targets)
    add_dependencies(lupine_runtime_clients ${lupine_runtime_client_targets})
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
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND LUPINE_BUILD_NVML)
            target_compile_definitions(lupine_cuda_server PRIVATE
                LUPINE_MONITORING_ENABLED)
        endif()
        if(MSVC)
            target_compile_options(lupine_cuda_server PRIVATE /wd4996)
        else()
            target_compile_options(lupine_cuda_server PRIVATE
                -Wno-deprecated-declarations)
        endif()
        target_link_libraries(lupine_cuda_server PRIVATE
            lupine_server_core lupine_cuda_checkpoint lupine_ipc
            lupine_cuda_smemcpy_module CUDA::cuda_driver)

        add_library(lupine_cudart_server STATIC cudart_server.cpp)
        lupine_internal_target(lupine_cudart_server)
        target_compile_definitions(lupine_cudart_server PRIVATE LUPINE_RPC_SERVER)
        target_link_libraries(lupine_cudart_server PRIVATE
            lupine_cudart_headers lupine_server_core ${CMAKE_DL_LIBS})

        if(LUPINE_BUILD_CUBLAS)
            add_library(lupine_cublas_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/cublas_server.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/cublaslt_server.cpp)
            lupine_internal_target(lupine_cublas_server)
            target_include_directories(lupine_cublas_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_cublas_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_CUFFT)
            add_library(lupine_cufft_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/cufft_server.cpp)
            lupine_internal_target(lupine_cufft_server)
            target_include_directories(lupine_cufft_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_cufft_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_CUDNN)
            add_library(lupine_cudnn_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/cudnn_server.cpp)
            lupine_internal_target(lupine_cudnn_server)
            target_include_directories(lupine_cudnn_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS} ${LUPINE_CUDNN_INCLUDE_DIR})
            target_link_libraries(lupine_cudnn_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_CURAND)
            add_library(lupine_curand_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/curand_server.cpp)
            lupine_internal_target(lupine_curand_server)
            target_include_directories(lupine_curand_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_curand_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_CUSPARSE)
            add_library(lupine_cusparse_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/cusparse_server.cpp)
            lupine_internal_target(lupine_cusparse_server)
            # The handlers forward the deprecated legacy calls as well.
            target_compile_definitions(lupine_cusparse_server PRIVATE
                DISABLE_CUSPARSE_DEPRECATED)
            target_include_directories(lupine_cusparse_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_cusparse_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_CUSOLVER)
            add_library(lupine_cusolver_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/cusolver_server.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/cusolvermg_server.cpp)
            lupine_internal_target(lupine_cusolver_server)
            target_include_directories(lupine_cusolver_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_cusolver_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_NVRTC)
            add_library(lupine_nvrtc_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_server.cpp)
            lupine_internal_target(lupine_nvrtc_server)
            target_include_directories(lupine_nvrtc_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_nvrtc_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_NCCL)
            add_library(lupine_nccl_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/nccl_server.cpp)
            lupine_internal_target(lupine_nccl_server)
            target_compile_definitions(lupine_nccl_server PRIVATE NCCL_OS_LINUX)
            target_include_directories(lupine_nccl_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS} ${LUPINE_NCCL_INCLUDE_DIR})
            target_link_libraries(lupine_nccl_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_NVJITLINK)
            add_library(lupine_nvjitlink_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/nvjitlink_server.cpp)
            lupine_internal_target(lupine_nvjitlink_server)
            target_compile_definitions(lupine_nvjitlink_server PRIVATE
                ${LUPINE_NVJITLINK_DEFINITIONS})
            target_include_directories(lupine_nvjitlink_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_nvjitlink_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_NVJPEG)
            add_library(lupine_nvjpeg_server STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_server.cpp)
            lupine_internal_target(lupine_nvjpeg_server)
            target_include_directories(lupine_nvjpeg_server PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(lupine_nvjpeg_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()

        if(LUPINE_BUILD_NPP)
            # One translation unit per NPP library, so each compiles on its own.
            add_library(lupine_npp_server STATIC)
            lupine_internal_target(lupine_npp_server)
            foreach(library IN LISTS LUPINE_NPP_LIBRARIES)
                if(library STREQUAL nppc)
                    add_library(lupine_${library}_server OBJECT
                        ${CMAKE_CURRENT_SOURCE_DIR}/nppc_server.cpp)
                else()
                    add_library(lupine_${library}_server OBJECT
                        ${CMAKE_CURRENT_SOURCE_DIR}/npp_server.cpp)
                    target_compile_definitions(lupine_${library}_server PRIVATE
                        LUPINE_NPP_SERVER_INC="codegen/gen_${library}_server.inc")
                endif()
                lupine_internal_target(lupine_${library}_server)
                target_compile_definitions(lupine_${library}_server PRIVATE
                    LUPINE_NPP_LIBRARY="${library}")
                target_include_directories(lupine_${library}_server PRIVATE
                    ${CUDAToolkit_INCLUDE_DIRS})
                target_link_libraries(lupine_${library}_server PRIVATE
                    lupine_server_core)
                target_sources(lupine_npp_server PRIVATE
                    $<TARGET_OBJECTS:lupine_${library}_server>)
            endforeach()
            target_link_libraries(lupine_npp_server PRIVATE
                lupine_server_core ${CMAKE_DL_LIBS})
        endif()
    endif()

    if(LUPINE_BUILD_NVML)
        add_library(lupine_nvml_server STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/nvml_server.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/nvml_runtime.cpp)
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
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND LUPINE_BUILD_CUDA AND
       LUPINE_BUILD_NVML)
        target_sources(lupine_driver_server PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/monitor_server.cpp)
        target_compile_definitions(lupine_driver_server PRIVATE
            LUPINE_MONITORING_ENABLED)
    endif()
    target_include_directories(lupine_driver_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(lupine_driver_server PRIVATE
        lupine_server_core lupine_ipc ${CMAKE_DL_LIBS})
    if(LUPINE_CLIENT_BUNDLE_INPUT)
        if(NOT IS_DIRECTORY "${LUPINE_CLIENT_BUNDLE_INPUT}")
            message(FATAL_ERROR
                "LUPINE_CLIENT_BUNDLE_INPUT is not a directory: ${LUPINE_CLIENT_BUNDLE_INPUT}")
        endif()
        find_program(UV_EXECUTABLE uv REQUIRED)
        set(LUPINE_CLIENT_BUNDLE_GENERATED_DIR
            ${CMAKE_CURRENT_BINARY_DIR}/client-bundles)

        function(lupine_embed_client_bundle tag platforms)
            set(files ${ARGN})
            set(input_dir
                ${LUPINE_CLIENT_BUNDLE_INPUT}/lupine-client-${tag})
            set(output_dir ${LUPINE_CLIENT_BUNDLE_GENERATED_DIR}/${tag})
            set(archive ${output_dir}/client.zip)
            set(source ${LUPINE_CLIENT_BUNDLE_GENERATED_DIR}/${tag}.cpp)
            string(REPLACE "-" "_" symbol ${tag})
            set(inputs)
            foreach(file IN LISTS files)
                list(APPEND inputs ${input_dir}/${file})
            endforeach()
            add_custom_command(
                OUTPUT ${source}
                BYPRODUCTS ${archive}
                COMMAND ${UV_EXECUTABLE} run --script
                        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bundle_codegen.py
                        ${input_dir} ${archive} "${platforms}"
                        lupine_embedded_client_bundle_${symbol} ${source} ${files}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bundle_codegen.py
                        ${inputs}
                COMMENT "Embedding ${tag} client bundle"
                VERBATIM)
            target_sources(lupine_driver_server PRIVATE ${source})
        endfunction()

        lupine_embed_client_bundle(linux-x86_64 "linux/amd64"
            libcuda.so.1 libcudart.so.13 libnvidia-ml.so.1)
        lupine_embed_client_bundle(linux-aarch64 "linux/arm64"
            libcuda.so.1 libcudart.so.13 libnvidia-ml.so.1)
        lupine_embed_client_bundle(macosx-universal2
            "macos/amd64;macos/arm64"
            libcuda.dylib libcudart.dylib libnvidia-ml.dylib)
        lupine_embed_client_bundle(win-amd64 "windows/amd64"
            nvcuda.dll cudart64_13.dll nvml.dll)
        lupine_embed_client_bundle(win-arm64 "windows/arm64"
            nvcuda.dll cudart64_13.dll nvml.dll)
        target_compile_definitions(lupine_driver_server PRIVATE
            LUPINE_EMBED_CLIENT_BUNDLES)
        message(STATUS
            "Embedding server client bundles from: ${LUPINE_CLIENT_BUNDLE_INPUT}")
    endif()
    if(LUPINE_BUILD_CUDA)
        target_compile_definitions(lupine_driver_server PRIVATE
            LUPINE_BUILD_CUDA_BACKEND
            LUPINE_BUILD_CUDART_BACKEND
            LUPINE_RPC_SERVER
            LUPINE_BACKEND_VERSION="${CUDAToolkit_VERSION}")
        target_link_libraries(lupine_driver_server PRIVATE
            lupine_cuda_server lupine_cudart_server lupine_cudart_headers)
        if(LUPINE_BUILD_CUBLAS)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_CUBLAS_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_cublas_server)
        endif()
        if(LUPINE_BUILD_CUFFT)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_CUFFT_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_cufft_server)
        endif()
        if(LUPINE_BUILD_CUDNN)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_CUDNN_BACKEND)
            target_include_directories(lupine_driver_server PRIVATE
                ${LUPINE_CUDNN_INCLUDE_DIR})
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_cudnn_server)
        endif()
        if(LUPINE_BUILD_CURAND)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_CURAND_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_curand_server)
        endif()
        if(LUPINE_BUILD_CUSPARSE)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_CUSPARSE_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_cusparse_server)
        endif()
        if(LUPINE_BUILD_CUSOLVER)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_CUSOLVER_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_cusolver_server)
        endif()
        if(LUPINE_BUILD_NVRTC)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_NVRTC_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_nvrtc_server)
        endif()
        if(LUPINE_BUILD_NCCL)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_NCCL_BACKEND NCCL_OS_LINUX)
            target_include_directories(lupine_driver_server PRIVATE
                ${LUPINE_NCCL_INCLUDE_DIR})
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_nccl_server)
        endif()
        if(LUPINE_BUILD_NVJITLINK)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_NVJITLINK_BACKEND ${LUPINE_NVJITLINK_DEFINITIONS})
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_nvjitlink_server)
        endif()
        if(LUPINE_BUILD_NVJPEG)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_NVJPEG_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_nvjpeg_server)
        endif()
        if(LUPINE_BUILD_NPP)
            target_compile_definitions(lupine_driver_server PRIVATE
                LUPINE_BUILD_NPP_BACKEND)
            target_link_libraries(lupine_driver_server PRIVATE
                lupine_npp_server)
        endif()
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

    add_executable(visible_devices_test
        ${CMAKE_CURRENT_SOURCE_DIR}/visible_devices_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/visible_devices.cpp)
    target_include_directories(visible_devices_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    lupine_apply_sanitizer(visible_devices_test)
    add_test(NAME visible_devices_test COMMAND visible_devices_test)


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
    add_executable(h2_test
        ${CMAKE_CURRENT_SOURCE_DIR}/h2_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/monitor_client.cpp)
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

    if(TARGET lupine_cublas_client)
        # cuSOLVER imports this undeclared export from cuBLAS 12.9 on.
        set(lupine_cublas_undeclared_check "true")
        if("${LUPINE_CUBLAS_SOVERSION}.${LUPINE_CUBLAS_MINOR}" VERSION_GREATER_EQUAL 12.9)
            set(lupine_cublas_undeclared_check
                "nm -D \"$<TARGET_FILE:lupine_cublas_client>\" | grep -q ' T cublasSetEnvironmentMode@@libcublas.so.${LUPINE_CUBLAS_SOVERSION}$'")
        endif()
        add_test(NAME cublas_exports
            COMMAND bash -c
                "${lupine_cublas_undeclared_check} && nm -D \"$<TARGET_FILE:lupine_cublas_client>\" | grep -q cublasSgemm_v2 && nm -D \"$<TARGET_FILE:lupine_cublas_client>\" | grep -q cublasGemmEx && nm -D \"$<TARGET_FILE:lupine_cublas_client>\" | grep -q cublasSetMatrix && nm -D \"$<TARGET_FILE:lupine_cublas_client>\" | grep -q cublasXtSgemm && nm -D \"$<TARGET_FILE:lupine_cublaslt_client>\" | grep -q cublasLtMatmul && nm -D \"$<TARGET_FILE:lupine_cublaslt_client>\" | grep -q cublasLtMatmulDescCreate")
    endif()

    if(TARGET lupine_hip_client)
        add_test(NAME hip_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_hip_client>\" | grep -q ' T hipMalloc$' && nm -D \"$<TARGET_FILE:lupine_hip_client>\" | grep -q ' T hipMemcpyHtoD$' && nm -D \"$<TARGET_FILE:lupine_hip_client>\" | grep -q ' T hipStreamCreate$' && nm -D \"$<TARGET_FILE:lupine_hip_client>\" | grep -q ' T hipGraphLaunch$' && nm -D \"$<TARGET_FILE:lupine_hip_client>\" | grep -q ' T hipChooseDevice$'")
    endif()

    if(TARGET lupine_cufft_client)
        add_test(NAME cufft_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_cufft_client>\" | grep -q cufftPlanMany && nm -D \"$<TARGET_FILE:lupine_cufft_client>\" | grep -q cufftExecC2C && nm -D \"$<TARGET_FILE:lupine_cufft_client>\" | grep -q cufftXtMemcpy")
    endif()

    if(TARGET lupine_cudnn_client)
        add_test(NAME cudnn_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_cudnn_client>\" | grep -q cudnnConvolutionForward && nm -D \"$<TARGET_FILE:lupine_cudnn_client>\" | grep -q cudnnBackendSetAttribute && nm -D \"$<TARGET_FILE:lupine_cudnn_client>\" | grep -q cudnnMultiHeadAttnForward")
    endif()

    if(TARGET lupine_curand_client)
        add_test(NAME curand_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_curand_client>\" | grep -q ' T curandCreateGeneratorHost@@libcurand.so.${LUPINE_CURAND_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_curand_client>\" | grep -q ' T curandGenerateNormal@@libcurand.so.${LUPINE_CURAND_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_curand_client>\" | grep -q ' T curandGetDirectionVectors64@@libcurand.so.${LUPINE_CURAND_SOVERSION}$'")
    endif()

    if(TARGET lupine_cusparse_client)
        set(lupine_cusparse_removed_check "true")
        if(LUPINE_CUSPARSE_SOVERSION LESS 12)
            set(lupine_cusparse_removed_check
                "nm -D \"$<TARGET_FILE:lupine_cusparse_client>\" | grep -q ' T cusparseScsrsv2_bufferSize@@libcusparse.so.${LUPINE_CUSPARSE_SOVERSION}$'")
        endif()
        add_test(NAME cusparse_exports
            COMMAND bash -c
                "${lupine_cusparse_removed_check} && nm -D \"$<TARGET_FILE:lupine_cusparse_client>\" | grep -q ' T cusparseSpMV@' && nm -D \"$<TARGET_FILE:lupine_cusparse_client>\" | grep -q ' T cusparseCreateCsr@' && nm -D \"$<TARGET_FILE:lupine_cusparse_client>\" | grep -q ' T cusparseXcsrsort@' && nm -D \"$<TARGET_FILE:lupine_cusparse_client>\" | grep -q ' T cusparseGetErrorString@'")
    endif()

    if(TARGET lupine_cusolver_client)
        add_test(NAME cusolver_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_cusolver_client>\" | grep -q ' T cusolverDnXgetrf@@libcusolver.so.${LUPINE_CUSOLVER_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_cusolver_client>\" | grep -q ' T cusolverSpDcsrqrsvBatched@@libcusolver.so.${LUPINE_CUSOLVER_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_cusolver_client>\" | grep -q ' T cusolverRfRefactor@@libcusolver.so.${LUPINE_CUSOLVER_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_cusolvermg_client>\" | grep -q ' T cusolverMgSyevd@@libcusolverMg.so.${LUPINE_CUSOLVER_SOVERSION}$'")
    endif()

    if(TARGET lupine_nvrtc_client)
        add_test(NAME nvrtc_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_nvrtc_client>\" | grep -q ' T nvrtcCompileProgram@@libnvrtc.so.${LUPINE_NVRTC_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvrtc_client>\" | grep -q ' T nvrtcGetLoweredName@@libnvrtc.so.${LUPINE_NVRTC_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvrtc_client>\" | grep -q ' T nvrtcGetPTX@@libnvrtc.so.${LUPINE_NVRTC_SOVERSION}$'")
    endif()

    if(TARGET lupine_nccl_client)
        add_test(NAME nccl_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_nccl_client>\" | grep -q ' T ncclAllReduce$' && nm -D \"$<TARGET_FILE:lupine_nccl_client>\" | grep -q ' T pncclAllReduce$' && nm -D \"$<TARGET_FILE:lupine_nccl_client>\" | grep -q ' T ncclGroupEnd$' && nm -D \"$<TARGET_FILE:lupine_nccl_client>\" | grep -q ' T ncclCommInitRank$'")
    endif()

    if(TARGET lupine_nvjitlink_client)
        add_test(NAME nvjitlink_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_nvjitlink_client>\" | grep -q ' T nvJitLinkCreate@@libnvJitLink.so.${LUPINE_NVJITLINK_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvjitlink_client>\" | grep -q ' T __nvJitLinkCreate_${LUPINE_NVJITLINK_MAJOR}_${LUPINE_NVJITLINK_MINOR}@@libnvJitLink.so.${LUPINE_NVJITLINK_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvjitlink_client>\" | grep -q ' T __nvJitLinkGetLinkedPtx_${LUPINE_NVJITLINK_MAJOR}_0@@libnvJitLink.so.${LUPINE_NVJITLINK_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvjitlink_client>\" | grep -q ' T nvJitLinkVersion@@libnvJitLink.so.${LUPINE_NVJITLINK_SOVERSION}$'")
    endif()

    if(TARGET lupine_nvjpeg_client)
        add_test(NAME nvjpeg_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_nvjpeg_client>\" | grep -q ' T nvjpegDecode@@libnvjpeg.so.${LUPINE_NVJPEG_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvjpeg_client>\" | grep -q ' T nvjpegDecodeBatchedEx@@libnvjpeg.so.${LUPINE_NVJPEG_SOVERSION}$' && nm -D \"$<TARGET_FILE:lupine_nvjpeg_client>\" | grep -q ' T nvjpegEncodeRetrieveBitstream@@libnvjpeg.so.${LUPINE_NVJPEG_SOVERSION}$'")
    endif()

    if(TARGET lupine_nppc_client)
        # One representative export per library, each on its library's node.
        set(npp_export_checks
            "nppc:nppGetStreamContext" "nppial:nppiAdd_8u_C1RSfs_Ctx"
            "nppicc:nppiRGBToGray_8u_C3C1R_Ctx" "nppidei:nppiCopy_8u_C1R_Ctx"
            "nppif:nppiFilterBoxBorder_8u_C1R_Ctx" "nppig:nppiResize_8u_C1R_Ctx"
            "nppim:nppiDilate3x3_8u_C1R_Ctx" "nppist:nppiMean_StdDev_8u_C1R_Ctx"
            "nppisu:nppiMalloc_8u_C1" "nppitc:nppiThreshold_8u_C1R_Ctx"
            "npps:nppsSum_32f_Ctx")
        foreach(check IN LISTS npp_export_checks)
            string(REPLACE ":" ";" check "${check}")
            list(GET check 0 library)
            list(GET check 1 symbol)
            add_test(NAME ${library}_exports
                COMMAND bash -c
                    "nm -D \"$<TARGET_FILE:lupine_${library}_client>\" | grep -q ' T ${symbol}@@lib${library}.so.${LUPINE_NPP_SOVERSION}$'")
        endforeach()
    endif()

    if(TARGET lupine_cufile_client)
        add_test(NAME cufile_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_cufile_client>\" | grep -q ' T cuFileHandleRegister$' && nm -D \"$<TARGET_FILE:lupine_cufile_client>\" | grep -q ' T cuFileRead$' && nm -D \"$<TARGET_FILE:lupine_cufile_client>\" | grep -q ' T cuFileWrite$' && nm -D \"$<TARGET_FILE:lupine_cufile_client>\" | grep -q ' T cuFileDriverClose$' && nm -D \"$<TARGET_FILE:lupine_cufile_client>\" | grep -q ' T cuFileDriverClose_v2$'")
    endif()

    if(TARGET lupine_cupti_client)
        # The entry points PyTorch's libtorch_cpu.so resolves from libcupti,
        # each under the SONAME's version node.
        add_test(NAME cupti_exports
            COMMAND bash -c
                "for symbol in cuptiActivityDisable cuptiActivityEnable cuptiActivityEnableCudaEventDeviceTimestamps cuptiActivityEnableDriverApi cuptiActivityEnableRuntimeApi cuptiActivityFlushAll cuptiActivityGetNextRecord cuptiActivityGetNumDroppedRecords cuptiActivityPopExternalCorrelationId cuptiActivityPushExternalCorrelationId cuptiActivityRegisterCallbacks cuptiActivityRegisterTimestampCallback cuptiActivitySetAttribute cuptiEnableCallback cuptiEnableDomain cuptiFinalize cuptiGetCallbackName cuptiGetResultString cuptiGetVersion cuptiSubscribe cuptiUnsubscribe; do nm -D \"$<TARGET_FILE:lupine_cupti_client>\" | grep -q \" T \${symbol}@@libcupti.so.${LUPINE_CUPTI_SOVERSION}$\" || exit 1; done")
    endif()

    if(TARGET lupine_nvshmem_client)
        # The twenty entry points PyTorch's libtorch_nvshmem.so resolves at
        # load time, each one in the NVSHMEM version node its reference names.
        add_test(NAME nvshmem_exports
            COMMAND bash -c
                "nm -D \"$<TARGET_FILE:lupine_nvshmem_client>\" | grep -cE ' T (nvshmem_free|nvshmem_info_get_version|nvshmem_init|nvshmem_malloc|nvshmem_ptr|nvshmem_team_n_pes|nvshmem_team_split_strided|nvshmemid_hostlib_finalize|nvshmemid_hostlib_init_attr|nvshmemid_init_status|nvshmemx_alltoallmem_on_stream|nvshmemx_broadcastmem_on_stream|nvshmemx_cumodule_init|nvshmemx_get_uniqueid|nvshmemx_getmem_on_stream|nvshmemx_mc_ptr|nvshmemx_putmem_on_stream|nvshmemx_putmem_signal_on_stream|nvshmemx_set_attr_uniqueid_args|nvshmemx_signal_wait_until_on_stream)@@NVSHMEM$' | grep -qx 20")
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
if(TARGET lupine_cudart_client)
    message(STATUS "Building CUDA runtime client: lupine_cudart_client")
endif()
if(TARGET lupine_cublas_client)
    message(STATUS
        "Building cuBLAS clients: lupine_cublas_client (libcublas.so.${LUPINE_CUBLAS_SOVERSION}), lupine_cublaslt_client (libcublasLt.so.${LUPINE_CUBLAS_SOVERSION})")
endif()
if(TARGET lupine_cufft_client)
    message(STATUS
        "Building cuFFT client: lupine_cufft_client (libcufft.so.${LUPINE_CUFFT_SOVERSION})")
endif()
if(TARGET lupine_cudnn_client)
    message(STATUS
        "Building cuDNN client: lupine_cudnn_client (libcudnn.so.${LUPINE_CUDNN_SOVERSION})")
endif()
if(TARGET lupine_curand_client)
    message(STATUS
        "Building cuRAND client: lupine_curand_client (libcurand.so.${LUPINE_CURAND_SOVERSION})")
endif()
if(TARGET lupine_cusparse_client)
    message(STATUS
        "Building cuSPARSE client: lupine_cusparse_client (libcusparse.so.${LUPINE_CUSPARSE_SOVERSION})")
endif()
if(TARGET lupine_cusolver_client)
    message(STATUS
        "Building cuSOLVER clients: lupine_cusolver_client, lupine_cusolvermg_client (libcusolver.so.${LUPINE_CUSOLVER_SOVERSION})")
endif()
if(TARGET lupine_nvrtc_client)
    message(STATUS
        "Building NVRTC client: lupine_nvrtc_client (libnvrtc.so.${LUPINE_NVRTC_SOVERSION})")
endif()
if(TARGET lupine_nccl_client)
    message(STATUS
        "Building NCCL client: lupine_nccl_client (libnccl.so.${LUPINE_NCCL_SOVERSION})")
endif()
if(TARGET lupine_nvjitlink_client)
    message(STATUS
        "Building nvJitLink client: lupine_nvjitlink_client (libnvJitLink.so.${LUPINE_NVJITLINK_SOVERSION}, release ${LUPINE_NVJITLINK_MAJOR}.${LUPINE_NVJITLINK_MINOR})")
endif()
if(TARGET lupine_nvjpeg_client)
    message(STATUS
        "Building nvJPEG client: lupine_nvjpeg_client (libnvjpeg.so.${LUPINE_NVJPEG_SOVERSION})")
endif()
if(TARGET lupine_nppc_client)
    message(STATUS
        "Building NPP clients: lupine_nppc_client and its image and signal libraries (libnppc.so.${LUPINE_NPP_SOVERSION})")
endif()
if(TARGET lupine_cufile_client)
    message(STATUS
        "Building cuFile client: lupine_cufile_client (libcufile.so.0, reporting version ${LUPINE_CUFILE_VERSION})")
endif()
if(TARGET lupine_cupti_client)
    message(STATUS
        "Building CUPTI client: lupine_cupti_client (libcupti.so.${LUPINE_CUPTI_SOVERSION})")
endif()
if(TARGET lupine_nvshmem_client)
    message(STATUS
        "Building nvSHMEM client: lupine_nvshmem_client (libnvshmem_host.so.${LUPINE_NVSHMEM_SOVERSION}, load-only)")
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
