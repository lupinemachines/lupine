get_property(enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)

if("CUDA" IN_LIST enabled_languages)
    add_library(lupine_cuda_smemcpy STATIC
        smemcpy.cu
        smemcpy_dispatch.cpp)
    target_include_directories(lupine_cuda_smemcpy PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(lupine_cuda_smemcpy PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)

    if(BUILD_TESTING)
        add_executable(smemcpy_test smemcpy_test.cu)
        target_link_libraries(smemcpy_test PRIVATE lupine_cuda_smemcpy)
        set_target_properties(smemcpy_test PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON)
        add_test(NAME smemcpy_test COMMAND smemcpy_test)
        set_tests_properties(smemcpy_test PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 120)
    endif()
endif()

if(LUPINE_CUDA_SMEMCPY_FATBIN)
    if(NOT EXISTS "${LUPINE_CUDA_SMEMCPY_FATBIN}")
        message(FATAL_ERROR
            "LUPINE_CUDA_SMEMCPY_FATBIN does not exist: ${LUPINE_CUDA_SMEMCPY_FATBIN}")
    endif()
    set(lupine_smemcpy_fatbin "${LUPINE_CUDA_SMEMCPY_FATBIN}")
elseif("CUDA" IN_LIST enabled_languages)
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-code
        OUTPUT_VARIABLE lupine_cuda_gpu_codes
        RESULT_VARIABLE lupine_cuda_gpu_codes_result)
    if(NOT lupine_cuda_gpu_codes_result EQUAL 0)
        message(FATAL_ERROR "nvcc --list-gpu-code failed")
    endif()
    string(REPLACE "\n" ";" lupine_cuda_gpu_codes
        "${lupine_cuda_gpu_codes}")
    set(lupine_cuda_oldest_arch "")
    set(lupine_cuda_gencode_args "")
    foreach(code IN LISTS lupine_cuda_gpu_codes)
        if(code MATCHES "^sm_([0-9]+)$")
            set(arch "${CMAKE_MATCH_1}")
            if(NOT lupine_cuda_oldest_arch OR
               arch LESS lupine_cuda_oldest_arch)
                set(lupine_cuda_oldest_arch "${arch}")
            endif()
            list(APPEND lupine_cuda_gencode_args
                "--generate-code=arch=compute_${arch},code=sm_${arch}")
        endif()
    endforeach()
    if(NOT lupine_cuda_oldest_arch)
        message(FATAL_ERROR "nvcc did not report a supported GPU target")
    endif()
    list(APPEND lupine_cuda_gencode_args
        "--generate-code=arch=compute_${lupine_cuda_oldest_arch},code=compute_${lupine_cuda_oldest_arch}")
    separate_arguments(lupine_cuda_fatbin_flags NATIVE_COMMAND
        "${CMAKE_CUDA_FLAGS}")
    set(lupine_smemcpy_fatbin
        "${CMAKE_CURRENT_BINARY_DIR}/lupine_smemcpy.fatbin")
    add_custom_command(
        OUTPUT "${lupine_smemcpy_fatbin}"
        COMMAND ${CMAKE_CUDA_COMPILER}
            --fatbin
            --std=c++17
            ${lupine_cuda_fatbin_flags}
            ${lupine_cuda_gencode_args}
            -I${PROJECT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/smemcpy.cu
            -o "${lupine_smemcpy_fatbin}"
        DEPENDS
            smemcpy.cu
            smemcpy.h
            smemcpy_dispatch.h
        VERBATIM)
endif()

if(lupine_smemcpy_fatbin)
    set(lupine_smemcpy_fatbin_cpp
        "${CMAKE_CURRENT_BINARY_DIR}/lupine_smemcpy_fatbin.cpp")
    add_executable(lupine_embed_fatbin embed_fatbin.cpp)
    add_custom_command(
        OUTPUT "${lupine_smemcpy_fatbin_cpp}"
        COMMAND $<TARGET_FILE:lupine_embed_fatbin>
            "${lupine_smemcpy_fatbin}"
            "${lupine_smemcpy_fatbin_cpp}"
        DEPENDS
            "${lupine_smemcpy_fatbin}"
            lupine_embed_fatbin
        VERBATIM)
    add_library(lupine_cuda_smemcpy_module STATIC
        smemcpy_dispatch.cpp
        smemcpy_module.cpp
        "${lupine_smemcpy_fatbin_cpp}")
    target_include_directories(lupine_cuda_smemcpy_module PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(lupine_cuda_smemcpy_module PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
endif()

if("HIP" IN_LIST enabled_languages)
    add_library(lupine_hip_smemcpy STATIC hip_smemcpy.hip)
    target_include_directories(lupine_hip_smemcpy PUBLIC ${PROJECT_SOURCE_DIR})
    set_target_properties(lupine_hip_smemcpy PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)

    if(BUILD_TESTING)
        add_executable(hip_smemcpy_test hip_smemcpy_test.hip)
        target_link_libraries(hip_smemcpy_test PRIVATE lupine_hip_smemcpy)
        set_target_properties(hip_smemcpy_test PROPERTIES
            HIP_STANDARD 17
            HIP_STANDARD_REQUIRED ON)
        add_test(NAME hip_smemcpy_test COMMAND hip_smemcpy_test)
        set_tests_properties(hip_smemcpy_test PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 120)
    endif()
endif()
