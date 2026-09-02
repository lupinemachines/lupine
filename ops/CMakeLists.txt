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

if(LUPINE_BUILD_SERVER AND LUPINE_BUILD_CUDA)
    if(LUPINE_PRECOMPILED_OPS)
        set(lupine_cuda_ops_dir "${LUPINE_PRECOMPILED_OPS}/cuda")
        set(lupine_smemcpy_include "${lupine_cuda_ops_dir}/smemcpy.inc")
        if(NOT EXISTS "${lupine_smemcpy_include}")
            message(FATAL_ERROR
                "Missing precompiled CUDA operation: ${lupine_smemcpy_include}")
        endif()
    else()
        set(lupine_built_ops "${CMAKE_CURRENT_BINARY_DIR}/precompiled-ops")
        set(lupine_cuda_ops_dir "${lupine_built_ops}/cuda")
        set(lupine_smemcpy_include "${lupine_cuda_ops_dir}/smemcpy.inc")
        add_custom_command(
            OUTPUT "${lupine_smemcpy_include}"
            COMMAND ${CMAKE_COMMAND}
                "-DLUPINE_PRECOMPILED_OPS=${lupine_built_ops}"
                "-DLUPINE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
                "-DLUPINE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/precompile.cmake"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/precompile.cmake"
                smemcpy.cu
                smemcpy.h
                smemcpy_dispatch.h
            VERBATIM)
    endif()
    add_library(lupine_cuda_smemcpy_module STATIC
        smemcpy_dispatch.cpp
        smemcpy_module.cpp
        "${lupine_smemcpy_include}")
    set_source_files_properties("${lupine_smemcpy_include}" PROPERTIES
        HEADER_FILE_ONLY TRUE)
    target_include_directories(lupine_cuda_smemcpy_module PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        PRIVATE ${lupine_cuda_ops_dir})
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
