if(LUPINE_OPS_CUDA)
    add_library(lupine_smemcpy STATIC smemcpy.cu)
    target_include_directories(lupine_smemcpy PUBLIC ${PROJECT_SOURCE_DIR})
    set_target_properties(lupine_smemcpy PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)

    if(BUILD_TESTING)
        add_executable(smemcpy_test smemcpy_test.cu)
        target_link_libraries(smemcpy_test PRIVATE lupine_smemcpy)
        set_target_properties(smemcpy_test PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON)
        add_test(NAME smemcpy_test COMMAND smemcpy_test)
        set_tests_properties(smemcpy_test PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 120)
    endif()
endif()

if(LUPINE_OPS_HIP)
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
