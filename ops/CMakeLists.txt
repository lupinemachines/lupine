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
