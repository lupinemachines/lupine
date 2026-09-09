cmake_minimum_required(VERSION 3.21)
project(LupineIntegrationTests NONE)
enable_testing()

set(LUPINE_LIB "$ENV{LUPINE_LIB}")
if(NOT LUPINE_LIB)
    set(LUPINE_LIB "${CMAKE_CURRENT_LIST_DIR}/../../build/libcuda.so.1")
endif()
get_filename_component(LUPINE_LIB "${LUPINE_LIB}" ABSOLUTE)
get_filename_component(shim_dir "${LUPINE_LIB}" DIRECTORY)
file(GLOB runtime_libs "${shim_dir}/libcudart.so*")
if(NOT EXISTS "${LUPINE_LIB}" OR NOT runtime_libs)
    message(FATAL_ERROR "Build lupine_cuda_client and lupine_cudart_client before configuring integration tests")
endif()

# Both variants use the same build. Only the libraries visible to the loader
# differ: never put the full build directory on the driver-only search path.
set(variants driver-only driver-runtime)
foreach(variant IN LISTS variants)
    set(lib_dir "${CMAKE_CURRENT_BINARY_DIR}/${variant}/lib")
    file(MAKE_DIRECTORY "${lib_dir}")
    file(CREATE_LINK "${LUPINE_LIB}" "${lib_dir}/libcuda.so.1" SYMBOLIC)
    file(CREATE_LINK "${LUPINE_LIB}" "${lib_dir}/libcuda.so" SYMBOLIC)
    if(variant STREQUAL "driver-runtime")
        foreach(lib IN LISTS runtime_libs)
            get_filename_component(name "${lib}" NAME)
            file(CREATE_LINK "${lib}" "${lib_dir}/${name}" SYMBOLIC)
        endforeach()
    endif()
endforeach()

set(results_dir "$ENV{INTEGRATION_RESULTS_DIR}")
if(NOT results_dir)
    set(results_dir "${CMAKE_CURRENT_BINARY_DIR}/results")
endif()
set(custom_build_dir "${CMAKE_CURRENT_BINARY_DIR}/custom-bin")
file(MAKE_DIRECTORY "${custom_build_dir}")
set(port 14900)

function(add_suite suite script)
    set(list_env ${ARGN})
    if(suite STREQUAL "cuda-python")
        set(list_env)
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env LIST_TESTS=1 SERVER_UPLOAD=0
                ${list_env} "${script}"
        OUTPUT_VARIABLE tests OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" tests "${tests}")

    foreach(name IN LISTS tests)
        if(suite STREQUAL "custom")
            # Compile each custom executable once, even when both variants run
            # in parallel or only one variant is selected with ctest -L.
            add_test(NAME "build.custom.${name}" COMMAND "${script}" "${name}")
            set_tests_properties("build.custom.${name}" PROPERTIES
                ENVIRONMENT "BUILD_ONLY=1;BUILD_TESTS=1;BUILD_DIR=${custom_build_dir};SERVER_UPLOAD=0;LUPINE_LIB=${LUPINE_LIB};LD_LIBRARY_PATH=;LD_PRELOAD="
                FIXTURES_SETUP "build.custom.${name}" TIMEOUT 900)
        endif()
        foreach(variant IN LISTS variants)
            set(test "${variant}.${suite}.${name}")
            set(lib_dir "${CMAKE_CURRENT_BINARY_DIR}/${variant}/lib")
            set(test_env
                "SERVER_PORT_BASE=${port}" "SERVER_UPLOAD=0"
                "RESULTS_DIR=${results_dir}/cases/${variant}/${suite}/${name}"
                "LUPINE_LIB=${lib_dir}/libcuda.so.1"
                "LD_LIBRARY_PATH=${lib_dir}" "LD_PRELOAD="
                "NVJPEG_ASSETS_DIR=${results_dir}/assets/${variant}"
                ${ARGN})
            add_test(NAME "${test}" COMMAND "${script}" "${name}")
            set_tests_properties("${test}" PROPERTIES
                ENVIRONMENT "${test_env}"
                LABELS "${variant};${suite}"
                RESOURCE_LOCK "${suite}.${name}"
                SKIP_REGULAR_EXPRESSION "SKIP:" TIMEOUT 900)
            if(suite STREQUAL "custom")
                set_property(TEST "${test}" APPEND PROPERTY ENVIRONMENT
                    "BUILD_TESTS=0" "BUILD_ONLY=0" "BUILD_DIR=${custom_build_dir}")
                set_property(TEST "${test}" PROPERTY FIXTURES_REQUIRED "build.custom.${name}")
            endif()
            math(EXPR port "${port} + 10")
        endforeach()
    endforeach()
    set(port "${port}" PARENT_SCOPE)
endfunction()

set(TEST_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
add_suite(cuda "${TEST_DIR}/run_cuda_samples.sh" BUILD_SAMPLES=0)
add_suite(pytorch "${TEST_DIR}/run_pytorch_lupine_tests.sh")
add_suite(cuda-python "${TEST_DIR}/run_cuda_python_tests.sh" CUDA_PYTHON_INSTALL=0)
add_suite(cuda-library "${TEST_DIR}/run_cuda_library_samples.sh" BUILD_SAMPLES=0)
add_suite(custom "${TEST_DIR}/run_custom_tests.sh")

foreach(variant IN LISTS variants)
    set(decoder "${variant}.cuda-library.nvJPEG/nvJPEG-Decoder")
    set(encoder "${variant}.cuda-library.nvJPEG/nvJPEG-Encoder-MultipleInstances")
    set_property(TEST "${encoder}" PROPERTY DEPENDS "${decoder}")
endforeach()
