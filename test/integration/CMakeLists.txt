cmake_minimum_required(VERSION 3.21)
project(LupineIntegrationTests NONE)
enable_testing()

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
        add_test(NAME "${suite}.${name}" COMMAND "${script}" "${name}")
        set_tests_properties("${suite}.${name}" PROPERTIES
            ENVIRONMENT "SERVER_PORT_BASE=${port};SERVER_UPLOAD=0;RESULTS_DIR=$ENV{INTEGRATION_RESULTS_DIR}/cases/${suite}/${name};${ARGN}"
            SKIP_REGULAR_EXPRESSION "SKIP:" TIMEOUT 900)
        math(EXPR port "${port} + 10")
    endforeach()
    set(port "${port}" PARENT_SCOPE)
endfunction()

set(TEST_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
add_suite(cuda "${TEST_DIR}/run_cuda_samples.sh" BUILD_SAMPLES=0)
add_suite(pytorch "${TEST_DIR}/run_pytorch_lupine_tests.sh")
add_suite(cuda-python "${TEST_DIR}/run_cuda_python_tests.sh" CUDA_PYTHON_INSTALL=0)
add_suite(cuda-library "${TEST_DIR}/run_cuda_library_samples.sh" BUILD_SAMPLES=0)
add_suite(custom "${TEST_DIR}/run_custom_tests.sh")

set(decoder "cuda-library.nvJPEG/nvJPEG-Decoder")
set(encoder "cuda-library.nvJPEG/nvJPEG-Encoder-MultipleInstances")
set_property(TEST "${encoder}" PROPERTY DEPENDS "${decoder}")
