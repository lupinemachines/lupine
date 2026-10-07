cmake_minimum_required(VERSION 3.21)
project(LupineROCmIntegrationTests NONE)
enable_testing()

# Explicitly configured by developers; not included by the root project or CI.
set(ROCM_SUITES "hip-tests;rocm-examples;rocm-libraries" CACHE STRING
    "Prepared ROCm suites to register")
set(port "$ENV{SERVER_PORT_BASE}")
if(NOT port)
    set(port 23000)
endif()
set(results_dir "$ENV{INTEGRATION_RESULTS_DIR}")
if(NOT results_dir)
    set(results_dir "${CMAKE_CURRENT_BINARY_DIR}/results")
endif()

foreach(suite IN LISTS ROCM_SUITES)
    if(suite STREQUAL "hip-tests")
        set(script run_hip_tests.sh)
    elseif(suite STREQUAL "rocm-examples")
        set(script run_rocm_examples.sh)
    elseif(suite STREQUAL "rocm-libraries")
        set(script run_rocm_libraries.sh)
    else()
        message(FATAL_ERROR "Unknown ROCm suite: ${suite}")
    endif()
    set(script "${CMAKE_CURRENT_LIST_DIR}/../${script}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env LIST_TESTS=1 BUILD_TESTS=0 BUILD_SAMPLES=0 "${script}"
        OUTPUT_VARIABLE projects OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" projects "${projects}")
    foreach(name IN LISTS projects)
        if(port GREATER 65520)
            message(FATAL_ERROR "ROCm tests exceed the available server port range")
        endif()
        add_test(NAME "${suite}.${name}" COMMAND "${script}" "${name}")
        set_tests_properties("${suite}.${name}" PROPERTIES
            ENVIRONMENT "LIST_TESTS=0;BUILD_ONLY=0;BUILD_TESTS=0;BUILD_SAMPLES=0;SERVER_PORT_BASE=${port};SERVER_PORT_COUNT=16;RESULTS_DIR=${results_dir}/${suite}/${name}"
            LABELS "rocm;${suite}"
            SKIP_RETURN_CODE 77
            TIMEOUT 900)
        # A sample can contain multiple executable units. Keep their ports
        # disjoint when CTest runs project invocations concurrently.
        math(EXPR port "${port} + 16")
    endforeach()
endforeach()
