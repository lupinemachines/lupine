cmake_minimum_required(VERSION 3.21)
project(LupineIntegrationTests NONE)
enable_testing()

set(LUPINE_LIB "$ENV{LUPINE_LIB}")
if(NOT LUPINE_LIB)
    set(LUPINE_LIB "${CMAKE_CURRENT_LIST_DIR}/../../build/libcuda.so.1")
endif()
get_filename_component(LUPINE_LIB "${LUPINE_LIB}" ABSOLUTE)
get_filename_component(shim_dir "${LUPINE_LIB}" DIRECTORY)
if(NOT EXISTS "${LUPINE_LIB}")
    message(FATAL_ERROR "Build lupine_cuda_client before configuring integration tests")
endif()

# Every suite loads the driver shim under NVIDIA's own runtime and libraries,
# with the NVML, NCCL and nvSHMEM shims beside it.
set(variant driver-only)
set(lib_dir "${CMAKE_CURRENT_BINARY_DIR}/${variant}/lib")
file(MAKE_DIRECTORY "${lib_dir}")
file(CREATE_LINK "${LUPINE_LIB}" "${lib_dir}/libcuda.so.1" SYMBOLIC)
file(CREATE_LINK "${LUPINE_LIB}" "${lib_dir}/libcuda.so" SYMBOLIC)
file(GLOB shim_libs "${shim_dir}/libnvidia-ml.so*" "${shim_dir}/libnccl.so*"
     "${shim_dir}/libnvshmem_host.so*")
foreach(lib IN LISTS shim_libs)
    get_filename_component(name "${lib}" NAME)
    file(CREATE_LINK "${lib}" "${lib_dir}/${name}" SYMBOLIC)
endforeach()

set(results_dir "$ENV{INTEGRATION_RESULTS_DIR}")
if(NOT results_dir)
    set(results_dir "${CMAKE_CURRENT_BINARY_DIR}/results")
endif()
set(port "$ENV{SERVER_PORT_BASE}")
if(NOT port)
    set(port 14900)
endif()

function(add_suite suite script)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env LIST_TESTS=1 SERVER_UPLOAD=0
                ${ARGN} "${script}"
        OUTPUT_VARIABLE tests OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" tests "${tests}")

    foreach(name IN LISTS tests)
        set(test "${variant}.${suite}.${name}")
        add_test(NAME "${test}" COMMAND "${script}" "${name}")
        set_tests_properties("${test}" PROPERTIES
            ENVIRONMENT "SERVER_PORT_BASE=${port};SERVER_UPLOAD=0;RESULTS_DIR=${results_dir}/cases/${variant}/${suite}/${name};LUPINE_LIB=${lib_dir}/libcuda.so.1;LD_LIBRARY_PATH=${lib_dir};LD_PRELOAD=;NVJPEG_ASSETS_DIR=${results_dir}/assets;NVJPEG2K_ASSETS_DIR=${results_dir}/assets/nvjpeg2k;NVTIFF_ASSETS_DIR=${results_dir}/assets/nvtiff;NPPPLUS_ASSETS_DIR=${results_dir}/assets/nppplus;${ARGN}"
            LABELS "${variant};${suite}"
            RESOURCE_LOCK "${suite}.${name}"
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

# These samples read files another sample writes into the assets dir.
function(add_sample_dependency consumer producer)
    set(consumer "${variant}.cuda-library.${consumer}")
    set(producer "${variant}.cuda-library.${producer}")
    if(TEST "${consumer}" AND TEST "${producer}")
        set_property(TEST "${consumer}" APPEND PROPERTY DEPENDS "${producer}")
    endif()
endfunction()
add_sample_dependency(nvJPEG/nvJPEG-Encoder-MultipleInstances nvJPEG/nvJPEG-Decoder)
add_sample_dependency(nvJPEG2000/nvJPEG2000-Encoder nvJPEG2000/nvJPEG2000-Decoder)
add_sample_dependency(nvTIFF/nvTIFF-GeoTIFF-Decode-Encode nvTIFF/nvTIFF-Encode-Options)
add_sample_dependency(nvTIFF/nvTIFF-Image-Info-Multi-Image nvTIFF/nvTIFF-Encode-Options)
