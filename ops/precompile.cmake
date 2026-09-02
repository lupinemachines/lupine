cmake_minimum_required(VERSION 3.18)

if(NOT LUPINE_PRECOMPILED_OPS)
    message(FATAL_ERROR
        "LUPINE_PRECOMPILED_OPS must name the output directory")
endif()

if(NOT LUPINE_CUDA_COMPILER)
    find_program(LUPINE_CUDA_COMPILER nvcc
        HINTS
            "${CUDAToolkit_ROOT}/bin"
            "$ENV{CUDA_HOME}/bin"
            "$ENV{CUDA_PATH}/bin")
endif()
if(NOT LUPINE_CUDA_COMPILER)
    message(FATAL_ERROR
        "nvcc was not found; set LUPINE_CUDA_COMPILER")
endif()

if(NOT LUPINE_CUDA_BIN2C)
    get_filename_component(lupine_cuda_bin_dir
        "${LUPINE_CUDA_COMPILER}" DIRECTORY)
    find_program(LUPINE_CUDA_BIN2C bin2c
        HINTS "${lupine_cuda_bin_dir}"
        NO_DEFAULT_PATH)
endif()
if(NOT LUPINE_CUDA_BIN2C)
    message(FATAL_ERROR
        "bin2c was not found next to nvcc; set LUPINE_CUDA_BIN2C")
endif()

execute_process(
    COMMAND "${LUPINE_CUDA_COMPILER}" --list-gpu-code
    OUTPUT_VARIABLE lupine_cuda_gpu_codes
    RESULT_VARIABLE lupine_cuda_gpu_codes_result)
if(NOT lupine_cuda_gpu_codes_result EQUAL 0)
    message(FATAL_ERROR "nvcc --list-gpu-code failed")
endif()

string(REGEX MATCHALL "sm_[0-9]+" lupine_cuda_gpu_codes
    "${lupine_cuda_gpu_codes}")
list(REMOVE_DUPLICATES lupine_cuda_gpu_codes)
set(lupine_cuda_oldest_arch "")
set(lupine_cuda_gencode_args "")
foreach(code IN LISTS lupine_cuda_gpu_codes)
    string(SUBSTRING "${code}" 3 -1 arch)
    if(lupine_cuda_oldest_arch STREQUAL "" OR
       arch LESS lupine_cuda_oldest_arch)
        set(lupine_cuda_oldest_arch "${arch}")
    endif()
    list(APPEND lupine_cuda_gencode_args
        "--generate-code=arch=compute_${arch},code=sm_${arch}")
endforeach()
if(lupine_cuda_oldest_arch STREQUAL "")
    message(FATAL_ERROR "nvcc did not report a supported GPU target")
endif()
list(APPEND lupine_cuda_gencode_args
    "--generate-code=arch=compute_${lupine_cuda_oldest_arch},code=compute_${lupine_cuda_oldest_arch}")

separate_arguments(lupine_cuda_flags NATIVE_COMMAND "${LUPINE_CUDA_FLAGS}")
get_filename_component(lupine_source_dir
    "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(lupine_cuda_ops_dir "${LUPINE_PRECOMPILED_OPS}/cuda")
set(lupine_smemcpy_fatbin "${lupine_cuda_ops_dir}/smemcpy.fatbin")
set(lupine_smemcpy_source "${lupine_cuda_ops_dir}/smemcpy.cpp")
file(MAKE_DIRECTORY "${lupine_cuda_ops_dir}")
file(REMOVE "${lupine_smemcpy_fatbin}" "${lupine_smemcpy_source}")
execute_process(
    COMMAND "${LUPINE_CUDA_COMPILER}"
        --fatbin
        --std=c++17
        ${lupine_cuda_flags}
        ${lupine_cuda_gencode_args}
        "-I${lupine_source_dir}"
        "${CMAKE_CURRENT_LIST_DIR}/smemcpy.cu"
        -o "${lupine_smemcpy_fatbin}"
    RESULT_VARIABLE lupine_smemcpy_result
    COMMAND_ECHO STDOUT)
if(NOT lupine_smemcpy_result EQUAL 0)
    file(REMOVE "${lupine_smemcpy_fatbin}")
    message(FATAL_ERROR "Failed to precompile ${lupine_smemcpy_fatbin}")
endif()

execute_process(
    COMMAND "${LUPINE_CUDA_BIN2C}"
        --const
        --name lupine_smemcpy_fatbin
        "${lupine_smemcpy_fatbin}"
    OUTPUT_FILE "${lupine_smemcpy_source}"
    RESULT_VARIABLE lupine_bin2c_result
    COMMAND_ECHO STDOUT)
if(NOT lupine_bin2c_result EQUAL 0)
    file(REMOVE "${lupine_smemcpy_fatbin}" "${lupine_smemcpy_source}")
    message(FATAL_ERROR "Failed to generate ${lupine_smemcpy_source}")
endif()
file(APPEND "${lupine_smemcpy_source}"
    "\nextern \"C\" const void *lupine_cuda_smemcpy_image() {\n"
    "  return lupine_smemcpy_fatbin;\n"
    "}\n")
file(REMOVE "${lupine_smemcpy_fatbin}")

message(STATUS "Precompiled operations: ${LUPINE_PRECOMPILED_OPS}")
