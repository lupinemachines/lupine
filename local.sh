#!/bin/bash

client_lib_path="$(pwd)/build/libcuda.so.1"
server_out_path="$(pwd)/build/lupine_driver_server"

ansi_format() {
  case "$1" in
  "pass")
    echo -e "\e[32m   ✓ $2\e[0m"
    ;;
  "fail")
    echo -e "\e[31m    ✗ $2\e[0m"
    ;;
  "emph")
    echo -e "\e[1m$2\e[0m"
    ;;
  *)
    echo "$2"
    ;;
  esac
}

#---- lists tests here ----#
test_cuda_available() {
  pass_message=$1
  fail_message=$2

  output=$(LD_PRELOAD="$client_lib_path" python3 -c "import torch; print(torch.cuda.is_available())" | tail -n 1)

  if [[ "$output" == "True" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "CUDA is not available. Expected True but got [$output]."
    return 1
  fi
}

test_tensor_to_cuda() {
  output=$(LD_PRELOAD="$client_lib_path" python3 -c "
import torch
print('Creating a tensor...')
tensor = torch.zeros(10, 10)
print('Moving tensor to CUDA...')
tensor = tensor.to('cuda:0')
print('Tensor successfully moved to CUDA')
" | tail -n 1)

  if [[ "$output" == "Tensor successfully moved to CUDA" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "Tensor failed. Got [$output]."
    return 1
  fi
}

test_tensor_to_cuda_to_cpu() {
  output=$(LD_PRELOAD="$client_lib_path" python3 -c "
import torch
print('Creating a tensor...')
tensor = torch.full((10, 10), 5)
print('Tensor created on CPU:')
print(tensor)

print('Moving tensor to CUDA...')
tensor_cuda = tensor.to('cuda:0')
print('Tensor successfully moved to CUDA')

print('Moving tensor back to CPU...')
tensor_cpu = tensor_cuda.to('cpu')
print('Tensor successfully moved back to CPU:')
" | tail -n 1)

  if [[ "$output" == "Tensor successfully moved back to CPU:" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "Tensor failed. Got [$output]."
    return 1
  fi
}

test_vector_add() {
  output=$(LD_PRELOAD="$client_lib_path" ./vector.o | tail -n 1)

  if [[ "$output" == "PASSED" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "vector_add failed. Got [$output]."
    return 1
  fi
}

test_cudnn() {
  output=$(LD_PRELOAD="$client_lib_path" ./cudnn.o | tail -n 1)

  if [[ "$output" == "New array: 0.5 0.731059 0.880797 0.952574 0.982014 0.993307 0.997527 0.999089 0.999665 0.999877 " ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "test_cudnn failed. Got [$output]."
    return 1
  fi
}

test_cublas_batched() {
  output=$(LD_PRELOAD="$client_lib_path" ./cublas_batched.o | tail -n 5)

  expected_output=$'=====\nC[1]\n111.00 122.00\n151.00 166.00\n====='

  # trim ugly output from the file
  output=$(echo "$output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  expected_output=$(echo "$expected_output" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

  if [[ "$output" == "$expected_output" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "test_cublas_batched failed. Got [$output]."
    return 1
  fi
}

test_unified_mem() {
  output=$(LD_PRELOAD="$client_lib_path" ./unified_pointer.o | tail -n 1)

  if [[ "$output" == "Max error: 0" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "vector_add failed. Got [$output]."
    return 1
  fi
}

test_graphs() {
  output=$(LD_PRELOAD="$client_lib_path" ./cuda_graphs_host_func.o | tail -n 1)

  if [[ "$output" == "[cudaGraphsManual] Host callback final reduced sum = 1.000000" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "vector_add failed. Got [$output]."
    return 1
  fi
}

#---- declare test cases ----#
declare -A test_cuda_avail=(
  ["function"]="test_cuda_available"
  ["pass"]="CUDA is available."
)

declare -A test_tensor_to_cuda=(
  ["function"]="test_tensor_to_cuda"
  ["pass"]="Tensor moved to CUDA successfully."
)

declare -A test_tensor_to_cuda_to_cpu=(
  ["function"]="test_tensor_to_cuda_to_cpu"
  ["pass"]="Tensor successfully moved to CUDA and back to CPU."
)

declare -A test_vector_add=(
  ["function"]="test_vector_add"
  ["pass"]="CUDA vector_add example works."
)

declare -A test_cudnn=(
  ["function"]="test_cudnn"
  ["pass"]="cuDNN correctly applies sigmoid activation on a tensor."
)

declare -A test_cublas_batched=(
  ["function"]="test_cublas_batched"
  ["pass"]="Batched cublas works via test/cublas_batched.cu."
)

declare -A test_unified_mem=(
  ["function"]="test_unified_mem"
  ["pass"]="Unified memory works as expected."
)

declare -A test_graphs=(
  ["function"]="test_graphs"
  ["pass"]="Graphs works as expected."
)

#---- assign them to our associative array ----#
tests=("test_cuda_avail" "test_tensor_to_cuda" "test_tensor_to_cuda_to_cpu" "test_vector_add" "test_cudnn" "test_cublas_batched" "test_unified_mem" "test_graphs")

test() {
  set_paths

  build_tests

  echo "running tests at: $client_lib_path"
  echo -e "\n\033[1mRunning test(s)...\033[0m"

  for test in "${tests[@]}"; do
    func_name=$(eval "echo \${${test}[function]}")
    pass_message=$(eval "echo \${${test}[pass]}")

    if ! eval "$func_name \"$pass_message\""; then
      echo -e "\033[31mTest failed. Exiting...\033[0m"
      return 1
    fi
  done

  echo -e "\n\033[1mRunning CUDA sample compliance test(s)...\033[0m"
  SAMPLE_SUITE="${SAMPLE_SUITE:-compliance}" \
    LUPINE_LIB="$client_lib_path" \
    SERVER_LOCAL_BIN="$server_out_path" \
    "$(pwd)/test/run_cuda_samples.sh"
}

build_tests() {
  echo "building test files..."

  nvcc --cudart=shared -lnvidia-ml -lcuda ./test/vector_add.cu -o vector.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn ./test/cudnn.cu -o cudnn.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/cublas_batched.cu -o cublas_batched.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/unified.cu -o unified.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/unified_pointer.cu -o unified_pointer.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/unified_linked.cu -o unified_linked.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/cublas_unified.cu -o cublas_unified.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/cudnn_managed.cu -o cudnn_managed.o
  nvcc --cudart=shared -lnvidia-ml -lcuda -lcudnn -lcublas ./test/cuda_graphs_host_func.cu -o cuda_graphs_host_func.o
}

set_paths() {
  if [[ ! -f "$client_lib_path" ]]; then
    echo "Error: missing client shim at $client_lib_path. Run: cmake -S . -B build && cmake --build build"
    exit 1
  fi

  if [[ ! -x "$server_out_path" ]]; then
    echo "Error: missing server binary at $server_out_path. Run: cmake -S . -B build && cmake --build build"
    exit 1
  fi

  echo "Using client: $client_lib_path -- server: $server_out_path"
}

run() {
  export LUPINE_SERVER="${LUPINE_SERVER:-0.0.0.0:14833}"
  set_paths
  LD_PRELOAD="$client_lib_path" nvidia-smi
}

test_ci() {
  cmake .
  cmake --build .
  ctest --output-on-failure -R h2_test

  set_paths
  
  build_tests

  echo "running tests at: $client_lib_path"
  echo -e "\n\033[1mRunning test(s)...\033[0m"

  for test in "${tests[@]}"; do
    func_name=$(eval "echo \${${test}[function]}")
    pass_message=$(eval "echo \${${test}[pass]}")

    if ! eval "$func_name \"$pass_message\""; then
      echo -e "\033[31mTest failed. Exiting...\033[0m"
      return 1
    fi
  done
}

server() {
  set_paths
  "$server_out_path"
}

# Main script logic using a switch case
case "$1" in
  build_tests)
    build_tests
    ;;
  run)
    run
    ;;
  server)
    server
    ;;
  test_ci)
    test_ci
    ;;
  test)
    test
    ;;
  *)
    echo "Usage: $0 {build|run|server}"
    exit 1
    ;;
esac
