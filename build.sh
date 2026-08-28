#!/bin/bash

set -e

build_dir="build"
precompiled_ops="$PWD/$build_dir/precompiled-ops"

cd codegen

# Run the Python script in the codegen directory
python3 ./codegen.py

# Move up to the parent directory
cd ..

# Run CMake to configure and build the project
cmake -DLUPINE_PRECOMPILED_OPS="$precompiled_ops" \
  -P ops/precompile.cmake
cmake -S . -B "$build_dir" \
  -DLUPINE_PRECOMPILED_OPS="$precompiled_ops"
cmake --build "$build_dir"
