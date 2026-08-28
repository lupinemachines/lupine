#!/bin/bash

set -e

build_dir="build"

cd codegen

# Run the Python script in the codegen directory
python3 ./codegen.py

# Move up to the parent directory
cd ..

# Run CMake to configure and build the project
cmake -S . -B "$build_dir"
cmake --build "$build_dir"
