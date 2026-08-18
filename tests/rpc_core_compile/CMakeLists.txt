cmake_minimum_required(VERSION 3.18)
project(lupine_rpc_core_compile LANGUAGES CXX)

add_library(rpc_core_compile OBJECT ../../rpc.cpp)
target_include_directories(rpc_core_compile PRIVATE ../..)
target_compile_features(rpc_core_compile PRIVATE cxx_std_17)
