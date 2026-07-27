add_library(pcg INTERFACE)

# Include relative to the base directory
target_include_directories(pcg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

