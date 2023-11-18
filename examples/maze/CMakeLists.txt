cmake_minimum_required(VERSION 3.12)
project(maze VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Target-specific include directories
include_directories(${nunn_SOURCE_DIR}/reinforcement/inc)
include_directories(${nunn_SOURCE_DIR}/common/inc)

# File GLOB
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# Add executable
add_executable(maze ${SOURCES})

# Link libraries
target_link_libraries(maze nunn)

# Compiler-specific configurations
if(MSVC)
    # Using Visual Studio C++
    target_compile_options(maze PRIVATE /W4 /permissive-)
else()
    # Other compilers like GCC, Clang
    target_compile_options(maze PRIVATE -Wall -Wextra -pedantic)
endif()

# Set target properties (optional, for more fine-grained control)
set_target_properties(maze PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

